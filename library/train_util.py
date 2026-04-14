# common functions for training (stripped for Anima LoRA standalone)
#
# This module was split into sub-packages for maintainability:
#   library.datasets   — dataset classes, bucket management, image utilities
#   library.training   — optimizer, scheduler, checkpoint utilities
#
# All public names are re-exported here so existing `train_util.X` references
# continue to work without modification.

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import subprocess
import time
from io import BytesIO
from typing import Dict, List, Optional

import safetensors.torch
import toml
import torch
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    DistributedDataParallelKwargs,
)
from huggingface_hub import hf_hub_download
from packaging.version import Version

import library.sai_model_spec as sai_model_spec
from library.device_utils import clean_memory_on_device  # noqa: F401 — used by strategy_anima via train_util
from library.utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports from library.datasets
# ---------------------------------------------------------------------------
from library.datasets import (  # noqa: F401, E402
    # buckets
    make_bucket_resolutions,
    BucketManager,
    BucketBatchIndex,
    # subsets
    split_train_val,
    ImageInfo,
    AugHelper,
    BaseSubset,
    DreamBoothSubset,
    # image_utils
    IMAGE_EXTENSIONS,
    IMAGE_TRANSFORMS,
    TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
    TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
    load_image,
    trim_and_resize_if_required,
    load_images_and_masks_for_caching,
    cache_batch_latents,
    save_text_encoder_outputs_to_disk,
    load_text_encoder_outputs_from_disk,
    glob_images,
    glob_images_pathlib,
    is_disk_cached_latents_is_expected,
    ImageLoadingDataset,
    # base
    BaseDataset,
    DreamBoothDataset,
    DatasetGroup,
    MinimalDataset,
    load_arbitrary_dataset,
    debug_dataset,
    collator_class,
    LossRecorder,
)

# ---------------------------------------------------------------------------
# Re-exports from library.training
# ---------------------------------------------------------------------------
from library.training import (  # noqa: F401, E402
    # metadata
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
    build_minimum_network_metadata,
    build_training_metadata,
    add_dataset_metadata,
    add_model_hash_metadata,
    finalize_metadata,
    # checkpoints
    EPOCH_STATE_NAME,
    EPOCH_FILE_NAME,
    EPOCH_DIFFUSERS_DIR_NAME,
    LAST_STATE_NAME,
    DEFAULT_EPOCH_NAME,
    DEFAULT_LAST_OUTPUT_NAME,
    DEFAULT_STEP_NAME,
    STEP_STATE_NAME,
    STEP_FILE_NAME,
    STEP_DIFFUSERS_DIR_NAME,
    default_if_none,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
    save_sd_model_on_train_end_common,
    get_checkpoint_state_dir,
    get_checkpoint_ckpt_name,
    save_checkpoint_state,
    # optimizers
    get_optimizer,
    get_optimizer_train_eval_fn,
    is_schedulefree_optimizer,
    # schedulers
    get_scheduler_fix,
    get_dummy_scheduler,
)

# ---------------------------------------------------------------------------
# HIGH_VRAM flag — kept here for backward compat, delegates to datasets.base
# ---------------------------------------------------------------------------
from library.datasets.base import HIGH_VRAM  # noqa: F401, E402
import library.datasets.base as _datasets_base  # noqa: E402


# ---------------------------------------------------------------------------
# Remaining code that stays in train_util
# (metadata, hashing, arg-parsing, accelerator setup, loss functions, etc.)
# ---------------------------------------------------------------------------

EPSILON = 1e-6


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:
        return "IsADirectory"
    except PermissionError:
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:
        return "IsADirectory"
    except PermissionError:
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    mh = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return mh, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "(unknown)"


# region arguments


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


# Metadata key constants, build_minimum_network_metadata, and full training
# metadata helpers are now in library.training.metadata and re-exported above.


def get_sai_model_spec(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,
    sd3: str = None,
    flux: str = None,
    lumina: str = None,
    optional_metadata: dict[str, str] | None = None,
):
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    model_config = {}
    if sd3 is not None:
        model_config["sd3"] = sd3
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina

    extracted_metadata = {}
    for attr_name in dir(args):
        if attr_name.startswith("metadata_") and not attr_name.startswith(
            "metadata___"
        ):
            value = getattr(args, attr_name, None)
            if value is not None:
                field_name = attr_name[9:]
                if field_name not in [
                    "title",
                    "author",
                    "description",
                    "license",
                    "tags",
                ]:
                    extracted_metadata[field_name] = value

    all_optional_metadata = {**extracted_metadata}
    if optional_metadata:
        all_optional_metadata.update(optional_metadata)

    metadata = sai_model_spec.build_metadata(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,
        model_config=model_config,
        optional_metadata=all_optional_metadata if all_optional_metadata else None,
    )
    return metadata


def get_sai_model_spec_dataclass(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,
    sd3: str = None,
    flux: str = None,
    lumina: str = None,
    hunyuan_image: str = None,
    anima: str = None,
    optional_metadata: dict[str, str] | None = None,
) -> sai_model_spec.ModelSpecMetadata:
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    model_config = {}
    if sd3 is not None:
        model_config["sd3"] = sd3
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina
    if hunyuan_image is not None:
        model_config["hunyuan_image"] = hunyuan_image
    if anima is not None:
        model_config["anima"] = anima
    return sai_model_spec.build_metadata_dataclass(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,
        model_config=model_config,
        optional_metadata=optional_metadata,
    )


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--v2",
        action="store_true",
        help="load Stable Diffusion v2.0 model",
    )
    parser.add_argument(
        "--v_parameterization",
        action="store_true",
        help="enable v-parameterization training",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training)",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Value '{value}' is not a valid percentage"
                )
        try:
            float_value = float(value)
            if float_value >= 1:
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use"
        "Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, "
        "DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, "
        "AdaFactor.",
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes)",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2.0e-6, help="learning rate"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping",
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...")',
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="",
        help="custom scheduler module",
    )
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100")',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler",
    )
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Combines backward pass and optimizer step to reduce VRAM usage.",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler and warmup decay scheduler",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to output trained model",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="base name of trained model file",
    )
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files",
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="huggingface token",
    )
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private)",
    )
    parser.add_argument(
        "--save_state_to_huggingface",
        action="store_true",
        help="save state to huggingface",
    )
    parser.add_argument(
        "--resume_from_huggingface", action="store_true", help="resume from huggingface"
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints)",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed)",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps)",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="saved state to resume training",
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=None,
        help="save resumable checkpoint every N epochs (overwrites previous, auto-resumes on next run)",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="batch size for training",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225],
        help="max token length of text encoder (default for 75, 150 or 225)",
    )
    parser.add_argument(
        "--mem_eff_attn",
        action="store_true",
        help="use memory efficient attention for CrossAttention",
    )
    parser.add_argument(
        "--profile_steps",
        type=str,
        default=None,
        help="profile CUDA kernels for the given step range, e.g. '3-5'. Exports a Chrome trace to profile_trace.json",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="use torch.compile (requires PyTorch 2.0)",
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="inductor",
        choices=[
            "eager",
            "aot_eager",
            "inductor",
            "aot_ts_nvfuser",
            "nvprims_nvfuser",
            "cudagraphs",
            "ofi",
            "fx2trt",
            "onnxrt",
            "tensort",
            "ipex",
            "tvm",
        ],
        help="dynamo backend type (default is inductor)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="blocks",
        choices=["blocks", "full"],
        help="torch.compile mode: 'blocks' compiles each DiT block individually (default), "
        "'full' compiles the entire model as one graph for cross-block memory optimization "
        "(incompatible with gradient checkpointing and block swap)",
    )
    parser.add_argument(
        "--xformers", action="store_true", help="use xformers for CrossAttention"
    )
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0)",
    )
    parser.add_argument(
        "--vae", type=str, default=None, help="path to checkpoint of vae to replace"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1600,
        help="training steps",
    )
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps)",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=4,
        help="max num workers for DataLoader",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers",
    )
    parser.add_argument(
        "--dataloader_pin_memory", action="store_true", help="pin DataLoader memory"
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=4,
        help="prefetch_factor for DataLoader workers (only valid when num_workers>0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="enable gradient checkpointing",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision",
    )
    parser.add_argument(
        "--full_fp16", action="store_true", help="fp16 training including gradients"
    )
    parser.add_argument(
        "--full_bf16", action="store_true", help="bf16 training including gradients"
    )
    # FP8 is not supported yet — flag kept for CLI compatibility but force-disabled in assert_extra_args.
    parser.add_argument(
        "--fp8_base",
        action="store_true",
        help="(not supported yet) use fp8 for base model. This flag is force-disabled.",
    )

    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="DDP timeout (min, None for default of accelerate)",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="enable gradient_as_bucket_view for DDP",
    )
    parser.add_argument(
        "--ddp_static_graph", action="store_true", help="enable static_graph for DDP"
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1)",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional).",
    )
    parser.add_argument(
        "--log_config", action="store_true", help="log training configuration"
    )

    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended)",
    )
    parser.add_argument(
        "--noise_offset_random_strength",
        action="store_true",
        help="use random strength between 0~noise_offset for noise offset.",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)",
    )
    parser.add_argument(
        "--ip_noise_gamma",
        type=float,
        default=None,
        help="enable input perturbation noise. recommended value: around 0.1",
    )
    parser.add_argument(
        "--ip_noise_gamma_random_strength",
        action="store_true",
        help="Use random strength between 0~ip_noise_gamma for input perturbation noise.",
    )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default)",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for U-Net training (0~999, default is 0)",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000)",
    )
    parser.add_argument(
        "--t_min",
        type=float,
        default=None,
        help="Restrict training sigma range: minimum sigma (0.0~1.0).",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=None,
        help="Restrict training sigma range: maximum sigma (0.0~1.0). Default 1.0.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2", "huber", "smooth_l1", "pseudo_huber"],
        help="The type of loss function to use (L1, L2, Huber, smooth L1, or pseudo-Huber), default is L2",
    )
    parser.add_argument(
        "--huber_schedule",
        type=str,
        default="snr",
        choices=["constant", "exponential", "snr"],
        help="The scheduling method for Huber loss. default is snr",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.1,
        help="The Huber loss decay parameter. default is 0.1",
    )
    parser.add_argument(
        "--huber_scale",
        type=float,
        default=1.0,
        help="The Huber loss scale parameter. default is 1.0",
    )
    parser.add_argument(
        "--pseudo_huber_c",
        type=float,
        default=0.03,
        help="Pseudo-Huber loss parameter c. Small c -> L1-like, large c -> MSE-like. default is 0.03",
    )
    parser.add_argument(
        "--multiscale_loss_weight",
        type=float,
        default=0,
        help="Weight for 2x-downsampled multiscale loss term. 0 = disabled. default is 0",
    )
    parser.add_argument(
        "--lowram", action="store_true", help="enable low RAM optimization."
    )
    parser.add_argument(
        "--highvram", action="store_true", help="disable low VRAM optimization."
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps",
    )
    parser.add_argument(
        "--sample_at_first",
        action="store_true",
        help="generate sample images before training",
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps)",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images",
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help="sampler (scheduler) type for sample images",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="method name under configs/methods/ (e.g. 'tlora', 'hydralora', 'postfix'). Merged after preset so method settings win on overlap.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        help="hardware preset section name in configs/presets.toml (e.g. 'default', 'fast_16gb', 'low_vram').",
    )
    parser.add_argument(
        "--output_config",
        action="store_true",
        help="output command line args to given .toml file",
    )
    if support_dreambooth:
        parser.add_argument(
            "--prior_loss_weight",
            type=float,
            default=1.0,
            help="loss weight for regularization images",
        )


def add_masked_loss_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory",
    )
    parser.add_argument(
        "--masked_loss", action="store_true", help="apply mask for calculating loss."
    )


def add_dit_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs",
        action="store_true",
        help="cache text encoder outputs",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)",
    )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none", "uniform"],
        help="weighting scheme for timestep distribution. Default is uniform",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean for logit_normal weighting scheme",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std for logit_normal weighting scheme",
    )
    parser.add_argument(
        "--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme"
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="[EXPERIMENTAL] Sets the number of blocks to swap during the forward and backward passes.",
    )


def get_sanitized_config_or_none(args: argparse.Namespace):
    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        if k not in sensitive_args + sensitive_path_args:
            if (
                v is None
                or isinstance(v, bool)
                or isinstance(v, str)
                or isinstance(v, float)
                or isinstance(v, int)
            ):
                filtered_args[k] = v
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


def verify_command_line_training_args(args: argparse.Namespace):
    wandb_enabled = args.log_with is not None and args.log_with != "tensorboard"
    if not wandb_enabled:
        return

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]

    for arg in sensitive_args:
        if getattr(args, arg, None) is not None:
            logger.warning(
                f"wandb is enabled, but option `{arg}` is included in the command line. It is recommended to move it to the `.toml` file."
            )

    for arg in sensitive_path_args:
        if getattr(args, arg, None) is not None and os.path.isabs(getattr(args, arg)):
            logger.info(
                f"wandb is enabled, but option `{arg}` is included in the command line and it is an absolute path. It is recommended to move it to the `.toml` file or use relative path."
            )

    if getattr(args, "config_file", None) is not None:
        logger.info(
            "wandb is enabled, but option `config_file` is included in the command line. Please be careful about the information included in the path."
        )

    if (
        args.huggingface_repo_id is not None
        and args.huggingface_repo_visibility != "public"
    ):
        logger.info(
            "wandb is enabled, but option huggingface_repo_id is included in the command line and huggingface_repo_visibility is not 'public'. It is recommended to move it to the `.toml` file."
        )


def enable_high_vram(args: argparse.Namespace):
    if args.highvram:
        logger.info("highvram is enabled")
        _datasets_base.enable_high_vram()


def verify_training_args(args: argparse.Namespace):
    enable_high_vram(args)

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        logger.warning(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled"
        )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError("v_pred_like_loss cannot be enabled with v_parameterization")

    if args.zero_terminal_snr and not args.v_parameterization:
        logger.warning(
            "zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
        )

    if args.sample_every_n_epochs is not None and args.sample_every_n_epochs <= 0:
        logger.warning(
            "sample_every_n_epochs is less than or equal to 0, so it will be disabled"
        )
        args.sample_every_n_epochs = None

    if args.sample_every_n_steps is not None and args.sample_every_n_steps <= 0:
        logger.warning(
            "sample_every_n_steps is less than or equal to 0, so it will be disabled"
        )
        args.sample_every_n_steps = None


def add_dataset_arguments(
    parser: argparse.ArgumentParser,
    support_dreambooth: bool,
    support_caption: bool,
    support_caption_dropout: bool,
):
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="directory for train images"
    )
    parser.add_argument(
        "--cache_info",
        action="store_true",
        help="cache meta information for faster dataset loading",
    )
    parser.add_argument(
        "--shuffle_caption", action="store_true", help="shuffle separated caption"
    )
    parser.add_argument(
        "--caption_separator", type=str, default=",", help="separator for caption"
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".caption",
        help="extension of caption files",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility)",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens",
    )
    parser.add_argument(
        "--keep_tokens_separator",
        type=str,
        default="",
        help="A custom separator to divide the caption into fixed and flexible parts.",
    )
    parser.add_argument(
        "--secondary_separator",
        type=str,
        default=None,
        help="a secondary separator for caption.",
    )
    parser.add_argument(
        "--enable_wildcard", action="store_true", help="enable wildcard for caption"
    )
    parser.add_argument(
        "--caption_prefix", type=str, default=None, help="prefix for caption text"
    )
    parser.add_argument(
        "--caption_suffix", type=str, default=None, help="suffix for caption text"
    )
    parser.add_argument(
        "--color_aug", action="store_true", help="enable weak color augmentation"
    )
    parser.add_argument(
        "--flip_aug", action="store_true", help="enable horizontal flip augmentation"
    )
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0)",
    )
    parser.add_argument("--random_crop", action="store_true", help="enable random crop")
    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height')",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage",
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=1, help="batch size for caching latents"
    )
    parser.add_argument(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage",
    )
    parser.add_argument(
        "--skip_cache_check",
        action="store_true",
        help="skip the content validation of cache",
    )
    parser.add_argument(
        "--enable_bucket",
        action="store_true",
        help="enable buckets for multi aspect ratio training",
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=256,
        help="minimum resolution for buckets",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=1024,
        help="maximum resolution for buckets",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="make bucket for each image without upscaling",
    )
    parser.add_argument(
        "--resize_interpolation",
        type=str,
        default=None,
        choices=[
            "lanczos",
            "nearest",
            "bilinear",
            "linear",
            "bicubic",
            "cubic",
            "area",
        ],
        help="Resize interpolation",
    )
    parser.add_argument(
        "--token_warmup_min", type=int, default=1, help="start learning at N tags"
    )
    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps",
    )
    parser.add_argument(
        "--alpha_mask",
        action="store_true",
        help="use alpha channel as mask for training",
    )
    parser.add_argument(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class)",
    )

    if support_caption_dropout:
        parser.add_argument(
            "--caption_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout caption(0.0~1.0)",
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0)",
        )

    if support_dreambooth:
        parser.add_argument(
            "--reg_data_dir",
            type=str,
            default=None,
            help="directory for regularization images",
        )

    if support_caption:
        parser.add_argument(
            "--in_json", type=str, default=None, help="json metadata for dataset"
        )
        parser.add_argument(
            "--dataset_repeats",
            type=int,
            default=1,
            help="repeat dataset when training with captions",
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original)",
    )
    parser.add_argument(
        "--use_safetensors", action="store_true", help="use safetensors format to save"
    )


def _flatten_toml(d: dict) -> dict:
    """Flatten top-level sections into a single namespace (ignores nesting)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[kk] = vv
        else:
            out[k] = v
    return out


def _load_toml_with_base(path: str) -> dict:
    """Load a TOML file and recursively resolve its 'base_config' reference."""
    with open(path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)
    base_ref = config_dict.pop("base_config", None)
    if base_ref is None:
        return _flatten_toml(config_dict)
    if not os.path.isabs(base_ref):
        base_ref = os.path.join(os.path.dirname(path), base_ref)
    logger.info(f"Loading base config from {base_ref}...")
    base_dict = _load_toml_with_base(base_ref)
    merged = dict(base_dict)
    merged.update(_flatten_toml(config_dict))
    return merged


def load_preset_section(preset: str, configs_dir: str = "configs") -> dict:
    """Load a named preset section from configs/presets.toml."""
    path = os.path.join(configs_dir, "presets.toml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        presets = toml.load(f)
    if preset not in presets:
        raise KeyError(
            f"Preset '{preset}' not found in {path}. Available: {sorted(presets)}"
        )
    section = presets[preset]
    if not isinstance(section, dict):
        raise ValueError(f"Preset '{preset}' in {path} is not a table")
    return dict(section)


def load_method_preset(method: str, preset: str = "default", configs_dir: str = "configs") -> dict:
    """Merge base.toml → presets.toml[<preset>] → methods/<method>.toml into a flat dict.

    Method settings win over preset settings on overlap (e.g. postfix can force
    blocks_to_swap=0 regardless of the hardware preset).
    """
    base_path = os.path.join(configs_dir, "base.toml")
    method_path = os.path.join(configs_dir, "methods", f"{method}.toml")
    for p in (base_path, method_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config file not found: {p}")

    merged: dict = {}
    with open(base_path, "r", encoding="utf-8") as f:
        merged.update(_flatten_toml(toml.load(f)))
    merged.update(load_preset_section(preset, configs_dir))
    with open(method_path, "r", encoding="utf-8") as f:
        merged.update(_flatten_toml(toml.load(f)))
    return merged


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    # New-style chain: --method / --preset
    method = getattr(args, "method", None)
    preset = getattr(args, "preset", None) or "default"
    if method is not None and not args.config_file:
        logger.info(f"Loading chain: base → presets/{preset} → methods/{method}")
        try:
            merged = load_method_preset(method, preset)
        except FileNotFoundError as e:
            logger.error(str(e))
            exit(1)

        config_args = argparse.Namespace(**merged)
        args = parser.parse_args(namespace=config_args)
        args.config_file = os.path.join("configs", "methods", f"{method}.toml")
        return args

    if not args.config_file:
        return args

    config_path = (
        args.config_file + ".toml"
        if not args.config_file.endswith(".toml")
        else args.config_file
    )

    if args.output_config:
        if os.path.exists(config_path):
            logger.error("Config file already exists. Aborting...")
            exit(1)

        args_dict = vars(args)

        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        default_args = vars(parser.parse_args([]))

        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        logger.info("Saved config file")
        exit(0)

    if not os.path.exists(config_path):
        logger.info(f"{config_path} not found.")
        exit(1)

    logger.info(f"Loading settings from {config_path}...")
    merged = _load_toml_with_base(config_path)

    config_args = argparse.Namespace(**merged)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]

    return args


# endregion

# region utils


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        logger.info(f"resume training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return

    logger.info(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(
        f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}"
    )

    from huggingface_hub import list_repo_files

    list_files = list_repo_files(
        repo_id=repo_id,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )
    list_files = [f for f in list_files if f.startswith(path_in_repo)]

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        asyncio.gather(*[download(filename=filename) for filename in list_files])
    )
    if len(results) == 0:
        raise ValueError("No files found in the specified repo id")
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(",")])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert len(args.resolution) == 2, "resolution must be 'size' or 'width,height'"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple(
            [float(r) for r in args.face_crop_aug_range.split(",")]
        )
        assert (
            len(args.face_crop_aug_range) == 2
            and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), "face_crop_aug_range must be two floats"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            logger.warning(
                "latents in npz is ignored when color_aug or random_crop is True"
            )


def prepare_accelerator(args: argparse.Namespace):
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = (
            args.logging_dir
            + "/"
            + log_prefix
            + time.strftime("%Y%m%d%H%M%S", time.localtime())
        )

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard")
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    dynamo_backend = "NO"
    if args.torch_compile and not getattr(args, "static_token_count", None):
        # When static_token_count is set, we compile individual blocks instead
        # of the full forward (which has varying input H/W per bucket).
        dynamo_backend = args.dynamo_backend

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo"
                if os.name == "nt" or not torch.cuda.is_available()
                else "nccl",
                init_method=(
                    "env://?use_libuv=False"
                    if os.name == "nt"
                    and Version(torch.__version__) >= Version("2.4.0")
                    else None
                ),
                timeout=datetime.timedelta(minutes=args.ddp_timeout)
                if args.ddp_timeout
                else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
                static_graph=args.ddp_static_graph,
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_backend=dynamo_backend,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def patch_accelerator_for_fp16_training(accelerator):
    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def get_timesteps(
    min_timestep: int, max_timestep: int, b_size: int, device: torch.device
) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def get_huber_threshold_if_needed(
    args, timesteps: torch.Tensor, noise_scheduler
) -> Optional[torch.Tensor]:
    if args.loss_type == "pseudo_huber":
        b_size = timesteps.shape[0]
        return torch.full((b_size,), args.pseudo_huber_c, device=timesteps.device)
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError(
                "Huber schedule 'snr' is not supported with the current model."
            )
        alphas_cumprod = torch.index_select(
            noise_scheduler.alphas_cumprod, 0, timesteps.cpu()
        )
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full(
            (b_size,), args.huber_c * args.huber_scale, device=timesteps.device
        )
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
):
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = (
            2
            * huber_c
            * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        )
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "pseudo_huber":
        if huber_c is None:
            raise ValueError("pseudo_huber_c is required for pseudo_huber loss")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")
    names.append("text_encoder3")

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    lrs = lr_scheduler.get_last_lr()

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if (
            optimizer_type.lower().startswith("DAdapt".lower())
            or optimizer_type.lower() == "Prodigy".lower()
        ):
            logs["lr/d*lr/" + name] = (
                lr_scheduler.optimizers[-1].param_groups[lr_index]["d"]
                * lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

            m = re.match(r"ctr (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["cfg_trunc_ratio"] = float(m.group(1))
                continue

            m = re.match(r"rcfg (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["renorm_cfg"] = float(m.group(1))
                continue

            m = re.match(r"fs (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["flow_shift"] = m.group(1)
                continue

        except ValueError as ex:
            logger.error("Exception in parsing")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> List[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [
            line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"
        ]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [
            dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]
        ]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def init_trackers(
    accelerator: Accelerator, args: argparse.Namespace, default_tracker_name: str
):
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            default_tracker_name
            if args.log_tracker_name is None
            else args.log_tracker_name,
            config=get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

            wandb_tracker.define_metric("epoch", hidden=True)
            wandb_tracker.define_metric("val_step", hidden=True)

            wandb_tracker.define_metric("global_step", hidden=True)


# endregion
