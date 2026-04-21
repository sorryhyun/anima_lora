# common functions for training (stripped for Anima LoRA standalone).
#
# Most code lives in domain subpackages (library.datasets, library.training,
# library.runtime, library.config). This module keeps its own helpers
# (argparse/metadata/prompt utilities) and re-exports the subset of names
# still accessed via `train_util.X` from train.py, library.anima.training,
# library.config.loader, and the test suite.

import argparse
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import safetensors
import toml
import torch
from accelerate import Accelerator

from library.models import sai_spec as sai_model_spec
from library.runtime.accelerator import (  # noqa: F401
    prepare_accelerator,
    prepare_dtype,
    patch_accelerator_for_fp16_training,
    resume_from_local_or_hf_if_specified,
)
from library.log import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# Re-exports still accessed as `train_util.X` by train.py, library.anima.training,
# library.config.loader, or tests. Everything else should import directly from
# its home module.
from library.datasets import (  # noqa: F401, E402
    DreamBoothSubset,
    DreamBoothDataset,
    DatasetGroup,
    MinimalDataset,
    load_arbitrary_dataset,
    debug_dataset,
    collator_class,
    LossRecorder,
)

from library.training import (  # noqa: F401, E402
    # losses
    get_huber_threshold_if_needed,
    # cli args
    add_sd_models_arguments,
    add_optimizer_arguments,
    add_training_arguments,
    add_masked_loss_arguments,
    add_dit_training_arguments,
    add_dataset_arguments,
    verify_command_line_training_args,
    verify_training_args,
    get_sanitized_config_or_none,
    # metadata
    build_training_metadata,
    add_dataset_metadata,
    add_model_hash_metadata,
    finalize_metadata,
    # checkpoints
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
    get_checkpoint_state_dir,
    get_checkpoint_ckpt_name,
    save_checkpoint_state,
    # optimizers
    get_optimizer,
    get_optimizer_train_eval_fn,
    # schedulers
    get_scheduler_fix,
)

# ---------------------------------------------------------------------------
# Remaining code that stays in train_util
# (metadata, hashing, arg-parsing, accelerator setup, loss functions, etc.)
# ---------------------------------------------------------------------------

EPSILON = 1e-6


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


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


def get_sai_model_spec_dataclass(
    args: argparse.Namespace,
    lora: bool,
    optional_metadata: dict[str, str] | None = None,
) -> sai_model_spec.ModelSpecMetadata:
    timestamp = time.time()

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        timesteps = (
            args.min_timestep if args.min_timestep is not None else 0,
            args.max_timestep if args.max_timestep is not None else 1000,
        )
    else:
        timesteps = None

    extracted_metadata = {}
    for attr_name in dir(args):
        if attr_name.startswith("metadata_") and not attr_name.startswith(
            "metadata___"
        ):
            value = getattr(args, attr_name, None)
            if value is not None:
                field_name = attr_name[9:]
                if field_name not in {"title", "author", "description", "license", "tags"}:
                    extracted_metadata[field_name] = value
    if optional_metadata:
        extracted_metadata.update(optional_metadata)

    return sai_model_spec.build_metadata_dataclass(
        lora,
        timestamp,
        title=title,
        reso=args.resolution,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        optional_metadata=extracted_metadata or None,
    )




# Config loader helpers live in library.config.io; only the public entry points
# used via `train_util.X` are re-exported here. Private helpers (`_flatten_toml`,
# `_render_merged_toml`, etc.) should be imported directly from library.config.io.
from library.config.io import (  # noqa: F401, E402
    load_dataset_config_from_base,
    load_method_preset,
    read_config_from_file,
)



# endregion

# region utils


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



def get_timesteps(
    min_timestep: int, max_timestep: int, b_size: int, device: torch.device
) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps



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
