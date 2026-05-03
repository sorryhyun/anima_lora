# Anima LoRA training script (merged standalone)

import importlib
import argparse
import math
import os
import typing
import json
from dataclasses import dataclass
from typing import Any, Callable, Union, Optional
import sys
import random
import time
from multiprocessing import Value
from tqdm import tqdm

import torch
import torch.nn as nn
from library.runtime.device import clean_memory_on_device

from accelerate.utils import set_seed
from accelerate import Accelerator
from library import (
    train_util,
)
from library.anima import (
    models as anima_models,
    training as anima_train_utils,
    weights as anima_utils,
    strategy as strategy_anima,
    text_strategies,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.models import sai_spec as sai_model_spec
from library.runtime import noise as noise_utils
from library.config import loader as config_util
from library.config.loader import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
    ValidationBaseline,
    resolve_adapters,
)
from library.config.io import (
    load_dataset_config_from_base,
    read_config_from_file,
)
from library.datasets import (
    DatasetGroup,
    MinimalDataset,
    collator_class,
    debug_dataset,
    load_arbitrary_dataset,
)
from library.datasets import base as _datasets_base
from library.runtime.accelerator import (
    patch_accelerator_for_fp16_training,
    prepare_accelerator,
    prepare_dtype,
    resume_from_local_or_hf_if_specified,
)
from library.training import (
    CheckpointSaver,
    LossContext,
    SAMPLER_REGISTRY,
    SamplerContext,
    add_custom_train_arguments,
    add_dataset_arguments,
    add_dataset_metadata,
    add_dit_training_arguments,
    add_masked_loss_arguments,
    add_model_hash_metadata,
    add_network_arguments,
    add_optimizer_arguments,
    add_sd_models_arguments,
    add_training_arguments,
    build_loss_composer,
    build_training_metadata,
    finalize_metadata,
    get_huber_threshold_if_needed,
    get_optimizer,
    get_optimizer_train_eval_fn,
    get_scheduler_fix,
    save_state_on_train_end,
    verify_command_line_training_args,
    verify_training_args,
)
from library.training.bias_metric import measure_bias_trajectory
from library.training.loop import build_loop_state, run_training_loop
from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma
from library.log import setup_logging, add_logging_arguments

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainCtx:
    """Training-wide state built once near the top of train() and passed to
    per-step / per-batch methods instead of 15-arg parameter lists. Fields here
    are fixed for the whole training run — per-call values (epoch, global_step,
    progress_bar, logging keys, …) stay explicit at call sites."""

    args: Any
    accelerator: Accelerator
    network: Any
    unet: Any
    vae: Any
    text_encoders: list
    noise_scheduler: Any
    text_encoding_strategy: Any
    tokenize_strategy: Any
    vae_dtype: torch.dtype
    weight_dtype: torch.dtype
    train_text_encoder: bool
    train_unet: bool
    optimizer_eval_fn: Callable
    optimizer_train_fn: Callable
    is_tracking: bool


@dataclass(frozen=True)
class ValCtx:
    """Validation-wide state fixed for the entire training run. The per-call
    val_loss_recorder (step vs epoch) stays explicit since it differs per call
    site; everything else here is shared."""

    dataloader: Any
    sigmas: list
    steps: int
    total_steps: int
    train_loss_recorder: Any
    original_t_min: float
    original_t_max: float


class AnimaTrainer:
    def __init__(self):
        self.sample_prompts_te_outputs = None
        self._padding_mask_cache = {}
        # Per-method extensions (EasyControl, IP-Adapter, APEX, …). Resolved
        # from args+network in train() right after _create_and_apply_network.
        self._adapters: list[MethodAdapter] = []
        # Per-step aux dict — adapters' ``extra_forwards`` returns are merged
        # here in ``get_noise_pred_and_target`` and consumed by the loss
        # composer in ``_process_batch_inner``.
        self._extras_for_step: dict = {}
        # Set by ``_process_batch_inner`` when the split-backward path runs
        # both ``accelerator.backward`` calls inline. The train loop reads
        # this and skips its own outer backward to avoid double-stepping or
        # crashing on the detached return tensor.
        self._split_backward_consumed: bool = False

    # region logging helpers

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if (
                args.optimizer_type.lower().startswith("DAdapt".lower())
                or args.optimizer_type.lower() == "Prodigy".lower()
            ):
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"]
                    * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower())
                and optimizer is not None
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if (
                    args.optimizer_type.lower().startswith("DAdapt".lower())
                    or args.optimizer_type.lower() == "Prodigy".lower()
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"]
                        * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if (
                    args.optimizer_type.lower().endswith(
                        "ProdigyPlusScheduleFree".lower()
                    )
                    and optimizer is not None
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
                    )

        return logs

    def step_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        self.accelerator_logging(accelerator, logs, global_step, global_step, epoch)

    def epoch_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        self.accelerator_logging(accelerator, logs, epoch, global_step, epoch)

    def val_logging(
        self,
        accelerator: Accelerator,
        logs: dict,
        global_step: int,
        epoch: int,
        val_step: int,
    ):
        self.accelerator_logging(
            accelerator, logs, global_step + val_step, global_step, epoch, val_step
        )

    def accelerator_logging(
        self,
        accelerator: Accelerator,
        logs: dict,
        step_value: int,
        global_step: int,
        epoch: int,
        val_step: Optional[int] = None,
    ):
        """
        step_value is for tensorboard, other values are for wandb
        """
        tensorboard_tracker = None
        wandb_tracker = None
        other_trackers = []
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tensorboard_tracker = accelerator.get_tracker("tensorboard")
            elif tracker.name == "wandb":
                wandb_tracker = accelerator.get_tracker("wandb")
            else:
                other_trackers.append(accelerator.get_tracker(tracker.name))

        if tensorboard_tracker is not None:
            tensorboard_tracker.log(logs, step=step_value)

        if wandb_tracker is not None:
            logs["global_step"] = global_step
            logs["epoch"] = epoch
            if val_step is not None:
                logs["val_step"] = val_step
            wandb_tracker.log(logs)

        for tracker in other_trackers:
            tracker.log(logs, step=step_value)

    # endregion

    # region Anima-specific methods (from AnimaNetworkTrainer overrides)

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        # FP8 is not supported yet — force-disable all fp8 flags.
        if getattr(args, "fp8_base", False):
            logger.warning("fp8_base is not supported yet — disabling.")
            args.fp8_base = False
        if getattr(args, "fp8_base_unet", False):
            logger.warning("fp8_base_unet is not supported yet — disabling.")
            args.fp8_base_unet = False

        if (
            args.cache_text_encoder_outputs_to_disk
            and not args.cache_text_encoder_outputs
        ):
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), (
                "when caching Text Encoder output, token_warmup_step or caption_tag_dropout_rate cannot be used"
            )
            if getattr(args, "cache_llm_adapter_outputs", False):
                # Adapter output caching is only valid when the adapter is frozen (no LoRA on adapter).
                if args.network_args is not None and any(
                    "train_llm_adapter" in a and "true" in a.lower()
                    for a in args.network_args
                ):
                    raise ValueError(
                        "--cache_llm_adapter_outputs is incompatible with --network_args train_llm_adapter=True"
                    )
        else:
            assert not getattr(args, "cache_llm_adapter_outputs", False), (
                "--cache_llm_adapter_outputs requires --cache_text_encoder_outputs"
            )

        assert args.network_train_unet_only or not args.cache_text_encoder_outputs, (
            "network for Text Encoder cannot be trained with caching Text Encoder outputs"
        )

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, (
            "blocks_to_swap is not supported with cpu_offload_checkpointing"
        )

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning(
                    "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled"
                )
                args.gradient_checkpointing = True
            assert not args.cpu_offload_checkpointing, (
                "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            )
            assert args.blocks_to_swap is None or args.blocks_to_swap == 0, (
                "blocks_to_swap is not supported with unsloth_offload_checkpointing"
            )

        # Propagate inversion_dir to datasets for functional-loss supervision (postfix-func).
        inversion_dir = getattr(args, "inversion_dir", None)
        if inversion_dir:
            num_runs = getattr(args, "functional_loss_num_runs", 3)
            for dataset in train_dataset_group.datasets:
                dataset.inversion_dir = inversion_dir
                dataset.inversion_num_runs = num_runs
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.inversion_dir = inversion_dir
                    dataset.inversion_num_runs = num_runs

        # Propagate IP-Adapter / REPA feature-cache flag so datasets load
        # {stem}_anima_{encoder}.safetensors sidecars into batch["ip_features"].
        # REPA forces this on automatically — the alignment loss is meaningless
        # without the cached PE features as alignment targets.
        if getattr(args, "use_repa", False) and not getattr(
            args, "ip_features_cache_to_disk", False
        ):
            args.ip_features_cache_to_disk = True
            logger.info(
                "REPA: --use_repa set; forcing --ip_features_cache_to_disk=true. "
                "Run `make preprocess-pe` if you haven't cached PE features yet."
            )
        if getattr(args, "ip_features_cache_to_disk", False):
            ip_encoder = getattr(args, "ip_encoder", "pe")
            for dataset in train_dataset_group.datasets:
                dataset.ip_features_cache_to_disk = True
                dataset.ip_features_encoder = ip_encoder
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.ip_features_cache_to_disk = True
                    dataset.ip_features_encoder = ip_encoder

        train_dataset_group.verify_bucket_reso_steps(
            16
        )  # WanVAE spatial downscale = 8 and patch size = 2
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = (
            args.blocks_to_swap is not None and args.blocks_to_swap > 0
        )

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
            args.qwen3, dtype=weight_dtype, device="cpu"
        )
        qwen3_text_encoder.eval()

        # Load VAE
        logger.info("Loading Anima VAE...")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )
        vae.to(weight_dtype)
        vae.eval()

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(
        self, args, weight_dtype, accelerator, text_encoders
    ) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        if attn_mode == "flash4":
            # Flash Attention 4 (flash-attention-sm120) is not supported yet.
            raise RuntimeError(
                "attn_mode='flash4' is not supported yet — the flash-attention-sm120 "
                "kernel is disabled in this build. Use 'flash', 'torch', 'flex', "
                "'sageattn', or 'xformers' instead."
            )
        elif attn_mode == "flash":
            from networks.attention_dispatch import flash_attn, flash_attn_func

            if flash_attn_func is not None:
                logger.info(
                    f"Using Flash Attention 2 (flash_attn {flash_attn.__version__})"
                )
            else:
                raise RuntimeError(
                    "attn_mode='flash' requested but flash_attn is not available."
                )
        else:
            logger.info(f"Using attention mode: {attn_mode}")

        # Frozen LoRA: merged into DiT weights at load time (no runtime hooks).
        # Used by postfix/prefix runs that train on top of a fixed LoRA.
        lora_weights_list = None
        lora_multipliers = None
        if getattr(args, "lora_path", None):
            from safetensors.torch import load_file

            logger.info(
                f"merging frozen LoRA from {args.lora_path} into DiT weights "
                f"(multiplier={args.lora_multiplier})"
            )
            lora_sd = load_file(args.lora_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
            lora_weights_list = [lora_sd]
            lora_multipliers = [args.lora_multiplier]

        # Load DiT
        attn_softmax_scale = getattr(args, "attn_softmax_scale", None)
        logger.info(
            f"Loading Anima DiT model with split_attn: {args.split_attn}, attn_softmax_scale: {attn_softmax_scale}..."
        )
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
            attn_softmax_scale=attn_softmax_scale,
        )

        # FP8 base weights (fp8_base_unet) are not supported yet — the fp8 path is disabled.
        # if args.fp8_base_unet:
        #     from library.anima.models import quantize_to_fp8
        #     n = quantize_to_fp8(model)
        #     logger.info(f"fp8_base_unet: quantized {n} linear layers to float8_e4m3fn")

        # Bucketed KV trimming for cross-attention
        model.trim_crossattn_kv = getattr(args, "trim_crossattn_kv", False)

        # Static token count (constant-shape padding for torch.compile)
        if getattr(args, "static_token_count", None) is not None:
            model.set_static_token_count(args.static_token_count)
            if (
                args.torch_compile
                and getattr(args, "compile_mode", "blocks") == "blocks"
            ):
                model.compile_blocks(
                    args.dynamo_backend,
                    mode=getattr(args, "compile_inductor_mode", None),
                )
            logger.info(f"static_token_count={args.static_token_count}")

        # Store unsloth preference so that when the base trainer calls
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        # Block swap
        self.is_swapping_blocks = (
            args.blocks_to_swap is not None and args.blocks_to_swap > 0
        )
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                False,
                cache_llm_adapter_outputs=getattr(
                    args, "cache_llm_adapter_outputs", False
                ),
                use_shuffled_caption_variants=getattr(
                    args, "use_shuffled_caption_variants", False
                ),
            )
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_noise_scheduler(
        self, args: argparse.Namespace, device: torch.device
    ) -> Any:
        noise_scheduler = noise_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        ctx: "TrainCtx",
        latents,
        batch,
        text_encoder_conds,
        *,
        is_train=True,
    ):
        args = ctx.args
        accelerator = ctx.accelerator
        noise_scheduler = ctx.noise_scheduler
        unet = ctx.unet
        network = ctx.network
        weight_dtype = ctx.weight_dtype
        anima: anima_models.Anima = unet

        # Reset per-step adapter aux so stale tensors from a prior step can't
        # leak into the loss composer.
        self._extras_for_step = {}

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        # Method-adapter pre-forward priming. IP-Adapter encodes the reference
        # image and primes per-block K/V; EasyControl runs the cond pre-pass
        # and primes per-block (K_c, V_c). Both run on the 4D latent layout
        # the patched DiT forward expects. The patched cross-attn / self-attn
        # closures consume the primed tensors during attention.
        if self._adapters:
            step_ctx = StepCtx(
                args=args,
                accelerator=accelerator,
                network=network,
                weight_dtype=weight_dtype,
            )
            for adapter in self._adapters:
                adapter.prime_for_forward(step_ctx, batch, latents, is_train=is_train)
        noise = torch.randn_like(latents)

        # Draw noisy input + timesteps via the sampler registry (M1).
        sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
        sampler_out = sampler_fn(
            SamplerContext(
                args=args,
                noise_scheduler=noise_scheduler,
                latents=latents,
                noise=noise,
                device=accelerator.device,
                weight_dtype=weight_dtype,
            )
        )
        noisy_model_input = sampler_out.noisy_input
        timesteps = sampler_out.timesteps  # [0,1]-scaled, float32
        sigmas = sampler_out.sigmas

        # Set timestep-dependent rank mask on LoRA and ReFT modules
        if hasattr(network, "set_timestep_mask"):
            network.set_timestep_mask(timesteps, max_timestep=1.0)
        if hasattr(network, "set_reft_timestep_mask"):
            network.set_reft_timestep_mask(timesteps, max_timestep=1.0)
        # σ-conditional HydraLoRA router (Track B, timestep-hydra.md). No-op
        # unless use_sigma_router is on and the variant is hydra/ortho_hydra.
        if hasattr(network, "set_sigma"):
            network.set_sigma(timesteps)
        # HydraLoRA expert-warmup: during the first ``expert_warmup_ratio`` of
        # training, only one randomly-chosen expert per module receives
        # gradient (forward still uses all experts via the learned gate).
        # No-op unless expert_warmup_ratio > 0.
        if is_train and hasattr(network, "step_expert_warmup"):
            network.step_expert_warmup(
                int(getattr(self, "_hydra_warmup_step", 0)),
                int(getattr(args, "max_train_steps", 0) or 0),
            )
            if hasattr(network, "step_balance_loss_warmup"):
                network.step_balance_loss_warmup(
                    int(getattr(self, "_hydra_warmup_step", 0)),
                    int(getattr(args, "max_train_steps", 0) or 0),
                )
            self._hydra_warmup_step = int(getattr(self, "_hydra_warmup_step", 0)) + 1

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            # Only require grads for text conditions when training the text encoder.
            # When using cached text encoder outputs (or training DiT-only), requiring grads here adds backward work.
            if self.is_train_text_encoder(args) and not args.cache_text_encoder_outputs:
                for t in text_encoder_conds:
                    if t is not None and t.dtype.is_floating_point:
                        t.requires_grad_(True)

        # Unpack text encoder conditions
        crossattn_emb = None
        if len(text_encoder_conds) == 5:
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                text_encoder_conds
            )
        else:
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds

        # Pre-compute max sequence length on CPU to avoid GPU sync in KV trimming
        _max_crossattn_seqlen = None
        if args.trim_crossattn_kv and t5_attn_mask is not None:
            _max_crossattn_seqlen = int(t5_attn_mask.sum(dim=-1).max())

        if crossattn_emb is None:
            # Move to device
            prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
            attn_mask = attn_mask.to(accelerator.device)
            t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
            t5_attn_mask = t5_attn_mask.to(accelerator.device)
        else:
            crossattn_emb = crossattn_emb.to(accelerator.device, dtype=weight_dtype)
            if args.trim_crossattn_kv or hasattr(network, "append_postfix"):
                t5_attn_mask = t5_attn_mask.to(accelerator.device)

        # On-device caption dropout. The freshly-transferred GPU tensors are
        # not aliased to the dataloader's CPU copies, so we can write in-place
        # — no clones, no main-thread CPU memcpy on the critical path.
        caption_dropout_rates = (
            batch.get("caption_dropout_rates") if isinstance(batch, dict) else None
        )
        if caption_dropout_rates is not None:
            if crossattn_emb is None:
                ctx.text_encoding_strategy.apply_caption_dropout_inplace(
                    caption_dropout_rates,
                    prompt_embeds=prompt_embeds,
                    attn_mask=attn_mask,
                    t5_input_ids=t5_input_ids,
                    t5_attn_mask=t5_attn_mask,
                )
            else:
                # In this branch prompt_embeds / attn_mask / t5_input_ids stay
                # on CPU because they're unused downstream — only zero what the
                # model actually consumes (and only touch t5_attn_mask if it
                # was moved to device above).
                ctx.text_encoding_strategy.apply_caption_dropout_inplace(
                    caption_dropout_rates,
                    crossattn_emb=crossattn_emb,
                    t5_attn_mask=(
                        t5_attn_mask
                        if t5_attn_mask is not None and t5_attn_mask.is_cuda
                        else None
                    ),
                )

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask_key = (bs, h_latent, w_latent, weight_dtype, accelerator.device)
        padding_mask = self._padding_mask_cache.get(padding_mask_key)
        if padding_mask is None:
            padding_mask = torch.zeros(
                bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device
            )
            self._padding_mask_cache[padding_mask_key] = padding_mask

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(
            2
        )  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            if crossattn_emb is None:
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )
            else:
                # crossattn_emb is already in target (T5-compatible) space
                # Prefix/postfix mode: inject learned vectors before DiT forward.
                # Pool text BEFORE injection so modulation guidance sees only real text.
                has_prefix_postfix = getattr(
                    network, "mode", None
                ) == "prefix" or hasattr(network, "append_postfix")
                kw = {}
                if has_prefix_postfix:
                    kw["pooled_text_override"] = crossattn_emb.max(dim=1).values
                if getattr(network, "mode", None) == "prefix":
                    crossattn_emb = network.prepend_prefix(crossattn_emb)
                elif hasattr(network, "append_postfix"):
                    seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
                    crossattn_emb = network.append_postfix(
                        crossattn_emb, seqlens, timesteps=timesteps
                    )
                if args.trim_crossattn_kv:
                    kw["crossattn_seqlens"] = t5_attn_mask.sum(dim=-1).to(torch.int32)
                    max_cs = _max_crossattn_seqlen
                    if has_prefix_postfix:
                        kw["crossattn_seqlens"] = (
                            kw["crossattn_seqlens"] + network.num_postfix_tokens
                        )
                        if max_cs is not None:
                            max_cs += network.num_postfix_tokens
                    kw["max_crossattn_seqlen"] = max_cs
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    crossattn_emb,
                    padding_mask=padding_mask,
                    **kw,
                )

                # Method-adapter extra forwards (APEX fake/mix branches, …).
                # Each adapter sees the primary forward's inputs + 5D output
                # and may run additional anima(...) calls inside this same
                # autocast / grad scope, returning aux loss tensors keyed for
                # the LossComposer.
                if self._adapters:
                    primary = ForwardArtifacts(
                        anima_call=anima,
                        noisy_model_input=noisy_model_input,
                        timesteps=timesteps,
                        crossattn_emb=crossattn_emb,
                        padding_mask=padding_mask,
                        forward_kwargs=kw,
                        model_pred=model_pred,
                        noise=noise,
                        latents=latents,
                        is_train=is_train,
                    )
                    step_ctx = StepCtx(
                        args=args,
                        accelerator=accelerator,
                        network=network,
                        weight_dtype=weight_dtype,
                    )
                    for adapter in self._adapters:
                        out = adapter.extra_forwards(step_ctx, primary)
                        if out:
                            self._extras_for_step.update(out)

                # --- Functional MSE loss against stochastic inversion run ---
                # If functional loss is enabled and the batch has inversions loaded,
                # run a second no-grad forward with a sampled inversion run as
                # crossattn_emb and compute MSE between the two sets of cross_attn
                # output_proj captures at the configured blocks.
                self._func_loss = None
                inv_runs = (
                    batch.get("inversion_runs") if isinstance(batch, dict) else None
                )
                inv_mask = (
                    batch.get("inversion_mask") if isinstance(batch, dict) else None
                )
                if (
                    is_train
                    and getattr(self, "_func_blocks", None)
                    and inv_runs is not None
                    and inv_mask is not None
                    and bool(inv_mask.any().item())
                ):
                    # Snapshot main-forward captures (still attached to postfix MLP graph)
                    cap_main = dict(self._func_captures)
                    missing = [bi for bi in self._func_blocks if bi not in cap_main]
                    if missing:
                        raise RuntimeError(
                            f"Functional loss: main forward did not populate captures for blocks {missing}"
                        )

                    # Sample one run per batch element
                    inv_runs_dev = inv_runs.to(accelerator.device, dtype=weight_dtype)
                    inv_mask_dev = inv_mask.to(accelerator.device)
                    B_inv, N_runs, _, _ = inv_runs_dev.shape
                    run_idx = torch.randint(
                        0, N_runs, (B_inv,), device=inv_runs_dev.device
                    )
                    sampled_inv = inv_runs_dev[
                        torch.arange(B_inv, device=inv_runs_dev.device), run_idx
                    ]  # [B, S, D]

                    # Same pooled_text_override so AdaLN modulation is identical;
                    # only cross-attn K/V differs between the two forwards.
                    inv_kw = {}
                    if has_prefix_postfix and "pooled_text_override" in kw:
                        inv_kw["pooled_text_override"] = kw["pooled_text_override"]

                    with torch.no_grad():
                        _ = anima(
                            noisy_model_input,
                            timesteps,
                            sampled_inv,
                            padding_mask=padding_mask,
                            **inv_kw,
                        )

                    cap_inv = {
                        bi: self._func_captures[bi].detach() for bi in self._func_blocks
                    }

                    mask_f = inv_mask_dev.float()
                    denom = mask_f.sum().clamp(min=1.0)
                    block_losses = []
                    for bi in self._func_blocks:
                        diff = cap_main[bi].float() - cap_inv[bi].float()
                        per_sample = diff.pow(2).mean(
                            dim=tuple(range(1, diff.ndim))
                        )  # [B]
                        block_losses.append((per_sample * mask_f).sum() / denom)
                    self._func_loss = sum(block_losses) / len(block_losses)
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Note: do NOT clear timestep mask here -- gradient checkpointing recomputes the forward
        # pass during backward, so the mask must remain set. It gets overwritten on the next step.

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )

        return model_pred, target, timesteps, weighting

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
    ):
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()
        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # Re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            return accelerator.prepare(unet)

        model = unet
        model = accelerator.prepare(
            model, device_placement=[not self.is_swapping_blocks]
        )
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(
            accelerator.device
        )
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model

    def on_validation_step_end(self, ctx: "TrainCtx", batch):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            ctx.accelerator.unwrap_model(ctx.unet).prepare_block_swap_before_forward()

    def process_batch(
        self,
        ctx: "TrainCtx",
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        """Override base process_batch to surface caption_dropout_rates for on-device dropout."""

        # The cached text-encoder outputs list arrives as
        # [..., caption_dropout_rates] from the dataset (see strategy.py
        # cache layout). Split the trailing rates tensor off so the inner
        # path sees the canonical 4- or 5-element conds list, and stash the
        # rates on the batch — get_noise_pred_and_target applies the dropout
        # in-place after the H2D transfer. Doing it here on CPU would clone
        # prompt_embeds / crossattn_emb on the critical path before the H2D
        # copy, blocking the main thread.
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            encoder_outputs = text_encoder_outputs_list[:-1]
            # Shallow copy so the original list (with rates appended) stays
            # intact for validation's per-sigma loop that reuses the batch.
            batch = {
                **batch,
                "text_encoder_outputs_list": encoder_outputs,
                "caption_dropout_rates": caption_dropout_rates,
            }

        return self._process_batch_inner(ctx, batch, is_train=is_train)

    def _process_batch_inner(
        self,
        ctx: "TrainCtx",
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        """
        Process a batch for the network (original NetworkTrainer.process_batch logic)
        """
        args = ctx.args
        accelerator = ctx.accelerator
        network = ctx.network
        vae = ctx.vae
        text_encoders = ctx.text_encoders
        text_encoding_strategy = ctx.text_encoding_strategy
        tokenize_strategy = ctx.tokenize_strategy
        noise_scheduler = ctx.noise_scheduler
        vae_dtype = ctx.vae_dtype
        weight_dtype = ctx.weight_dtype
        train_text_encoder = ctx.train_text_encoder
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(
                    torch.FloatTensor, batch["latents"].to(accelerator.device)
                )
            else:
                if (
                    args.vae_batch_size is None
                    or len(batch["images"]) <= args.vae_batch_size
                ):
                    latents = self.encode_images_to_latents(
                        args,
                        vae,
                        batch["images"].to(accelerator.device, dtype=vae_dtype),
                    )
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size]
                        for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(
                                args, vae, chunk.to(accelerator.device, dtype=vae_dtype)
                            )
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(
                        torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents)
                    )

            latents = self.shift_scale_latents(args, latents)

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = (
                text_encoder_outputs_list  # List of text encoder outputs
            )

        if (
            len(text_encoder_conds) == 0
            or text_encoder_conds[0] is None
            or train_text_encoder
        ):
            with (
                torch.set_grad_enabled(is_train and train_text_encoder),
                accelerator.autocast(),
            ):
                if args.weighted_captions:
                    input_ids_list, weights_list = (
                        tokenize_strategy.tokenize_with_weights(batch["captions"])
                    )
                    encoded_text_encoder_conds = (
                        text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(
                                args, accelerator, text_encoders
                            ),
                            input_ids_list,
                            weights_list,
                        )
                    )
                else:
                    input_ids = [
                        ids.to(accelerator.device) for ids in batch["input_ids_list"]
                    ]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(
                            args, accelerator, text_encoders
                        ),
                        input_ids,
                    )
                if args.full_fp16:
                    encoded_text_encoder_conds = [
                        c.to(weight_dtype) for c in encoded_text_encoder_conds
                    ]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            ctx,
            latents,
            batch,
            text_encoder_conds,
            is_train=is_train,
        )

        huber_c = get_huber_threshold_if_needed(args, timesteps, noise_scheduler)

        # Assemble aux dict for the composer: extra_forwards returns from each
        # method adapter (APEX tensors + schedule already mixed in, etc.) plus
        # the trainer-owned functional-loss capture (next adapter target).
        loss_aux: dict = dict(self._extras_for_step)

        func_loss = getattr(self, "_func_loss", None)
        if func_loss is not None:
            loss_aux["func_loss"] = func_loss

        composer = build_loss_composer(args, getattr(self, "_network", network))

        def _build_loss_ctx(aux: dict) -> LossContext:
            return LossContext(
                args=args,
                batch=batch,
                model_pred=noise_pred,
                target=target,
                timesteps=timesteps,
                weighting=weighting,
                huber_c=huber_c,
                loss_weights=batch["loss_weights"],
                network=getattr(self, "_network", network),
                aux=aux,
            )

        # Split-backward: APEX runs two grad-tracked DiT forwards per step
        # (real branch via L_mix, fake branch via L_fake) whose autograd
        # graphs are disjoint — forward 3's input is built from
        # ``model_pred.detach()``. Composing+backwarding both as one scalar
        # keeps both graphs live until backward, roughly doubling peak
        # activation memory. When an adapter opts in, backward the real
        # branch inline so forward-1 activations are freed before forward 3
        # runs, then run forward 3 + L_fake. Total gradient = real + fake.
        split_backward = is_train and any(
            a.wants_split_backward(is_train=is_train) for a in self._adapters
        )

        # APEX warmup: ``lam_f_eff <= 0`` means ``apex_fake`` short-circuits
        # to a no-graph zero tensor and forward 3 is not needed at all. Fall
        # back to the legacy single-pass compose so the trainer's outer
        # backward sees a graph from apex_mix. Also catches the case where
        # the adapter returned no aux (e.g. crossattn_emb is None).
        if split_backward:
            apex_aux = loss_aux.get("apex") or {}
            if float(apex_aux.get("lam_f_eff", 0.0)) <= 0.0:
                split_backward = False

        if not split_backward:
            return composer.compose(_build_loss_ctx(loss_aux))

        # --- real branch ---
        loss_real = composer.compose_real_branch(_build_loss_ctx(loss_aux))
        # accelerator.backward handles gradient_accumulation scaling; a second
        # backward in the same accumulate scope just deposits more gradient
        # into .grad.
        accelerator.backward(loss_real)

        # --- deferred fake branch (forward 3 + L_fake) ---
        step_ctx = StepCtx(
            args=args,
            accelerator=accelerator,
            network=network,
            weight_dtype=weight_dtype,
        )
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            for adapter in self._adapters:
                if not adapter.wants_split_backward(is_train=is_train):
                    continue
                out = adapter.extra_forwards_fake(step_ctx)
                if not out:
                    continue
                # Merge into loss_aux: extend nested dicts (e.g. "apex") rather
                # than overwriting the real-branch keys T_mix_v / lam_inner_eff.
                for k, v in out.items():
                    if (
                        k in loss_aux
                        and isinstance(loss_aux[k], dict)
                        and isinstance(v, dict)
                    ):
                        loss_aux[k] = {**loss_aux[k], **v}
                    else:
                        loss_aux[k] = v

        loss_fake = composer.compose_fake_branch(_build_loss_ctx(loss_aux))
        if loss_fake.requires_grad:
            accelerator.backward(loss_fake)
        # Tell the train loop we've already consumed both branches' graphs;
        # the returned tensor is detached and only carries the composite scalar
        # for logging / metrics.
        self._split_backward_consumed = True
        return loss_real.detach() + loss_fake.detach()

    # endregion

    # region Methods only in NetworkTrainer (not overridden by Anima)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        self._network = (
            network  # composer reads _network for ortho / balance regularizers
        )
        self._func_loss = None
        self._func_hooks = []
        self._func_captures = {}
        self._func_blocks = []
        if getattr(args, "functional_loss_weight", 0.0) > 0.0 and getattr(
            args, "inversion_dir", None
        ):
            blocks_str = getattr(args, "functional_loss_blocks", "8,12,16,20")
            try:
                self._func_blocks = sorted(
                    int(b.strip()) for b in blocks_str.split(",") if b.strip()
                )
            except ValueError as e:
                raise ValueError(
                    f"functional_loss_blocks must be comma-separated integers, got {blocks_str!r}"
                ) from e

            def _make_hook(block_idx: int):
                def _hook(_module, _inputs, output):
                    # Save the cross_attn.output_proj output for this block.
                    # Hook fires twice per step (main forward + inversion forward);
                    # the main forward runs first, we snapshot before second forward overwrites.
                    self._func_captures[block_idx] = output

                return _hook

            blocks_list = unet.blocks  # nn.ModuleList of 28 Anima DiT blocks
            num_blocks = len(blocks_list)
            for bi in self._func_blocks:
                if not (0 <= bi < num_blocks):
                    raise ValueError(
                        f"functional_loss_blocks contains out-of-range index {bi} (model has {num_blocks} blocks)"
                    )
                module = blocks_list[bi].cross_attn.output_proj
                h = module.register_forward_hook(_make_hook(bi))
                self._func_hooks.append(h)
            logger.info(
                f"Functional loss enabled: hooks on cross_attn.output_proj at blocks {self._func_blocks}, "
                f"weight={args.functional_loss_weight}, num_runs={args.functional_loss_num_runs}"
            )

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(
            args, lora=True
        ).to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def prepare_text_encoder_fp8(
        self, index, text_encoder, te_weight_dtype, weight_dtype
    ):
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def get_text_encoders_train_flags(self, args, text_encoders):
        return (
            [True] * len(text_encoders)
            if self.is_train_text_encoder(args)
            else [False] * len(text_encoders)
        )

    def on_step_start(self, ctx: "TrainCtx", batch, *, is_train: bool = True):
        if not self._adapters:
            return
        step_ctx = StepCtx(
            args=ctx.args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )
        for adapter in self._adapters:
            adapter.on_step_start(step_ctx, batch, is_train=is_train)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cast_text_encoder(self, args):
        return True

    def cast_vae(self, args):
        return True

    def cast_unet(self, args):
        return not getattr(args, "fp8_base_unet", False)

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        **kwargs,
    ):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator: Accelerator,
        unet,
        vae,
        text_encoders,
        dataset: DatasetGroup,
        weight_dtype,
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # We cannot move DiT to CPU because of block swap, so only move VAE
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            llm_adapter = None
            models_for_cache = text_encoders
            if getattr(args, "cache_llm_adapter_outputs", False):
                logger.info("Loading LLM adapter for caching outputs...")
                llm_adapter = anima_utils.load_llm_adapter(
                    args.pretrained_model_name_or_path,
                    args.llm_adapter_path,
                    dtype=weight_dtype,
                    device=accelerator.device,
                )
                models_for_cache = [text_encoders[0], llm_adapter]

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(models_for_cache, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(
                    f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}"
                )

                tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
                text_encoding_strategy = (
                    text_strategies.TextEncodingStrategy.get_strategy()
                )

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [
                            prompt_dict.get("prompt", ""),
                            prompt_dict.get("negative_prompt", ""),
                        ]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = (
                                    text_encoding_strategy.encode_tokens(
                                        tokenize_strategy,
                                        text_encoders,
                                        tokens_and_masks,
                                    )
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            if llm_adapter is not None:
                logger.info("move LLM adapter back to cpu")
                llm_adapter.to("cpu")

            # move text encoder back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder to device for encoding during training/validation
            text_encoders[0].to(accelerator.device)

    # endregion

    # region Main training loop

    @staticmethod
    def _parse_profile_steps(args) -> tuple[int, int] | None:
        """Parse --profile_steps 'start-end' into (start, end) or None.

        When set, the loop calls ``torch.cuda.profiler.start()`` at ``start``
        and ``stop()`` after ``end``, so pair this with::

            nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
                accelerate launch ... train.py --profile_steps 3-5
        """
        raw = getattr(args, "profile_steps", None)
        if not raw:
            return None
        if "-" in raw:
            a, b = raw.split("-", 1)
            return int(a), int(b)
        n = int(raw)
        return n, n + 2

    @staticmethod
    def _switch_rng_state(
        seed: int,
    ) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()

        torch.manual_seed(seed)
        random.seed(seed)

        return (cpu_rng_state, gpu_rng_state, python_rng_state)

    @staticmethod
    def _restore_rng_state(
        rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple],
    ):
        cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state)
        random.setstate(python_rng_state)

    def _prepare_dataset(self, args):
        """Build train/val dataset groups and the collator shared by both loaders."""
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(
                ConfigSanitizer(support_dropout=True)
            )
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                base_ds = load_dataset_config_from_base(overrides=vars(args))
                if base_ds is not None:
                    logger.info("Loading dataset config from configs/base.toml")
                    user_config = base_ds
                    use_user_config = True
                elif use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            # Global --sample_ratio override (used by the `[half]` preset).
            sample_ratio = getattr(args, "sample_ratio", None)
            if sample_ratio is not None:
                for ds in user_config.get("datasets", []):
                    for sub in ds.get("subsets", []):
                        sub["sample_ratio"] = sample_ratio
                logger.info(f"Applied --sample_ratio={sample_ratio} to all subsets")

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = (
                config_util.generate_dataset_group_by_blueprint(
                    blueprint.dataset_group,
                    constant_token_buckets=getattr(args, "static_token_count", None)
                    is not None,
                )
            )

            rates = [
                subset.caption_dropout_rate
                for ds in train_dataset_group.datasets
                for subset in ds.subsets
            ]
            if rates and any(r > 0 for r in rates):
                logger.info(f"caption dropout ENABLED — per-subset rates: {rates}")
            else:
                logger.info("caption dropout DISABLED (rate=0.0 on all subsets)")
        else:
            # use arbitrary dataset class
            train_dataset_group = load_arbitrary_dataset(args)
            val_dataset_group = (
                None  # placeholder until validation dataset supported for arbitrary
            )

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        return (
            train_dataset_group,
            val_dataset_group,
            current_epoch,
            current_step,
            collator,
            use_user_config,
            use_dreambooth_method,
        )

    def _create_and_apply_network(
        self,
        args,
        accelerator,
        vae,
        text_encoder,
        unet,
        text_encoders,
        weight_dtype,
    ):
        """Import network module, merge base weights, build LoRA, apply to the model."""
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                if (
                    args.base_weights_multiplier is None
                    or len(args.base_weights_multiplier) <= i
                ):
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(
                    f"merging module: {weight_path} with multiplier {multiplier}"
                )

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(
                    text_encoder,
                    unet,
                    weights_sd,
                    weight_dtype,
                    accelerator.device if args.lowram else "cpu",
                )

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        # Forward known network-arg keys from top-level config (TOML) to net_kwargs.
        # CLI --network_args take precedence over top-level config keys.
        # Source of truth: `networks.all_network_kwargs()` (union of
        # `SHARED_KWARG_FLAGS` and each `NetworkSpec.kwarg_flags`), plus a
        # small tail of top-level training args the network modules still
        # want to read (e.g. postfix contrastive's step-boundary window).
        for key in NETWORK_KWARG_ALLOWLIST + _EXTRA_FORWARDED_TOP_LEVEL_ARGS:
            if (
                key not in net_kwargs
                and hasattr(args, key)
                and getattr(args, key) is not None
            ):
                net_kwargs[key] = str(getattr(args, key))

        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(
                1, args.network_weights, vae, text_encoder, unet, **net_kwargs
            )
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(
            network, "apply_max_norm_regularization"
        ):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it"
            )
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(
                f"load network weights from {args.network_weights}: {info}"
            )

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(
                text_encoders, self.get_text_encoders_train_flags(args, text_encoders)
            ):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()  # may have no effect

        return network, net_kwargs, train_unet, train_text_encoder

    def _setup_optimizer_and_dataloader(
        self,
        args,
        accelerator,
        network,
        train_dataset_group,
        val_dataset_group,
        collator,
    ):
        """Build optimizer, dataloaders, and LR scheduler; finalize max_train_steps."""
        accelerator.print("prepare optimizer, data loader etc.")

        # make backward compatibility for text_encoder_lr
        support_multiple_lrs = hasattr(
            network, "prepare_optimizer_params_with_multiple_te_lrs"
        )
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            if (
                args.text_encoder_lr is None
                or isinstance(args.text_encoder_lr, float)
                or isinstance(args.text_encoder_lr, int)
            ):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = (
                    None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
                )
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            else:
                results = network.prepare_optimizer_params(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError:
            trainable_params = network.prepare_optimizer_params(
                text_encoder_lr, args.unet_lr
            )
            lr_descriptions = None

        optimizer_name, optimizer_args, optimizer = get_optimizer(
            args, trainable_params
        )
        optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(
            optimizer, args
        )

        # prepare dataloader
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        persistent_workers = args.persistent_data_loader_workers and n_workers > 0

        dataloader_kwargs = {
            "batch_size": 1,
            "collate_fn": collator,
            "num_workers": n_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        if n_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            shuffle=True,
            **dataloader_kwargs,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            **dataloader_kwargs,
        )

        # Calculate training steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader)
                / accelerator.num_processes
                / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is"
            )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr scheduler
        lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

        return (
            optimizer,
            optimizer_name,
            optimizer_args,
            optimizer_train_fn,
            optimizer_eval_fn,
            text_encoder_lr,
            lr_descriptions,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )

    def _prepare_with_accelerator(
        self,
        args,
        accelerator,
        network,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        unet,
        text_encoders,
        text_encoder,
        vae,
        vae_dtype,
        weight_dtype,
        train_unet,
        train_text_encoder,
        cache_latents,
    ):
        """Cast model dtypes, run accelerator.prepare, flip train/eval, optional torch.compile."""
        # full fp16/bf16 training
        if args.full_fp16:
            assert args.mixed_precision == "fp16", (
                "full_fp16 requires mixed precision='fp16'"
            )
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed precision='bf16'"
            )
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype

        unet.requires_grad_(False)
        if self.cast_unet(args):
            unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu" and self.cast_text_encoder(args):
                t_enc.to(dtype=te_weight_dtype)

                # nn.Embedding not support FP8
                if te_weight_dtype != weight_dtype:
                    self.prepare_text_encoder_fp8(
                        i, t_enc, te_weight_dtype, weight_dtype
                    )

        # accelerator preparation (no deepspeed)
        if train_unet:
            unet = self.prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            unet.to(
                accelerator.device,
                dtype=unet_weight_dtype if self.cast_unet(args) else None,
            )
        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(
                    text_encoders,
                    self.get_text_encoders_train_flags(args, text_encoders),
                )
            ]
            if len(text_encoders) > 1:
                text_encoder = text_encoders
            else:
                text_encoder = text_encoders[0]
        # else: text_encoder is unchanged; device and dtype are already set above

        network, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
            accelerator.prepare(
                network, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
        )
        training_model = network

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for i, (t_enc, frag) in enumerate(
                zip(
                    text_encoders,
                    self.get_text_encoders_train_flags(args, text_encoders),
                )
            ):
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        # compile_mode='full': narrow torch.compile to _run_blocks (the constant-
        # shape block stack). Pre-blocks (patch/embed/static-pad/RoPE-pad/t_embedder/
        # BlockMask) and post-blocks (unpad/final_layer/unpatchify) stay eager —
        # their shapes vary per CONSTANT_TOKEN_BUCKETS entry, so wrapping them
        # would force one CUDAGraph per bucket. Pinning the compile boundary to
        # the shape-invariant region yields a single CUDAGraph across all buckets.
        if args.torch_compile and getattr(args, "compile_mode", "blocks") == "full":
            assert not args.gradient_checkpointing, (
                "compile_mode='full' is incompatible with gradient checkpointing"
            )
            assert not self.is_swapping_blocks, (
                "compile_mode='full' is incompatible with block swap"
            )
            inductor_mode = getattr(args, "compile_inductor_mode", None)
            # Compile on the unwrapped DiT so the instance-bound method sticks
            # regardless of accelerator wrapping (DDP/etc resolve self._run_blocks
            # against the underlying module's __dict__).
            accelerator.unwrap_model(unet).compile_core(
                backend=args.dynamo_backend, mode=inductor_mode
            )
            logger.info(
                f"compile_core: _run_blocks compiled "
                f"(backend={args.dynamo_backend}, mode={inductor_mode})"
            )

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # patch for fp16 grad scale
        if args.full_fp16:
            patch_accelerator_for_fp16_training(accelerator)

        return (
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            training_model,
            unet,
            text_encoders,
            text_encoder,
            unet_weight_dtype,
        )

    def _run_validation(
        self,
        ctx: "TrainCtx",
        val: "ValCtx",
        *,
        val_loss_recorder,
        epoch,
        global_step,
        progress_bar,
        progress_desc,
        postfix_label,
        log_avg_key,
        log_div_key,
        logging_fn,
    ):
        """Run a validation pass over val.dataloader x val.sigmas."""
        args = ctx.args
        accelerator = ctx.accelerator
        ctx.optimizer_eval_fn()
        accelerator.unwrap_model(ctx.network).eval()
        unwrapped_unet = accelerator.unwrap_model(ctx.unet)
        if hasattr(unwrapped_unet, "switch_block_swap_for_inference"):
            unwrapped_unet.switch_block_swap_for_inference()
        rng_states = self._switch_rng_state(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        # Adapter-supplied "without-this-method" baselines (e.g. IP-Adapter's
        # "no_ip"). For each (val_step, sigma) we'll re-run process_batch with
        # the adapter perturbed and log the delta — this is what isolates the
        # method's actual contribution when the primary FM loss is dominated
        # by paths the method shortcircuits.
        baselines: list[ValidationBaseline] = []
        for adapter in self._adapters:
            baselines.extend(adapter.validation_baselines())

        val_progress_bar = tqdm(
            range(val.total_steps * (1 + len(baselines))),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc=progress_desc,
        )
        val_timesteps_step = 0
        per_sigma_losses = {s: [] for s in val.sigmas}
        per_baseline_per_sigma: dict[str, dict[float, list[float]]] = {
            b.name: {s: [] for s in val.sigmas} for b in baselines
        }

        def _cudagraph_step_begin() -> None:
            if not self._cudagraph_mark_step:
                return
            net_unwrapped = accelerator.unwrap_model(ctx.network)
            if hasattr(net_unwrapped, "clear_step_caches"):
                net_unwrapped.clear_step_caches()
            torch.compiler.cudagraph_mark_step_begin()

        for val_step, batch in enumerate(val.dataloader):
            if val_step >= val.steps:
                break

            for sigma in val.sigmas:
                self.on_step_start(ctx, batch, is_train=False)

                # Pin sigma via t_min/t_max (what the noise function reads)
                args.t_min = args.t_max = sigma

                # Snapshot RNG so each baseline forward sees the same noise
                # the primary did — only the adapter state differs, so the
                # delta is the adapter's contribution rather than noise.
                rng_pre_primary = (
                    (
                        torch.get_rng_state(),
                        torch.cuda.get_rng_state(),
                        random.getstate(),
                    )
                    if baselines
                    else None
                )

                # Mirror the training loop's cudagraph step-begin marker so
                # cudagraph_trees doesn't re-record / leak a memory pool on
                # each val forward (see the train-loop call site for context).
                _cudagraph_step_begin()

                loss = self.process_batch(ctx, batch, is_train=False)

                current_loss = loss.detach().item()
                val_loss_recorder.add(
                    epoch=epoch, step=val_timesteps_step, loss=current_loss
                )
                per_sigma_losses[sigma].append(current_loss)
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(
                    {
                        postfix_label: val_loss_recorder.moving_average,
                        "sigma": f"{sigma:.2f}",
                    }
                )

                self.on_validation_step_end(ctx, batch)
                val_timesteps_step += 1

                # --- Baseline sweep: same (batch, sigma, noise), perturbed
                # adapter state. Logged separately as `<prefix>_baseline_<n>`.
                if baselines:
                    rng_post_primary = (
                        torch.get_rng_state(),
                        torch.cuda.get_rng_state(),
                        random.getstate(),
                    )
                    for baseline in baselines:
                        # Rewind RNG to pre-primary so the baseline draws the
                        # same noise (and any other randomness) the primary
                        # saw — without this, the delta is dominated by
                        # noise variance not the adapter's contribution.
                        torch.set_rng_state(rng_pre_primary[0])
                        torch.cuda.set_rng_state(rng_pre_primary[1])
                        random.setstate(rng_pre_primary[2])

                        baseline.enter()
                        try:
                            _cudagraph_step_begin()
                            bl_loss = (
                                self.process_batch(ctx, batch, is_train=False)
                                .detach()
                                .item()
                            )
                        finally:
                            baseline.exit()
                        per_baseline_per_sigma[baseline.name][sigma].append(bl_loss)
                        val_progress_bar.update(1)
                        val_progress_bar.set_postfix(
                            {
                                postfix_label: val_loss_recorder.moving_average,
                                "sigma": f"{sigma:.2f}",
                                f"Δ_{baseline.name}": f"{bl_loss - current_loss:+.4f}",
                            }
                        )
                        self.on_validation_step_end(ctx, batch)
                    # Resume the RNG stream where the primary left it so the
                    # next (val_step, sigma) draws fresh noise as before.
                    torch.set_rng_state(rng_post_primary[0])
                    torch.cuda.set_rng_state(rng_post_primary[1])
                    random.setstate(rng_post_primary[2])

        if ctx.is_tracking:
            loss_validation_divergence = (
                val_loss_recorder.moving_average
                - val.train_loss_recorder.moving_average
            )
            logs = {
                log_avg_key: val_loss_recorder.moving_average,
                log_div_key: loss_validation_divergence,
            }
            for s, losses in per_sigma_losses.items():
                if losses:
                    logs[f"loss/validation/sigma_{s:.2f}"] = sum(losses) / len(losses)

            # Baseline aggregates. Prefix mirrors the primary key
            # ("loss/validation/step_average" → "loss/validation/step") so
            # step- and epoch-validation get distinct baseline series.
            primary_avg = val_loss_recorder.moving_average
            base_prefix = log_avg_key.removesuffix("_average")
            for b in baselines:
                per_sigma = per_baseline_per_sigma[b.name]
                all_losses = [v for losses in per_sigma.values() for v in losses]
                if not all_losses:
                    continue
                bl_avg = sum(all_losses) / len(all_losses)
                logs[f"{base_prefix}_baseline_{b.name}"] = bl_avg
                logs[f"{base_prefix}_baseline_{b.name}_delta"] = bl_avg - primary_avg
                for s, losses in per_sigma.items():
                    if losses:
                        logs[f"{base_prefix}_baseline_{b.name}/sigma_{s:.2f}"] = sum(
                            losses
                        ) / len(losses)

            logging_fn(accelerator, logs, global_step, epoch + 1)

        self._restore_rng_state(rng_states)
        args.t_min = val.original_t_min
        args.t_max = val.original_t_max
        ctx.optimizer_train_fn()
        accelerator.unwrap_model(ctx.network).train()
        if hasattr(unwrapped_unet, "switch_block_swap_for_training"):
            unwrapped_unet.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)
        progress_bar.unpause()

    _BIAS_INCOMPATIBLE_ADAPTERS = frozenset({"apex", "ip_adapter", "easycontrol"})

    def _run_bias_validation(
        self,
        ctx: "TrainCtx",
        val: "ValCtx",
        *,
        epoch: int,
        global_step: int,
        progress_bar,
        log_prefix: str,
        logging_fn,
    ) -> None:
        """SNR-t bias diagnostic — paired forward/reverse velocity-norm gap
        across the inference schedule, averaged over the val dataloader.

        The headline scalar (``<prefix>/integrated_signed_gap`` over the late
        half) tracks the same bias DCW patches at sampler-time, so it
        correlates with sample quality where FM-MSE val loss does not. See
        ``docs/methods/dcw.md``.

        LoRA family only — incompatible adapters short-circuit with a one-time
        warning. Requires cached ``crossattn_emb`` (LoRA training default);
        batches without it are skipped.
        """
        args = ctx.args
        if not getattr(args, "validation_bias_metric", False):
            return

        skipped = sorted(
            getattr(a, "name", "?")
            for a in self._adapters
            if getattr(a, "name", None) in self._BIAS_INCOMPATIBLE_ADAPTERS
        )
        if skipped:
            if not getattr(self, "_bias_skip_warned", False):
                ctx.accelerator.print(
                    f"[bias-metric] disabled — incompatible adapter(s) active: {skipped}"
                )
                self._bias_skip_warned = True
            return

        accelerator = ctx.accelerator
        device = accelerator.device
        weight_dtype = ctx.weight_dtype
        unwrapped_net = accelerator.unwrap_model(ctx.network)
        unwrapped_unet = accelerator.unwrap_model(ctx.unet)

        progress_bar.pause() if hasattr(progress_bar, "pause") else None
        ctx.optimizer_eval_fn()
        unwrapped_net.eval()
        if hasattr(unwrapped_unet, "switch_block_swap_for_inference"):
            unwrapped_unet.switch_block_swap_for_inference()
        # Inference semantics: full-rank T-LoRA / ReFT (training masks are
        # not applied at sampling time; see networks/CLAUDE.md).
        if hasattr(unwrapped_net, "clear_timestep_mask"):
            unwrapped_net.clear_timestep_mask()
        rng_states = self._switch_rng_state(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        num_steps = int(getattr(args, "validation_bias_steps", 12))
        flow_shift = float(getattr(args, "flow_shift", 1.0))
        val_seed_base = int(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        sum_v_fwd = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_rev = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_fwd_LL = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_rev_LL = torch.zeros(num_steps, dtype=torch.float64)
        n_batches = 0
        skipped_no_crossattn = 0

        bias_pbar = tqdm(
            total=val.steps,
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc=f"{log_prefix.split('/')[-1]} bias",
        )
        try:
            for val_step, batch in enumerate(val.dataloader):
                if val_step >= val.steps:
                    break

                latents = batch["latents"]
                if latents is None:
                    bias_pbar.update(1)
                    continue
                latents = latents.to(device, dtype=weight_dtype)
                if latents.ndim == 4:
                    latents = latents.unsqueeze(2)  # (B, C, 1, H, W)

                teo = batch.get("text_encoder_outputs_list")
                if teo is None or len(teo) < 6 or teo[4] is None:
                    skipped_no_crossattn += 1
                    bias_pbar.update(1)
                    continue
                crossattn_emb = teo[4].to(device, dtype=weight_dtype)
                t5_attn_mask = teo[3]
                if t5_attn_mask is not None:
                    t5_attn_mask = t5_attn_mask.to(device)

                B, _, _, h_lat, w_lat = latents.shape
                pm_key = (B, h_lat, w_lat, weight_dtype, device)
                padding_mask = self._padding_mask_cache.get(pm_key)
                if padding_mask is None:
                    padding_mask = torch.zeros(
                        B, 1, h_lat, w_lat, dtype=weight_dtype, device=device
                    )
                    self._padding_mask_cache[pm_key] = padding_mask

                fwd_kwargs: dict = {}
                if getattr(args, "trim_crossattn_kv", False) and t5_attn_mask is not None:
                    seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
                    fwd_kwargs["crossattn_seqlens"] = seqlens
                    fwd_kwargs["max_crossattn_seqlen"] = int(seqlens.max())

                def forward_fn(x_t, sigma_b, _emb=crossattn_emb, _pm=padding_mask, _kw=fwd_kwargs):
                    set_hydra_sigma(unwrapped_unet, sigma_b)
                    with accelerator.autocast():
                        return unwrapped_unet(x_t, sigma_b, _emb, padding_mask=_pm, **_kw)

                seed = val_seed_base + 1000 * val_step
                traj = measure_bias_trajectory(
                    forward_fn,
                    latents,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    noise_seed=seed,
                )
                sum_v_fwd += traj["v_fwd"]
                sum_v_rev += traj["v_rev"]
                sum_v_fwd_LL += traj["v_fwd_LL"]
                sum_v_rev_LL += traj["v_rev_LL"]
                n_batches += 1
                bias_pbar.update(1)
                bias_pbar.set_postfix_str(
                    f"|gap|={float(traj['gap'].abs().sum()):.2f}"
                )
        finally:
            bias_pbar.close()
            clear_hydra_sigma(unwrapped_unet)

        self._restore_rng_state(rng_states)
        ctx.optimizer_train_fn()
        unwrapped_net.train()
        if hasattr(unwrapped_unet, "switch_block_swap_for_training"):
            unwrapped_unet.switch_block_swap_for_training()
        clean_memory_on_device(device)
        if hasattr(progress_bar, "unpause"):
            progress_bar.unpause()

        if skipped_no_crossattn and not getattr(self, "_bias_no_crossattn_warned", False):
            accelerator.print(
                f"[bias-metric] skipped {skipped_no_crossattn} batch(es) without "
                "cached crossattn_emb — set cache_text_encoder_outputs=true "
                "(default for LoRA) to enable."
            )
            self._bias_no_crossattn_warned = True

        if n_batches == 0 or not ctx.is_tracking:
            return

        v_fwd = sum_v_fwd / n_batches
        v_rev = sum_v_rev / n_batches
        v_fwd_LL = sum_v_fwd_LL / n_batches
        v_rev_LL = sum_v_rev_LL / n_batches
        gap = v_rev - v_fwd
        gap_LL = v_rev_LL - v_fwd_LL
        late = num_steps // 2

        # Stash the final aggregated trajectory so the end-of-training
        # recipe-write hook can reuse it without re-running validation
        # (see _maybe_calibrate_dcw_recipe). Sigmas come from the same
        # get_timesteps_sigmas call used inside measure_bias_trajectory.
        from library.inference.sampling import get_timesteps_sigmas

        _, sig_t = get_timesteps_sigmas(num_steps, flow_shift, device)
        self._last_bias_validation = {
            "gap_LL": gap_LL.tolist(),
            "sigmas": sig_t.cpu()[:num_steps].to(torch.float64).tolist(),
            "num_steps": num_steps,
            "flow_shift": flow_shift,
            "n_batches": n_batches,
        }

        logs: dict = {
            f"{log_prefix}/integrated_signed_gap": float(gap.sum()),
            f"{log_prefix}/integrated_abs_gap": float(gap.abs().sum()),
            f"{log_prefix}/integrated_signed_gap_late": float(gap[late:].sum()),
            f"{log_prefix}/integrated_abs_gap_late": float(gap[late:].abs().sum()),
            f"{log_prefix}/integrated_signed_gap_LL_late": float(gap_LL[late:].sum()),
            f"{log_prefix}/integrated_abs_gap_LL_late": float(gap_LL[late:].abs().sum()),
        }
        for i in range(num_steps):
            logs[f"{log_prefix}/gap_step_{i:02d}"] = float(gap[i])
            logs[f"{log_prefix}/gap_LL_step_{i:02d}"] = float(gap_LL[i])
            logs[f"{log_prefix}/v_fwd_step_{i:02d}"] = float(v_fwd[i])
            logs[f"{log_prefix}/v_rev_step_{i:02d}"] = float(v_rev[i])
        logging_fn(accelerator, logs, global_step, epoch + 1)

    def _run_bias_LL_pass(
        self,
        ctx: "TrainCtx",
        val: "ValCtx",
        *,
        num_steps: int,
        flow_shift: float,
        val_seed_base: int,
        dcw_probe_lambda: float,
        dcw_probe_schedule: str,
        pbar: "tqdm | None" = None,
        pbar_label: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """One sweep of the val dataloader returning summed LL-band
        ``v_fwd`` / ``v_rev`` trajectories (length ``num_steps``, float64
        on CPU) and the count of usable batches. With
        ``dcw_probe_lambda=0.0`` this is a baseline pass; with a nonzero λ
        it's the calibration probe pass.

        When ``pbar`` is provided, advances by 1 per val batch (skipped or
        not) and updates the postfix with ``pbar_label`` plus the running
        mean late-half ``|gap_LL|`` from this pass so far. Caller owns the
        bar's lifecycle.

        Caller is responsible for putting models in eval mode, switching
        block-swap, clearing timestep masks, and restoring RNG / train
        state — this just iterates and accumulates.
        """
        args = ctx.args
        accelerator = ctx.accelerator
        device = accelerator.device
        weight_dtype = ctx.weight_dtype
        unwrapped_unet = accelerator.unwrap_model(ctx.unet)

        sum_v_fwd_LL = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_rev_LL = torch.zeros(num_steps, dtype=torch.float64)
        n_batches = 0
        late = num_steps // 2
        try:
            for val_step, batch in enumerate(val.dataloader):
                if val_step >= val.steps:
                    break
                latents = batch["latents"]
                if latents is None:
                    if pbar is not None:
                        pbar.update(1)
                    continue
                latents = latents.to(device, dtype=weight_dtype)
                if latents.ndim == 4:
                    latents = latents.unsqueeze(2)

                teo = batch.get("text_encoder_outputs_list")
                if teo is None or len(teo) < 6 or teo[4] is None:
                    if pbar is not None:
                        pbar.update(1)
                    continue
                crossattn_emb = teo[4].to(device, dtype=weight_dtype)
                t5_attn_mask = teo[3]
                if t5_attn_mask is not None:
                    t5_attn_mask = t5_attn_mask.to(device)

                B, _, _, h_lat, w_lat = latents.shape
                pm_key = (B, h_lat, w_lat, weight_dtype, device)
                padding_mask = self._padding_mask_cache.get(pm_key)
                if padding_mask is None:
                    padding_mask = torch.zeros(
                        B, 1, h_lat, w_lat, dtype=weight_dtype, device=device
                    )
                    self._padding_mask_cache[pm_key] = padding_mask

                fwd_kwargs: dict = {}
                if getattr(args, "trim_crossattn_kv", False) and t5_attn_mask is not None:
                    seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
                    fwd_kwargs["crossattn_seqlens"] = seqlens
                    fwd_kwargs["max_crossattn_seqlen"] = int(seqlens.max())

                def forward_fn(x_t, sigma_b, _emb=crossattn_emb, _pm=padding_mask, _kw=fwd_kwargs):
                    set_hydra_sigma(unwrapped_unet, sigma_b)
                    with accelerator.autocast():
                        return unwrapped_unet(x_t, sigma_b, _emb, padding_mask=_pm, **_kw)

                seed = val_seed_base + 1000 * val_step
                traj = measure_bias_trajectory(
                    forward_fn,
                    latents,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    noise_seed=seed,
                    dcw_probe_lambda=dcw_probe_lambda,
                    dcw_probe_schedule=dcw_probe_schedule,
                )
                sum_v_fwd_LL += traj["v_fwd_LL"]
                sum_v_rev_LL += traj["v_rev_LL"]
                n_batches += 1
                if pbar is not None:
                    pbar.update(1)
                    mean_gap_LL = (sum_v_rev_LL - sum_v_fwd_LL) / n_batches
                    abs_gap_late = float(mean_gap_LL[late:].abs().sum())
                    pbar.set_postfix_str(
                        f"{pbar_label} |gap_LL|_late={abs_gap_late:.2f} "
                        f"({n_batches} batch{'es' if n_batches != 1 else ''})"
                    )
        finally:
            clear_hydra_sigma(unwrapped_unet)

        return sum_v_fwd_LL, sum_v_rev_LL, n_batches

    def _calibrate_dcw_recipe(
        self,
        ctx: "TrainCtx",
        val: "ValCtx",
        *,
        metadata: dict,
    ) -> None:
        """End-of-training: solve a per-LoRA LL-band DCW recipe and stuff
        it into the safetensors metadata.

        Iterates the val dataloader once with a small λ_LL=−ε DCW probe
        injected into the reverse rollout, computes per-step
        ``gap_LL_probe``, then closed-form solves for the late-half
        optimum λ_LL* against a baseline ``gap_LL``. The baseline is
        reused from the most recent ``--validation_bias_metric`` pass
        when available, otherwise an extra non-probe pass is run inline
        — so calibration works whether or not bias-metric validation was
        enabled during training.

        Skipped silently when the active adapter is bias-incompatible
        (APEX / IP-Adapter / EasyControl) or the val pass produces zero
        usable batches. See ``docs/proposal/lora-dcw-proposal.md``.
        """
        args = ctx.args
        if not getattr(args, "calibrate_dcw_recipe", True):
            return

        skipped = sorted(
            getattr(a, "name", "?")
            for a in self._adapters
            if getattr(a, "name", None) in self._BIAS_INCOMPATIBLE_ADAPTERS
        )
        if skipped:
            return

        accelerator = ctx.accelerator
        if not accelerator.is_main_process:
            return

        device = accelerator.device
        unwrapped_net = accelerator.unwrap_model(ctx.network)
        unwrapped_unet = accelerator.unwrap_model(ctx.unet)

        ctx.optimizer_eval_fn()
        unwrapped_net.eval()
        if hasattr(unwrapped_unet, "switch_block_swap_for_inference"):
            unwrapped_unet.switch_block_swap_for_inference()
        if hasattr(unwrapped_net, "clear_timestep_mask"):
            unwrapped_net.clear_timestep_mask()
        rng_states = self._switch_rng_state(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        anchors = list(
            getattr(args, "dcw_calibration_anchors", [-0.015, -0.022, -0.030])
        )
        anchors = [float(a) for a in anchors if float(a) != 0.0]
        if not anchors:
            accelerator.print(
                "[dcw-calibration] no nonzero anchors configured — skipping"
            )
            return
        schedule = getattr(args, "dcw_calibration_schedule", "one_minus_sigma")
        val_seed_base = int(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        flow_shift = float(getattr(args, "flow_shift", 1.0))
        num_steps = int(
            getattr(args, "dcw_calibration_steps", None)
            or getattr(args, "validation_bias_steps", 12)
        )

        # Cached bias-metric baseline (cheap per-cycle diagnostic) is only
        # reusable when its step count matches calibration's. Otherwise σ
        # samples differ and the late-half gap is not comparable across
        # the baseline-vs-anchor passes the selector reads.
        baseline = getattr(self, "_last_bias_validation", None)
        if baseline is not None and baseline["num_steps"] != num_steps:
            baseline = None

        anchor_gaps: list[tuple[float, list[float]]] = []
        n_batches = 0
        n_passes = (1 if baseline is None else 0) + len(anchors)
        cal_pbar = tqdm(
            total=n_passes * val.steps,
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc=f"dcw-calib (steps={num_steps})",
        )
        try:
            if baseline is None:
                accelerator.print(
                    f"[dcw-calibration] running baseline val pass at "
                    f"num_steps={num_steps} before {len(anchors)} anchor pass(es)"
                )
                fwd_LL_base, rev_LL_base, n_batches_base = self._run_bias_LL_pass(
                    ctx,
                    val,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    val_seed_base=val_seed_base,
                    dcw_probe_lambda=0.0,
                    dcw_probe_schedule=schedule,
                    pbar=cal_pbar,
                    pbar_label="baseline",
                )
                if n_batches_base == 0:
                    return
                gap_LL_base_list = (
                    (rev_LL_base - fwd_LL_base) / n_batches_base
                ).tolist()
                from library.inference.sampling import get_timesteps_sigmas

                _, sig_t = get_timesteps_sigmas(num_steps, flow_shift, device)
                baseline = {
                    "gap_LL": gap_LL_base_list,
                    "sigmas": sig_t.cpu()[:num_steps].to(torch.float64).tolist(),
                    "num_steps": num_steps,
                    "flow_shift": flow_shift,
                    "n_batches": n_batches_base,
                }

            for lam in anchors:
                sum_v_fwd_LL, sum_v_rev_LL, n_batches = self._run_bias_LL_pass(
                    ctx,
                    val,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    val_seed_base=val_seed_base,
                    dcw_probe_lambda=lam,
                    dcw_probe_schedule=schedule,
                    pbar=cal_pbar,
                    pbar_label=f"λ={lam:+.4f}",
                )
                if n_batches == 0:
                    accelerator.print(
                        f"[dcw-calibration] anchor λ={lam:+.4f} produced no usable "
                        "batches — skipping recipe"
                    )
                    return
                gap_LL_at_lam = (
                    (sum_v_rev_LL - sum_v_fwd_LL) / n_batches
                ).tolist()
                anchor_gaps.append((lam, gap_LL_at_lam))
        finally:
            cal_pbar.close()
            self._restore_rng_state(rng_states)
            ctx.optimizer_train_fn()
            unwrapped_net.train()
            if hasattr(unwrapped_unet, "switch_block_swap_for_training"):
                unwrapped_unet.switch_block_swap_for_training()
            clean_memory_on_device(device)

        from library.inference.dcw_calibration import select_best_anchor_LL

        try:
            selected = select_best_anchor_LL(
                baseline["gap_LL"],
                anchor_gaps,
                baseline["sigmas"],
                schedule=schedule,
            )
        except ValueError as exc:
            accelerator.print(
                f"[dcw-calibration] selector failed: {exc} — skipping recipe"
            )
            return

        recipe_payload = {
            "lambda_LL": selected["lambda_LL"],
            "schedule_LL": selected["schedule_LL"],
        }
        late_idx = num_steps // 2
        gap_LL_base = baseline["gap_LL"]
        provenance = {
            "version": 2,  # bump: was linear-response solver, now anchor selection
            "band": "LL",
            "anchors": anchors,
            "num_steps": num_steps,
            "n_batches": n_batches,
            "n_batches_baseline": baseline.get("n_batches", n_batches),
            "baseline_signed_gap_LL_late": float(sum(gap_LL_base[late_idx:])),
            "baseline_abs_gap_LL_late": float(
                sum(abs(g) for g in gap_LL_base[late_idx:])
            ),
            "baseline_gap_LL_late_loss": selected["baseline_gap_LL_late"],
            "best_gap_LL_late_loss": selected["best_gap_LL_late"],
            "candidate_losses": [
                {"lambda": c["lambda"], "loss": c["loss"]}
                for c in selected["candidates"]
            ],
            "num_late_steps": selected["num_late_steps"],
        }

        metadata["ss_dcw_recipe"] = json.dumps(
            recipe_payload, separators=(",", ":")
        )
        metadata["ss_dcw_calibration"] = json.dumps(
            provenance, separators=(",", ":")
        )

        cand_str = " ".join(
            f"λ={c['lambda']:+.4f}→{c['loss']:.1f}"
            for c in selected["candidates"]
        )
        winner_note = (
            "baseline (no DCW)" if selected["baseline_was_best"]
            else f"λ={selected['lambda_LL']:+.4f}"
        )
        accelerator.print(
            f"[dcw-calibration] selected {winner_note} ({selected['schedule_LL']}); "
            f"candidates: {cand_str}; n_batches={n_batches}"
        )

    def train(self, args):
        from networks.methods.apex import (
            promote_warmstart_to_merge as _apex_promote_warmstart_to_merge,
        )

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        verify_training_args(args)
        _apex_promote_warmstart_to_merge(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Whether inductor will have CUDAGraphs active — governs whether the
        # training loop needs to call torch.compiler.cudagraph_mark_step_begin()
        # each step (see the call site inside the accumulate block).
        self._cudagraph_mark_step = bool(
            getattr(args, "torch_compile", False)
            and getattr(args, "compile_inductor_mode", None)
            in ("reduce-overhead", "max-autotune")
        )

        tokenize_strategy = self.get_tokenize_strategy(args)
        text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(
            tokenize_strategy
        )  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        text_strategies.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        (
            train_dataset_group,
            val_dataset_group,
            current_epoch,
            current_step,
            collator,
            use_user_config,
            use_dreambooth_method,
        ) = self._prepare_dataset(args)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images)"
            )
            return

        if cache_latents:
            assert train_dataset_group.is_latent_cacheable(), (
                "when caching latents, either color_aug or random_crop cannot be used"
            )
            if val_dataset_group is not None:
                assert val_dataset_group.is_latent_cacheable(), (
                    "when caching latents, either color_aug or random_crop cannot be used"
                )

        self.assert_extra_args(
            args, train_dataset_group, val_dataset_group
        )  # may change some args

        # Prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precision dtype
        weight_dtype, save_dtype = prepare_dtype(args)
        vae_dtype = (
            (torch.float32 if args.no_half_vae else weight_dtype)
            if self.cast_vae(args)
            else None
        )

        # load target models: unet may be None for lazy loading
        model_version, text_encoder, vae, unet = self.load_target_model(
            args, weight_dtype, accelerator
        )
        if vae_dtype is None:
            vae_dtype = vae.dtype
            logger.info(
                f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false"
            )

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )

        # prepare dataset for latents caching if needed
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)
            if val_dataset_group is not None:
                val_dataset_group.new_cache_latents(vae, accelerator)

            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        text_strategies.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = (
            self.get_text_encoder_outputs_caching_strategy(args)
        )
        if text_encoder_outputs_caching_strategy is not None:
            text_strategies.TextEncoderOutputsCachingStrategy.set_strategy(
                text_encoder_outputs_caching_strategy
            )
        self.cache_text_encoder_outputs_if_needed(
            args,
            accelerator,
            unet,
            vae,
            text_encoders,
            train_dataset_group,
            weight_dtype,
        )
        if val_dataset_group is not None:
            self.cache_text_encoder_outputs_if_needed(
                args,
                accelerator,
                unet,
                vae,
                text_encoders,
                val_dataset_group,
                weight_dtype,
            )

        if unet is None:
            # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
            unet, text_encoders = self.load_unet_lazily(
                args, weight_dtype, accelerator, text_encoders
            )

        network_result = self._create_and_apply_network(
            args, accelerator, vae, text_encoder, unet, text_encoders, weight_dtype
        )
        if network_result is None:
            return
        network, net_kwargs, train_unet, train_text_encoder = network_result

        # Resolve and run on_network_built for each method adapter (EasyControl,
        # IP-Adapter, APEX, …). Each adapter validates its runtime contract and
        # logs/sets up auxiliary state before optimizer / accelerator wiring.
        self._adapters = resolve_adapters(args, network)
        if self._adapters:
            setup_ctx = SetupCtx(
                args=args,
                accelerator=accelerator,
                network=network,
                unet=unet,
                text_encoders=text_encoders,
                weight_dtype=weight_dtype,
            )
            for adapter in self._adapters:
                adapter.on_network_built(setup_ctx)

        (
            optimizer,
            optimizer_name,
            optimizer_args,
            optimizer_train_fn,
            optimizer_eval_fn,
            text_encoder_lr,
            lr_descriptions,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = self._setup_optimizer_and_dataloader(
            args,
            accelerator,
            network,
            train_dataset_group,
            val_dataset_group,
            collator,
        )

        (
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            training_model,
            unet,
            text_encoders,
            text_encoder,
            unet_weight_dtype,
        ) = self._prepare_with_accelerator(
            args,
            accelerator,
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            unet,
            text_encoders,
            text_encoder,
            vae,
            vae_dtype,
            weight_dtype,
            train_unet,
            train_text_encoder,
            cache_latents,
        )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = (
                math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
            )

        total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        accelerator.print("running training")
        accelerator.print("  num train images * repeats")
        accelerator.print("  num validation images * repeats")
        accelerator.print("  num reg images")
        accelerator.print("  num batches per epoch")
        accelerator.print("  num epochs")
        accelerator.print("  batch size per device")
        accelerator.print("  gradient accumulation steps")
        accelerator.print("  total optimization steps")

        metadata = build_training_metadata(
            args,
            session_id=session_id,
            training_started_at=training_started_at,
            text_encoder_lr=text_encoder_lr,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            model_version=model_version,
            num_train_images=train_dataset_group.num_train_images,
            num_val_images=val_dataset_group.num_train_images
            if val_dataset_group is not None
            else 0,
            num_reg_images=train_dataset_group.num_reg_images,
            num_batches_per_epoch=len(train_dataloader),
            num_train_epochs=num_train_epochs,
        )
        self.update_metadata(metadata, args)  # architecture specific metadata
        add_dataset_metadata(
            metadata,
            train_dataset_group,
            args,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
            total_batch_size=total_batch_size,
        )
        add_model_hash_metadata(metadata, args)
        metadata, minimum_metadata = finalize_metadata(
            metadata, net_kwargs=net_kwargs if args.network_args else None
        )

        # Saver owns every save / remove operation plus the accelerator
        # save/load pre-hooks that persist train_state.json. Hooks must be
        # registered before resume_from_local_or_hf_if_specified() so the
        # load hook fires and populates saver.steps_from_state.
        saver = CheckpointSaver(
            args=args,
            accelerator=accelerator,
            save_dtype=save_dtype,
            metadata=metadata,
            minimum_metadata=minimum_metadata,
            get_sai_model_spec_fn=self.get_sai_model_spec,
            current_epoch=current_epoch,
            current_step=current_step,
        )
        saver.register_hooks(network)

        # auto-resume from the resumable checkpoint if one exists
        saver.auto_resume()

        # resume
        resume_from_local_or_hf_if_specified(accelerator, args)
        steps_from_state = saver.steps_from_state

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader)
                    / accelerator.num_processes
                    / args.gradient_accumulation_steps
                )
        else:
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert args.max_train_steps > initial_step, (
                "max_train_steps should be greater than initial step"
            )

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                if not args.resume:
                    logger.info(
                        "initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                    )
                logger.info(f"skipping {initial_step} steps")
                initial_step *= args.gradient_accumulation_steps

                epoch_to_start = initial_step // math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
            else:
                epoch_to_start = initial_step // math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
                initial_step = 0  # do not skip

        # Drop dataset-group locals before loop entry so the cached references
        # release memory; the dataloaders already hold the data they need.
        del train_dataset_group
        if val_dataset_group is not None:
            del val_dataset_group

        loop_state = build_loop_state(
            self,
            args=args,
            accelerator=accelerator,
            saver=saver,
            network=network,
            unet=unet,
            text_encoder=text_encoder,
            text_encoders=text_encoders,
            vae=vae,
            tokenizers=tokenizers,
            training_model=training_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_descriptions=lr_descriptions,
            optimizer_train_fn=optimizer_train_fn,
            optimizer_eval_fn=optimizer_eval_fn,
            weight_dtype=weight_dtype,
            unet_weight_dtype=unet_weight_dtype,
            vae_dtype=vae_dtype,
            text_encoding_strategy=text_encoding_strategy,
            tokenize_strategy=tokenize_strategy,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            current_epoch=current_epoch,
            current_step=current_step,
            num_train_epochs=num_train_epochs,
            epoch_to_start=epoch_to_start,
            initial_step=initial_step,
            metadata=metadata,
            train_ctx_cls=TrainCtx,
            val_ctx_cls=ValCtx,
        )

        run_training_loop(self, loop_state)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            save_state_on_train_end(args, accelerator)

        # Per-LoRA DCW recipe is computed BEFORE save_final so the solver's
        # output lands in saver.metadata (the same dict CheckpointSaver
        # holds by reference) and gets serialized into the .safetensors.
        self._calibrate_dcw_recipe(
            loop_state.train_ctx,
            loop_state.val_ctx,
            metadata=metadata,
        )

        saver.cleanup_resumable()
        saver.save_final(network, loop_state.global_step, num_train_epochs)

    # endregion


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    add_dataset_arguments(parser, True, True, True)
    add_training_arguments(parser, True)
    add_masked_loss_arguments(parser)
    add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        "",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save metadata in output model",
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors)",
    )

    parser.add_argument(
        "--unet_lr",
        type=float,
        default=None,
        help="learning rate for U-Net",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple",
    )
    # FP8 is not supported yet — flag is kept for CLI compatibility but force-disabled in assert_extra_args.
    parser.add_argument(
        "--fp8_base_unet",
        action="store_true",
        help="(not supported yet) use fp8 for U-Net (or DiT). This flag is force-disabled.",
    )

    add_network_arguments(parser)
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + "",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + "",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset",
    )
    parser.add_argument(
        "--validation_sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Sigma values for validation loss (0.0~1.0). Low values = fine detail. Default: 0.1 0.4 0.7",
    )
    parser.add_argument(
        "--validation_bias_metric",
        action="store_true",
        help="Log the SNR-t bias gap (||v(x̂_t)|| − ||v(x_t_fwd)||, integrated "
        "over the late half) on the validation set, alongside FM-MSE. The "
        "headline scalar correlates with sample quality where FM-MSE often "
        "does not. LoRA family only — no APEX / IP-Adapter / EasyControl. "
        "See docs/methods/dcw.md.",
    )
    parser.add_argument(
        "--validation_bias_steps",
        type=int,
        default=12,
        help="Inference-schedule length for the bias-metric reverse rollout. "
        "Default 12 (half of inference's 24) — late-half gap envelope is "
        "preserved while keeping val cost bounded. Raise to 24 for "
        "production-resolution numbers.",
    )
    parser.add_argument(
        "--calibrate_dcw_recipe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At end-of-training, run one reverse-rollout per --dcw_calibration_anchors "
        "(plus a baseline if --validation_bias_metric didn't cache one), pick the "
        "anchor that empirically minimized the late-half weighted |gap_LL|², and "
        "write it to safetensors metadata as ss_dcw_recipe (read at inference by "
        "--dcw). Baseline (λ=0) is always a candidate so flat-style LoRAs that DCW "
        "hurts can land on no correction. Cost: N+(0|1) validation passes where N "
        "is len(anchors). See docs/proposal/lora-dcw-proposal.md.",
    )
    parser.add_argument(
        "--dcw_calibration_anchors",
        type=float,
        nargs="+",
        default=[-0.015, -0.022, -0.030],
        help="λ_LL anchor values to probe; recipe writes the empirically best one "
        "(lowest weighted late-half |gap_LL|²). Defaults span the safe LL operating "
        "range from docs/methods/dcw.md §Re-tuning λ for LL-only. Earlier versions "
        "ran a single ε-probe and closed-form-solved for λ* under linear-response, "
        "which extrapolated past LL's nonlinear knee and could pick λ ≈ −0.1 (visible "
        "drift). Picking from measured anchors avoids the extrapolation. Ignored "
        "without --calibrate_dcw_recipe.",
    )
    parser.add_argument(
        "--dcw_calibration_steps",
        type=int,
        default=28,
        help="Reverse-rollout step count for the end-of-training calibration "
        "passes. Decoupled from --validation_bias_steps (which is the cheap "
        "per-cycle diagnostic, default 12). Default 28 matches typical "
        "inference settings — the σ schedule is sampled at the same density "
        "the saved λ will be applied with. When this differs from "
        "--validation_bias_steps the cached baseline can't be reused, so "
        "calibration always runs an extra baseline pass (N+1 passes total). "
        "Ignored without --calibrate_dcw_recipe.",
    )
    parser.add_argument(
        "--dcw_calibration_schedule",
        type=str,
        default="one_minus_sigma",
        choices=["one_minus_sigma", "sigma_i", "const"],
        help="σ-schedule the calibration solves against (and writes to the "
        "recipe). Should match the schedule the LoRA will be applied with "
        "at inference. Default one_minus_sigma per docs/methods/dcw.md "
        "§Re-tuning λ for LL-only.",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        help="Dump the fully merged config (base → preset → method → CLI) as TOML "
        "with provenance comments, then exit 0. Does not start training.",
    )
    parser.add_argument(
        "--config-snapshot",
        dest="config_snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output/<output_name>.snapshot.toml next to the checkpoint on every real "
        "run (provenance + git SHA). Pass --no-config-snapshot to disable.",
    )
    parser.add_argument(
        "--config-strict",
        dest="config_strict",
        action="store_true",
        help="Treat config-schema warnings (unknown keys, off-list choices) as errors.",
    )
    return parser


from library.config import schema as _config_schema  # noqa: E402
from networks import all_network_kwargs as _all_network_kwargs  # noqa: E402


# Network-module-consumed flags (networks.lora_anima / networks.methods.postfix).
# These don't flow through argparse directly because `create_network` reads
# them from ``kwargs``. Derived from the registry in ``networks/__init__.py``
# (``SHARED_KWARG_FLAGS`` ∪ per-``NetworkSpec.kwarg_flags``) so adding a new
# kwarg to a variant spec automatically registers it here.
NETWORK_KWARG_ALLOWLIST: tuple[str, ...] = _all_network_kwargs()

# Top-level training args that aren't network kwargs but still flow through
# ``net_kwargs`` because a network module reads them. Kept explicit — any
# growth here should be reviewed, since the right answer is usually to
# expose the value as a proper argparse flag the network module reads
# directly rather than tunneling it through kwargs.
_EXTRA_FORWARDED_TOP_LEVEL_ARGS: tuple[str, ...] = (
    # Postfix contrastive resets its intra-step reference set on step
    # boundary, so it needs the grad-accum window.
    "gradient_accumulation_steps",
)


def build_network_extras() -> dict[str, _config_schema.ConfigKey]:
    return {
        k: _config_schema.ConfigKey(name=k, type="str", source="network_module")
        for k in NETWORK_KWARG_ALLOWLIST
    }


if __name__ == "__main__":
    parser = setup_parser()
    _config_schema.populate_schema(parser, extras=build_network_extras())

    args = parser.parse_args()
    verify_command_line_training_args(args)
    args = read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    artist = getattr(args, "artist_filter", None)
    if artist:
        _datasets_base.set_artist_filter(artist)
        slug = artist.lstrip("@")
        args.output_dir = "output/ckpt-artist"
        args.output_name = f"{args.output_name}_{slug}"
        logger.info(
            f"artist_filter active: '{artist}' → output_dir={args.output_dir}, "
            f"output_name={args.output_name}"
        )

    trainer = AnimaTrainer()
    trainer.train(args)
