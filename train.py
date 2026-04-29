# Anima LoRA training script (merged standalone)

import gc
import importlib
import argparse
import math
import os
import typing
from dataclasses import dataclass
from typing import Any, Callable, Union, Optional
import sys
import random
import time
import json
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
    resolve_adapters,
)
from library.config.io import (
    load_dataset_config_from_base,
    read_config_from_file,
)
from library.datasets import (
    DatasetGroup,
    LossRecorder,
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
    LossContext,
    MetricContext,
    SAMPLER_REGISTRY,
    SamplerContext,
    collect_metrics,
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
    get_checkpoint_ckpt_name,
    get_checkpoint_state_dir,
    get_epoch_ckpt_name,
    get_huber_threshold_if_needed,
    get_last_ckpt_name,
    get_optimizer,
    get_optimizer_train_eval_fn,
    get_remove_epoch_no,
    get_remove_step_no,
    get_scheduler_fix,
    get_step_ckpt_name,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_checkpoint_state,
    save_state_on_train_end,
    verify_command_line_training_args,
    verify_training_args,
)
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
                "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"
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

        # Install smart caption shuffle for Anima (respects @artist prefix and "on the ..." sections)
        if args.shuffle_caption:
            for dataset in train_dataset_group.datasets:
                dataset.custom_shuffle_caption_fn = (
                    anima_train_utils.anima_smart_shuffle_caption
                )
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.custom_shuffle_caption_fn = (
                        anima_train_utils.anima_smart_shuffle_caption
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

        # Propagate IP-Adapter feature-cache flag so datasets load
        # {stem}_anima_{encoder}.safetensors sidecars into batch["ip_features"].
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
        """Override base process_batch for caption dropout with cached text encoder outputs."""

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = (
            ctx.text_encoding_strategy
        )
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            encoder_outputs = text_encoder_outputs_list[:-1]

            # Apply caption dropout to cached outputs
            encoder_outputs = (
                anima_text_encoding_strategy.drop_cached_text_encoder_outputs(
                    *encoder_outputs, caption_dropout_rates=caption_dropout_rates
                )
            )
            # Use a shallow-copied batch so the original text_encoder_outputs_list
            # (with caption_dropout_rates appended) stays intact for validation's
            # multi-timestep loop which reuses the same batch.
            batch = {**batch, "text_encoder_outputs_list": encoder_outputs}

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
        """Parse --profile_steps 'start-end' into (start, end) or None."""
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

        val_progress_bar = tqdm(
            range(val.total_steps),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc=progress_desc,
        )
        val_timesteps_step = 0
        per_sigma_losses = {s: [] for s in val.sigmas}
        for val_step, batch in enumerate(val.dataloader):
            if val_step >= val.steps:
                break

            for sigma in val.sigmas:
                self.on_step_start(ctx, batch, is_train=False)

                # Pin sigma via t_min/t_max (what the noise function reads)
                args.t_min = args.t_max = sigma

                # Mirror the training loop's cudagraph step-begin marker so
                # cudagraph_trees doesn't re-record / leak a memory pool on
                # each val forward (see the train-loop call site for context).
                if self._cudagraph_mark_step:
                    net_unwrapped = accelerator.unwrap_model(ctx.network)
                    if hasattr(net_unwrapped, "clear_step_caches"):
                        net_unwrapped.clear_step_caches()
                    torch.compiler.cudagraph_mark_step_begin()

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

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)

            # save current epoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            logger.info(
                f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value + 1}"
            )
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "current_epoch": current_epoch.value,
                        "current_step": current_step.value + 1,
                    },
                    f,
                )

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)

            # load current epoch and step
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # auto-resume from checkpoint if checkpointing_epochs is set and a checkpoint exists
        if getattr(args, "checkpointing_epochs", None) and not args.resume:
            checkpoint_state_dir = get_checkpoint_state_dir(args)
            if os.path.exists(checkpoint_state_dir):
                train_state_file = os.path.join(
                    checkpoint_state_dir, "train_state.json"
                )
                if os.path.exists(train_state_file):
                    with open(train_state_file, "r", encoding="utf-8") as f:
                        ckpt_data = json.load(f)
                    ckpt_step = ckpt_data.get("current_step", 0)
                    if ckpt_step < args.max_train_steps:
                        args.resume = checkpoint_state_dir
                        args.skip_until_initial_step = True
                        logger.info(
                            f"auto-resuming from checkpoint at step {ckpt_step}: {checkpoint_state_dir}"
                        )
                    else:
                        logger.info(
                            f"checkpoint already reached max_train_steps ({ckpt_step} >= {args.max_train_steps}), starting fresh"
                        )

        # resume
        resume_from_local_or_hf_if_specified(accelerator, args)

        # calculate epochs
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

        global_step = 0

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        train_util.init_trackers(accelerator, args, "network_train")

        loss_recorder = LossRecorder()
        val_step_loss_recorder = LossRecorder()
        val_epoch_loss_recorder = LossRecorder()

        del train_dataset_group
        if val_dataset_group is not None:
            del val_dataset_group

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
        else:

            def on_step_start_for_network(*args, **kwargs):
                return None

        # function for saving/removing
        def save_model(
            ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False
        ):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = self.get_sai_model_spec(args)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)
            # Also remove HydraLoRA _moe sibling if present
            moe_file = os.path.splitext(old_ckpt_file)[0] + "_moe.safetensors"
            if os.path.exists(moe_file):
                accelerator.print(f"removing old checkpoint: {moe_file}")
                os.remove(moe_file)

        # if text_encoder is not needed for training, delete it to save memory.
        if self.is_text_encoder_not_needed_for_training(args):
            logger.info(
                "text_encoder is not needed for training. deleting to save memory."
            )
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None
            gc.collect()
            clean_memory_on_device(accelerator.device)

        # For --sample_at_first
        optimizer_eval_fn()
        self.sample_images(
            accelerator,
            args,
            0,
            global_step,
            accelerator.device,
            vae,
            tokenizers,
            text_encoder,
            unet,
        )
        optimizer_train_fn()
        is_tracking = len(accelerator.trackers) > 0
        if is_tracking:
            accelerator.log({}, step=0)

        ctx = TrainCtx(
            args=args,
            accelerator=accelerator,
            network=network,
            unet=unet,
            vae=vae,
            text_encoders=text_encoders,
            noise_scheduler=noise_scheduler,
            text_encoding_strategy=text_encoding_strategy,
            tokenize_strategy=tokenize_strategy,
            vae_dtype=vae_dtype,
            weight_dtype=weight_dtype,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            optimizer_eval_fn=optimizer_eval_fn,
            optimizer_train_fn=optimizer_train_fn,
            is_tracking=is_tracking,
        )

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            global_step = initial_step // args.gradient_accumulation_steps
            for skip_epoch in range(epoch_to_start):
                logger.info(
                    f"skipping epoch {skip_epoch + 1} because initial_step (multiplied) is {initial_step}"
                )
                initial_step -= len(train_dataloader)

        # log device and dtype for each model
        logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            params_itr = t_enc.parameters()
            params_itr.__next__()  # skip the first parameter
            params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
            param_3rd = params_itr.__next__()
            logger.info(
                f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}"
            )

        clean_memory_on_device(accelerator.device)

        progress_bar = tqdm(
            range(args.max_train_steps - global_step),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc="steps",
        )

        validation_steps = (
            min(args.max_validation_steps, len(val_dataloader))
            if args.max_validation_steps is not None
            else len(val_dataloader)
        )
        # Validate at fixed sigma values across the schedule:
        # 0.1 = near-clean / fine detail, 0.4 = mid / bulk structure,
        # 0.7 = high noise / coarse denoising (early inference steps).
        validation_sigmas = (
            args.validation_sigmas
            if args.validation_sigmas is not None
            else [0.1, 0.4, 0.7]
        )
        validation_total_steps = validation_steps * len(validation_sigmas)
        original_t_min = args.t_min
        original_t_max = args.t_max

        val = ValCtx(
            dataloader=val_dataloader,
            sigmas=validation_sigmas,
            steps=validation_steps,
            total_steps=validation_total_steps,
            train_loss_recorder=loss_recorder,
            original_t_min=original_t_min,
            original_t_max=original_t_max,
        )

        # --- Profiler setup ---
        profile_range = self._parse_profile_steps(args)
        profiler_ctx = None

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}\n")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(
                text_encoder, unet
            )  # network.train() is called here

            # TRAINING
            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(
                    train_dataloader, initial_step - 1
                )
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step
                if initial_step > 0:
                    initial_step -= 1
                    continue

                # --- Profiler: start recording ---
                if (
                    profile_range
                    and global_step == profile_range[0]
                    and profiler_ctx is None
                ):
                    accelerator.print(f"\n[profiler] starting at step {global_step}")
                    profiler_ctx = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        with_stack=True,
                    )
                    profiler_ctx.__enter__()

                with accelerator.accumulate(training_model):
                    on_step_start_for_network(text_encoder, unet)

                    # preprocess batch for each model
                    self.on_step_start(ctx, batch, is_train=True)

                    # CUDAGraphs (reduce-overhead / max-autotune) need an explicit
                    # iteration boundary for inductor's cudagraph_trees. Without
                    # this call, the "pending, uninvoked backwards" fast-path
                    # check fails every step and cudagraphs silently fall back to
                    # the eager path — you pay compile latency and keep launch
                    # overhead. Must be called before the forward on every step.
                    #
                    # Also clear Python references to last-step gate/σ tensors
                    # *before* marking — those tensors live in the cudagraph
                    # memory pool, and a lingering self._last_gate/self._sigma
                    # reference keeps the pool pinned regardless of the mark
                    # call, which defeats the whole point.
                    if self._cudagraph_mark_step:
                        net_unwrapped = accelerator.unwrap_model(network)
                        if hasattr(net_unwrapped, "clear_step_caches"):
                            net_unwrapped.clear_step_caches()
                        torch.compiler.cudagraph_mark_step_begin()

                    loss = self.process_batch(ctx, batch, is_train=True)

                    # Split-backward path (APEX) backwards both branches
                    # inline inside process_batch and returns a detached
                    # scalar for logging. Skip the outer backward in that
                    # case so we don't double-step or crash on a no-grad
                    # tensor during warmup.
                    if getattr(self, "_split_backward_consumed", False):
                        self._split_backward_consumed = False
                    else:
                        accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # HydraLoRA "best-expert" warmup: keep grads only on
                        # top-k experts by per-expert grad-norm during warmup.
                        # No-op unless expert_best_warmup_ratio > 0. Runs
                        # before clip_grad_norm so clipping sees the masked grads.
                        net_unwrapped = accelerator.unwrap_model(network)
                        if hasattr(
                            net_unwrapped, "step_expert_best_warmup_post_backward"
                        ):
                            net_unwrapped.step_expert_best_warmup_post_backward(
                                int(getattr(self, "_hydra_warmup_step", 0)),
                                int(getattr(args, "max_train_steps", 0) or 0),
                            )
                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(
                                network
                            ).get_trainable_params()
                            accelerator.clip_grad_norm_(
                                params_to_clip, args.max_grad_norm
                            )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # --- Profiler: stop recording ---
                if profiler_ctx is not None and global_step >= profile_range[1]:
                    profiler_ctx.__exit__(None, None, None)
                    trace_path = os.path.join(args.output_dir, "profile_trace.json")
                    profiler_ctx.export_chrome_trace(trace_path)
                    accelerator.print(f"\n[profiler] stopped at step {global_step}")
                    accelerator.print(f"[profiler] trace saved to {trace_path}")
                    accelerator.print(
                        "[profiler] open in https://ui.perfetto.dev for visual inspection\n"
                    )
                    key_avg = profiler_ctx.key_averages(group_by_stack_n=0)
                    accelerator.print("[profiler] top 30 CUDA kernels by total time:\n")
                    accelerator.print(
                        key_avg.table(sort_by="cuda_time_total", row_limit=30)
                    )
                    profiler_ctx = None
                    profile_range = None  # don't re-trigger

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(
                        network
                    ).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    mean_grad_norm = None
                    mean_combined_norm = None
                    max_mean_logs = {
                        "Keys Scaled": keys_scaled,
                        "Average key norm": mean_norm,
                    }
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None
                    mean_grad_norm = None
                    mean_combined_norm = None
                    max_mean_logs = {}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    optimizer_eval_fn()
                    self.sample_images(
                        accelerator,
                        args,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        tokenizers,
                        text_encoder,
                        unet,
                    )
                    progress_bar.unpause()

                    # Save model at specified steps
                    if (
                        args.save_every_n_steps is not None
                        and global_step % args.save_every_n_steps == 0
                    ):
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = get_step_ckpt_name(
                                args, "." + args.save_model_as, global_step
                            )
                            save_model(
                                ckpt_name,
                                accelerator.unwrap_model(network),
                                global_step,
                                epoch,
                            )

                            if args.save_state:
                                save_and_remove_state_stepwise(
                                    args, accelerator, global_step
                                )

                            remove_step_no = get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = get_step_ckpt_name(
                                    args, "." + args.save_model_as, remove_step_no
                                )
                                remove_model(remove_ckpt_name)
                    optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}
                _unwrapped_net = accelerator.unwrap_model(network)
                if getattr(_unwrapped_net, "_use_hydra", False):
                    _router_H = _unwrapped_net.get_router_entropy()
                    if _router_H is not None:
                        logs["router_H"] = f"{_router_H:.3f}"
                progress_bar.set_postfix(refresh=False, **{**max_mean_logs, **logs})

                log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
                should_log_step = (global_step % log_every == 0) or (
                    global_step >= args.max_train_steps
                )
                if is_tracking and should_log_step:
                    logs = self.generate_step_logs(
                        args,
                        current_loss,
                        avr_loss,
                        lr_scheduler,
                        lr_descriptions,
                        optimizer,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                        mean_grad_norm,
                        mean_combined_norm,
                    )
                    trainer_state: dict = {}
                    for adapter in self._adapters:
                        trainer_state.update(adapter.state_for_metrics())
                    logs.update(
                        collect_metrics(
                            MetricContext(
                                args=args,
                                network=_unwrapped_net,
                                trainer_state=trainer_state,
                            )
                        )
                    )
                    self.step_logging(accelerator, logs, global_step, epoch + 1)

                # VALIDATION PER STEP: global_step is already incremented
                should_validate_step = (
                    args.validate_every_n_steps is not None
                    and global_step % args.validate_every_n_steps == 0
                )
                if (
                    accelerator.sync_gradients
                    and validation_steps > 0
                    and should_validate_step
                ):
                    self._run_validation(
                        ctx,
                        val,
                        val_loss_recorder=val_step_loss_recorder,
                        epoch=epoch,
                        global_step=global_step,
                        progress_bar=progress_bar,
                        progress_desc="validation steps",
                        postfix_label="val_avg_loss",
                        log_avg_key="loss/validation/step_average",
                        log_div_key="loss/validation/step_divergence",
                        logging_fn=self.step_logging,
                    )

                if global_step >= args.max_train_steps:
                    break

            # EPOCH VALIDATION
            should_validate_epoch = (
                (epoch + 1) % args.validate_every_n_epochs == 0
                if args.validate_every_n_epochs is not None
                else True
            )

            if should_validate_epoch and len(val_dataloader) > 0:
                self._run_validation(
                    ctx,
                    val,
                    val_loss_recorder=val_epoch_loss_recorder,
                    epoch=epoch,
                    global_step=global_step,
                    progress_bar=progress_bar,
                    progress_desc="epoch validation steps",
                    postfix_label="val_epoch_avg_loss",
                    log_avg_key="loss/validation/epoch_average",
                    log_div_key="loss/validation/epoch_divergence",
                    logging_fn=self.epoch_logging,
                )

            # END OF EPOCH
            if is_tracking:
                logs = {"loss/epoch_average": loss_recorder.moving_average}
                self.epoch_logging(accelerator, logs, global_step, epoch + 1)

            # Per-method end-of-epoch hooks (IP-Adapter diagnostic dump, …).
            # Main process only — adapters that need cross-rank reduction
            # should do that internally.
            if self._adapters and is_main_process:
                epoch_end_ctx = StepCtx(
                    args=args,
                    accelerator=accelerator,
                    network=network,
                    weight_dtype=weight_dtype,
                )
                for adapter in self._adapters:
                    adapter.on_epoch_end(epoch_end_ctx)

            accelerator.wait_for_everyone()

            # Save model at specified epochs
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                    epoch + 1
                ) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = get_epoch_ckpt_name(
                        args, "." + args.save_model_as, epoch + 1
                    )
                    save_model(
                        ckpt_name,
                        accelerator.unwrap_model(network),
                        global_step,
                        epoch + 1,
                    )

                    remove_epoch_no = get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = get_epoch_ckpt_name(
                            args, "." + args.save_model_as, remove_epoch_no
                        )
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            # Save resumable checkpoint at specified epoch intervals (overwrites previous)
            if args.checkpointing_epochs is not None and args.checkpointing_epochs > 0:
                if (epoch + 1) % args.checkpointing_epochs == 0 and (
                    epoch + 1
                ) < num_train_epochs:
                    if is_main_process:
                        ckpt_name = get_checkpoint_ckpt_name(
                            args, "." + args.save_model_as
                        )
                        save_model(
                            ckpt_name,
                            accelerator.unwrap_model(network),
                            global_step,
                            epoch + 1,
                        )
                    save_checkpoint_state(args, accelerator)

            self.sample_images(
                accelerator,
                args,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                tokenizers,
                text_encoder,
                unet,
            )
            progress_bar.unpause()
            optimizer_train_fn()

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            save_state_on_train_end(args, accelerator)

        # clean up checkpoint files after successful completion
        if is_main_process and getattr(args, "checkpointing_epochs", None):
            checkpoint_state_dir = get_checkpoint_state_dir(args)
            if os.path.exists(checkpoint_state_dir):
                import shutil

                logger.info(
                    f"training complete, removing checkpoint state: {checkpoint_state_dir}"
                )
                shutil.rmtree(checkpoint_state_dir)
            checkpoint_ckpt = os.path.join(
                args.output_dir,
                get_checkpoint_ckpt_name(args, "." + args.save_model_as),
            )
            if os.path.exists(checkpoint_ckpt):
                logger.info(f"removing checkpoint weights: {checkpoint_ckpt}")
                os.remove(checkpoint_ckpt)

        if is_main_process:
            ckpt_name = get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(
                ckpt_name,
                network,
                global_step,
                num_train_epochs,
                force_sync_upload=True,
            )

            logger.info("model saved.")

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
