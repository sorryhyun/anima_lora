# Anima LoRA training script (merged standalone)

import gc
import importlib
import argparse
import math
import os
import typing
from typing import Any, Union, Optional
import sys
import random
import time
import json
from multiprocessing import Value
from tqdm import tqdm

import torch
import torch.nn as nn
from library.device_utils import clean_memory_on_device

from accelerate.utils import set_seed
from accelerate import Accelerator
from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    noise_utils,
    qwen_image_autoencoder_kl,
    sai_model_spec,
    strategy_anima,
    strategy_base,
    train_util,
)
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


class AnimaTrainer:
    def __init__(self):
        self.sample_prompts_te_outputs = None
        self._padding_mask_cache = {}

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
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
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
            from networks.attention import flash_attn, flash_attn_func

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
            attn_softmax_scale=attn_softmax_scale,
        )

        # FP8 base weights (fp8_base_unet) are not supported yet — the fp8 path is disabled.
        # if args.fp8_base_unet:
        #     from library.anima_models import quantize_to_fp8
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
                model.compile_blocks(args.dynamo_backend)
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
                caption_shuffle_variants=getattr(args, "caption_shuffle_variants", 0),
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
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        anima: anima_models.Anima = unet

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        noise = torch.randn_like(latents)

        # Get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = (
            noise_utils.get_noisy_model_input_and_timesteps(
                args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
            )
        )
        timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

        # Set timestep-dependent rank mask on LoRA and ReFT modules
        if hasattr(network, "set_timestep_mask"):
            network.set_timestep_mask(timesteps, max_timestep=1.0)
        if hasattr(network, "set_reft_timestep_mask"):
            network.set_reft_timestep_mask(timesteps, max_timestep=1.0)

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

        # Set HydraLoRA gate weights from crossattn_emb
        if crossattn_emb is not None and hasattr(network, "set_hydra_gate"):
            network.set_hydra_gate(crossattn_emb)

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

        # Check for CFG postfix mode
        cfg_postfix = (
            getattr(network, "mode", None) == "cfg"
            and hasattr(network, "postfix_pos")
            and crossattn_emb is None
        )

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            if cfg_postfix:
                # CFG-aware dual postfix: two forward passes
                adapter = anima.llm_adapter

                # Positive pass: real caption + postfix_pos
                adapter.postfix_embeds = network.postfix_pos
                pos_pred = anima(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )

                # Negative pass: unconditional (zeroed Qwen3) + postfix_neg
                adapter.postfix_embeds = network.postfix_neg
                uncond_embeds = torch.zeros_like(prompt_embeds)
                uncond_attn_mask = torch.zeros_like(attn_mask)
                uncond_t5_ids = torch.zeros_like(t5_input_ids)
                uncond_t5_ids[:, 0] = 1  # </s> token
                uncond_t5_attn = torch.zeros_like(t5_attn_mask)
                uncond_t5_attn[:, 0] = 1
                neg_pred = anima(
                    noisy_model_input,
                    timesteps,
                    uncond_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=uncond_t5_ids,
                    target_attention_mask=uncond_t5_attn,
                    source_attention_mask=uncond_attn_mask,
                )

                # Restore positive postfix as default
                adapter.postfix_embeds = network.postfix_pos

                # CFG combine
                cfg_scale = network.cfg_scale
                model_pred = neg_pred + cfg_scale * (pos_pred - neg_pred)
            elif crossattn_emb is None:
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
                    crossattn_emb = network.append_postfix(crossattn_emb, seqlens)
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

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        # Orthogonality regularization for OrthoLoRA
        if (
            hasattr(self, "_network")
            and getattr(self._network, "_ortho_reg_weight", 0) > 0
        ):
            ortho_reg = self._network.get_ortho_regularization()
            loss = loss + self._network._ortho_reg_weight * ortho_reg
        # HydraLoRA load-balancing loss
        if (
            hasattr(self, "_network")
            and getattr(self._network, "_balance_loss_weight", 0) > 0
        ):
            balance_loss = self._network.get_balance_loss()
            loss = loss + self._network._balance_loss_weight * balance_loss
        return loss

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

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
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

    def on_validation_step_end(
        self, args, accelerator, network, text_encoders, unet, batch, weight_dtype
    ):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """Override base process_batch for caption dropout with cached text encoder outputs."""

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = (
            text_encoding_strategy
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

        return self._process_batch_inner(
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train,
            train_text_encoder,
            train_unet,
        )

    def _process_batch_inner(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """
        Process a batch for the network (original NetworkTrainer.process_batch logic)
        """
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
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
        )

        huber_c = train_util.get_huber_threshold_if_needed(
            args, timesteps, noise_scheduler
        )
        loss = train_util.conditional_loss(
            noise_pred.float(), target.float(), args.loss_type, "none", huber_c
        )
        if weighting is not None:
            loss = loss * weighting
        if args.masked_loss or (
            "alpha_masks" in batch and batch["alpha_masks"] is not None
        ):
            loss = apply_masked_loss(loss, batch)
        loss = loss.mean(
            dim=list(range(1, loss.ndim))
        )  # mean over all dims except batch

        loss_weights = batch["loss_weights"]  # per-sample weight
        loss = loss * loss_weights

        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

        scalar_loss = loss.mean()

        # Multiscale loss: additional MSE term at 2x-downsampled resolution
        if getattr(args, "multiscale_loss_weight", 0):
            ms_weight = args.multiscale_loss_weight
            h, w = noise_pred.shape[-2:]
            side_length = math.sqrt(h * w) * 8  # approximate pixel-space side
            if side_length >= 1024 * 0.9 and h >= 2 and w >= 2:
                pred_ds = torch.nn.functional.avg_pool2d(noise_pred.float(), 2)
                target_ds = torch.nn.functional.avg_pool2d(target.float(), 2)
                ms_loss = torch.nn.functional.mse_loss(pred_ds, target_ds)
                scalar_loss = (scalar_loss + ms_loss * ms_weight) / (1.0 + ms_weight)

        return scalar_loss

    # endregion

    # region Methods only in NetworkTrainer (not overridden by Anima)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        self._network = (
            network  # store reference for ortho regularization in post_process_loss
        )

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(
            None, args, False, True, False, anima="preview"
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

    def on_step_start(
        self,
        args,
        accelerator,
        network,
        text_encoders,
        unet,
        batch,
        weight_dtype,
        is_train: bool = True,
    ):
        pass

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
        dataset: train_util.DatasetGroup,
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

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = (
                    strategy_base.TextEncodingStrategy.get_strategy()
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

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(
            tokenize_strategy
        )  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # Prepare dataset
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
                if use_dreambooth_method:
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

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = (
                config_util.generate_dataset_group_by_blueprint(
                    blueprint.dataset_group,
                    constant_token_buckets=getattr(args, "static_token_count", None)
                    is not None,
                )
            )
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args)
            val_dataset_group = (
                None  # placeholder until validation dataset supported for arbitrary
            )

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = train_util.collator_class(
            current_epoch, current_step, ds_for_collator
        )

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            train_util.debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                train_util.debug_dataset(val_dataset_group)
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
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precision dtype
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
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
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = (
            self.get_text_encoder_outputs_caching_strategy(args)
        )
        if text_encoder_outputs_caching_strategy is not None:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
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

        # Load network module
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
        _NETWORK_ARG_KEYS = [
            "train_llm_adapter",
            "exclude_patterns",
            "include_patterns",
            "use_dora",
            "use_ortho",
            "sig_type",
            "ortho_reg_weight",
            "use_timestep_mask",
            "min_rank",
            "alpha_rank_scale",
            "use_hydra",
            "num_experts",
            "balance_loss_weight",
            "rank_dropout",
            "module_dropout",
            "verbose",
            "network_reg_lrs",
            "network_reg_dims",
            "loraplus_lr_ratio",
            "loraplus_unet_lr_ratio",
            "loraplus_text_encoder_lr_ratio",
            "layer_start",
            "layer_end",
        ]
        for key in _NETWORK_ARG_KEYS:
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
            return
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
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # Prepare optimizer, data loader etc.
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

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(
            args, trainable_params
        )
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(
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
        lr_scheduler = train_util.get_scheduler_fix(
            args, optimizer, accelerator.num_processes
        )

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
        else:
            pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

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

        del t_enc

        # Full-model torch.compile: compile after LoRA apply + accelerator prepare
        # so dynamo traces through LoRA-patched forwards without _orig_mod prefix issues.
        if args.torch_compile and getattr(args, "compile_mode", "blocks") == "full":
            assert not args.gradient_checkpointing, (
                "compile_mode='full' is incompatible with gradient checkpointing"
            )
            assert not self.is_swapping_blocks, (
                "compile_mode='full' is incompatible with block swap"
            )
            unet = torch.compile(unet, backend=args.dynamo_backend, dynamic=True)
            logger.info(f"full-model torch.compile (backend={args.dynamo_backend})")

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # patch for fp16 grad scale
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

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
            checkpoint_state_dir = train_util.get_checkpoint_state_dir(args)
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
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

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

        metadata = train_util.build_training_metadata(
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
        train_util.add_dataset_metadata(
            metadata,
            train_dataset_group,
            args,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
            total_batch_size=total_batch_size,
        )
        train_util.add_model_hash_metadata(metadata, args)
        metadata, minimum_metadata = train_util.finalize_metadata(
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

        loss_recorder = train_util.LossRecorder()
        val_step_loss_recorder = train_util.LossRecorder()
        val_epoch_loss_recorder = train_util.LossRecorder()

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
        # Validate at fixed sigma values (low-t focused for texture fidelity)
        validation_sigmas = (
            args.validation_sigmas
            if args.validation_sigmas is not None
            else [0.05, 0.10, 0.20, 0.35]
        )
        validation_total_steps = validation_steps * len(validation_sigmas)
        original_t_min = args.t_min
        original_t_max = args.t_max

        def switch_rng_state(
            seed: int,
        ) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state()
            python_rng_state = random.getstate()

            torch.manual_seed(seed)
            random.seed(seed)

            return (cpu_rng_state, gpu_rng_state, python_rng_state)

        def restore_rng_state(
            rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple],
        ):
            cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state)
            random.setstate(python_rng_state)

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
                    self.on_step_start(
                        args,
                        accelerator,
                        network,
                        text_encoders,
                        unet,
                        batch,
                        weight_dtype,
                        is_train=True,
                    )

                    loss = self.process_batch(
                        batch,
                        text_encoders,
                        unet,
                        network,
                        vae,
                        noise_scheduler,
                        vae_dtype,
                        weight_dtype,
                        accelerator,
                        args,
                        text_encoding_strategy,
                        tokenize_strategy,
                        is_train=True,
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        self.all_reduce_network(
                            accelerator, network
                        )  # sync DDP grad manually
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
                            ckpt_name = train_util.get_step_ckpt_name(
                                args, "." + args.save_model_as, global_step
                            )
                            save_model(
                                ckpt_name,
                                accelerator.unwrap_model(network),
                                global_step,
                                epoch,
                            )

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(
                                    args, accelerator, global_step
                                )

                            remove_step_no = train_util.get_remove_step_no(
                                args, global_step
                            )
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(
                                    args, "." + args.save_model_as, remove_step_no
                                )
                                remove_model(remove_ckpt_name)
                    optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}
                progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if is_tracking:
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
                    optimizer_eval_fn()
                    accelerator.unwrap_model(network).eval()
                    rng_states = switch_rng_state(
                        args.validation_seed
                        if args.validation_seed is not None
                        else args.seed
                    )

                    val_progress_bar = tqdm(
                        range(validation_total_steps),
                        smoothing=0,
                        disable=not accelerator.is_local_main_process,
                        desc="validation steps",
                    )
                    val_timesteps_step = 0
                    per_sigma_losses = {s: [] for s in validation_sigmas}
                    for val_step, batch in enumerate(val_dataloader):
                        if val_step >= validation_steps:
                            break

                        for sigma in validation_sigmas:
                            self.on_step_start(
                                args,
                                accelerator,
                                network,
                                text_encoders,
                                unet,
                                batch,
                                weight_dtype,
                                is_train=False,
                            )

                            # Pin sigma via t_min/t_max (what the noise function reads)
                            args.t_min = args.t_max = sigma

                            loss = self.process_batch(
                                batch,
                                text_encoders,
                                unet,
                                network,
                                vae,
                                noise_scheduler,
                                vae_dtype,
                                weight_dtype,
                                accelerator,
                                args,
                                text_encoding_strategy,
                                tokenize_strategy,
                                is_train=False,
                                train_text_encoder=train_text_encoder,
                                train_unet=train_unet,
                            )

                            current_loss = loss.detach().item()
                            val_step_loss_recorder.add(
                                epoch=epoch, step=val_timesteps_step, loss=current_loss
                            )
                            per_sigma_losses[sigma].append(current_loss)
                            val_progress_bar.update(1)
                            val_progress_bar.set_postfix(
                                {
                                    "val_avg_loss": val_step_loss_recorder.moving_average,
                                    "sigma": f"{sigma:.2f}",
                                }
                            )

                            self.on_validation_step_end(
                                args,
                                accelerator,
                                network,
                                text_encoders,
                                unet,
                                batch,
                                weight_dtype,
                            )
                            val_timesteps_step += 1

                    if is_tracking:
                        loss_validation_divergence = (
                            val_step_loss_recorder.moving_average
                            - loss_recorder.moving_average
                        )
                        logs = {
                            "loss/validation/step_average": val_step_loss_recorder.moving_average,
                            "loss/validation/step_divergence": loss_validation_divergence,
                        }
                        for s, losses in per_sigma_losses.items():
                            if losses:
                                logs[f"loss/validation/sigma_{s:.2f}"] = sum(
                                    losses
                                ) / len(losses)
                        self.step_logging(
                            accelerator, logs, global_step, epoch=epoch + 1
                        )

                    restore_rng_state(rng_states)
                    args.t_min = original_t_min
                    args.t_max = original_t_max
                    optimizer_train_fn()
                    accelerator.unwrap_model(network).train()
                    progress_bar.unpause()

                if global_step >= args.max_train_steps:
                    break

            # EPOCH VALIDATION
            should_validate_epoch = (
                (epoch + 1) % args.validate_every_n_epochs == 0
                if args.validate_every_n_epochs is not None
                else True
            )

            if should_validate_epoch and len(val_dataloader) > 0:
                optimizer_eval_fn()
                accelerator.unwrap_model(network).eval()
                rng_states = switch_rng_state(
                    args.validation_seed
                    if args.validation_seed is not None
                    else args.seed
                )

                val_progress_bar = tqdm(
                    range(validation_total_steps),
                    smoothing=0,
                    disable=not accelerator.is_local_main_process,
                    desc="epoch validation steps",
                )

                val_timesteps_step = 0
                per_sigma_losses = {s: [] for s in validation_sigmas}
                for val_step, batch in enumerate(val_dataloader):
                    if val_step >= validation_steps:
                        break

                    for sigma in validation_sigmas:
                        # Pin sigma via t_min/t_max
                        args.t_min = args.t_max = sigma

                        self.on_step_start(
                            args,
                            accelerator,
                            network,
                            text_encoders,
                            unet,
                            batch,
                            weight_dtype,
                            is_train=False,
                        )

                        loss = self.process_batch(
                            batch,
                            text_encoders,
                            unet,
                            network,
                            vae,
                            noise_scheduler,
                            vae_dtype,
                            weight_dtype,
                            accelerator,
                            args,
                            text_encoding_strategy,
                            tokenize_strategy,
                            is_train=False,
                            train_text_encoder=train_text_encoder,
                            train_unet=train_unet,
                        )

                        current_loss = loss.detach().item()
                        val_epoch_loss_recorder.add(
                            epoch=epoch, step=val_timesteps_step, loss=current_loss
                        )
                        per_sigma_losses[sigma].append(current_loss)
                        val_progress_bar.update(1)
                        val_progress_bar.set_postfix(
                            {
                                "val_epoch_avg_loss": val_epoch_loss_recorder.moving_average,
                                "sigma": f"{sigma:.2f}",
                            }
                        )

                        self.on_validation_step_end(
                            args,
                            accelerator,
                            network,
                            text_encoders,
                            unet,
                            batch,
                            weight_dtype,
                        )
                        val_timesteps_step += 1

                if is_tracking:
                    avr_loss: float = val_epoch_loss_recorder.moving_average
                    loss_validation_divergence = (
                        val_epoch_loss_recorder.moving_average
                        - loss_recorder.moving_average
                    )
                    logs = {
                        "loss/validation/epoch_average": avr_loss,
                        "loss/validation/epoch_divergence": loss_validation_divergence,
                    }
                    for s, losses in per_sigma_losses.items():
                        if losses:
                            logs[f"loss/validation/sigma_{s:.2f}"] = sum(losses) / len(
                                losses
                            )
                    self.epoch_logging(accelerator, logs, global_step, epoch + 1)

                restore_rng_state(rng_states)
                args.t_min = original_t_min
                args.t_max = original_t_max
                optimizer_train_fn()
                accelerator.unwrap_model(network).train()
                progress_bar.unpause()

            # END OF EPOCH
            if is_tracking:
                logs = {"loss/epoch_average": loss_recorder.moving_average}
                self.epoch_logging(accelerator, logs, global_step, epoch + 1)

            accelerator.wait_for_everyone()

            # Save model at specified epochs
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                    epoch + 1
                ) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(
                        args, "." + args.save_model_as, epoch + 1
                    )
                    save_model(
                        ckpt_name,
                        accelerator.unwrap_model(network),
                        global_step,
                        epoch + 1,
                    )

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(
                            args, "." + args.save_model_as, remove_epoch_no
                        )
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(
                            args, accelerator, epoch + 1
                        )

            # Save resumable checkpoint at specified epoch intervals (overwrites previous)
            if args.checkpointing_epochs is not None and args.checkpointing_epochs > 0:
                if (epoch + 1) % args.checkpointing_epochs == 0 and (
                    epoch + 1
                ) < num_train_epochs:
                    if is_main_process:
                        ckpt_name = train_util.get_checkpoint_ckpt_name(
                            args, "." + args.save_model_as
                        )
                        save_model(
                            ckpt_name,
                            accelerator.unwrap_model(network),
                            global_step,
                            epoch + 1,
                        )
                    train_util.save_checkpoint_state(args, accelerator)

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
            train_util.save_state_on_train_end(args, accelerator)

        # clean up checkpoint files after successful completion
        if is_main_process and getattr(args, "checkpointing_epochs", None):
            checkpoint_state_dir = train_util.get_checkpoint_state_dir(args)
            if os.path.exists(checkpoint_state_dir):
                import shutil

                logger.info(
                    f"training complete, removing checkpoint state: {checkpoint_state_dir}"
                )
                shutil.rmtree(checkpoint_state_dir)
            checkpoint_ckpt = os.path.join(
                args.output_dir,
                train_util.get_checkpoint_ckpt_name(args, "." + args.save_model_as),
            )
            if os.path.exists(checkpoint_ckpt):
                logger.info(f"removing checkpoint weights: {checkpoint_ckpt}")
                os.remove(checkpoint_ckpt)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
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
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    train_util.add_dit_training_arguments(parser)
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

    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="pretrained weights for network",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        help="network module to train",
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network)",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value)",
    )
    parser.add_argument(
        "--network_train_unet_only",
        action="store_true",
        help="only training U-Net part",
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point)",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training",
    )
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
        help="Sigma values for validation loss (0.0~1.0). Low values = fine detail. Default: 0.05 0.10 0.20 0.35",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    trainer = AnimaTrainer()
    trainer.train(args)
