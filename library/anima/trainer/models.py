"""Model loading for the Anima trainer: Qwen3 TE + VAE first (needed for
caching), the DiT after the caches are populated (lazy-loading order)."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from library.anima import weights as anima_utils
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.training.mem_reweight import MemGapTracker

logger = logging.getLogger(__name__)


def load_text_encoder_and_vae(args, weight_dtype, *, load_qwen3=True, load_vae=True):
    """``(qwen3_text_encoder | None, vae | None)`` on CPU. Each is skipped when
    every output it would produce is already cached and nothing live
    (sampling, CMMD, live encoding) needs it."""
    if load_qwen3:
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
            args.qwen3, dtype=weight_dtype, device="cpu"
        )
        qwen3_text_encoder.eval()
    else:
        logger.info(
            "Skipping Qwen3 text encoder load: all text-encoder outputs cached."
        )
        qwen3_text_encoder = None

    if load_vae:
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
    else:
        logger.info("Skipping VAE load: all latents cached and no sampling.")
        vae = None

    return qwen3_text_encoder, vae


def load_dit(args, weight_dtype, accelerator, *, swapping: bool) -> nn.Module:
    """Load the DiT (frozen LoRA merged in, pooled_text_proj installed, block
    swap enabled). Not compiled here: ``compile_blocks`` must follow
    ``apply_to`` + ``load_weights`` (see ``network.create_and_apply_network``
    and library/runtime/harness.py)."""
    loading_device = "cpu" if swapping else accelerator.device

    from library.runtime.backend import (
        needs_rocm_attention_fallback,
        resolve_attention_mode,
    )

    requested_attn_mode = args.attn_mode
    attn_mode = resolve_attention_mode(requested_attn_mode, torch)
    if needs_rocm_attention_fallback(requested_attn_mode, torch):
        logger.warning(
            "ROCm detected: using PyTorch SDPA instead of CUDA Flash Attention"
        )

    if attn_mode == "flash4":
        raise RuntimeError(
            "attn_mode='flash4' is not supported yet -- the flash-attention-sm120 "
            "kernel is disabled in this build. Use 'flash', 'torch', 'flex', "
            "or 'sageattn' instead."
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
    # Used by postfix runs that train on top of a fixed LoRA.
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

    attn_softmax_scale = getattr(args, "attn_softmax_scale", None)
    logger.info(
        f"Loading Anima DiT model with attn_softmax_scale: {attn_softmax_scale}..."
    )
    model = anima_utils.load_anima_model(
        accelerator.device,
        args.pretrained_model_name_or_path,
        attn_mode,
        loading_device,
        weight_dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        attn_softmax_scale=attn_softmax_scale,
        # Sample prompts tokenized through the pack need its rows behind
        # the ext ids (same table the TE caches were built with).
        vocab_pack=getattr(args, "vocab_pack", None),
    )

    # Mod-aware training: install the distilled pooled_text_proj so an
    # adaln LoRA trains against the operating point mod-guidance perturbs
    # at inference. Frozen implicitly (excluded from the LoRA factory).
    if getattr(args, "pooled_text_proj", None):
        anima_utils.load_pooled_text_proj(model, args.pooled_text_proj, "cpu")
        model.pooled_text_proj.to(device=loading_device, dtype=weight_dtype)

    if swapping:
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        model.enable_block_swap(args.blocks_to_swap, accelerator.device)

    return model


def build_mem_tracker(args, *, swapping: bool) -> MemGapTracker | None:
    """Online memorization Δ-gap tracker for ``--mem_reweight_mode``. The
    measurement is a second per-step DiT forward (the trainable DiT with
    ``network.set_multiplier(0)``, as for VR), unaudited against the
    block-swap offloader — raise, not warn."""
    mem_mode = str(getattr(args, "mem_reweight_mode", "") or "")
    if not mem_mode:
        return None
    if swapping:
        raise ValueError(
            "--mem_reweight_mode requires blocks_to_swap=0 — the "
            "Δ-gap measurement forward has not been validated against "
            "the block-swap offloader and can silently desync it. "
            "Use block compile for memory instead."
        )
    save_path = os.path.join(
        args.output_dir,
        f"{getattr(args, 'output_name', None) or 'lora'}_memgap.json",
    )
    tracker = MemGapTracker(args, save_path)
    logger.info(
        f"memorization Δ-gap tracker enabled (mode={mem_mode}, "
        f"σ≤{args.mem_sigma_max}, K={args.mem_measure_every}) — "
        f"state → {save_path}"
    )
    return tracker
