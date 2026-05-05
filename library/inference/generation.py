"""Core generation logic for Anima inference: denoising loops and tiled diffusion."""

import argparse
import logging
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Any, Dict, Tuple, Union

import torch
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from library.anima import models as anima_models
from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma
from library.inference import sampling as inference_utils
from library.inference.output import check_inputs
from library.inference.text import prepare_text_inputs
from library.inference.models import load_dit_model
from library.inference.mod_guidance import setup_mod_guidance

logger = logging.getLogger(__name__)

# Spectrum runner registry. The spectrum implementation lives in
# networks/spectrum.py (or a downstream package); it self-registers on import.
# generation.py never imports it directly so the dep edge can point inward
# from a downstream inference package without inverting.
_SPECTRUM_RUNNER = None


def register_spectrum_runner(fn):
    """Plug in a spectrum_denoise implementation.

    The runner must accept the same kwargs as networks.spectrum.spectrum_denoise.
    Called by networks/spectrum.py at import time, or by a downstream inference
    package that ships its own spectrum module.
    """
    global _SPECTRUM_RUNNER
    _SPECTRUM_RUNNER = fn


class GenerationSettings:
    def __init__(
        self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
    ):
        self.device = device
        self.dit_weight_dtype = (
            dit_weight_dtype  # not used currently because model may be optimized
        )


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = torch.bfloat16  # default

    logger.info(
        f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}"
    )

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


# region Tiling helpers


def compute_tile_positions(
    h_latent: int, w_latent: int, tile_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """Compute (y, x) start positions for overlapping tiles covering the full latent grid."""
    stride = tile_size - overlap
    positions = []
    y = 0
    while y < h_latent:
        if y + tile_size > h_latent:
            y = h_latent - tile_size  # clamp last row
        x = 0
        while x < w_latent:
            if x + tile_size > w_latent:
                x = w_latent - tile_size  # clamp last column
            positions.append((y, x))
            if x + tile_size >= w_latent:
                break
            x += stride
        if y + tile_size >= h_latent:
            break
        y += stride
    return positions


def create_tile_blend_weight(
    tile_h: int,
    tile_w: int,
    overlap: int,
    y: int,
    x: int,
    h_latent: int,
    w_latent: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a (1, 1, 1, tile_h, tile_w) blend weight with cosine ramps on overlapping edges."""
    weight = torch.ones(1, 1, 1, tile_h, tile_w, device=device, dtype=dtype)
    if overlap <= 0:
        return weight

    # Precompute cosine ramp: (1 - cos(pi * t)) / 2 for t in [0, 1]
    ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
    ramp = (1.0 - torch.cos(math.pi * ramp)) / 2.0

    # Top edge
    if y > 0:
        weight[:, :, :, :overlap, :] *= ramp[None, None, None, :, None]
    # Bottom edge
    if y + tile_h < h_latent:
        weight[:, :, :, -overlap:, :] *= ramp.flip(0)[None, None, None, :, None]
    # Left edge
    if x > 0:
        weight[:, :, :, :, :overlap] *= ramp[None, None, None, None, :]
    # Right edge
    if x + tile_w < w_latent:
        weight[:, :, :, :, -overlap:] *= ramp.flip(0)[None, None, None, None, :]

    return weight


# endregion


# region Tiled generation


def generate_body_tiled(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """MultiDiffusion-style tiled denoising for high-resolution generation."""
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(
        f"Tiled diffusion: image size {height}x{width} (HxW), infer_steps: {args.infer_steps}"
    )

    tile_size = args.tile_size
    overlap = args.tile_overlap
    patch_spatial = anima.patch_spatial

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Prefix tuning: prepend learned vectors to cached adapter output
    prefix_weight = getattr(args, "prefix_weight", None)
    if prefix_weight is not None:
        from networks.methods.postfix import create_network_from_weights

        prefix_net, prefix_sd = create_network_from_weights(
            multiplier=1.0, file=prefix_weight, ae=None, text_encoders=None, unet=None
        )
        prefix_net.load_weights(prefix_weight)
        prefix_net.to(device, dtype=torch.bfloat16)
        embed = prefix_net.prepend_prefix(embed)
        negative_embed = prefix_net.prepend_prefix(negative_embed)
        logger.info(
            f"Prefix: prepended {prefix_net.num_postfix_tokens} tokens, embed shape now {embed.shape}"
        )

    # Postfix tuning: append learned vectors after real text tokens
    postfix_weight = getattr(args, "postfix_weight", None)
    if postfix_weight is not None:
        from networks.methods.postfix import create_network_from_weights

        postfix_net, postfix_sd = create_network_from_weights(
            multiplier=1.0, file=postfix_weight, ae=None, text_encoders=None, unet=None
        )
        if postfix_net.mode == "cond-timestep":
            raise NotImplementedError(
                "cond-timestep postfix + tiled diffusion is not yet supported. "
                "Disable --tiled_diffusion or use a cond/postfix checkpoint."
            )
        postfix_net.load_weights(postfix_weight)
        postfix_net.to(device, dtype=torch.bfloat16)
        embed_mask = context["embed"][3].to(device)
        embed_seqlens = embed_mask.sum(dim=-1).to(torch.int32)
        embed = postfix_net.append_postfix(embed, embed_seqlens)
        neg_mask = context_null["embed"][3].to(device)
        neg_seqlens = neg_mask.sum(dim=-1).to(torch.int32)
        negative_embed = postfix_net.append_postfix(negative_embed, neg_seqlens)
        logger.info(
            f"Postfix: appended {postfix_net.num_postfix_tokens} tokens after text"
        )

    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    h_latent = height // 8
    w_latent = width // 8
    shape = (1, num_channels_latents, 1, h_latent, w_latent)
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    # Compute tile positions and precompute blend weights
    positions = compute_tile_positions(h_latent, w_latent, tile_size, overlap)
    logger.info(
        f"Tiled diffusion: {len(positions)} tiles, tile_size={tile_size}, overlap={overlap}"
    )

    blend_weights = {}
    for y, x in positions:
        tile_h = min(tile_size, h_latent - y)
        tile_w = min(tile_size, w_latent - x)
        blend_weights[(y, x)] = create_tile_blend_weight(
            tile_h, tile_w, overlap, y, x, h_latent, w_latent, device, torch.bfloat16
        )

    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps /= 1000
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    # Create sampler
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(sigmas, seed=args.seed, device=device)

    do_cfg = args.guidance_scale != 1.0
    autocast_enabled = args.fp8

    # P-GRAFT: get network reference for mid-denoising cutoff
    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    try:
        with tqdm(total=len(timesteps), desc="Denoising steps (tiled)") as pbar:
            for i, t in enumerate(timesteps):
                # P-GRAFT: disable LoRA at cutoff step
                if (
                    pgraft_network is not None
                    and lora_cutoff_step is not None
                    and i == lora_cutoff_step
                ):
                    pgraft_network.set_enabled(False)
                    logger.info(f"P-GRAFT: Disabled LoRA at step {i}/{len(timesteps)}")

                t_expand = t.expand(latents.shape[0])
                set_hydra_sigma(anima, t_expand)

                noise_acc = torch.zeros_like(latents)
                weight_acc = torch.zeros(
                    1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
                )

                if do_cfg:
                    uncond_noise_acc = torch.zeros_like(latents)
                    uncond_weight_acc = torch.zeros(
                        1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
                    )

                for y, x in positions:
                    tile_h = min(tile_size, h_latent - y)
                    tile_w = min(tile_size, w_latent - x)
                    tile_latent = latents[:, :, :, y : y + tile_h, x : x + tile_w]
                    tile_padding_mask = torch.zeros(
                        1, 1, tile_h, tile_w, dtype=torch.bfloat16, device=device
                    )

                    h_off = y // patch_spatial
                    w_off = x // patch_spatial

                    bw = blend_weights[(y, x)]

                    # Conditional pass
                    if anima.blocks_to_swap:
                        anima.prepare_block_swap_before_forward()
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=autocast_enabled,
                        ),
                    ):
                        tile_pred = anima(
                            tile_latent,
                            t_expand,
                            embed,
                            padding_mask=tile_padding_mask,
                            h_offset=h_off,
                            w_offset=w_off,
                        )
                    noise_acc[:, :, :, y : y + tile_h, x : x + tile_w] += (
                        tile_pred * bw
                    )
                    weight_acc[:, :, :, y : y + tile_h, x : x + tile_w] += bw

                    # Unconditional pass
                    if do_cfg:
                        if anima.blocks_to_swap:
                            anima.prepare_block_swap_before_forward()
                        with (
                            torch.no_grad(),
                            torch.autocast(
                                device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=autocast_enabled,
                            ),
                        ):
                            uncond_tile_pred = anima(
                                tile_latent,
                                t_expand,
                                negative_embed,
                                padding_mask=tile_padding_mask,
                                h_offset=h_off,
                                w_offset=w_off,
                            )
                        uncond_noise_acc[:, :, :, y : y + tile_h, x : x + tile_w] += (
                            uncond_tile_pred * bw
                        )
                        uncond_weight_acc[:, :, :, y : y + tile_h, x : x + tile_w] += bw

                noise_pred = noise_acc / weight_acc
                if do_cfg:
                    uncond_noise_pred = uncond_noise_acc / uncond_weight_acc
                    noise_pred = uncond_noise_pred + args.guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                denoised = latents.float() - sigmas[i] * noise_pred.float()
                if er_sde is not None:
                    new_latents = er_sde.step(latents, denoised, i)
                else:
                    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

                if getattr(args, "dcw", False) and float(sigmas[i + 1]) > 0.0:
                    from networks.dcw import apply_dcw, parse_band_mask

                    new_latents = apply_dcw(
                        new_latents.float(),
                        denoised,
                        float(sigmas[i]),
                        lam=args.dcw_lambda,
                        schedule=args.dcw_schedule,
                        bands=parse_band_mask(getattr(args, "dcw_band_mask", "LL")),
                    )

                latents = new_latents.to(latents.dtype)
                pbar.update()
    finally:
        clear_hydra_sigma(anima)
        # P-GRAFT: restore LoRA for next generation
        if pgraft_network is not None and lora_cutoff_step is not None:
            pgraft_network.set_enabled(True)

    return latents


# endregion


# region Core generation


def generate_body(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: Union[int, List[int]],
    latents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Core denoising loop for Anima generation.

    Args:
        args: Generation arguments (image_size, infer_steps, guidance_scale, etc.)
        anima: Loaded DiT model.
        context: Dict with "embed" key containing text encoder outputs.
        context_null: Dict with negative prompt embeddings (or None for unconditional).
        device: Target device.
        seed: Single seed or list of seeds (for batch generation).
        latents: Optional pre-created latent noise tensor.  When provided, the
            batch dimension is taken from this tensor and seed is ignored for
            noise creation.  This enables callers (e.g. batch mode) to construct
            multi-seed batched latents externally.

    Returns:
        Denoised latent tensor (batch dimension preserved).
    """

    height, width = check_inputs(args)

    # Dispatch to tiled diffusion if enabled and latent exceeds tile size
    h_latent = height // 8
    w_latent = width // 8
    if (
        latents is None
        and getattr(args, "tiled_diffusion", False)
        and (h_latent > args.tile_size or w_latent > args.tile_size)
    ):
        return generate_body_tiled(args, anima, context, context_null, device, seed)

    # Create latents if not provided
    if latents is None:
        seed_g = torch.Generator(device="cpu")
        seed_g.manual_seed(seed if isinstance(seed, int) else seed[0])

        logger.info(
            f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}"
        )

        num_channels_latents = anima_models.Anima.LATENT_CHANNELS
        shape = (1, num_channels_latents, 1, height // 8, width // 8)
        latents = randn_tensor(
            shape, generator=seed_g, device=device, dtype=torch.bfloat16
        )

    bs = latents.shape[0]
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]

    logger.info(
        f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}, batch: {bs}"
    )

    # image generation ######

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context  # dummy for unconditional
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Prefix/postfix tuning: inject learned vectors into cross-attention embeddings.
    # Pool text BEFORE injection so modulation guidance sees only real text tokens.
    _pooled_text_pos = None
    _pooled_text_neg = None

    prefix_weight = getattr(args, "prefix_weight", None)
    postfix_weight = getattr(args, "postfix_weight", None)
    if prefix_weight is not None or postfix_weight is not None:
        _pooled_text_pos = embed.max(dim=1).values  # (1, 1024)
        _pooled_text_neg = negative_embed.max(dim=1).values

    if prefix_weight is not None:
        from networks.methods.postfix import create_network_from_weights

        prefix_net, prefix_sd = create_network_from_weights(
            multiplier=1.0, file=prefix_weight, ae=None, text_encoders=None, unet=None
        )
        prefix_net.load_weights(prefix_weight)
        prefix_net.to(device, dtype=torch.bfloat16)
        embed = prefix_net.prepend_prefix(embed)
        negative_embed = prefix_net.prepend_prefix(negative_embed)
        logger.info(
            f"Prefix: prepended {prefix_net.num_postfix_tokens} tokens, embed shape now {embed.shape}"
        )

    postfix_net = None
    embed_seqlens = None
    neg_seqlens = None
    postfix_base_embed = None
    postfix_base_neg = None
    if postfix_weight is not None:
        from networks.methods.postfix import create_network_from_weights

        postfix_net, postfix_sd = create_network_from_weights(
            multiplier=1.0, file=postfix_weight, ae=None, text_encoders=None, unet=None
        )
        postfix_net.load_weights(postfix_weight)
        postfix_net.to(device, dtype=torch.bfloat16)
        # Compute seqlens from attention masks
        embed_mask = context["embed"][3].to(device)
        embed_seqlens = embed_mask.sum(dim=-1).to(torch.int32)
        neg_mask = context_null["embed"][3].to(device)
        neg_seqlens = neg_mask.sum(dim=-1).to(torch.int32)
        if postfix_net.mode == "cond-timestep":
            # σ-conditional: defer postfix application to inside the denoising loop
            # so it's recomputed per timestep. Stash the un-postfixed base embeds.
            postfix_base_embed = embed
            postfix_base_neg = negative_embed
            logger.info(
                f"Postfix (cond-timestep): deferring per-step injection of "
                f"{postfix_net.num_postfix_tokens} tokens"
            )
        else:
            embed = postfix_net.append_postfix(embed, embed_seqlens)
            negative_embed = postfix_net.append_postfix(negative_embed, neg_seqlens)
            logger.info(
                f"Postfix: appended {postfix_net.num_postfix_tokens} tokens after text"
            )

    # Create padding mask
    padding_mask = torch.zeros(
        bs, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device
    )

    logger.info(
        f"Embed: {embed.shape}, negative_embed: {negative_embed.shape}, latents: {latents.shape}"
    )

    # Expand embeddings to batch size
    if embed.shape[0] < bs:
        embed = embed.expand(bs, -1, -1)
    if negative_embed.shape[0] < bs:
        negative_embed = negative_embed.expand(bs, -1, -1)

    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    # Keep the σ-conditional postfix base in sync with shaping/cast above.
    if postfix_base_embed is not None:
        postfix_base_embed = embed
        postfix_base_neg = negative_embed
        if embed_seqlens.shape[0] < bs:
            embed_seqlens = embed_seqlens.expand(bs)
            neg_seqlens = neg_seqlens.expand(bs)

    # Prepare timesteps
    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps /= 1000  # scale to [0,1] range
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    # DCW: load + setup the learnable calibrator if requested.
    dcw_calibrator = None
    calibrator_path = getattr(args, "dcw_calibrator", None) or getattr(args, "dcw_v4", None)
    if calibrator_path:
        from library.inference.dcw_calibrator import OnlineDCWCalibrator

        artifact_path = Path(calibrator_path)
        if artifact_path.is_dir():
            artifact_path = artifact_path / "fusion_head.safetensors"
        try:
            dcw_calibrator = OnlineDCWCalibrator.from_safetensors(
                artifact_path, device=device
            )
        except Exception as e:
            logger.warning("DCW calibrator: failed to load %s: %s — disabling", artifact_path, e)
            dcw_calibrator = None
        if dcw_calibrator is not None:
            calib_embed_mask = (
                context["embed"][3].to(device)
                if len(context["embed"]) > 3 and context["embed"][3] is not None
                else None
            )
            dcw_calibrator.setup(
                embed=embed, embed_mask=calib_embed_mask,
                gain=getattr(args, "dcw_calibrator_gain", None)
                    or getattr(args, "dcw_v4_alpha_gain", 1.0),
            )
            if not dcw_calibrator.is_active:
                dcw_calibrator = None  # graceful degrade to scalar/none

    # Create sampler
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(sigmas, seed=args.seed, device=device)

    # Denoising loop
    do_cfg = args.guidance_scale != 1.0
    autocast_enabled = args.fp8

    # P-GRAFT: get network reference for mid-denoising cutoff
    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    if getattr(args, "spectrum", False):
        if _SPECTRUM_RUNNER is None:
            raise RuntimeError(
                "--spectrum was passed but no spectrum runner is registered. "
                "Import networks.spectrum (or the downstream inference package "
                "that registers it) before calling generate()."
            )

        latents = _SPECTRUM_RUNNER(
            anima,
            latents,
            timesteps,
            sigmas,
            embed,
            negative_embed,
            padding_mask,
            args.guidance_scale,
            er_sde,
            device,
            window_size=getattr(args, "spectrum_window_size", 2.0),
            flex_window=getattr(args, "spectrum_flex_window", 0.25),
            warmup_steps=getattr(args, "spectrum_warmup", 6),
            w=getattr(args, "spectrum_w", 0.3),
            m=getattr(args, "spectrum_m", 3),
            lam=getattr(args, "spectrum_lam", 0.1),
            stop_caching_step=getattr(args, "spectrum_stop_caching_step", -1),
            calibration_strength=getattr(args, "spectrum_calibration", 0.0),
            autocast_enabled=autocast_enabled,
            pgraft_network=pgraft_network,
            pooled_text_pos=_pooled_text_pos,
            pooled_text_neg=_pooled_text_neg,
            lora_cutoff_step=lora_cutoff_step,
            postfix_net=postfix_net if postfix_base_embed is not None else None,
            postfix_base_embed=postfix_base_embed,
            postfix_base_neg=postfix_base_neg,
            postfix_embed_seqlens=embed_seqlens,
            postfix_neg_seqlens=neg_seqlens,
            dcw=getattr(args, "dcw", False),
            dcw_lambda=getattr(args, "dcw_lambda", -0.015),
            dcw_schedule=getattr(args, "dcw_schedule", "one_minus_sigma"),
            dcw_band_mask=getattr(args, "dcw_band_mask", "LL"),
            dcw_calibrator=dcw_calibrator,
        )
    else:
        try:
            with tqdm(total=len(timesteps), desc=f"Denoising steps ({bs}x)") as pbar:
                for i, t in enumerate(timesteps):
                    # P-GRAFT: disable LoRA at cutoff step (reference model takes over)
                    if (
                        pgraft_network is not None
                        and lora_cutoff_step is not None
                        and i == lora_cutoff_step
                    ):
                        pgraft_network.set_enabled(False)
                        logger.info(
                            f"P-GRAFT: Disabled LoRA at step {i}/{len(timesteps)}"
                        )

                    t_expand = t.expand(latents.shape[0])
                    set_hydra_sigma(anima, t_expand)

                    # σ-conditional postfix: recompute per step against base embeds.
                    if postfix_base_embed is not None:
                        step_embed = postfix_net.append_postfix(
                            postfix_base_embed, embed_seqlens, timesteps=t_expand
                        )
                        step_negative = postfix_net.append_postfix(
                            postfix_base_neg, neg_seqlens, timesteps=t_expand
                        )
                    else:
                        step_embed = embed
                        step_negative = negative_embed

                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=autocast_enabled,
                        ),
                    ):
                        _pos_kw = (
                            {"pooled_text_override": _pooled_text_pos}
                            if _pooled_text_pos is not None
                            else {}
                        )
                        noise_pred = anima(
                            latents,
                            t_expand,
                            step_embed,
                            padding_mask=padding_mask,
                            **_pos_kw,
                        )

                    if do_cfg:
                        with (
                            torch.no_grad(),
                            torch.autocast(
                                device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=autocast_enabled,
                            ),
                        ):
                            _neg_kw = (
                                {"pooled_text_override": _pooled_text_neg}
                                if _pooled_text_neg is not None
                                else {}
                            )
                            uncond_noise_pred = anima(
                                latents,
                                t_expand,
                                step_negative,
                                padding_mask=padding_mask,
                                **_neg_kw,
                            )
                        noise_pred = uncond_noise_pred + args.guidance_scale * (
                            noise_pred - uncond_noise_pred
                        )

                    # ensure latents dtype is consistent
                    denoised = latents.float() - sigmas[i] * noise_pred.float()
                    if er_sde is not None:
                        new_latents = er_sde.step(latents, denoised, i)
                    else:
                        new_latents = inference_utils.step(
                            latents, noise_pred, sigmas, i
                        )

                    if dcw_calibrator is not None:
                        dcw_calibrator.record(i, noise_pred)
                        dcw_calibrator.fire_head_if_due(i)

                    lam_i_calib: Optional[float] = None
                    if float(sigmas[i + 1]) > 0.0 and (
                        dcw_calibrator is not None or getattr(args, "dcw", False)
                    ):
                        from networks.dcw import apply_dcw, parse_band_mask

                        if dcw_calibrator is not None:
                            lam_i_calib = dcw_calibrator.lambda_for_step(i, float(sigmas[i]))
                            new_latents = apply_dcw(
                                new_latents.float(), denoised, float(sigmas[i]),
                                lam=lam_i_calib, schedule="const",
                                bands=frozenset({"LL"}),
                            )
                        else:
                            new_latents = apply_dcw(
                                new_latents.float(),
                                denoised,
                                float(sigmas[i]),
                                lam=args.dcw_lambda,
                                schedule=args.dcw_schedule,
                                bands=parse_band_mask(
                                    getattr(args, "dcw_band_mask", "LL")
                                ),
                            )

                    latents = new_latents.to(latents.dtype)

                    if dcw_calibrator is not None and dcw_calibrator.is_active:
                        # λ_scalar = α̂·gain — the prompt-adaptive equivalent of
                        # the bench's fixed `lam=0.01, schedule="one_minus_sigma"` scalar.
                        # λ_i is the post-envelope per-step value actually applied.
                        lam_scalar = dcw_calibrator.gain * dcw_calibrator.alpha_eff
                        if i < dcw_calibrator.k_warmup:
                            pbar.set_postfix_str(
                                f"λ_i={lam_i_calib:+.4f} (warmup {i + 1}/{dcw_calibrator.k_warmup})"
                                if lam_i_calib is not None else f"warmup {i + 1}/{dcw_calibrator.k_warmup}"
                            )
                        else:
                            pbar.set_postfix_str(
                                f"λ_scalar={lam_scalar:+.4f} λ_i={lam_i_calib:+.4f} "
                                f"α={dcw_calibrator.alpha_eff:+.4g}"
                                if lam_i_calib is not None
                                else f"λ_scalar={lam_scalar:+.4f} α={dcw_calibrator.alpha_eff:+.4g}"
                            )

                    pbar.update()
        finally:
            clear_hydra_sigma(anima)
            # P-GRAFT: restore LoRA for next generation
            if pgraft_network is not None and lora_cutoff_step is not None:
                pgraft_network.set_enabled(True)

    return latents


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> torch.Tensor:
    """Main function for generation.

    Returns:
        torch.Tensor: generated latent
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    if shared_models is None or "model" not in shared_models:
        # load DiT model
        anima = load_dit_model(args, device, dit_weight_dtype)

        if shared_models is not None:
            shared_models["model"] = anima
    else:
        # use shared model
        logger.info("Using shared DiT model.")
        anima: anima_models.Anima = shared_models["model"]

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]

    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, anima, shared_models)

    # Phase 2 modulation guidance: compute guidance delta once
    if (
        getattr(args, "pooled_text_proj", None) is not None
        and getattr(args, "mod_w", 0.0) != 0.0
    ):
        setup_mod_guidance(args, anima, device, shared_models)
    else:
        anima.reset_mod_guidance()

    # IP-Adapter: load + apply network, encode reference image, prime per-block
    # K/V on the network. The patched cross-attn closures pull from the cache
    # for both cond and uncond passes (image stays on through CFG; text is the
    # CFG steering knob).
    _setup_ip_adapter(args, anima, device)

    # EasyControl: load + apply network, VAE-encode reference image, run cond
    # pre-pass to prime per-block (K_c, V_c). Phase 1 — recomputed every step
    # at training; at inference we run it once here (no KV cache yet).
    _setup_easycontrol(args, anima, device, shared_models)

    return generate_body(args, anima, context, context_null, device, seed)


def _setup_ip_adapter(args, anima, device):
    ip_weight = getattr(args, "ip_adapter_weight", None)
    ip_image = getattr(args, "ip_image", None)
    if ip_weight is None and ip_image is None:
        return None
    if ip_weight is None or ip_image is None:
        raise ValueError(
            "--ip_adapter_weight and --ip_image must be passed together "
            f"(got ip_adapter_weight={ip_weight!r}, ip_image={ip_image!r})"
        )

    from PIL import Image
    from torchvision import transforms

    from networks.methods.ip_adapter import create_network_from_weights
    from library.vision import encode_pe_from_imageminus1to1, load_pe_encoder

    # Aspect-match: snap --image_size to the CONSTANT_TOKEN_BUCKETS entry whose
    # aspect is closest to the reference. Done BEFORE encoding so the same
    # aspect drives both the generated latent and the PE-side bucket pick.
    if getattr(args, "ip_image_match_size", False):
        from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS

        with Image.open(ip_image) as _ref_for_size:
            _rw, _rh = _ref_for_size.size
        _target = _rw / _rh
        _best_wh = min(
            CONSTANT_TOKEN_BUCKETS, key=lambda wh: abs((wh[0] / wh[1]) - _target)
        )
        # check_inputs reads args.image_size as [H, W].
        args.image_size = [_best_wh[1], _best_wh[0]]
        logger.info(
            f"IP-Adapter: image_size auto-picked from ref (aspect w/h={_target:.3f}) "
            f"-> {tuple(args.image_size)} (HxW)"
        )

    create_kwargs = {}
    if getattr(args, "ip_scale", None) is not None:
        create_kwargs["ip_scale"] = float(args.ip_scale)

    network, _sd = create_network_from_weights(
        multiplier=1.0,
        file=ip_weight,
        ae=None,
        text_encoders=None,
        unet=anima,
        **create_kwargs,
    )
    network.load_weights(ip_weight)
    network.to(device, dtype=torch.bfloat16)
    network.apply_to(text_encoders=None, unet=anima)

    bundle = load_pe_encoder(device, name=network.encoder_name, dtype=torch.bfloat16)

    img = Image.open(ip_image).convert("RGB")
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    img_t = tfm(img).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1, 3, H, W] in [-1, 1]

    with torch.no_grad():
        feats_list = encode_pe_from_imageminus1to1(bundle, img_t, same_bucket=True)
        ip_features = torch.stack(feats_list, dim=0)  # [1, T_pe, d_enc]
        ip_tokens = network.encode_ip_tokens(ip_features.to(torch.bfloat16))

    network.set_ip_tokens(ip_tokens)
    logger.info(
        f"IP-Adapter: loaded {ip_weight} (encoder={network.encoder_name}, "
        f"K={network.num_ip_tokens}, scale={network.get_effective_scale():.3f})"
    )
    # Stash on anima so the caller can keep the network alive for the duration
    # of generation (Python won't GC it while it's reachable from the model).
    anima._ip_adapter_network = network
    return network


def _setup_easycontrol(args, anima, device, shared_models):
    """Load EasyControl weights, VAE-encode the reference image, prime cond KV cache.

    The cond stream is deterministic across denoising steps (cond_temb at t=0,
    no dependence on noisy target, frozen DiT + frozen LoRA), so we run it
    once via ``network.precompute_cond_kv()`` and reuse the per-block
    (K_c, V_c) tensors for every step and every CFG branch — the patched
    Block.forward then bypasses the cond stream entirely.
    """
    ec_weight = getattr(args, "easycontrol_weight", None)
    ec_image = getattr(args, "easycontrol_image", None)
    if ec_weight is None and ec_image is None:
        return None
    if ec_weight is None or ec_image is None:
        raise ValueError(
            "--easycontrol_weight and --easycontrol_image must be passed together "
            f"(got easycontrol_weight={ec_weight!r}, easycontrol_image={ec_image!r})"
        )

    from PIL import Image
    from torchvision import transforms

    from networks.methods.easycontrol import create_network_from_weights
    from library.models import qwen_vae as qwen_image_autoencoder_kl

    if getattr(args, "easycontrol_image_match_size", False):
        from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS

        with Image.open(ec_image) as _ref_for_size:
            _rw, _rh = _ref_for_size.size
        _target = _rw / _rh
        _best_wh = min(
            CONSTANT_TOKEN_BUCKETS, key=lambda wh: abs((wh[0] / wh[1]) - _target)
        )
        args.image_size = [_best_wh[1], _best_wh[0]]
        logger.info(
            f"EasyControl: image_size auto-picked from ref (aspect w/h={_target:.3f}) "
            f"-> {tuple(args.image_size)} (HxW)"
        )

    create_kwargs = {}
    if getattr(args, "easycontrol_scale", None) is not None:
        create_kwargs["cond_scale"] = float(args.easycontrol_scale)

    network, _sd = create_network_from_weights(
        multiplier=1.0,
        file=ec_weight,
        ae=None,
        text_encoders=None,
        unet=anima,
        **create_kwargs,
    )
    network.load_weights(ec_weight)
    network.to(device, dtype=torch.bfloat16)
    network.apply_to(text_encoders=None, unet=anima)

    # VAE-encode the reference image -> 4D latent.
    # Resize to args.image_size first so the cond bucket matches the target.
    h_pix, w_pix = args.image_size
    img = Image.open(ec_image).convert("RGB").resize((w_pix, h_pix), Image.LANCZOS)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    img_t = tfm(img).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1,3,H,W] in [-1,1]

    vae = (shared_models or {}).get("vae")
    vae_was_shared = vae is not None
    if vae is None:
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
        )
        vae.to(torch.bfloat16)
        vae.eval()
        vae.to(device)

    with torch.no_grad():
        cond_latent_5d = vae.encode_pixels_to_latents(img_t)  # [1, C, 1, H', W']
        cond_latent = cond_latent_5d.squeeze(2)  # [1, C, H', W']

    if not vae_was_shared:
        vae.to("cpu")
        del vae
        torch.cuda.empty_cache()

    network.set_cond(cond_latent.to(device, dtype=torch.bfloat16))
    # KV cache: walk the cond stream once and pin per-block (K_c, V_c). Every
    # subsequent denoising step (and CFG branch) feeds these into target's
    # extended self-attention without re-running the cond stream.
    network.precompute_cond_kv()
    logger.info(
        f"EasyControl: loaded {ec_weight} "
        f"(r={network.cond_lora_dim}, scale={network.get_effective_scale():.3f}, kv-cached)"
    )
    anima._easycontrol_network = network
    return network


# endregion
