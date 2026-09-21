"""Core generation logic for Anima inference: denoising loops and tiled diffusion."""

import argparse
import logging
import math
import os
import random
from types import SimpleNamespace
from typing import Optional, List, Any, Dict, Tuple, Union

import torch
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from library.anima import models as anima_models
from library.inference.adapters import (
    clear_hydra_fei,
    clear_hydra_sigma,
    compute_and_set_hydra_fei,
    set_hydra_crossattn,
    set_hydra_sigma,
    set_step_expert_index,
    set_xattn_boost_state,
)
from library.inference import sampling as inference_utils
from library.inference.cfg_delta_probe import CfgDeltaProbe
from library.inference.output import check_inputs
from library.inference.text import prepare_text_inputs
from library.inference.models import load_dit_model
from library.inference.corrections.mod_guidance import setup_mod_guidance
from library.inference.corrections.smc_cfg import SMCCFGState
from library.inference.corrections.fsg import FSGCalibrator
from library.inference.sampler_context import SamplerSideChannels
from library.inference.traj_stats import TrajStatsRecorder

logger = logging.getLogger(__name__)


def _build_cns_recolorer(args):
    """Build a CNS noise recolorer from ``--cns`` (path or "auto"), or None.

    Only meaningful on the stochastic ``er_sde`` path — on euler/lcm there is
    no injected noise to recolor.
    """
    spec = getattr(args, "cns", None)
    if not spec:
        return None
    if getattr(args, "sampler", "euler") != "er_sde":
        logger.warning(
            "--cns is set but --sampler=%s injects no noise; CNS is a no-op "
            "(use --sampler er_sde).",
            getattr(args, "sampler", "euler"),
        )
        return None
    from library.inference.corrections.cns import CNSRecolorer

    recolorer = CNSRecolorer.from_path(
        spec, strength=getattr(args, "cns_strength", 1.0)
    )
    logger.info("CNS noise recoloring enabled (strength=%.2f).", recolorer.strength)
    return recolorer


# Runner registries: networks/spectrum.py, networks/spd.py, networks/foveated.py
# self-register on import (generation.py never imports them directly, so the
# dep edge can point inward from a downstream package without inverting).
_SPECTRUM_RUNNER = None
_SPD_RUNNER = None
_FOVEATED_RUNNER = None


def _setup_soft_tokens(args, anima, device):
    """Build + apply the soft_tokens network from ``--soft_tokens_weight``.

    Returns ``None`` when the flag isn't set. Otherwise returns the network with
    ``apply_to(unet=anima)`` already called — the per-block ``Block.forward``
    monkey-patches are live, but ``_step_layer_tokens`` is empty until the
    caller fires ``network.append_postfix(..., timesteps=t)`` each step.
    """
    soft_weight = getattr(args, "soft_tokens_weight", None)
    if soft_weight is None:
        return None
    from networks.methods.soft_tokens import create_network_from_weights

    net, _ = create_network_from_weights(
        multiplier=1.0,
        file=soft_weight,
        ae=None,
        text_encoders=None,
        unet=anima,
        for_inference=True,
    )
    net.load_weights(soft_weight)
    net.to(device, dtype=torch.bfloat16)
    net.apply_to(
        text_encoders=None, unet=anima, apply_text_encoder=False, apply_unet=True
    )
    logger.info(
        f"soft_tokens: loaded {soft_weight} "
        f"(n_layers={net.n_layers}, K={net.num_tokens}, "
        f"n_t_buckets={net.n_t_buckets}, splice={net.splice_position})"
    )
    return net


def _seqlens_from_context(context_dict, device):
    """Extract per-sample text seqlens from the context's attention mask.

    ``context['embed'][3]`` is the cached attention mask (1 inside text, 0 in
    padding) — sum along the sequence axis gives the real token count per
    sample, which the ``front_of_padding`` splice needs.
    """
    embed_mask = context_dict["embed"][3].to(device)
    return embed_mask.sum(dim=-1).to(torch.int32)


def _parse_spectrum_delta(raw):
    """Parse ``--spectrum_delta`` (str 'auto' or a float) into the runner's arg.

    Returns ``None`` for auto-calibration (the SEA accumulator self-tunes δ to
    the refresh ratio), or a float for an explicit threshold.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if str(raw).strip().lower() == "auto":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def register_spectrum_runner(fn):
    """Plug in a spectrum_denoise implementation (networks.spectrum's signature:
    core positional args + a ``SamplerSideChannels`` + spectrum keyword knobs).
    Called by networks/spectrum.py at import time.
    """
    global _SPECTRUM_RUNNER
    _SPECTRUM_RUNNER = fn


def register_spd_runner(fn):
    """Plug in an spd_denoise implementation. Mirrors register_spectrum_runner;
    called by networks/spd.py at import time."""
    global _SPD_RUNNER
    _SPD_RUNNER = fn


def register_foveated_runner(fn):
    """Plug in a foveated_denoise implementation. Mirrors
    register_spectrum_runner; called by networks/foveated.py at import time."""
    global _FOVEATED_RUNNER
    _FOVEATED_RUNNER = fn


def _resolve_spd_schedule(args) -> Tuple[List[float], List[float]]:
    """Resolve (stages, transition_sigmas) for --spd from CLI args.

    Default is the bench-recommended single-late knee: one handoff
    ``0.5 → 1.0`` at σ≈0.7 (conservative; gentler trajectory than σ0.5). A
    final ``1.0`` stage is appended automatically. ``--spd_transition_sigmas``
    must have ``len(stages) - 1`` entries.
    """
    stages = list(getattr(args, "spd_stages", None) or [0.5])
    if not stages or stages[-1] != 1.0:
        stages.append(1.0)
    transition_sigmas = list(getattr(args, "spd_transition_sigmas", None) or [])
    if not transition_sigmas:
        # one default σ per handoff; single-late knee for the common 2-stage case
        transition_sigmas = [0.7] * (len(stages) - 1)
    if len(transition_sigmas) != len(stages) - 1:
        raise ValueError(
            f"--spd_transition_sigmas needs {len(stages) - 1} value(s) for "
            f"stages {stages}, got {transition_sigmas}"
        )
    return stages, transition_sigmas


class GenerationSettings:
    # DiT always runs bf16 for inference (forced in load_dit_model).
    def __init__(self, device: torch.device):
        self.device = device


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    # programmatic callers (GenerationRequest.to_args(), bench probes) skip
    # parse_args' device-resolution block, so resolve cuda-else-cpu here too
    dev = getattr(args, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(dev)
    logger.info(f"Using device: {device}, DiT weight precision: bfloat16")
    return GenerationSettings(device=device)


def resolve_seed(args: argparse.Namespace) -> int:
    """Return the seed to use: ``args.seed`` if set, else a fresh random one.

    Pure — does **not** mutate ``args``. Callers that need ``args.seed`` set
    for downstream saving should assign the return value themselves.
    """
    return args.seed if args.seed is not None else random.randint(0, 2**32 - 1)


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

    if y > 0:
        weight[:, :, :, :overlap, :] *= ramp[None, None, None, :, None]
    if y + tile_h < h_latent:
        weight[:, :, :, -overlap:, :] *= ramp.flip(0)[None, None, None, :, None]
    if x > 0:
        weight[:, :, :, :, :overlap] *= ramp[None, None, None, None, :]
    if x + tile_w < w_latent:
        weight[:, :, :, :, -overlap:] *= ramp.flip(0)[None, None, None, None, :]

    return weight


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

    # soft tokens — see generate_body() for details
    soft_tokens_net = _setup_soft_tokens(args, anima, device)
    soft_tokens_embed_seqlens = (
        _seqlens_from_context(context, device) if soft_tokens_net is not None else None
    )
    soft_tokens_neg_seqlens = (
        _seqlens_from_context(context_null, device)
        if soft_tokens_net is not None
        else None
    )

    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    h_latent = height // 8
    w_latent = width // 8
    shape = (1, num_channels_latents, 1, h_latent, w_latent)
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

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
        args.infer_steps,
        args.flow_shift,
        device,
        tail_power=getattr(args, "sigma_tail_power", 1.0),
    )
    timesteps = timesteps.to(device, dtype=torch.bfloat16)  # σ∈[0,1] — DiT time arg

    # `er_sde` also holds an LCMSampler when selected; both share .step()
    cns = _build_cns_recolorer(args)
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(
            sigmas, seed=args.seed, device=device, cns=cns
        )
    elif args.sampler == "lcm":
        er_sde = inference_utils.LCMSampler(sigmas, seed=args.seed, device=device)

    do_cfg = args.guidance_scale != 1.0
    smc_cfg = (
        SMCCFGState(
            lam=args.smc_cfg_lambda,
            alpha=args.smc_cfg_alpha,
        )
        if do_cfg and getattr(args, "smc_cfg", False)
        else None
    )
    traj_stats = TrajStatsRecorder.from_args(args, seed=getattr(args, "seed", None))

    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    # xattn_boost: cond-forward-only cross-attn gain at σ ≥ band, per tile
    _boost_raw = float(getattr(args, "xattn_boost", 1.0) or 1.0)
    xattn_boost = _boost_raw if _boost_raw != 1.0 else None
    xattn_boost_band = float(getattr(args, "xattn_boost_band", 0.85))
    xattn_renorm = str(getattr(args, "xattn_boost_renorm", "img"))
    xattn_renorm_frac = float(getattr(args, "xattn_boost_renorm_frac", 0.5))

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
                set_step_expert_index(anima, i)
                # computed on the full latent (pre-tile) so every tile sees the
                # same per-sample FEI; no-op when no FEI router is attached
                compute_and_set_hydra_fei(anima, latents)

                noise_acc = torch.zeros_like(latents)
                weight_acc = torch.zeros(
                    1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
                )

                if do_cfg:
                    uncond_noise_acc = torch.zeros_like(latents)
                    uncond_weight_acc = torch.zeros(
                        1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
                    )

                _boost_step = (
                    xattn_boost is not None and float(sigmas[i]) >= xattn_boost_band
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

                    if anima.blocks_to_swap:
                        anima.prepare_block_swap_before_forward()
                    # caption-dependent routers; fire separately for cond vs uncond
                    set_hydra_crossattn(anima, embed)
                    if soft_tokens_net is not None:
                        soft_tokens_net.append_postfix(
                            embed, soft_tokens_embed_seqlens, timesteps=t_expand
                        )
                    if _boost_step:
                        set_xattn_boost_state(
                            anima,
                            xattn_boost,
                            renorm_mode=xattn_renorm,
                            frac=xattn_renorm_frac,
                        )
                    with torch.no_grad():
                        tile_pred = anima(
                            tile_latent,
                            t_expand,
                            embed,
                            padding_mask=tile_padding_mask,
                            h_offset=h_off,
                            w_offset=w_off,
                        )
                    if _boost_step:
                        # uncond runs at identity
                        set_xattn_boost_state(anima, 1.0)
                    noise_acc[:, :, :, y : y + tile_h, x : x + tile_w] += tile_pred * bw
                    weight_acc[:, :, :, y : y + tile_h, x : x + tile_w] += bw

                    if do_cfg:
                        if anima.blocks_to_swap:
                            anima.prepare_block_swap_before_forward()
                        set_hydra_crossattn(anima, negative_embed)
                        if soft_tokens_net is not None:
                            soft_tokens_net.append_postfix(
                                negative_embed,
                                soft_tokens_neg_seqlens,
                                timesteps=t_expand,
                            )
                        with torch.no_grad():
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
                    if smc_cfg is not None:
                        noise_pred = smc_cfg.combine(
                            noise_pred, uncond_noise_pred, args.guidance_scale
                        )
                    else:
                        noise_pred = uncond_noise_pred + args.guidance_scale * (
                            noise_pred - uncond_noise_pred
                        )

                if traj_stats is not None:
                    traj_stats.record(
                        i,
                        sigmas[i],  # tensor, NOT float() — float() = stream sync
                        latents,
                        noise_pred,
                        uncond_noise_pred if do_cfg else None,
                    )
                denoised = latents.float() - sigmas[i] * noise_pred.float()
                if er_sde is not None:
                    new_latents = er_sde.step(latents, denoised, i)
                else:
                    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

                latents = new_latents.to(latents.dtype)
                pbar.update()
    finally:
        if xattn_boost is not None:
            set_xattn_boost_state(anima, 1.0)
        clear_hydra_sigma(anima)
        clear_hydra_fei(anima)
        # P-GRAFT: restore LoRA for next generation
        if pgraft_network is not None and lora_cutoff_step is not None:
            pgraft_network.set_enabled(True)

    if traj_stats is not None:
        traj_stats.flush()
    return latents


def generate_body(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: Union[int, List[int]],
    latents: Optional[torch.Tensor] = None,
    context_alt: Optional[Dict[str, Any]] = None,
    tag_drop_sigma: Optional[float] = None,
    tag_boost_scale: Optional[float] = None,
) -> torch.Tensor:
    """Core denoising loop for Anima generation.

    Args:
        args: generation arguments (image_size, infer_steps, guidance_scale, etc).
        anima: loaded DiT model.
        context: dict with "embed" key containing text encoder outputs.
        context_null: negative-prompt embeddings dict, or None for unconditional.
        device: target device.
        seed: single seed or list of seeds (batch generation).
        latents: optional pre-created noise tensor; when given, batch size
            comes from it and ``seed`` is ignored for noise creation.
        context_alt / tag_drop_sigma / tag_boost_scale: commitment-σ probe —
            below ``tag_drop_sigma`` the conditional pass swaps from
            ``context`` to ``context_alt`` (e.g. the same caption with one tag
            dropped), or, with ``tag_boost_scale`` set, to an amplified
            ``embed_alt + scale·(embed − embed_alt)`` extrapolation — testing
            when/whether a dropped tag's contribution is localized structure
            vs. a global tone shift. No-op unless ``context_alt`` and
            ``tag_drop_sigma`` are both set.

    Returns:
        Denoised latent tensor (batch dimension preserved).
    """

    height, width = check_inputs(args)

    h_latent = height // 8
    w_latent = width // 8
    if (
        latents is None
        and getattr(args, "tiled_diffusion", False)
        and (h_latent > args.tile_size or w_latent > args.tile_size)
    ):
        return generate_body_tiled(args, anima, context, context_null, device, seed)

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

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context  # dummy for unconditional
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # optional pooled-text override for modulation guidance; downstream guards
    # on `is not None`
    _pooled_text_pos = None
    _pooled_text_neg = None

    # soft tokens: monkey-patches built once; per-step append_postfix() fires
    # in the loop below (and is mirrored in the Spectrum runner)
    soft_tokens_net = _setup_soft_tokens(args, anima, device)
    soft_tokens_embed_seqlens = (
        _seqlens_from_context(context, device) if soft_tokens_net is not None else None
    )
    soft_tokens_neg_seqlens = (
        _seqlens_from_context(context_null, device)
        if soft_tokens_net is not None
        else None
    )

    padding_mask = torch.zeros(
        bs, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device
    )

    logger.info(
        f"Embed: {embed.shape}, negative_embed: {negative_embed.shape}, latents: {latents.shape}"
    )

    if embed.shape[0] < bs:
        embed = embed.expand(bs, -1, -1)
    if negative_embed.shape[0] < bs:
        negative_embed = negative_embed.expand(bs, -1, -1)

    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    # commitment-σ probe alt embedding (see docstring). Text encoder outputs
    # are max-padded, so this shares embed's sequence length (padding-sink
    # invariant — trimming would produce black images).
    embed_alt = None
    embed_boost = None
    if context_alt is not None and tag_drop_sigma is not None:
        embed_alt = context_alt["embed"][0].to(device, dtype=torch.bfloat16)
        if embed_alt.shape[0] < bs:
            embed_alt = embed_alt.expand(bs, -1, -1)
        embed_alt = embed_alt.to(torch.bfloat16)
        if tag_boost_scale is not None:
            # amplify the tag's embedding delta (embed - embed_alt) in fp32
            # before casting back, to minimize rounding at scale>1
            embed_boost = (
                embed_alt.float()
                + float(tag_boost_scale) * (embed.float() - embed_alt.float())
            ).to(torch.bfloat16)

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps,
        args.flow_shift,
        device,
        tail_power=getattr(args, "sigma_tail_power", 1.0),
    )
    timesteps = timesteps.to(device, dtype=torch.bfloat16)  # σ∈[0,1] — DiT time arg

    # `er_sde` also holds an LCMSampler when selected; both share .step()
    cns = _build_cns_recolorer(args)
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(
            sigmas, seed=args.seed, device=device, cns=cns
        )
    elif args.sampler == "lcm":
        er_sde = inference_utils.LCMSampler(sigmas, seed=args.seed, device=device)

    do_cfg = args.guidance_scale != 1.0
    smc_cfg = (
        SMCCFGState(
            lam=args.smc_cfg_lambda,
            alpha=args.smc_cfg_alpha,
        )
        if do_cfg and getattr(args, "smc_cfg", False)
        else None
    )

    # FSG pre-step latent calibration. CFG-only (needs the cond/uncond gap).
    # Composes with --spectrum; ignored under --spd.
    fsg = None
    if getattr(args, "fsg", False):
        if not do_cfg:
            logger.warning("--fsg requires CFG (guidance_scale != 1.0); ignoring.")
        elif getattr(args, "spd", False):
            logger.warning(
                "--fsg is ignored under --spd (it replaces the denoise loop); "
                "FSG×SPD is a v2 item."
            )
        else:
            _band = getattr(args, "fsg_band", [0.59, 0.75])
            fsg = FSGCalibrator(
                band=(float(_band[0]), float(_band[1])),
                k=getattr(args, "fsg_k", 3),
                d_sigma=getattr(args, "fsg_d_sigma", 0.1),
                gamma=getattr(args, "fsg_gamma", None),
            )

    # CFG++ substrate (paper App A.2): a σ-scheduled guidance REWEIGHT, so it's
    # a pure change to the cond/uncond combine and works unchanged under Euler
    # and er_sde/lcm, and (via the side-channel) under --spectrum. Mutually
    # exclusive with --smc_cfg (alternative combine) and unsupported under
    # --spd (mid-loop σ re-spacing).
    cfgpp_lambda = None
    if getattr(args, "cfgpp", False):
        if not do_cfg:
            logger.warning("--cfgpp requires CFG (guidance_scale != 1.0); ignoring.")
        elif smc_cfg is not None:
            logger.warning(
                "--cfgpp and --smc_cfg both replace the cond/uncond combine; "
                "ignoring --cfgpp."
            )
        elif getattr(args, "spd", False):
            logger.warning(
                "--cfgpp is ignored under --spd (mid-loop σ re-spacing; CFG++ is "
                "not wired into the SPD runner)."
            )
        else:
            cfgpp_lambda = float(getattr(args, "cfgpp_lambda", 1.5))
            _sampler_name = "er_sde" if er_sde is not None else "euler"
            _loop = "spectrum" if getattr(args, "spectrum", False) else _sampler_name
            logger.info(
                f"CFG++ substrate active (λ={cfgpp_lambda}, sampler={_sampler_name}"
                f", loop={_loop})."
            )

    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    # passive per-step trajectory recorder; threaded through the side channels
    # so the spectrum runner records too, flushed once before return
    traj_stats = TrajStatsRecorder.from_args(args, seed=seed)

    # handed to whichever loop runner is active (spectrum / spd); the standard
    # inline loop below reads the locals directly instead
    _side_channels = SamplerSideChannels.from_args(
        args,
        pgraft_network=pgraft_network,
        lora_cutoff_step=lora_cutoff_step,
        pooled_text_pos=_pooled_text_pos,
        pooled_text_neg=_pooled_text_neg,
        smc_cfg=smc_cfg,
        fsg=fsg,
        cfgpp_lambda=cfgpp_lambda,
        soft_tokens_net=soft_tokens_net,
        soft_tokens_embed_seqlens=soft_tokens_embed_seqlens,
        soft_tokens_neg_seqlens=soft_tokens_neg_seqlens,
        traj_stats=traj_stats,
    )

    _fovea_sigma_c = float(getattr(args, "fovea_sigma_c", 0.0) or 0.0)
    if _fovea_sigma_c > 0.0:
        if getattr(args, "spd", False) or getattr(args, "spectrum", False):
            raise ValueError(
                "--fovea_sigma_c is mutually exclusive with --spd / --spectrum "
                "(all replace the denoise loop). The foveated-Spectrum compose "
                "was closed by foveated P3 (_archive/bench/foveated/) — do not "
                "re-propose."
            )
        if _FOVEATED_RUNNER is None:
            raise RuntimeError(
                "--fovea_sigma_c was passed but no foveated runner is "
                "registered. Import networks.foveated before calling generate()."
            )
        latents = _FOVEATED_RUNNER(
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
            _side_channels,
            sigma_c=_fovea_sigma_c,
            fovea_frac=getattr(args, "fovea_frac", 0.35),
            mask_source=getattr(args, "fovea_mask_source", "combo"),
        )
    elif getattr(args, "spd", False):
        if getattr(args, "spectrum", False):
            raise ValueError(
                "--spd and --spectrum are mutually exclusive (both replace the "
                "denoise loop). Compose them via a future SPD∘Spectrum runner, "
                "not by passing both."
            )
        if _SPD_RUNNER is None:
            raise RuntimeError(
                "--spd was passed but no SPD runner is registered. "
                "Import networks.spd before calling generate()."
            )

        stages, transition_sigmas = _resolve_spd_schedule(args)
        latents = _SPD_RUNNER(
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
            _side_channels,
            stages=stages,
            transition_sigmas=transition_sigmas,
            seed=seed if isinstance(seed, int) else seed[0],
        )
    elif getattr(args, "spectrum", False):
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
            _side_channels,
            window_size=getattr(args, "spectrum_window_size", 2.0),
            flex_window=getattr(args, "spectrum_flex_window", 0.25),
            warmup_steps=getattr(args, "spectrum_warmup", 6),
            w=getattr(args, "spectrum_w", 0.3),
            m=getattr(args, "spectrum_m", 3),
            lam=getattr(args, "spectrum_lam", 0.1),
            stop_caching_step=getattr(args, "spectrum_stop_caching_step", -1),
            calibration_strength=getattr(args, "spectrum_calibration", 0.0),
            schedule=getattr(args, "spectrum_schedule", "window"),
            delta=_parse_spectrum_delta(getattr(args, "spectrum_delta", "auto")),
            refresh_ratio=getattr(args, "spectrum_refresh_ratio", -1.0),
        )
    else:
        cfg_delta_probe = CfgDeltaProbe.maybe_create()
        # cross-attn residual gain on the cond forward for σ ≥ band; reset
        # right after so uncond (and FSG's internal forwards) run at identity —
        # the finally below guarantees no gain leaks into later generations
        xattn_boost = _side_channels.xattn_boost
        xattn_boost_band = _side_channels.xattn_boost_band
        xattn_renorm = _side_channels.xattn_boost_renorm
        xattn_renorm_frac = _side_channels.xattn_boost_renorm_frac
        if xattn_boost is not None:
            _renorm_desc = (
                "raw"
                if xattn_renorm == "off"
                else f"renorm={xattn_renorm} ρ={xattn_renorm_frac:g}"
            )
            logger.info(
                f"xattn boost active: λ={xattn_boost} at σ ≥ {xattn_boost_band} "
                f"({_renorm_desc})"
                + ("" if do_cfg else " (CFG 1.0 — boosting the single forward)")
            )
        # ANIMA_TEXT_KNOCKOUT_SIGMA=<cutoff>: below it, continue unconditionally
        # (noise_pred = uncond) to measure how much late-σ text still steers
        # the output. Off (None) => normal generation.
        _knockout_env = os.environ.get("ANIMA_TEXT_KNOCKOUT_SIGMA")
        text_knockout_sigma = float(_knockout_env) if _knockout_env else None
        # ANIMA_FREEZE_GUIDANCE_SIGMA=<cutoff>: below it, keep the per-step
        # guidance magnitude but lock its direction to the unit delta captured
        # at the first sub-cutoff step — tests whether late cross-attn
        # re-tweaking (not its presence) drives the tag-region effect.
        _freeze_env = os.environ.get("ANIMA_FREEZE_GUIDANCE_SIGMA")
        freeze_guidance_sigma = float(_freeze_env) if _freeze_env else None
        _frozen_guidance_dir = None  # unit delta, captured once below the cutoff
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
                    # mutates `latents` before the real forward; hydra/FEI
                    # setters below recompute on the calibrated latent
                    if fsg is not None and fsg.scheduled(float(sigmas[i])):
                        latents = fsg.calibrate(
                            anima,
                            latents,
                            float(sigmas[i]),
                            i,
                            embed,
                            negative_embed,
                            padding_mask,
                            args.guidance_scale,
                            pooled_pos=_pooled_text_pos,
                            pooled_neg=_pooled_text_neg,
                        )
                    set_hydra_sigma(anima, t_expand)
                    set_step_expert_index(anima, i)
                    compute_and_set_hydra_fei(anima, latents)

                    # below cutoff: alternate (or boosted) embedding; else embed
                    if embed_alt is not None and float(sigmas[i]) < tag_drop_sigma:
                        cond_embed = (
                            embed_boost if embed_boost is not None else embed_alt
                        )
                    else:
                        cond_embed = embed
                    set_hydra_crossattn(anima, cond_embed)
                    if soft_tokens_net is not None:
                        soft_tokens_net.append_postfix(
                            cond_embed, soft_tokens_embed_seqlens, timesteps=t_expand
                        )
                    _boost_step = (
                        xattn_boost is not None and float(sigmas[i]) >= xattn_boost_band
                    )
                    if _boost_step:
                        set_xattn_boost_state(
                            anima,
                            xattn_boost,
                            renorm_mode=xattn_renorm,
                            frac=xattn_renorm_frac,
                        )
                    with torch.no_grad():
                        _pos_kw = (
                            {"pooled_text_override": _pooled_text_pos}
                            if _pooled_text_pos is not None
                            else {}
                        )
                        noise_pred = anima(
                            latents,
                            t_expand,
                            cond_embed,
                            padding_mask=padding_mask,
                            **_pos_kw,
                        )
                    if _boost_step:
                        # uncond runs at identity
                        set_xattn_boost_state(anima, 1.0)

                    if do_cfg:
                        set_hydra_crossattn(anima, negative_embed)
                        if soft_tokens_net is not None:
                            soft_tokens_net.append_postfix(
                                negative_embed,
                                soft_tokens_neg_seqlens,
                                timesteps=t_expand,
                            )
                        with torch.no_grad():
                            _neg_kw = (
                                {"pooled_text_override": _pooled_text_neg}
                                if _pooled_text_neg is not None
                                else {}
                            )
                            uncond_noise_pred = anima(
                                latents,
                                t_expand,
                                negative_embed,
                                padding_mask=padding_mask,
                                **_neg_kw,
                            )
                        if cfg_delta_probe is not None:
                            # log text-driven velocity before the combine
                            # overwrites noise_pred (still conditional here)
                            cfg_delta_probe.record(
                                i, float(sigmas[i]), noise_pred, uncond_noise_pred
                            )
                        if cfgpp_lambda is not None:
                            # noise_pred = v^u + w_eff·(v^c − v^u); see
                            # sampling.cfgpp_guidance_weight
                            w_eff = inference_utils.cfgpp_guidance_weight(
                                float(sigmas[i]), float(sigmas[i + 1]), cfgpp_lambda
                            )
                            noise_pred = uncond_noise_pred + w_eff * (
                                noise_pred - uncond_noise_pred
                            )
                        elif smc_cfg is not None:
                            noise_pred = smc_cfg.combine(
                                noise_pred, uncond_noise_pred, args.guidance_scale
                            )
                        else:
                            delta = noise_pred - uncond_noise_pred
                            if (
                                freeze_guidance_sigma is not None
                                and float(sigmas[i]) < freeze_guidance_sigma
                            ):
                                # keep magnitude, lock direction; per-sample L2
                                # over all non-batch dims (5D latent)
                                flat = delta.float().reshape(delta.shape[0], -1)
                                mag = flat.norm(dim=1).clamp_min(1e-12)
                                unit = (flat / mag[:, None]).reshape(delta.shape)
                                if _frozen_guidance_dir is None:
                                    # capture once, at the first sub-cutoff step
                                    _frozen_guidance_dir = unit
                                mag_shape = (delta.shape[0],) + (1,) * (delta.dim() - 1)
                                delta = _frozen_guidance_dir.to(
                                    delta.dtype
                                ) * mag.reshape(mag_shape).to(delta.dtype)
                            noise_pred = uncond_noise_pred + args.guidance_scale * delta

                        if (
                            text_knockout_sigma is not None
                            and float(sigmas[i]) < text_knockout_sigma
                        ):
                            noise_pred = uncond_noise_pred

                    if traj_stats is not None:
                        traj_stats.record(
                            i,
                            sigmas[i],  # tensor, NOT float() — float() = stream sync
                            latents,
                            noise_pred,
                            uncond_noise_pred if do_cfg else None,
                        )
                    denoised = latents.float() - sigmas[i] * noise_pred.float()
                    if er_sde is not None:
                        new_latents = er_sde.step(latents, denoised, i)
                    else:
                        new_latents = inference_utils.step(
                            latents, noise_pred, sigmas, i
                        )

                    latents = new_latents.to(latents.dtype)

                    pbar.update()
        finally:
            if cfg_delta_probe is not None:
                cfg_delta_probe.flush()
            if xattn_boost is not None:
                set_xattn_boost_state(anima, 1.0)
            clear_hydra_sigma(anima)
            clear_hydra_fei(anima)
            # P-GRAFT: restore LoRA for next generation
            if pgraft_network is not None and lora_cutoff_step is not None:
                pgraft_network.set_enabled(True)

    # single flush point for every branch (runners record but never flush);
    # idempotent, so flushing early is harmless
    if traj_stats is not None:
        traj_stats.flush()
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
    device = gen_settings.device
    seed = resolve_seed(args)  # doesn't mutate args — see resolve_seed docstring

    if shared_models is None or "model" not in shared_models:
        anima = load_dit_model(args, device, torch.bfloat16)

        if shared_models is not None:
            shared_models["model"] = anima
    else:
        logger.info("Using shared DiT model.")
        anima: anima_models.Anima = shared_models["model"]

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]

    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, anima, shared_models)

    # Modulation guidance: compute guidance delta once
    if (
        getattr(args, "pooled_text_proj", None) is not None
        and getattr(args, "mod_w", 0.0) != 0.0
    ):
        setup_mod_guidance(args, anima, device, shared_models)
    else:
        anima.reset_mod_guidance()

    _setup_easycontrol(args, anima, device, shared_models)

    # DAVE: per-block DC-attenuation hooks (training-free diversity edit). Must
    # be removed after generation — the model is shared across seeds, so a
    # stacked hook set would compound the attenuation. See dave.py.
    dave_hooks = None
    if getattr(args, "dave", None):
        from library.inference.corrections.dave import setup_dave

        dave_hooks = setup_dave(args, anima, device)
    else:
        anima.reset_dave()

    try:
        return generate_body(args, anima, context, context_null, device, seed)
    finally:
        if dave_hooks is not None:
            dave_hooks.remove()
            anima.reset_dave()


def _setup_easycontrol(args, anima, device, shared_models):
    """Load EasyControl weights, VAE-encode the reference image, prime cond KV cache.

    The cond stream is deterministic across denoising steps (frozen DiT +
    frozen LoRA, no dependence on the noisy target), so it's run once via
    ``network.precompute_cond_kv()`` and the (K_c, V_c) tensors are reused
    for every step and CFG branch.
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
        from library.datasets.buckets import (
            choose_edge,
            freefit_band_for_edge,
            freefit_bucket,
        )

        with Image.open(ec_image) as _ref_for_size:
            _rw, _rh = _ref_for_size.size
        # free-fit the reference's native aspect into the canonical 1024 tier band
        _edge = choose_edge(_rw, _rh, [1024])
        _bw, _bh = freefit_bucket(_rw, _rh, freefit_band_for_edge(_edge))
        args.image_size = [_bh, _bw]
        logger.info(
            f"EasyControl: image_size auto-picked from ref (aspect w/h={_rw / _rh:.3f}) "
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
    # b_cond is a live logit bias, not baked into the KV cache — must offset
    # before apply_to/precompute so every block's closure captures the shifted
    # Parameter. Mirrors scripts/edit.py.
    b_offset = getattr(args, "easycontrol_b_offset", None)
    if b_offset is not None:
        with torch.no_grad():
            for b in network.b_cond:
                b += float(b_offset)
        logger.info("EasyControl: b_cond offset %+g applied to every block", b_offset)
    network.to(device, dtype=torch.bfloat16)
    network.apply_to(text_encoders=None, unet=anima)

    # resize to args.image_size first so the cond bucket matches the target
    h_pix, w_pix = args.image_size
    img = Image.open(ec_image).convert("RGB").resize((w_pix, h_pix), Image.LANCZOS)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    img_t = (
        tfm(img).unsqueeze(0).to(device, dtype=torch.bfloat16)
    )  # [1,3,H,W] in [-1,1]

    vae = (shared_models or {}).get("vae")
    vae_was_shared = vae is not None
    if vae is None:
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
            vae_2d=getattr(args, "vae_2d", True),
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
    network.precompute_cond_kv()
    logger.info(
        f"EasyControl: loaded {ec_weight} "
        f"(r={network.cond_lora_dim}, scale={network.get_effective_scale():.3f}, kv-cached)"
    )
    anima._easycontrol_network = network
    return network
