"""Spectral Progressive Diffusion (SPD) — training-free inference acceleration.

Xiao et al., arXiv:2605.18736. Grow spatial resolution along the denoising
trajectory: run early (noise-dominated) steps at low resolution, then inject
high-frequency detail via *spectral noise expansion* only when finer
frequencies emerge from noise. The latent power spectrum decays as a power law
(`P_ω ∝ |ω|^{-β}`, β=2.26 on Anima — `bench/spd/`), so HF carries far less
signal and is cheap to defer.

Inference-only. Like ``networks/spectrum.py``, a sampler-level runner that
replaces the denoise loop and self-registers with ``library.inference.generation``
at import time, so ``library.inference`` has no hard edge into ``networks/``.
Dispatched from ``generate_body`` on ``--spd``.

Scope:
  * **Euler only.** Spectral expansion re-spaces the remaining σ schedule
    mid-loop (Sec 4.3); ``ERSDESampler``/``LCMSampler`` precompute coefficients
    from the full schedule, so a stochastic sampler request falls back to Euler
    with a warning.
  * **No SMC-CFG** (unvalidated against the mid-loop reshape); warns and ignores.
  * Composes with LoRA / Hydra / soft-tokens / P-GRAFT: the per-Linear LoRA
    delta is token-count-agnostic, so it runs at any stage resolution.
"""

from __future__ import annotations

import logging
from typing import List

import torch
from tqdm import tqdm

from library.inference.adapters import (
    compute_and_set_hydra_fei,
    set_hydra_crossattn,
    set_hydra_sigma,
    set_xattn_boost_state,
)
from library.inference.sampler_context import SamplerSideChannels

# Spectral math lives in spd_core.py (vendored into the ComfyUI SPEED node).
from networks.spd_core import dct_lowpass_init, spectral_expand

log = logging.getLogger(__name__)


@torch.no_grad()
def spd_denoise(
    anima,
    latents: torch.Tensor,
    timesteps: torch.Tensor,  # unused (SPD builds its own t from the live σ); kept for runner-signature parity
    sigmas: torch.Tensor,
    embed: torch.Tensor,
    negative_embed: torch.Tensor,
    padding_mask: torch.Tensor,
    guidance_scale: float,
    sampler,  # ERSDESampler / LCMSampler / None — SPD forces Euler (see module docstring)
    device: torch.device,
    ctx: SamplerSideChannels,
    *,
    stages: List[float],
    transition_sigmas: List[float],
    seed: int = 0,
) -> torch.Tensor:
    """Multi-resolution SPD denoising loop.

    ``stages`` is ascending resolution scales (e.g. ``[0.5, 1.0]``);
    ``transition_sigmas`` (len = len(stages)-1) are the σ thresholds at which to
    spectral-expand to the next stage. ``stages=[1.0]`` + ``[]`` is the plain
    full-res baseline.

    The first stage starts from a DCT low-pass of the full-res init latent; each
    transition fills the newly representable HF slots with σ-scaled noise and
    re-spaces the remaining σ schedule (Sec 4.3). ``padding_mask`` is rebuilt at
    each stage to match the new token grid.

    ``ctx`` carries the shared conditioning side-channels
    (``library.inference.sampler_context``); SMC-CFG is ignored.
    """
    pgraft_network = ctx.pgraft_network
    lora_cutoff_step = ctx.lora_cutoff_step
    pooled_text_pos = ctx.pooled_text_pos
    pooled_text_neg = ctx.pooled_text_neg
    soft_tokens_net = ctx.soft_tokens_net
    soft_tokens_embed_seqlens = ctx.soft_tokens_embed_seqlens
    soft_tokens_neg_seqlens = ctx.soft_tokens_neg_seqlens
    # --xattn_boost: cond-forward-only cross-attn gain at σ ≥ band; the gate
    # reads the re-spaced per-step σ.
    xattn_boost = ctx.xattn_boost
    xattn_boost_band = ctx.xattn_boost_band
    xattn_renorm = ctx.xattn_boost_renorm
    xattn_renorm_frac = ctx.xattn_boost_renorm_frac

    if sampler is not None:
        log.warning(
            "--spd forces Euler; the requested stochastic sampler is ignored "
            "(spectral expansion re-spaces σ mid-loop, which precomputed "
            "ER-SDE/LCM coefficients cannot follow)."
        )
    if ctx.smc_cfg is not None:
        log.warning(
            "--spd v0 does not compose with SMC-CFG (it acts on the "
            "re-spaced σ boundary and is unvalidated against the mid-loop "
            "reshape); ignoring. See docs/anima/inference/spd.md."
        )

    do_cfg = guidance_scale != 1.0
    patch = anima.patch_spatial
    H_full, W_full = latents.shape[-2], latents.shape[-1]
    sigmas = sigmas.clone().float()
    gen = torch.Generator(device=device).manual_seed(int(seed) + 10_000)

    cur_scale = stages[0]
    x5 = latents
    if cur_scale < 1.0:
        x5 = dct_lowpass_init(x5, cur_scale, patch)
    stage_idx = 0

    def _padding_mask_for(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1], dtype=torch.bfloat16, device=device
        )

    pad = _padding_mask_for(x5)

    def velocity(
        x: torch.Tensor, sigma_scalar: float, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        # timestep == σ in [0,1] for Anima flow-matching (fed unscaled).
        t = x.new_full((x.shape[0],), float(sigma_scalar))
        set_hydra_sigma(anima, t)
        compute_and_set_hydra_fei(anima, x)
        set_hydra_crossattn(anima, embed)
        if soft_tokens_net is not None:
            soft_tokens_net.append_postfix(
                embed, soft_tokens_embed_seqlens, timesteps=t
            )
        _pos_kw = (
            {"pooled_text_override": pooled_text_pos}
            if pooled_text_pos is not None
            else {}
        )
        _boost_step = (
            xattn_boost is not None and float(sigma_scalar) >= xattn_boost_band
        )
        if _boost_step:
            set_xattn_boost_state(
                anima, xattn_boost, renorm_mode=xattn_renorm, frac=xattn_renorm_frac
            )
        try:
            v_c = anima(x, t, embed, padding_mask=pad_mask, **_pos_kw)
        finally:
            if _boost_step:
                set_xattn_boost_state(anima, 1.0)  # uncond runs at identity
        if not do_cfg:
            return v_c
        set_hydra_crossattn(anima, negative_embed)
        if soft_tokens_net is not None:
            soft_tokens_net.append_postfix(
                negative_embed, soft_tokens_neg_seqlens, timesteps=t
            )
        _neg_kw = (
            {"pooled_text_override": pooled_text_neg}
            if pooled_text_neg is not None
            else {}
        )
        v_u = anima(x, t, negative_embed, padding_mask=pad_mask, **_neg_kw)
        return v_u + guidance_scale * (v_c - v_u)

    n = len(sigmas) - 1
    with tqdm(total=n, desc=f"SPD denoising ({x5.shape[0]}x)") as pbar:
        for i in range(n):
            # P-GRAFT: disable LoRA at cutoff step (reference model takes over).
            if (
                pgraft_network is not None
                and lora_cutoff_step is not None
                and i == lora_cutoff_step
            ):
                pgraft_network.set_enabled(False)
                log.info("P-GRAFT: Disabled LoRA at step %d/%d", i, n)

            sigma = float(sigmas[i])
            # Expand through any stage whose transition σ we've crossed.
            while (
                stage_idx < len(transition_sigmas)
                and sigma <= transition_sigmas[stage_idx]
            ):
                nxt = stages[stage_idx + 1]
                if nxt > cur_scale:
                    orig = float(sigmas[i])
                    x5, sigma_new = spectral_expand(
                        x5, sigma, cur_scale, nxt, H_full, W_full, patch, gen
                    )
                    pad = _padding_mask_for(x5)
                    cur_scale = nxt
                    if orig > 0 and sigma_new != orig:  # re-space remaining σ (Sec 4.3)
                        sigmas[i + 1 :] = sigma_new * (sigmas[i + 1 :] / orig)
                    sigma = sigma_new
                stage_idx += 1

            v = velocity(x5, sigma, pad).float()
            dt = float(sigmas[i + 1]) - sigma
            x5 = (x5.float() + v * dt).to(latents.dtype)
            pbar.update(1)

    if cur_scale < 1.0:  # never handed off to full res — bicubic rescue so decode works
        import torch.nn.functional as F

        x5 = (
            F.interpolate(
                x5.squeeze(2).float(),
                size=(H_full, W_full),
                mode="bicubic",
                align_corners=False,
            )
            .unsqueeze(2)
            .to(latents.dtype)
        )
    return x5


# Side-effect registration (mirrors the register_spectrum_runner call in networks/spectrum.py).
from library.inference.generation import register_spd_runner  # noqa: E402

register_spd_runner(spd_denoise)
