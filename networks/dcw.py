"""DCW: post-step correction for SNR-t bias on flow-matching DiTs.

Anima form (pixel mode, opposite sign from paper, λ ≈ -0.015 for LL-only):
    prev += λ · sched(σ_i) · (prev - x0_pred)

where sched ∈ {one_minus_sigma, sigma_i, const, none}. Default
schedule one_minus_sigma matches Anima's bias envelope (concentrates
correction at low σ where |gap| is largest). See docs/methods/dcw.md.

The correction can be restricted to a subset of single-level Haar
subbands ({LL, LH, HL, HH}) via the ``bands`` argument. LL-only is the
shipped default — the per-band sweep at
``bench/dcw/results/20260503-2102-band-mask-eyeball/`` showed the
broadband correction degrades detail bands while LL-only improves all
four bands. ``bands == ALL_BANDS`` falls through to the cheap
no-DWT path (orthonormal Haar reconstructs exactly to float roundoff).

Paper: Yu et al., "Elucidating the SNR-t Bias of Diffusion Probabilistic
Models" (CVPR 2026, arXiv:2604.16044).
"""

from typing import Literal

import torch

Schedule = Literal["one_minus_sigma", "sigma_i", "const", "none"]

BANDS = ("LL", "LH", "HL", "HH")
ALL_BANDS = frozenset(BANDS)


def parse_band_mask(label: str) -> frozenset[str]:
    """CLI string → frozenset of band names. ``all`` → all four bands.

    Format: ``LL``, ``HH``, ``LH+HL+HH``, ``all``. Case-insensitive on
    the band names; ``all`` must be exactly that token.
    """
    if label == "all":
        return ALL_BANDS
    parts = [p.upper() for p in label.split("+") if p]
    bad = [p for p in parts if p not in BANDS]
    if bad or not parts:
        raise ValueError(
            f"unknown band(s) in mask {label!r}: {bad or '<empty>'}; "
            f"valid bands {BANDS!r} or 'all'"
        )
    return frozenset(parts)


def _haar_dwt_2d(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2D orthonormal Haar DWT on the last two dims.

    Returns (LL, LH, HL, HH), each (..., H/2, W/2).
    """
    a = v[..., 0::2, 0::2]
    b = v[..., 0::2, 1::2]
    c = v[..., 1::2, 0::2]
    d = v[..., 1::2, 1::2]
    s = 0.5
    LL = (a + b + c + d) * s
    LH = (a + b - c - d) * s
    HL = (a - b + c - d) * s
    HH = (a - b - c + d) * s
    return LL, LH, HL, HH


def _haar_idwt_2d(
    LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor
) -> torch.Tensor:
    s = 0.5
    a = (LL + LH + HL + HH) * s
    b = (LL + LH - HL - HH) * s
    c = (LL - LH + HL - HH) * s
    d = (LL - LH - HL + HH) * s
    out = torch.empty(
        *LL.shape[:-2], LL.shape[-2] * 2, LL.shape[-1] * 2,
        dtype=LL.dtype, device=LL.device,
    )
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def _sched(sigma_i: float, schedule: Schedule) -> float:
    if schedule == "one_minus_sigma":
        return 1.0 - sigma_i
    if schedule == "sigma_i":
        return sigma_i
    if schedule == "const":
        return 1.0
    return 0.0  # "none" — for ablation


def apply_dcw(
    prev_sample: torch.Tensor,
    x0_pred: torch.Tensor,
    sigma_i: float,
    *,
    lam: float = -0.015,
    schedule: Schedule = "one_minus_sigma",
    bands: frozenset[str] = ALL_BANDS,
) -> torch.Tensor:
    """Apply pixel-mode DCW correction to prev_sample, optionally restricted
    to a subset of Haar subbands.

    Returns prev_sample unchanged if lam == 0 or schedule == "none".
    Callers should also skip the final step (sigma_{i+1} == 0) where
    prev_sample == x0_pred exactly and the (1-σ_i) weight is near 1.

    When ``bands == ALL_BANDS`` (default), the full pixel-mode update
    runs without any DWT round-trip — bit-identical to the pre-band-sweep
    behaviour. Otherwise the differential ``(prev - x0_pred)`` is
    decomposed via single-level Haar DWT, non-mask bands are zeroed,
    reconstructed, and added. Latent space is 16-ch Qwen-VAE; the
    "low/high freq" labels are *latent*, not pixel-space.
    """
    s = lam * _sched(sigma_i, schedule)
    if s == 0.0:
        return prev_sample
    if bands == ALL_BANDS:
        return prev_sample + s * (prev_sample - x0_pred)
    diff = (prev_sample - x0_pred).float()
    LL, LH, HL, HH = _haar_dwt_2d(diff)
    z = torch.zeros_like(LL)
    LL_m = LL if "LL" in bands else z
    LH_m = LH if "LH" in bands else z
    HL_m = HL if "HL" in bands else z
    HH_m = HH if "HH" in bands else z
    masked = _haar_idwt_2d(LL_m, LH_m, HL_m, HH_m).to(prev_sample.dtype)
    return prev_sample + s * masked
