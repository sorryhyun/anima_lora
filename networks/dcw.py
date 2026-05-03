"""DCW: post-step correction for SNR-t bias on flow-matching DiTs.

Anima form (pixel mode, opposite sign from paper, λ ≈ -0.010):
    prev += λ · sched(σ_i) · (prev - x0_pred)

where sched ∈ {one_minus_sigma, sigma_i, const, none}. Default
schedule one_minus_sigma matches Anima's bias envelope (concentrates
correction at low σ where |gap| is largest). See bench/dcw/findings.md.

Paper: Yu et al., "Elucidating the SNR-t Bias of Diffusion Probabilistic
Models" (CVPR 2026, arXiv:2604.16044).
"""

from typing import Literal

import torch

Schedule = Literal["one_minus_sigma", "sigma_i", "const", "none"]


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
    lam: float = -0.010,
    schedule: Schedule = "one_minus_sigma",
) -> torch.Tensor:
    """Apply pixel-mode DCW correction to prev_sample.

    Returns prev_sample unchanged if lam == 0 or schedule == "none".
    Callers should also skip the final step (sigma_{i+1} == 0) where
    prev_sample == x0_pred exactly and the (1-σ_i) weight is near 1.
    """
    s = lam * _sched(sigma_i, schedule)
    if s == 0.0:
        return prev_sample
    return prev_sample + s * (prev_sample - x0_pred)
