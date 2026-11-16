"""Pure-compute kernels for the HydraLoRA / FeRA / σ-band router at inference.

The training tree, the inference loop and the ComfyUI adapter node
(``ComfyUI-Anima_lora-Adapter``, vendored by ``scripts/release/sync_vendor.py``)
all import these. Router weights are trained against this exact functional
form: a permuted band ordering or a different sinusoidal frequency schedule
silently corrupts samples.

Public surface:

* ``gaussian_blur_2d`` — separable 2D Gaussian on ``[B, C, H, W]``.
* ``compute_fei_2band(z, sigma_low) -> (B, 2)`` simplex
  ``[e_low, e_high]`` (``stacked_experts_global_fei``).
* ``compute_fei_nband_high_to_low(z, sigma_low, num_bands) -> (B, N)``
  simplex ordered ``[high, ..., low]`` (author-faithful FeRA). The ordering
  differs from ``compute_fei_2band``; do not unify the two.
* ``fei_sigma_low(h_lat, w_lat, div) -> float`` — bucket-invariant σ_low
  scaling (``min(H_lat, W_lat) / div``).
* ``sigma_sinusoidal_features(sigma, dim)`` — sinusoidal σ features matching
  the DiT t_embedder functional form.
* ``apply_sigma_band_mask`` — mask out-of-band expert logits to -inf for
  the σ-band partition routing.
* ``fei_temperature(fei, tau) -> simplex`` — ChimeraHydra hardwired-FEI freq
  gate (``freq_router_mode="fei"``).
"""

from __future__ import annotations

from typing import List

import torch

from library.runtime.fei import (
    compute_fei_2band,
    fei_sigma_low,
    gaussian_blur_2d,
)

# The σ math lives beside router_state's buffer helpers; re-exported publicly here.
from networks.lora_modules.router_state import (
    _apply_sigma_band_mask as apply_sigma_band_mask,
    _sigma_sinusoidal_features as sigma_sinusoidal_features,
)


def fei_temperature(fei: torch.Tensor, tau: float) -> torch.Tensor:
    """Hardwired-FEI freq gate: ``π_f = normalize(FEI ** (1/τ))``.

    Serves ChimeraHydra ``freq_router_mode="fei"`` checkpoints in the ComfyUI
    node; τ=1.0 returns FEI unchanged. The power form avoids ``log(0)`` on
    high-σ steps where ``e_low ≈ 0``.
    """
    if abs(tau - 1.0) < 1e-8:
        return fei
    p = fei.clamp_min(0).pow(1.0 / max(tau, 1e-6))
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def compute_fei_nband_high_to_low(
    z: torch.Tensor, sigma_low: float, num_bands: int
) -> torch.Tensor:
    """Return ``(B, num_bands)`` simplex energies, ordered ``[high, ..., low]``.

    Matches FeRA's ``FrequencyEnergyIndicator``: differences of adjacent
    pyramid levels (high-freq first), then the coarsest LP as the low band;
    σ doubles outward from ``sigma_low``. fp32 internally (squared norms
    underflow in bf16). FeRA router weights were trained against this
    ordering, so a permutation corrupts the gate.
    """
    z = z.float()
    sigmas: List[float] = [sigma_low * (2.0**k) for k in range(num_bands - 1)]
    pyramid = [z]
    for s in sigmas:
        pyramid.append(gaussian_blur_2d(pyramid[-1], s))
    bands = [pyramid[k] - pyramid[k + 1] for k in range(num_bands - 1)]
    bands.append(pyramid[-1])
    energies = torch.stack([b.pow(2).flatten(1).sum(-1) for b in bands], dim=-1)
    return energies / energies.sum(dim=-1, keepdim=True).clamp_min(1e-12)


__all__ = [
    "gaussian_blur_2d",
    "compute_fei_2band",
    "compute_fei_nband_high_to_low",
    "fei_sigma_low",
    "sigma_sinusoidal_features",
    "apply_sigma_band_mask",
    "fei_temperature",
]
