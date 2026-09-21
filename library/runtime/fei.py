"""Frequency-Energy Index (FEI) on latents.

FEI = soft band-energy on the simplex, computed by stacked Gaussian
low-pass on a 4D `[B, C, H, W]` latent. Used as the routing key for
HydraLoRA-FEI (see ``networks/lora_modules/hydra.py`` and
``docs/methods/hydra-lora.md``).

**2 bands**, not the FeRA paper's 3: on flow-matching latents the mid band
carried <8% of the energy at every σ_mid tried
(``_archive/bench/fera/probe_fei.py``).

Bucket-invariant scaling: ``σ_low = min(H_lat, W_lat) / fei_sigma_low_div``.
Default **4.0** maximised population std(e_low) on real training latents
(``_archive/bench/fera/probe_fei_dataset.py``); 8.0 remains a Pareto choice
(mean |Δ FEI| < 0.02 between mirror buckets).

Both training (`train.py`) and inference (`library/inference/generation.py`)
call ``compute_fei_2band`` once per step on the current `z_t`, then
push it into every HydraLoRA module via
``library/inference/adapters.py::set_hydra_fei``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# Module-level cache so a long training run doesn't
# re-emit `arange + exp` once per call. Keyed on (sigma, device, dtype).
_GAUSS_CACHE: dict[tuple[float, torch.device, torch.dtype], torch.Tensor] = {}


def _gaussian_kernel_1d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (round(sigma, 4), device, dtype)
    cached = _GAUSS_CACHE.get(key)
    if cached is not None:
        return cached
    half = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    _GAUSS_CACHE[key] = k
    return k


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian along (H, W) with reflect padding. ``x`` is ``[B, C, H, W]``."""
    if sigma <= 0:
        return x
    k1 = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    K = k1.numel()
    pad = K // 2
    C = x.shape[1]
    kw = k1.view(1, 1, 1, K).expand(C, 1, 1, K).contiguous()
    kh = k1.view(1, 1, K, 1).expand(C, 1, K, 1).contiguous()
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kw, groups=C)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kh, groups=C)
    return x


def compute_fei_2band(z: torch.Tensor, sigma_low: float) -> torch.Tensor:
    """Return ``[B, 2]`` simplex ``(e_low, e_high)`` on latent ``z``.

    Two-band Laplacian decomposition::

        b_low  = LP(z, σ_low)
        b_high = z − LP(z, σ_low)
        e_k    = ||b_k||²  (sum over C, H, W)
        e      = e_k / Σ e

    `z` is ``[B, C, H, W]`` (call ``.squeeze(2)`` on a `(B, C, T, H, W)`
    Anima latent first). Result is float32 on the input's device — caller
    casts when stuffing into a router buffer.

    We promote `z` to fp32 internally to avoid silent precision loss on
    bf16 latents (the squared norm + division can underflow at small
    `e_low`). The convolution itself runs in fp32; for the typical
    `H_lat·W_lat ≈ 4096` patch grid this is negligible vs the DiT forward.
    """
    z = z.float()
    lp = gaussian_blur_2d(z, sigma_low)
    e_low = lp.pow(2).flatten(1).sum(-1)
    e_high = (z - lp).pow(2).flatten(1).sum(-1)
    energies = torch.stack([e_low, e_high], dim=-1)
    return energies / energies.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def fei_sigma_low(h_lat: int, w_lat: int, fei_sigma_low_div: float) -> float:
    """``σ_low = min(H_lat, W_lat) / fei_sigma_low_div``.

    Bucket-adaptive — keeps the band semantic across aspect ratios with
    no per-bucket router head. Default ``4.0`` is set on ``LoRANetworkCfg``.
    """
    return float(min(h_lat, w_lat)) / float(fei_sigma_low_div)
