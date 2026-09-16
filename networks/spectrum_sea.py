"""Spectral-Evolution-Aware (SEA) filter — SeaCache's cache-decision metric.

Library home for the SEA filter that `networks/spectrum.py` uses as an
alternative *when-to-skip* trigger (``schedule="sea"``), replacing Spectrum's
content-blind growing window. The Chebyshev feature **forecasting** (the reuse
path) is unchanged — only the decision metric is swapped, which keeps the
per-step ``noise_pred`` reconstruction (and therefore SMC-CFG / mod-guidance
composition) intact. See ``docs/inference/spectrum.md`` §"SEA schedule" and
the validation in ``docs/findings/seacache_sea_decision_metric.md``.

Reconstructs the timestep-dependent filter from SeaCache (Chung et al.,
"SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models",
arXiv:2602.18993v2), §4.1. The filter downweights the high-frequency *noise*
component and preserves the low-frequency *content* component, so a cache
distance measured on filtered features tracks content evolution rather than
stochastic detail.

Math (paper Eq. 5 / 7):

    G_t(f)      = a_t · S_x(f) / (a_t² · S_x(f) + b_t²)          (Wiener-like)
    G_t^norm(f) = ν_t · G_t(f),   ν_t = (mean_f G_t(f))^{-1}      (unit-mean gain)
    P_t(I)      = iFFT( G_t^norm(f) ⊙ FFT(I) )                    (Eq. 6)

with ``f`` the radial spatial frequency, ``S_x(f) ∝ f^{-β}`` a natural-image
power-law spectrum (β≈2, paper refs [7,18,61,62]), and ``(a_t, b_t)`` the
forward-noising signal / noise coefficients. For the rectified-flow schedule
Anima uses, ``x_t = (1−σ)·x0 + σ·ε``, so ``a_t = 1−σ`` and ``b_t = σ``.

Sanity:
  * σ→0 (clean): a=1,b=0 → G=1 everywhere → P is identity (passes all detail).
  * σ→1 (pure noise): a=0 → G=0 → P→0 (no signal to keep).
  * mid-σ: low frequencies pass, high frequencies attenuated (paper Fig. 4).

Applied per channel over the spatial ``(H, W)`` axes; any leading batch/channel
dims broadcast against the shared ``(H, W)`` gain.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch

_EPS = 1e-8


def radial_freq(h: int, w: int, device, dtype) -> torch.Tensor:
    """Radial spatial-frequency magnitude grid for ``fft2`` over ``(h, w)``.

    Returns an ``(h, w)`` tensor in cycles/pixel; DC (f=0) sits at index [0, 0]
    in fft layout. The Nyquist-normalized magnitude is ``sqrt(fy² + fx²)`` with
    ``fy, fx`` from :func:`torch.fft.fftfreq` (range ``[-0.5, 0.5)``).
    """
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(gy * gy + gx * gx)


def sea_gain(h: int, w: int, sigma: float, beta: float, device, dtype) -> torch.Tensor:
    """Density-normalized SEA gain ``G_t^norm(f)`` as an ``(h, w)`` mask.

    ``sigma`` is the flow-matching noise fraction (``b_t``); ``a_t = 1 − sigma``.
    DC frequency is floored at ``1/max(h,w)`` so ``S_x = f^{-β}`` stays finite.
    """
    a = 1.0 - float(sigma)
    b = float(sigma)
    f = radial_freq(h, w, device, dtype)
    f_floor = 1.0 / max(h, w)
    sx = f.clamp_min(f_floor) ** (-beta)  # natural-image power law
    g = a * sx / (a * a * sx + b * b + _EPS)  # Wiener-like response
    g = g / g.mean().clamp_min(_EPS)  # unit-mean gain over freq bins (Eq. 7)
    return g


def sea_filter(x: torch.Tensor, sigma: float, beta: float = 2.0) -> torch.Tensor:
    """Apply the SEA filter ``P_t`` over the spatial (last two) axes of ``x``.

    Accepts ``(C, H, W)`` (the bench shape), ``(B, C, H, W)`` (a 4D latent), or
    any ``(..., H, W)`` tensor: FFT over the last two dims, multiply by the
    σ-dependent radial gain (shared across all leading dims), inverse FFT,
    return the real part (same shape/dtype).
    """
    assert x.dim() >= 2, f"expected (..., H, W), got {tuple(x.shape)}"
    h, w = x.shape[-2], x.shape[-1]
    g = sea_gain(h, w, sigma, beta, x.device, torch.float32)
    xf = torch.fft.fft2(x.to(torch.float32), dim=(-2, -1))
    yf = xf * g  # broadcast (H,W) over all leading dims
    y = torch.fft.ifft2(yf, dim=(-2, -1)).real
    return y.to(x.dtype)


def l1rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L1 distance ``‖a − b‖₁ / (‖b‖₁ + ξ)`` — SeaCache Eq. 3."""
    return float((a - b).abs().sum() / (b.abs().sum() + _EPS))


# ---------------------------------------------------------------------------
# Auto-δ calibration (refresh-ratio matching)
#
# The SEA trigger refreshes when the accumulated SEA distance since the last
# refresh crosses δ. δ is the latency/quality dial (SeaCache Eq. 4/8) and wants
# per-config calibration. ``solve_delta_for_refresh_ratio`` binary-searches the
# δ whose accumulate-reset rule, replayed over a recorded distance trace, hits a
# target fraction of refreshes — making the SEA arm a like-for-like swap at
# matched compute (docs/inference/spectrum.md §"SEA schedule").
# ---------------------------------------------------------------------------


def window_decision_fraction(
    num_steps: int,
    warmup_steps: int,
    stop_at: int,
    window_size: float,
    flex_window: float,
    forced_steps: frozenset = frozenset(),
) -> float:
    """Refresh fraction the growing-window schedule spends in the decision region.

    Replays the exact window rule (the Spectrum loop's ``else`` branch + its
    curr_ws advance) and returns ``actual_decision_steps / decision_steps``. The
    SEA auto-δ target defaults to *this* so the SEA arm is a like-for-like swap at
    matched compute for any step count (a fixed fraction only matches one step
    count).

    ``forced_steps`` are step indices forced to an actual forward by an external
    consumer (FSG-scheduled calibration steps). They are treated exactly like
    warmup/tail — forced actual, excluded from the decision denominator — so the
    fraction (and therefore the SEA δ target) is matched against the window
    baseline *over the same adaptive budget*; FSG's fixed forward cost lands
    identically on both arms. Default empty reproduces the plain window schedule.
    """
    curr_ws = window_size
    consec = 0
    actual_dec = 0
    n_dec = 0
    for i in range(num_steps):
        if i < warmup_steps or i >= stop_at or i in forced_steps:
            actual = True
        else:
            actual = (consec + 1) % max(1, math.floor(curr_ws)) == 0
            n_dec += 1
            actual_dec += int(actual)
        if actual:
            if i >= warmup_steps and i not in forced_steps:
                curr_ws = round(curr_ws + flex_window, 3)
            consec = 0
        else:
            consec += 1
    return actual_dec / max(1, n_dec)


def count_refreshes(dists: Sequence[float], delta: float) -> int:
    """Number of refreshes the accumulate-until-δ rule fires over ``dists``.

    Monotonically non-increasing in ``delta`` (larger δ → fewer refreshes),
    which is what makes the binary search below well-posed.
    """
    accum = 0.0
    n = 0
    for d in dists:
        accum += float(d)
        if accum >= delta:
            n += 1
            accum = 0.0
    return n


def solve_delta_for_refresh_ratio(
    dists: Sequence[float], refresh_ratio: float, iters: int = 60
) -> float:
    """δ such that ``count_refreshes(dists, δ)`` ≈ ``round(refresh_ratio·len)``.

    ``dists`` is the per-step SEA distance trace over the *decision-eligible*
    steps (post-warmup, pre-stop). Returns the threshold δ at which the refresh
    count transitions to ≤ the target; clamps to a positive value.
    """
    dists = [float(d) for d in dists]
    n = len(dists)
    if n == 0:
        return _EPS
    target = max(1, round(float(refresh_ratio) * n))
    lo, hi = 0.0, max(sum(dists), _EPS)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if count_refreshes(dists, mid) > target:
            lo = mid  # too many refreshes → δ too small, raise it
        else:
            hi = mid
    return max(hi, _EPS)


def accumulate_distances(seas: Sequence[torch.Tensor]) -> List[float]:
    """Per-step consecutive SEA distances ``l1rel(sea_i, sea_{i-1})``.

    Convenience for offline calibration / tests; the live loop accumulates
    inline. The first element has no predecessor and is omitted, so the result
    has ``len(seas) - 1`` entries.
    """
    return [l1rel(seas[i], seas[i - 1]) for i in range(1, len(seas))]
