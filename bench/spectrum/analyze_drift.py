#!/usr/bin/env python
"""Drift analyzer — measures how Spectrum-style prediction error on cached
steps propagates through different samplers × schedulers. No DiT required.

Motivation
----------
When Spectrum replaces an actual forward with a Chebyshev prediction, the
returned `denoised` carries a small error relative to the true denoised. How
that error evolves into a visible latent drift depends on the sampler's
update equation AND on how the denoiser responds to a perturbed x at the
next step:

  * Deterministic single-step (euler): drift accumulates ~linearly in step
    count, roughly proportional to the per-step cache error.
  * Ancestral (euler_a): each step injects fresh Gaussian noise whose variance
    depends on the (possibly wrong) denoised. The stochastic kicks don't
    cancel out in the diff between perturbed and reference runs — they
    compound as an extra random walk on top of the deterministic drift.
  * Deterministic multistep (dpmpp_2m): the update uses
    (denoised[i] - denoised[i-1]) / (λ[i] - λ[i-1]). A perturbation on one
    cached denoised enters the next step's finite difference divided by the
    λ-gap, which is tiny at low sigma → error amplified by ~1/Δλ.
  * Multistep SDE (er_sde): same finite-difference amplification plus the
    stochastic kick compounding. Worst-case without the guard.

Closed-loop denoiser
--------------------
The denoiser is a Tweedie posterior mean for a toy Gaussian data prior
``N(target, prior_var · I)``:

    denoised(x, σ) = (σ² · target + prior_var · x) / (σ² + prior_var)

Jacobian w.r.t. x is ``prior_var / (σ² + prior_var)`` — near 0 at high σ
(matches the legacy oracle), → 1 at low σ. That's exactly when er_sde+karras
(tiny late-step Δλ) and euler_a (random-walk noise riding feedback) blow up
in real sampling. ``--prior_var 0`` recovers the old x-independent oracle.

This script simulates the sampler update equations directly, injecting
Gaussian error on the subset of steps Spectrum would cache (determined by
the same log-sigma-uniform heuristic as the ComfyUI custom node). It reports
per-step ``||x_spec − x_ref||`` averaged over trials, plus endpoint distance
to target so you can see whether spec actually converged.

Ref and spec runs share the same ancestral-noise RNG sequence so that any
difference between trajectories is attributable to the injected cache error
alone, not to the sampler's intrinsic stochasticity.

What this tells you
-------------------
* Which sampler × scheduler pair amplifies a given cache-error magnitude
  through the sampler equations *and* through closed-loop feedback.
* Where in the schedule (which step range) the amplification peaks.
* Whether the cell still converges to the data manifold (endpoint_err) or
  has been knocked off it (corruption proxy).

Interpretation guide
--------------------
* peak_drift / final_drift — paired-trajectory L2 between spec and ref.
* spec_endpoint_err / ref_endpoint_err — distance from x_final to target. If
  spec_endpoint_err >> ref_endpoint_err, caching pushed the trajectory off
  the data manifold (image-corruption proxy).
* endpoint_err_ratio = spec / ref — single-number "how much extra damage did
  caching cost". 1.0 = caching is free; 5+ = visible corruption in real
  images.
* Rising drift curves through the low-sigma tail → late-schedule cache
  errors dominate → push ``stop_caching_step`` lower.
* Flat drift then sudden spike → finite-difference amplification at a
  specific sigma gap.

Usage
-----
    python bench/spectrum/analyze_drift.py
    python bench/spectrum/analyze_drift.py --error_mag 0.01 --trials 40
    python bench/spectrum/analyze_drift.py --prior_var 0.1        # weaker feedback
    python bench/spectrum/analyze_drift.py --prior_var 0          # legacy oracle
    python bench/spectrum/analyze_drift.py --forbid_consec_all    # all-guarded
    python bench/spectrum/analyze_drift.py --no_caching           # sanity: drift=0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Plotting is optional — fail gracefully if matplotlib isn't installed.
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Scheduler sigma grids (matched to comfy.k_diffusion.sampling formulas)
# ---------------------------------------------------------------------------

def sigmas_simple(n: int, s_min: float = 0.02, s_max: float = 14.6) -> np.ndarray:
    """Linear-in-sigma, roughly log-uniform for Anima's sigma range."""
    return np.concatenate([np.linspace(s_max, s_min, n), [0.0]])


def sigmas_karras(n: int, s_min: float = 0.02, s_max: float = 14.6,
                  rho: float = 7.0) -> np.ndarray:
    """Karras: sigma^(1/rho) linearly spaced. Front-loaded."""
    ramp = np.linspace(0, 1, n)
    x = (s_max ** (1 / rho) + ramp * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    return np.concatenate([x, [0.0]])


def sigmas_exponential(n: int, s_min: float = 0.02, s_max: float = 14.6) -> np.ndarray:
    """Log-uniform — equivalent to karras with rho=1 in log-space."""
    return np.concatenate([
        np.exp(np.linspace(math.log(s_max), math.log(s_min), n)),
        [0.0],
    ])


def sigmas_kl_optimal(n: int, s_min: float = 0.02, s_max: float = 14.6) -> np.ndarray:
    """KL-optimal-ish: aggressive front-load via large rho."""
    rho = 10.0
    ramp = np.linspace(0, 1, n)
    x = (s_max ** (1 / rho) + ramp * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    return np.concatenate([x, [0.0]])


def sigmas_karras_rho3(n: int, s_min: float = 0.02, s_max: float = 14.6) -> np.ndarray:
    """Karras with rho=3 — milder front-load, more late-tail snap."""
    return sigmas_karras(n, s_min, s_max, rho=3.0)


def sigmas_linear_quadratic(n: int, s_min: float = 0.02, s_max: float = 14.6,
                            split_ratio: float = 0.25) -> np.ndarray:
    """Linear σ for the first split_ratio fraction, quadratic decay after.

    Approximates ComfyUI's linear_quadratic scheduler. Hybrid that retains
    some of simple's late-tail snap while front-loading more carefully.
    """
    k = max(2, int(round(split_ratio * n)))
    s_mid = s_max - (s_max - s_min) * 0.5
    linear_part = np.linspace(s_max, s_mid, k, endpoint=False)
    rest = n - k
    ramp = np.linspace(0.0, 1.0, rest)
    quad_part = s_mid + (s_min - s_mid) * (ramp ** 2)
    return np.concatenate([linear_part, quad_part, [0.0]])


def sigmas_polyexp(n: int, s_min: float = 0.02, s_max: float = 14.6,
                   rho: float = 1.5) -> np.ndarray:
    """Power-law in sigma with adjustable rho. rho=1 → exponential,
    rho>1 → simple-flavored with growing tail snap."""
    ramp = np.linspace(0.0, 1.0, n)
    log_min, log_max = math.log(s_min), math.log(s_max)
    # Bias the ramp toward the simple end via x → x^(1/rho), giving wider
    # late log-σ gaps than plain exponential.
    biased = ramp ** (1.0 / rho)
    return np.concatenate([np.exp(log_max + biased * (log_min - log_max)), [0.0]])


SCHEDULERS = {
    "simple": sigmas_simple,
    "karras": sigmas_karras,
    "karras_rho3": sigmas_karras_rho3,
    "exponential": sigmas_exponential,
    "kl_optimal": sigmas_kl_optimal,
    "linear_quadratic": sigmas_linear_quadratic,
    "polyexp": sigmas_polyexp,
}


def auto_stop_caching_step(sigmas: np.ndarray, base_keep: int = 3) -> int:
    """Schedule-aware stop_caching_step.

    Schedules with a strong late-Δλ "tail-snap" (e.g. simple) self-correct
    accumulated cache error in the last few steps. Uniform schedules (karras,
    exponential, kl_optimal) lack this wash-out, so any mid-schedule cache
    error persists to the endpoint. We measure the ratio of the last-3-gap
    mean to the overall mean: ≥2 means strong tail-snap (keep base_keep
    actuals at the end), <2 means uniform (push stop earlier so more late
    actuals can dilute drift).
    """
    n = len(sigmas) - 1  # number of denoising steps
    log_s = np.log(np.maximum(sigmas[:n], 1e-8))
    gaps = np.abs(np.diff(log_s))
    if gaps.size < 4:
        return max(0, n - base_keep)
    tail3 = float(gaps[-3:].mean())
    overall = float(gaps.mean())
    ratio = tail3 / max(overall, 1e-6)
    if ratio >= 2.0:
        keep = base_keep
    else:
        keep = base_keep + max(1, int(round(5.0 * (2.0 - ratio))))
    return max(0, n - keep)


# ---------------------------------------------------------------------------
# Toy denoised function
# ---------------------------------------------------------------------------

def posterior_denoised(
    x: np.ndarray, sigma: float, target: np.ndarray, prior_var: float
) -> np.ndarray:
    """Tweedie posterior mean for data ~ N(target, prior_var · I).

    For x_t = x_0 + σ·ε with x_0 ~ N(target, prior_var·I), the optimal
    denoiser is closed-form:

        E[x_0 | x_t, σ] = (σ² · target + prior_var · x_t) / (σ² + prior_var)

    Jacobian w.r.t. x is prior_var / (σ² + prior_var). At high σ this is
    near 0 (matches an oracle that ignores x); at low σ it tends to 1, so
    cache errors that drift x propagate fully into the next denoised. This
    closed loop is the dominant amplification mode in real diffusion that
    the prior x-independent oracle missed. ``prior_var = 0`` recovers the
    oracle (denoised = target everywhere).
    """
    if prior_var <= 0.0:
        return target
    s2 = sigma * sigma
    return (s2 * target + prior_var * x) / (s2 + prior_var)


# ---------------------------------------------------------------------------
# Samplers (k-diffusion-flavored, flow-matching compatible)
# ---------------------------------------------------------------------------
# Each sampler takes (x, sigma, sigma_next, denoised, state, noise, denoise_fn)
# and returns x_next. `state` is a dict for multistep history. `noise` is a
# pre-generated Gaussian sample of shape x — shared between ref and spec so
# ancestral stochasticity cancels out of the drift measurement. `denoise_fn`
# is an oracle callback `(x, sigma) -> denoised` for samplers that need extra
# denoiser calls inside one outer step (e.g. heun's corrector). The cache
# error is only applied to the *primary* denoised passed in; sub-step
# denoisers are always actual — modeling spectrum's per-forward cache trigger
# where the corrector pass typically misses the cache window.

def sampler_euler(x, sigma, sigma_next, denoised, state, noise, denoise_fn):
    if sigma == 0:
        return denoised
    d = (x - denoised) / sigma
    return x + (sigma_next - sigma) * d


def sampler_euler_ancestral(x, sigma, sigma_next, denoised, state, noise, denoise_fn, eta=1.0):
    if sigma_next == 0:
        return denoised
    # k-diffusion get_ancestral_step
    sigma_up = min(sigma_next, eta * math.sqrt(
        max(0.0, (sigma_next ** 2) * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2))
    ))
    sigma_down = math.sqrt(max(0.0, sigma_next ** 2 - sigma_up ** 2))
    d = (x - denoised) / sigma
    x_det = x + (sigma_down - sigma) * d
    return x_det + sigma_up * noise


def sampler_heun(x, sigma, sigma_next, denoised, state, noise, denoise_fn):
    """2nd-order predictor-corrector. Corrector self-corrects single-step
    cache errors when the corrector denoised is actual."""
    if sigma_next == 0:
        return denoised
    d = (x - denoised) / sigma
    x_pred = x + (sigma_next - sigma) * d
    denoised_2 = denoise_fn(x_pred, sigma_next)
    d_2 = (x_pred - denoised_2) / max(sigma_next, 1e-9)
    return x + (sigma_next - sigma) * 0.5 * (d + d_2)


def sampler_dpmpp_2s_ancestral(x, sigma, sigma_next, denoised, state, noise,
                               denoise_fn, eta=1.0):
    """2nd-order single-step ancestral DPM-Solver++. Ancestral noise + a
    corrector step at the midpoint."""
    if sigma_next == 0:
        return denoised
    sigma_up = min(sigma_next, eta * math.sqrt(
        max(0.0, (sigma_next ** 2) * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2))
    ))
    sigma_down = math.sqrt(max(0.0, sigma_next ** 2 - sigma_up ** 2))
    if sigma_down <= 0:
        return denoised + sigma_up * noise
    t = -math.log(max(sigma, 1e-9))
    t_d = -math.log(max(sigma_down, 1e-9))
    h = t_d - t
    s_mid = math.exp(-(t + h * 0.5))
    x_mid = (s_mid / sigma) * x - math.expm1(-h * 0.5) * denoised
    denoised_2 = denoise_fn(x_mid, s_mid)
    x_det = (sigma_down / sigma) * x - math.expm1(-h) * denoised_2
    return x_det + sigma_up * noise


def sampler_dpmpp_2m(x, sigma, sigma_next, denoised, state, noise, denoise_fn):
    if sigma_next == 0:
        return denoised
    t_fn = lambda s: -math.log(max(s, 1e-9))
    h = t_fn(sigma_next) - t_fn(sigma)
    denoised_prev = state.get("denoised_prev")
    if denoised_prev is None:
        x_next = (sigma_next / sigma) * x - math.expm1(-h) * denoised
    else:
        h_prev = state["h_prev"]
        r = h_prev / h
        # 2nd-order extrapolation of denoised
        denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * denoised_prev
        x_next = (sigma_next / sigma) * x - math.expm1(-h) * denoised_d
    state["denoised_prev"] = denoised
    state["h_prev"] = h
    return x_next


def sampler_dpmpp_2m_sde(x, sigma, sigma_next, denoised, state, noise,
                         denoise_fn, eta=1.0):
    """Multistep DPM-Solver++ with stochastic kick. Combines FD amplification
    (cached denoised in 2nd-order extrapolation) with ancestral-style noise."""
    if sigma_next == 0:
        return denoised
    t_fn = lambda s: -math.log(max(s, 1e-9))
    h = t_fn(sigma_next) - t_fn(sigma)
    eta_h = eta * h
    x_next = ((sigma_next / sigma) * math.exp(-eta_h) * x
              + (1.0 - math.exp(-(h + eta_h))) * denoised)
    prev = state.get("denoised_prev")
    if prev is not None:
        h_prev = state["h_prev"]
        r = h_prev / h
        # Same FD-amplification source as dpmpp_2m, weighted differently.
        x_next = x_next + (1.0 - math.exp(-(h + eta_h))) * (1.0 / (-2.0 * r)) * (denoised - prev)
    sigma_up = sigma_next * math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * eta_h)))
    state["denoised_prev"] = denoised
    state["h_prev"] = h
    return x_next + sigma_up * noise


def sampler_er_sde_3(x, sigma, sigma_next, denoised, state, noise, denoise_fn):
    """Simplified 3rd-order ER-SDE.

    Not bit-exact to comfy's implementation — uses the qualitatively-
    equivalent form: first-order Euler + FD correction from prior denoised +
    SDE noise scaled by the log-sigma gap. The key error-amplification
    pattern is the ``(denoised - denoised_prev) / dλ`` term which divides
    cache errors by the λ-gap magnitude.
    """
    if sigma_next == 0:
        return denoised
    lam = math.log(max(sigma, 1e-9))
    lam_next = math.log(max(sigma_next, 1e-9))
    dl = lam_next - lam
    # First-order Euler-ish term
    d = (x - denoised) / sigma
    x_next = x + (sigma_next - sigma) * d
    # 2nd-order FD correction (the dangerous piece)
    prev = state.get("denoised_prev")
    if prev is not None:
        dl_prev = state["dl_prev"]
        # (denoised - denoised_prev) / Δλ — amplifies cache error as Δλ shrinks.
        fd = (denoised - prev) / max(abs(dl_prev), 1e-6)
        x_next = x_next + 0.5 * (sigma_next - sigma) * fd
    state["denoised_prev"] = denoised
    state["dl_prev"] = dl
    # SDE stochastic term — scale by sqrt(|Δλ|), small fraction of step size.
    x_next = x_next + 0.3 * math.sqrt(abs(dl)) * abs(sigma_next - sigma) * noise
    return x_next


SAMPLERS = {
    "euler": sampler_euler,
    "euler_a": sampler_euler_ancestral,
    "heun": sampler_heun,
    "dpmpp_2s_a": sampler_dpmpp_2s_ancestral,
    "dpmpp_2m": sampler_dpmpp_2m,
    "dpmpp_2m_sde": sampler_dpmpp_2m_sde,
    "er_sde_3": sampler_er_sde_3,
}
# Samplers we currently mark as "fragile" (guarded) in the custom node.
# Multistep / SDE samplers FD-divide cached errors. Ancestral samplers ride
# closed-loop feedback through their stochastic kicks. Both benefit from
# the no-consecutive-caches guard.
#
# `er_sde_3` was previously here but was removed from the production guard
# set: empirically er_sde + simple runs cleanly without the guard (the
# simple scheduler's tail-snap absorbs the FD amplification). Mirror that
# here so the simulation matches production behavior. er_sde + karras /
# exponential / kl_optimal will look worse in this bench as a result —
# that's intentional and matches what the production node will produce
# off-recommendation.
FRAGILE = {"dpmpp_2m", "dpmpp_2m_sde", "euler_a", "dpmpp_2s_a"}


# ---------------------------------------------------------------------------
# Spectrum cache-decision mirror (matches the ComfyUI node's should_cache)
# ---------------------------------------------------------------------------

def spectrum_cache_mask(
    sigmas: np.ndarray,
    warmup_steps: int,
    window_size: float,
    flex_window: float,
    stop_caching_step: int | None,
    forbid_consec: bool,
    min_dl_factor: float = 0.0,
) -> np.ndarray:
    """Return a boolean mask of length len(sigmas)-1. True = this step is
    cached by Spectrum (the forecaster replaces the actual forward).

    `min_dl_factor` (k): Δλ-aware guard. Skip caching at step i when the
    forward gap |log σ_i − log σ_{i+1}| < k · dlu (mean gap in caching
    region). Caching step i means at step i+1 the FD term divides by that
    forward gap — small gap → blown-up error. k=0 disables, k=0.5 trims
    karras/exp tails, k=1.0 only allows caching where local gap meets the
    region average.
    """
    n = len(sigmas) - 1
    log_s_all = np.log(np.maximum(sigmas[:-1], 1e-8))
    # Forward gaps |log σ_i − log σ_{i+1}|. For the last step there's no
    # "next" sigma in the caching loop sense; pad with +inf so the guard
    # never trips on it (irrelevant — it's past stop_caching_step anyway).
    log_s_next = np.log(np.maximum(sigmas[1:n], 1e-8)) if n >= 2 else np.array([])
    forward_dl = np.full(n, np.inf, dtype=np.float64)
    if log_s_next.size > 0:
        forward_dl[: log_s_next.size] = np.abs(log_s_all[: log_s_next.size] - log_s_next)
    if stop_caching_step is None or stop_caching_step < 0:
        stop_caching_step = max(warmup_steps, n - 3)

    lo = max(0, min(warmup_steps, n - 1))
    hi = max(lo + 2, min(stop_caching_step + 1, n))
    gaps = np.abs(np.diff(log_s_all[lo:hi]))
    dlu = float(gaps.mean()) if gaps.size > 0 else 0.1
    dlu = max(dlu, 1e-6)
    min_dl_thresh = min_dl_factor * dlu

    mask = np.zeros(n, dtype=bool)
    delta_ls = window_size * dlu
    last_actual_ls: float | None = None
    consec = 0
    for i in range(n):
        if i < warmup_steps or i >= stop_caching_step:
            mask[i] = False
            last_actual_ls = log_s_all[i]
            consec = 0
            if i >= warmup_steps:
                delta_ls += flex_window * dlu
            continue
        if forbid_consec and consec >= 1:
            mask[i] = False
            last_actual_ls = log_s_all[i]
            consec = 0
            delta_ls += flex_window * dlu
            continue
        if last_actual_ls is None:
            mask[i] = False
            last_actual_ls = log_s_all[i]
            continue
        # Δλ guard: don't cache if the forward gap is too small (would
        # amplify cache error in FD-using samplers at step i+1).
        if forward_dl[i] < min_dl_thresh:
            mask[i] = False
            last_actual_ls = log_s_all[i]
            consec = 0
            delta_ls += flex_window * dlu
            continue
        if abs(log_s_all[i] - last_actual_ls) < delta_ls:
            mask[i] = True
            consec += 1
        else:
            mask[i] = False
            last_actual_ls = log_s_all[i]
            consec = 0
            delta_ls += flex_window * dlu
    return mask


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    trajectory: np.ndarray  # shape: (n_steps + 1, dim)


def simulate(
    sigmas: np.ndarray,
    sampler_fn,
    cache_mask: np.ndarray,
    error_mag: float,
    target: np.ndarray,
    x0: np.ndarray,
    step_noise: np.ndarray,
    cache_error: np.ndarray,
    prior_var: float,
) -> SimResult:
    """Run one sampling trajectory with optional per-step denoised error.

    Reference runs pass cache_mask all False (or error_mag=0). Both ref and
    spec runs must share identical target, x0, step_noise, and prior_var —
    only the mask/error differ — so the drift diff is purely the injected
    error's propagation, not sampler stochasticity.
    """
    x = x0.copy()
    traj = [x.copy()]
    state: dict = {}
    denoise_fn = lambda x_, s_: posterior_denoised(x_, s_, target, prior_var)
    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        denoised = denoise_fn(x, sigma)
        if cache_mask[i]:
            scale = error_mag * max(float(np.abs(denoised).mean()), 1e-8)
            denoised = denoised + scale * cache_error[i]
        x = sampler_fn(x, sigma, sigma_next, denoised, state, step_noise[i], denoise_fn)
        traj.append(x.copy())
    return SimResult(trajectory=np.stack(traj))


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def sweep(args, out_dir: Path):
    rng_master = np.random.default_rng(args.seed)

    # Per-trial invariants shared across all (sampler, scheduler) cells so
    # trials are paired — cross-cell comparisons aren't noise-masked.
    trials = []
    for t in range(args.trials):
        target = rng_master.standard_normal(args.dim).astype(np.float32)
        x0_unit = rng_master.standard_normal(args.dim).astype(np.float32)  # scale by sigma_max per scheduler
        # One step-noise and cache-error sequence per trial — same across all cells.
        step_noise = rng_master.standard_normal((args.steps, args.dim)).astype(np.float32)
        cache_error = rng_master.standard_normal((args.steps, args.dim)).astype(np.float32)
        trials.append({
            "target": target, "x0_unit": x0_unit,
            "step_noise": step_noise, "cache_error": cache_error,
        })

    rows = []
    # Store per-cell drift trajectories for plotting.
    curves: dict[tuple[str, str], np.ndarray] = {}

    for sched_name, sched_fn in SCHEDULERS.items():
        sigmas = sched_fn(args.steps).astype(np.float32)
        for samp_name, samp_fn in SAMPLERS.items():
            forbid = (samp_name in FRAGILE) and not args.no_guards
            if args.forbid_consec_all:
                forbid = True
            stop = (auto_stop_caching_step(sigmas) if args.auto_stop
                    else args.stop_caching_step)
            mask = (spectrum_cache_mask(
                sigmas,
                warmup_steps=args.warmup_steps,
                window_size=args.window_size,
                flex_window=args.flex_window,
                stop_caching_step=stop,
                forbid_consec=forbid,
                min_dl_factor=args.min_dl_factor,
            ) if not args.no_caching else np.zeros(len(sigmas) - 1, dtype=bool))

            drifts = np.zeros((args.trials, len(sigmas)), dtype=np.float32)
            ref_endpoint = np.zeros(args.trials, dtype=np.float32)
            spec_endpoint = np.zeros(args.trials, dtype=np.float32)
            for ti, t_ in enumerate(trials):
                x0 = t_["x0_unit"] * sigmas[0]  # scale noise to sigma_max
                ref = simulate(
                    sigmas, samp_fn,
                    cache_mask=np.zeros_like(mask),
                    error_mag=0.0,
                    target=t_["target"], x0=x0,
                    step_noise=t_["step_noise"],
                    cache_error=t_["cache_error"],
                    prior_var=args.prior_var,
                )
                spec = simulate(
                    sigmas, samp_fn,
                    cache_mask=mask,
                    error_mag=args.error_mag,
                    target=t_["target"], x0=x0,
                    step_noise=t_["step_noise"],
                    cache_error=t_["cache_error"],
                    prior_var=args.prior_var,
                )
                drifts[ti] = np.linalg.norm(spec.trajectory - ref.trajectory, axis=1)
                ref_endpoint[ti] = np.linalg.norm(ref.trajectory[-1] - t_["target"])
                spec_endpoint[ti] = np.linalg.norm(spec.trajectory[-1] - t_["target"])

            mean_drift = drifts.mean(axis=0)
            std_drift = drifts.std(axis=0)
            n_cached = int(mask.sum())
            curves[(sched_name, samp_name)] = np.stack([mean_drift, std_drift])

            ref_ep_mean = float(ref_endpoint.mean())
            spec_ep_mean = float(spec_endpoint.mean())
            ratio = spec_ep_mean / ref_ep_mean if ref_ep_mean > 1e-8 else float("nan")

            rows.append({
                "sampler": samp_name,
                "scheduler": sched_name,
                "n_cached": n_cached,
                "n_steps": len(sigmas) - 1,
                "forbid_consec": forbid,
                "final_drift_mean": float(mean_drift[-1]),
                "final_drift_std": float(std_drift[-1]),
                "peak_drift_mean": float(mean_drift.max()),
                "peak_drift_step": int(mean_drift.argmax()),
                "ref_endpoint_err": ref_ep_mean,
                "spec_endpoint_err": spec_ep_mean,
                "endpoint_err_ratio": ratio,
            })

    # CSV
    csv_path = out_dir / "drift_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV → {csv_path}")

    # JSON with full trajectories
    json_path = out_dir / "drift_curves.json"
    json_payload = {
        "config": vars(args),
        "curves": {
            f"{s}__{p}": {"mean": m.tolist(), "std": sd.tolist()}
            for (s, p), arr in curves.items()
            for m, sd in [arr]
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2))
    print(f"JSON → {json_path}")

    # Plot
    if not HAS_MPL and not args.no_plot:
        print("[plot skipped — matplotlib not installed. Run `uv add matplotlib` "
              "for plotting, or --no_plot to silence this message]")
    if HAS_MPL and not args.no_plot:
        # Per-step drift curves: one subplot per scheduler. Wrap to 2 rows
        # if many schedulers.
        n_sched = len(SCHEDULERS)
        ncols = min(4, n_sched)
        nrows = (n_sched + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.2 * ncols, 3.2 * nrows),
                                 sharey=True, squeeze=False)
        axes_flat = axes.flatten()
        for ax, (sched_name, _) in zip(axes_flat, SCHEDULERS.items()):
            for samp_name in SAMPLERS:
                mean, std = curves[(sched_name, samp_name)]
                xs = np.arange(len(mean))
                ax.plot(xs, mean, label=samp_name, linewidth=1.4)
                ax.fill_between(xs, mean - std, mean + std, alpha=0.12)
            ax.set_title(sched_name)
            ax.set_xlabel("step")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[n_sched:]:
            ax.set_visible(False)
        axes[0, 0].set_ylabel(r"$\|x_\mathrm{spec} - x_\mathrm{ref}\|_2$")
        axes_flat[n_sched - 1].legend(loc="upper left", framealpha=0.9, fontsize=8,
                                      ncol=1)
        title = (f"Spectrum drift (ε={args.error_mag:g}, "
                 f"prior_var={args.prior_var:g}, "
                 f"{args.trials} trials, {args.steps} steps, "
                 f"guards {'ON' if not args.no_guards else 'OFF'}, "
                 f"auto_stop={'ON' if args.auto_stop else 'OFF'})")
        fig.suptitle(title, y=1.00)
        plt.tight_layout()
        png_path = out_dir / "drift.png"
        fig.savefig(png_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot → {png_path}")

        # Heatmap: final_drift_mean per (sampler × scheduler). Headline view
        # for spotting the best combo.
        sampler_order = list(SAMPLERS.keys())
        sched_order = list(SCHEDULERS.keys())
        H = np.full((len(sampler_order), len(sched_order)), np.nan)
        lookup = {(r["sampler"], r["scheduler"]): r["final_drift_mean"] for r in rows}
        for i, sa in enumerate(sampler_order):
            for j, sc in enumerate(sched_order):
                H[i, j] = lookup.get((sa, sc), np.nan)
        log_H = np.log10(np.maximum(H, 1e-4))
        fig2, ax = plt.subplots(figsize=(1.0 * len(sched_order) + 2.5,
                                         0.55 * len(sampler_order) + 1.8))
        im = ax.imshow(log_H, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(sched_order)))
        ax.set_xticklabels(sched_order, rotation=35, ha="right")
        ax.set_yticks(range(len(sampler_order)))
        ax.set_yticklabels(sampler_order)
        # Mark the best (lowest drift) cell with a red border.
        best_i, best_j = np.unravel_index(np.nanargmin(H), H.shape)
        # Annotate every cell with the raw drift value.
        vmid = (np.nanmin(log_H) + np.nanmax(log_H)) * 0.5
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H[i, j]
                if np.isnan(v):
                    continue
                color = "white" if log_H[i, j] > vmid else "black"
                weight = "bold" if (i, j) == (best_i, best_j) else "normal"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight=weight)
        ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                   fill=False, edgecolor="red", linewidth=2.0))
        ax.set_title(f"final_drift_mean — bold = best ({sampler_order[best_i]} "
                     f"+ {sched_order[best_j]} = {H[best_i, best_j]:.3f})")
        plt.colorbar(im, ax=ax, label=r"$\log_{10}$ final drift")
        plt.tight_layout()
        heatmap_path = out_dir / "drift_heatmap.png"
        fig2.savefig(heatmap_path, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"Heatmap → {heatmap_path}")

    # Summary table — `final_drift_mean` is the headline number; rank cells
    # by it within each sampler. Endpoint columns live in the CSV.
    print(f"\n{'sampler':12s} {'scheduler':12s} {'cached':>8s}  "
          f"{'peak@step':>10s}  {'final drift':>22s}")
    print("-" * 74)
    for r in sorted(rows, key=lambda r: (r["sampler"], r["scheduler"])):
        print(
            f"{r['sampler']:12s} {r['scheduler']:12s} "
            f"{r['n_cached']:>3d}/{r['n_steps']:<3d}   "
            f"{r['peak_drift_mean']:>6.3f}@{r['peak_drift_step']:<3d}  "
            f"{r['final_drift_mean']:>10.3f} ± {r['final_drift_std']:<7.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--dim", type=int, default=256,
                    help="Latent vector dim (higher = tighter std error bars)")
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--error_mag", type=float, default=0.02,
                    help="Relative Gaussian error on cached denoised. 0.02 ~ "
                    "2%% of |denoised| — typical Spectrum prediction error.")
    ap.add_argument("--prior_var", type=float, default=1.0,
                    help="Toy data prior variance for the Tweedie denoiser. "
                    "Controls feedback strength: Jacobian = prior_var/(σ²+prior_var). "
                    "0 → x-independent oracle (legacy behavior). 1.0 (default) "
                    "gives meaningful low-σ feedback. Higher = more amplification.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup_steps", type=int, default=7)
    ap.add_argument("--window_size", type=float, default=2.0)
    ap.add_argument("--flex_window", type=float, default=0.25)
    ap.add_argument("--stop_caching_step", type=int, default=-1)
    ap.add_argument("--no_guards", action="store_true",
                    help="Disable the fragile-sampler consecutive-cache guard")
    ap.add_argument("--forbid_consec_all", action="store_true",
                    help="Apply the no-consecutive-cached guard to every sampler")
    ap.add_argument("--min_dl_factor", type=float, default=0.0,
                    help="Δλ-aware cache guard. Skip caching at step i if "
                    "|log σ_i − log σ_{i+1}| < k · dlu. 0 disables (legacy), "
                    "0.5 trims karras/exp tail, 1.0 only caches where local "
                    "gap ≥ region average. Try sweeping 0 / 0.3 / 0.5 / 0.7.")
    ap.add_argument("--auto_stop", action="store_true",
                    help="Pick stop_caching_step per scheduler from tail-snap "
                    "ratio (last-3-gap mean / overall mean). Strong tail-snap "
                    "(simple ≈ 6) keeps last 3 actuals; uniform schedules "
                    "(karras / exp / kl_optimal, ratio ≤ 1.5) push stop "
                    "earlier so more late actuals dilute accumulated drift.")
    ap.add_argument("--no_caching", action="store_true",
                    help="Sanity check — all steps actual, drift should be 0")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--out", default="bench/spectrum/results/drift")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"out_dir = {out_dir}")

    sweep(args, out_dir)


if __name__ == "__main__":
    main()
