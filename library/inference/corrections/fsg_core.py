"""Foresight Guidance (FSG) — velocity-source-agnostic core.

Pure compute (no model / comfy / adapter imports), shared by the library plugin
(``corrections/fsg.py``, direct DiT forward) and the ComfyUI Spectrum node
(``calc_cond_batch`` → ``v = (x − x0)/σ``), which vendors it via
``scripts/release/sync_vendor.py``.

FSG treats CFG as a fixed-point calibration: at scheduled timesteps it runs
``K`` forward(cond)–backward(uncond) iterations over a long interval ``Δσ`` to
pull ``x_t → x̂_t`` onto the path where both velocities agree, then the step
proceeds from ``x̂_t``. See ``docs/anima/inference/fsg.md``.

Paper: "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed
Point Iterations" (NeurIPS 2025, arXiv 23177). The paper is ε-prediction + DDIM;
Anima is velocity-prediction flow-matching, so the forward-backward operator maps
onto the reversible Euler ODE (no DDIM machinery):

    v^γ  = v^u + γ·(v^c − v^u)              # CFG-guided velocity
    x'   = x  − Δσ · v^γ(x,   σ)            # denoise σ → σ−Δσ (guided)
    x''  = x' + Δσ · v^u(x',  σ−Δσ)         # re-noise back (unconditional)
    F(x) = x'' ;  iterate x ← F(x), K times

DO NOT EDIT the vendored copy — regenerate it via scripts/release/sync_vendor.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

# Caller-supplied velocity callbacks (velocity-space tensors):
#   vel_cond_uncond(x, σ) -> (v_cond, v_uncond)   — both branches at one σ
#   vel_uncond(x, σ)      -> v_uncond             — uncond only (re-noise leg)
VelCondUncond = Callable[[torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]]
VelUncond = Callable[[torch.Tensor, float], torch.Tensor]


def cfgpp_guidance_weight(sigma_i: float, sigma_next: float, lam: float) -> float:
    """CFG++ effective guidance weight for one step (integrator-agnostic).

    CFG++ differs from CFG only in guidance-strength scheduling (arXiv 23177
    App A.2, eqs 18-19). The flow-matching form is

        w_eff = λ · (1 − σ_next) · σ_i / (σ_i − σ_next)

    Applied as ``noise_pred = v^u + w_eff·(v^c − v^u)`` it equals the Euler
    "calibrate x̂ = x − λ(1−σ')σ·Δv then step along v^u" form, but as a reweight
    of the combine it composes with any integrator (Euler, ER-SDE, LCM).
    At σ_next → 0 it collapses to ``w_eff = λ``.
    """
    ds = sigma_i - sigma_next
    if ds <= 0.0:
        return lam
    return lam * (1.0 - sigma_next) * sigma_i / ds


@dataclass
class FSGCalibrator:
    """Forward-backward fixed-point calibrator config + scheduler + pure loop.

    Each side subclasses it with a ``calibrate(...)`` that builds the velocity
    callbacks and delegates to ``run_fixed_point``.

    Args:
        band: ``(σ_lo, σ_hi)`` — calibrate only when the step's σ falls inside.
            Default [0.59, 0.75] is the 28-step er_sde band; it shifts down
            with step count.
        k: fixed-point iterations per scheduled step (error ~ρ^K, ρ≈0.93 ⇒
            K=3–4 captures ~all the gain). ``k=0`` makes the calibrator inert.
        d_sigma: calibration interval Δσ (the forward-backward stride; *not* the
            sampler's own per-step Δσ). Too-large Δσ is what makes σ≈0.94 diverge.
        gamma: calibration guidance γ. ``None`` → use the outer ``guidance_scale``
            passed at call time (the paper's operator uses plain γ-combine).
    """

    band: Tuple[float, float] = (0.59, 0.75)
    k: int = 3
    d_sigma: float = 0.1
    gamma: Optional[float] = None

    def __post_init__(self) -> None:
        self.k = int(self.k)
        self.d_sigma = float(self.d_sigma)
        lo, hi = float(self.band[0]), float(self.band[1])
        self.band = (lo, hi)

    def scheduled(self, sigma_i: float) -> bool:
        """True iff this step's σ is in-band and the calibrator is active (K>0)."""
        lo, hi = self.band
        return self.k > 0 and lo <= float(sigma_i) <= hi

    @torch.no_grad()
    def run_fixed_point(
        self,
        latents: torch.Tensor,
        sigma_i: float,
        guidance_scale: float,
        vel_cond_uncond: VelCondUncond,
        vel_uncond: VelUncond,
    ) -> torch.Tensor:
        """Return ``x̂`` after K forward-backward iterations, or ``latents``
        unchanged (same object) when this step is not scheduled. Costs ``3·K``
        velocity evaluations.
        """
        if not self.scheduled(sigma_i):
            return latents

        gamma = float(guidance_scale) if self.gamma is None else float(self.gamma)
        ds = self.d_sigma
        s_lo = max(float(sigma_i) - ds, 1e-3)

        x = latents
        for _ in range(self.k):
            vc, vu = vel_cond_uncond(x, float(sigma_i))
            vg = vu + gamma * (vc - vu)
            x_fwd = x - ds * vg  # denoise σ → σ−Δσ (guided)
            vu_lo = vel_uncond(x_fwd, s_lo)
            x = x_fwd + ds * vu_lo  # invert back (uncond)
        return x.to(latents.dtype)
