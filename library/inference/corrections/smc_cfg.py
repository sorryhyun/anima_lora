"""Sliding-Mode Control CFG (SMC-CFG), α-adaptive variant.

Adapted from Wang et al., "CFG-Ctrl: Control-Based Classifier-Free Diffusion
Guidance" (arXiv:2603.03281), with the paper's fixed switching gain replaced
by α-adaptive sliding-mode control (Plestan et al., 2010 — classical SMC
literature).

Drop-in modification of the CFG cond/uncond combine. At each denoising step:

    e_t        = v_cond − v_uncond              (semantic error, velocity-space)
    s_t        = (e_t − e_prev) + λ · e_prev    (sliding-mode surface)
    k_t        = α · mean(|e_t|)                (adaptive switching gain)
    Δe         = −k_t · sign(s_t)               (bang-bang switching correction)
    v̂_t        = v_uncond + w · (e_t + Δe)

The adaptive gain keeps the controller in-band across model / CFG / σ /
sample (the paper's fixed k=0.1 is off by ~14× on Anima at CFG=4; see
_archive/bench/smc_cfg/analysis_and_proposal.md §A). α=0.2 is the default.

No tanh boundary layer: at α=0.2 the ±k_t bang-bang stays below visibility,
while tanh-with-auto-ε concentrates the correction and surfaces as grain.

`e_prev` is the raw e from the previous step (None → e_prev := e_t on the
first step, matching the paper's `if e(t+1) is None then e(t+1) ← e(t)`).
We store the *uncontrolled* e_prev so the sliding surface tracks the real
discrepancy, not the controller's own feedback.

No extra DiT forwards. One velocity-shaped buffer of state.
Composes with mod-guidance (AdaLN-side).

This module is **pure compute** (torch only — no anima/comfy deps) and is the
single source of truth shared verbatim with the ComfyUI Spectrum node (vendored
by scripts/release/sync_vendor.py) and the DirectEdit node.
"""

from __future__ import annotations

from typing import Optional

import torch


class SMCCFGState:
    def __init__(
        self,
        lam: float = 5.0,
        alpha: float = 0.2,
    ):
        self.lam = float(lam)
        # alpha: dimensionless adaptive gain. k_t = alpha · |e_t|.mean() per
        # step — self-scales across model / CFG / σ / sample (see
        # _archive/bench/smc_cfg/analysis_and_proposal.md §A).
        self.alpha = float(alpha)
        self._e_prev: Optional[torch.Tensor] = None

    def combine(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        e = cond - uncond
        # Resolution can change mid-loop (SPD/SPEED spectral expansion at the
        # low→full handoff): the stored previous-step residual then carries a
        # stale spatial shape and can't form the sliding surface. Treat the
        # handoff as a cold start — drop the stale history. Inert (bit-exact)
        # whenever the shape is unchanged, which is every non-SPD step.
        if self._e_prev is not None and self._e_prev.shape != e.shape:
            self._e_prev = None
        e_prev = e if self._e_prev is None else self._e_prev
        s = (e - e_prev) + self.lam * e_prev

        k_t = self.alpha * e.abs().mean().clamp_min(1e-12)
        delta_e = -k_t * torch.sign(s)
        self._e_prev = e.detach()
        return uncond + guidance_scale * (e + delta_e)
