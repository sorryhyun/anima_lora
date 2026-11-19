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

The adaptive gain self-scales across model / CFG / σ / sample (the paper's
fixed k=0.1 is off by ~14× on Anima at CFG=4; ``docs/anima/inference/smc_cfg.md``).
No tanh boundary layer: at α=0.2 the bang-bang stays below visibility, while
tanh concentrates the correction into grain.

`e_prev` is the *uncontrolled* e from the previous step (e_t on the first
step, as in the paper), so the surface tracks the real discrepancy rather than
the controller's own feedback.

No extra DiT forwards. Pure torch; vendored into the ComfyUI Spectrum and
DirectEdit nodes by scripts/release/sync_vendor.py.
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
        # k_t = alpha · |e_t|.mean() per step.
        self.alpha = float(alpha)
        self._e_prev: Optional[torch.Tensor] = None

    def combine(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        e = cond - uncond
        # SPD/SPEED changes resolution at the low→full handoff; the stale
        # residual can't form the surface, so cold-start there.
        if self._e_prev is not None and self._e_prev.shape != e.shape:
            self._e_prev = None
        e_prev = e if self._e_prev is None else self._e_prev
        s = (e - e_prev) + self.lam * e_prev

        k_t = self.alpha * e.abs().mean().clamp_min(1e-12)
        delta_e = -k_t * torch.sign(s)
        self._e_prev = e.detach()
        return uncond + guidance_scale * (e + delta_e)
