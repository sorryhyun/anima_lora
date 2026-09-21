"""Foresight Guidance (FSG) — library binding of the velocity seam.

The framework-agnostic operator (config gate + K-iteration fixed-point loop +
the CFG++ weight) lives in :mod:`library.inference.corrections.fsg_core`, shared
verbatim with the ComfyUI Spectrum node. This module binds the velocity
callbacks to a **direct DiT forward** — ``anima(x, t, embed)`` with the
hydra/step/FEI/content/crossattn context set exactly as ``generate_body`` does,
so adapter-routed checkpoints calibrate with the routing the real step will use.

**Anima-specific band.** The paper concentrates iterations in the noisiest
stages; on Anima that is the dead zone (σ≈0.94 diverges, ρ>1). The operator
contracts only in **mid-σ**, and the contracting band **moves down with step
count**: at 20-step Euler it was [0.75, 0.85], but at the 28-step er_sde
production schedule σ≈0.84 stops contracting and the sweet spot is σ≈0.75, so
the default is the **[0.59, 0.75]** band.
This is a ``pre-step latent calibration`` seam: it mutates ``latents`` *before* the
real per-step forward and is otherwise invisible to the rest of the loop. Re-probe
the band with ``_archive/bench/fsg`` if you change ``infer_steps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from library.inference.adapters import (
    compute_and_set_hydra_fei,
    set_hydra_crossattn,
    set_hydra_sigma,
    set_step_expert_index,
)
from library.inference.corrections.fsg_core import FSGCalibrator as _FSGCalibratorCore
from library.inference.corrections.fsg_core import cfgpp_guidance_weight

__all__ = ["FSGCalibrator", "cfgpp_guidance_weight"]


@dataclass
class FSGCalibrator(_FSGCalibratorCore):
    """FSG calibrator bound to a direct Anima DiT forward.

    Inherits ``band`` / ``k`` / ``d_sigma`` / ``gamma`` config, the ``scheduled``
    σ-band gate, and the pure ``run_fixed_point`` loop from
    :class:`~library.inference.corrections.fsg_core.FSGCalibrator`.
    """

    @staticmethod
    @torch.no_grad()
    def _velocity(anima, x, sigma, step_i, embed, padding_mask, pooled):
        """Full DiT velocity forward, mirroring ``generate_body``'s per-step call.

        Sets the hydra/step/FEI/content/crossattn context exactly as the sampler
        does (all no-ops on a base DiT), so adapter-routed checkpoints calibrate
        with the same routing the real step will use. ``embed`` selects
        conditional vs unconditional.
        """
        t_b = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
        set_hydra_sigma(anima, t_b)
        set_step_expert_index(anima, step_i)
        compute_and_set_hydra_fei(anima, x)
        set_hydra_crossattn(anima, embed)
        kw = {"pooled_text_override": pooled} if pooled is not None else {}
        return anima(x, t_b, embed, padding_mask=padding_mask, **kw)

    @torch.no_grad()
    def calibrate(
        self,
        anima,
        latents: torch.Tensor,
        sigma_i: float,
        step_i: int,
        embed: torch.Tensor,
        negative_embed: torch.Tensor,
        padding_mask: torch.Tensor,
        guidance_scale: float,
        *,
        pooled_pos: Optional[torch.Tensor] = None,
        pooled_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``x̂`` after K forward-backward iterations, or ``latents``
        unchanged (same object, bit-exact) when this step is not scheduled.

        Costs ``3·K`` extra DiT forwards per scheduled step (v^c + v^u at σ,
        v^u at σ−Δσ). Deterministic — composes with mod-guidance / DAVE / CNS
        for free (they patch the model or the post-step x-space, both of
        which FSG's forwards inherit / precede).
        """
        if not self.scheduled(sigma_i):
            return latents

        def vel_cond_uncond(x, sigma):
            vc = self._velocity(
                anima, x, sigma, step_i, embed, padding_mask, pooled_pos
            )
            vu = self._velocity(
                anima, x, sigma, step_i, negative_embed, padding_mask, pooled_neg
            )
            return vc, vu

        def vel_uncond(x, sigma):
            return self._velocity(
                anima, x, sigma, step_i, negative_embed, padding_mask, pooled_neg
            )

        return self.run_fixed_point(
            latents, sigma_i, guidance_scale, vel_cond_uncond, vel_uncond
        )
