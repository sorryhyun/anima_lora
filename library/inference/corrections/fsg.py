"""Foresight Guidance (FSG): binds the :mod:`~library.inference.corrections.fsg_core`
operator to a direct DiT forward.

Anima-specific band: the paper iterates in the noisiest stages, but on Anima
σ≈0.94 diverges (ρ>1). The operator contracts only in mid-σ, and the band moves
down with step count, so the [0.59, 0.75] default is tuned for the 28-step
er_sde schedule; re-probe it if ``infer_steps`` changes
(``docs/anima/inference/fsg.md``).

A pre-step latent calibration seam: it mutates ``latents`` before the real
per-step forward and is otherwise invisible to the loop.
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
    """FSG calibrator bound to a direct Anima DiT forward."""

    @staticmethod
    @torch.no_grad()
    def _velocity(anima, x, sigma, step_i, embed, padding_mask, pooled):
        """Full DiT velocity forward, mirroring ``generate_body``'s per-step call.

        Sets the hydra/step/FEI/crossattn routing context as the sampler does
        (no-ops on a base DiT), so adapter-routed checkpoints calibrate with the
        routing the real step will use. ``embed`` selects cond vs uncond.
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
        unchanged (same object) when this step is not scheduled.

        Costs ``3·K`` extra DiT forwards per scheduled step. Composes with
        mod-guidance / DAVE / CNS (they patch the model or the post-step
        x-space).
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
