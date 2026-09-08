"""MIST — Moment-aligned Invariant Stability Transform (training-free CFG fix).

Reimplements Peng et al., "MIST: Moment-Aligned Invariant Stability Transform
for Robust Flow Matching" (ICML 2026, paper in repo root 18927_MIST_*.pdf).

Drop-in replacement of the CFG cond/uncond combine, in the same slot as
SMC-CFG (`SMCCFGState.combine`). MIST decomposes the CFG-induced velocity
perturbation δv = v_cond − v_uncond into its first two moments and corrects
both, with NO extra DiT forwards::

    v_MIST = IA( v_uncond + w · ST(δv) )

ST — **Stability Thresholding** (local, applied to the *raw* δv first):

  * Temporal Decay (TD, eq 4): clamp the per-sample guidance L2 norm to be ≤
    the previous step's, enforcing monotone decay; skipped for the first
    `t_clip` steps (early formation is naturally volatile)::
        δv̂ = δv · min(1, ‖δv_prev‖₂ / ‖δv‖₂)
  * Spatial Suppression (SS, eq 5-6): per spatial location, suppress guidance
    where the local guidance-to-structure ratio exceeds γ — a safety anchor
    that damps only the unstable, feature-conflicted regions::
        ρ_{i,j} = w·‖δv̂_{i,j}‖₂ / (‖v_u,{i,j}‖₂ + ε)
        δv_ST_{i,j} = δv̂_{i,j} · clip(γ / ρ_{i,j}, 0, 1)

IA — **Invariant Alignment** (global statistical wrapper on the combined velocity):

  * Drift Correction (DC, eq 2): re-center the guided velocity so its per-channel
    spatial mean matches the unconditional velocity's — removes the "barycentric
    drift" (1st moment shift) CFG injects.
  * Energy Matching (EM, eq 3): renormalize the guided velocity's per-channel
    spatial std back to the unconditional velocity's — tames the "quadratic
    energetic instability" (2nd moment / variance explosion)::
        v_IA = μ_u + (v_dc − μ_u) · σ_u / (σ_dc + ε)

The paper's ablation (Tables 2/9) finds IA — specifically DC — carries most of
the gain; ST adds margins. All four sub-components are individually toggleable
so a bench can reproduce that ablation. ``γ`` defaults to None → use the
guidance scale ``w`` (the same "auto = w" convention FSG uses for its γ).

Reduction granularity is **per-channel spatial** (reduce over every dim from 2
onward, keep B and C) for the IA moments and **per-location** (reduce the
channel dim only) for SS — both computed from ``x.dim()`` so the transform is
ndim-agnostic across the 4D ``(B,C,H,W)`` and 5D ``(B,C,T=1,H,W)`` latents
Anima hands the sampler boundary (never bare-``squeeze`` — the singleton frame
axis stays at dim 2).

``_e_prev_norm`` is the previous step's *raw* per-sample δv norm (matching the
paper's δv_{t+1} reference; None on the first step → TD is inert). Like
SMC-CFG, on a mid-loop resolution change (SPD/SPEED handoff) the stale history
is dropped — TD/SS recompute from the new shape, IA is per-step stateless.

Pure compute (torch only — no anima/comfy deps), so it can be vendored verbatim
into the ComfyUI Spectrum node by scripts/release/sync_vendor.py, exactly like
smc_cfg.py / fsg_core.py.
"""

from __future__ import annotations

from typing import Optional

import torch


def _spatial_dims(x: torch.Tensor) -> tuple[int, ...]:
    """Dims from 2 onward — spatial (+singleton frame), keeping B(0) and C(1)."""
    return tuple(range(2, x.dim()))


class MISTState:
    def __init__(
        self,
        *,
        gamma: Optional[float] = 1.5,
        t_clip: int = 1,
        use_td: bool = True,
        use_ss: bool = True,
        use_dc: bool = True,
        use_em: bool = True,
        ia_strength: float = 1.0,
        eps: float = 1e-8,
    ):
        # γ — SS guidance-to-structure ceiling. Paper default 1.5 (App C, Table
        # 12). None → use the runtime guidance scale w. Lower γ suppresses more
        # (clip(γ/ρ): a location with ρ>γ is damped toward γ).
        self.gamma = None if gamma is None else float(gamma)
        # Skip TD for the first t_clip steps (eq 4 — early formation is volatile).
        self.t_clip = int(t_clip)
        self.use_td = bool(use_td)
        self.use_ss = bool(use_ss)
        self.use_dc = bool(use_dc)
        self.use_em = bool(use_em)
        # IA blend in [0,1]: 0 → plain guided velocity (IA off), 1 → full
        # renorm to the unconditional moments. The paper has no such knob; it's
        # the natural lever to dial back IA's over-correction (on Anima full
        # EM strips guidance-driven color — see the bench).
        self.ia_strength = float(ia_strength)
        self.eps = float(eps)
        self._e_prev_norm: Optional[torch.Tensor] = None  # (B,) raw δv L2, prev step
        self._step: int = 0  # internal counter when step_i is not supplied

    def reset(self) -> None:
        """Cold-start the cross-step state (call once per generation)."""
        self._e_prev_norm = None
        self._step = 0

    def combine(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
        step_i: Optional[int] = None,
    ) -> torch.Tensor:
        w = float(guidance_scale)
        gamma = w if self.gamma is None else self.gamma
        eps = self.eps
        step = self._step if step_i is None else int(step_i)

        e = cond - uncond  # δv, guidance delta

        # ---- ST: Temporal Decay (per-sample monotone-norm clamp) ----
        e_norm = e.flatten(1).norm(dim=1)  # (B,)
        if self.use_td and step >= self.t_clip and self._e_prev_norm is not None:
            # Resolution can change mid-loop (SPD handoff) → stale per-sample
            # norm shape; drop it (cold start) rather than misalign.
            if self._e_prev_norm.shape == e_norm.shape:
                factor = (self._e_prev_norm / e_norm.clamp_min(eps)).clamp(max=1.0)
                e_hat = e * factor.view(-1, *([1] * (e.dim() - 1)))
            else:
                e_hat = e
        else:
            e_hat = e
        self._e_prev_norm = e_norm.detach()  # store RAW norm (paper's δv_{t+1})

        # ---- ST: Spatial Suppression (per-location guidance-to-structure clip) ----
        if self.use_ss:
            e_loc = e_hat.norm(dim=1, keepdim=True)  # (B,1,*spatial) — over channels
            v_loc = uncond.norm(dim=1, keepdim=True)
            rho = w * e_loc / (v_loc + eps)
            supp = (gamma / rho.clamp_min(eps)).clamp(max=1.0)  # min(1, γ/ρ)
            e_st = e_hat * supp
        else:
            e_st = e_hat

        # ---- combine: v_t(x) + w·ST(δv) ----
        g = uncond + w * e_st
        guided = g  # pre-IA guided velocity (for the ia_strength blend)

        # ---- IA: Drift Correction + Energy Matching (per-channel spatial moments) ----
        sdims = _spatial_dims(g)
        mu_u = uncond.mean(dim=sdims, keepdim=True)
        if self.use_dc:
            g = g - g.mean(dim=sdims, keepdim=True) + mu_u  # re-center to uncond mean
        if self.use_em:
            std_u = uncond.std(dim=sdims, keepdim=True)
            center = g.mean(dim=sdims, keepdim=True)
            std_g = g.std(dim=sdims, keepdim=True)
            g = center + (g - center) * (std_u / (std_g + eps))

        # ---- IA strength blend (off-paper): dial back IA toward plain guidance ----
        if self.ia_strength != 1.0:
            g = guided + self.ia_strength * (g - guided)

        self._step = step + 1
        return g
