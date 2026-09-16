"""Straight-Through Estimation blend for flow-matching cycle/reverse losses.

BYG (`bench/byg/README.md`, paper arXiv 2606.03911 §4.3,
Eq. 4) decouples *what a forward sees* from *what gradients flow through* when
a loss wants clean conditioning but the only differentiable estimate is a
blurry one-step prediction:

    ŷ_hyb = sg(ỹ_0) + (ŷ − sg(ŷ))                            (Eq. 4)

The forward *value* equals the clean multi-step estimate ``ỹ_0`` (so the
reverse/cycle pass conditions on inference-quality inputs), while the
*gradient* flows through the cheap one-step prediction ``ŷ`` (which is
differentiable w.r.t. the trainable forward velocity). ``sg`` is stop-gradient.

It applies to any loss that wants clean ``x_0``-space conditioning while
keeping gradients connected to the noisy state.

5D-latent invariant (repo CLAUDE.md): every velocity/latent the DiT touches is
5D ``(B, C, T=1, H, W)`` with the singleton at **dim 2**. The blend adds two
such tensors element-wise, so ndim/shape are asserted rather than left to
broadcasting.
"""

from __future__ import annotations

import torch


def ste_clean_blend(y0_clean: torch.Tensor, y_onestep: torch.Tensor) -> torch.Tensor:
    """Return ``sg(y0_clean) + (y_onestep − sg(y_onestep))`` (BYG Eq. 4).

    Forward value equals ``y0_clean``; gradient flows only through
    ``y_onestep``. Both tensors must be 5D ``(B, C, 1, H, W)`` with the
    singleton frame axis at dim 2 and identical shapes.
    """
    if y0_clean.dim() != 5 or y_onestep.dim() != 5:
        raise ValueError(
            "ste_clean_blend expects 5D (B,C,1,H,W) latents; got "
            f"y0_clean.dim()={y0_clean.dim()}, y_onestep.dim()={y_onestep.dim()}. "
            "unsqueeze(2) into the DiT layout before blending (never bare squeeze)."
        )
    if y0_clean.shape[2] != 1 or y_onestep.shape[2] != 1:
        raise ValueError(
            "ste_clean_blend expects the singleton temporal axis at dim 2 "
            f"(T=1); got y0_clean.shape={tuple(y0_clean.shape)}, "
            f"y_onestep.shape={tuple(y_onestep.shape)}."
        )
    if y0_clean.shape != y_onestep.shape:
        raise ValueError(
            "ste_clean_blend operands must match shape; got "
            f"{tuple(y0_clean.shape)} vs {tuple(y_onestep.shape)}."
        )
    return y0_clean.detach() + (y_onestep - y_onestep.detach())
