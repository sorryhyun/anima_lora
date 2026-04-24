# Memory-saving autograd functions for the LoRA down projection.
#
# Motivation: ``F.linear(x.float(), weight.float())`` causes autograd to save
# the fp32-cast input for backward. With static_token_count=4096 that's
# ~32 MiB per 2048-wide Linear and ~128 MiB for the 8192-wide MLP layer2 input,
# accumulated across adapted modules. These two Functions save the original
# low-precision ``x`` instead and recompute the fp32 cast in backward — a
# targeted application of the gradient-checkpointing idea to a single op.
#
# The fp32 accumulation of the bottleneck matmul is preserved: forward still
# runs ``F.linear(x_work.float(), weight.float())`` and backward runs the same
# shape/precision matmuls (``go.float() @ w.float()`` / ``go.float().T @ x_f``),
# so gradients are bitwise-identical to the existing path for deterministic
# kernels.
#
# Two separate functions (scaled vs. unscaled) keep the graph shape fixed for
# ``torch.compile`` — no optional tensors, no shape-dependent branches.

from __future__ import annotations

import torch


class LoRADownProjectFn(torch.autograd.Function):
    """``F.linear(x.float(), weight.float())`` without retaining the fp32 cast."""

    @staticmethod
    def forward(ctx, x, weight):
        out = torch.nn.functional.linear(x.float(), weight.float())
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        go = grad_out.float()
        w_f = weight.float()
        x_f = x.float()

        grad_x = go.matmul(w_f).to(x.dtype)
        grad_weight = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
            x_f.reshape(-1, x_f.shape[-1])
        )
        return grad_x, grad_weight.to(weight.dtype)


class ScaledLoRADownProjectFn(torch.autograd.Function):
    """Scaled variant: forward is ``F.linear((x * inv_scale).float(), weight.float())``.

    ``inv_scale`` is a calibration buffer (no gradient). Saving bf16 ``x`` plus
    the 1-D ``inv_scale`` (size == in_features) avoids retaining the
    materialized fp32 ``x * inv_scale`` that the current path otherwise holds.
    """

    @staticmethod
    def forward(ctx, x, weight, inv_scale):
        x_work = x * inv_scale
        out = torch.nn.functional.linear(x_work.float(), weight.float())
        ctx.save_for_backward(x, weight, inv_scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, inv_scale = ctx.saved_tensors
        x_work = x * inv_scale
        go = grad_out.float()
        w_f = weight.float()
        x_f = x_work.float()

        grad_x_work = go.matmul(w_f)
        grad_weight = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
            x_f.reshape(-1, x_f.shape[-1])
        )

        # dL/dx = dL/dx_work * inv_scale  (elementwise broadcast over in-features)
        grad_x = (grad_x_work * inv_scale).to(x.dtype)
        return grad_x, grad_weight.to(weight.dtype), None


def lora_down_project(x, weight, inv_scale):
    """Dispatch helper: picks the scaled or unscaled Function based on inv_scale."""
    if inv_scale is None:
        return LoRADownProjectFn.apply(x, weight)
    return ScaledLoRADownProjectFn.apply(x, weight, inv_scale)
