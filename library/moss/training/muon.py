"""``torch.optim.Muon`` batched by matrix shape.

A LoRA has hundreds of matrices in a handful of shapes. torch's step runs them one by
one, ~34 kernels each, and the launches cost more than the kernels: 154 ms of a 1.30 s
step for 40 ms of GPU time (17k kernels). Same update, one bf16 Newton-Schulz per shape.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import Tensor


def zeropower(update: Tensor, coefficients, steps: int, eps: float) -> Tensor:
    """``_zeropower_via_newtonschulz`` on ``(B, H, W)``; bf16, as there."""
    a, b, c = coefficients
    tall = update.size(1) > update.size(2)
    x = update.bfloat16()
    if tall:
        x = x.mT
    x = x / torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.mT
        gram = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, gram, x, beta=a)
    return x.mT if tall else x


class BatchedMuon(torch.optim.Muon):
    """Takes ``adjust_lr_fn`` ``match_rms_adamw`` only, the one the trainer runs."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["adjust_lr_fn"] != "match_rms_adamw":
                raise ValueError(f"adjust_lr_fn {group['adjust_lr_fn']}")
            by_shape = defaultdict(list)
            for p in group["params"]:
                if p.grad is not None:
                    by_shape[p.shape].append(p)
            for shape, params in by_shape.items():
                self._step_shape(group, shape, params)
        return loss

    def _step_shape(self, group: dict, shape: torch.Size, params: list[Tensor]) -> None:
        lr, momentum = group["lr"], group["momentum"]
        for p in params:
            self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
        buffers = [self.state[p]["momentum_buffer"] for p in params]
        grads = torch.stack([p.grad for p in params])
        stacked = torch.stack(buffers).lerp_(grads, 1 - momentum)
        torch._foreach_copy_(buffers, stacked.unbind(0))
        update = grads.lerp(stacked, momentum) if group["nesterov"] else stacked
        update = zeropower(
            update, group["ns_coefficients"], group["ns_steps"], group["eps"]
        ).to(params[0].dtype)
        torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
        torch._foreach_add_(
            params, update.unbind(0), alpha=-lr * 0.2 * math.sqrt(max(shape))
        )
