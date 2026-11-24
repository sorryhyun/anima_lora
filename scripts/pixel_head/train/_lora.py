"""A LoRA on a pixel head's linear layers, for a rank-limited finish.

:func:`add_lora` freezes the head and gives every ``nn.Linear`` in it the update
``(alpha / rank) · B A`` (A random, B zero: the head's output is unchanged until it is
trained). Biases, norms, layer scales and depthwise kernels stay frozen. The head's own
keys and attributes are untouched (``weight`` and ``bias`` stay where they were), so the
forward, the read and ``torch.compile`` see an ordinary head.

:func:`merged_state` is the head's state dict with every update folded into its weight:
the file a trainer writes loads as a plain head. :func:`lora_state` is the factors alone.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Linear):
    """An ``nn.Linear`` (frozen) plus ``lora_scale · lora_B @ lora_A``. Made from an
    existing layer by :func:`add_lora`, not constructed."""

    lora_scale: float

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return y + self.lora_scale * F.linear(F.linear(x, self.lora_A), self.lora_B)

    def delta(self) -> torch.Tensor:
        return self.lora_scale * (self.lora_B.float() @ self.lora_A.float())


def add_lora(head: nn.Module, rank: int, alpha: float | None = None) -> int:
    """Freeze ``head`` and put a rank-``rank`` LoRA on each of its linear layers, in
    place → the number of trainable values. ``alpha`` defaults to ``rank`` (scale 1)."""
    assert not has_lora(head), "add_lora: the head already carries one"
    alpha = float(rank) if not alpha else alpha
    head.requires_grad_(False)
    n = 0
    for m in list(head.modules()):
        if type(m) is not nn.Linear:
            continue
        m.__class__ = LoRALinear
        m.lora_scale = alpha / rank
        a = torch.empty(rank, m.in_features, device=m.weight.device)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        m.lora_A = nn.Parameter(a)
        m.lora_B = nn.Parameter(torch.zeros(m.out_features, rank, device=a.device))
        n += m.lora_A.numel() + m.lora_B.numel()
    return n


def has_lora(head: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in head.modules())


def lora_state(head: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().contiguous().cpu()
        for k, v in head.state_dict().items()
        if k.rsplit(".", 1)[-1] in ("lora_A", "lora_B")
    }


@torch.no_grad()
def merged_state(head: nn.Module) -> dict[str, torch.Tensor]:
    """The head's state dict under its plain keys, each LoRA folded into its weight."""
    sd = {
        k: v.detach().clone()
        for k, v in head.state_dict().items()
        if k.rsplit(".", 1)[-1] not in ("lora_A", "lora_B")
    }
    for name, m in head.named_modules():
        if isinstance(m, LoRALinear):
            w = sd[f"{name}.weight"]
            sd[f"{name}.weight"] = (w.float() + m.delta()).to(w.dtype)
    return sd
