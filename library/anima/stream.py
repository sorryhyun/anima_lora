"""fp32 residual stream for Anima: the stream, modulation, q / k norms, RoPE, embedder and
head run in fp32 around the block GEMMs and FA2, which stay bf16 (or int8). User doc:
``docs/anima/optimizations/fp32_stream.md``.

:func:`stream_fp32_` runs at the point int8 converts: after the conversion and before the
adapters attach and ``compile_blocks``, so a LoRA's ``org_forward`` captures the fp32
forward. No parameter changes dtype, so the fp32 linears cost no VRAM and a later
``.to(bf16)`` cannot undo them. The swapped classes run outside autocast, and everything
else inside the DiT forward runs under bf16 autocast, which rounds the block linears' input
and feeds the int8 linears fp32.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.anima.models import Anima, RMSNorm


class Fp32Linear(nn.Linear):
    """Computes in fp32 on the kept weight; casts for the call only."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=False):
            bias = None if self.bias is None else self.bias.float()
            return F.linear(x.float(), self.weight.float(), bias)


class Fp32RMSNorm(RMSNorm):
    """Keeps fp32 on the way out, so RoPE rotates q and k in fp32. ``Attention.forward``
    rounds them on entry to FA2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()) * self.weight.float()


def fp32_linears(dit: Anima) -> Iterator[nn.Linear]:
    """The embedder, head and modulation linears."""
    yield dit.x_embedder.proj[1]
    yield dit.final_layer.linear
    for part in (dit.t_embedder, dit.final_layer.adaln_modulation):
        yield from (m for m in part.modules() if isinstance(m, nn.Linear))
    for block in dit.blocks:
        for name, m in block.named_modules():
            if name.startswith("adaln_") and isinstance(m, nn.Linear):
                yield m


def stream_fp32_(dit: Anima, rope: bool = True) -> None:
    """``rope=False`` leaves the q / k norms and RoPE in the block's dtype."""
    for m in fp32_linears(dit):
        m.__class__ = Fp32Linear
    if rope:
        for block in dit.blocks:
            for attn in (block.self_attn, block.cross_attn):
                attn.q_norm.__class__ = Fp32RMSNorm
                attn.k_norm.__class__ = Fp32RMSNorm

    forward = dit.forward_mini_train_dit

    def under_autocast(x: torch.Tensor, *args, **kwargs):
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            return forward(x, *args, **kwargs)

    dit.forward_mini_train_dit = under_autocast
    dit.stream = "fp32"


def latent_dtype(dit: nn.Module) -> torch.dtype:
    """The sampler state's dtype on ``dit``."""
    return torch.float32 if getattr(dit, "stream", "bf16") == "fp32" else torch.bfloat16
