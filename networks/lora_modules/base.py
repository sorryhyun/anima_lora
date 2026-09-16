# Refs:
#   https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#   https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import random

import torch
from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _absorb_channel_scale(
    weight: torch.Tensor, channel_scale: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """SmoothQuant-style channel-scale absorption into a Linear's input columns.

    Mutates ``weight`` so ``W[:, c] *= s_norm[c]``; returns
    ``inv_scale = 1 / s_norm`` (caller applies ``x * inv_scale`` at forward).
    Output is unchanged — rebalances per-column gradient magnitude. See
    ``_archive/bench/channel_stats/channel_dominance_analysis.md``.
    """
    assert channel_scale.ndim == 1, (
        f"channel_scale must be 1D, got shape {tuple(channel_scale.shape)}"
    )
    assert channel_scale.shape[0] == weight.shape[1], (
        f"channel_scale length {channel_scale.shape[0]} does not match "
        f"weight in_features {weight.shape[1]}"
    )
    s = channel_scale.detach().to(dtype=torch.float32).clamp_min(eps)
    s = s / s.mean().clamp_min(eps)
    with torch.no_grad():
        weight.mul_(s.to(weight).unsqueeze(0))
    # Must track weight's device (forward multiply + save-time bake can't
    # straddle cuda/cpu); fp32 storage is intentional, only the device moves.
    return (1.0 / s).to(weight.device).contiguous()


class BaseLoRAModule(torch.nn.Module):
    """Shared scaffolding: alpha→scale, multiplier, dropouts, channel_scale,
    timestep masking, ``apply_to`` monkey-patching. Subclasses own ``forward``."""

    supports_conv2d: bool = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d" and not self.supports_conv2d:
            raise ValueError(f"{type(self).__name__} does not support Conv2d")

        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self._has_channel_scale = False
        # All-ones mask by default → identity multiply, so forward never needs a
        # None-vs-Tensor guard under torch.compile. Rebound live by
        # LoRANetwork.set_timestep_mask (T-LoRA).
        self.register_buffer(
            "_timestep_mask",
            torch.ones(1, lora_dim, dtype=torch.float32),
            persistent=False,
        )
        self.enabled = True

    def _register_channel_scale(
        self,
        target_weight: torch.Tensor,
        channel_scale,
        *,
        linear_only: bool = True,
    ) -> None:
        if channel_scale is None:
            return
        if linear_only and target_weight.dim() != 2:
            raise ValueError(
                "channel_scale is only supported for Linear LoRA modules, "
                f"got weight with dim {target_weight.dim()}"
            )
        inv_scale = _absorb_channel_scale(target_weight, channel_scale)
        self.register_buffer("inv_scale", inv_scale, persistent=True)
        self._has_channel_scale = True

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _skip_module(self) -> bool:
        return (
            self.module_dropout is not None
            and self.training
            and random.random() < self.module_dropout
        )

    def _rebalance(self, x: torch.Tensor) -> torch.Tensor:
        # inv_scale stays fp32 in storage; cast at the multiply site so
        # bf16 × fp32 → bf16 instead of promoting to fp32.
        if not self._has_channel_scale:
            return x
        return x * self.inv_scale.to(device=x.device, dtype=x.dtype)

    def _apply_rank_dropout(self, lx: torch.Tensor):
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            return lx, self.scale * (1.0 / (1.0 - self.rank_dropout))
        return lx, self.scale

    # Forward scaffold (template method): the invariant chain (enable/fuse
    # short-circuit, eval delegation, module dropout, dtype policy, T-LoRA
    # gate, dropout, residual add) lives here once; two-GEMM variants (LoRA,
    # OrthoInit, StepExpert) supply only _down/_gate/_up. Cayley modules
    # (OrthoLoRA/OrthoHydra, one batched solve shared between down/up) and
    # router-gated MoE modules (Hydra/StackedExperts/Chimera, gate consumed
    # inside the up-projection) keep their own forward.

    def forward(self, x):
        if not self.enabled or getattr(self, "_fused", False):
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            # T-LoRA is training-only; inference always runs full rank.
            return org_forwarded + self._eval_delta(x, org_forwarded)

        if self._skip_module():
            return org_forwarded

        # Rank GEMMs run in the model compute dtype (org_forwarded.dtype), not
        # x.dtype — AdaLN's LayerNorm hands fp32 under autocast(bf16), which
        # would leave the rank path fp32 (OOMs in _rebalance). Pinned by
        # tests/test_lora_dtype_policy.py.
        work = org_forwarded.dtype
        x_lora = self._rebalance(x.to(work))
        lx = self._down(x_lora, work)
        lx = self._gate(lx, work)
        if self.dropout is not None:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx, scale = self._apply_rank_dropout(lx)
        lx = self._up(lx.to(work), work)
        return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)

    def _gate(self, lx: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        """Default T-LoRA gate: ``lx * mask`` (fp32 mask promotes ``lx``; ``_up``
        casts back to ``work``). Override to gate differently, e.g. OrthoInit's
        ``lambda_layer``."""
        return lx * self._timestep_mask

    def _down(self, x_lora: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _down"
        )

    def _up(self, lx: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _up"
        )

    def _eval_delta(self, x: torch.Tensor, org_forwarded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _eval_delta"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
