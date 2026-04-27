# Shared scaffolding for LoRA-family modules.
#
# Reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import random

import torch
from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _absorb_channel_scale(
    weight: torch.Tensor, channel_scale: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Absorb per-channel input scale into a Linear weight's input columns.

    Given `weight` of shape ``[out, in]`` and `channel_scale` of shape ``[in]``
    (already post-alpha — the caller is responsible for the ``.pow(alpha)``),
    mutate `weight` so ``W[:, c] *= s_norm[c]`` where ``s_norm = s / s.mean()``,
    and return ``inv_scale = 1 / s_norm``. At forward time callers must apply
    ``x * inv_scale`` before multiplying by the absorbed weight, which preserves
    the original output exactly but rebalances the per-column gradient magnitudes.

    Rationale: with SmoothQuant-style absorption, the gradient of the down
    projection's column ``c`` goes from being proportional to ``|x[c]|^2`` to
    proportional to ``|x[c] / s[c]|^2`` — uniform across channels when
    ``s[c] ~ (mean|x[c]|)^alpha``. See ``archive/bench/channel_dominance_analysis.md``.
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
    return (1.0 / s).contiguous()


class BaseLoRAModule(torch.nn.Module):
    """Shared scaffolding for LoRA-family modules.

    Centralizes the parts every variant implements identically: alpha→scale
    normalization, multiplier, dropout/module_dropout/rank_dropout bookkeeping,
    channel_scale absorption, timestep masking, and ``apply_to`` monkey-patching.
    Subclasses still own their own ``forward`` but call the helpers here to
    avoid re-implementing the dropout/rebalance/rank-dropout boilerplate.
    """

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
        # Default timestep-rank mask: a shape-(1, lora_dim) all-ones buffer.
        # Multiplying by ones is a no-op, so every LoRA-family forward can
        # apply ``lx * self._timestep_mask`` unconditionally — no None-vs-
        # Tensor guard fires under ``compile_mode=full``. When T-LoRA is
        # active, ``LoRANetwork.set_timestep_mask`` reassigns this buffer to
        # a shared live-updated mask (see network.py); ``clear_timestep_mask``
        # fills the shared mask with ones to restore the neutral state.
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
        """Absorb ``channel_scale`` into ``target_weight`` in-place and register
        ``inv_scale`` as a persistent buffer for use at forward time."""
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
        """True if ``module_dropout`` fires for this step — caller should return
        the untouched org output."""
        return (
            self.module_dropout is not None
            and self.training
            and random.random() < self.module_dropout
        )

    def _rebalance(self, x: torch.Tensor) -> torch.Tensor:
        """SmoothQuant-style input rebalancing — no-op when not calibrated."""
        return x * self.inv_scale if self._has_channel_scale else x

    def _apply_rank_dropout(self, lx: torch.Tensor):
        """Apply rank dropout to the rank-r intermediate and return (lx, scale).
        Returns ``self.scale`` unchanged when rank_dropout is disabled."""
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask
            # scaling for rank dropout: treat as if the rank is changed
            return lx, self.scale * (1.0 / (1.0 - self.rank_dropout))
        return lx, self.scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
