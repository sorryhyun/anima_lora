# Deprecated LoRA module variants
# These modules are kept for checkpoint compatibility but are no longer actively used.
# OrthoLoRA + postfix tuning supersedes DoRA with lower VRAM and better quality.

import random
import warnings

import torch

from networks.lora_modules import LoRAModule


class DoRAModule(LoRAModule):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.
    Decomposes pretrained weight into magnitude and direction, applies LoRA to the direction only.
    Reference: https://arxiv.org/abs/2402.09353

    DEPRECATED: Use OrthoLoRA + postfix tuning instead.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
    ):
        warnings.warn(
            "DoRAModule is deprecated. Use OrthoLoRA + postfix tuning instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            channel_scale=channel_scale,
        )

        # Initialize magnitude to per-row L2 norms of the original weight
        weight = org_module.weight.detach().float()
        self.magnitude = torch.nn.Parameter(weight.norm(p=2, dim=1))  # [out_features]

    def apply_to(self):
        # Compute and cache the frozen original weight row-norms before parent deletes org_module
        org_weight = self.org_module.weight.detach().float()
        self.register_buffer(
            "_org_weight_norm",
            org_weight.norm(p=2, dim=1).clamp(
                min=torch.finfo(org_weight.dtype).eps
            ),  # [out_features]
        )
        super().apply_to()

    def forward(self, x):
        org_out = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if random.random() < self.module_dropout:
                return org_out

        # per-channel input rebalancing (SmoothQuant-style, see LoRAModule.forward)
        x_lora = x * self.inv_scale if self._has_channel_scale else x

        # fp32 accumulation: compute LoRA delta in fp32 for better precision
        if self.fp32_accumulation:
            lx = torch.nn.functional.linear(
                x_lora.float(), self.lora_down.weight.float()
            )
        else:
            lx = self.lora_down(x_lora)

        # timestep-dependent rank masking
        if self._timestep_mask is not None and self.training:
            lx = lx * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        if self.fp32_accumulation:
            lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
            # Cast LoRA delta back to native dtype early
            lora_out = (lx * self.multiplier * scale).to(org_out.dtype)
        else:
            lx = self.lora_up(lx)
            lora_out = lx * self.multiplier * scale

        # DoRA: scale by (magnitude / ||W_original||)
        mag_scale = (self.magnitude / self._org_weight_norm).to(org_out.dtype)

        # Broadcast mag_scale to match output shape
        if len(org_out.shape) == 3:
            mag_scale = mag_scale.unsqueeze(0).unsqueeze(0)  # [1, 1, out_dim]
        elif len(org_out.shape) == 2:
            mag_scale = mag_scale.unsqueeze(0)  # [1, out_dim]

        return mag_scale * (org_out + lora_out)
