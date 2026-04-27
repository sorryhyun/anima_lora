# Classic LoRA — single class for both training and inference, with merge
# (per-LoRA state-dict slice into the base weight) and fuse (delta baked into
# the base weight at runtime, forward becomes a no-op) helpers.

import math

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.custom_autograd import lora_down_project


class LoRAModule(BaseLoRAModule):
    supports_conv2d = True

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
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv2d(
                self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
            )
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        # Opt-in: save bf16 x instead of retaining fp32 x_lora for backward.
        # Linear-only (Conv2d takes the legacy path). Set externally by the
        # network factory when use_custom_down_autograd is enabled.
        self.use_custom_down_autograd = False

        # Held in a list so PyTorch's nn.Module __setattr__ does not register
        # org_module as a submodule (would double-count params and pollute
        # state_dict). apply_to() in BaseLoRAModule deletes self.org_module
        # after rerouting forward, so this list is the only handle the LoRA
        # object retains for fuse/unfuse.
        self.org_module_ref = [org_module]
        self._fused = False

    def forward(self, x):
        # Inference fast path: delta is fused into the base weight, or the
        # adapter has been disabled — skip the LoRA branch entirely.
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            # Inference: native-dtype matmuls via the nn.Linear/Conv2d wrappers.
            x_lora = self._rebalance(x)
            lx = self.lora_up(self.lora_down(x_lora))
            return org_forwarded + lx * self.multiplier * self.scale

        # Training. Policy: bf16 storage, fp32 for the bottleneck matmuls. The
        # down-proj accumulates over embed_dim (large) and the up-proj output
        # is added back to the bf16 base; running both matmuls in fp32 recovers
        # mantissa precision that bf16 would shed.
        if self._skip_module():
            return org_forwarded

        # Per-channel input rebalancing (SmoothQuant-style). Absorbed scale is
        # baked into lora_down at init; we divide x here so the net forward is
        # unchanged but per-column gradients are balanced.
        if self.use_custom_down_autograd and isinstance(
            self.lora_down, torch.nn.Linear
        ):
            inv_scale = self.inv_scale if self._has_channel_scale else None
            lx = lora_down_project(x, self.lora_down.weight, inv_scale)
        else:
            x_lora = self._rebalance(x)
            lx = torch.nn.functional.linear(
                x_lora.float(), self.lora_down.weight.float()
            )

        # Timestep-dependent rank masking. Mask is always a Tensor (default
        # all-ones buffer → identity); LoRANetwork.set_timestep_mask rebinds
        # it to a shared live-updated mask when T-LoRA is active. No None-vs-
        # Tensor guard means no recompile under compile_mode=full.
        lx = lx * self._timestep_mask

        if self.dropout is not None:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
        return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)

    def get_weight(self, multiplier=None):
        """Return the LoRA delta as a tensor matching org_module.weight shape."""
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # Undo per-channel absorption so the merged weight is equivalent to the
        # LoRA delta applied to raw (unscaled) inputs.
        if self._has_channel_scale and down_weight.dim() == 2:
            down_weight = down_weight * self.inv_scale.to(down_weight).unsqueeze(0)

        if len(down_weight.size()) == 2:
            weight = multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            weight = (
                multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight
            ).permute(1, 0, 2, 3)
            weight = multiplier * conved * self.scale

        return weight

    def merge_to(self, sd, dtype, device):
        """Merge a per-LoRA state-dict slice into org_module.weight in-place.

        Used by the alternative-to-apply_to inference path: deltas land directly
        on the base model's Linear/Conv2d weight, no forward hooks. The sd
        parameter holds the raw checkpoint slice because the LoRA modules
        themselves haven't been load_state_dict'd at this point.
        """
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()

            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            # Undo per-channel absorption before merging into the base weight so
            # that the merged forward (no x rebalancing) produces the same output.
            if "inv_scale" in sd:
                inv_scale = sd["inv_scale"].to(torch.float).to(device)
                if down_weight.dim() == 2:
                    down_weight = down_weight * inv_scale.unsqueeze(0)

            if len(w.size()) == 2:
                w += self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                w += (
                    self.multiplier
                    * (
                        up_weight.squeeze(3).squeeze(2)
                        @ down_weight.squeeze(3).squeeze(2)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)
                w += self.multiplier * conved * self.scale

            weight.data.copy_(w.to(dtype))

    def fuse_weight(self):
        """Bake LoRA delta into org_module.weight; subsequent forwards no-op."""
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        """Subtract a previously fused LoRA delta back out of org_module.weight."""
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False
