# Classic LoRA and the inference-only subclass with merge/fuse support.

import math

import torch

from networks.lora_modules.base import BaseLoRAModule


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
        """
        if alpha == 0 or None, alpha is rank (no scaling).
        """
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

    def forward(self, x):
        # Policy: bf16 storage, fp32 for the bottleneck matmuls. The down-proj
        # accumulates over embed_dim (large) and the up-proj output is added
        # back to the bf16 base; running both matmuls in fp32 recovers mantissa
        # precision that bf16 would shed. Activation copies are transient.
        org_forwarded = self.org_forward(x)

        if self._skip_module():
            return org_forwarded

        # per-channel input rebalancing (SmoothQuant-style). Absorbed scale is
        # baked into lora_down at init; we divide x here so the net forward is
        # unchanged but per-column gradients are balanced.
        x_lora = self._rebalance(x)

        lx = torch.nn.functional.linear(
            x_lora.float(), self.lora_down.weight.float()
        )

        # timestep-dependent rank masking
        if self._timestep_mask is not None and self.training:
            lx = lx * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
        return org_forwarded + (lx * self.multiplier * scale).to(
            org_forwarded.dtype
        )


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        channel_scale=None,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            channel_scale=channel_scale,
        )

        self.org_module_ref = [org_module]
        self.enabled = True
        self.network = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device):
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

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # Undo per-channel absorption so the merged weight is equivalent to the
        # LoRA delta applied to raw (unscaled) inputs.
        if self._has_channel_scale and down_weight.dim() == 2:
            down_weight = down_weight * self.inv_scale.to(down_weight).unsqueeze(0)

        if len(down_weight.size()) == 2:
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight
            ).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def default_forward(self, x):
        x_lora = x * self.inv_scale if self._has_channel_scale else x
        lx = self.lora_down(x_lora)
        lx = self.lora_up(lx)
        return self.org_forward(x) + lx * self.multiplier * self.scale

    def fuse_weight(self):
        """Merge LoRA delta into org_module weight. Forward becomes a no-op (just org_forward)."""
        if getattr(self, "_fused", False):
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        """Remove LoRA delta from org_module weight."""
        if not getattr(self, "_fused", False):
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        return self.default_forward(x)
