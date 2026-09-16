# Step-expert LoRA: shared down-projection + K up-heads selected by the
# diffusion step index (no learned router). Used by the turbo DP-DMD student
# when ``per_step_expert`` is on (e.g. head A serves step 0, head B step 1).
# Selection is a plain Python int, not a buffer/tensor —
# torch.compile guards on it and specializes one graph per step value (K
# graphs) instead of forcing an ``.item()`` graph break.

import logging
import math

import torch

from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class StepExpertLoRAModule(BaseLoRAModule):
    """Shared-A, dual(-K)-B-head LoRA with hard step-index head selection.

    ``forward`` reads ``self._step`` (set by the network's
    ``set_step_index`` / coordinator's ``set_student_step``) and routes through
    ``self.lora_ups[self._step]``; ``lora_down`` is shared across heads.
    Linear-only. No ``merge_to`` / ``fuse_weight`` — K per-step heads can't
    fold into one static DiT weight, so ``make merge`` refuses per-step-expert
    turbo.
    """

    supports_conv2d = False

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
        step_expert_K: int = 2,
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
            raise ValueError("StepExpertLoRAModule supports Linear only")

        self.K = int(step_expert_K)
        if self.K < 1:
            raise ValueError(f"step_expert_K must be >= 1, got {self.K}")

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        # K up-heads off the shared down-proj, zero-init so ΔW=0 at start.
        self.lora_ups = torch.nn.ModuleList(
            [torch.nn.Linear(self.lora_dim, out_dim, bias=False) for _ in range(self.K)]
        )

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        for up in self.lora_ups:
            torch.nn.init.zeros_(up.weight)

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        self._step = 0  # plain int, not a buffer — see module header

    def set_step(self, k: int) -> None:
        """Select the active up-head for subsequent forwards (0 <= k < K)."""
        if not (0 <= k < self.K):
            raise IndexError(
                f"step index {k} out of range for {self.K} heads ({self.lora_name})"
            )
        self._step = int(k)

    # Forward is the BaseLoRAModule scaffold; this class supplies only the
    # shared-down / step-selected-up GEMMs and the eval delta.

    def _down(self, x_lora, work):
        return torch.nn.functional.linear(x_lora, self.lora_down.weight.to(work))

    def _up(self, lx, work):
        up = self.lora_ups[self._step]
        return torch.nn.functional.linear(lx, up.weight.to(work))

    def _eval_delta(self, x, org_forwarded):
        up = self.lora_ups[self._step]
        x_lora = self._rebalance(x)
        lx = up(self.lora_down(x_lora))
        return lx * self.multiplier * self.scale
