# Step-expert LoRA: shared down-projection + K up-heads selected by the
# diffusion step index (no learned router — the step counter is known at call
# time). Used by the turbo DP-DMD student when ``per_step_expert`` is on so
# head A serves step 0 (diversity) and head B serves step 1 (quality) without
# the two conflicting gradients fighting over one set of up-weights.
#
# Layout mirrors a plain ``LoRAModule`` (Linear-only) but with the up-projection
# replaced by a ``ModuleList`` of K heads off one shared ``lora_down``. Only the
# head at ``self._step`` contributes per forward, so per-step inference compute
# is identical to a single-head LoRA. Selection is a plain Python int attribute
# (not a tensor read) so ``torch.compile`` guards on it and specializes one
# graph per step value (K graphs) instead of forcing an ``.item()`` graph break.

import logging
import math

import torch

from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class StepExpertLoRAModule(BaseLoRAModule):
    """Shared-A, dual(-K)-B-head LoRA with hard step-index head selection.

    ``forward`` reads ``self._step`` (set by the network's
    ``set_step_index`` / coordinator's ``set_student_step``) and routes the
    bottleneck through ``self.lora_ups[self._step]``. ``lora_down`` is shared
    across heads — the common subspace both objectives train; only the up-heads
    specialize.

    Linear-only (the turbo student targets fused attention / MLP projections,
    all ``nn.Linear``). ``merge_to`` / ``fuse_weight`` are intentionally absent:
    K per-step heads cannot fold into one static DiT weight, so ``make merge``
    refuses per-step-expert turbo (see ``scripts/merge_to_dit.py``).
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
        # K up-heads off the single shared down-proj. zero-init every head so
        # ΔW = 0 at start (same invariant LoRAModule relies on); the head a step
        # routes to is the only one that ever receives gradient for that step.
        self.lora_ups = torch.nn.ModuleList(
            [torch.nn.Linear(self.lora_dim, out_dim, bias=False) for _ in range(self.K)]
        )

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        for up in self.lora_ups:
            torch.nn.init.zeros_(up.weight)

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        # Hard step-index selection. Plain Python int (not a buffer): the LoRA
        # forward is monkey-patched into the compiled DiT block, so reading a
        # tensor + ``.item()`` here would force a graph break. As a guarded int,
        # dynamo specializes one graph per distinct value — exactly the K we
        # cycle through (mirrors the 2-graphs-by-token-count compile story).
        self._step = 0

    def set_step(self, k: int) -> None:
        """Select the active up-head for subsequent forwards (0 <= k < K)."""
        if not (0 <= k < self.K):
            raise IndexError(
                f"step index {k} out of range for {self.K} heads ({self.lora_name})"
            )
        self._step = int(k)

    # Forward is the shared BaseLoRAModule scaffold; this class supplies the
    # shared-down / step-selected-up GEMMs and the eval delta. The active
    # up-head is ``self.lora_ups[self._step]`` (a guarded Python int, so Dynamo
    # specializes one graph per step value). T-LoRA gate is the inherited
    # default; dtype policy lives in the base.

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
