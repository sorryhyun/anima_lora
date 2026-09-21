# Network-level router for the shared-A Hydra layout.
#
# A two-layer MLP router firing once per step and broadcasting its gates into
# every routing-aware module's shared buffer (see LoRANetwork's
# set_routing_weights). Also re-exported from network.py.
#
# Contract:
#   * fp32 compute is load-bearing — bf16 logits + softmax(logits/τ) underflow
#     at small τ. Inference casts the parent LoRANetwork to bf16, so the
#     forward re-pins the router weights to fp32 on first use.
#   * (B, L, D) inputs are RMS-pooled over the sequence axis to (B, D).
#   * parameterless LayerNorm (elementwise_affine=False) keeps the state_dict
#     free of ln_* keys, so on/off is deterministic from cfg with no metadata
#     stamp for the LN tensors themselves.

from typing import Optional

import torch

# Post-LLM-adapter crossattn_emb width, fixed by the Anima DiT
# (``crossattn_emb_channels = 1024`` in ``library/anima/models.py``).
CROSSATTN_EMB_DIM: int = 1024


class GlobalRouter(torch.nn.Module):
    """Single network-level router feeding every routing-aware module.

    Two-layer MLP → softmax/τ. Final layer is zero-init so step-0 gates are
    uniform; combined with zero-init expert ups this guarantees ΔW=0 at the
    first optimizer step. Owned by
    ``LoRANetwork`` when ``cfg.route_per_layer=False`` and ``cfg.use_moe_style``
    selects an MoE layout; reads the per-step signal via ``set_fei`` /
    ``set_sigma`` and broadcasts gates ``(B, E)`` through
    ``LoRANetwork.set_routing_weights``. ``_last_gates`` / ``_last_input``
    (detached, per-forward) feed the metrics layer; ``apply_layer_norm`` is
    used by the ``crossattn_emb`` source.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 64,
        tau: float = 0.7,
        apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"GlobalRouter: input_dim must be > 0, got {input_dim}")
        if num_experts <= 1:
            raise ValueError(
                f"GlobalRouter: num_experts must be > 1, got {num_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.tau = float(tau)
        self.apply_layer_norm = bool(apply_layer_norm)
        self.ln_in: Optional[torch.nn.LayerNorm] = (
            torch.nn.LayerNorm(self.input_dim, elementwise_affine=False)
            if self.apply_layer_norm
            else None
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_experts),
        )
        # Uniform-at-init: zero the output layer so softmax(0/τ) = 1/E.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

        # Per-step diagnostics (overwritten + detached each forward); _last_fei
        # aliases _last_input under the FEI source.
        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None
        self._last_fei: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
            if self.ln_in is not None:
                self.ln_in.float()
        x32 = x.float()
        if x32.dim() == 3:  # crossattn_emb source: pool (B, L, D) → (B, D)
            x32 = x32.pow(2).mean(dim=1).sqrt()
        if self.ln_in is not None:
            x32 = self.ln_in(x32)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        self._last_fei = self._last_input
        return gates
