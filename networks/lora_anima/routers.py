# Network-level routers for the LoRA-family MoE layouts.
#
# Three two-layer MLP routers, all firing once per step and broadcasting their
# gates into every routing-aware module's shared buffer (see LoRANetwork's
# set_*_routing_weights). Also re-exported from network.py.
#
# Shared contract across all three:
#   * fp32 compute is load-bearing — bf16 logits + softmax(logits/τ) underflow
#     at small τ. Inference casts the parent LoRANetwork to bf16, so each
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
    uniform; combined with zero-init expert ups (free mode) / ``lambda_layer``
    (ortho mode) this guarantees ΔW=0 at the first optimizer step. Owned by
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


class FreqRouter(torch.nn.Module):
    """ChimeraHydra freq-pool router (one per network).

    Two-layer MLP → softmax/τ over the ``K_f`` freq experts. Input is
    ``concat(FEI(z_t), sinusoidal-σ-features)``; broadcasts ``π_f`` to every
    chimera module's ``_freq_routing_weights`` (grad_fn preserved) so
    ``∂L_denoise/∂π_f`` reaches the router (eq. 6-7, 11).

    Output layer is ``N(0, init_std)``. Without a centered gate a zero-init
    freq router is a fixed point (uniform gates, zero router gradient);
    ``LoRANetwork`` builds it with ``init_std=0`` because chimera's centered
    gate breaks that symmetry. When both ``fei_dim`` and ``sigma_dim`` are
    > 0, per-modality parameterless LayerNorm balances their differing
    per-channel variance budgets before the MLP.
    """

    def __init__(
        self,
        input_dim: int,
        num_freq_experts: int,
        *,
        hidden_dim: int = 32,
        tau: float = 1.0,
        init_std: float = 0.1,
        fei_dim: int = 0,
        sigma_dim: int = 0,
        apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"FreqRouter: input_dim must be > 0, got {input_dim}")
        if num_freq_experts <= 1:
            raise ValueError(
                f"FreqRouter: num_freq_experts must be > 1, got {num_freq_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_freq_experts = int(num_freq_experts)
        self.tau = float(tau)
        self.fei_dim = int(fei_dim)
        self.sigma_dim = int(sigma_dim)
        # LN only fires when both modalities are present (variance balance across
        # the concat) and the dims sum to input_dim (guards the rebuild path).
        self.apply_layer_norm = bool(apply_layer_norm) and (
            self.fei_dim > 0
            and self.sigma_dim > 0
            and self.fei_dim + self.sigma_dim == self.input_dim
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_freq_experts),
        )
        with torch.no_grad():
            # Output layer only (see class docstring).
            torch.nn.init.normal_(self.net[-1].weight, std=float(init_std))
            torch.nn.init.zeros_(self.net[-1].bias)

        self.ln_fei: Optional[torch.nn.LayerNorm] = None
        self.ln_sigma: Optional[torch.nn.LayerNorm] = None
        if self.apply_layer_norm:
            self.ln_fei = torch.nn.LayerNorm(self.fei_dim, elementwise_affine=False)
            self.ln_sigma = torch.nn.LayerNorm(self.sigma_dim, elementwise_affine=False)

        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
        x32 = x.float()
        if self.apply_layer_norm:
            fei_part = self.ln_fei(x32[..., : self.fei_dim])
            sigma_part = self.ln_sigma(
                x32[..., self.fei_dim : self.fei_dim + self.sigma_dim]
            )
            x32 = torch.cat([fei_part, sigma_part], dim=-1)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        return gates


class ContentRouter(torch.nn.Module):
    """ChimeraHydra content-pool router, network-level (one per network).

    Same ``Linear → SiLU → Linear → softmax/τ`` shape as FreqRouter, but the
    input is a pooled ``crossattn_emb`` (per-sample text features). Output
    ``π_c`` is broadcast to every chimera module's ``_content_routing_weights``
    (grad_fn preserved) — the only source of π_c. Output is zero-init by
    default (``content_router_init_std=0.0``): the
    content pool's disjoint ``P_bases_c·λ_c`` residual breaks symmetry under
    the always-on centered gate, so uniform π_c at step 0 keeps ΔW_c=0 while
    the router still gets gradient. Parameterless input LN normalizes the
    wide per-channel variance of the pooled T5-space vector.
    """

    def __init__(
        self,
        input_dim: int,
        num_content_experts: int,
        *,
        hidden_dim: int = 64,
        tau: float = 1.0,
        init_std: float = 0.1,
        apply_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"ContentRouter: input_dim must be > 0, got {input_dim}")
        if num_content_experts <= 1:
            raise ValueError(
                f"ContentRouter: num_content_experts must be > 1, got {num_content_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_content_experts = int(num_content_experts)
        self.tau = float(tau)
        self.apply_layer_norm = bool(apply_layer_norm)
        self.ln_in: Optional[torch.nn.LayerNorm] = (
            torch.nn.LayerNorm(self.input_dim, elementwise_affine=False)
            if self.apply_layer_norm
            else None
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_content_experts),
        )
        with torch.no_grad():
            torch.nn.init.normal_(self.net[-1].weight, std=float(init_std))
            torch.nn.init.zeros_(self.net[-1].bias)

        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
            if self.ln_in is not None:
                self.ln_in.float()
        x32 = x.float()
        if x32.dim() == 3:  # raw crossattn_emb: RMS-pool over seq → (B, D)
            x32 = x32.pow(2).mean(dim=1).sqrt()
        if self.ln_in is not None:
            x32 = self.ln_in(x32)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        return gates
