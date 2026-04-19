# LoRA module building blocks
# Extracted from lora_flux.py — generic LoRA, OrthoLoRA, and inference modules
#
# Reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import random
from typing import Optional

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
    ``s[c] ~ (mean|x[c]|)^alpha``. See ``bench/channel_dominance_analysis.md``.
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
        self._timestep_mask = None
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


def _sigma_sinusoidal_features(
    sigma: torch.Tensor, sigma_feature_dim: int
) -> torch.Tensor:
    """Sinusoidal σ features matching the DiT t_embedder functional form.

    Shared helper (also used by postfix-sigma, inlined there for historical
    self-containedness). Kept here so HydraLoRAModule / OrthoHydraLoRAExpModule
    can reuse the identical spectrum without cross-module coupling.
    """
    t = sigma.flatten().float()
    half_dim = sigma_feature_dim // 2
    exponent = (
        -math.log(10000)
        * torch.arange(half_dim, dtype=torch.float32, device=t.device)
        / max(half_dim, 1)
    )
    freqs = torch.exp(exponent)
    angles = t[:, None] * freqs[None, :]  # [B, half_dim]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class HydraLoRAModule(BaseLoRAModule):
    """
    HydraLoRA: MoE-style multi-head LoRA with layer-local routing.
    Shared lora_down captures common features; per-expert lora_up heads specialize.
    Each module owns its own router that reads the layer input and emits per-sample gates,
    so specialization is learned per-layer rather than globally.
    Reference: docs/methods/hydra-lora.md

    Optional σ-conditional routing (Track B, timestep-hydra.md): when
    ``sigma_feature_dim > 0``, a small 2-layer MLP maps sinusoidal(σ) to an
    additive bias on the gate logits. Zero-init on the final layer means
    training starts identical to base HydraLoRA; σ-dependence only emerges if
    gradients push it. ``|sigma_mlp[-1].weight|`` at convergence is a direct
    diagnostic of how much σ-conditioning was actually used.
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
        num_experts=4,
        channel_scale=None,
        sigma_feature_dim: int = 0,
        sigma_hidden_dim: int = 128,
    ):
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

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.num_experts = num_experts
        self.in_dim = in_dim

        # Shared down projection
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        # Fused per-expert up projections: (num_experts, out_dim, lora_dim)
        self.lora_up_weight = torch.nn.Parameter(
            torch.zeros(num_experts, out_dim, self.lora_dim)
        )

        # Local router: reads pooled layer input → per-sample expert gates.
        # Init is deliberately small (std=0.01) so initial gates are near-uniform.
        # Relying on xavier init here produces saturated softmax in bf16 when the
        # pooled input has DC-bias outlier channels (see bench/channel_dominance_analysis.md).
        self.router = torch.nn.Linear(in_dim, num_experts, bias=True)
        torch.nn.init.normal_(self.router.weight, std=0.01)
        torch.nn.init.zeros_(self.router.bias)

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        self.sigma_feature_dim = int(sigma_feature_dim)
        self.sigma_hidden_dim = int(sigma_hidden_dim)
        if self.sigma_feature_dim > 0:
            # σ-conditional router bias: sinusoidal(σ) -> 2-layer MLP -> E logits.
            # Zero-init the final layer so step 0 logits == base router output.
            self.sigma_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.sigma_feature_dim, self.sigma_hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.sigma_hidden_dim, num_experts, bias=False),
            )
            torch.nn.init.zeros_(self.sigma_mlp[-1].weight)
        else:
            self.sigma_mlp = None

        self._last_gate = None  # (B, num_experts), cached each forward for balance loss
        self._sigma = None  # (B,) σ tensor; set externally by LoRANetwork.set_sigma

    def _compute_gate(self, x_lora: torch.Tensor) -> torch.Tensor:
        """Pool layer input over sequence dim (if any), run router, softmax.

        Uses mean pool, not max pool: DiT layer inputs have DC-bias outlier
        channels with peak/mean ratios of 80–96× (see channel_dominance_analysis.md).
        Max pool would surface those outliers straight into the router and blow
        up softmax in bf16. Mean pool averages them out.

        When ``sigma_mlp`` is present and ``_sigma`` is set, adds a σ-conditional
        bias to the logits before softmax (zero at init → identity to base).
        """
        if x_lora.dim() >= 3:
            B = x_lora.shape[0]
            pooled = x_lora.reshape(B, -1, x_lora.shape[-1]).mean(dim=1)
        else:
            pooled = x_lora
        logits = self.router(pooled)  # (B, num_experts)
        if self.sigma_mlp is not None and self._sigma is not None:
            sigma_feat = _sigma_sinusoidal_features(
                self._sigma, self.sigma_feature_dim
            ).to(logits.dtype)
            logits = logits + self.sigma_mlp(sigma_feat)
        return torch.softmax(logits, dim=-1)

    def forward(self, x):
        # Policy: bf16 storage, fp32 for the bottleneck matmuls. See
        # LoRAModule.forward for rationale. Gate/router stays in autocast
        # dtype — softmax over num_experts is fine in bf16 with the
        # small-std router init.
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        # per-channel input rebalancing (SmoothQuant-style, see LoRAModule.forward)
        x_lora = self._rebalance(x)

        lx = torch.nn.functional.linear(
            x_lora.float(), self.lora_down.weight.float()
        )

        # timestep-dependent rank masking (T-LoRA compatibility)
        if self._timestep_mask is not None and self.training:
            lx = lx * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # Layer-local routing: gate is computed from this module's own input.
        gate = self._compute_gate(x_lora)  # (B, num_experts)
        if self.training:
            self._last_gate = gate  # cache for network-level balance loss

        # Gate-weighted combined weight per batch element: (B, out_dim, lora_dim)
        combined = torch.einsum(
            "be,eod->bod", gate.float(), self.lora_up_weight.float()
        )
        # Apply: lx is (B, ..., lora_dim), combined is (B, out_dim, lora_dim)
        orig_shape = lx.shape
        B = orig_shape[0]
        lx_3d = lx.reshape(B, -1, orig_shape[-1])  # (B, *, lora_dim)
        out = torch.bmm(lx_3d, combined.transpose(1, 2))  # (B, *, out_dim)
        out = out.reshape(*orig_shape[:-1], -1)  # restore prefix dims

        return org_forwarded + (out * self.multiplier * scale).to(
            org_forwarded.dtype
        )


class OrthoLoRAExpModule(BaseLoRAModule):
    """
    Experimental OrthoLoRA with Cayley parameterization + SVD-informed init.

    Instead of trainable (d, r) / (r, d) basis matrices with soft orthogonality
    regularization, uses frozen SVD-derived bases rotated by Cayley-parameterized
    (r, r) orthogonal matrices.  Trainable parameters are two skew-symmetric
    matrices S_p, S_q (r×r) and a diagonal scale vector lambda (1×r).

    Cayley transform: R = (I - A)(I + A)^{-1}, A = S - S^T (skew-symmetric).
    Guarantees R^T R = I at every gradient step — no reg hyperparameter.

    Forward:
        P_eff = P_basis @ cayley(S_p)        # (out, r) — columns stay orthonormal
        Q_eff = cayley(S_q) @ Q_basis        # (r, in) — rows stay orthonormal
        out   = x @ Q_eff^T @ diag(λ) @ P_eff^T

    Init: S_p = S_q = 0 → R = I, λ = 0 → ΔW = 0 (zero output at init).

    Reference: PSOFT (Wu et al., ICLR 2026) for Cayley + SVD-init idea.
    We keep frozen full-dim bases (not PSOFT's frozen principal-subspace restriction)
    so expressiveness is limited to rotations within the initial basis span —
    this is the tradeoff being benchmarked via use_ortho.
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

        # --- SVD-informed initialization ---
        # Extract top-r singular vectors from the pretrained weight. Randomized
        # SVD (torch.svd_lowrank) is ~10-100x faster than full SVD for r ≪ min(m,n)
        # and gives near-machine-precision on the top-r factors we keep.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        q = min(lora_dim + 6, min(W.shape))
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
        # U: (out, q), V: (in, q) — V is returned directly (not Vh).
        P_init = U[:, :lora_dim].clone().contiguous()  # (out, r)
        Q_init = V[:, :lora_dim].T.clone().contiguous()  # (r, in)
        del U, _S_vals, V, W

        # Frozen bases — define the subspace; Cayley rotates within it
        self.register_buffer("P_basis", P_init.cpu())  # (out_dim, r)
        self.register_buffer("Q_basis", Q_init.cpu())  # (r, in_dim)

        # Trainable skew-symmetric seeds (r × r).  Cayley(0) = I, so at init
        # P_eff = P_basis, Q_eff = Q_basis.
        self.S_p = torch.nn.Parameter(torch.zeros(lora_dim, lora_dim))
        self.S_q = torch.nn.Parameter(torch.zeros(lora_dim, lora_dim))

        # Diagonal scale — zero-init gives ΔW = 0 at init (standard LoRA convention)
        self.lambda_layer = torch.nn.Parameter(torch.zeros(1, lora_dim))

        # Per-channel input pre-scaling (SmoothQuant-style).
        # Absorb into Q_basis so the frozen path is rebalanced.
        self._register_channel_scale(self.Q_basis, channel_scale)

    # --- Cayley transform (exact inverse, r×r is tiny) ---
    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """Cayley transform: R = (I - A)(I + A)^{-1}, A = S - S^T."""
        A = S - S.T  # guaranteed skew-symmetric
        eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(eye + A, eye - A)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self._skip_module():
            return org_forwarded

        dtype = self.P_basis.dtype
        x_lora = self._rebalance(x.to(dtype))

        # Cayley-parameterized orthogonal rotations (r × r)
        R_q = self._cayley(self.S_q)  # (r, r)
        Q_eff = R_q @ self.Q_basis  # (r, in_dim)

        # timestep mask
        mask = self._timestep_mask

        # x @ Q_eff^T → (*, r), then scale by lambda
        lx = torch.nn.functional.linear(x_lora, Q_eff)  # (*, r)
        if mask is not None:
            lx = lx * self.lambda_layer * mask
        else:
            lx = lx * self.lambda_layer

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        R_p = self._cayley(self.S_p)  # (r, r)
        P_eff = self.P_basis @ R_p  # (out_dim, r)
        out = torch.nn.functional.linear(lx, P_eff)  # (*, out_dim)

        lora_out = out * self.multiplier * scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley guarantees orthogonality structurally."""
        zero = torch.tensor(0.0, device=self.S_p.device)
        return zero, zero


class OrthoHydraLoRAExpModule(BaseLoRAModule):
    """
    OrthoLoRAExp + HydraLoRA: Cayley-parameterized MoE LoRA.

    Shared down projection uses a Cayley-rotated SVD basis (frozen Q_basis +
    trainable S_q); per-expert up projections each rotate a shared frozen
    P_basis through independent Cayley parameters (S_p: num_experts × r × r).
    A shared lambda diagonal scales the bottleneck (zero-init → ΔW = 0).

    Cayley guarantees orthogonality at every step without regularization.
    Per-expert rotations keep experts in distinct output subspaces, complementing
    the balance loss that prevents utilization collapse.
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
        num_experts=4,
        channel_scale=None,
        sigma_feature_dim: int = 0,
        sigma_hidden_dim: int = 128,
    ):
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

        in_dim = org_module.in_features
        self.num_experts = num_experts
        self.in_dim = in_dim

        # --- SVD-informed init (same as OrthoLoRAExp) ---
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        q = min(lora_dim + 6, min(W.shape))
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
        P_init = U[:, :lora_dim].clone().contiguous()  # (out, r)
        Q_init = V[:, :lora_dim].T.clone().contiguous()  # (r, in)
        del U, _S_vals, V, W

        # Frozen bases — shared across experts
        self.register_buffer("P_basis", P_init.cpu())  # (out_dim, r)
        self.register_buffer("Q_basis", Q_init.cpu())  # (r, in_dim)

        # Shared Q rotation: Cayley(0) = I → Q_eff = Q_basis at init
        self.S_q = torch.nn.Parameter(torch.zeros(lora_dim, lora_dim))

        # Per-expert P rotations: each expert rotates shared output basis differently
        self.S_p = torch.nn.Parameter(torch.zeros(num_experts, lora_dim, lora_dim))

        # Shared diagonal scale — zero-init → ΔW = 0 at init
        self.lambda_layer = torch.nn.Parameter(torch.zeros(1, lora_dim))

        # Layer-local router (same as HydraLoRAModule)
        self.router = torch.nn.Linear(in_dim, num_experts, bias=True)
        torch.nn.init.normal_(self.router.weight, std=0.01)
        torch.nn.init.zeros_(self.router.bias)

        # Per-channel input pre-scaling (SmoothQuant-style)
        self._register_channel_scale(self.Q_basis, channel_scale)

        self.sigma_feature_dim = int(sigma_feature_dim)
        self.sigma_hidden_dim = int(sigma_hidden_dim)
        if self.sigma_feature_dim > 0:
            self.sigma_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.sigma_feature_dim, self.sigma_hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.sigma_hidden_dim, num_experts, bias=False),
            )
            torch.nn.init.zeros_(self.sigma_mlp[-1].weight)
        else:
            self.sigma_mlp = None

        self._last_gate = None  # cached each forward for balance loss
        self._sigma = None  # (B,) σ; set by LoRANetwork.set_sigma

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """Cayley transform: R = (I - A)(I + A)^{-1}, A = S - S^T.
        Supports both 2D (r, r) and batched 3D (E, r, r) input."""
        A = S - S.transpose(-2, -1)  # skew-symmetric
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)

    def _compute_gate(self, x_lora: torch.Tensor) -> torch.Tensor:
        """Pool layer input over sequence dim (if any), run router, softmax.

        σ-conditional bias added when ``sigma_mlp`` is present and ``_sigma``
        is set. See ``HydraLoRAModule._compute_gate`` for rationale.
        """
        if x_lora.dim() >= 3:
            B = x_lora.shape[0]
            pooled = x_lora.reshape(B, -1, x_lora.shape[-1]).mean(dim=1)
        else:
            pooled = x_lora
        logits = self.router(pooled)  # (B, num_experts)
        if self.sigma_mlp is not None and self._sigma is not None:
            sigma_feat = _sigma_sinusoidal_features(
                self._sigma, self.sigma_feature_dim
            ).to(logits.dtype)
            logits = logits + self.sigma_mlp(sigma_feat)
        return torch.softmax(logits, dim=-1)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        dtype = self.P_basis.dtype
        x_lora = self._rebalance(x.to(dtype))

        # Shared down: Cayley-parameterized Q
        R_q = self._cayley(self.S_q)  # (r, r)
        Q_eff = R_q @ self.Q_basis  # (r, in)

        lx = torch.nn.functional.linear(x_lora, Q_eff)  # (*, r)

        # Scale by lambda + timestep mask
        mask = self._timestep_mask
        if mask is not None:
            lx = lx * self.lambda_layer * mask
        else:
            lx = lx * self.lambda_layer

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # Per-expert up: batched Cayley on P rotations
        R_p = self._cayley(self.S_p)  # (E, r, r)
        # P_basis: (out, r) → (1, out, r); R_p: (E, r, r) → P_eff: (E, out, r)
        P_eff = self.P_basis.unsqueeze(0) @ R_p

        # Layer-local routing
        gate = self._compute_gate(x_lora)  # (B, E)
        if self.training:
            self._last_gate = gate

        # Gate-weighted combined P: (B, out, r)
        P_combined = torch.einsum("be,eor->bor", gate, P_eff)

        # Apply: lx (B, ..., r) × P_combined^T (B, r, out) → (B, ..., out)
        orig_shape = lx.shape
        B = orig_shape[0]
        lx_3d = lx.reshape(B, -1, orig_shape[-1])  # (B, *, r)
        out = torch.bmm(lx_3d, P_combined.transpose(1, 2))  # (B, *, out)
        out = out.reshape(*orig_shape[:-1], -1)

        lora_out = out * self.multiplier * scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley guarantees orthogonality structurally."""
        zero = torch.tensor(0.0, device=self.S_p.device)
        return zero, zero


class ReFTModule(torch.nn.Module):
    """
    LoReFT: Low-Rank Representation Fine-Tuning.
    Applies a learned low-rank subspace edit to the output representation:
        h_new = h + R^T(ΔW·h + b) * scale * multiplier
    where R is an orthogonal rotation selecting the intervention subspace and
    ΔW (``learned_source``) is the learned delta within that subspace. The
    paper's form ``(Wh + b) − Rh`` is algebraically identical under
    ``ΔW = W − R``; parameterizing ΔW directly avoids the activation-level
    cancellation, so the module runs in the ambient dtype (bf16 under mixed
    precision) without fp32 upcasts.

    Intervention target: the paper defines ReFT on the residual stream at
    specific layers (Wu et al., 2024 §3.3). ``org_module`` here is usually a
    DiT Block whose output is the block-level residual-stream hidden state;
    wrapping Blocks (not each internal Linear) keeps the parameter and
    activation budget aligned with the paper.

    Zero-init: learned_source is zero-initialized so delta=0 at init.
    Reference: Wu et al., "ReFT: Representation Finetuning for Language Models" (NeurIPS 2024)
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        embed_dim: Optional[int] = None,
        multiplier=1.0,
        reft_dim=4,
        alpha=1,
        dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if embed_dim is None:
            if hasattr(org_module, "out_features"):
                embed_dim = org_module.out_features
            else:
                raise ValueError(
                    "embed_dim must be provided when wrapping a non-Linear module "
                    f"(got {type(org_module).__name__})"
                )
        self.reft_dim = reft_dim

        # R: orthogonal rotation (projects to intervention subspace)
        self.rotate_layer = torch.nn.Linear(embed_dim, reft_dim, bias=False)
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        r_rand = torch.randn(embed_dim, reft_dim, device=init_device)
        r_orth, _ = torch.linalg.qr(r_rand)  # (embed_dim, reft_dim)
        self.rotate_layer.weight.data = r_orth.T.cpu().clone().contiguous()
        del r_rand, r_orth

        # ΔW: learned delta in R's subspace — zero-init gives delta=0 at step 0.
        self.learned_source = torch.nn.Linear(embed_dim, reft_dim)
        torch.nn.init.zeros_(self.learned_source.weight)
        torch.nn.init.zeros_(self.learned_source.bias)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = reft_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / reft_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.module_dropout = module_dropout

        self._timestep_mask = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, *args, **kwargs):
        # Works for wrapped Linear (forward(x)) and wrapped DiT Block
        # (forward(x_B_T_H_W_D, emb, crossattn, attn_params, rope, adaln_lora_3D)).
        h = self.org_forward(*args, **kwargs)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return h

        # ΔW·h + b in ambient dtype — no cancellation, no fp32 copy of h.
        # Last-dim linear broadcasts over any leading shape (B,L,D) or (B,T,H,W,D).
        delta = torch.nn.functional.linear(
            h, self.learned_source.weight, self.learned_source.bias
        )

        if self._timestep_mask is not None and self.training:
            delta = delta * self._timestep_mask

        if self.dropout is not None and self.training:
            delta = torch.nn.functional.dropout(delta, p=self.dropout)

        edit = torch.nn.functional.linear(delta, self.rotate_layer.weight.T)
        return h + edit * (self.multiplier * self.scale)

    def regularization(self):
        """Orthogonality regularization: ||R R^T - I||^2"""
        R = self.rotate_layer.weight  # (reft_dim, embed_dim)
        reg = torch.sum((R @ R.T - torch.eye(self.reft_dim, device=R.device)) ** 2)
        return reg


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
