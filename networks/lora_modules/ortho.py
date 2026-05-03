# OrthoLoRA variants: Cayley-parameterized orthogonal low-rank adapters,
# plus the OrthoHydra MoE combination.

from typing import List, Optional

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.custom_autograd import lora_down_project
from networks.lora_modules.hydra import (
    _apply_sigma_band_mask,
    _clear_sigma_feature_cache,
    _register_sigma_band_partition,
    _register_sigma_feature_cache,
    _set_sigma_feature_cache,
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

        # Opt-in: save bf16 x instead of retaining the fp32 upcast for backward.
        # Applies to the Q_eff projection only; the P_eff multiply stays on the
        # legacy path (its input ``lx`` is already rank-sized and cheap).
        self.use_custom_down_autograd = False

        # Pre-allocated identity for Cayley solves; allocating fresh `torch.eye`
        # in every forward emitted ~2 small kernels per module per step.
        self.register_buffer(
            "_eye_r",
            torch.eye(lora_dim, dtype=torch.float32),
            persistent=False,
        )

    # --- Cayley transform (exact inverse, r×r is tiny) ---
    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """Cayley transform: R = (I - A)(I + A)^{-1}, A = S - S^T.

        Kept for save-time SVD distillation (`networks/lora_save.py`); the
        forward now batches the S_q / S_p solves into a single call instead.
        """
        A = S - S.T  # guaranteed skew-symmetric
        eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(eye + A, eye - A)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self._skip_module():
            return org_forwarded

        # Batched Cayley: stack S_q and S_p into one (2, r, r) solve. Halves
        # the LU/TRSM launch count vs. solving them separately.
        skew = torch.stack([self.S_q, self.S_p])  # (2, r, r)
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)  # (2, r, r)
        R_q, R_p = R[0], R[1]
        Q_eff = R_q @ self.Q_basis  # (r, in_dim)

        # x @ Q_eff^T → (*, r), then scale by lambda
        if self.use_custom_down_autograd and self.training:
            inv_scale = self.inv_scale if self._has_channel_scale else None
            lx = lora_down_project(x, Q_eff, inv_scale)  # (*, r)
        else:
            dtype = self.P_basis.dtype
            x_lora = self._rebalance(x.to(dtype))
            lx = torch.nn.functional.linear(x_lora, Q_eff)  # (*, r)
        # timestep mask: always a Tensor (default all-ones → identity), so no
        # None-vs-Tensor guard fires under compile.
        lx = lx * self.lambda_layer * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # R_p was computed in the batched Cayley solve above.
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
    OrthoLoRAExp + HydraLoRA: Cayley-parameterized MoE LoRA with
    disjoint per-expert output subspaces.

    Shared down projection uses a Cayley-rotated SVD basis (frozen Q_basis +
    trainable S_q). Up projections take the top-``E*r`` singular vectors of
    the pretrained weight and partition them into ``E`` disjoint slices of
    ``r`` columns each — every expert owns its own orthonormal output basis
    ``P_bases[e]: (out, r)``, rotated within its slice by a per-expert Cayley
    ``R_p[e]``. Because the SVD columns are mutually orthonormal, experts are
    **structurally orthogonal in output space** (``P_bases[i]^T P_bases[j] = 0``
    for ``i ≠ j``), regardless of training dynamics.

    Why disjoint slices (not a shared P_basis + per-expert rotation): with a
    shared basis every ``P_eff[e]`` lives in the same rank-``r`` column span,
    so ``P_eff[i]^T P_eff[j] = R_p[i]^T R_p[j]`` is an orthogonal matrix — it
    cannot be zero. Experts would differ only by an r×r rotation inside a
    shared subspace, giving the router near-identical ``score_e`` for all
    experts and no gradient to differentiate them (MoE cold-start deadlock
    plus σ-blindness; see bench results 2026-04-21). Disjoint slices make
    ``score_e`` genuinely different because each expert writes into a
    distinct output subspace, which is what breaks the deadlock.

    Fallback: if ``min(out_dim, in_dim) < num_experts * lora_dim`` we cannot
    partition disjointly, so ``P_bases`` degenerates to the legacy shared
    ``P_basis`` replicated ``E`` times (warning logged). In that case all
    experts start identical (shared basis + zero ``S_p`` + zero
    ``lambda_layer``); ``expert_warmup_ratio`` is the only symmetry-breaker —
    do not run narrow-layer Hydra with ``expert_warmup_ratio=0``.
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
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: int = 1,
        sigma_bucket_boundaries: Optional[List[float]] = None,
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

        # --- SVD-informed init with disjoint per-expert output slices ---
        # Take the top (E*r) singular vectors and partition the U columns into
        # E slices of r. Each slice is orthonormal and mutually orthogonal to
        # every other — experts write into structurally-disjoint subspaces.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        target_cols = num_experts * lora_dim
        max_cols = min(W.shape)
        disjoint = target_cols <= max_cols
        q = min(target_cols + 6, max_cols) if disjoint else min(lora_dim + 6, max_cols)
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
        Q_init = V[:, :lora_dim].T.clone().contiguous()  # (r, in) — shared
        if disjoint:
            # (out, E*r) -> (out, E, r) -> (E, out, r)
            P_stack = U[:, :target_cols].reshape(out_dim, num_experts, lora_dim)
            P_bases_init = P_stack.permute(1, 0, 2).clone().contiguous()
        else:
            # Fallback: not enough singular directions to disjoint-partition.
            # Replicate the top-r slice across experts (legacy behavior).
            # Symmetry-breaking falls to expert_warmup_ratio.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"{lora_name}: min(out={out_dim}, in={in_dim})={max_cols} < "
                f"num_experts*lora_dim={target_cols}; falling back to shared "
                "P_basis (experts rely on expert_warmup_ratio for differentiation)."
            )
            P_shared = U[:, :lora_dim].clone().contiguous()  # (out, r)
            P_bases_init = P_shared.unsqueeze(0).expand(
                num_experts, -1, -1
            ).contiguous()
        del U, _S_vals, V, W

        # Frozen bases — per-expert disjoint P_bases, shared Q_basis.
        self.register_buffer("P_bases", P_bases_init.cpu())  # (E, out_dim, r)
        self.register_buffer("Q_basis", Q_init.cpu())  # (r, in_dim)
        self._disjoint_basis = disjoint

        # Shared Q rotation: Cayley(0) = I → Q_eff = Q_basis at init
        self.S_q = torch.nn.Parameter(torch.zeros(lora_dim, lora_dim))

        # Per-expert P rotations: each expert rotates its own output basis.
        # With disjoint P_bases, zero-init S_p still yields E distinct P_eff
        # slices (one per basis partition) — no manual symmetry-breaking needed.
        # In the narrow-layer fallback, expert_warmup_ratio carries that role.
        self.S_p = torch.nn.Parameter(torch.zeros(num_experts, lora_dim, lora_dim))

        # Shared diagonal scale — zero-init → ΔW = 0 at init
        self.lambda_layer = torch.nn.Parameter(torch.zeros(1, lora_dim))

        # Layer-local router (same as HydraLoRAModule): reads the pooled rank-R
        # signal (post Q_eff projection, pre-λ) concatenated with sinusoidal(σ)
        # when σ routing is enabled. See HydraLoRAModule.__init__ for the full
        # rationale on direct-input σ vs additive-bias sigma_mlp.
        self.sigma_feature_dim = int(sigma_feature_dim)
        self.sigma_hidden_dim = int(sigma_hidden_dim)  # unused; kept for API compat
        router_in_dim = lora_dim + self.sigma_feature_dim
        self.router = torch.nn.Linear(router_in_dim, num_experts, bias=True)
        with torch.no_grad():
            self.router.weight.zero_()
            torch.nn.init.normal_(self.router.weight[:, :lora_dim], std=0.01)
            self.router.bias.zero_()

        # Per-channel input pre-scaling (SmoothQuant-style)
        self._register_channel_scale(self.Q_basis, channel_scale)

        # Opt-in: save bf16 x instead of retaining the fp32 upcast for backward.
        # Applies to the shared Q_eff projection; router + P_eff paths are
        # rank/expert-sized and stay on the legacy path.
        self.use_custom_down_autograd = False

        self._last_gate = None  # cached each forward for balance loss
        # σ tensor; always a Tensor (never None) so the sinusoidal branch in
        # _compute_gate can run unconditionally. Registered as a non-persistent
        # buffer so .to(device) moves the placeholder with the module.
        # See ``HydraLoRAModule`` for the None-vs-Tensor guard rationale.
        _register_sigma_feature_cache(self, self.sigma_feature_dim)
        # Hard σ-band expert partition (see HydraLoRAModule for rationale).
        self._sigma_band_partition: bool = bool(specialize_experts_by_sigma_buckets)
        if self._sigma_band_partition:
            _register_sigma_band_partition(
                self, num_experts, num_sigma_buckets, sigma_bucket_boundaries
            )
        # Expert-warmup gradient masking. See HydraLoRAModule for full
        # rationale — for OrthoHydra the mask gates gradient into S_p (which
        # parameterises per-expert P rotations). Default all-ones → applied
        # unconditionally, collapses to identity outside warmup. No Python-
        # bool guard means no dynamo recompile at the warmup transition.
        self.register_buffer(
            "_expert_grad_mask",
            torch.ones(num_experts, dtype=torch.float32),
            persistent=False,
        )

        # Pre-allocated identity for Cayley solves; allocating fresh `torch.eye`
        # per forward emitted ~2 small kernels per module per step.
        self.register_buffer(
            "_eye_r",
            torch.eye(lora_dim, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """Cayley transform: R = (I - A)(I + A)^{-1}, A = S - S^T.

        Supports both 2D (r, r) and batched 3D (E, r, r) input. Kept for
        save-time SVD distillation (`networks/lora_save.py`); the forward
        batches the S_q / S_p solves into a single call instead.
        """
        A = S - S.transpose(-2, -1)  # skew-symmetric
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)

    def _compute_gate(self, lx: torch.Tensor) -> torch.Tensor:
        """Pool rank-R signal over sequence dim, optionally concat sinusoidal(σ),
        run router, softmax.

        RMS pool over the post-``Q_eff`` activations (pre-λ, pre-mask). λ is
        zero-init, so pooling the post-λ signal would zero the router input at
        step 0 and freeze gradient. See ``HydraLoRAModule._compute_gate`` for
        the rationale on direct-input σ routing and the always-a-Tensor
        ``_sigma`` pattern.
        """
        if lx.dim() >= 3:
            B = lx.shape[0]
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        pooled = pooled.to(self.router.weight.dtype)
        if self.sigma_feature_dim > 0:
            sigma_feat = self._sigma_features.to(pooled.dtype)
            sigma_feat = sigma_feat.expand(pooled.shape[0], -1)
            router_in = torch.cat([pooled, sigma_feat], dim=-1)
        else:
            router_in = pooled
        logits = self.router(router_in)  # (B, num_experts)
        if self._sigma_band_partition:
            logits = _apply_sigma_band_mask(
                logits, self._sigma, self._expert_band, self._sigma_edges
            )
        return torch.softmax(logits, dim=-1)

    def set_sigma(
        self, sigmas: torch.Tensor, sigma_features: torch.Tensor | None = None
    ) -> None:
        _set_sigma_feature_cache(self, sigmas, sigma_features)

    def clear_sigma(self) -> None:
        _clear_sigma_feature_cache(self)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        # Expert-warmup masking — see HydraLoRAModule.forward for rationale.
        # Applied unconditionally: outside warmup the mask is all-ones and
        # ``S_p*1 + S_p.detach()*0`` collapses to ``S_p`` (autograd-equivalent),
        # so no Python-bool guard is needed. Computed early so S_p_eff joins
        # the batched Cayley solve below.
        expert_mask = self._expert_grad_mask.to(self.S_p.dtype).view(-1, 1, 1)
        S_p_eff = self.S_p * expert_mask + self.S_p.detach() * (1.0 - expert_mask)

        # Batched Cayley: stack S_q with S_p_eff into one (E+1, r, r) solve.
        # Single LU + TRSM launch covers both the shared down rotation and
        # all per-expert up rotations, instead of two separate solves.
        skew = torch.cat([self.S_q.unsqueeze(0), S_p_eff], dim=0)  # (E+1, r, r)
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)  # (E+1, r, r)
        R_q = R[0]  # (r, r)
        R_p = R[1:]  # (E, r, r)
        Q_eff = R_q @ self.Q_basis  # (r, in)

        if self.use_custom_down_autograd and self.training:
            inv_scale = self.inv_scale if self._has_channel_scale else None
            lx = lora_down_project(x, Q_eff, inv_scale)  # (*, r)
        else:
            dtype = self.P_bases.dtype
            x_lora = self._rebalance(x.to(dtype))
            lx = torch.nn.functional.linear(x_lora, Q_eff)  # (*, r)

        # Layer-local routing from raw rank-R signal, before λ scaling /
        # timestep masking / dropout. λ is zero-init so post-λ lx carries no
        # signal to the router at step 0; pooling pre-λ keeps the gradient
        # flowing into the router from the start.
        gate = self._compute_gate(lx)  # (B, E)
        if self.training:
            # Plain STORE_ATTR inline (not a @compiler.disable helper) —
            # see ``HydraLoRAModule.forward`` for the AOT-autograd memory
            # rationale (disabled helper ≡ graph break per module ≡ OOM
            # under compile_mode=full).
            self._last_gate = gate

        # Scale by lambda + timestep mask. Mask is always a Tensor (default
        # all-ones buffer → identity), so no None-vs-Tensor guard fires.
        lx = lx * self.lambda_layer * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # R_p was computed in the batched Cayley solve above.
        # P_bases: (E, out, r), R_p: (E, r, r) → P_eff: (E, out, r)
        P_eff = self.P_bases @ R_p

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
