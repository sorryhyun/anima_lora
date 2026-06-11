# OrthoLoRA variants: Cayley-parameterized orthogonal low-rank adapters
# (OrthoLoRA + OrthoHydra MoE), plus OrthoInit — top-r SVD of W0 as a
# *trainable* init (no frozen-subspace cap; full LoRA expressivity).

from typing import Dict, List, Optional

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.router_state import (
    RouterStateMixin,
    _apply_sigma_band_mask,
    _register_sigma_band_partition,
)


class OrthoLoRAModule(BaseLoRAModule):
    """OrthoLoRA: Cayley-rotated SVD basis (no orthogonality reg hyperparameter).

    Frozen P_basis (out, r) / Q_basis (r, in) from the base weight's top-r SVD,
    rotated by Cayley(S_q) / Cayley(S_p) where R = (I - A)(I + A)^{-1},
    A = S - S^T. Trainable: S_p, S_q (r×r), lambda_layer (1, r).

        out = x @ Q_eff^T @ diag(λ) @ P_eff^T
        where Q_eff = cayley(S_q) @ Q_basis, P_eff = P_basis @ cayley(S_p)

    Zero-init S_p, S_q, λ → ΔW=0 at step 0. Frozen *full-dim* bases (not
    PSOFT's principal-subspace restriction): expressiveness is limited to
    rotations within the initial basis span — the tradeoff under benchmark.

    Ref: PSOFT (Wu et al., ICLR 2026).
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

        # SVD-informed init. Randomized lowrank is ~10-100× faster than full
        # SVD at r ≪ min(m,n) and near-machine-precision on the kept slice.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        q = min(lora_dim + 6, min(W.shape))
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
        P_init = U[:, :lora_dim].clone().contiguous()  # (out, r)
        Q_init = V[:, :lora_dim].T.clone().contiguous()  # (r, in)
        del U, _S_vals, V, W

        # Frozen subspace bases; Cayley rotates within them.
        self.register_buffer("P_basis", P_init.cpu())
        self.register_buffer("Q_basis", Q_init.cpu())

        # Cayley(0) = I → at init P_eff = P_basis, Q_eff = Q_basis.
        self.S_p = torch.nn.Parameter(torch.zeros(lora_dim, lora_dim))
        self.S_q = torch.nn.Parameter(torch.zeros(lora_dim, lora_dim))

        # ΔW = 0 at init (standard LoRA convention).
        self.lambda_layer = torch.nn.Parameter(torch.zeros(1, lora_dim))

        # Absorb into Q_basis so the frozen path carries the rebalance.
        # Run while the buffer is still fp32 so the in-place ``weight.mul_``
        # happens at fp32 precision; downcast immediately after.
        self._register_channel_scale(self.Q_basis, channel_scale)

        # Frozen bases → bf16. The activation chain (lx, out) inherits this
        # dtype via ``dtype = self.P_basis.dtype`` below, halving the
        # saved-for-backward budget. Cayley solve stays fp32 in ``forward``
        # (orthogonality invariant: ``R^T R = I`` ~1e-7 in fp32 vs ~1e-2 in bf16).
        self.P_basis = self.P_basis.to(torch.bfloat16)
        self.Q_basis = self.Q_basis.to(torch.bfloat16)

        # Pre-allocated identity for the batched Cayley solve (avoids ~2
        # small kernels per module per step from a fresh torch.eye).
        self.register_buffer(
            "_eye_r",
            torch.eye(lora_dim, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """R = (I - A)(I + A)^{-1}, A = S - S^T.

        Kept for save-time SVD distillation in :meth:`distill_save_state_dict`;
        forward uses a batched solve instead.
        """
        A = S - S.T
        eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(eye + A, eye - A)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self._skip_module():
            return org_forwarded

        work = self.P_basis.dtype  # bf16 — bases live here, chain follows

        # Stack S_q + S_p into one (2, r, r) solve — halves LU/TRSM launches.
        # Cayley island stays fp32; we cast R only at the boundary into the
        # basis matmuls so orthogonality is preserved while downstream
        # activations stay bf16.
        skew = torch.stack([self.S_q, self.S_p])
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
        R_q = R[0].to(work)
        R_p = R[1].to(work)
        Q_eff = R_q @ self.Q_basis  # bf16

        x_lora = self._rebalance(x.to(work))
        lx = torch.nn.functional.linear(x_lora, Q_eff)
        # λ stays a fp32 Parameter (Adam state precision); cast at multiply
        # so the chain remains bf16. Same for the timestep mask buffer.
        lx = lx * self.lambda_layer.to(work) * self._timestep_mask.to(work)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        P_eff = self.P_basis @ R_p  # bf16
        out = torch.nn.functional.linear(lx, P_eff)

        lora_out = out * self.multiplier * scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley guarantees orthogonality structurally."""
        zero = torch.tensor(0.0, device=self.S_p.device)
        return zero, zero

    @classmethod
    def distill_save_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> None:
        """OrthoLoRA → standard LoRA: Cayley + frozen SVD → ``lora_down``/``lora_up``.

        Mutates ``state_dict`` in place.

        Discriminator: ``.S_p`` keys with ``dim == 2``. The OrthoHydra path
        owns the 3-D ``.S_p`` shape and must run before this method, otherwise
        the 2-D handler would mis-reduce 3-D tensors. Sets are tracked by
        prefix so each module's keys are converted atomically.

        Sqrt-splits ``λ`` between the two factors so the on-disk product
        ``ΔW = P_eff @ diag(λ) @ Q_eff`` is preserved bit-exactly under
        the ``(lora_down, lora_up)`` factorization.
        """
        prefixes = set()
        for key in state_dict.keys():
            if key.endswith(".S_p") and state_dict[key].dim() == 2:
                prefixes.add(key[: -len(".S_p")])

        for prefix in prefixes:
            S_p = state_dict[f"{prefix}.S_p"]
            S_q = state_dict[f"{prefix}.S_q"]
            P_basis = state_dict[f"{prefix}.P_basis"]
            Q_basis = state_dict[f"{prefix}.Q_basis"]
            lam = state_dict[f"{prefix}.lambda_layer"]  # (1, r)
            alpha = state_dict.get(f"{prefix}.alpha")
            save_dtype = dtype if dtype is not None else P_basis.dtype

            R_p = cls._cayley(S_p.float())
            R_q = cls._cayley(S_q.float())
            P_eff = P_basis.float() @ R_p  # (out, r)
            Q_eff = R_q @ Q_basis.float()  # (r, in)

            lam_1d = lam.squeeze(0).float()
            lam_abs = lam_1d.abs()
            lam_sign = lam_1d.sign()
            lam_sqrt = lam_abs.sqrt()
            lora_up = (
                (P_eff * (lam_sqrt * lam_sign).unsqueeze(0))
                .to(save_dtype)
                .cpu()
                .contiguous()
            )
            lora_down = (
                (Q_eff * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()
            )

            for suffix in ("S_p", "S_q", "lambda_layer", "P_basis", "Q_basis"):
                state_dict.pop(f"{prefix}.{suffix}", None)
            # inv_scale stays — shared buffer, not an ortho-exp-only key.

            state_dict[f"{prefix}.lora_up.weight"] = lora_up
            state_dict[f"{prefix}.lora_down.weight"] = lora_down
            if alpha is not None:
                state_dict[f"{prefix}.alpha"] = alpha


class OrthoInitLoRAModule(BaseLoRAModule):
    """OrthoInit: top-r SVD of W₀ as *initialization only* — trainable, uncapped.

    OrthoLoRA freezes the top-r SVD basis and only Cayley-rotates *within* it,
    so ``colspace(ΔW) ⊆ top-r(W₀)`` for the whole run — the rotation can never
    leave the principal subspace, which is what makes T-LoRA+ortho / chimera-
    ortho feel weak. OrthoInit keeps the same three-factor ``P diag(λ) Q`` shape
    but makes ``P_init`` / ``Q_init`` **trainable Parameters** seeded from the
    SVD (no frozen buffer, no Cayley). ΔW can then reach *any* rank-r subspace
    (full LoRA expressivity) while still getting the W₀-aligned warm start.

        out = x @ Q_init^T @ diag(λ) @ P_init^T

    λ zero-init keeps ΔW=0 at step 0. Because ∂L/∂P, ∂L/∂Q ∝ λ, the magnitudes
    fit first along the principal directions, then the bases rotate off-subspace
    as the loss demands — SVD as a *starting point*, not a cage.

    Saved as standard LoRA (sqrt-split λ into ``lora_down`` / ``lora_up``), so the
    on-disk format, merge path, and inference loader are identical to a distilled
    OrthoLoRA. The ``lambda_layer`` + ``_timestep_mask`` factor is preserved at
    train time so T-LoRA's per-rank schedule still gates the singular values.
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

        # SVD-informed init (same randomized low-rank as OrthoLoRA), but the
        # bases become trainable rather than frozen subspace buffers.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        q = min(lora_dim + 6, min(W.shape))
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
        P_init = U[:, :lora_dim].clone().contiguous()  # (out, r)
        Q_init = V[:, :lora_dim].T.clone().contiguous()  # (r, in)
        del U, _S_vals, V, W

        # Trainable factors. fp32 master (Adam state precision); cast to the
        # activation dtype at the matmul boundary in forward — same convention
        # as ``lambda_layer``. No Cayley, no frozen P_basis/Q_basis buffers.
        self.P_init = torch.nn.Parameter(P_init.cpu())
        self.Q_init = torch.nn.Parameter(Q_init.cpu())
        # ΔW = 0 at init; also gates ∂L/∂P, ∂L/∂Q off until λ ramps.
        self.lambda_layer = torch.nn.Parameter(torch.zeros(1, lora_dim))

        # Absorb channel scale into the input-side Q_init while still fp32
        # (mutates ``.data`` in place; output unchanged, gradient rebalanced).
        self._register_channel_scale(self.Q_init.data, channel_scale)

    # Forward is the shared BaseLoRAModule scaffold; this class supplies the
    # trainable SVD-seeded down / up GEMMs and folds ``lambda_layer`` into the
    # T-LoRA gate. The fp32-master → ``work`` cast dance + dtype-policy rationale
    # (mirrors the lora.py 8c2005c fix) live in the base scaffold.

    def _down(self, x_lora, work):
        return torch.nn.functional.linear(x_lora, self.Q_init.to(work))

    def _gate(self, lx, work):
        # λ (fp32 master Parameter) + the T-LoRA mask gate the rank latent. lx
        # is (B, L, r) here — tiny — so the fp32 promotion is negligible and the
        # base scaffold casts back to ``work`` before the up GEMM.
        return lx * self.lambda_layer.float() * self._timestep_mask

    def _up(self, lx, work):
        return torch.nn.functional.linear(lx, self.P_init.to(work))

    def _eval_delta(self, x, org_forwarded):
        # bf16 fast path (mirrors LoRAModule). Distilled checkpoints infer as
        # plain LoRA; this branch only fires for sample-during-training.
        x_lora = self._rebalance(x)
        lx = torch.nn.functional.linear(x_lora, self.Q_init.to(x.dtype))
        lx = lx * self.lambda_layer.to(x.dtype) * self._timestep_mask.to(x.dtype)
        lx = torch.nn.functional.linear(lx, self.P_init.to(x.dtype))
        return lx * self.multiplier * self.scale

    @classmethod
    def distill_save_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> None:
        """OrthoInit → standard LoRA: ``P_init`` / ``Q_init`` / λ → down/up.

        Mutates ``state_dict`` in place. Discriminator: ``.P_init`` keys (no
        other variant uses that name). Sqrt-splits λ between the two factors so
        the on-disk product ``ΔW = P_init @ diag(λ) @ Q_init`` is preserved
        bit-exactly under the ``(lora_down, lora_up)`` factorization.
        """
        prefixes = set()
        for key in state_dict.keys():
            if key.endswith(".P_init"):
                prefixes.add(key[: -len(".P_init")])

        for prefix in prefixes:
            P = state_dict[f"{prefix}.P_init"].float()  # (out, r)
            Q = state_dict[f"{prefix}.Q_init"].float()  # (r, in)
            lam = state_dict[f"{prefix}.lambda_layer"]  # (1, r)
            alpha = state_dict.get(f"{prefix}.alpha")
            save_dtype = (
                dtype if dtype is not None else state_dict[f"{prefix}.P_init"].dtype
            )

            lam_1d = lam.squeeze(0).float()
            lam_abs = lam_1d.abs()
            lam_sign = lam_1d.sign()
            lam_sqrt = lam_abs.sqrt()
            lora_up = (
                (P * (lam_sqrt * lam_sign).unsqueeze(0))
                .to(save_dtype)
                .cpu()
                .contiguous()
            )
            lora_down = (Q * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()

            for suffix in ("P_init", "Q_init", "lambda_layer"):
                state_dict.pop(f"{prefix}.{suffix}", None)
            # inv_scale stays — shared buffer, not an orthoinit-only key.

            state_dict[f"{prefix}.lora_up.weight"] = lora_up
            state_dict[f"{prefix}.lora_down.weight"] = lora_down
            if alpha is not None:
                state_dict[f"{prefix}.alpha"] = alpha


class OrthoHydraLoRAModule(RouterStateMixin, BaseLoRAModule):
    """OrthoLoRA + HydraLoRA: Cayley-rotated MoE with disjoint per-expert P-bases.

    Shared Q_basis + trainable S_q (down). Up takes the top E*r singular
    vectors and partitions them into E disjoint slices of r columns; expert e
    owns ``P_bases[e]: (out, r)``, rotated by per-expert Cayley R_p[e].
    Because SVD columns are orthonormal, ``P_bases[i]^T P_bases[j] = 0`` for
    i≠j — experts are structurally orthogonal in output space.

    Disjoint slices (not shared P + per-expert R_p): with a shared basis,
    ``P_eff[i]^T P_eff[j] = R_p[i]^T R_p[j]`` is orthogonal and never zero, so
    every expert lives in the same rank-r span and the router gets near-
    identical ``score_e`` (MoE cold-start deadlock; bench 2026-04-21).
    Disjoint subspaces make ``score_e`` genuinely different from step 0.

    Fallback when ``min(out, in) < E*r``: ``P_bases`` replicates the top-r
    slice E times (warning logged). All experts start identical — narrow-
    layer OrthoHydra relies entirely on the per-expert ``S_p`` rotations
    diverging during training; if the router collapses they never will.
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
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: int = 1,
        sigma_bucket_boundaries: Optional[List[float]] = None,
        fei_feature_dim: int = 0,
        use_global_router: bool = False,
        centered_gate: bool = False,
        lambda_init: float = 0.0,
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
        # Centered-gate residual: forward uses (g_e - 1/E) and λ starts
        # nonzero so ΔW=0 at init but router logits get step-0 gradient.
        self._centered_gate = bool(centered_gate)

        # SVD-informed init with disjoint per-expert P slices. Top E*r U columns
        # split into E slices of r — each slice orthonormal, mutually orthogonal.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        target_cols = num_experts * lora_dim
        max_cols = min(W.shape)
        disjoint = target_cols <= max_cols
        q = min(target_cols + 6, max_cols) if disjoint else min(lora_dim + 6, max_cols)
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
        Q_init = V[:, :lora_dim].T.clone().contiguous()  # (r, in) shared
        if disjoint:
            P_stack = U[:, :target_cols].reshape(out_dim, num_experts, lora_dim)
            P_bases_init = P_stack.permute(1, 0, 2).clone().contiguous()
        else:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                f"{lora_name}: min(out={out_dim}, in={in_dim})={max_cols} < "
                f"num_experts*lora_dim={target_cols}; falling back to shared "
                "P_basis (experts start identical, rely on S_p divergence)."
            )
            P_shared = U[:, :lora_dim].clone().contiguous()
            P_bases_init = (
                P_shared.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
            )
        del U, _S_vals, V, W

        self.register_buffer("P_bases", P_bases_init.cpu())  # (E, out, r)
        self.register_buffer("Q_basis", Q_init.cpu())
        self._disjoint_basis = disjoint

        self.S_q = torch.nn.Parameter(torch.zeros(lora_dim, lora_dim))
        # Per-expert P rotations. With disjoint P_bases zero-init still yields
        # E distinct P_eff slices; in the narrow-layer fallback all experts
        # start identical and must diverge through training-time updates.
        self.S_p = torch.nn.Parameter(torch.zeros(num_experts, lora_dim, lora_dim))
        # λ zero-init keeps ΔW=0 at step 0 but gates the router gradient off
        # until λ ramps. Under centered_gate the gate residual (g_e - 1/E)
        # already preserves ΔW=0 at init, so λ can start small-nonzero and the
        # router receives gradient at literal step 0.
        self.lambda_layer = torch.nn.Parameter(
            torch.full((1, lora_dim), float(lambda_init))
        )

        self.use_global_router = bool(use_global_router)
        # Layer-local router: see HydraLoRAModule for σ + FEI routing surface.
        if self.use_global_router:
            self.sigma_feature_dim = 0
            self.fei_feature_dim = 0
        else:
            self.sigma_feature_dim = int(sigma_feature_dim)
            self.fei_feature_dim = int(fei_feature_dim)
            router_in_dim = lora_dim + self.sigma_feature_dim + self.fei_feature_dim
            self.router = torch.nn.Linear(router_in_dim, num_experts, bias=True)
            with torch.no_grad():
                self.router.weight.zero_()
                # The normal seed is the legacy symmetry-breaker for zero-init
                # λ (it gives the gate a tiny non-uniform tilt at step 0). Under
                # centered_gate the symmetry break comes from (P_k - mean)·λ0
                # instead, and we want the gate *exactly* uniform at init so the
                # residual vanishes — so leave the router fully zero-init.
                if not self._centered_gate:
                    torch.nn.init.normal_(self.router.weight[:, :lora_dim], std=0.01)
                self.router.bias.zero_()

        # Channel-scale absorption runs in fp32; downcast the bases afterward.
        self._register_channel_scale(self.Q_basis, channel_scale)

        # Frozen bases → bf16. ``dtype = self.P_bases.dtype`` downstream
        # carries the bf16 through the activation chain (lx, P_combined, out).
        # Cayley stays fp32 (orthogonality invariant; see forward).
        self.P_bases = self.P_bases.to(torch.bfloat16)
        self.Q_basis = self.Q_basis.to(torch.bfloat16)

        self._last_gate = None
        # See router_state.py + HydraLoRAModule for the always-a-Tensor +
        # pointer-stable buffer protocol that drops the compile guards.
        self._register_router_io_buffers(num_experts)
        if specialize_experts_by_sigma_buckets and self.use_global_router:
            raise ValueError(
                "specialize_experts_by_sigma_buckets is incompatible with "
                "use_global_router=True (no per-layer logits to mask)."
            )
        self._sigma_band_partition: bool = bool(specialize_experts_by_sigma_buckets)
        if self._sigma_band_partition:
            _register_sigma_band_partition(
                self, num_experts, num_sigma_buckets, sigma_bucket_boundaries
            )

        self.register_buffer(
            "_eye_r",
            torch.eye(lora_dim, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """R = (I - A)(I + A)^{-1}, A = S - S^T. 2D or batched 3D.

        Kept for save-time SVD distillation; forward uses a batched solve.
        """
        A = S - S.transpose(-2, -1)
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)

    def _compute_gate(self, lx: torch.Tensor) -> torch.Tensor:
        """RMS-pool post-Q_eff (pre-λ, pre-mask), concat σ/FEI, router, softmax.

        Pre-λ pool: λ is zero-init, so post-λ pooling would zero the router
        input at step 0 and freeze gradient. See HydraLoRAModule._compute_gate
        for the routing surface; under use_global_router lx is ignored.
        """
        if self.use_global_router:
            B = lx.shape[0] if lx.dim() >= 1 else 1
            w = self._routing_weights
            if w.dim() == 1:
                w = w.unsqueeze(0)
            return w.to(lx.dtype).expand(B, -1)
        if lx.dim() >= 3:
            B = lx.shape[0]
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        pooled = pooled.to(self.router.weight.dtype)
        parts = [pooled]
        if self.sigma_feature_dim > 0:
            sigma_feat = self._sigma_features.to(pooled.dtype)
            sigma_feat = sigma_feat.expand(pooled.shape[0], -1)
            parts.append(sigma_feat)
        if self.fei_feature_dim > 0:
            fei_feat = self._fei.to(pooled.dtype).expand(pooled.shape[0], -1)
            parts.append(fei_feat)
        router_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        logits = self.router(router_in)  # (B, num_experts)
        if self._sigma_band_partition:
            logits = _apply_sigma_band_mask(
                logits, self._sigma, self._expert_band, self._sigma_edges
            )
        return torch.softmax(logits, dim=-1)

    # σ / FEI / routing-weights method surface is inherited from
    # RouterStateMixin (set_sigma / clear_sigma / set_fei / clear_fei /
    # set_routing_weights / clear_routing_weights — each buffer-presence-guarded).

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        work = self.P_bases.dtype  # bf16 — bases + downstream activations

        # Stack S_q with S_p into one (E+1, r, r) solve — single LU+TRSM
        # launch covers shared Q rotation and all per-expert P rotations.
        # Cayley solve stays fp32; boundary cast feeds R into the basis matmuls.
        skew = torch.cat([self.S_q.unsqueeze(0), self.S_p], dim=0)
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
        R_q = R[0].to(work)
        R_p = R[1:].to(work)
        Q_eff = R_q @ self.Q_basis  # bf16

        x_lora = self._rebalance(x.to(work))
        lx = torch.nn.functional.linear(x_lora, Q_eff)

        # Pool pre-λ (zero-init λ would zero the router input at step 0).
        gate = self._compute_gate(lx)  # (B, E) — fp32 from the router
        if self.training:
            # Plain STORE_ATTR — see HydraLoRAModule.forward. Keep native
            # dtype here so the balance loss reads the router-native gate.
            self._last_gate = gate

        lx = lx * self.lambda_layer.to(work) * self._timestep_mask.to(work)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        P_eff = self.P_bases @ R_p  # (E, out, r) bf16
        # Centered-gate residual: (g_e - 1/E). With a uniform (zero-init)
        # router this is exactly 0 at init → ΔW=0 (base preserved), while
        # ∂Δy/∂a_k ∝ (P_k - P̄)·diag(λ0)·ℓ ≠ 0 → router gradient at step 0.
        # _last_gate stays the raw softmax so the balance loss still reads a
        # proper distribution.
        gate_eff = gate
        if self._centered_gate:
            gate_eff = gate - (1.0 / self.num_experts)
        # Cast gate at the einsum boundary so bf16 × fp32 doesn't promote
        # P_combined back to fp32 and inflate the saved activation.
        P_combined = torch.einsum("be,eor->bor", gate_eff.to(work), P_eff)

        orig_shape = lx.shape
        B = orig_shape[0]
        lx_3d = lx.reshape(B, -1, orig_shape[-1])
        out = torch.bmm(lx_3d, P_combined.transpose(1, 2))
        out = out.reshape(*orig_shape[:-1], -1)

        lora_out = out * self.multiplier * scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley guarantees orthogonality structurally."""
        zero = torch.tensor(0.0, device=self.S_p.device)
        return zero, zero

    def subspace_overlap(self) -> float:
        """Mean normalized cross-expert column-space overlap of ``P_bases``.

        ``‖P_iᵀ P_j‖_F / √r`` averaged over expert pairs i<j: ~0 when the
        per-expert slices are disjoint (the design invariant), ~1 in the
        narrow-layer shared-basis fallback. Constant across training — the
        Cayley ``R_p`` rotations preserve each slice's span, so this is a
        structural *guardrail* (catches a basis/init regression), not a
        learning signal. Computed once from the frozen buffer and cached.
        """
        cached = getattr(self, "_subspace_overlap_cached", None)
        if cached is not None:
            return cached
        with torch.no_grad():
            P = self.P_bases.float()  # (E, out, r)
            E = P.shape[0]
            r = P.shape[-1]
            if E < 2 or r == 0:
                self._subspace_overlap_cached = 0.0
                return 0.0
            # (E, E, r, r) cross-Gram, Frobenius over the (r, r) block.
            cross = torch.einsum("ior,jos->ijrs", P, P)
            fro = cross.pow(2).sum(dim=(-1, -2)).sqrt()  # (E, E)
            off = (fro.sum() - fro.diagonal().sum()) / (E * (E - 1))
            val = float((off / (r**0.5)).item())
        self._subspace_overlap_cached = val
        return val

    @classmethod
    def distill_save_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> None:
        """OrthoHydra → Hydra runtime form (shared down + stacked ups).

        Mutates ``state_dict`` in place. Discriminator: ``.S_p`` with
        ``dim == 3`` AND co-located ``.S_q`` with ``dim == 2``. The 2-D
        ``S_q`` is the only thing that distinguishes OrthoHydra (shared
        Q rotation across experts) from StackedExperts-ortho (per-expert
        ``S_q`` is 3-D), and ``StackedExpertsLoRAModule.distill_save_state_dict``
        must run before this method.

        Outputs: ``.lora_down.weight`` (shared) + ``.lora_up_weight``
        (stacked ``(E, out, r)`` — the Hydra training runtime layout). The
        downstream MoE writer in :meth:`HydraLoRAModule.build_moe_state_dict`
        expands this into per-expert ``.lora_ups.{i}.weight`` keys.

        NOTE (centered_gate parity): the runtime MoE form combines per-expert
        ups with the raw softmax gate, i.e. ``Σ_e g_e P_e``. A checkpoint
        trained with ``ortho_centered_gate=True`` actually computed
        ``Σ_e (g_e - 1/E) P_e = Σ_e g_e P_e - (1/E)Σ_e P_e`` — an extra fixed
        ``-(1/E)Σ_e P_e`` mean-expert bias (a plain-LoRA branch) that this
        runtime form does NOT carry. In-training CMMD validation runs the
        live (centered) module forward and is faithful; deploying the
        distilled ``_moe`` checkpoint is not, until the router-live node
        subtracts the mean expert. Treat centered_gate as train/research-only
        for now. See proposal.md / [[project_crossattn_emb_router_source]].
        """
        prefixes = set()
        for key in list(state_dict.keys()):
            if not (key.endswith(".S_p") and state_dict[key].dim() == 3):
                continue
            prefix = key[: -len(".S_p")]
            S_q_key = f"{prefix}.S_q"
            if S_q_key not in state_dict or state_dict[S_q_key].dim() != 2:
                continue
            prefixes.add(prefix)

        for prefix in prefixes:
            S_p = state_dict[f"{prefix}.S_p"]  # (E, r, r)
            S_q = state_dict[f"{prefix}.S_q"]  # (r, r)
            # Per-expert disjoint bases (new) or legacy shared basis (old ckpts).
            P_bases = state_dict.get(f"{prefix}.P_bases")
            if P_bases is None:
                P_bases = state_dict[f"{prefix}.P_basis"]  # (out, r) legacy
            Q_basis = state_dict[f"{prefix}.Q_basis"]  # (r, in)
            lam = state_dict[f"{prefix}.lambda_layer"]  # (1, r)
            alpha = state_dict.get(f"{prefix}.alpha")
            save_dtype = dtype if dtype is not None else P_bases.dtype

            R_q = cls._cayley(S_q.float())  # (r, r)
            Q_eff = R_q @ Q_basis.float()  # (r, in)

            R_p = cls._cayley(S_p.float())  # (E, r, r)
            if P_bases.dim() == 3:
                # (E, out, r) @ (E, r, r) = (E, out, r)
                P_eff = P_bases.float() @ R_p
            else:
                # legacy shared (out, r): broadcast over experts
                P_eff = P_bases.float().unsqueeze(0) @ R_p  # (E, out, r)

            # sqrt-split lambda so ΔW = P @ diag(λ) @ Q is preserved bit-exactly
            lam_1d = lam.squeeze(0).float()
            lam_abs = lam_1d.abs()
            lam_sign = lam_1d.sign()
            lam_sqrt = lam_abs.sqrt()

            lora_down = (
                (Q_eff * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()
            )
            lora_up_weight = (
                (P_eff * (lam_sqrt * lam_sign).unsqueeze(0).unsqueeze(0))
                .to(save_dtype)
                .cpu()
                .contiguous()
            )

            for suffix in (
                "S_p",
                "S_q",
                "lambda_layer",
                "P_basis",
                "P_bases",
                "Q_basis",
            ):
                state_dict.pop(f"{prefix}.{suffix}", None)

            state_dict[f"{prefix}.lora_down.weight"] = lora_down
            state_dict[f"{prefix}.lora_up_weight"] = lora_up_weight
            if alpha is not None:
                state_dict[f"{prefix}.alpha"] = alpha
