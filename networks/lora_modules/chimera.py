# ChimeraHydra: dual-pool additive MoE with TWO Cayley A's per Linear — content
# pool (K_c B-heads, routed by pooled crossattn_emb via the network-level
# ContentRouter) + freq pool (K_f B-heads, routed by FEI(z_t) via the
# network-level FreqRouter). T-LoRA's rank mask hits the content branch only
# — the freq branch stays full-rank at every t (TimeStep Master-style
# asymmetric mixture). See docs/experimental/chimera-hydra.md.
#
# Per Linear:
#
#     A_c = Cayley(S_q_c) · Q_basis_c          (r, in)   — content latent
#     A_f = Cayley(S_q_f) · Q_basis_f          (r, in)   — freq    latent
#
#     B_c[k] = P_bases_c[k] · Cayley(S_p_c[k]) (out, r)  k = 0..K_c-1
#     B_f[j] = P_bases_f[j] · Cayley(S_p_f[j]) (out, r)  j = 0..K_f-1
#
#     Δy = Σ_c π_c[c] · B_c[c] (A_c x · λ_c · mask_t(σ))     ◄ content branch
#        + Σ_f π_f[f] · B_f[f] (A_f x · λ_f)                  ◄ freq    branch
#
# SVD partition gives free orthogonality on BOTH sides: top 2r right-singular
# vectors of W split r/r into Q_basis_c / Q_basis_f (row-spaces orthogonal);
# top (K_c+K_f)·r left-singular vectors split K_c·r / K_f·r into P_bases_c /
# P_bases_f (every expert's col-space orthogonal to every other's, within and
# across pools).
#
# Both pools are network-routed and centered-gate by construction (the only
# shipped configuration): each pool's gate is recentered to ``π − 1/K`` and
# λ_c/λ_f start at ``lambda_init`` (>0), so ΔW=0 at init while each pool's
# disjoint per-expert P-subspaces still feed its router a nonzero step-0
# gradient. No per-Linear router — ``ContentRouter``/``FreqRouter`` live on
# ``LoRANetwork`` and write π_c/π_f into the slot-assigned buffers below.

import logging
from typing import Dict, List, Optional

import torch

from networks.attn_fuse import match_fused_spec
from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale
from networks.lora_modules.lora import defuse_standard_qkv

logger = logging.getLogger(__name__)


class _ChimeraRoutingMixin:
    """Shared dual-pool routing plumbing for the train + inference modules.

    Both chimera classes carry two gate buffers — ``_content_routing_weights``
    (π_c) and ``_freq_routing_weights`` (π_f) — slot-assigned (NO
    ``.detach()``, NO ``.copy_()``) so the buffer keeps the router's
    ``grad_fn`` and ``∂L/∂π`` reaches the router parameters. Mirrors
    ``router_state._set_routing_weights`` / ``HydraLoRAModule``.
    """

    def _register_routing_buffers(self, K_c: int, K_f: int) -> None:
        # Uniform 1/K placeholders, non-persistent (re-derived on construction).
        self.register_buffer(
            "_content_routing_weights",
            torch.full((1, K_c), 1.0 / max(K_c, 1), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_freq_routing_weights",
            torch.full((1, K_f), 1.0 / max(K_f, 1), dtype=torch.float32),
            persistent=False,
        )
        self._last_gate = None  # (B, K_c+K_f), read by LoRANetwork's balance loss

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign the freq pool's gates (preserves grad_fn)."""
        self._freq_routing_weights = self._coerce_gate(
            weights, self._freq_routing_weights
        )

    def clear_freq_routing_weights(self) -> None:
        """Reset to uniform 1/K_f without rebinding the pointer."""
        K_f = int(self._freq_routing_weights.shape[-1])
        self._freq_routing_weights.fill_(1.0 / max(K_f, 1))

    def set_content_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign π_c from the network-level ContentRouter (preserves
        grad_fn — same contract as :meth:`set_freq_routing_weights`)."""
        self._content_routing_weights = self._coerce_gate(
            weights, self._content_routing_weights
        )

    def clear_content_routing_weights(self) -> None:
        K_c = int(self._content_routing_weights.shape[-1])
        self._content_routing_weights.fill_(1.0 / max(K_c, 1))

    @staticmethod
    def _coerce_gate(weights: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
        w = weights.to(dtype=buf.dtype, device=buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        return w

    @staticmethod
    def _broadcast_gate(pi: torch.Tensor, batch: int) -> torch.Tensor:
        """(K,) / (1, K) → (batch, K); already-batched passes through."""
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)
        if pi.shape[0] == 1 and batch > 1:
            pi = pi.expand(batch, -1)
        return pi

    def _content_gate_raw(self, batch: int) -> torch.Tensor:
        """π_c broadcast to the batch, fp32 (the einsum casts at its boundary)."""
        return self._broadcast_gate(self._content_routing_weights, batch).float()

    def _freq_gate_raw(self, batch: int) -> torch.Tensor:
        """π_f broadcast to the batch in its native dtype (grad_fn intact)."""
        return self._broadcast_gate(self._freq_routing_weights, batch)

    def _full_gate(self, pi_c_raw: torch.Tensor) -> torch.Tensor:
        """(B, K_c+K_f) raw (uncentered) simplex for the per-pool balance loss.

        ``LoRANetwork._get_chimera_balance_loss`` slices at
        ``num_experts_content`` to recover the two halves.
        """
        pi_f = self._broadcast_gate(self._freq_routing_weights, pi_c_raw.shape[0])
        pi_f = pi_f.to(pi_c_raw.dtype)
        if pi_f.shape[0] != pi_c_raw.shape[0]:
            pi_f = pi_f.expand(pi_c_raw.shape[0], -1)
        return torch.cat([pi_c_raw, pi_f], dim=-1)

    @staticmethod
    def _center(pi: torch.Tensor, num_experts: int) -> torch.Tensor:
        """Recenter a gate to ``π − 1/K`` (NOT in-place — preserves grad_fn so
        the FreqRouter / ContentRouter still receive gradient). A uniform gate
        then contributes exactly 0, keeping ΔW=0 at init."""
        return pi - (1.0 / num_experts)

    @staticmethod
    def _combine_up(lx_c, lx_f, comb_c, comb_f, scale):
        """Cat both pools along the rank axis → ONE bmm (only one full
        hidden-size tensor live at a time). ``scale`` is shared (rank_dropout
        returns the same scalar for both pools), so it folds out of the sum.
        """
        orig_shape = lx_c.shape
        B = orig_shape[0]
        lx_cat = torch.cat(
            [lx_c.reshape(B, -1, orig_shape[-1]), lx_f.reshape(B, -1, orig_shape[-1])],
            dim=-1,
        )  # (B, L, 2r)
        comb_cat = torch.cat([comb_c, comb_f], dim=-1)  # (B, out, 2r)
        out = torch.bmm(lx_cat, comb_cat.transpose(1, 2))
        return (out * scale).reshape(*orig_shape[:-1], -1)


class ChimeraHydraLoRAModule(_ChimeraRoutingMixin, BaseLoRAModule):
    """ChimeraHydra training-time module: two Cayley A's, two B-pools, both
    network-routed (content via ContentRouter, freq via FreqRouter). See the
    module header for the shape/formula summary.

    ``use_ortho_init=True`` swaps each pool's frozen-basis + Cayley
    parameterization for **trainable** fp32 SVD-seeded bases (``Q_basis_*`` /
    ``P_bases_*`` become Parameters, ``S_*``/Cayley drop out) — ΔW can then
    leave the principal 2r-subspace while keeping the W₀-aligned warm start;
    ΔW=0 at init still holds (comes from the centered gate, not the basis).

    Save distills Cayley → free-form per pool (:meth:`distill_save_state_dict`
    / :meth:`build_moe_state_dict`; OrthoInit is the same distill with R = I).
    Either way load rebuilds a ``ChimeraHydraInferenceModule`` — the on-disk
    ``*_chimera.safetensors`` layout is identical.
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
        num_experts_content: int = 3,
        num_experts_freq: int = 3,
        channel_scale=None,
        lambda_init: float = 0.0,
        use_ortho_init: bool = False,
        expert_basis_mult: int = 1,
        expert_diag: bool = False,
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

        if num_experts_content <= 0 or num_experts_freq <= 0:
            raise ValueError(
                f"ChimeraHydra requires both pools non-empty: "
                f"K_c={num_experts_content}, K_f={num_experts_freq}"
            )

        K_c = int(num_experts_content)
        K_f = int(num_experts_freq)
        r = int(lora_dim)

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.num_experts_content = K_c
        self.num_experts_freq = K_f
        self.num_experts = K_c + K_f
        self.in_dim = in_dim

        self.use_ortho_init = bool(use_ortho_init)

        # Per-expert capability levers (frozen-Cayley path only; forced off
        # under ortho-init since its bases are already free). Both distill
        # into the standard (K, out, r) up-stack — on-disk layout unchanged.
        #   * expert_basis_mult (m>=1): each expert gets an over-complete
        #     (out, M=m*r) frozen pool from a disjoint U-slice + an M×M Cayley
        #     rotation; forward selects the first r columns (Stiefel(M, r)) —
        #     depth without breaking cross-expert orthogonality.
        #   * expert_diag: per-expert (K, r) trainable diagonal σ (init 1)
        #     folded into the up-projection — the singular-spectrum DOF the
        #     orthogonal-only frozen path otherwise lacks.
        m_mult = 1 if self.use_ortho_init else max(1, int(expert_basis_mult))
        self._expert_diag = bool(expert_diag) and not self.use_ortho_init

        # SVD partition: each pool wants its own r right-singular vectors
        # (Q_basis_{c,f}, (r, in)) and its own K_*·M left-singular vectors
        # (P_bases_{c,f}, (K_*, out, M) where M=m*r). One low-rank SVD with q
        # big enough to cover both pools.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        M = r * m_mult
        target_right = 2 * r  # need this many V columns
        max_cols = min(W.shape)
        target = max((K_c + K_f) * M, target_right)
        disjoint = target <= max_cols
        if not disjoint and M > r:
            # Over-complete pool overflows this Linear's width — fall back to
            # the canonical r-slice for this layer (still disjoint where it
            # fits). The expander degrades per-layer, never globally.
            logger.warning(
                f"{lora_name}: expert_basis_mult={m_mult} needs "
                f"(K_c+K_f)·m·r={(K_c + K_f) * M} > min(out, in)={max_cols}; "
                "using M=r (no over-complete pool) for this layer."
            )
            M = r
            target = max((K_c + K_f) * r, target_right)
            disjoint = target <= max_cols
        self._M = M
        q = min(target + 6, max_cols) if disjoint else min(r + 6, max_cols)
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)

        if disjoint:
            # Right-singular split: V is (in, q). Top r → content, next r →
            # freq — subsets of the same SVD basis, so orthonormal columns.
            Q_basis_c = V[:, :r].T.clone().contiguous()  # (r, in)
            Q_basis_f = V[:, r : 2 * r].T.clone().contiguous()  # (r, in)

            # Left-singular split: U is (out, q). First K_c*M -> content P
            # stack, next K_f*M -> freq P stack, each reshape-partitioned into
            # disjoint M-wide per-expert slices (same trick as OrthoHydra) —
            # every expert's col-space orthogonal to every other's, within and
            # across pools. With m>1 each slice is an m*r pool; the Stiefel
            # selector picks an r-dim subspace within it at forward time.
            U_c = U[:, : K_c * M].reshape(out_dim, K_c, M)
            P_bases_c_init = U_c.permute(1, 0, 2).clone().contiguous()
            U_f = U[:, K_c * M : (K_c + K_f) * M].reshape(out_dim, K_f, M)
            P_bases_f_init = U_f.permute(1, 0, 2).clone().contiguous()
        else:
            # Narrow-layer fallback: replicate top-r slice into each pool
            # (M==r here — over-complete already downgraded above). Pool-
            # orthogonality is lost; both pools rely on the Cayley rotations
            # diverging during training.
            logger.warning(
                f"{lora_name}: min(out={out_dim}, in={in_dim})={max_cols} < "
                f"max(K_c+K_f, 2)·r = {target}; falling back to shared "
                "SVD slice (pools start identical, rely on Cayley divergence)."
            )
            Q_shared = V[:, :r].T.clone().contiguous()
            Q_basis_c = Q_shared.clone()
            Q_basis_f = Q_shared.clone()
            P_shared = U[:, :r].clone().contiguous()
            P_bases_c_init = P_shared.unsqueeze(0).expand(K_c, -1, -1).contiguous()
            P_bases_f_init = P_shared.unsqueeze(0).expand(K_f, -1, -1).contiguous()
        del U, _S_vals, V, W
        self._disjoint_basis = disjoint

        if self.use_ortho_init:
            # OrthoInit: SVD bases become TRAINABLE fp32 parameters (no frozen
            # buffer, no Cayley) — ΔW can leave the principal 2r-subspace while
            # keeping the W₀-aligned warm start. ΔW=0 at init still holds via
            # the centered gate. Pool orthogonality holds at init, free to
            # drift thereafter (as OrthoInitLoRAModule intends).
            self.Q_basis_c = torch.nn.Parameter(Q_basis_c.cpu().float())
            self.Q_basis_f = torch.nn.Parameter(Q_basis_f.cpu().float())
            self.P_bases_c = torch.nn.Parameter(P_bases_c_init.cpu().float())
            self.P_bases_f = torch.nn.Parameter(P_bases_f_init.cpu().float())
        else:
            # Frozen subspace bases (one per pool). P bases are M=m·r wide.
            self.register_buffer("Q_basis_c", Q_basis_c.cpu())
            self.register_buffer("Q_basis_f", Q_basis_f.cpu())
            self.register_buffer("P_bases_c", P_bases_c_init.cpu())  # (K_c, out, M)
            self.register_buffer("P_bases_f", P_bases_f_init.cpu())  # (K_f, out, M)

            # Cayley(0)=I -> at init each effective basis equals its frozen
            # buffer. P-side skew is M×M (rotates within each expert's m·r
            # pool); A-side stays r×r.
            self.S_q_c = torch.nn.Parameter(torch.zeros(r, r))
            self.S_q_f = torch.nn.Parameter(torch.zeros(r, r))
            self.S_p_c = torch.nn.Parameter(torch.zeros(K_c, M, M))
            self.S_p_f = torch.nn.Parameter(torch.zeros(K_f, M, M))

            # Per-expert trainable diagonal. Init 1 → no-op at the
            # basis level; ΔW=0 at init still holds via the centered gate.
            if self._expert_diag:
                self.sigma_c = torch.nn.Parameter(torch.ones(K_c, r))
                self.sigma_f = torch.nn.Parameter(torch.ones(K_f, r))

        # Per-pool λ, independent magnitudes. Centered-gate starts λ at
        # lambda_init (>0): the recentered gate keeps ΔW=0 at init while
        # feeding both routers a step-0 gradient.
        lam0 = float(lambda_init)
        self.lambda_c = torch.nn.Parameter(torch.full((1, r), lam0))
        self.lambda_f = torch.nn.Parameter(torch.full((1, r), lam0))

        # Channel-scale rebalance happens once at the input (via inv_scale);
        # both A_c and A_f need their input columns pre-scaled to compensate.
        # _register_channel_scale handles Q_basis_c + registers inv_scale;
        # Q_basis_f gets the same column-scale applied manually.
        if channel_scale is not None:
            # .data for OrthoInit Parameters (in-place rescale); frozen-path
            # buffers pass through directly.
            q_c = self.Q_basis_c.data if self.use_ortho_init else self.Q_basis_c
            q_f = self.Q_basis_f.data if self.use_ortho_init else self.Q_basis_f
            self._register_channel_scale(q_c, channel_scale)
            _absorb_channel_scale(q_f, channel_scale)

        if not self.use_ortho_init:
            # Frozen bases -> bf16 (saved-for-backward halved). Cayley solve
            # stays fp32 (R^T R = I to ~1e-7 in fp32 vs ~1e-2 in bf16).
            # OrthoInit keeps the fp32 master since bases are trained directly.
            self.Q_basis_c = self.Q_basis_c.to(torch.bfloat16)
            self.Q_basis_f = self.Q_basis_f.to(torch.bfloat16)
            self.P_bases_c = self.P_bases_c.to(torch.bfloat16)
            self.P_bases_f = self.P_bases_f.to(torch.bfloat16)

        # Pre-allocated identity for the batched Cayley solve — (K_c+K_f+2)
        # skew-symmetric matrices share one fp32 LU+TRSM call. Skipped under
        # OrthoInit (no Cayley solve). When M>r, the A's (size r) and B-pools
        # (size M) solve separately, so a second M-sized identity is kept.
        if not self.use_ortho_init:
            self.register_buffer(
                "_eye_r",
                torch.eye(r, dtype=torch.float32),
                persistent=False,
            )
            if self._M > r:
                self.register_buffer(
                    "_eye_p",
                    torch.eye(self._M, dtype=torch.float32),
                    persistent=False,
                )

        self._register_routing_buffers(K_c, K_f)

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """R = (I - A)(I + A)^{-1}, A = S - S^T. 2D or batched 3D.

        Kept for save-time SVD distillation in :meth:`distill_save_state_dict`;
        forward uses a batched solve over the cat'd skew stack.
        """
        A = S - S.transpose(-2, -1)
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        K_c = self.num_experts_content
        K_f = self.num_experts_freq
        # bf16 for Cayley (frozen-basis); fp32 for OrthoInit (trainable bases
        # are the fp32 master, mirroring OrthoInitLoRAModule).
        work = self.P_bases_c.dtype
        r = self.lora_dim

        if self.use_ortho_init:
            # Trainable bases, no rotation: A_eff/B_eff ARE the parameters.
            Q_eff_c = self.Q_basis_c  # (r, in)
            Q_eff_f = self.Q_basis_f  # (r, in)
            R_p_c = R_p_f = None  # P_eff read straight from the bases below
        elif self._M == r:
            # One batched (2 + K_c + K_f, r, r) Cayley solve covers both A's
            # and both B-pools' rotations. Single LU+TRSM kernel launch.
            skew = torch.cat(
                [
                    self.S_q_c.unsqueeze(0),
                    self.S_q_f.unsqueeze(0),
                    self.S_p_c,
                    self.S_p_f,
                ],
                dim=0,
            )
            A = skew - skew.transpose(-2, -1)
            R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
            R_q_c = R[0].to(work)
            R_q_f = R[1].to(work)
            R_p_c = R[2 : 2 + K_c].to(work)
            R_p_f = R[2 + K_c : 2 + K_c + K_f].to(work)

            Q_eff_c = R_q_c @ self.Q_basis_c  # (r, in)
            Q_eff_f = R_q_f @ self.Q_basis_f  # (r, in)
        else:
            # Over-complete (M>r): the A's rotate at size r, the B-pools at
            # size M, so they can't share one solve. Two batched LU+TRSM calls.
            skew_q = torch.cat(
                [self.S_q_c.unsqueeze(0), self.S_q_f.unsqueeze(0)], dim=0
            )
            A_q = skew_q - skew_q.transpose(-2, -1)
            R_q = torch.linalg.solve(self._eye_r + A_q, self._eye_r - A_q)
            R_q_c = R_q[0].to(work)
            R_q_f = R_q[1].to(work)

            skew_p = torch.cat([self.S_p_c, self.S_p_f], dim=0)  # (K_c+K_f, M, M)
            A_p = skew_p - skew_p.transpose(-2, -1)
            R_p = torch.linalg.solve(self._eye_p + A_p, self._eye_p - A_p)
            R_p_c = R_p[:K_c].to(work)  # (K_c, M, M)
            R_p_f = R_p[K_c : K_c + K_f].to(work)

            Q_eff_c = R_q_c @ self.Q_basis_c  # (r, in)
            Q_eff_f = R_q_f @ self.Q_basis_f  # (r, in)

        # Single rank-cat down-projection for both pools: two separate matmuls
        # make backward materialize TWO (B, L, in) grad_x tensors. Concatenating
        # Q_eff along the rank axis computes grad_x once; the split below is a
        # free view. Bit-identical to the per-pool calls — see
        # test_chimera_down_proj_rank_cat_matches_separate.
        #
        # GEMMs run in the adapter compute dtype (work), not x.dtype (see
        # base.py's forward()); OrthoInit gets its fp32 bottleneck via work.
        comp = work
        Q_eff_cat = torch.cat([Q_eff_c, Q_eff_f], dim=0)  # (2r, in)
        x_lora = self._rebalance(x.to(comp))
        lx_down_cat = torch.nn.functional.linear(x_lora, Q_eff_cat.to(comp))
        lx_c = lx_down_cat[..., :r]
        lx_f = lx_down_cat[..., r:]

        # Cache the RAW (uncentered) simplex for the balance loss before
        # recentering.
        pi_c = self._content_gate_raw(lx_c.shape[0])  # (B, K_c) fp32
        if self.training:
            # Plain STORE_ATTR — see HydraLoRAModule.forward.
            self._last_gate = self._full_gate(pi_c)
        pi_c = self._center(pi_c, K_c)

        # λ + T-LoRA mask (content only) — freq branch keeps full rank at
        # every t (TimeStep Master-style asymmetric mixture).
        lx_c = lx_c * self.lambda_c.to(work) * self._timestep_mask.to(work)
        lx_f = lx_f * self.lambda_f.to(work)

        if self.dropout is not None and self.training:
            lx_c = torch.nn.functional.dropout(lx_c, p=self.dropout)
            lx_f = torch.nn.functional.dropout(lx_f, p=self.dropout)

        lx_c, scale_c = self._apply_rank_dropout(lx_c)
        lx_f, scale_f = self._apply_rank_dropout(lx_f)

        # Cast π at the einsum boundary so bf16 × fp32 doesn't promote
        # P_combined back to fp32 (would inflate the saved activation).
        if self.use_ortho_init:
            P_eff_c = self.P_bases_c  # (K_c, out, r)
            P_eff_f = self.P_bases_f  # (K_f, out, r)
        else:
            P_eff_c = self.P_bases_c @ R_p_c  # (K_c, out, M)
            P_eff_f = self.P_bases_f @ R_p_f  # (K_f, out, M)
            if self._M != r:
                # Stiefel(M, r) select: the first r columns of the rotated
                # over-complete pool. Orthonormal r-dim colspace ⊆ the
                # expert's private m·r slice — trainable subspace, still
                # disjoint across experts.
                P_eff_c = P_eff_c[..., :r]  # (K_c, out, r)
                P_eff_f = P_eff_f[..., :r]  # (K_f, out, r)

        if self._expert_diag:
            # Per-expert learnable singular spectrum (the magnitude DOF the
            # orthogonal-only frozen path lacks). Scales each expert's r cols.
            P_eff_c = P_eff_c * self.sigma_c.to(P_eff_c.dtype).unsqueeze(1)
            P_eff_f = P_eff_f * self.sigma_f.to(P_eff_f.dtype).unsqueeze(1)

        pi_c_w = pi_c.to(comp)
        pi_f_w = self._center(self._freq_gate_raw(lx_c.shape[0]), K_f).to(comp)

        P_combined_c = torch.einsum("bc,cor->bor", pi_c_w, P_eff_c.to(comp))
        P_combined_f = torch.einsum("bf,for->bor", pi_f_w, P_eff_f.to(comp))

        out = self._combine_up(
            lx_c.to(comp), lx_f.to(comp), P_combined_c, P_combined_f, scale_c
        )

        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)

    def regularization(self):
        """No-op on both paths: Cayley guarantees orthogonality structurally,
        and OrthoInit leaves the bases unconstrained. ``P_bases_c`` exists
        under either layout."""
        zero = torch.tensor(0.0, device=self.P_bases_c.device)
        return zero, zero

    # Save-pipeline hooks: dual-A distill + per-pool MoE writer.

    @classmethod
    def distill_save_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> None:
        """Chimera training-form → free-form per-pool (Cayley → lora_{down,up}).

        Mutates ``state_dict`` in place. Discriminator: co-located
        ``.Q_basis_c``/``.Q_basis_f`` keys — the ``_c``/``_f`` suffix pair is
        chimera-only (OrthoLoRA fallbacks use a bare ``.Q_basis``), covering
        both the Cayley path (frozen buffers + ``.S_q_c``) and OrthoInit
        (trainable bases, no ``S_*``). Must run FIRST in the save pipeline so
        subsequent converters see a chimera-free state_dict.

        Per pool, distill the Cayley-rotated SVD layout into free-form
        (``.lora_down_{c,f}.weight``, ``.lora_up_{c,f}_weight``);
        :meth:`build_moe_state_dict` then expands the stacked ups into
        per-expert ``.lora_ups_{c,f}.{i}.weight`` keys and q/k/v splits.
        """
        prefixes = set()
        for key in list(state_dict.keys()):
            if not key.endswith(".Q_basis_c"):
                continue
            prefix = key[: -len(".Q_basis_c")]
            if state_dict.get(f"{prefix}.Q_basis_f") is None:
                continue
            prefixes.add(prefix)

        for prefix in prefixes:
            # OrthoInit chimera carries no Cayley skew params; its trainable
            # bases ARE the effective factors (R = I).
            ortho_init = state_dict.get(f"{prefix}.S_q_c") is None
            Q_basis_c = state_dict[f"{prefix}.Q_basis_c"]
            Q_basis_f = state_dict[f"{prefix}.Q_basis_f"]
            P_bases_c = state_dict[f"{prefix}.P_bases_c"]  # (K_c, out, r)
            P_bases_f = state_dict[f"{prefix}.P_bases_f"]  # (K_f, out, r)
            lam_c = state_dict[f"{prefix}.lambda_c"]
            lam_f = state_dict[f"{prefix}.lambda_f"]
            alpha = state_dict.get(f"{prefix}.alpha")
            save_dtype = dtype if dtype is not None else P_bases_c.dtype

            r = lam_c.shape[-1]
            if ortho_init:
                Q_eff_c = Q_basis_c.float()  # (r, in)
                Q_eff_f = Q_basis_f.float()
                P_eff_c = P_bases_c.float()  # (K_c, out, r)
                P_eff_f = P_bases_f.float()
            else:
                S_q_c = state_dict[f"{prefix}.S_q_c"]
                S_q_f = state_dict[f"{prefix}.S_q_f"]
                S_p_c = state_dict[f"{prefix}.S_p_c"]  # (K_c, M, M)
                S_p_f = state_dict[f"{prefix}.S_p_f"]  # (K_f, M, M)
                R_q_c = cls._cayley(S_q_c.float())
                R_q_f = cls._cayley(S_q_f.float())
                R_p_c = cls._cayley(S_p_c.float())
                R_p_f = cls._cayley(S_p_f.float())
                Q_eff_c = R_q_c @ Q_basis_c.float()  # (r, in)
                Q_eff_f = R_q_f @ Q_basis_f.float()
                P_eff_c = P_bases_c.float() @ R_p_c  # (K_c, out, M)
                P_eff_f = P_bases_f.float() @ R_p_f
                if P_eff_c.shape[-1] != r:
                    # Over-complete Stiefel(M, r) select — mirror the forward.
                    P_eff_c = P_eff_c[..., :r]  # (K_c, out, r)
                    P_eff_f = P_eff_f[..., :r]

            # Fold the per-expert diagonal into the saved ups, so the
            # distilled free-form reproduces the trained forward exactly.
            sig_c = state_dict.get(f"{prefix}.sigma_c")
            sig_f = state_dict.get(f"{prefix}.sigma_f")
            if sig_c is not None:
                P_eff_c = P_eff_c * sig_c.float().unsqueeze(1)
            if sig_f is not None:
                P_eff_f = P_eff_f * sig_f.float().unsqueeze(1)

            def _split(P_eff, Q_eff, lam):
                lam_1d = lam.squeeze(0).float()
                lam_sqrt = lam_1d.abs().sqrt()
                lam_sign = lam_1d.sign()
                lora_down = (
                    (Q_eff * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()
                )
                lora_up_weight = (
                    (P_eff * (lam_sqrt * lam_sign).unsqueeze(0).unsqueeze(0))
                    .to(save_dtype)
                    .cpu()
                    .contiguous()
                )
                return lora_down, lora_up_weight

            lora_down_c, lora_up_c_weight = _split(P_eff_c, Q_eff_c, lam_c)
            lora_down_f, lora_up_f_weight = _split(P_eff_f, Q_eff_f, lam_f)

            for suffix in (
                "S_q_c",
                "S_q_f",
                "S_p_c",
                "S_p_f",
                "Q_basis_c",
                "Q_basis_f",
                "P_bases_c",
                "P_bases_f",
                "lambda_c",
                "lambda_f",
                "sigma_c",
                "sigma_f",
            ):
                state_dict.pop(f"{prefix}.{suffix}", None)

            state_dict[f"{prefix}.lora_down_c.weight"] = lora_down_c
            state_dict[f"{prefix}.lora_up_c_weight"] = lora_up_c_weight
            state_dict[f"{prefix}.lora_down_f.weight"] = lora_down_f
            state_dict[f"{prefix}.lora_up_f_weight"] = lora_up_f_weight
            if alpha is not None:
                state_dict[f"{prefix}.alpha"] = alpha

    @staticmethod
    def build_moe_state_dict(
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> Dict[str, torch.Tensor]:
        """Build the ``*_chimera.safetensors`` payload.

        Expects :meth:`distill_save_state_dict` to have already run. Two
        transforms: (1) expand stacked ``.lora_up_{c,f}_weight`` into
        per-expert ``.lora_ups_{c,f}.{i}.weight``; (2) per-pool fused-qkv
        defuse on attention prefixes (both pools share the prefix, so a
        fused frag splits both pools' down+ups per component; ``alpha``/
        ``inv_scale`` clone into each). ``content_router.*``/``freq_router.*``
        keys pass through untouched.

        Remaining fused-qkv prefixes after the per-pool split are the
        OrthoLoRA attention fallbacks (excluded from ``router_targets``,
        already distilled to plain LoRA) — run through the shared
        :func:`defuse_standard_qkv` so they land in the split q/k/v layout
        ComfyUI's cosmos backbone expects.
        """
        sd: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            v = v.detach().clone().to("cpu")
            if k.endswith(".lora_up_c_weight"):
                prefix = k.removesuffix(".lora_up_c_weight")
                for i in range(v.size(0)):
                    sd[f"{prefix}.lora_ups_c.{i}.weight"] = v[i]
            elif k.endswith(".lora_up_f_weight"):
                prefix = k.removesuffix(".lora_up_f_weight")
                for j in range(v.size(0)):
                    sd[f"{prefix}.lora_ups_f.{j}.weight"] = v[j]
            else:
                sd[k] = v

        # Per-pool q/k/v split. Detect by either pool's down key (both
        # should be present per chimera prefix; iterating one set is
        # sufficient).
        fused_groups: List[tuple] = []
        for key in list(sd.keys()):
            if not key.endswith(".lora_down_c.weight"):
                continue
            prefix = key.removesuffix(".lora_down_c.weight")
            spec = match_fused_spec(prefix)
            if spec is not None:
                fused_groups.append((prefix, spec))

        for prefix, spec in fused_groups:
            suffixes = spec.component_letters
            n = len(suffixes)
            down_c = sd.pop(f"{prefix}.lora_down_c.weight")
            down_f = sd.pop(f"{prefix}.lora_down_f.weight")
            alpha = sd.pop(f"{prefix}.alpha", None)
            inv_scale = sd.pop(f"{prefix}.inv_scale", None)

            ups_c_keys = sorted(
                (
                    k
                    for k in list(sd.keys())
                    if k.startswith(f"{prefix}.lora_ups_c.") and k.endswith(".weight")
                ),
                key=lambda k: int(
                    k.removeprefix(f"{prefix}.lora_ups_c.").removesuffix(".weight")
                ),
            )
            ups_f_keys = sorted(
                (
                    k
                    for k in list(sd.keys())
                    if k.startswith(f"{prefix}.lora_ups_f.") and k.endswith(".weight")
                ),
                key=lambda k: int(
                    k.removeprefix(f"{prefix}.lora_ups_f.").removesuffix(".weight")
                ),
            )
            ups_c = [sd.pop(k) for k in ups_c_keys]
            ups_f = [sd.pop(k) for k in ups_f_keys]
            ups_c_chunked = [u.chunk(n, dim=0) for u in ups_c]
            ups_f_chunked = [u.chunk(n, dim=0) for u in ups_f]

            base_prefix = prefix.removesuffix(spec.fused_frag)
            for ci, letter in enumerate(suffixes):
                new_prefix = base_prefix + spec.component_frag(letter)
                sd[f"{new_prefix}.lora_down_c.weight"] = down_c.clone()
                sd[f"{new_prefix}.lora_down_f.weight"] = down_f.clone()
                for ei, u_chunks in enumerate(ups_c_chunked):
                    sd[f"{new_prefix}.lora_ups_c.{ei}.weight"] = (
                        u_chunks[ci].contiguous().clone()
                    )
                for ei, u_chunks in enumerate(ups_f_chunked):
                    sd[f"{new_prefix}.lora_ups_f.{ei}.weight"] = (
                        u_chunks[ci].contiguous().clone()
                    )
                if alpha is not None:
                    sd[f"{new_prefix}.alpha"] = alpha.clone()
                if inv_scale is not None:
                    sd[f"{new_prefix}.inv_scale"] = inv_scale.clone()

        # Plain-LoRA leg defuse on any remaining fused attention prefixes.
        defuse_standard_qkv(sd)

        if dtype is not None:
            sd = {k: v.to(dtype) for k, v in sd.items()}
        return sd


class ChimeraHydraInferenceModule(_ChimeraRoutingMixin, BaseLoRAModule):
    """Free-form inference form of ChimeraHydra, loaded from a distilled
    ``*_chimera.safetensors`` — explicit per-pool (lora_down, stacked lora_up)
    instead of Cayley-rotated SVD bases, produced by
    :meth:`ChimeraHydraLoRAModule.distill_save_state_dict` at save time.

    Both pools' gates are recentered to ``π − 1/K`` before the combine (λ is
    folded symmetrically into the saved ups, so this reproduces the trained
    forward exactly). No T-LoRA mask at inference, consistent with the rest
    of the LoRA family.
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
        num_experts_content: int = 3,
        num_experts_freq: int = 3,
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

        K_c = int(num_experts_content)
        K_f = int(num_experts_freq)
        r = int(lora_dim)
        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.num_experts_content = K_c
        self.num_experts_freq = K_f
        self.num_experts = K_c + K_f
        self.in_dim = in_dim

        # Free-form down-projections (one per pool). Initialized empty;
        # actual weights overwritten by load_state_dict.
        self.lora_down_c = torch.nn.Linear(in_dim, r, bias=False)
        self.lora_down_f = torch.nn.Linear(in_dim, r, bias=False)
        # Stacked B's, fused (K_*, out, r). Loader expands per-expert
        # ``.lora_ups_*.{i}.weight`` into these stacks before calling
        # load_state_dict — see factory.create_network_from_weights.
        self.lora_up_c_weight = torch.nn.Parameter(torch.zeros(K_c, out_dim, r))
        self.lora_up_f_weight = torch.nn.Parameter(torch.zeros(K_f, out_dim, r))

        if channel_scale is not None:
            self._register_channel_scale(self.lora_down_c.weight.data, channel_scale)
            _absorb_channel_scale(self.lora_down_f.weight.data, channel_scale)

        self._register_routing_buffers(K_c, K_f)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        x_lora = self._rebalance(x)
        x_f32 = x_lora.float()
        lx_c = torch.nn.functional.linear(x_f32, self.lora_down_c.weight.float())
        lx_f = torch.nn.functional.linear(x_f32, self.lora_down_f.weight.float())

        # Centered-gate parity with training: subtract per-pool 1/K. λ is
        # already baked into the saved ups, so this reproduces the trained
        # ``Σ (π[k] - 1/K)·B[k]`` combine exactly.
        pi_c = self._center(
            self._content_gate_raw(lx_c.shape[0]), self.num_experts_content
        )
        pi_f = self._center(
            self._freq_gate_raw(lx_c.shape[0]).float(), self.num_experts_freq
        )

        if self.dropout is not None and self.training:
            lx_c = torch.nn.functional.dropout(lx_c, p=self.dropout)
            lx_f = torch.nn.functional.dropout(lx_f, p=self.dropout)
        lx_c, scale_c = self._apply_rank_dropout(lx_c)
        lx_f, scale_f = self._apply_rank_dropout(lx_f)

        # Gate-weighted up projection per pool.
        comb_c = torch.einsum(
            "bc,cor->bor", pi_c.float(), self.lora_up_c_weight.float()
        )
        comb_f = torch.einsum(
            "bf,for->bor", pi_f.float(), self.lora_up_f_weight.float()
        )

        out = self._combine_up(lx_c, lx_f, comb_c, comb_f, scale_c)

        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)
