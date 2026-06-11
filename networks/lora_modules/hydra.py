# HydraLoRA: MoE-style multi-head LoRA with layer-local routing.

import math
from typing import Dict, List, Optional

import torch

from networks.attn_fuse import match_fused_spec
from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.router_state import (
    RouterStateMixin,
    _apply_sigma_band_mask,
    _register_sigma_band_partition,
    _sigma_sinusoidal_features,
)

# Re-exported through ``networks.lora_modules.__init__``; ``network.py``
# imports ``_sigma_sinusoidal_features`` from there.
__all__ = [
    "HydraLoRAModule",
    "_apply_sigma_band_mask",
    "_sigma_sinusoidal_features",
]


class HydraLoRAModule(RouterStateMixin, BaseLoRAModule):
    """HydraLoRA: shared lora_down + per-expert lora_up, layer-local routing.

    See docs/methods/hydra-lora.md.

    Routing inputs (concatenated into the router's input):
      * pooled rank-R signal (always)
      * sinusoidal(σ) when ``sigma_feature_dim > 0``
      * FEI(z_t) when ``fei_feature_dim > 0``

    σ goes into the router *input* (not as additive bias) so its gradient
    survives even when ``score_e`` is near-uniform — a bias-only path's
    ``dL/d logits · d_sigma_feat`` vanishes during the cold-start window
    where the router has nothing to differentiate.

    ``use_global_router`` (shared_A + ``route_per_layer=False``): the per-layer
    router is dropped; gates arrive via the broadcast ``_routing_weights``
    buffer from the network-level GlobalRouter. σ-band partition is rejected
    in this mode (no local logits to mask). Balance loss is silently inert —
    ``_last_gate`` is the detached broadcast.

    ``num_experts_content > 0`` (ChimeraHydra runtime form): the E experts
    split into a content pool (``num_experts_content``, routed by the local
    per-Linear router on pooled rank-R) and a freq pool (``num_experts_freq
    = num_experts - num_experts_content``, routed by the network-level
    FreqRouter through a separate broadcast buffer ``_freq_routing_weights``).
    The router is narrowed to K_c outputs; ``_compute_gate`` cats
    ``[π_c | π_f]`` into the full (B, E) gate. σ/FEI feature dims must be 0
    in this mode (the FreqRouter owns those axes — see chimera.py docstring).
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
        expert_init_std: float = 0.0,
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: int = 1,
        sigma_bucket_boundaries: Optional[List[float]] = None,
        fei_feature_dim: int = 0,
        use_global_router: bool = False,
        num_experts_content: int = 0,
        use_global_content_router: bool = False,
        centered_gate: bool = False,
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
        # Centered-gate runtime parity: a checkpoint distilled from an
        # OrthoHydra trained with ``ortho_centered_gate`` combined experts with
        # ``(g_e - 1/E)`` rather than the raw softmax. λ is folded symmetrically
        # into the saved per-expert ups, so reproducing it here is exactly
        # ``gate -= 1/E`` before the combine. Single-pool only (never chimera —
        # the concat gate isn't a single E-simplex). See ortho.py distill note.
        self._centered_gate = bool(centered_gate)

        # Shared down projection.
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        # Per-expert up projections, fused: (E, out, r). Zero-init → ΔW=0 at
        # step 0. `expert_init_std` is the paper-baseline knob (Tian et al.
        # NeurIPS'24); production runs leave it at 0.0.
        self.lora_up_weight = torch.nn.Parameter(
            torch.zeros(num_experts, out_dim, self.lora_dim)
        )
        if expert_init_std > 0.0:
            torch.nn.init.normal_(self.lora_up_weight, mean=0.0, std=expert_init_std)

        self.use_global_router = bool(use_global_router)
        # ChimeraHydra dual-pool flag (load-time form): the per-Linear router
        # produces K_c content gates; freq gates arrive via FreqRouter
        # broadcast. Validated invariants:
        #   * num_experts_content > 0 ⇒ num_experts_freq = E - K_c > 0
        #   * Mutually exclusive with use_global_router (chimera owns its own
        #     network-level router; FeRA's GlobalRouter is a different surface).
        #   * σ/FEI feature dims must be 0 (FreqRouter takes those axes).
        self.num_experts_content = int(num_experts_content)
        self.num_experts_freq = (
            num_experts - self.num_experts_content
            if self.num_experts_content > 0
            else 0
        )
        self.use_global_content_router = bool(use_global_content_router)
        if self.num_experts_content > 0:
            if self.num_experts_freq <= 0:
                raise ValueError(
                    f"num_experts_content={self.num_experts_content} must be < "
                    f"num_experts={num_experts} (freq pool would be empty)."
                )
            if self.use_global_router:
                raise ValueError(
                    "num_experts_content > 0 is incompatible with "
                    "use_global_router=True (chimera owns its own freq router)."
                )
            if int(sigma_feature_dim) > 0 or int(fei_feature_dim) > 0:
                raise ValueError(
                    "num_experts_content > 0 requires sigma_feature_dim == 0 and "
                    "fei_feature_dim == 0 — those axes belong to the FreqRouter."
                )
        elif self.use_global_content_router:
            raise ValueError(
                "use_global_content_router=True requires num_experts_content > 0 "
                "(global content router only runs on the chimera content pool)."
            )
        # Local router on pooled rank-R (not raw in_dim): raw DiT inputs have
        # 80–96× DC-bias outliers + 4096 tokens, mean-pool collapses to DC and
        # the router gets no trainable gradient. lora_down is trained jointly,
        # so signal-carrying directions accumulate in rank-R space without
        # outlier saturation. See docs/methods/hydra-lora.md §Fixes.
        if self.use_global_router:
            self.sigma_feature_dim = 0
            self.fei_feature_dim = 0
        elif self.use_global_content_router:
            # Chimera load form with the network-level ContentRouter active.
            # Per-Linear router is absent on disk; π_c arrives via the
            # ``_content_routing_weights`` slot-assigned buffer below.
            self.sigma_feature_dim = 0
            self.fei_feature_dim = 0
            self.router = None
        else:
            self.sigma_feature_dim = int(sigma_feature_dim)
            # FEI default (fei_dim=2) = raw 2-band simplex (e_low, e_high) from
            # library.runtime.fei.compute_fei_2band. See
            # ``[[project_fera_probe_2band_decision]]``.
            self.fei_feature_dim = int(fei_feature_dim)
            router_in_dim = (
                self.lora_dim + self.sigma_feature_dim + self.fei_feature_dim
            )
            # Chimera narrows the router to K_c outputs (its forward output IS
            # π_c); plain Hydra keeps the standard E-output router.
            router_out_dim = (
                self.num_experts_content
                if self.num_experts_content > 0
                else num_experts
            )
            self.router = torch.nn.Linear(router_in_dim, router_out_dim, bias=True)
            # Split init: small-std on rank-R columns, zeros on σ/FEI columns.
            # Step-0 gate matches σ=off+FEI=off; conditioning emerges as those
            # columns train.
            with torch.no_grad():
                self.router.weight.zero_()
                torch.nn.init.normal_(self.router.weight[:, : self.lora_dim], std=0.01)
                self.router.bias.zero_()

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        self._last_gate = None  # (B, E), cached each forward for balance loss
        # σ / FEI / routing-weights placeholders: always-a-Tensor invariant +
        # pointer-stable buffers (see router_state.py). Routes through the
        # registration helper so the cat / branch in ``_compute_gate`` runs
        # unconditionally — no None-vs-Tensor guard under torch.compile.
        self._register_router_io_buffers(num_experts)
        # ChimeraHydra freq-pool gate buffer. Uniform 1/K_f placeholder; the
        # network-level FreqRouter overwrites via direct slot assignment
        # (``set_freq_routing_weights`` — no .detach()/.copy_(), grad_fn
        # preserved). Non-persistent (state_dict re-derives on construction).
        if self.num_experts_content > 0:
            placeholder = torch.full(
                (1, self.num_experts_freq),
                1.0 / max(self.num_experts_freq, 1),
                dtype=torch.float32,
            )
            self.register_buffer("_freq_routing_weights", placeholder, persistent=False)
            # Content-pool gate buffer for the global-router path. Same
            # contract; placeholder uniform 1/K_c. Registered unconditionally
            # on chimera modules so ``_wire_shared_content_buffers`` can
            # identify them by buffer presence — per-Linear (default) form
            # still computes π_c from its own router and the buffer is dead.
            content_placeholder = torch.full(
                (1, self.num_experts_content),
                1.0 / max(self.num_experts_content, 1),
                dtype=torch.float32,
            )
            self.register_buffer(
                "_content_routing_weights", content_placeholder, persistent=False
            )
        # σ-band partition: experts split into num_sigma_buckets bands;
        # out-of-band logits masked to -inf before softmax, soft routing
        # within each band. Independent of σ-feature router. Incompatible
        # with use_global_router — no local logits to mask.
        if specialize_experts_by_sigma_buckets and self.use_global_router:
            raise ValueError(
                "specialize_experts_by_sigma_buckets is incompatible with "
                "use_global_router=True (no per-layer logits to mask). Pick "
                "one: per-layer σ partition, or network-level GlobalRouter."
            )
        self._sigma_band_partition: bool = bool(specialize_experts_by_sigma_buckets)
        if self._sigma_band_partition:
            _register_sigma_band_partition(
                self, num_experts, num_sigma_buckets, sigma_bucket_boundaries
            )

    def _compute_gate(self, lx: torch.Tensor) -> torch.Tensor:
        """RMS-pool rank-R signal, concat σ/FEI if enabled, router, softmax.

        RMS (not mean) pool: zero-mean activations don't cancel by √N over the
        L≈4096 sequence. Safe in rank-R space because lora_down strips the raw
        DiT 80–96× DC-bias outliers that break RMS in bf16 (see
        docs/methods/hydra-lora.md §Fixes).

        Under ``use_global_router`` ``lx`` is ignored — gate is the broadcast
        ``_routing_weights`` buffer.

        Under ``num_experts_content > 0`` (chimera) the router outputs K_c
        content gates; ``_freq_routing_weights`` provides K_f freq gates from
        the network-level FreqRouter; the two are concatenated into the
        full (B, E) gate.
        """
        if self.use_global_router:
            B = lx.shape[0] if lx.dim() >= 1 else 1
            w = self._routing_weights
            if w.dim() == 1:
                w = w.unsqueeze(0)
            return w.to(lx.dtype).expand(B, -1)
        if self.use_global_content_router:
            # Chimera global-content path: π_c is broadcast from the
            # network-level ContentRouter; π_f from the FreqRouter. No
            # per-Linear router call — ``self.router`` is None.
            B = lx.shape[0] if lx.dim() >= 1 else 1
            pi_c = self._content_routing_weights
            if pi_c.dim() == 1:
                pi_c = pi_c.unsqueeze(0)
            if pi_c.shape[0] == 1 and B > 1:
                pi_c = pi_c.expand(B, -1)
            pi_c = pi_c.to(lx.dtype)
            pi_f = self._freq_routing_weights
            if pi_f.dim() == 1:
                pi_f = pi_f.unsqueeze(0)
            pi_f = pi_f.to(pi_c.dtype).expand(pi_c.shape[0], -1)
            return torch.cat([pi_c, pi_f], dim=-1)
        if lx.dim() >= 3:
            B = lx.shape[0]
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        # lx is in the compute dtype (activation dtype at training, fp32 at
        # inference); router weights are in storage dtype.
        pooled = pooled.to(self.router.weight.dtype)
        parts = [pooled]
        if self.sigma_feature_dim > 0:
            # Placeholder (1, D) broadcasts to batch pre-set_sigma; expand is a
            # no-op once set_sigma rebinds to (B, D). Same rule for FEI below.
            sigma_feat = self._sigma_features.to(pooled.dtype).expand(
                pooled.shape[0], -1
            )
            parts.append(sigma_feat)
        if self.fei_feature_dim > 0:
            fei_feat = self._fei.to(pooled.dtype).expand(pooled.shape[0], -1)
            parts.append(fei_feat)
        router_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        logits = self.router(router_in)  # (B, K_c) under chimera, (B, E) otherwise
        if self._sigma_band_partition:
            logits = _apply_sigma_band_mask(
                logits, self._sigma, self._expert_band, self._sigma_edges
            )
        if self.num_experts_content > 0:
            # Chimera dual-pool: softmax each pool independently, concat.
            pi_c = torch.softmax(logits, dim=-1)  # (B, K_c)
            pi_f = self._freq_routing_weights
            if pi_f.dim() == 1:
                pi_f = pi_f.unsqueeze(0)
            pi_f = pi_f.to(pi_c.dtype).expand(pi_c.shape[0], -1)
            return torch.cat([pi_c, pi_f], dim=-1)  # (B, E)
        return torch.softmax(logits, dim=-1)

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign the freq pool's gates (preserves grad_fn).

        Direct slot assignment (NO .detach(), NO .copy_()) — the buffer
        must carry the FreqRouter's grad_fn so ``∂L/∂π_f`` reaches the
        FreqRouter parameters. Mirrors
        ``router_state._set_routing_weights`` and the original
        ChimeraHydraLoRAModule helper.
        """
        if self.num_experts_content <= 0:
            return
        buf = self._freq_routing_weights
        w = weights.to(dtype=buf.dtype, device=buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        self._freq_routing_weights = w

    def clear_freq_routing_weights(self) -> None:
        """Reset to uniform 1/K_f without rebinding the pointer."""
        if self.num_experts_content <= 0:
            return
        K_f = int(self._freq_routing_weights.shape[-1])
        self._freq_routing_weights.fill_(1.0 / max(K_f, 1))

    def set_content_routing_weights(self, weights: torch.Tensor) -> None:
        """Inference-side slot-assign for the chimera global-content path.

        Mirrors :meth:`set_freq_routing_weights`. No-op on non-chimera
        modules (those have no ``_content_routing_weights`` buffer).
        """
        if self.num_experts_content <= 0:
            return
        buf = self._content_routing_weights
        w = weights.to(dtype=buf.dtype, device=buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        self._content_routing_weights = w

    def clear_content_routing_weights(self) -> None:
        if self.num_experts_content <= 0:
            return
        K_c = int(self._content_routing_weights.shape[-1])
        self._content_routing_weights.fill_(1.0 / max(K_c, 1))

    # σ / FEI / routing-weights method surface (set_sigma / clear_sigma /
    # set_fei / clear_fei / set_routing_weights / clear_routing_weights) is
    # inherited from RouterStateMixin. The chimera dual-pool freq/content
    # setters above stay local — they wrap two extra buffers the mixin
    # doesn't know about.

    def forward(self, x):
        # Training computes the rank GEMMs in the activation dtype — under the
        # trainer's autocast(bf16) that is bit-identical to the retired
        # fp32-bottleneck path (autocast re-cast its ``.float()`` inputs back
        # to bf16 before every GEMM; see bench/lora_fp32_bottleneck), minus
        # the dead cast traffic. Inference (no autocast anywhere in the
        # engine) keeps the historical fp32 compute so router-live
        # checkpoints produce unchanged outputs.
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        # Training: compute in the model COMPUTE dtype (``org_forwarded.dtype`` =
        # the frozen base's output = the autocast/model dtype), not x.dtype.
        # ``x`` arrives fp32 from the AdaLN LayerNorm under autocast(bf16), so
        # ``comp = x.dtype`` left the rank path fp32 and let ``_rebalance``
        # allocate a fp32 activation (``inv_scale``) — OOM for no numeric gain
        # (autocast re-casts the GEMM to bf16 anyway). LoRA params are fp32
        # master, so cast x + weights DOWN to the base dtype. Inference keeps
        # fp32. See bench/lora_fp32_bottleneck.
        comp = org_forwarded.dtype if self.training else torch.float32
        x_lora = self._rebalance(x.to(comp))
        lx = torch.nn.functional.linear(x_lora, self.lora_down.weight.to(comp))

        # Gate from rank-R signal pre-mask/dropout — those are training-time
        # perturbations and the gate must behave identically at inference.
        gate = self._compute_gate(lx)  # (B, E)
        if self.training:
            # Plain STORE_ATTR (NOT @compiler.disable): a disabled helper
            # forces a graph break per LoRA forward and explodes
            # saved-for-backward memory under torch.compile (observed
            # OOM at 56 MoE + 140 OrthoLoRA modules on T4-class budget).
            self._last_gate = gate

        if self.training:
            lx = lx * self._timestep_mask

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # Centered-gate parity (single-pool only). The concat gate of a chimera
        # dual-pool form is not one E-simplex, so centering is gated off there.
        gate_eff = gate
        if self._centered_gate and self.num_experts_content == 0:
            gate_eff = gate - (1.0 / self.num_experts)

        # Gate-weighted up projection: (B, out, r) per batch element.
        combined = torch.einsum(
            "be,eod->bod", gate_eff.to(comp), self.lora_up_weight.to(comp)
        )
        orig_shape = lx.shape
        B = orig_shape[0]
        lx_3d = lx.reshape(B, -1, orig_shape[-1]).to(comp)
        out = torch.bmm(lx_3d, combined.transpose(1, 2))
        out = out.reshape(*orig_shape[:-1], -1)

        return org_forwarded + (out * self.multiplier * scale).to(org_forwarded.dtype)

    # ------------------------------------------------------------------
    # Save-pipeline hook. The training runtime keeps experts stacked under
    # ``.lora_up_weight (E, out, r)`` — ComfyUI's HydraLoRA custom node
    # expects per-expert ``.lora_ups.{i}.weight`` keys, so save expands
    # them here. Fused-qkv prefixes are split per-expert per-component.
    # ------------------------------------------------------------------

    @staticmethod
    def build_moe_state_dict(
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> Dict[str, torch.Tensor]:
        """Build the Hydra ``*_moe.safetensors`` payload.

        Expects the state_dict to already be in the training-runtime form
        (stacked ``.lora_up_weight``) — :meth:`OrthoHydraLoRAModule.distill_save_state_dict`
        runs first if the live checkpoint came from the ortho-hydra path.

        Two transforms:
          1. Expand ``.lora_up_weight (E, out, r)`` → per-expert
             ``.lora_ups.{i}.weight`` keys.
          2. Per-pool fused-qkv defuse on attention prefixes. ``lora_down``
             / ``alpha`` / ``router.*`` / ``sigma_mlp.*`` / ``inv_scale``
             are shared across q/k/v (same Linear input, same routing
             decision), so clone them into each split component. The
             plain-LoRA leg (modules excluded from ``router_targets``)
             gets its own per-component split — the fused qkv carries
             standard ``.lora_up.weight``.
        """
        hydra_sd: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            v = v.detach().clone().to("cpu")
            if k.endswith(".lora_up_weight"):
                prefix = k.removesuffix(".lora_up_weight")
                for i in range(v.size(0)):
                    hydra_sd[f"{prefix}.lora_ups.{i}.weight"] = v[i]
            else:
                hydra_sd[k] = v

        hydra_fused_groups: List[tuple] = []
        for key in list(hydra_sd.keys()):
            if not key.endswith(".lora_down.weight"):
                continue
            prefix = key.removesuffix(".lora_down.weight")
            spec = match_fused_spec(prefix)
            if spec is not None:
                hydra_fused_groups.append((prefix, spec))

        for prefix, spec in hydra_fused_groups:
            suffixes = spec.component_letters
            n = len(suffixes)
            down = hydra_sd.pop(f"{prefix}.lora_down.weight")
            alpha = hydra_sd.pop(f"{prefix}.alpha", None)
            router_w = hydra_sd.pop(f"{prefix}.router.weight", None)
            router_b = hydra_sd.pop(f"{prefix}.router.bias", None)
            inv_scale = hydra_sd.pop(f"{prefix}.inv_scale", None)
            sigma_mlp_keys = [
                k for k in list(hydra_sd.keys()) if k.startswith(f"{prefix}.sigma_mlp.")
            ]
            sigma_mlp_state = {k: hydra_sd.pop(k) for k in sigma_mlp_keys}

            ups_keys = sorted(
                (
                    k
                    for k in list(hydra_sd.keys())
                    if k.startswith(f"{prefix}.lora_ups.") and k.endswith(".weight")
                ),
                key=lambda k: int(
                    k.removeprefix(f"{prefix}.lora_ups.").removesuffix(".weight")
                ),
            )
            ups = [hydra_sd.pop(k) for k in ups_keys]
            ups_chunked = [u.chunk(n, dim=0) for u in ups]

            # Plain-LoRA leg (present when router_targets excluded this
            # module). Split these per-component so q/k/v keys are
            # consistent with the already-split ``.lora_down.weight`` above.
            plain_up = hydra_sd.pop(f"{prefix}.lora_up.weight", None)
            plain_up_chunks = plain_up.chunk(n, dim=0) if plain_up is not None else None

            base_prefix = prefix.removesuffix(spec.fused_frag)
            for ci, letter in enumerate(suffixes):
                new_prefix = base_prefix + spec.component_frag(letter)
                hydra_sd[f"{new_prefix}.lora_down.weight"] = down.clone()
                for ei, u_chunks in enumerate(ups_chunked):
                    hydra_sd[f"{new_prefix}.lora_ups.{ei}.weight"] = (
                        u_chunks[ci].contiguous().clone()
                    )
                if plain_up_chunks is not None:
                    hydra_sd[f"{new_prefix}.lora_up.weight"] = (
                        plain_up_chunks[ci].contiguous().clone()
                    )
                if alpha is not None:
                    hydra_sd[f"{new_prefix}.alpha"] = alpha.clone()
                if router_w is not None:
                    hydra_sd[f"{new_prefix}.router.weight"] = router_w.clone()
                if router_b is not None:
                    hydra_sd[f"{new_prefix}.router.bias"] = router_b.clone()
                if inv_scale is not None:
                    hydra_sd[f"{new_prefix}.inv_scale"] = inv_scale.clone()
                for k, v in sigma_mlp_state.items():
                    subkey = k.removeprefix(f"{prefix}.")
                    hydra_sd[f"{new_prefix}.{subkey}"] = v.clone()

        if dtype is not None:
            hydra_sd = {k: v.to(dtype) for k, v in hydra_sd.items()}
        return hydra_sd
