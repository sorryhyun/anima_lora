# Shared σ / FEI / routing-weights buffer protocol for HydraLoRAModule,
# OrthoHydraLoRAModule, and StackedExpertsLoRAModule. Lives here so the
# global-router gradient path stays identical across all three (the proposal
# flagged a drift where OrthoHydra had detached its routing buffer).
#
# Cross-module aliasing dance (`_wire_shared_*` / Module._apply recovery) is
# tangled with cudagraph pointer stability and stays in LoRANetwork.
#
# Buffer protocol:
#   * `_sigma` / `_sigma_features`: rebound by LoRANetwork.set_sigma.
#   * `_fei`: rebound by LoRANetwork.set_fei.
#   * `_routing_weights`: rebound by LoRANetwork.set_routing_weights via
#     direct slot assignment (NO .detach(), NO .copy_()) — the buffer must
#     carry the router's grad_fn so L_denoise backprop reaches GlobalRouter.
#     See FeRA eq. 6-7, 11.
#
# Pointer-stable placeholders + always-a-Tensor invariant → no None-vs-Tensor
# guards in the forwards, so routed paths stay compile-clean.

import math
from typing import List, Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────


def _copy_or_rebind_buffer(
    module: torch.nn.Module, name: str, value: torch.Tensor
) -> None:
    """In-place copy (pointer-preserving) when shape matches, else rebind.

    σ / FEI shape can drift across train/val batch sizes; in-place keeps
    cudagraph pointers stable on the steady-state path.
    """
    buf = getattr(module, name)
    if buf.shape == value.shape and buf.device == value.device:
        buf.copy_(value.to(buf.dtype))
    else:
        setattr(module, name, value.to(buf.dtype).clone())


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal σ features (shared with postfix-sigma, which inlines its own copy)
# ─────────────────────────────────────────────────────────────────────────────


# freqs depend only on (half_dim, device); cache to avoid emitting a fresh
# arange+exp per module per step.
_FREQS_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _sigma_sinusoidal_features(
    sigma: torch.Tensor, sigma_feature_dim: int
) -> torch.Tensor:
    """Sinusoidal σ features matching the DiT t_embedder functional form."""
    t = sigma.flatten().float()
    half_dim = sigma_feature_dim // 2
    key = (half_dim, t.device)
    freqs = _FREQS_CACHE.get(key)
    if freqs is None:
        exponent = (
            -math.log(10000)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / max(half_dim, 1)
        )
        freqs = torch.exp(exponent)
        _FREQS_CACHE[key] = freqs
    angles = t[:, None] * freqs[None, :]  # [B, half_dim]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def _fei_temperature(fei: torch.Tensor, tau: float) -> torch.Tensor:
    """Hardwired-FEI freq gate: ``π_f = normalize(FEI ** (1/τ))``.

    Used by the ChimeraHydra ``freq_router_mode="fei"`` path, which routes the
    freq pool directly by the FEI band-simplex instead of a learned MLP. FEI
    is already a normalized simplex, so τ=1.0 returns it unchanged
    (``softmax(log p) = p``). τ<1 sharpens the low/high crossover, τ>1 flattens
    it. The power form (rather than ``softmax(log FEI / τ)``) avoids ``log(0)``
    on the high-σ steps where ``e_low ≈ 0`` (``0 ** (1/τ) = 0``).
    """
    if abs(tau - 1.0) < 1e-8:
        return fei
    p = fei.clamp_min(0).pow(1.0 / max(tau, 1e-6))
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# σ feature cache
# ─────────────────────────────────────────────────────────────────────────────


def _register_sigma_feature_cache(
    module: torch.nn.Module, sigma_feature_dim: int
) -> None:
    module.register_buffer(
        "_sigma", torch.zeros(1, dtype=torch.float32), persistent=False
    )
    if sigma_feature_dim <= 0:
        return
    zero_feat = _sigma_sinusoidal_features(module._sigma, sigma_feature_dim)
    module.register_buffer("_sigma_features", zero_feat, persistent=False)


def _set_sigma_feature_cache(
    module: torch.nn.Module,
    sigmas: torch.Tensor,
    sigma_features: torch.Tensor | None = None,
) -> None:
    sigmas = sigmas.detach()
    _copy_or_rebind_buffer(module, "_sigma", sigmas)
    if getattr(module, "sigma_feature_dim", 0) <= 0:
        return
    if sigma_features is None:
        sigma_features = _sigma_sinusoidal_features(sigmas, module.sigma_feature_dim)
    _copy_or_rebind_buffer(module, "_sigma_features", sigma_features.detach())


def _clear_sigma_feature_cache(module: torch.nn.Module) -> None:
    module._sigma.zero_()
    if getattr(module, "sigma_feature_dim", 0) > 0:
        zero_feat = _sigma_sinusoidal_features(module._sigma, module.sigma_feature_dim)
        _copy_or_rebind_buffer(module, "_sigma_features", zero_feat)


# ─────────────────────────────────────────────────────────────────────────────
# FEI feature cache
# ─────────────────────────────────────────────────────────────────────────────


def _register_fei_feature_cache(module: torch.nn.Module, fei_feature_dim: int) -> None:
    """Register `_fei` placeholder. Width-1 zero when fei_feature_dim == 0
    keeps Module._apply parity with the σ side."""
    width = max(int(fei_feature_dim), 1)
    module.register_buffer(
        "_fei", torch.zeros(1, width, dtype=torch.float32), persistent=False
    )


def _set_fei_feature_cache(module: torch.nn.Module, fei: torch.Tensor) -> None:
    fei = fei.detach()
    _copy_or_rebind_buffer(module, "_fei", fei)


def _clear_fei_feature_cache(module: torch.nn.Module) -> None:
    module._fei.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# σ-band expert partition
# ─────────────────────────────────────────────────────────────────────────────


def _register_sigma_band_partition(
    module: torch.nn.Module,
    num_experts: int,
    num_sigma_buckets: int,
    sigma_bucket_boundaries: Optional[List[float]] = None,
) -> None:
    """Register `_expert_band` (E,) and `_sigma_edges` (B-1,) for σ-band routing.

    Interleaved band assignment (`e mod num_sigma_buckets`) — with sequential
    SVD slicing in OrthoHydra this gives every band a representative spread
    of singular slices instead of binding band 0 to the top slice.

    `sigma_bucket_boundaries` is optionally a length-(B+1) edge list (0.0 …
    1.0); interior B-1 cuts feed `torch.bucketize`. None defaults to uniform
    linspace.
    """
    band = torch.arange(num_experts, dtype=torch.long) % num_sigma_buckets
    module.register_buffer("_expert_band", band, persistent=False)
    if sigma_bucket_boundaries is None:
        edges = torch.linspace(0.0, 1.0, num_sigma_buckets + 1)
    else:
        edges = torch.tensor(list(sigma_bucket_boundaries), dtype=torch.float32)
    interior = edges[1:-1].contiguous()
    module.register_buffer("_sigma_edges", interior, persistent=False)
    module._sigma_num_buckets = int(num_sigma_buckets)


def _apply_sigma_band_mask(
    logits: torch.Tensor,
    sigma: torch.Tensor,
    expert_band: torch.Tensor,
    sigma_edges: torch.Tensor,
) -> torch.Tensor:
    """Mask out-of-band expert logits to -inf so softmax renormalises in-band.

    sigma may broadcast from (1,) when set_sigma hasn't fired this forward.
    torch.bucketize default (right=False) maps σ-on-edge to the upper bucket.
    """
    num_buckets = int(sigma_edges.numel()) + 1
    bucket_ids = torch.bucketize(sigma.float(), sigma_edges).clamp(0, num_buckets - 1)
    if bucket_ids.shape[0] == 1 and logits.shape[0] > 1:
        bucket_ids = bucket_ids.expand(logits.shape[0])
    in_band = bucket_ids[:, None] == expert_band[None, :]  # (B, E) bool
    return logits.masked_fill(~in_band, float("-inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Routing-weights buffer (network-level ``GlobalRouter`` broadcast target)
# ─────────────────────────────────────────────────────────────────────────────


def _register_routing_weights_buffer(module: torch.nn.Module, num_experts: int) -> None:
    """Pointer-stable `_routing_weights`, uniform 1/E placeholder.

    Forward gate-weighting branch runs unconditionally (no None guard under
    compile). LoRANetwork.set_routing_weights rebinds across every module via
    the shared-buffer aliasing protocol — see [[project_set_sigma_aliasing_bug]].
    """
    placeholder = torch.full(
        (1, num_experts),
        1.0 / max(int(num_experts), 1),
        dtype=torch.float32,
    )
    module.register_buffer("_routing_weights", placeholder, persistent=False)


def _set_routing_weights(module: torch.nn.Module, weights: torch.Tensor) -> None:
    """Replace `_routing_weights` with the live router output.

    Direct slot assignment (NOT .copy_()) and no .detach() — the buffer must
    carry the router's grad_fn so ∂L/∂α flows back to GlobalRouter. This is
    the FeRA gradient path (eq. 6-7, 11): α_t enters y_t as a live multiplier,
    so plain L_denoise backprop trains the router. Shared across all three
    routing-aware modules so the contract is identical regardless of layout.
    """
    buf = module._routing_weights
    w = weights.to(dtype=buf.dtype, device=buf.device)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    module._routing_weights = w


def _clear_routing_weights(module: torch.nn.Module) -> None:
    """Reset to uniform 1/E without rebinding the pointer."""
    E = int(module._routing_weights.shape[-1])
    module._routing_weights.fill_(1.0 / max(E, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Per-module method surface
# ─────────────────────────────────────────────────────────────────────────────


class RouterStateMixin:
    """Shared σ / FEI / routing-weights *method surface* for the routing-aware
    LoRA variants (HydraLoRA / OrthoHydra / StackedExperts).

    The free functions above own the buffer mechanics — pointer-stable rebind,
    the grad-carrying ``_routing_weights`` slot-assign contract (NO ``.detach()``
    / ``.copy_()`` so ``∂L/∂α`` reaches the GlobalRouter, FeRA eq. 6-7, 11).
    What was still pasted verbatim into each class was the thin method wrapping;
    this mixin holds it once, so a future router source is a one-place edit.

    Each setter is **buffer-presence-guarded** (``hasattr``): a module that
    registered only a subset of the buffers inherits the full surface as safe
    no-ops. StackedExperts (``_routing_weights`` only, no σ/FEI) is the case
    that matters — its ``set_sigma`` / ``set_fei`` become no-ops, exactly
    equivalent to never defining them (the network keys its ``_*_aware_loras``
    lists on *buffer* presence — ``network.py::_wire_shared_*`` — and every
    external caller probes by method name and tolerates a no-op). The
    ``_routing_weights`` guard also subsumes the old
    ``getattr(self, "use_global_router", False)`` check: that buffer is
    registered iff ``use_global_router`` on Hydra/OrthoHydra, and always on
    StackedExperts, so ``hasattr`` is the exact, layout-agnostic condition.

    Chimera carries two routing buffers (π_c / π_f) and a different method
    surface — it keeps ``_ChimeraRoutingMixin`` instead.
    """

    def set_sigma(
        self, sigmas: torch.Tensor, sigma_features: torch.Tensor | None = None
    ) -> None:
        if not hasattr(self, "_sigma"):
            return
        _set_sigma_feature_cache(self, sigmas, sigma_features)

    def clear_sigma(self) -> None:
        if not hasattr(self, "_sigma"):
            return
        _clear_sigma_feature_cache(self)

    def set_fei(self, fei: torch.Tensor) -> None:
        if not hasattr(self, "_fei"):
            return
        _set_fei_feature_cache(self, fei)

    def clear_fei(self) -> None:
        if not hasattr(self, "_fei"):
            return
        _clear_fei_feature_cache(self)

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        if not hasattr(self, "_routing_weights"):
            return
        _set_routing_weights(self, weights)

    def clear_routing_weights(self) -> None:
        if not hasattr(self, "_routing_weights"):
            return
        _clear_routing_weights(self)

    def _register_router_io_buffers(self, num_experts: int) -> None:
        """Register the σ / FEI / routing-weights placeholder buffers shared by
        the per-Linear-router variants (Hydra / OrthoHydra).

        Reads ``self.sigma_feature_dim`` / ``self.fei_feature_dim`` /
        ``self.use_global_router`` (all assigned before this call). The routing
        buffer is registered only under the global router — its presence is
        exactly what the ``set_routing_weights`` guard above keys on. σ-band
        partition and the chimera dual-pool buffers stay in each class (their
        registration is interleaved with class-specific validation / layout).
        """
        _register_sigma_feature_cache(self, self.sigma_feature_dim)
        _register_fei_feature_cache(self, self.fei_feature_dim)
        if self.use_global_router:
            _register_routing_weights_buffer(self, num_experts)
