"""Inference-time adapter state helpers."""

from collections.abc import Iterable
from typing import Any, Optional

import torch

from library.runtime.fei import compute_fei_2band, fei_sigma_low


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return value
    return (value,)


def iter_hydra_networks(model: Any) -> Iterable[Any]:
    """Yield attached HydraLoRA networks without duplicating aliases."""
    candidates = []
    containers = [model]
    orig_mod = getattr(model, "_orig_mod", None)
    if orig_mod is not None and orig_mod is not model:
        containers.append(orig_mod)

    for container in containers:
        candidates.extend(_as_iterable(getattr(container, "_hydra_networks", None)))
        candidates.extend(_as_iterable(getattr(container, "_hydra_network", None)))

        # Accept the aliased P-GRAFT slot only for routing-aware nets (σ/FEI) so
        # regular P-GRAFT LoRAs stay untouched.
        pgraft_network = getattr(container, "_pgraft_network", None)
        if getattr(pgraft_network, "use_sigma_router", False) or getattr(
            pgraft_network, "use_fei_router", False
        ):
            candidates.append(pgraft_network)

    seen = set()
    for network in candidates:
        if network is None:
            continue
        ident = id(network)
        if ident in seen:
            continue
        seen.add(ident)
        yield network


def set_hydra_sigma(model: Any, timesteps: torch.Tensor) -> None:
    """Propagate current denoising sigma to router-live HydraLoRA adapters."""
    sigma = timesteps.detach().flatten().to(dtype=torch.float32)
    for network in iter_hydra_networks(model):
        set_sigma = getattr(network, "set_sigma", None)
        if callable(set_sigma):
            set_sigma(sigma)


def set_xattn_gain(model: Any, gain: float) -> None:
    """Set the per-block cross-attn residual gain.

    Writes each Block's non-persistent ``_xattn_gain`` buffer in place so
    compiled block graphs pick the new value up without a recompile. 1.0
    restores exact identity; callers gating on σ must reset it themselves
    (and before any uncond forward when boosting the cond pass only).
    """
    for block in model.blocks:
        block._xattn_gain.fill_(gain)


def set_xattn_renorm(
    model: Any, on: bool, *, per_token: bool = True, frac: float = 1.0
) -> None:
    """Toggle norm-matched cross-attn gain.

    While True, each Block rescales its post-cross-attn hidden state back
    toward the gain-1.0 norm, so ``set_xattn_gain`` rotates the state
    toward the cross-attn residual without leaving the norm shell.
    ``per_token=False`` matches the per-image MEAN token norm instead of
    each token's (per-token matching flattens the token-norm distribution →
    grey tone, muted highlights). ``frac`` ρ
    applies ``scale**ρ`` (1.0 = full correction, 0.0 = raw boost). Same
    cond-only discipline as :func:`set_xattn_gain` — toggle together with
    the gain and reset both before uncond forwards.
    """
    for block in model.blocks:
        block._xattn_renorm = on
        block._xattn_renorm_pertoken = per_token
        block._xattn_renorm_frac = frac


def set_xattn_boost_state(
    model: Any, gain: float, *, renorm_mode: str = "off", frac: float = 1.0
) -> None:
    """Set the cross-attn boost gain + its norm matching in one call.

    The single entry point denoise-loop runners should use for
    ``--xattn_boost`` (gain λ on the cond forward at σ ≥ band, reset with
    ``gain=1.0`` before uncond forwards and on loop exit). ``renorm_mode``:
    ``"img"`` (shipped default — per-image mean-norm matching, keeps the
    token-norm peaks that carry neon/highlights), ``"tok"`` (per-token,
    flattens peaks → grey tone; bench reference), ``"off"`` (raw gain).
    ``frac`` is the partial exponent ρ (scale**ρ). Renorm is forced off at
    gain 1.0 so a reset always restores exact identity.
    """
    set_xattn_gain(model, gain)
    on = renorm_mode != "off" and gain != 1.0
    set_xattn_renorm(
        model,
        on,
        per_token=renorm_mode == "tok",
        frac=frac if on else 1.0,
    )


def set_xattn_kbias(model: Any, bias: Optional[torch.Tensor]) -> None:
    """Set (or clear, with ``None``) the cross-attn per-key logit bias.

    ``bias`` is a ``(L_ctx,)`` float tensor added to every cross-attn QK^T
    row (allocation probe). Writes each
    Block's ``cross_attn._ctx_k_bias`` buffer; ``None`` restores exact
    identity (and the fused flash path — a set bias drops that call to
    SDPA). Same cond-only discipline as :func:`set_xattn_gain`: callers
    must clear it before any uncond forward.
    """
    for block in model.blocks:
        block.cross_attn._ctx_k_bias = bias


def clear_hydra_sigma(model: Any) -> None:
    """Clear cached sigma from router-live HydraLoRA adapters."""
    for network in iter_hydra_networks(model):
        clear_sigma = getattr(network, "clear_sigma", None)
        if callable(clear_sigma):
            clear_sigma()


def set_hydra_fei(model: Any, fei: torch.Tensor) -> None:
    """Propagate per-sample FEI ``[B, 2]`` to router-live HydraLoRA adapters.

    Parallel to ``set_hydra_sigma``. ``fei`` is ``(B, fei_feature_dim)`` —
    the simplex energy from ``library.runtime.fei.compute_fei_2band``.
    Networks without a FEI router silently ignore the call (no
    ``set_fei`` attribute).
    """
    fei = fei.detach().to(dtype=torch.float32)
    for network in iter_hydra_networks(model):
        set_fei = getattr(network, "set_fei", None)
        if callable(set_fei):
            set_fei(fei)


def clear_hydra_fei(model: Any) -> None:
    """Clear cached FEI from router-live HydraLoRA adapters."""
    for network in iter_hydra_networks(model):
        clear_fei = getattr(network, "clear_fei", None)
        if callable(clear_fei):
            clear_fei()


def _resolve_fei_sigma_low_div(model: Any) -> Optional[float]:
    """First fei-aware network's ``cfg.fei_sigma_low_div``, or None if no
    FEI router is attached.

    Used by ``compute_and_set_hydra_fei`` to pick σ_low without the caller
    having to thread it through. All currently-shipped variants share one
    σ_low_div across the whole network.
    """
    for network in iter_hydra_networks(model):
        if not getattr(network, "use_fei_router", False):
            continue
        cfg = getattr(network, "cfg", None)
        if cfg is None:
            continue
        return float(getattr(cfg, "fei_sigma_low_div", 8.0))
    return None


def set_hydra_content(model: Any, crossattn_emb: torch.Tensor) -> None:
    """Fire the ChimeraHydra ContentRouter on a pooled text vector.

    Mirrors :func:`set_hydra_fei`. ``crossattn_emb`` is the post-LLM-adapter
    text feature tensor (B, L, D) — the same one fed into the DiT's
    cross-attention. No-op on networks without a ContentRouter (chimera off).

    Call BEFORE each forward in the denoising loop, separately for cond
    and uncond branches — the two have different captions and therefore
    different ``π_c`` gates. The freq router has no cond/uncond asymmetry
    (FEI is identical) so it fires once per step.
    """
    for network in iter_hydra_networks(model):
        if not getattr(network, "use_content_router", False):
            continue
        set_content = getattr(network, "set_content", None)
        if callable(set_content):
            set_content(crossattn_emb)


def clear_hydra_content(model: Any) -> None:
    for network in iter_hydra_networks(model):
        clear_content = getattr(network, "clear_content_routing_weights", None)
        if callable(clear_content):
            clear_content()


def set_hydra_crossattn(model: Any, crossattn_emb: torch.Tensor) -> None:
    """Fire the network-level GlobalRouter on a pooled text vector.

    Mirrors :func:`set_hydra_content`, but for the non-chimera Hydra / FeRA
    pool routed on text (``router_source="crossattn_emb"``,
    ``route_per_layer=False``). ``crossattn_emb`` is the post-LLM-adapter
    feature tensor (B, L, D) fed into the DiT's cross-attention. No-op on
    networks without a crossattn GlobalRouter.

    Call BEFORE each forward, separately for cond and uncond branches — the
    two have different captions and therefore different gates.
    """
    for network in iter_hydra_networks(model):
        if not getattr(network, "use_crossattn_router", False):
            continue
        set_crossattn = getattr(network, "set_crossattn_routing", None)
        if callable(set_crossattn):
            set_crossattn(crossattn_emb)


def iter_step_expert_networks(model: Any) -> Iterable[Any]:
    """Yield attached per-step-expert turbo networks (deduped, compile-aware)."""
    candidates = []
    containers = [model]
    orig_mod = getattr(model, "_orig_mod", None)
    if orig_mod is not None and orig_mod is not model:
        containers.append(orig_mod)
    for container in containers:
        candidates.extend(
            _as_iterable(getattr(container, "_step_expert_networks", None))
        )
    seen = set()
    for network in candidates:
        if network is None:
            continue
        ident = id(network)
        if ident in seen:
            continue
        seen.add(ident)
        yield network


def set_step_expert_index(model: Any, step: int) -> None:
    """Select the active step-expert up-head for the current denoise step.

    No-op when no per-step-expert turbo network is attached. The head index is
    clamped to ``[0, K-1]`` so an ``infer_steps`` that overshoots the trained
    head count repeats the last (quality) head rather than raising — the test
    command warns on the mismatch (see ``cmd_test_turbo``).
    """
    for network in iter_step_expert_networks(model):
        set_idx = getattr(network, "set_step_index", None)
        if not callable(set_idx):
            continue
        K = 1
        for lora in getattr(network, "unet_loras", []):
            K = max(K, int(getattr(lora, "K", 1)))
        set_idx(min(step, K - 1))


def compute_and_set_hydra_fei(model: Any, z: torch.Tensor) -> None:
    """One-shot per-step FEI compute + propagate.

    ``z`` may be a 4D ``(B, C, H, W)`` latent or Anima's 5D
    ``(B, C, T, H, W)`` (with ``T == 1``) — the singleton temporal dim is
    squeezed automatically. No-op when no attached network has a FEI router
    (so non-FEI variants pay zero overhead beyond the one ``getattr`` per
    network).
    """
    div = _resolve_fei_sigma_low_div(model)
    if div is None:
        return
    if z.dim() == 5:
        z = z.squeeze(2)
    h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
    sigma_low = fei_sigma_low(h_lat, w_lat, div)
    fei = compute_fei_2band(z, sigma_low)
    set_hydra_fei(model, fei)
