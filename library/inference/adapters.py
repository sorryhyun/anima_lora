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

    Writes each Block's ``_xattn_gain`` buffer in place so compiled block
    graphs pick it up without a recompile. 1.0 restores identity; callers
    gating on σ reset it themselves, and before any uncond forward.
    """
    for block in model.blocks:
        block._xattn_gain.fill_(gain)


def set_xattn_renorm(
    model: Any, on: bool, *, per_token: bool = True, frac: float = 1.0
) -> None:
    """Toggle norm-matched cross-attn gain.

    While on, each Block rescales its post-cross-attn state back toward the
    gain-1.0 norm, so the gain rotates the state without leaving the norm
    shell. ``per_token=False`` matches the per-image mean token norm.
    ``frac`` ρ applies ``scale**ρ``. Reset together with the gain before
    uncond forwards.
    """
    for block in model.blocks:
        block._xattn_renorm = on
        block._xattn_renorm_pertoken = per_token
        block._xattn_renorm_frac = frac


def set_xattn_boost_state(
    model: Any, gain: float, *, renorm_mode: str = "off", frac: float = 1.0
) -> None:
    """Set the cross-attn boost gain + its norm matching in one call.

    The entry point runners use for ``--xattn_boost``: gain λ on the cond
    forward at σ ≥ band, reset with ``gain=1.0`` before uncond forwards and on
    loop exit. ``renorm_mode``: ``"img"`` (default, per-image mean-norm; keeps
    the token-norm peaks that carry highlights), ``"tok"`` (per-token, flattens
    them into a grey tone), ``"off"`` (raw gain). ``frac`` is the exponent ρ.
    Renorm is forced off at gain 1.0 so a reset restores identity.
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

    ``bias`` is ``(L_ctx,)`` (added to every QK^T row) or ``(N_q, L_ctx)``
    (one row per image token, block-flattened order). A set bias drops the
    call from flash to SDPA; ``None`` restores both. Clear it before any
    uncond forward.
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

    Networks without a ``set_fei`` ignore the call.
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
    """First FEI-router network's ``cfg.fei_sigma_low_div``, or None."""
    for network in iter_hydra_networks(model):
        if not getattr(network, "use_fei_router", False):
            continue
        cfg = getattr(network, "cfg", None)
        if cfg is None:
            continue
        return float(getattr(cfg, "fei_sigma_low_div", 4.0))
    return None


def set_hydra_crossattn(model: Any, crossattn_emb: torch.Tensor) -> None:
    """Fire the network-level GlobalRouter on a pooled text vector.

    For ``router_source="crossattn_emb"``, ``route_per_layer=False``.
    ``crossattn_emb`` is the post-LLM-adapter (B, L, D) tensor. Call before
    each forward, separately for cond and uncond: their gates differ.
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

    Clamped to ``[0, K-1]``: an ``infer_steps`` past the trained head count
    repeats the last head (``cmd_test_turbo`` warns).
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

    ``z`` is 4D ``(B, C, H, W)`` or Anima's 5D ``(B, C, 1, H, W)`` (squeezed
    at dim 2). No-op without a FEI router.
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
