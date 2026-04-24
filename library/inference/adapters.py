"""Inference-time adapter state helpers."""

from collections.abc import Iterable
from typing import Any

import torch


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

        # Hydra inference also aliases the same network into the P-GRAFT slot for
        # cutoff support. Keep this fallback for older call sites, but only accept
        # sigma-aware networks so regular P-GRAFT LoRAs remain untouched.
        pgraft_network = getattr(container, "_pgraft_network", None)
        if getattr(pgraft_network, "use_sigma_router", False):
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


def clear_hydra_sigma(model: Any) -> None:
    """Clear cached sigma from router-live HydraLoRA adapters."""
    for network in iter_hydra_networks(model):
        clear_sigma = getattr(network, "clear_sigma", None)
        if callable(clear_sigma):
            clear_sigma()
