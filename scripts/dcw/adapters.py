"""Adapter attach (LoRA / HydraLoRA). Mirrors library/inference/models.py."""

from __future__ import annotations

import logging

import torch
from safetensors.torch import load_file

from library.inference.models import _is_hydra_moe

log = logging.getLogger("dcw-bench")


def attach_loras(
    anima,
    paths: list[str],
    mults: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    from networks import lora_anima

    hydra_flags = [_is_hydra_moe(p) for p in paths]
    if any(hydra_flags) and not all(hydra_flags):
        raise SystemExit(
            "Mixing HydraLoRA moe files with regular LoRA files in --lora_weight "
            "is not supported (matches inference-time restriction)."
        )
    any_hydra = any(hydra_flags)
    log.info(
        f"attaching {len(paths)} adapter(s) as "
        f"{'router-live HydraLoRA' if any_hydra else 'LoRA'} hooks…"
    )

    if len(mults) == 1:
        mults = mults * len(paths)
    if len(mults) != len(paths):
        raise SystemExit(
            f"--lora_multiplier has {len(mults)} entries but --lora_weight has "
            f"{len(paths)}. Pass one multiplier per weight, or one shared."
        )

    for path, mult in zip(paths, mults):
        sd = load_file(path)
        sd = {k: v for k, v in sd.items() if k.startswith("lora_unet_")}
        network, weights_sd = lora_anima.create_network_from_weights(
            multiplier=mult,
            file=None,
            ae=None,
            text_encoders=[],
            unet=anima,
            weights_sd=sd,
            for_inference=True,
        )
        network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
        info = network.load_state_dict(weights_sd, strict=False)
        if info.unexpected_keys:
            log.warning(f"{path}: unexpected (first 5): {info.unexpected_keys[:5]}")
        if info.missing_keys:
            log.warning(f"{path}: missing (first 5): {info.missing_keys[:5]}")
        network.to(device, dtype=dtype)
        network.eval().requires_grad_(False)
        if any_hydra:
            hydra_networks = list(getattr(anima, "_hydra_networks", []))
            hydra_networks.append(network)
            anima._hydra_networks = hydra_networks
            anima._hydra_network = network
        log.info(f"  attached {path} (mult={mult}, modules={len(network.unet_loras)})")
