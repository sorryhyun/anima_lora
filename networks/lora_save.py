"""Save-pipeline orchestrator for the LoRA / Hydra family.

The per-variant save logic — MoE write layout, qkv defuse — lives on the
variant's module class in ``networks/lora_modules/``
(``HydraLoRAModule.build_moe_state_dict``, ``lora.defuse_and_bake_standard``).
This file is the thin layer that calls them and writes the resulting file.

The standard write path relays adaln keys from the runtime names to the
ComfyUI layout (``_relayout_adaln_to_comfy``), after the qkv defuse and
before hashing — so the shipped file is ComfyUI-native end to end. The Hydra
MoE variant returns early and is not ComfyUI-loadable regardless.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from library.log import setup_logging
from networks.lora_modules import HydraLoRAModule
from networks.lora_modules.lora import defuse_and_bake_standard

setup_logging()
logger = logging.getLogger(__name__)


def _relayout_adaln_to_comfy(
    state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Rename adaln LoRA keys from the in-repo runtime names
    (``adaln_up_{br}``) to the ComfyUI state-dict layout
    (``adaln_modulation_{br}_2``) and stamp ``ss_adaln_layout`` — see the
    layout note in ``networks/lora_utils.py``. The attn/MLP keys already ship
    in the defused split layout, so only the adaln keys move.

    Presence-gated — an adaln-less checkpoint is untouched, metadata and
    all. Mutates ``state_dict`` in place; returns the metadata to write
    (a dict is allocated if the stamp needs one and none was passed).
    """
    from networks.lora_utils import relayout_adaln_runtime_to_comfy

    renamed = relayout_adaln_runtime_to_comfy(state_dict)
    if renamed.keys() == state_dict.keys():
        return metadata  # no runtime adaln keys present — nothing to relayout

    state_dict.clear()
    state_dict.update(renamed)
    if metadata is None:
        metadata = {}
    metadata["ss_adaln_layout"] = "comfy"
    n_adaln = sum(
        1 for k in renamed if "adaln_modulation_" in k and k.endswith(".alpha")
    )
    logger.info(
        f"relaid {n_adaln} adaln modules to the ComfyUI layout "
        "(loads natively in ComfyUI; in-repo loader renames back on load)"
    )
    return metadata


def build_standard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    """Run the standard finalize chain WITHOUT writing a file.

    Defuse fused qkv + bake channel scaling, relay adaln keys to the ComfyUI
    layout, cast dtype. Returns the finalized ``(state_dict, metadata)``. Factored
    out of :func:`save_network_weights` so the dual-pool turbo save can finalize
    each pool to its on-disk plain-LoRA layout and concat the two exactly.
    """
    defuse_and_bake_standard(state_dict)
    metadata = _relayout_adaln_to_comfy(state_dict, metadata)
    if dtype is not None:
        for key in list(state_dict.keys()):
            state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)
    return state_dict, metadata


def save_network_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    save_variant: str,
) -> None:
    """Run the save pipeline: variant write.

    Mutates ``state_dict`` in place.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    # Variant dispatch:
    #   * hydra_moe: shared-A Hydra → *_moe.safetensors
    #   * standard: defuse qkv → *.safetensors
    # Auto-fallback: any surviving ``.lora_up_weight`` key implies a Hydra
    # payload — kept for callers that don't plumb ``save_variant`` through.
    is_hydra_variant = save_variant == "hydra_moe" or any(
        k.endswith(".lora_up_weight") for k in state_dict.keys()
    )

    if is_hydra_variant:
        hydra_file = os.path.splitext(file)[0] + "_moe.safetensors"
        hydra_sd = HydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(hydra_sd, hydra_file, metadata or {})
        logger.info(f"HydraLoRA full format saved to {hydra_file}")
        # The _moe file is the only useful artifact for HydraLoRA —
        # a uniform expert average defeats layer-local routing.
        return

    # Standard write path.
    state_dict, metadata = build_standard_state_dict(state_dict, dtype, metadata)

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)
