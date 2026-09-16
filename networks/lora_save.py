"""Save-pipeline orchestrator for the LoRA / Ortho / Hydra family.

The per-variant save logic — Cayley distillation, MoE write layout,
qkv defuse — lives on the variant's module class in
``networks/lora_modules/`` (``OrthoLoRAModule.distill_save_state_dict``,
``HydraLoRAModule.build_moe_state_dict``, etc). This file is the thin
ordering layer that calls them and writes the resulting file(s).

Ordering of the conversion pipeline is load-bearing:

  1. ``ChimeraHydraLoRAModule.distill_save_state_dict``
     (gated on co-located ``.Q_basis_c`` + ``.Q_basis_f`` — covers both the
     Cayley and OrthoInit chimera parameterizations)
  2. ``StackedExpertsLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 3-D ``.S_q``)
  3. ``OrthoHydraLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 2-D ``.S_q``)
  4. ``OrthoLoRAModule.distill_save_state_dict``
     (gated on 2-D ``.S_p``)
  4b. ``OrthoInitLoRAModule.distill_save_state_dict``
     (gated on ``.P_init`` — a name no other variant uses, so order vs the
     ``.S_p``-keyed steps above is independent; placed here for readability)
  5. legacy sig-type OrthoLoRA → standard LoRA
     (gated on ``.base_lambda``; no live module class emits these keys —
     kept so old artifacts remain re-bakeable)

The ``.S_p`` / ``.S_q`` dimensionality is the discriminator — every step
checks both dims explicitly so the matchers never overlap on the same
prefix.

The standard write path then relays adaln keys from the runtime names to
the ComfyUI layout (``_relayout_adaln_to_comfy``), after the qkv defuse and
before hashing — so the shipped file is ComfyUI-native end to end. The MoE
variants return early and are not ComfyUI-loadable regardless.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from library.log import setup_logging
from networks.lora_modules import (
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    OrthoHydraLoRAModule,
    OrthoInitLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
)
from networks.lora_modules.lora import defuse_and_bake_standard

setup_logging()
logger = logging.getLogger(__name__)


# Legacy sig-type OrthoLoRA → standard LoRA via 2r-dim SVD (see step 5 above).


def _convert_legacy_ortho_to_lora(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".base_lambda"):
            prefixes.add(key[: -len(".base_lambda")])

    for prefix in prefixes:
        P = state_dict[f"{prefix}.p_layer.weight"]  # (out, r)
        Q = state_dict[f"{prefix}.q_layer.weight"]  # (r, in)
        lam = state_dict[f"{prefix}.lambda_layer"]
        P_base = state_dict[f"{prefix}.base_p_weight"]
        Q_base = state_dict[f"{prefix}.base_q_weight"]
        lam_base = state_dict[f"{prefix}.base_lambda"]
        alpha = state_dict.get(f"{prefix}.alpha")
        rank = Q.shape[0]

        # ΔW = P·diag(λ)·Q − P_base·diag(λ_base)·Q_base is rank ≤ 2r. SVD
        # works in the small 2r-dim column/row space instead of on the full
        # (out × in) matrix: ΔW = [P|P_base] @ M @ [Q; Q_base], then SVD of M.
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        save_dtype = dtype if dtype is not None else P.dtype

        P_cat = torch.cat([P, P_base], dim=1).float().to(svd_device)  # (out, 2r)
        Q_cat = torch.cat([Q, Q_base], dim=0).float().to(svd_device)  # (2r, in)
        lam_diag = torch.diag(lam.squeeze(0).float().to(svd_device))
        lam_base_diag = torch.diag(lam_base.squeeze(0).float().to(svd_device))

        M = torch.zeros(2 * rank, 2 * rank, device=svd_device)
        M[:rank, :rank] = lam_diag
        M[rank:, rank:] = -lam_base_diag

        Qp, Rp = torch.linalg.qr(P_cat)
        Qq, Rq = torch.linalg.qr(Q_cat.T)

        core = Rp @ M @ Rq.T
        Uc, Sc, Vhc = torch.linalg.svd(core)

        lora_up = (
            (Qp @ Uc[:, :rank] * Sc[:rank].sqrt().unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )
        lora_down = (
            (Sc[:rank].sqrt().unsqueeze(1) * Vhc[:rank, :] @ Qq.T)
            .to(save_dtype)
            .cpu()
            .contiguous()
        )

        for suffix in (
            "p_layer.weight",
            "q_layer.weight",
            "lambda_layer",
            "base_p_weight",
            "base_q_weight",
            "base_lambda",
        ):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


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


# Imported directly by tests/test_global_router.py.


def _build_stacked_experts_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Thin shim → :meth:`StackedExpertsLoRAModule.build_moe_state_dict`."""
    return StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)


def build_standard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    """Run the standard (lora/ortho) finalize chain WITHOUT writing a file.

    Defuse fused qkv + bake channel scaling, relay adaln keys to the ComfyUI
    layout, cast dtype. Returns the finalized ``(state_dict, metadata)``. Factored
    out of :func:`save_network_weights` so the dual-pool turbo save can finalize
    each pool to its on-disk plain-LoRA layout and concat the two exactly (the
    distill chain must have already run on the caller's side for ortho/chimera
    stacks — plain-LoRA turbo pools have no such keys, so it is a no-op there).
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
    """Run the full save pipeline: distill chain + variant write.

    Mutates ``state_dict`` in place.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    # Distill chain. Order is load-bearing — see module docstring.
    ChimeraHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    StackedExpertsLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoInitLoRAModule.distill_save_state_dict(state_dict, dtype)
    _convert_legacy_ortho_to_lora(state_dict, dtype)

    # Variant dispatch:
    #   * stacked_experts_global_fei: independent-A → *_moe.safetensors
    #   * chimera_hydra_moe: dual-A per-pool + freq_router.* → *_chimera.safetensors
    #   * hydra_moe / ortho_hydra_to_hydra: shared-A Hydra → *_moe.safetensors
    #   * standard: defuse qkv → *.safetensors
    # Auto-fallback: any surviving ``.lora_up_weight`` key implies a Hydra
    # payload — kept for callers that don't plumb ``save_variant`` through.
    is_stacked_experts_variant = save_variant == "stacked_experts_global_fei"
    is_chimera_variant = save_variant == "chimera_hydra_moe"
    is_hydra_variant = (
        save_variant in ("hydra_moe", "ortho_hydra_to_hydra")
        or (
            not is_chimera_variant
            and any(k.endswith(".lora_up_weight") for k in state_dict.keys())
        )
    ) and not is_stacked_experts_variant

    if is_stacked_experts_variant:
        se_file = os.path.splitext(file)[0] + "_moe.safetensors"
        se_sd = StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(se_sd, se_file, metadata or {})
        logger.info(f"StackedExperts full format saved to {se_file}")
        return

    if is_chimera_variant:
        chimera_file = os.path.splitext(file)[0] + "_chimera.safetensors"
        chimera_sd = ChimeraHydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(chimera_sd, chimera_file, metadata or {})
        logger.info(f"ChimeraHydra full format saved to {chimera_file}")
        return

    if is_hydra_variant:
        hydra_file = os.path.splitext(file)[0] + "_moe.safetensors"
        hydra_sd = HydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(hydra_sd, hydra_file, metadata or {})
        logger.info(f"HydraLoRA full format saved to {hydra_file}")
        # The _moe file is the only useful artifact for HydraLoRA —
        # a uniform expert average defeats layer-local routing.
        return

    # Standard (lora / ortho) write path.
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
