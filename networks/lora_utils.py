import os
import re
from typing import Dict, List, Optional, Union
import torch
from tqdm import tqdm
from library.runtime.device import synchronize_device
from library.io.safetensors import (
    MemoryEfficientSafeOpen,
    TensorWeightAdapter,
    WeightTransformHooks,
    get_split_weight_filenames,
)
from library.log import setup_logging
from networks.attn_fuse import ATTN_FUSE_SPECS

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# AdaLN LoRA key layouts. The same modulation up-projection has two names: the
# in-repo runtime module name (``adaln_up_{branch}``, post ``_dit_rename_hook``)
# and the ComfyUI state-dict path (``adaln_modulation_{branch}.2`` → LoRA key
# ``adaln_modulation_{branch}_2``). ComfyUI's generic key map only knows the
# latter; the in-repo loader only knows the former. These two helpers convert a
# flat LoRA state dict between them so one trained checkpoint round-trips both
# ecosystems (see adaln.md §Key-naming contract). Non-adaln keys pass through.
_ADALN_BRANCHES = "self_attn|cross_attn|mlp"
_ADALN_RUNTIME_KEY_RE = re.compile(
    rf"^(lora_unet_blocks_\d+_)adaln_up_({_ADALN_BRANCHES})(\..+)$"
)
_ADALN_COMFY_KEY_RE = re.compile(
    rf"^(lora_unet_blocks_\d+_)adaln_modulation_({_ADALN_BRANCHES})_2(\..+)$"
)


def relayout_adaln_runtime_to_comfy(
    weights_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rename runtime adaln keys (``adaln_up_{br}``) to the ComfyUI layout
    (``adaln_modulation_{br}_2``) so the file loads natively in ComfyUI."""
    return {
        (
            f"{m.group(1)}adaln_modulation_{m.group(2)}_2{m.group(3)}"
            if (m := _ADALN_RUNTIME_KEY_RE.match(k))
            else k
        ): v
        for k, v in weights_sd.items()
    }


def relayout_adaln_comfy_to_runtime(
    weights_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`relayout_adaln_runtime_to_comfy` — ComfyUI adaln keys
    back to in-repo runtime module names so a comfy-native adaln LoRA also loads
    through ``create_network_from_weights`` (in-repo inference / merge)."""
    return {
        (
            f"{m.group(1)}adaln_up_{m.group(2)}{m.group(3)}"
            if (m := _ADALN_COMFY_KEY_RE.match(k))
            else k
        ): v
        for k, v in weights_sd.items()
    }


def has_comfy_adaln_keys(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """True if any key is in the ComfyUI adaln layout (needs comfy→runtime rename)."""
    return any(_ADALN_COMFY_KEY_RE.match(k) for k in weights_sd)


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(
            f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}"
        )

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {
            k: v for k, v in weights_sd.items() if not regex_exclude.search(k)
        }
        logger.info(
            f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
        )

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


# Derived from `ATTN_FUSE_SPECS` so the q/k/v ↔ qkv fusion layout has a single
# source of truth. Save / adapter-load walk the underscore-fragment form via
# `iter_split_groups`; this base-model merge path needs the dotted module-path
# form (e.g. `self_attn.qkv_proj` matches against `blocks.0.self_attn.qkv_proj`).
_FUSED_PROJ_FALLBACK = {
    f"{spec.attn_type}.{spec.fused_letters}_proj": (
        f"{spec.attn_type}.{{}}_proj",
        spec.component_letters,
    )
    for spec in ATTN_FUSE_SPECS
}


def _try_merge_unfused_lora(
    model_key_without_weight: str,
    lora_weight_keys: set,
    lora_sd: dict,
    multiplier: float,
    calc_device: torch.device,
    model_weight: torch.Tensor,
) -> "torch.Tensor | None":
    """Try to merge old unfused LoRA keys (q_proj/k_proj/v_proj) into a fused model weight.

    Returns the concatenated delta tensor, or None if no old keys found.
    """
    for fused_fragment, (template, suffixes) in _FUSED_PROJ_FALLBACK.items():
        if fused_fragment not in model_key_without_weight:
            continue

        deltas = []
        all_found = True
        keys_to_remove = []
        for suffix in suffixes:
            old_module = model_key_without_weight.replace(
                fused_fragment, template.format(suffix)
            )
            found_prefix = None
            for prefix in ["lora_unet_", ""]:
                lora_name = prefix + old_module.replace(".", "_")
                down_key = lora_name + ".lora_down.weight"
                up_key = lora_name + ".lora_up.weight"
                alpha_key = lora_name + ".alpha"
                if down_key in lora_weight_keys and up_key in lora_weight_keys:
                    found_prefix = prefix
                    break
            if found_prefix is None:
                all_found = False
                break

            down = lora_sd[down_key].to(calc_device)
            up = lora_sd[up_key].to(calc_device)
            dim = down.size(0)
            alpha = lora_sd.get(alpha_key, dim)
            scale = alpha / dim

            delta = multiplier * (up @ down) * scale
            deltas.append(delta)
            keys_to_remove.extend(
                [k for k in (down_key, up_key, alpha_key) if k in lora_weight_keys]
            )

        if not all_found or not deltas:
            continue

        for k in keys_to_remove:
            lora_weight_keys.discard(k)

        fused_delta = torch.cat(deltas, dim=0)
        return fused_delta

    return None


def load_safetensors_with_lora(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): List of dictionaries of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading safetensors.
        weight_transform_hooks (Optional[WeightTransformHooks]): Hooks for transforming weights during loading.
    """

    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for i, lora_sd in enumerate(lora_weights_list):
            # Strip _orig_mod_ from LoRA keys (inserted by torch.compile during training)
            normalized = {}
            for k, v in lora_sd.items():
                normalized[k.replace("__orig_mod_", "_")] = v
            # Shipped checkpoints carry adaln LoRA keys in the ComfyUI layout
            # (``adaln_modulation_{br}_2`` — lora_save relays them at write
            # time), but this hook sees the DiT's *runtime* key names
            # (``adaln_up_{br}``, post ``_dit_rename_hook``). Without the
            # rename the adaln rows of every ``train_adaln`` LoRA silently
            # fell into the "not all LoRA keys are used" warning on the
            # static-merge path (create_network_from_weights already did
            # this for the live-hook path). Presence-gated: no-op otherwise.
            if has_comfy_adaln_keys(normalized):
                normalized = relayout_adaln_comfy_to_runtime(normalized)
            lora_weights_list[i] = normalized
            lora_weight_keys = set(normalized.keys())
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        # A scalar multiplier (e.g. the `--lora_multiplier` argparse default of
        # 1.0, which is a bare float under nargs="*") broadcasts to all adapters.
        # Coerce to a list so the length checks below don't len() a float.
        if isinstance(lora_multipliers, (int, float)):
            lora_multipliers = [float(lora_multipliers)] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        logger.info(
            f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}"
        )

        def weight_hook_func(
            model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False
        ):
            nonlocal \
                list_of_lora_weight_keys, \
                lora_weights_list, \
                lora_multipliers, \
                calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)

            for lora_weight_keys, lora_sd, multiplier in zip(
                list_of_lora_weight_keys, lora_weights_list, lora_multipliers
            ):
                lora_name_without_prefix = model_weight_key.rsplit(".", 1)[0]
                found = False
                for prefix in ["lora_unet_", ""]:
                    lora_name = prefix + lora_name_without_prefix.replace(".", "_")
                    down_key = lora_name + ".lora_down.weight"
                    up_key = lora_name + ".lora_up.weight"
                    alpha_key = lora_name + ".alpha"
                    if down_key in lora_weight_keys and up_key in lora_weight_keys:
                        found = True
                        break
                if not found:
                    # Fallback: check for old unfused LoRA keys on fused projections
                    fused_delta = _try_merge_unfused_lora(
                        lora_name_without_prefix,
                        lora_weight_keys,
                        lora_sd,
                        multiplier,
                        calc_device,
                        model_weight,
                    )
                    if fused_delta is not None:
                        model_weight = model_weight + fused_delta
                    continue

                down_weight = lora_sd[down_key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                down_weight = down_weight.to(calc_device)
                up_weight = up_weight.to(calc_device)

                if len(model_weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    delta = multiplier * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    delta = (
                        multiplier
                        * (
                            up_weight.squeeze(3).squeeze(2)
                            @ down_weight.squeeze(3).squeeze(2)
                        )
                        .unsqueeze(2)
                        .unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3), up_weight
                    ).permute(1, 0, 2, 3)
                    delta = multiplier * conved * scale

                model_weight = model_weight + delta

                lora_weight_keys.remove(down_key)
                lora_weight_keys.remove(up_key)
                if alpha_key in lora_weight_keys:
                    lora_weight_keys.remove(alpha_key)

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device)
            return model_weight

        weight_hook = weight_hook_func

    state_dict = _load_safetensors_with_hook(
        model_files,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        weight_hook=weight_hook,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        if len(lora_weight_keys) > 0:
            logger.warning(
                f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}"
            )

    return state_dict


def _load_safetensors_with_hook(
    model_files: list[str],
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    weight_hook: callable = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors files and run an optional per-tensor hook."""
    logger.info(
        f"Loading state dict. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
    )
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(
            model_file, disable_numpy_memmap=disable_numpy_memmap
        ) as original_f:
            f = (
                TensorWeightAdapter(weight_transform_hooks, original_f)
                if weight_transform_hooks is not None
                else original_f
            )
            for key in tqdm(
                f.keys(),
                desc=f"Loading {os.path.basename(model_file)}",
                leave=False,
            ):
                if weight_hook is None and move_to_device:
                    value = f.get_tensor(
                        key, device=calc_device, dtype=dit_weight_dtype
                    )
                else:
                    value = f.get_tensor(
                        key
                    )  # we cannot directly load to device because get_tensor does non-blocking transfer
                    if weight_hook is not None:
                        value = weight_hook(
                            key, value, keep_on_calc_device=move_to_device
                        )
                    if move_to_device:
                        value = value.to(
                            calc_device, dtype=dit_weight_dtype, non_blocking=True
                        )
                    elif dit_weight_dtype is not None:
                        value = value.to(dit_weight_dtype)

                state_dict[key] = value
    if move_to_device:
        synchronize_device(calc_device)

    return state_dict
