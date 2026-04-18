"""Anima adapter loader for ComfyUI.

Unified loader for Anima LoRA safetensors files. Auto-detects and applies:

  - Plain LoRA (lora_down/lora_up keys) — ComfyUI weight patching.
  - HydraLoRA multi-head (lora_ups.N.weight) — experts baked down with uniform
    weighting (the trained per-layer router cannot be replayed under
    ComfyUI's weight-patch model) and then applied as standard LoRA.
  - LoReFT residual-stream intervention (reft_unet_blocks_<idx>.*) — per-block
    forward override via ModelPatcher.add_object_patch, adding
    strength · scale · R^T·(ΔW·h + b) to the output of each trained block.

Any combination of the three can coexist in one file (e.g. LoRA + ReFT);
separate strength inputs control how much of each to apply.
"""

import logging
import re
from typing import Dict, Optional

import torch

import folder_paths

logger = logging.getLogger(__name__)

# Cache: path -> parsed adapter bundle ({"lora": dict|None, "hydra": dict|None, "reft": dict|None}).
_adapter_cache: Dict[str, dict] = {}

_REFT_KEY_RE = re.compile(r"^reft_unet_blocks_(\d+)\.(.+)$")


def _parse_reft(weights_sd: Dict[str, torch.Tensor]) -> Optional[Dict[int, dict]]:
    """Group ReFT keys by block index. Returns None if no ReFT keys present."""
    by_idx: Dict[int, Dict[str, torch.Tensor]] = {}
    for key, value in weights_sd.items():
        m = _REFT_KEY_RE.match(key)
        if m is None:
            continue
        idx = int(m.group(1))
        by_idx.setdefault(idx, {})[m.group(2)] = value

    if not by_idx:
        return None

    reft: Dict[int, dict] = {}
    for idx, d in by_idx.items():
        if "rotate_layer.weight" not in d or "learned_source.weight" not in d:
            logger.warning(
                f"ReFT block {idx} is missing rotate_layer.weight or "
                f"learned_source.weight — skipping."
            )
            continue
        rotate = d["rotate_layer.weight"]
        reft_dim = rotate.size(0)
        alpha_t = d.get("alpha")
        if alpha_t is None:
            alpha = float(reft_dim)
        else:
            alpha = float(alpha_t.item() if hasattr(alpha_t, "item") else alpha_t)
        reft[idx] = {
            "rotate": rotate,                              # (reft_dim, x_dim)
            "source_w": d["learned_source.weight"],        # (reft_dim, x_dim)
            "source_b": d["learned_source.bias"],          # (reft_dim,)
            "scale": alpha / reft_dim,
        }
    return reft or None


def _parse_hydra(weights_sd: Dict[str, torch.Tensor]) -> Optional[dict]:
    """Group Hydra multi-head keys. Returns None if no per-expert ups found."""
    modules: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if key.startswith("reft_"):
            continue
        parts = key.split(".")
        prefix = parts[0]
        rest = ".".join(parts[1:])
        mod = modules.setdefault(prefix, {})
        if rest == "lora_down.weight":
            mod["lora_down"] = value
        elif rest.startswith("lora_ups.") and rest.endswith(".weight"):
            idx = int(rest.split(".")[1])
            mod.setdefault("lora_ups", {})[idx] = value
        elif rest == "alpha":
            mod["alpha"] = value
        elif rest == "inv_scale":
            mod["inv_scale"] = value

    num_experts = 0
    for mod in modules.values():
        if "lora_ups" in mod:
            num_experts = max(num_experts, max(mod["lora_ups"].keys()) + 1)
    if num_experts == 0:
        return None
    return {"num_experts": num_experts, "modules": modules}


def _extract_lora_sd(weights_sd: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    """Pull standard LoRA keys (lora_down/lora_up/alpha/dora_scale).
    Returns None if no lora_up.weight keys are present.
    """
    out: Dict[str, torch.Tensor] = {}
    has_up = False
    for key, value in weights_sd.items():
        if key.startswith("reft_"):
            continue
        if key.endswith(".lora_up_weight"):
            continue  # Hydra stacked ups — handled via _parse_hydra
        if ".lora_ups." in key:
            continue  # Per-expert Hydra ups — handled via _parse_hydra
        out[key] = value
        if key.endswith(".lora_up.weight"):
            has_up = True
    return out if has_up else None


def _load_adapter(file_path: str) -> dict:
    """Parse an Anima adapter file once, cache by path."""
    if file_path in _adapter_cache:
        return _adapter_cache[file_path]

    from safetensors.torch import load_file

    weights_sd = load_file(file_path)

    if any(k.startswith("_hydra_router") for k in weights_sd.keys()):
        raise ValueError(
            f"{file_path} uses the deprecated global HydraLoRA router format. "
            "Retrain with the current codebase."
        )

    bundle = {
        "path": file_path,
        "lora": _extract_lora_sd(weights_sd),
        "hydra": _parse_hydra(weights_sd),
        "reft": _parse_reft(weights_sd),
    }
    _adapter_cache[file_path] = bundle

    summary = []
    if bundle["lora"] is not None:
        summary.append(
            f"{sum(1 for k in bundle['lora'] if k.endswith('.lora_up.weight'))} LoRA modules"
        )
    if bundle["hydra"] is not None:
        summary.append(
            f"Hydra({bundle['hydra']['num_experts']} experts, "
            f"{len(bundle['hydra']['modules'])} modules)"
        )
    if bundle["reft"] is not None:
        summary.append(f"ReFT({len(bundle['reft'])} blocks)")
    logger.info(
        f"Loaded Anima adapter: {', '.join(summary) or 'empty'} from {file_path}"
    )
    return bundle


def _bake_hydra_to_lora(
    hydra_data: dict, expert_weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Uniform-weight expert bake-down (see module docstring for why)."""
    result: Dict[str, torch.Tensor] = {}
    for prefix, mod in hydra_data["modules"].items():
        if "lora_down" not in mod or "lora_ups" not in mod:
            continue
        ups = mod["lora_ups"]
        stacked = torch.stack([ups[i] for i in sorted(ups.keys())], dim=0)
        combined_up = torch.einsum(
            "e,eor->or",
            expert_weights.to(device=stacked.device, dtype=stacked.dtype),
            stacked,
        )
        down_weight = mod["lora_down"]
        if "inv_scale" in mod and down_weight.dim() == 2:
            inv_scale = mod["inv_scale"].to(
                dtype=down_weight.dtype, device=down_weight.device
            )
            down_weight = down_weight * inv_scale.unsqueeze(0)
        result[f"{prefix}.lora_down.weight"] = down_weight
        result[f"{prefix}.lora_up.weight"] = combined_up
        if "alpha" in mod:
            result[f"{prefix}.alpha"] = mod["alpha"]
    return result


def _apply_lora_sd_to_model(model, lora_sd: Dict[str, torch.Tensor], strength: float):
    """Apply a standard LoRA state_dict via ComfyUI's weight patching."""
    import comfy.lora
    import comfy.lora_convert

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    model.add_patches(loaded, strength)


def _make_reft_wrapped_forward(orig_forward, params: dict, strength: float):
    """Build a ``forward`` replacement that adds ReFT's residual edit to the
    original block output. Params are moved/cast to the input's device/dtype
    on first call and cached for subsequent calls.
    """
    scale = params["scale"]
    state = {
        "rotate": params["rotate"],
        "source_w": params["source_w"],
        "source_b": params["source_b"],
        "ready": False,
    }

    def wrapped(*args, **kwargs):
        h = orig_forward(*args, **kwargs)
        if (
            not state["ready"]
            or state["rotate"].device != h.device
            or state["rotate"].dtype != h.dtype
        ):
            state["rotate"] = state["rotate"].to(device=h.device, dtype=h.dtype)
            state["source_w"] = state["source_w"].to(device=h.device, dtype=h.dtype)
            state["source_b"] = state["source_b"].to(device=h.device, dtype=h.dtype)
            state["ready"] = True
        delta = torch.nn.functional.linear(h, state["source_w"], state["source_b"])
        edit = torch.nn.functional.linear(delta, state["rotate"].T)
        return h + edit * (strength * scale)

    return wrapped


def _apply_reft_to_model(model, reft_blocks: Dict[int, dict], strength: float) -> int:
    """Install per-block ReFT overrides as ComfyUI object patches.

    Returns the number of blocks actually patched. Object patches are
    restored by ``ModelPatcher.unpatch_model`` when the clone is disposed,
    so this does not leak into sibling workflows.
    """
    diffusion = model.get_model_object("diffusion_model")
    if not hasattr(diffusion, "blocks"):
        raise ValueError(
            "ReFT adapter requires a DiT with `.blocks` ModuleList "
            f"(got {type(diffusion).__name__})."
        )
    num_blocks = len(diffusion.blocks)

    patched = 0
    for idx, params in reft_blocks.items():
        if idx < 0 or idx >= num_blocks:
            logger.warning(
                f"ReFT block index {idx} out of range [0, {num_blocks}); skipping"
            )
            continue
        # Resolve the block's current forward (could already be patched by a
        # previous node) and close over it so we compose cleanly.
        block = diffusion.blocks[idx]
        orig_forward = getattr(block, "forward")
        wrapped = _make_reft_wrapped_forward(orig_forward, params, strength)
        model.add_object_patch(f"diffusion_model.blocks.{idx}.forward", wrapped)
        patched += 1
    return patched


def _apply_bundle(model, bundle: dict, strength_lora: float, strength_reft: float):
    """Apply whichever of {lora, hydra, reft} the bundle contains to a fresh
    clone of ``model``. Returns the new ModelPatcher.
    """
    new_model = model.clone()
    applied_any = False

    if bundle["hydra"] is not None:
        num_experts = bundle["hydra"]["num_experts"]
        uniform = torch.ones(num_experts, dtype=torch.float32) / num_experts
        lora_sd = _bake_hydra_to_lora(bundle["hydra"], uniform)
        if lora_sd:
            _apply_lora_sd_to_model(new_model, lora_sd, strength_lora)
            applied_any = True
    elif bundle["lora"] is not None:
        # Plain LoRA — apply directly. Hydra + plain-lora in the same file is
        # not produced by the save pipeline, so ``elif`` is sufficient.
        _apply_lora_sd_to_model(new_model, bundle["lora"], strength_lora)
        applied_any = True

    if bundle["reft"] is not None:
        n = _apply_reft_to_model(new_model, bundle["reft"], strength_reft)
        if n > 0:
            applied_any = True

    if not applied_any:
        logger.warning(
            f"Anima adapter at {bundle['path']} contained no recognizable "
            "LoRA, Hydra, or ReFT keys."
        )
    return new_model


class AnimaAdapterLoader:
    """Load an Anima adapter file and apply LoRA/Hydra/ReFT components.

    Separate strength inputs control the LoRA weight-patch magnitude and the
    ReFT residual-edit magnitude. Either can be set to 0 to disable that
    component independently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "adapter": (
                    folder_paths.get_filename_list("loras"),
                    {
                        "tooltip": (
                            "Anima adapter file. May contain any combination "
                            "of LoRA, HydraLoRA (*_moe.safetensors), and "
                            "ReFT (residual-stream) weights."
                        )
                    },
                ),
                "strength_lora": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength for LoRA / Hydra weight patches.",
                    },
                ),
                "strength_reft": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength for ReFT residual-stream edits.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima adapter loader. Auto-detects LoRA, HydraLoRA, and ReFT "
        "content in the file and applies each with its own strength. "
        "HydraLoRA experts are baked down with uniform weighting "
        "(per-layer routing cannot be replayed under weight-patching). "
        "ReFT is applied as per-block forward overrides via object patches."
    )

    def apply(self, model, adapter, strength_lora=1.0, strength_reft=1.0):
        file_path = folder_paths.get_full_path("loras", adapter)
        bundle = _load_adapter(file_path)
        return (_apply_bundle(model, bundle, strength_lora, strength_reft),)


class HydraLoRALoader:
    """Back-compat shim: same file-sniffing loader as AnimaAdapterLoader, but
    with a single ``strength`` input and the old node name so existing
    workflows keep working. New workflows should prefer AnimaAdapterLoader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "hydra_lora": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "Anima adapter file (LoRA / Hydra / ReFT)."},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Legacy HydraLoRA loader. Same as AnimaAdapterLoader but with a "
        "single strength applied to both LoRA and ReFT components."
    )

    def apply(self, model, hydra_lora, strength=1.0):
        file_path = folder_paths.get_full_path("loras", hydra_lora)
        bundle = _load_adapter(file_path)
        return (_apply_bundle(model, bundle, strength, strength),)


NODE_CLASS_MAPPINGS = {
    "AnimaAdapterLoader": AnimaAdapterLoader,
    "HydraLoRALoader": HydraLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaAdapterLoader": "Anima Adapter Loader (LoRA / Hydra / ReFT)",
    "HydraLoRALoader": "HydraLoRA Loader (legacy)",
}
