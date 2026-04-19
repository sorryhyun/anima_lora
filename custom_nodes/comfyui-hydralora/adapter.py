"""LoRA / HydraLoRA / ReFT loading and application for the Anima adapter node.

A single safetensors file may contain any combination of the three; each is
auto-detected from key patterns. Plain LoRA / Hydra are applied through
ComfyUI's weight-patch path; ReFT is applied as per-block ``forward_hook``s
swapped in via ``ModelPatcher.add_object_patch`` (overriding ``forward``
strands block weights on CPU under ComfyUI's cast-weights path).
"""

import logging
import re
from collections import OrderedDict
from typing import Dict, Optional

import torch

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


def _extract_lora_sd(
    weights_sd: Dict[str, torch.Tensor],
) -> Optional[Dict[str, torch.Tensor]]:
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


def load_adapter(file_path: str) -> dict:
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
    """Uniform-weight expert bake-down (per-layer router can't run under
    ComfyUI's weight-patch model).
    """
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


def _make_reft_hook(params: dict, strength: float):
    """Forward hook that adds ReFT's residual edit to the block output.

    Hook params (rotate / source_w / source_b) are moved/cast to the input's
    device/dtype on first call and cached for subsequent calls.
    """
    scale = params["scale"]
    state = {
        "rotate": params["rotate"],
        "source_w": params["source_w"],
        "source_b": params["source_b"],
        "ready": False,
    }

    def reft_hook(module, inputs, output):
        h = output
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

    return reft_hook


def _apply_reft_to_model(model, reft_blocks: Dict[int, dict], strength: float) -> int:
    """Install per-block ReFT edits as ComfyUI object patches.

    Uses a ``forward_hook`` per block (swapped in by replacing the block's
    ``_forward_hooks`` OrderedDict via ``add_object_patch``) instead of
    overriding ``block.forward``. Replacing ``forward`` interferes with
    ComfyUI's weight-loading path — the block's Linears were ending up with
    ``comfy_cast_weights=False`` and their weights left on CPU, producing a
    device mismatch when the block ran. A forward hook leaves ``forward``
    (and ComfyUI's view of it) untouched, and torch.compile traces cleanly
    through it.

    Returns the number of blocks actually patched. The original
    ``_forward_hooks`` dict is restored on ``ModelPatcher.unpatch_model``.
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
        block = diffusion.blocks[idx]
        hook = _make_reft_hook(params, strength)
        new_hooks = OrderedDict(block._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(
            f"diffusion_model.blocks.{idx}._forward_hooks", new_hooks
        )
        patched += 1
    return patched


def apply_adapter(
    model, file_path: str, strength_lora: float, strength_reft: float
) -> bool:
    """Apply LoRA / Hydra / ReFT components from ``file_path`` to ``model``
    in place. ``model`` must already be a clone. Returns True if anything
    was applied.
    """
    bundle = load_adapter(file_path)
    applied_any = False

    if bundle["hydra"] is not None:
        num_experts = bundle["hydra"]["num_experts"]
        uniform = torch.ones(num_experts, dtype=torch.float32) / num_experts
        lora_sd = _bake_hydra_to_lora(bundle["hydra"], uniform)
        if lora_sd:
            _apply_lora_sd_to_model(model, lora_sd, strength_lora)
            applied_any = True
    elif bundle["lora"] is not None:
        # Plain LoRA — apply directly. Hydra + plain-lora in the same file is
        # not produced by the save pipeline, so ``elif`` is sufficient.
        _apply_lora_sd_to_model(model, bundle["lora"], strength_lora)
        applied_any = True

    if bundle["reft"] is not None:
        n = _apply_reft_to_model(model, bundle["reft"], strength_reft)
        if n > 0:
            applied_any = True

    if not applied_any:
        logger.warning(
            f"Anima adapter at {file_path} contained no recognizable "
            "LoRA, Hydra, or ReFT keys."
        )
    return applied_any
