"""HydraLoRA loader for ComfyUI.

Loads a HydraLoRA multi-head safetensors file (*_moe.safetensors) and applies
it to the model with uniform expert weighting. HydraLoRA uses layer-local
routing at training time (each adapted Linear has its own router that reads
live layer input), but ComfyUI's weight-patching model cannot replicate
per-sample per-layer routing decisions — so experts are baked down with
equal weights.
"""

import logging
from typing import Dict

import torch

import folder_paths

logger = logging.getLogger(__name__)

# Cache: path -> parsed HydraLoRA data
_hydra_cache: Dict[str, dict] = {}


def _load_hydra(file_path: str) -> dict:
    """Load and parse a HydraLoRA multi-head safetensors file.

    Expected keys per module:
      <prefix>.lora_down.weight
      <prefix>.lora_ups.<i>.weight   for i in [0, num_experts)
      <prefix>.alpha                  (optional)
      <prefix>.router.weight          (optional — ignored by manual loader)
      <prefix>.router.bias            (optional — ignored by manual loader)
      <prefix>.inv_scale              (optional — per-channel input absorption;
                                       folded into lora_down at bake time)
    """
    if file_path in _hydra_cache:
        return _hydra_cache[file_path]

    from safetensors.torch import load_file

    weights_sd = load_file(file_path)

    # Reject old global-router format with a clear message.
    if any(k.startswith("_hydra_router") for k in weights_sd.keys()):
        raise ValueError(
            f"{file_path} uses the old global HydraLoRA router format. "
            "Retrain with the current codebase to get the per-module format."
        )

    # Group by module prefix.
    modules: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        parts = key.split(".")
        prefix = parts[0]
        rest = ".".join(parts[1:])

        if prefix not in modules:
            modules[prefix] = {}

        if rest == "lora_down.weight":
            modules[prefix]["lora_down"] = value
        elif rest.startswith("lora_ups.") and rest.endswith(".weight"):
            idx = int(rest.split(".")[1])
            if "lora_ups" not in modules[prefix]:
                modules[prefix]["lora_ups"] = {}
            modules[prefix]["lora_ups"][idx] = value
        elif rest == "alpha":
            modules[prefix]["alpha"] = value
        elif rest == "inv_scale":
            modules[prefix]["inv_scale"] = value
        # .router.weight / .router.bias are ignored here (manual blending only).

    # Infer num_experts from any module that has lora_ups.
    num_experts = 0
    for mod in modules.values():
        if "lora_ups" in mod:
            num_experts = max(num_experts, max(mod["lora_ups"].keys()) + 1)
    if num_experts == 0:
        raise ValueError(f"No HydraLoRA expert up-projections found in {file_path}")

    result = {
        "num_experts": num_experts,
        "modules": modules,
    }
    _hydra_cache[file_path] = result
    logger.info(
        f"Loaded HydraLoRA: {len(modules)} modules, {num_experts} experts from {file_path}"
    )
    return result


def _bake_down(
    hydra_data: dict, expert_weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Bake down multi-head LoRA to standard LoRA with given expert weights.

    Args:
        hydra_data: parsed HydraLoRA data from _load_hydra
        expert_weights: (num_experts,) weights for each expert. Typically
            normalized to sum to 1 so the blended magnitude matches the
            training-time softmax gate.

    Returns:
        Standard LoRA state_dict with lora_down.weight, lora_up.weight, alpha per module
    """
    result = {}
    for prefix, mod_data in hydra_data["modules"].items():
        if "lora_down" not in mod_data or "lora_ups" not in mod_data:
            continue

        # Weighted combination of expert up-projections.
        ups = mod_data["lora_ups"]
        stacked = torch.stack(
            [ups[i] for i in sorted(ups.keys())], dim=0
        )  # (E, out, rank)
        combined_up = torch.einsum(
            "e,eor->or", expert_weights.to(stacked.device), stacked
        )

        # Undo the per-channel input absorption that HydraLoRAModule applies
        # at forward time (`x_lora = x * inv_scale`). The ComfyUI patcher
        # merges LoRA as a raw weight delta, so we have to fold inv_scale
        # into lora_down here — mirrors LoRAInfModule.merge_to.
        down_weight = mod_data["lora_down"]
        if "inv_scale" in mod_data and down_weight.dim() == 2:
            inv_scale = mod_data["inv_scale"].to(
                dtype=down_weight.dtype, device=down_weight.device
            )
            down_weight = down_weight * inv_scale.unsqueeze(0)

        result[f"{prefix}.lora_down.weight"] = down_weight
        result[f"{prefix}.lora_up.weight"] = combined_up
        if "alpha" in mod_data:
            result[f"{prefix}.alpha"] = mod_data["alpha"]

    return result


def _apply_lora_to_model(model, lora_sd: Dict[str, torch.Tensor], strength: float):
    """Apply a standard LoRA state_dict to a ComfyUI model using patching."""
    import comfy.lora
    import comfy.lora_convert

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    new_model = model.clone()
    new_model.add_patches(loaded, strength)
    return new_model


class HydraLoRALoader:
    """Load a HydraLoRA file and apply with uniform expert weighting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "hydra_lora": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "HydraLoRA multi-head file (*_moe.safetensors)"},
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
        "Load a HydraLoRA multi-head LoRA and apply with uniform expert weighting. "
        "Experts are baked down with equal weights since ComfyUI's weight-patching "
        "cannot replicate the trained layer-local routing."
    )

    def apply(self, model, hydra_lora, strength=1.0):
        file_path = folder_paths.get_full_path("loras", hydra_lora)
        hydra_data = _load_hydra(file_path)
        num_experts = hydra_data["num_experts"]

        expert_weights = torch.ones(num_experts, dtype=torch.float32) / num_experts
        lora_sd = _bake_down(hydra_data, expert_weights)
        return (_apply_lora_to_model(model, lora_sd, strength),)


NODE_CLASS_MAPPINGS = {
    "HydraLoRALoader": HydraLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HydraLoRALoader": "HydraLoRA Loader",
}
