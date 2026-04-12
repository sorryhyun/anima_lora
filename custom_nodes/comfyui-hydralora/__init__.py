"""HydraLoRA: Multi-style LoRA loader with MoE routing for ComfyUI.

Loads a HydraLoRA multi-head safetensors file (*_hydra.safetensors) and applies
it to the model with either:
  - Manual expert weights (sliders per expert)
  - Automatic routing from text conditioning (via the learned router)

The node produces a MODEL output with the HydraLoRA applied as a model patch,
compatible with any ComfyUI sampler.
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F

import folder_paths

logger = logging.getLogger(__name__)

# Cache: path -> parsed HydraLoRA data
_hydra_cache: Dict[str, dict] = {}


def _load_hydra(file_path: str) -> dict:
    """Load and parse a HydraLoRA multi-head safetensors file."""
    if file_path in _hydra_cache:
        return _hydra_cache[file_path]

    from safetensors.torch import load_file

    weights_sd = load_file(file_path)

    # Parse structure
    router_weight = weights_sd.get("_hydra_router.weight")
    router_bias = weights_sd.get("_hydra_router.bias")
    if router_weight is None:
        raise ValueError(
            f"Not a HydraLoRA file (missing _hydra_router.weight): {file_path}"
        )

    num_experts = router_weight.shape[0]

    # Group by module prefix
    modules = {}  # prefix -> {lora_down, lora_ups: [up0, up1, ...], alpha}
    for key, value in weights_sd.items():
        if key.startswith("_hydra_router"):
            continue
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

    result = {
        "router_weight": router_weight,
        "router_bias": router_bias,
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
        expert_weights: (num_experts,) normalized weights for each expert

    Returns:
        Standard LoRA state_dict with lora_down.weight, lora_up.weight, alpha per module
    """
    result = {}
    for prefix, mod_data in hydra_data["modules"].items():
        if "lora_down" not in mod_data or "lora_ups" not in mod_data:
            continue

        # Weighted average of expert up-projections
        ups = mod_data["lora_ups"]
        stacked = torch.stack(
            [ups[i] for i in sorted(ups.keys())], dim=0
        )  # (E, out, rank)
        combined_up = torch.einsum(
            "e,eor->or", expert_weights.to(stacked.device), stacked
        )

        result[f"{prefix}.lora_down.weight"] = mod_data["lora_down"]
        result[f"{prefix}.lora_up.weight"] = combined_up
        if "alpha" in mod_data:
            result[f"{prefix}.alpha"] = mod_data["alpha"]

    return result


def _apply_lora_to_model(model, lora_sd: Dict[str, torch.Tensor], strength: float):
    """Apply a standard LoRA state_dict to a ComfyUI model using patching."""
    import comfy.lora

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    new_model = model.clone()
    comfy.lora.apply_lora(new_model, loaded, strength)
    return new_model


class HydraLoRALoader:
    """Load a HydraLoRA file with per-expert weight sliders for manual style blending."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "hydra_lora": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "HydraLoRA multi-head file (*_hydra.safetensors)"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                **{
                    f"expert_{i}": (
                        "FLOAT",
                        {
                            "default": 1.0,
                            "min": 0.0,
                            "max": 5.0,
                            "step": 0.05,
                            "tooltip": f"Weight for expert {i}",
                        },
                    )
                    for i in range(16)  # support up to 16 experts
                },
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Load a HydraLoRA multi-head LoRA and apply with per-expert weight control. "
        "Adjust expert_N sliders to blend different style heads. "
        "Only the first num_experts sliders are used; extras are ignored."
    )

    def apply(self, model, hydra_lora, strength=1.0, **kwargs):
        file_path = folder_paths.get_full_path("loras", hydra_lora)
        hydra_data = _load_hydra(file_path)
        num_experts = hydra_data["num_experts"]

        # Collect expert weights from kwargs
        raw_weights = []
        for i in range(num_experts):
            w = kwargs.get(f"expert_{i}", 1.0)
            raw_weights.append(w)
        raw_weights = torch.tensor(raw_weights, dtype=torch.float32)

        # Normalize to sum to 1
        total = raw_weights.sum()
        if total > 0:
            expert_weights = raw_weights / total
        else:
            expert_weights = torch.ones(num_experts) / num_experts

        lora_sd = _bake_down(hydra_data, expert_weights)
        return (_apply_lora_to_model(model, lora_sd, strength),)


class HydraLoRAAutoRouter:
    """Load a HydraLoRA file with automatic style routing from text conditioning.

    The learned router computes expert weights from the conditioning embeddings,
    selecting the appropriate style blend automatically based on the prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "hydra_lora": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "HydraLoRA multi-head file (*_hydra.safetensors)"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = (
        "Model with HydraLoRA applied using auto-routed expert weights.",
    )
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Load a HydraLoRA and automatically route to style experts based on the "
        "text conditioning. The learned router max-pools the conditioning embeddings "
        "and computes a softmax gate over experts."
    )

    def apply(self, model, conditioning, hydra_lora, strength=1.0):
        file_path = folder_paths.get_full_path("loras", hydra_lora)
        hydra_data = _load_hydra(file_path)

        # Extract crossattn_emb from conditioning
        # ComfyUI conditioning format: list of (tensor, dict) tuples
        # The tensor is the crossattn embedding (B, seq, dim)
        cond_tensor = conditioning[0][0]  # first conditioning entry, tensor part

        # Run router: max_pool -> linear -> softmax
        device = cond_tensor.device
        router_w = hydra_data["router_weight"].to(device=device, dtype=torch.float32)
        router_b = hydra_data["router_bias"]
        if router_b is not None:
            router_b = router_b.to(device=device, dtype=torch.float32)

        # Max pool over sequence dimension
        pooled = cond_tensor.float().max(dim=1).values  # (B, 1024)
        # Average over batch for a single gate
        pooled = pooled.mean(dim=0, keepdim=True)  # (1, 1024)

        gate_logits = F.linear(pooled, router_w, router_b)  # (1, num_experts)
        gate = F.softmax(gate_logits, dim=-1).squeeze(0)  # (num_experts,)

        logger.info(
            f"HydraLoRA auto-route: {', '.join(f'E{i}={g:.3f}' for i, g in enumerate(gate))}"
        )

        lora_sd = _bake_down(hydra_data, gate)
        return (_apply_lora_to_model(model, lora_sd, strength),)


NODE_CLASS_MAPPINGS = {
    "HydraLoRALoader": HydraLoRALoader,
    "HydraLoRAAutoRouter": HydraLoRAAutoRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HydraLoRALoader": "HydraLoRA Loader (Manual)",
    "HydraLoRAAutoRouter": "HydraLoRA Loader (Auto Router)",
}
