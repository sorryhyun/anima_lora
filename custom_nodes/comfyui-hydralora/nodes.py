"""Unified Anima adapter loader node.

One node, two independently-toggled sections:
  - Adapter: LoRA / HydraLoRA / ReFT (auto-detected from file keys)
  - Postfix: prefix / postfix / cond context splicing (auto-detected)

Both operate on the same cloned ``ModelPatcher``. Adapter weight patches
and ReFT block hooks are installed first; postfix wraps
``diffusion_model.forward`` last, so the wrapper sees the model with
adapter modifications already in place.
"""

import folder_paths

from .adapter import apply_adapter
from .postfix import apply_postfix


class AnimaAdapterLoader:
    """Apply an Anima adapter (LoRA / Hydra / ReFT) and/or a postfix.

    Each section is gated by a boolean toggle. When a toggle is off, that
    file dropdown and its strength inputs are ignored — there is no need
    for a None sentinel in the dropdown.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "use_adapter": (
                    "BOOLEAN",
                    {"default": True, "label_on": "on", "label_off": "off"},
                ),
                "adapter": (
                    loras,
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
                "use_postfix": (
                    "BOOLEAN",
                    {"default": False, "label_on": "on", "label_off": "off"},
                ),
                "postfix": (
                    loras,
                    {
                        "tooltip": (
                            "Postfix / prefix / cond file (prefix_embeds, "
                            "postfix_embeds, or cond_mlp.* keys)."
                        )
                    },
                ),
                "strength_postfix": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength multiplier for the postfix vectors.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima adapter loader. Toggle adapter (LoRA / Hydra / ReFT) and "
        "postfix sections independently. HydraLoRA installs per-Linear "
        "forward hooks that compute the trained per-sample router gate from "
        "each Linear's input and blend per-expert lora_up heads — full live "
        "routing including σ-conditional bias. ReFT installs per-block "
        "forward hooks. Postfix wraps diffusion_model.forward to splice "
        "learned vectors after the LLM adapter; positive-batch rows only "
        "(CFG-safe)."
    )

    def apply(
        self,
        model,
        use_adapter,
        adapter,
        strength_lora,
        strength_reft,
        use_postfix,
        postfix,
        strength_postfix,
    ):
        new_model = model.clone()

        if use_adapter:
            file_path = folder_paths.get_full_path("loras", adapter)
            apply_adapter(new_model, file_path, strength_lora, strength_reft)

        if use_postfix:
            file_path = folder_paths.get_full_path("loras", postfix)
            apply_postfix(new_model, file_path, strength_postfix)

        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "AnimaAdapterLoader": AnimaAdapterLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaAdapterLoader": "Anima Adapter Loader",
}
