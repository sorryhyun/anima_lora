"""Anima adapter ComfyUI custom node.

Single unified loader (``AnimaAdapterLoader``) that auto-detects and applies:
  - Plain LoRA (lora_down/lora_up keys) — ComfyUI weight patching.
  - HydraLoRA multi-head (lora_ups.N.weight) — experts baked down with
    uniform weighting (the trained per-layer router cannot be replayed
    under ComfyUI's weight-patch model) and applied as standard LoRA.
  - LoReFT residual-stream intervention (reft_unet_blocks_<idx>.*) — per-
    block forward_hook via ModelPatcher.add_object_patch.
  - Prefix / postfix / cond context splicing — ``diffusion_model.forward``
    wrapper that splices learned vectors after the LLM adapter.

Adapter (LoRA/Hydra/ReFT) and postfix are independently toggleable inside
the same node.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
