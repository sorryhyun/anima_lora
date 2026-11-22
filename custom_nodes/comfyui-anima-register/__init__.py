"""Anima register-token adapter ComfyUI custom node.

One node:

* ``AnimaRegisterAdapter`` — loads a register ``adapter.safetensors`` and patches
  a stock Anima ``MODEL`` so a normal KSampler downstream runs with the register
  tokens + trained QKV surface live. The trainer does not write register
  checkpoints; the node loads existing ones.

Comfy-native (split q/k/v against ``MiniTrainDIT``, no vendored training code).
Do NOT stack with the Block Compile node — the register path runs eager.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
