"""Anima LoRA Trainer — ComfyUI custom node.

Inserts the anima_lora workspace root onto sys.path so that `library.*`,
`networks.*`, and `train` are importable from inside ComfyUI's process.
Assumes this folder lives at ``<anima_lora_root>/custom_nodes/comfyui-anima-trainer``.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANIMA_LORA_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ANIMA_LORA_ROOT not in sys.path:
    sys.path.insert(0, _ANIMA_LORA_ROOT)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
