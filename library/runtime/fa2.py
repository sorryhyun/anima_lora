"""Puts the sm_120 FlashAttention build ahead of the pinned wheel in repo jobs.

The build (private repo sorryhyun/flash-attention-sm120) holds the 128×32 forward tile
and the fp16-accumulate backward, and kernels for sm_120 alone, so it cannot be the
``pyproject.toml`` pin. ``make fa2-sm120`` unpacks it under ``models/fa2_sm120/<tag>``;
without that folder, and on any other card, the pinned wheel loads as before. ComfyUI
shares the venv and never sees this path. Call :func:`use_sm120_build` before the first
``import flash_attn``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from library.env import resolve_under_home

logger = logging.getLogger(__name__)

TAG = "v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14"
REPO = "sorryhyun/flash-attention-sm120"
_TORCH_MINOR = "2.14"
_PYTHON = (3, 13)
_active: bool | None = None


def build_dir() -> Path:
    return resolve_under_home(f"models/fa2_sm120/{TAG}")


def _skip_reason(root: Path) -> str | None:
    """Why the unpacked build at ``root`` cannot load here; None when it can."""
    import torch

    if sys.platform != "linux" or sys.version_info[:2] != _PYTHON:
        return f"built for CPython {_PYTHON[0]}.{_PYTHON[1]} on Linux"
    if ".".join(torch.__version__.split(".")[:2]) != _TORCH_MINOR:
        return f"built on torch {_TORCH_MINOR}, this is {torch.__version__}"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
        return "not a sm_120 card"
    if "flash_attn_2_cuda" in sys.modules:
        return "the pinned wheel is already imported"
    return None


def use_sm120_build() -> bool:
    global _active
    if _active is None:
        root = build_dir()
        _active = False
        if (root / "flash_attn").is_dir():
            reason = _skip_reason(root)
            if reason is None:
                sys.path.insert(0, str(root))
                _active = True
                logger.info("FlashAttention: sm_120 build %s", TAG)
            else:
                logger.warning("FlashAttention: pinned wheel (%s)", reason)
    return _active
