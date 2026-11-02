"""Puts the sm_120 FlashAttention build ahead of the pinned wheel in repo jobs.

The build (sorryhyun/flash-attention-fp16) holds the 128×32 forward tile and the
fp16-accumulate backward, and kernels for sm_120 alone, so it cannot be the
``pyproject.toml`` pin. ``make fa2-sm120`` unpacks it under ``models/fa2_sm120/<tag>``;
without that folder, on any other card, or when it fails to load, the pinned wheel loads
as before. ComfyUI shares the venv and never sees this path. Call
:func:`use_sm120_build` before the first ``import flash_attn``.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from urllib.parse import quote

from library.env import resolve_under_home

logger = logging.getLogger(__name__)

TAG = "v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14"
REPO = "sorryhyun/flash-attention-fp16"
_LOCAL_VERSION = "sm120.hdim64128.fp16acc.cu132torch2.14"
_WHEEL_PLATFORMS = {"linux": "linux_x86_64", "win32": "win_amd64"}
_TORCH_MINOR = "2.14"
_PYTHON = (3, 13)
_active: bool | None = None


def build_dir() -> Path:
    return resolve_under_home(f"models/fa2_sm120/{TAG}")


def wheel_url() -> str:
    name = (
        f"flash_attn-2.8.3+{_LOCAL_VERSION}-cp313-cp313-"
        f"{_WHEEL_PLATFORMS[sys.platform]}.whl"
    )
    return f"https://github.com/{REPO}/releases/download/{TAG}/{quote(name)}"


def skip_reason() -> str | None:
    """Why the build cannot load here; None when it can."""
    import torch

    if (
        sys.platform not in _WHEEL_PLATFORMS
        or platform.machine().lower() not in ("x86_64", "amd64")
        or sys.version_info[:2] != _PYTHON
    ):
        return (
            f"built for CPython {_PYTHON[0]}.{_PYTHON[1]} on x86_64 Linux and Windows"
        )
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
            reason = skip_reason()
            if reason is None:
                sys.path.insert(0, str(root))
                try:
                    import flash_attn_2_cuda  # noqa: F401
                except ImportError as e:
                    sys.path.remove(str(root))
                    logger.warning("FlashAttention: pinned wheel (load failed: %s)", e)
                else:
                    _active = True
                    logger.info("FlashAttention: sm_120 build %s", TAG)
            else:
                logger.warning("FlashAttention: pinned wheel (%s)", reason)
    return _active
