"""CUDA caching-allocator defaults (stdlib-only; importable before torch).

Free-fit bucketing hands the allocator a different seq_len every step, which
fragments the reserved pool over a long run; ``expandable_segments`` grows
segments in place instead. The env var is read when the allocator initializes,
so this must run before the first CUDA allocation (train.py's pre-torch
prologue).
"""

from __future__ import annotations

import os
import sys


def default_expandable_segments() -> bool:
    """Default ``PYTORCH_CUDA_ALLOC_CONF`` to ``expandable_segments:True``.

    Linux-only (no cuMem* backend on Windows). An existing
    ``PYTORCH_CUDA_ALLOC_CONF`` or ``ANIMA_EXPANDABLE_SEGMENTS=0`` leaves it
    alone. Returns True when the default was applied.
    """
    if sys.platform != "linux":
        return False
    if os.environ.get("ANIMA_EXPANDABLE_SEGMENTS", "1").lower() in ("0", "false"):
        return False
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return True
