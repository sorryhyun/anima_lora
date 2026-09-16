"""CUDA caching-allocator defaults (stdlib-only; importable before torch).

Free-fit native-shape bucketing hands the allocator a different seq_len every
step, so segment reuse degrades and the reserved pool fragments over a long
run. ``expandable_segments`` lets the allocator grow segments in place
instead of carving a new one per novel size.

The env var is read when the CUDA caching allocator initializes, so this must
run before the first CUDA allocation — train.py calls it in its pre-torch
prologue.
"""

from __future__ import annotations

import os
import sys


def default_expandable_segments() -> bool:
    """Default ``PYTORCH_CUDA_ALLOC_CONF`` to ``expandable_segments:True``.

    Linux-only (torch does not enable the cuMem* virtual-memory backend on
    Windows). Respects an existing ``PYTORCH_CUDA_ALLOC_CONF`` verbatim
    (never appends to a user-chosen config) and the
    ``ANIMA_EXPANDABLE_SEGMENTS=0`` opt-out. Returns True when the default
    was applied.
    """
    if sys.platform != "linux":
        return False
    if os.environ.get("ANIMA_EXPANDABLE_SEGMENTS", "1").lower() in ("0", "false"):
        return False
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return True
