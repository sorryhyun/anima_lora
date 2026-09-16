"""Perception Encoder (PE) vision tower, re-exported from ``anime_tools.vision.pe``.

Trainer code (REPA, CMMD, the PE feature cache, IP-Adapter bench) imports it
from here. Same module object, so monkeypatching through either path hits the
real thing.
"""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("anime_tools.vision.pe")
