"""What the pixel_readout scripts share that is the bench's own: run lookup and sizes.

Model code is ``library/anima/pixel_head.py``; training pieces, losses and sheets are
``scripts/pixel_head/train/_common.py``, ``_x1_trainer.py``, ``_x1_data.py`` and ``_x2_trainer.py``. A script here
imports those and this package's ``_*.py``, never another script.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from library.anima.pixel_head import PATCH, PixelHead, load_pixel_head
from library.env import resolve_under_home

RESULTS = "bench/pixel_readout/results"
P1_RUN = f"{RESULTS}/20261118-2221-readout_1118_2219"
TARGET_TOKENS = 4100


def resolve_run(run: str | Path) -> Path:
    """A results directory, or a bare label → the newest ``results/*-<label>``."""
    path = Path(resolve_under_home(str(run)))
    if path.is_dir():
        return path
    hits = sorted(Path(resolve_under_home(RESULTS)).glob(f"*-{run}"))
    if not hits:
        raise FileNotFoundError(
            f"no run {run} (neither a path nor a label in {RESULTS})"
        )
    return hits[-1]


def load_head(run: str | Path, d_feat: int, device) -> PixelHead:
    """A trained head from ``<run>/head.safetensors`` (shape from its metadata)."""
    return load_pixel_head(resolve_run(run) / "head.safetensors", device, d_feat)


def fit_size(w: int, h: int) -> tuple[int, int]:
    s = math.sqrt(TARGET_TOKENS * PATCH * PATCH / (w * h))
    return max(PATCH, round(w * s / PATCH) * PATCH), max(
        PATCH, round(h * s / PATCH) * PATCH
    )


def bicubic2(img: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        img[None].float(), scale_factor=2, mode="bicubic", align_corners=False
    )[0].clamp(-1, 1)
