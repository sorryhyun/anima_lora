"""Repo paths and the output dirs."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "output" / "wake_probe"
CORPUS_TRAIN = REPO / "post_image_dataset" / "render" / "ja" / "resized"
CORPUS_HELD = REPO / "post_image_dataset" / "render" / "ja" / "heldout"
FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


# read OUT at call time so a caller can redirect it


def data_dir(a) -> Path:
    return OUT / ("data" + (f"_{a.data_tag}" if a.data_tag else ""))


def arm_dir(a) -> Path:
    return OUT / (
        a.arm
        + (f"_{a.data_tag}" if a.data_tag else "")
        + (f"_{a.arm_tag}" if a.arm_tag else "")
    )
