"""Repo paths and the output dirs.

Vendored into ``project/cjk_anima_scale/src/`` (2026-09-25). Path plumbing
only differs from the source: ``OUT`` is this line's output root, and
``data_dir`` / ``arm_dir`` take an explicit ``data_path`` / ``arm_path`` on
the namespace (a run's ``<run>/data`` and its eval arm dirs) before falling
back to the tag-named layout the old ``data_<stage>_<tag>`` / ``rows_<stage>_<tag>``
records use.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "output" / "cjk_anima_scale"
CORPUS_TRAIN = REPO / "post_image_dataset" / "render" / "ja" / "resized"
CORPUS_HELD = REPO / "post_image_dataset" / "render" / "ja" / "heldout"
FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


# read OUT at call time so a caller can redirect it


def data_dir(a) -> Path:
    if getattr(a, "data_path", ""):
        return Path(a.data_path)
    return OUT / ("data" + (f"_{a.data_tag}" if a.data_tag else ""))


def arm_dir(a) -> Path:
    if getattr(a, "arm_path", ""):
        return Path(a.arm_path)
    return OUT / (
        a.arm
        + (f"_{a.data_tag}" if a.data_tag else "")
        + (f"_{a.arm_tag}" if a.arm_tag else "")
    )
