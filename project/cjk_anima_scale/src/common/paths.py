"""Repo paths and the output dirs.

Vendored into ``project/cjk_anima_scale/src/`` (2026-09-25). Path plumbing
only differs from the source: ``OUT`` is this line's output root, and
``data_dir`` / ``arm_dir`` are the explicit ``data_path`` / ``arm_path`` on
the namespace (a run's ``<run>/data`` and its eval arm dirs; a record dir
``data_<stage>_<tag>`` / ``rows_<stage>_<tag>`` is named by its path too).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "output" / "cjk_anima_scale"
CORPUS_TRAIN = REPO / "post_image_dataset" / "render" / "ja" / "resized"
CORPUS_HELD = REPO / "post_image_dataset" / "render" / "ja" / "heldout"
FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def data_dir(a) -> Path:
    assert a.data_path, "--data_path: name the data dir"
    return Path(a.data_path)


def arm_dir(a) -> Path:
    assert a.arm_path, "--arm_path: name the arm dir"
    return Path(a.arm_path)
