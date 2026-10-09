"""Repo paths and the output dirs.

Vendored from the scale line's ``src/`` (2026-10-09; there since
2026-09-25). ``OUT`` stays the scale line's output root and is read-only from
here: the scene pools (``scenes_*``) and the EN reference cache
(``native_enref``) live there. ``FONT_DIR`` is this line's
``assets/fonts``. ``data_dir`` / ``arm_dir`` are the explicit ``data_path`` /
``arm_path`` on the namespace.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "output" / "cjk_anima_scale"  # read-only: scene pools, EN refs
CORPUS_TRAIN = REPO / "post_image_dataset" / "render" / "ja" / "resized"
CORPUS_HELD = REPO / "post_image_dataset" / "render" / "ja" / "heldout"
FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def data_dir(a) -> Path:
    assert a.data_path, "--data_path: name the data dir"
    return Path(a.data_path)


def arm_dir(a) -> Path:
    assert a.arm_path, "--arm_path: name the arm dir"
    return Path(a.arm_path)
