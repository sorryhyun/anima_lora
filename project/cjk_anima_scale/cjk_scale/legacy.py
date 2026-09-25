"""legacy — the stage configs before the 2026-09-25 collapse, read-only, for
the ``experiments/`` that re-read old ``data_<stage>_<tag>`` dirs.

``load_stage(stage)`` returns what those scripts took from the old
``config.load(stage, None)``: ``band``, ``data`` (the stage's ``[data]`` over
the old defaults) and ``train`` (its ``[train]`` over the old defaults). The
files are ``_archive/configs/<stage>.toml``; nothing in the line trains or
builds from them.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .paths import LINE

ARCHIVE = LINE / "_archive" / "configs"

# the old config.py defaults, as they stood on 2026-09-25
_DATA = {
    "scenes": "s1,s1w,sl1w,ja_comic",
    "scene_one_bubble": "ja_comic",
    "single_scenes": "s1,s1w",
    "single_max_ar": 2.0,
    "horizontal_scenes": "sl1w",
    "shapes": "448,512:2,448x512,512x448",
    "vertical": True,
    "stroke": 0.0,
    "horizontal_frac": 0.3,
}
_TRAIN = {
    "batch": 4,
    "lr_rows": 1e-3,
    "init_anchor": 0.0,
    "box_share": 0.25,
    "box_share_cap": 0.5,
    "box_share_glyphs": 8,
    "grid_box": 1,
    "steps": 28,
    "cfg": 4.0,
    "seed": 0,
}


@dataclass(frozen=True)
class LegacyStage:
    stage: str
    band: tuple
    data: dict
    train: dict
    path: Path


def load_stage(stage: str, run=None) -> LegacyStage:
    assert run is None, "legacy stages load without a run"
    path = ARCHIVE / f"{stage}.toml"
    assert path.is_file(), f"no archived stage config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    data = {k: v for k, v in raw.get("data", {}).items() if k != "mix"}
    return LegacyStage(
        stage=raw.get("stage", stage),
        band=tuple(float(x) for x in raw["band"]),
        data={**_DATA, **data},
        train={**_TRAIN, **raw.get("train", {})},
        path=path,
    )
