"""legacy — the stage configs before the 2026-09-25 collapse, read-only, for
the ``experiments/`` that re-read old ``data_<stage>_<tag>`` dirs.

The pre-collapse stage files are split records now (data generation and
training kept apart): ``configs/data_build/<stage>.toml`` = band + pools +
``[[mix]]`` recipes, ``configs/train/<stage>.toml`` = the trainer values.
``load_stage(stage)`` reads both halves and returns what those scripts took
from the old ``config.load(stage, None)``: ``band``, ``data`` (the data-build
keys over the old defaults) and ``train`` (the trainer keys over the old
defaults). Nothing in the line trains or builds from them.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .paths import CONFIGS

DATA_BUILD = CONFIGS / "data_build"
TRAIN = CONFIGS / "train"
# data-build keys that are not [data] pool keys
_BUILD_META = ("band", "gate", "min_overlap", "joint_from", "mix")

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
    dpath = DATA_BUILD / f"{stage}.toml"
    tpath = TRAIN / f"{stage}.toml"
    assert dpath.is_file(), f"no legacy data-build config {dpath}"
    assert tpath.is_file(), f"no legacy train config {tpath}"
    draw = tomllib.loads(dpath.read_text(encoding="utf-8"))
    traw = tomllib.loads(tpath.read_text(encoding="utf-8"))
    data = {k: v for k, v in draw.items() if k not in _BUILD_META}
    train = {k: v for k, v in traw.items() if k != "warm_from"}
    return LegacyStage(
        stage=stage,
        band=tuple(float(x) for x in draw["band"]),
        data={**_DATA, **data},
        train={**_TRAIN, **train},
        path=dpath,
    )
