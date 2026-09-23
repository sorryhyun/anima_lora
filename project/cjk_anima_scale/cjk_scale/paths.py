"""Where the line lives, where the probe line's primitives are, where runs land.

Everything lands under ``output/cjk_anima_scale/``: the stage dirs
(``data_scale_<stage>_<tag>`` / ``rows_scale_<stage>_<tag>``, the probe's
layout), the scene pools (``scenes_<tag>``), the EN reference cache
(``native_enref``) and the seed table (``rows_step1_0921_merged``).
``bootstrap()`` points the probe's ``common.paths.OUT`` at the same root, so
its ``load_scenes`` / ``eval`` / ``native`` / ``cf_sense`` and every
``probe/*.py`` reader open a stage table exactly as they open a probe table.
The probe line keeps ``output/wake_probe/`` for its own runs; the pools it
shares with this line are symlinks there.
"""

from __future__ import annotations

import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_anima_scale
REPO = LINE.parents[1]
PROBE_SRC = REPO / "project" / "cjk_renderable_anima" / "src"
CONFIGS = LINE / "configs"
RUNS = LINE / "runs"
LEDGER = RUNS / "ledger.jsonl"
OUT = REPO / "output" / "cjk_anima_scale"
UNITS_DIR = REPO / "project" / "cjk_renderable_anima" / "assets" / "units"


def bootstrap() -> None:
    """Put the repo and the probe line's ``src/`` on ``sys.path`` and point the
    probe's output root at ours (idempotent). The probe's packages are
    top-level names (``common``, ``data``, ``train``, ``eval``); this package
    is ``cjk_scale`` so the two never shadow each other.

    The redirect must happen before any other probe module is imported:
    ``data.synth`` / ``eval.enref`` / ``scenes.stage`` bind ``OUT`` at import
    (``from common.paths import OUT``), so they read whatever it was when
    they first loaded. ``scale.py`` and the tests call this first."""
    for p in (REPO, PROBE_SRC):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    import common.paths as probe_paths

    probe_paths.OUT = OUT


def run_tag(stage: str, tag: str) -> str:
    assert tag and "/" not in tag and " " not in tag, f"bad tag {tag!r}"
    return f"scale_{stage}_{tag}"


def data_dir(stage: str, tag: str) -> Path:
    return OUT / f"data_{run_tag(stage, tag)}"


def arm_dir(stage: str, tag: str) -> Path:
    return OUT / f"rows_{run_tag(stage, tag)}"
