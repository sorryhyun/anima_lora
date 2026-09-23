"""Where the line lives, where the probe line's primitives are, where runs land.

Outputs go under the probe line's ``output/wake_probe/`` with a ``scale_``
prefix (``data_scale_<stage>_<tag>`` / ``rows_scale_<stage>_<tag>``), so the
probe's ``eval`` / ``native`` / ``cf_sense`` stages and every ``probe/*.py``
reader open a stage table exactly as they open a probe table.
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
OUT = REPO / "output" / "wake_probe"
UNITS_DIR = REPO / "project" / "cjk_renderable_anima" / "assets" / "units"


def bootstrap() -> None:
    """Put the repo and the probe line's ``src/`` on ``sys.path`` (idempotent).
    The probe's packages are top-level names (``common``, ``data``, ``train``,
    ``eval``); this package is ``cjk_scale`` so the two never shadow each other."""
    for p in (REPO, PROBE_SRC):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def run_tag(stage: str, tag: str) -> str:
    assert tag and "/" not in tag and " " not in tag, f"bad tag {tag!r}"
    return f"scale_{stage}_{tag}"


def data_dir(stage: str, tag: str) -> Path:
    return OUT / f"data_{run_tag(stage, tag)}"


def arm_dir(stage: str, tag: str) -> Path:
    return OUT / f"rows_{run_tag(stage, tag)}"
