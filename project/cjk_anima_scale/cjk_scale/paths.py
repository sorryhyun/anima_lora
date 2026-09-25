"""Where the line lives, where its vendored stage packages are, where runs land.

A run lands in one dir, ``output/cjk_anima_scale/<run>/``:

    data/          the items (``img/``, ``train.jsonl``, ``eval.json``,
                   ``vocabs.json``, ``build.json``, latent + TE caches)
    trained.pt     the table (the run's vocabs' rows only)
    ctx/ floor/    the eval arms — ``overwrite(seed, trained)`` and
                   ``load(seed)`` — each a rows-arm dir the stages write into
    sheet.png      floor and trained side by side, every ruler
    reads.json     the numbers behind the sheet
    conflict/      ``scale.py <run> conflict``

Beside the runs: the scene pools (``scenes_<tag>``), the EN reference cache
(``native_enref``), the seed table (``rows_step1_0921_merged`` — its floor
reads of record live flat in it) and the old stage-layout records
(``{data,rows}_<stage>_<tag>``, readable through ``legacy_*``).

The stage packages (``common`` / ``data`` / ``train`` / ``eval`` /
``scenes``, ``cli``, ``stages``) are this line's own ``src/`` — top-level
names, vendored 2026-09-25; ``bootstrap()`` puts ``src/`` on ``sys.path``.
Their ``common.paths.OUT`` is this same root, and their ``data_dir`` /
``arm_dir`` take the run's dirs through ``--data_path`` / ``--arm_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_anima_scale
REPO = LINE.parents[1]
SRC = LINE / "src"
CONFIGS = LINE / "configs"
RUN_CONFIGS = CONFIGS / "runs"
RUNS = LINE / "runs"
LEDGER = RUNS / "ledger.jsonl"
OUT = REPO / "output" / "cjk_anima_scale"
UNITS_DIR = LINE / "assets" / "units"
# the seed table (one constant, plan.md § 2): the probe line's step1_0921 +
# step1_0921z merge, 2 274 rows. Every run's vocabs train from it, every other
# row rides frozen at it, and the floor arm is it whole.
SEED_TABLE = OUT / "rows_step1_0921_merged" / "trained.pt"


def bootstrap() -> None:
    """Put the repo and the line's ``src/`` on ``sys.path`` (``src/`` first:
    its top-level ``train`` must win over the repo root's ``train.py``) and
    pin the stages' output root to ours (idempotent; ``common.paths``
    already defaults to it)."""
    for p in (REPO, SRC):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    import common.paths as stage_paths

    stage_paths.OUT = OUT


def _check(run: str) -> str:
    assert run and "/" not in run and " " not in run, f"bad run name {run!r}"
    return run


def run_dir(run: str) -> Path:
    return OUT / _check(run)


def data_dir(run: str) -> Path:
    return run_dir(run) / "data"


def table_path(run: str) -> Path:
    return run_dir(run) / "trained.pt"


def arm_dir(run: str, arm: str) -> Path:
    """An eval arm dir under the run (``ctx`` / ``floor``): a rows-arm dir
    holding its ``trained.pt`` and every ruler's reads."""
    assert arm in ("ctx", "floor"), arm
    return run_dir(run) / arm


# ---------------------------------------------------------------------------
# the stage layout before 2026-09-25 — records only (experiments/ read them)


def legacy_data_dir(stage: str, tag: str) -> Path:
    return OUT / f"data_{stage}_{tag}"


def legacy_arm_dir(stage: str, tag: str) -> Path:
    return OUT / f"rows_{stage}_{tag}"
