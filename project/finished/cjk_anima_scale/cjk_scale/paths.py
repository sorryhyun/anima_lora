"""Where the line lives, where its vendored stage packages are, where runs land.

A run lands in one dir, ``output/cjk_anima_scale/<run>/``:

    data/          the items (``img/``, ``train.jsonl``, ``eval.json``,
                   ``vocabs.json``, ``build.json``, latent + TE caches)
    trained.pt     the merged rows — the seed's rows (rescaled into the run's
                   ``row_scale``) with the run's vocabs' rows on top, whole
    native/ …      the trained side's ruler outputs, at the run root (the
                   run dir is the trained arm; no ``ctx/`` sidecar)
    sheet.png      floor and trained side by side, every ruler
    reads.json     the numbers behind the sheet
    conflict/      ``scale.py <run> conflict``

Beside the runs: the scene pools (``scenes_<tag>``), the EN reference cache
(``native_enref``), the seed rows (``seed_retrain_0930`` — the floor arm:
every run's floor reads are cached flat in it; ``rows_step1_0921_merged`` the
old seed's) and the old stage-layout records
(``{data,rows}_<stage>_<tag>``, read-only).

The stage packages (``common`` / ``data`` / ``train`` / ``eval`` /
``scenes``, ``cli``, ``stages``) are this line's own ``src/`` — top-level
names, vendored 2026-09-25; ``bootstrap()`` puts ``src/`` on ``sys.path``.
Their ``common.paths.OUT`` is this same root, and their ``data_dir`` /
``arm_dir`` take the run's dirs through ``--data_path`` / ``--arm_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/finished/cjk_anima_scale
REPO = LINE.parents[2]
SRC = LINE / "src"
CONFIGS = LINE / "configs"
RUN_CONFIGS = CONFIGS / "runs"
RUNS = LINE / "runs"
LEDGER = RUNS / "ledger.jsonl"
OUT = REPO / "output" / "cjk_anima_scale"
VOCABS_DIR = LINE / "assets" / "vocabs"
# the seed rows (one constant, plan.md § 2): since 2026-09-30 the singles retrain
# (plan_retrain § 3) — ``retrain_kanji_b4``'s merged rows (the old seed + retrain_kana
# + kanji b1–b4, 2 683 rows) copied whole into their own dir (``seed.json`` there),
# so the floor cache is not a run's dir. Every run's vocabs train from it, every
# other row rides frozen at it, and the floor arm is it whole.
SEED_ROWS = OUT / "seed_retrain_0930" / "trained.pt"
# the old seed: the probe line's step1_0921 + step1_0921z merge, 2 274 rows — the
# floor of record the retrain itself is read against (``floor_score.md``); the
# retrain's experiments read their floor from it
SEED_ROWS_0921 = OUT / "rows_step1_0921_merged" / "trained.pt"
# the raw pack's digest (``VocabPack.digest`` with the encode fold left out — the
# load log said ``sha 7b9fce0bb57b…`` before the pack json took ``fold``):
# the seed rows are deltas over it and a cold row starts at its rows, so the
# trainer refuses any other attached pack (colab.md's pack row)
RAW_PACK_SHA = "7b9fce0bb57b"
# the punct base pack (``../../cjk_anima_reseed/punct_pack.py``, 10-05): the raw
# pack's rows and ids, a `…` row appended and dot runs (``dots``) on top —
# accepted beside it (the same digest, fold left out)
PUNCT_PACK_SHA = "6c09442810af"


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


def pin_old_seed() -> None:
    """Point this process's seed rows, and so its floor cache, at the old seed:
    the retrain's experiments (``experiments/``) read against the floor of
    record, not the seed they produced."""
    global SEED_ROWS
    SEED_ROWS = SEED_ROWS_0921


def _check(run: str) -> str:
    assert run and "/" not in run and " " not in run, f"bad run name {run!r}"
    return run


def run_dir(run: str) -> Path:
    return OUT / _check(run)


def data_dir(run: str) -> Path:
    return run_dir(run) / "data"


def trained_path(run: str) -> Path:
    """The run's merged rows: the seed's rows with the run's on top, one
    ``row_scale`` (written whole by ``rows.Rows.state_dict``)."""
    return run_dir(run) / "trained.pt"


def floor_dir(seed: Path | None = None) -> Path:
    """The floor arm: the seed rows' own dir (its ``trained.pt`` is the seed,
    whole). One read cache for every run — a ruler renders only the strings
    no earlier run read (``eval.ensure_floor``); since 2026-09-26, before
    which each run carried a ``<run>/floor/`` copy. ``seed``: another seed's
    rows (``SEED_ROWS_0921`` = the old floor of record)."""
    return OUT / (seed or SEED_ROWS).parent.name


def load_experiment(name: str):
    """``experiments/<name>/run_exp.py`` as a module (``<name>_exp``), for an
    experiment that builds on another's builders or reads. Executed fresh on
    every call, never cached in ``sys.modules`` — its import-time side
    effects (``pin_old_seed``) run again, as each experiment's own copy did."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"{name}_exp", LINE / "experiments" / name / "run_exp.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
