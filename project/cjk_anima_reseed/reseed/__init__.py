"""reseed — the reseed line's run configs and data build.

One table (``table.TABLE``): every tier carries its own glyph px, σ band
and share; the builder draws each tier once and stamps its band. Where the
table came from and what it leaves out of the scale builder: ``README.md``.

``src/`` holds the renderers, scene pools, inventory, readers and train
plumbing, vendored from the scale line's ``src/`` under their top-level
names (``common`` / ``data`` / ``train`` / ``eval``); the trainer is
``trainer`` / ``rows`` / ``loss``, ported from ``cjk_scale.{train, rows,
loss}``. No live file imports ``cjk_scale`` (``tests/test_boundary.py``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]  # project/cjk_anima_reseed
REPO = HOME.parents[1]
OUT = REPO / "output" / "cjk_anima_reseed"
CONFIGS = HOME / "configs"
SRC = HOME / "src"
# the scale line's outputs, read from here: the seed rows, the arms of record,
# the scene pools, the EN refs; ``pools`` appends to its two scene caches
# (``experiments/scene_{bubble_check,colorful}.json``)
SCALE_OUT = REPO / "output" / "cjk_anima_scale"
# the seed rows: ``seed_retrain_0930`` (the singles retrain's, 2 683 rows) and
# the old seed (the probe line's step1_0921 merge, 2 274 rows)
SEED_ROWS = SCALE_OUT / "seed_retrain_0930" / "trained.pt"
SEED_ROWS_0921 = SCALE_OUT / "rows_step1_0921_merged" / "trained.pt"
# the raw pack's digest with its encode fold left out (the seed rows are deltas
# over it, a cold row starts at it) and the punct pack's (``punct_pack.py``:
# the raw pack's rows and ids, a ``…`` row appended); the trainer takes either
RAW_PACK_SHA = "7b9fce0bb57b"
PUNCT_PACK_SHA = "6c09442810af"


def bootstrap() -> None:
    """``src/`` first on ``sys.path`` (its ``train`` must win over the repo's
    ``train.py``), then the repo; the pack named."""
    os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")
    for p in (REPO, SRC):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
