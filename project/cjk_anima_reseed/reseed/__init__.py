"""reseed — the reseed line's run configs and data build.

One table (``table.TABLE``): every tier carries its own glyph px, σ band
and share; the builder draws each tier once and stamps its band. Where the
table came from and what it leaves out of the scale builder: ``README.md``.

``src/`` holds the renderers, scene pools, inventory, readers and train
plumbing, vendored from ``../cjk_anima_scale/src/`` under their top-level
names (``common`` / ``data`` / ``train`` / ``eval``). The trainer is still
``cjk_scale.{paths, config, budget, train}``, read-only from here; its
``cjk_scale.builder`` / ``recipes`` rebuild the seed of record and are never
imported (``tests/``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]  # project/cjk_anima_reseed
SCALE = HOME.parent / "cjk_anima_scale"
REPO = HOME.parents[1]
OUT = REPO / "output" / "cjk_anima_reseed"
CONFIGS = HOME / "configs"
SRC = HOME / "src"


def bootstrap() -> None:
    """``src/`` first on ``sys.path`` (its ``train`` must win over the repo's
    ``train.py``), then the repo, then the scale line's dir (``cjk_scale``
    importable; its ``common`` / ``data`` / ``train`` resolve to ``src/``);
    the pack named."""
    os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")
    for p in (SCALE, REPO, SRC):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
