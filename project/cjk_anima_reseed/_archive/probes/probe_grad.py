#!/usr/bin/env python
"""probe_grad — where a large glyph's identity gradient sits in σ (2026-10-03)

The question (user, 10-03): a grid tier above ``grid_44`` — at what band?
``retrain_kana``'s large tier (``b0709``, σ 0.7–0.9, grid singles at a median
91 px) is what wrote the glyph alone on a blank page at eval, so its band is
not taken on trust. This reads it off the gradient the way
``_archive/reports/grad_bands_2026_10_03.md`` set the table's bands: the scale line's
``experiments/grad_identity`` pass 2 (row u, the true caption, the image
re-drawn with slot k's glyph swapped) at the same initial point (the 82
hiragana rows cold, every other row at the 0921 seed), on tiers drawn here.

- ``data`` (CPU): ``PROBE`` tiers by ``reseed.builder`` over the grid_44
  read's 82 rows → ``output/cjk_anima_reseed/probe_grad/data``. The bands
  stamped are placeholders; pass 2 sweeps σ.
- ``grad`` (GPU): ``grad_identity --render --captions plain`` with its data
  dirs and tiers pointed here → its ``results/<stamp>-<label>/``. ``grid_44``
  is in ``PROBE`` as the anchor: the 10-02 read put its f half point at 0.75
  and ‖I‖'s low half at 0.55.

    .venv/bin/python project/cjk_anima_reseed/probes/probe_grad.py data
    .venv/bin/python project/cjk_anima_reseed/probes/probe_grad.py grad --dry_run   # CPU: the renders
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_grad.py grad"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()

NAME = "probe_grad"
# the grid_44 read's rows (run1002_grid_44/data/vocabs.json): grad_identity's
# initial point asserts these 82
ROWS = (
    "chars:ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでと"
    "どなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをんーゔ"
)
ITEMS = 80  # drawn per tier; pass 2 keeps 40 (`--items`)
PLACEHOLDER = (0.7, 0.9)  # stamped, not read: pass 2 sweeps σ


def probe_tiers() -> tuple:
    from reseed.table import ITEMS_PER_ROW, _grid

    share = ITEMS / (ITEMS_PER_ROW * 82)
    return (
        _grid("grid_44", share, PLACEHOLDER, [44, 62], px_keep=(40, None)),
        _grid("grid_64", share, PLACEHOLDER, [66, 92], px_keep=(56, None)),
        _grid("grid_96", share, PLACEHOLDER, [100, 140], px_keep=(84, None)),
        _grid("lone_64", share, PLACEHOLDER, [66, 92], lone=True, px_keep=(56, None)),
        _grid("lone_96", share, PLACEHOLDER, [100, 140], lone=True, px_keep=(84, None)),
    )


def run():
    from reseed.config import Run

    return Run(
        name=NAME,
        path=Path(__file__),
        rows=(ROWS,),
        read=(),
        seed="0921",
        steps_per_row=0,
    )


def data(workers):
    import reseed.table as T
    from reseed.builder import build

    T.TABLE = probe_tiers()  # Run.table() reads it at call time
    build(run(), workers, 1.0)


def grad(label: str, dry_run: bool):
    from cjk_scale.paths import load_experiment

    d = run().data
    assert (d / "train.jsonl").exists(), f"no {d} — run `probe_grad.py data` first"
    names = tuple(t.name for t in probe_tiers())
    GI = load_experiment("grad_identity")
    GI.DIRS = {"plain": d, "clause": d}
    GI.TIERS = names
    GI.REPORT_TIERS = names
    GI.TRAINED = GI.TRAINED | {n: PLACEHOLDER for n in names}
    sys.argv = [
        GI.__file__,
        "--label",
        label,
        "--render",
        "--captions",
        "plain",
        "--render_tiers",
        *names,
    ] + (["--dry_run"] if dry_run else [])
    GI.main()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["data", "grad"])
    p.add_argument("--label", default="big")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    if a.verb == "data":
        data(a.workers)
    else:
        grad(a.label, a.dry_run)


if __name__ == "__main__":
    main()
