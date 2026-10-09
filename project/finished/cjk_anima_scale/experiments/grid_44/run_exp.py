#!/usr/bin/env python
"""grid_44 — grid_lone with a 44 px tier: half of grid_29 and a third of the lone items

`grid_lone` drew its grids at two sizes, `grid_29` 30 % and `grid_16` 15 %
of the items, and nothing above 40 px. This arm (user, 10-02) moves half of
`grid_29` to a larger tier and gives it a lone twin, the lone share split
evenly over the three sizes:

| tier | σ | share | | of the items |
|---|---|---|---|---|
| `grid_44` | 0.7–0.9 | 0.225 | `grid` 2×2 – 3×3, glyph 44–62 font px | 15 % |
| `grid_29` | 0.5–0.7 | 0.225 | `grid` 2×2 – 3×3, glyph 28–42 font px | 15 % |
| `grid_16` | 0.3–0.5 | 0.225 | `grid` 2×2 – 3×3, glyph 14–26 font px | 15 % |
| `lone_44` | 0.7–0.9 | 0.075 | `grid` 1×1, glyph 46–64 font px | 5 % |
| `lone_28` | 0.5–0.7 | 0.075 | `grid` 1×1, glyph 28–42 font px | 5 % |
| `lone_16` | 0.3–0.5 | 0.075 | `grid` 1×1, glyph 14–26 font px | 5 % |
| `bubbleN_34` + `bubble1_32` | 0.5–0.7 | 0.3 | `builder.TABLE`'s, 0.7 : 0.3 | 20 % |
| `bubbleN_18` | 0.3–0.5 | 0.3 | `builder.TABLE`'s | 20 % |

Σ shares 1.5, rows (81 hiragana + ``ー``), budget (60 / row), bubble fit,
jitter, windows and seed as `grid_lone`. The 44 px tiers take the band the
law gives a single at ≥ 40 px, 0.7–0.9 (`windows.ROWS`): `grid_44` through
the gate (an item under 40 px is re-drawn), `lone_44` ungated at its grid
twin's band, as the other lone tiers.

Legs, as `grid_lone`'s:
- ``data`` (CPU) → ``OUT/run1002_grid_44/data``;
- ``recap`` (CPU) → ``OUT/run1002_grid_44/data_<tag>``: the plain captions on
  every grid and lone item (`grid_small.derive`); with ``--band lo hi``
  every item's band is replaced by it as well (`grid_small`'s ``b7593``:
  the items are the per-px build's — what the gate kept — and only the σ
  they train at moves);
- ``train`` (GPU) → ``OUT/experiments/grid_44_cold_hira[_<tag>]``;
- ``read`` (GPU): `grid_small`'s (the hiragana words `en` + singles `swap`
  against ``retrain_kana``'s reads of record);
- ``read_plain`` (GPU): `grid_lone`'s plain clause on this arm, on
  ``grid_lone``'s recap arm and on ``retrain_kana``, paired.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/grid_44/run_exp.py \\
      --label recap --legs data recap train read read_plain --tag recap"
    # the same, every item at σ 0.75–0.93 (user, 10-02)
    … --label recap_b7593 --legs data recap train read read_plain \\
      --tag recap_b7593 --band 0.75 0.93
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale import paths  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS_0921, bootstrap, load_experiment  # noqa: E402

bootstrap()
paths.pin_old_seed()  # the kana run's seed: the build's context and the floor
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "grid_44"
DATA_RUN = f"run1002_{NAME}"
ARM = f"{NAME}_cold_hira"
# size → (band, the grid's font px, its lone twin and its font px). The 44 px
# grid keeps the ≥ 40 px side of its draw (the gate), the ungated lone twin
# does not: at the grid's font px it is built at 42.5, so it is drawn 2 px up
SIZES = {
    "44": ((0.7, 0.9), [44, 62], "lone_44", [46, 64]),
    "29": ((0.5, 0.7), [28, 42], "lone_28", [28, 42]),
    "16": ((0.3, 0.5), [14, 26], "lone_16", [14, 26]),
}
GRID_SHARE, LONE_SHARE = 0.225, 0.075


def table(GS, GL) -> tuple:
    from cjk_scale.builder import Group, Tier, tiers

    def tier(name: str, px: list, grids: str) -> tuple:
        params = {
            "grids": grids,
            "glyph_px": px,
            "bubble_frac": 0.5,
            "bubble_fit": GS.BUBBLE_FIT,
            "cell_jitter": GS.CELL_JITTER,
            "mark_horizontal": True,
        }
        if grids == GL.LONE:
            params["gate"] = "group"  # the grid twin's band, whatever the ink px
        return (Tier(name, "grid", 1.0, params),)

    return (
        *(
            Group("single", band, GRID_SHARE, tier(f"grid_{k}", px, GS.GRIDS))
            for k, (band, px, _lone, _px) in SIZES.items()
        ),
        *(
            Group("single", band, LONE_SHARE, tier(lone, px, GL.LONE))
            for band, _px, lone, px in SIZES.values()
        ),
        Group("single", (0.5, 0.7), 0.3, tiers("bubbleN_34", "bubble1_32")),
        Group("single", (0.3, 0.5), 0.3, tiers("bubbleN_18")),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "recap", "train", "read", "read_plain"],
    )
    p.add_argument("--tag", default="", help="recap: suffix for the data dir and arm")
    p.add_argument("--band", type=float, nargs=2, help="recap: one band for every item")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    assert ("recap" in args.legs) <= bool(args.tag), "recap takes --tag"
    assert not args.band or "recap" in args.legs, "--band is the recap leg's"
    GS = load_experiment("grid_small")
    GL = load_experiment("grid_lone")
    rc, rows = GL.run_config()
    rc = dataclasses.replace(rc, name=DATA_RUN)
    tbl = table(GS, GL)
    arm = ARM + (f"_{args.tag}" if args.tag else "")
    data_dir = OUT / DATA_RUN / ("data" + (f"_{args.tag}" if args.tag else ""))
    steps = GL.STEPS_PER_ROW * len(rows)
    shares = {g.label: g.share for g in tbl}
    print(
        f"{arm}: {len(rows)} rows (hiragana + {GL.EXTRA_ROWS}) cold × "
        f"{GL.STEPS_PER_ROW} = {steps} steps on {SEED_ROWS_0921}; tiers {shares} "
        f"(Σ {sum(shares.values()):g}); data {data_dir}"
        + (f"; one band {args.band}" if args.band else ""),
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {
        "rows": len(rows),
        "steps": steps,
        "shares": shares,
        "arm": arm,
        "band": args.band,
    }
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = GS.data(rc, rows, args.workers, tbl)
        print(json.dumps(metrics["data"], ensure_ascii=False, indent=1), flush=True)
    if "recap" in args.legs:
        metrics["derive"] = GS.derive(
            OUT / DATA_RUN / "data",
            data_dir,
            tuple(args.band) if args.band else None,
            True,
        )
        print(json.dumps(metrics["derive"], ensure_ascii=False, indent=1), flush=True)
    if "train" in args.legs:
        from cjk_scale import train as T

        T.train(
            dataclasses.replace(rc, name=arm),
            data=data_dir,
            out=OUT / "experiments" / arm,
            cold=True,
            steps_per_row=GL.STEPS_PER_ROW,
            context=SEED_ROWS_0921,
        )
    if "read" in args.legs:
        metrics["read"] = GS.read(arm)
    if "read_plain" in args.legs:
        metrics["read_plain"] = GL.read_plain(
            {
                arm: OUT / "experiments" / arm,
                f"{GL.ARM}_recap": OUT / "experiments" / f"{GL.ARM}_recap",
                GL.SRC_RUN: OUT / GL.SRC_RUN,
            }
        )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(data_dir), str(OUT / "experiments" / arm)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
