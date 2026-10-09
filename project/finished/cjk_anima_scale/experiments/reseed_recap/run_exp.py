#!/usr/bin/env python
"""reseed_recap — grid_44's recap on hiragana + katakana, at the kana run's 135 / row

`grid_lone recap` (small glyphs only, plain grid captions) and its reband
`recap_hp` (the bands read off the gradient) ran on the 81 hiragana rows +
``ー`` at 60 steps / row and read far under ``retrain_kana`` (plain read:
words ≤ 1 edit 14–16 vs 45 of 72, singles official 12 vs 34 of 64) with
rows, budget and composition all different from it
(`../../../cjk_anima_reseed/reports/grad_bands_2026_10_03.md` § What it does
not show). This arm (user, 10-03) gives the recipe the kana run's rows and
budget: the 81 hiragana + 85 katakana rows (``retrain_kana``'s vocabs less
its 8 punctuation singles — ``・ ー ヴ ヶ ヵ`` stay), cold, 135 steps / row =
22 410, old seed underneath.

The table is `grid_44`'s (three sizes of grid and lone, bubble fit, jitter,
the lone gate) with half of `grid_44` given to ``builder.TABLE``'s
`bubble1_52` — the one glyph filling a scene bubble, the form the gradient
read pays ≈ 3× a grid cell and 10–20× a lone 1×1 per draw, and the tier
every grid arm lacked (the kana run: 17 % of its items). A fifth of it is
over 64 px (42 / 52 / 75, max 136). The captions are the ``recap`` (plain)
ones. ``--bands``:

| tier | share | of the items | `hp` (default) | `law` |
|---|---|---|---|---|
| `grid_44` | 0.1125 | 7.5 % | 0.55–0.75 | 0.7–0.9 |
| `bubble1_52` | 0.1125 | 7.5 % | 0.55–0.8 | 0.7–0.9 |
| `lone_44` | 0.075 | 5 % | 0.45–0.8 | 0.7–0.9 |
| `grid_29` | 0.225 | 15 % | 0.4–0.7 | 0.5–0.7 |
| `lone_28` | 0.075 | 5 % | 0.35–0.65 | 0.5–0.7 |
| `grid_16` | 0.225 | 15 % | 0.2–0.6 | 0.3–0.5 |
| `lone_16` | 0.075 | 5 % | 0.2–0.5 | 0.3–0.5 |
| `bubbleN_34` + `bubble1_32` | 0.3 | 14 + 6 % | 0.45–0.7, 0.35–0.6 | 0.5–0.7 |
| `bubbleN_18` | 0.3 | 20 % | 0.2–0.5 | 0.3–0.5 |

`hp` = upper edge at the identity share's half point, lower edge at half
the identity gradient's peak (`grad_bands_2026_10_03.md`): `grid_lone`'s
``recap_hp`` bands, which tied the law's at 60 / row, and for the three
44–50 px tiers the report's untrained ones — there the upper edge comes
down from 0.9, where the small tiers' bands only widened downward. `law` =
the band law's per-px bands, as built. Σ shares 1.5 → ≈ 16 600 items.
Windows are drawn from the 164 letter rows (``・`` is punctuation: lone
only), the kana run's `read` held out.

Legs:
- ``data`` (CPU) → ``OUT/run1003_reseed_recap/data``;
- ``recap`` (CPU) → ``…/data_recap`` (`grid_small.derive`: the plain
  captions, the law's bands), and for ``--bands hp`` ``…/data_recap_hp``
  (`grid_lone.reband`);
- ``train`` (GPU) → ``OUT/experiments/reseed_recap_cold_kana_<bands>``;
- ``read`` (GPU): `kana_reband`'s on every key — the kana run's 13 words `en`
  + 14 singles `swap`, paired against ``retrain_kana``'s reads of record;
- ``read_plain`` (GPU): `grid_lone`'s plain clause on the same 27 keys, this
  arm against ``retrain_kana`` (its katakana keys rendered once), and on the
  hiragana keys against `grid_lone`'s ``recap`` / ``recap_hp`` caches.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/reseed_recap/run_exp.py \\
      --label hp --legs data recap train read read_plain"
    # the law's bands on the same items
    … --label law --legs recap train read read_plain --bands law
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

NAME = "reseed_recap"
SRC_RUN = "retrain_kana"
DATA_RUN = f"run1003_{NAME}"
ARM = f"{NAME}_cold_kana"
STEPS_PER_ROW = 135  # the kana run's
PUNCT = "、。「」〜！？～"  # the kana run's punctuation singles: left at the seed
LARGE_SHARE = 0.1125  # `grid_44` and `bubble1_52`: grid_44's 0.225, halved
# the 44–50 px tiers' bands from grad_identity's pass 2, as `grid_lone.BANDS`
# (grad_bands_2026_10_03.md's table: untrained before this arm)
BANDS_44 = {
    "grid_44": (0.55, 0.75),
    "lone_44": (0.45, 0.8),
    "bubble1_52": (0.55, 0.8),
}


def table(GS, GL) -> tuple:
    from cjk_scale.builder import Group, tiers

    groups = load_experiment("grid_44").table(GS, GL)
    assert groups[0].label == "grid_44", groups[0].label
    return (
        dataclasses.replace(groups[0], share=LARGE_SHARE),
        Group("single", (0.7, 0.9), LARGE_SHARE, tiers("bubble1_52")),
        *groups[1:],
    )


def run_config():
    from cjk_scale.config import load_run

    src = json.loads((OUT / SRC_RUN / "data" / "vocabs.json").read_text("utf-8"))
    rows = [v for v in src if v not in PUNCT]
    hira = [v for v in rows if "ぁ" <= v <= "ゖ"]
    assert (len(hira), len(rows) - len(hira)) == (81, 85), (len(hira), len(rows))
    return dataclasses.replace(
        load_run(SRC_RUN), name=DATA_RUN, vocabs=["chars:" + "".join(rows)]
    ), rows


def read(arm: str) -> dict:
    from cjk_scale import reads as R

    KR = load_experiment("kana_reband")
    out = KR.read(arm, "all")
    RR = load_experiment("retrain_read")
    c2 = [w for w in out["words"] if w in RR.C2_WORDS]
    print("===== p1_cold · C2 words (cache)", flush=True)
    out["p1_cold"] = {
        "en": R.tally(
            RR.grid_hits(
                RR.P2.EXP / "p1_cold" / RR.FLOOR_CACHE["en"] / "native_reads.json",
                c2,
                "en",
            )
        )
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "recap", "train", "read", "read_plain"],
    )
    p.add_argument("--bands", default="hp", choices=["hp", "law"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    GS = load_experiment("grid_small")
    GL = load_experiment("grid_lone")
    KR = load_experiment("kana_reband")
    rc, rows = run_config()
    tbl = table(GS, GL)
    arm = f"{ARM}_{args.bands}"
    root = OUT / DATA_RUN
    data_dir = root / ("data_recap_hp" if args.bands == "hp" else "data_recap")
    steps = STEPS_PER_ROW * len(rows)
    shares = {g.label: g.share for g in tbl}
    budget = KR.check_trainer(STEPS_PER_ROW)
    print(
        f"{arm}: {len(rows)} rows (hiragana 81 + katakana 85) cold × {STEPS_PER_ROW} = "
        f"{steps} steps on {SEED_ROWS_0921}; groups {shares} "
        f"(Σ {sum(shares.values()):g}); bands {args.bands}; data {data_dir}; {budget}",
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {
        "rows": len(rows),
        "steps": steps,
        "shares": shares,
        "arm": arm,
        "bands": args.bands,
        **budget,
    }
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = GS.data(rc, rows, args.workers, tbl, max_px=None)
        print(json.dumps(metrics["data"], ensure_ascii=False, indent=1), flush=True)
    if "recap" in args.legs:
        if not (root / "data_recap" / "train.jsonl").exists():
            metrics["derive"] = GS.derive(
                root / "data", root / "data_recap", None, True
            )
            print(
                json.dumps(metrics["derive"], ensure_ascii=False, indent=1), flush=True
            )
        if args.bands == "hp":
            metrics["reband"] = GL.reband(
                root / "data_recap", data_dir, GL.BANDS["recap_hp"] | BANDS_44
            )
            print(
                json.dumps(metrics["reband"], ensure_ascii=False, indent=1), flush=True
            )
    if "train" in args.legs:
        from cjk_scale import train as T

        T.train(
            dataclasses.replace(rc, name=arm),
            data=data_dir,
            out=OUT / "experiments" / arm,
            cold=True,
            steps_per_row=STEPS_PER_ROW,
            context=SEED_ROWS_0921,
        )
    if "read" in args.legs:
        metrics["read"] = read(arm)
    if "read_plain" in args.legs:
        here = {arm: OUT / "experiments" / arm}
        metrics["read_plain"] = GL.read_plain(here | {SRC_RUN: OUT / SRC_RUN}, "all")
        hira_arms = {
            f"{GL.ARM}_{t}": OUT / "experiments" / f"{GL.ARM}_{t}"
            for t in ("recap", "recap_hp")
        }
        metrics["read_plain_hira"] = GL.read_plain(here | hira_arms)
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
