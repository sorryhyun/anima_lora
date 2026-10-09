#!/usr/bin/env python
"""grid_small — the hiragana rows cold on small glyphs only, grids the larger share

The seed's lone tiers are large (`bubble1_52`, `grid_82`, `lone_190`, at
σ 0.7–0.9), drawn large for identity. `p1_cold` (36 hiragana,
cold, in-word items at 0.3–0.7 only) says identity does not need it: words
≤ 1 edit 65 / 128 with no large glyph (retrain_experiments § 4). This arm
asks it of grids: the 81 hiragana rows cold, **no lone glyph above 40 px**
(as built: grid glyphs 12–36 px, words 14–64 px, p95 50), and the lone tier
replaced by multi-cell grids of small glyphs at the band law's bands for
their px.

| tier | σ | share | |
|---|---|---|---|
| `grid_29` | 0.5–0.7 | 0.45 | `grid` 2×2 – 3×3, glyph 28–42 font px |
| `grid_16` | 0.3–0.5 | 0.45 | `grid` 2×2 – 3×3, glyph 14–26 font px |
| `bubbleN_34` + `bubble1_32` | 0.5–0.7 | 0.3 | `builder.TABLE`'s, 0.7 : 0.3 |
| `bubbleN_18` | 0.3–0.5 | 0.3 | `builder.TABLE`'s |

(The data dir of record was built before the tiers were named by px,
2026-10-02: its records say `g0507` / `g0305` / `b0507` / `b0305` and
`grid_single` / `scene_window` / `scene_single_small` — `builder.tier_of`.)

Σ shares 1.5 as `builder.TABLE`'s single kind, so the items (≈ 8 100) and
the budget (135 / row = 10 935 steps) are the kana run's per row; grids are
60 % of the items (the kana run: 17 %, cells 52–232 px). No 1×1 (a flat
canvas with one small glyph). A cell's bubble is sized to its glyph
(`render_grid(bubble_fit=…)`, user 10-02: the cell-filling bubble around a
16 px glyph is not a speech bubble), and the glyph sits at its cell's centre
± 8 % of the cell (`cell_jitter`: the position clause names the cell, and a
small glyph placed anywhere in a 170–256 px cell does not show the grid). Windows are drawn from the hiragana rows
alone, the kana run's `read` held out. Old seed underneath, as the kana run.

Legs:
- ``data`` (CPU) → ``OUT/run1002_grid_small/data``;
- ``reband`` (CPU) → ``OUT/run1002_grid_small/data_<tag>``: ``data``'s records
  with every band replaced by ``--band`` (images, latents and the TE cache
  shared with ``data`` — same files, same captions, same order). The
  per-px arm is the control; this is the one-band treatment (user, 10-02);
- ``recap`` (CPU) → the same ``data_<tag>``: every grid item's caption
  rewritten plain (user, 10-02) — no ``manga``, no ``japanese text``, the
  clause without its language (``simple background, no humans, multiple
  speech bubbles. On the top left, text reads as "ご". …`` / ``white
  background, simple background, no humans, text focus. …``), its wording
  drawn per item from ``common.prompts.GRID_CLAUSES``. Images, latents and
  order are ``data``'s; the TE cache is the dir's own (the captions are its
  key). Scene captions are untouched. Takes ``--band`` with ``reband``;
- ``train`` (GPU) → ``OUT/experiments/grid_small_cold_hira[_<tag>]``;
- ``read`` (GPU): ``kana_reband``'s read — `retrain_read`'s grid, 9 hiragana
  words `en` + 8 singles `swap`, paired against ``retrain_kana``'s reads of
  record; C2's words also against the `p1_cold` / `p1_mix` caches.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/grid_small/run_exp.py \\
      --label r0 --legs train read"
    # the one-band treatment on the same items
    … --label b7593 --legs reband train read --band 0.75 0.93 --tag b7593
    # the plain grid captions on the same items (per-px bands)
    … --label recap --legs recap train read --tag recap
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
from collections import Counter
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

NAME = "grid_small"
SRC_RUN = "retrain_kana"
DATA_RUN = f"run1002_{NAME}"
ARM = f"{NAME}_cold_hira"
STEPS_PER_ROW = 135  # the kana run's
GRIDS = "2x2:1,3x3:1,2x3:1,3x2:1"
BUBBLE_FIT = [1.15, 1.7]  # the bubble's inscribed rectangle, × the glyph
CELL_JITTER = 0.08  # the glyph at its cell's centre ± this share of the cell


def table() -> tuple:
    from cjk_scale.builder import Group, Tier, tiers

    def grid(name: str, px: list) -> tuple:
        return (
            Tier(
                name,
                "grid",
                1.0,
                {
                    "grids": GRIDS,
                    "glyph_px": px,
                    "bubble_frac": 0.5,
                    "bubble_fit": BUBBLE_FIT,
                    "cell_jitter": CELL_JITTER,
                    "mark_horizontal": True,
                },
            ),
        )

    return (
        Group("single", (0.5, 0.7), 0.45, grid("grid_29", [28, 42])),
        Group("single", (0.3, 0.5), 0.45, grid("grid_16", [14, 26])),
        Group("single", (0.5, 0.7), 0.3, tiers("bubbleN_34", "bubble1_32")),
        Group("single", (0.3, 0.5), 0.3, tiers("bubbleN_18")),
    )


def run_config():
    from cjk_scale.config import load_run

    src = json.loads((OUT / SRC_RUN / "data" / "vocabs.json").read_text("utf-8"))
    hira = [v for v in src if all("ぁ" <= c <= "ゖ" for c in v)]
    assert len(hira) == 81, len(hira)
    return dataclasses.replace(
        load_run(SRC_RUN), name=DATA_RUN, vocabs=["chars:" + "".join(hira)]
    ), hira


def data(
    rc,
    hira: list,
    workers: int | None,
    tbl: tuple | None = None,
    max_px: float | None = 64,
    prepare=None,
    base_chars: str = "",
) -> dict:
    """``prepare`` is `builder.build`'s; ``base_chars`` = what the items may
    draw beside the rows (`reseed_anchor`: ！ ？ and the EN words' letters)."""
    import statistics as st

    from cjk_scale.builder import build

    dst = build(rc, workers, tbl or table(), prepare)
    recs = [
        json.loads(ln) for ln in (dst / "train.jsonl").read_text("utf-8").splitlines()
    ]
    drawn = {c for r in recs for c in r["text"].replace(" ", "")} - set(base_chars)
    assert drawn <= set(hira), drawn - set(hira)
    # `reseed_recap` draws `bubble1_52` (a fifth of it over 64 px): no cap there
    assert max_px is None or max(r["px"] for r in recs) <= max_px, max(
        r["px"] for r in recs
    )
    by: dict = {}
    for r in recs:
        by.setdefault(r["tier"], []).append(r["px"])
    return {
        "items": len(recs),
        "rows_drawn": len(drawn),
        "grid_share": round(sum(r["layout"] == "grid" for r in recs) / len(recs), 3),
        "bands": dict(Counter(json.dumps(r["band"]) for r in recs)),
        "px": {
            k: {
                "n": len(v),
                "p5": round(st.quantiles(v, n=20)[0], 1),
                "p50": round(st.median(v), 1),
                "p95": round(st.quantiles(v, n=20)[-1], 1),
                "max": round(max(v), 1),
            }
            for k, v in sorted(by.items())
        },
    }


def plain_caption(r: dict, rng: random.Random) -> tuple:
    """A grid record's caption in the plain frame, its clause drawn from
    `GRID_CLAUSES` (one wording per item)."""
    from cjk_scale.recipes import GRIDS as SHAPES
    from common.prompts import GRID_CLAUSES, GRID_CLAUSES_BUBBLE, grid_caption

    cols, rows, _ = SHAPES[r["grid"]]
    assert len(r["units"]) == cols * rows and not r["horizontal"], r["file"]
    frame = "bubble" if r["bubble"] else "flat"
    clause = rng.choice(
        [c for c in GRID_CLAUSES if frame == "bubble" or c not in GRID_CLAUSES_BUBBLE]
    )
    return grid_caption(frame, cols, rows, r["units"], clause=clause), clause


def derive(src: Path, dst: Path, band: tuple | None, recap: bool) -> dict:
    """``src``'s records at one band (``band``) and / or with the grid
    captions rewritten plain (``recap``); images and latents shared by
    symlink, the TE cache too unless the captions changed."""
    from cjk_scale.builder import tier_of

    recs = [
        json.loads(ln) for ln in (src / "train.jsonl").read_text("utf-8").splitlines()
    ]
    out: dict = {"from": str(src), "items": len(recs)}
    note: dict = {"derived_from": str(src)}
    if band:
        old = Counter((tier_of(r), tuple(r["band"])) for r in recs)
        for r in recs:
            r["band"] = list(band)
        out["old_bands"] = {f"{g} {a}-{b}": n for (g, (a, b)), n in sorted(old.items())}
        out["band"] = note["band"] = list(band)
    if recap:
        rng = random.Random(0)
        for r in recs:
            # the grids and the 1×1 (grid_lone); `grid_single` in the data of record
            if r["recipe"] in ("grid", "grid_single"):
                r["caption"], r["clause"] = plain_caption(r, rng)
        clauses = Counter(
            f"{'bubble' if r['bubble'] else 'flat'}/{r['clause']}"
            for r in recs
            if "clause" in r
        )
        out["recap"] = note["recap"] = dict(sorted(clauses.items()))
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
    )
    for f in ("vocabs.json", "eval.json"):
        shutil.copy2(src / f, dst / f)
    bj = json.loads((src / "build.json").read_text("utf-8"))
    (dst / "build.json").write_text(
        json.dumps(bj | note, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for d in src.iterdir():
        shared = d.name.startswith("latents_") or (d.name == "te_cache" and not recap)
        if d.is_dir() and shared:
            link = dst / d.name
            if not link.exists():
                link.symlink_to(d)
    return out


def read(arm: str) -> dict:
    from cjk_scale import reads as R

    KR = load_experiment("kana_reband")
    out = KR.read(arm, "hira")
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
        choices=["data", "reband", "recap", "train", "read"],
    )
    p.add_argument("--band", type=float, nargs=2, default=None, help="reband: one band")
    p.add_argument("--tag", default="", help="suffix for the data dir and the arm")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    derived = {"reband", "recap"} & set(args.legs)
    assert ("reband" in args.legs) == bool(args.band), "--band is reband's"
    assert bool(derived) == bool(args.tag), (
        "reband / recap take --tag; the other legs do not"
    )
    rc, hira = run_config()
    arm = ARM + (f"_{args.tag}" if args.tag else "")
    data_dir = OUT / DATA_RUN / ("data" + (f"_{args.tag}" if args.tag else ""))
    steps = STEPS_PER_ROW * len(hira)
    shares = {g.label: g.share for g in table()}
    print(
        f"{arm}: {len(hira)} hiragana rows cold × {STEPS_PER_ROW} = {steps} steps on "
        f"{SEED_ROWS_0921}; groups {shares} (Σ {sum(shares.values()):g}); data "
        f"{data_dir}" + (f"; one band {args.band}" if args.band else ""),
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {
        "rows": len(hira),
        "steps": steps,
        "shares": shares,
        "arm": arm,
        "band": args.band,
    }
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = data(rc, hira, args.workers)
        print(json.dumps(metrics["data"], ensure_ascii=False, indent=1), flush=True)
    if derived:
        metrics["derive"] = derive(
            OUT / DATA_RUN / "data",
            data_dir,
            tuple(args.band) if args.band else None,
            "recap" in args.legs,
        )
        print(json.dumps(metrics["derive"], ensure_ascii=False, indent=1), flush=True)
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
