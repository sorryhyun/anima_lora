#!/usr/bin/env python
"""grid_lone — grid_small with the lone glyph back, small: hiragana + ー at 60 / row

`grid_small` drew no 1×1 and read under the kana run (r0: words ≤ 1 edit
12 vs 33 of 72, singles contained 31 vs 44 of 64); its one-band arm
(`b7593`, every item at σ 0.75–0.93) read 0. This arm (user, 10-02) puts the
lone glyph back **at the grids' px** — a 1×1 canvas, bare or one bubble,
15 % of the items, taken from ``grid_16`` (30 → 15 %) — on the 81 hiragana
rows + ``ー`` (82), at 60 steps / row (4 920).

| tier | σ | share | |
|---|---|---|---|
| `grid_29` | 0.5–0.7 | 0.45 | `grid` 2×2 – 3×3, glyph 28–42 font px |
| `grid_16` | 0.3–0.5 | 0.225 | `grid` 2×2 – 3×3, glyph 14–26 font px |
| `lone_28` | 0.5–0.7 | 0.1125 | `grid` 1×1, glyph 28–42 font px |
| `lone_16` | 0.3–0.5 | 0.1125 | `grid` 1×1, glyph 14–26 font px |
| `bubbleN_34` + `bubble1_32` | 0.5–0.7 | 0.3 | `builder.TABLE`'s, 0.7 : 0.3 |
| `bubbleN_18` | 0.3–0.5 | 0.3 | `builder.TABLE`'s |

(The data dirs of record were built before the tiers were named by px,
2026-10-02: their records say `g0507` / `g0305` / `l0507` / `l0305` /
`b0507` / `b0305` and `grid_single` / `scene_window` /
`scene_single_small` — `builder.tier_of`.)

Σ shares 1.5, as `grid_small`. The 1×1's 15 % is split evenly over the two
font-px ranges, each at its grid twin's band **ungated** (`gate = "group"`):
a lone glyph's px is its own ink box, so the first build's gate dropped
``ー`` from both tiers (0.38 × its font px) and most small kana from
`lone_28` (っ 0.58, ぅ 0.60; the median glyph 0.83) — a grid item averages
its cells and keeps them. Bubble fit and jitter are `grid_small`'s: the
bubble is sized to its glyph, and the glyph sits at the canvas centre ± 8 %.
Windows are drawn from the 82 rows alone, the kana run's `read` held out.
Old seed underneath, as the kana run.

Legs:
- ``data`` (CPU) → ``OUT/run1002_grid_lone/data``;
- ``recap`` (CPU) → ``OUT/run1002_grid_lone/data_<tag>``: `grid_small`'s
  plain captions on every grid and lone item, the 1×1 with one bubble and no
  position header (``… no humans, speech bubble. Text reads as "ご".``);
- ``reband`` (CPU) → ``OUT/run1002_grid_lone/data_<tag>``: ``data_recap``'s
  records with each tier's band replaced by ``BANDS[<tag>]``, from
  `reports/grad_identity_2026_10_02.md`. Same items, captions, latents and
  TE cache; the arm against ``recap`` (the law's per-px bands) is the band
  alone (user, 10-02 / 10-03):
  ``recap_h0`` — σ from 0 up to the tier's identity half point (where the
  glyph-dependent share f of a cold row's gradient is half its plateau), no
  lower edge: read under ``recap`` (the draws below the identity peak are
  draws the rows did not get);
  ``recap_hp`` — the same upper edge, the lower edge where the identity
  gradient's size ‖I‖ is half its peak on the low side (0.2, the lowest σ
  read, where a 15–19 px tier's ‖I‖ is still over half its peak);
- ``train`` (GPU) → ``OUT/experiments/grid_lone_cold_hira[_<tag>]``;
- ``read`` (GPU): `grid_small`'s (the hiragana words `en` + singles `swap`
  against ``retrain_kana``'s reads of record);
- ``read_plain`` (GPU): the same words and singles asked in the recap's
  wording (``<scene prompt>. Text reads as "…".`` — no ``japanese text``
  tag, no language in the clause) on both arms and on ``retrain_kana``
  (rendered once into its ``native_r4_plain/``), paired arm against arm.
  The `en` read asks with the clause `r0`'s grid items carry and `recap`'s
  do not (user, 10-02).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/grid_lone/run_exp.py \\
      --label r0 --legs data"
    … --label recap --legs recap train read --tag recap
    … --label plain --legs read_plain
    # the half-point caps on the recap items, no lower edge
    … --label recap_h0 --legs reband train read read_plain --tag recap_h0
    # … with the lower edge at half the identity peak
    … --label recap_hp --legs reband train read read_plain --tag recap_hp
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

NAME = "grid_lone"
SRC_RUN = "retrain_kana"
DATA_RUN = f"run1002_{NAME}"
ARM = f"{NAME}_cold_hira"
STEPS_PER_ROW = 60
LONE = "1x1:1"
EXTRA_ROWS = "ー"  # the long-vowel mark, beside the hiragana
PLAIN_CLAUSE = '{p}. Text reads as "{k}".'  # the recap's wording on a scene prompt
# `reband`, by tag: the tier's band from grad_identity's pass 2. Upper edge =
# f's half point (0.62 / 0.72 / 0.53 / 0.66 / 0.52 / 0.68 / 0.60); `recap_hp`'s
# lower edge = ‖I‖ at half its peak, low side
BANDS = {
    "recap_h0": {
        "grid_16": (0.0, 0.6),
        "grid_29": (0.0, 0.7),
        "lone_16": (0.0, 0.5),
        "lone_28": (0.0, 0.65),
        "bubbleN_18": (0.0, 0.5),
        "bubbleN_34": (0.0, 0.7),
        "bubble1_32": (0.0, 0.6),
    },
    "recap_hp": {
        "grid_16": (0.2, 0.6),
        "grid_29": (0.4, 0.7),
        "lone_16": (0.2, 0.5),  # no ‖I‖ peak: grid_16's lower edge
        "lone_28": (0.35, 0.65),
        "bubbleN_18": (0.2, 0.5),
        "bubbleN_34": (0.45, 0.7),
        "bubble1_32": (0.35, 0.6),
    },
}


def reband(src: Path, dst: Path, bands: dict) -> dict:
    """``src``'s records with every tier's band set to ``bands[tier]``;
    images, latents and the TE cache shared by symlink (captions unchanged)."""
    import shutil
    from collections import Counter

    from cjk_scale.builder import tier_of

    recs = [
        json.loads(ln) for ln in (src / "train.jsonl").read_text("utf-8").splitlines()
    ]
    old = Counter((tier_of(r), tuple(r["band"])) for r in recs)
    for r in recs:
        r["band"] = list(bands[tier_of(r)])
    new = Counter((tier_of(r), tuple(r["band"])) for r in recs)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
    )
    for f in ("vocabs.json", "eval.json"):
        shutil.copy2(src / f, dst / f)
    bj = json.loads((src / "build.json").read_text("utf-8"))
    note = {
        "derived_from": str(src),
        "tier_bands": {t: list(b) for t, b in bands.items()},
    }
    (dst / "build.json").write_text(
        json.dumps(bj | note, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for d in src.iterdir():
        if d.is_dir() and (d.name.startswith("latents_") or d.name == "te_cache"):
            link = dst / d.name
            if not link.exists():
                link.symlink_to(d.resolve())

    def fmt(c):
        return {f"{t} {a:g}-{b:g}": n for (t, (a, b)), n in sorted(c.items())}

    return {"from": str(src), "items": len(recs), "old": fmt(old), "new": fmt(new)}


def table(GS) -> tuple:
    from cjk_scale.builder import Group, Tier, tiers

    def grid(name: str, px: list, grids: str) -> tuple:
        params = {
            "grids": grids,
            "glyph_px": px,
            "bubble_frac": 0.5,
            "bubble_fit": GS.BUBBLE_FIT,
            "cell_jitter": GS.CELL_JITTER,
            "mark_horizontal": True,
        }
        if grids == LONE:
            params["gate"] = "group"  # the grid twin's band, whatever the ink px
        return (Tier(name, "grid", 1.0, params),)

    return (
        Group("single", (0.5, 0.7), 0.45, grid("grid_29", [28, 42], GS.GRIDS)),
        Group("single", (0.3, 0.5), 0.225, grid("grid_16", [14, 26], GS.GRIDS)),
        Group("single", (0.5, 0.7), 0.1125, grid("lone_28", [28, 42], LONE)),
        Group("single", (0.3, 0.5), 0.1125, grid("lone_16", [14, 26], LONE)),
        Group("single", (0.5, 0.7), 0.3, tiers("bubbleN_34", "bubble1_32")),
        Group("single", (0.3, 0.5), 0.3, tiers("bubbleN_18")),
    )


def run_config():
    from cjk_scale.config import load_run

    src = json.loads((OUT / SRC_RUN / "data" / "vocabs.json").read_text("utf-8"))
    rows = [v for v in src if all("ぁ" <= c <= "ゖ" for c in v) or v in EXTRA_ROWS]
    assert len(rows) == 81 + len(EXTRA_ROWS), len(rows)
    return dataclasses.replace(
        load_run(SRC_RUN), name=DATA_RUN, vocabs=["chars:" + "".join(rows)]
    ), rows


def read_plain(arms: dict, rows: str = "hira") -> dict:
    """``arms`` (name → run dir) on `retrain_read`'s grid under
    ``PLAIN_CLAUSE`` (renders in each dir's ``native_r4_plain/``), every pair
    of arms paired on the shared prompt × seed. ``rows="all"`` adds the
    katakana words and singles (`reseed_recap`), paired per script as well."""
    from itertools import combinations

    from cjk_scale import reads as R
    from cjk_scale.config import load_run
    from common import prompts

    prompts.NATIVE_CLAUSES["plain"] = PLAIN_CLAUSE  # in-process, as polish's frame
    RR = load_experiment("retrain_read")
    KR = load_experiment("kana_reband")
    rc = load_run(SRC_RUN)  # its `read` words; the arm is each of `arms`
    words = [w for w in rc.read if rows == "all" or KR.is_hira(w)]
    singles = list(RR.HIRA + (RR.KATA if rows == "all" else ()))
    out: dict = {"clause": PLAIN_CLAUSE, "words": words, "singles": singles}
    hits = {}
    for name, arm in arms.items():
        assert (arm / "trained.pt").exists(), f"{arm}: not trained yet"
        path = RR.ensure(rc, arm, words + singles, "plain")
        hits[name] = RR.grid_hits(path, words + singles, "plain")
        print(f"===== {name} · plain", flush=True)
        out[name] = R.tally(hits[name])
    groups = [("words", words), ("singles", singles)]
    if rows == "all":
        groups += [
            ("words_hira", [w for w in words if KR.is_hira(w)]),
            ("words_kata", [w for w in words if not KR.is_hira(w)]),
            ("singles_hira", list(RR.HIRA)),
            ("singles_kata", list(RR.KATA)),
        ]
    out["paired"] = {}
    for a, b in combinations(arms, 2):
        for g, keys in groups:
            pr = R.paired(
                *({k: v for k, v in hits[x].items() if k[0] in keys} for x in (a, b))
            )
            out["paired"][f"{g}: {a} vs {b}"] = pr
            print(f"  {g}: {a} vs {b} {pr}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "recap", "reband", "train", "read", "read_plain"],
    )
    p.add_argument(
        "--tag", default="", help="recap / reband: suffix for the data dir and arm"
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    assert bool({"recap", "reband"} & set(args.legs)) == bool(args.tag), (
        "recap / reband take --tag, alone"
    )
    GS = load_experiment("grid_small")
    rc, rows = run_config()
    tbl = table(GS)
    arm = ARM + (f"_{args.tag}" if args.tag else "")
    data_dir = OUT / DATA_RUN / ("data" + (f"_{args.tag}" if args.tag else ""))
    steps = STEPS_PER_ROW * len(rows)
    shares = {g.label: g.share for g in tbl}
    print(
        f"{arm}: {len(rows)} rows (hiragana + {EXTRA_ROWS}) cold × {STEPS_PER_ROW} = "
        f"{steps} steps on {SEED_ROWS_0921}; groups {shares} "
        f"(Σ {sum(shares.values()):g}); data {data_dir}",
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {"rows": len(rows), "steps": steps, "shares": shares, "arm": arm}
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = GS.data(rc, rows, args.workers, tbl)
        print(json.dumps(metrics["data"], ensure_ascii=False, indent=1), flush=True)
    if "recap" in args.legs:
        metrics["derive"] = GS.derive(OUT / DATA_RUN / "data", data_dir, None, True)
        print(json.dumps(metrics["derive"], ensure_ascii=False, indent=1), flush=True)
    if "reband" in args.legs:
        assert args.tag in BANDS, f"reband: --tag is one of {sorted(BANDS)}"
        metrics["reband"] = reband(
            OUT / DATA_RUN / "data_recap", data_dir, BANDS[args.tag]
        )
        print(json.dumps(metrics["reband"], ensure_ascii=False, indent=1), flush=True)
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
        metrics["read"] = GS.read(arm)
    if "read_plain" in args.legs:
        arms = {
            f"{ARM}_recap": OUT / "experiments" / f"{ARM}_recap",
            ARM: OUT / "experiments" / ARM,
            SRC_RUN: OUT / SRC_RUN,
        }
        if "reband" in args.legs:  # the reband arm first: its pairs lead
            others = {
                f"{ARM}_{t}": OUT / "experiments" / f"{ARM}_{t}"
                for t in BANDS
                if t != args.tag
                and (OUT / "experiments" / f"{ARM}_{t}" / "trained.pt").exists()
            }
            arms = {arm: OUT / "experiments" / arm} | arms | others
        metrics["read_plain"] = read_plain(arms)
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
