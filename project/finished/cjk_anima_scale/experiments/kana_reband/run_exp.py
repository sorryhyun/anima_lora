#!/usr/bin/env python
"""kana_reband — ``retrain_kana``'s rows again, cold, with every item at σ 0.75–0.93

findings.md puts glyph identity at σ 0.9–0.7 on the trajectory; the seed's
kana run trained two thirds of its items below 0.7 (``builder.TABLE``:
b0709 / b0507 / b0305, 5 800 items each). This arm is ``retrain_kana`` with
the band alone changed: the same 17 400 items (images stay in
``retrain_kana/data/img``), every one at ``--band`` (default 0.75–0.93), the
174 kana rows cold from the pack rows on the old seed's rows, the kana run's
trainer and budget (135 / row = 23 490 steps, μ 0, lr 1e-3 cosine, seed 0 —
``check_trainer`` holds them to ``retrain_kana/train_record.json``).

``--rows hira`` (the arm that ran): the 81 hiragana rows alone, on the 6 521
items whose every glyph is hiragana — katakana, punctuation and the items
that carry them are left out, so no untrained row sits in a caption. The
filter keeps 42 of the 2 314 multi-cell grids (a grid's cells mix scripts):
the lone tier is ``scene_single`` and 1×1 grids here.

Legs:
- ``data`` (CPU) → ``OUT/run1002_kana_reband/data[_<tag>]``: the records with
  ``band`` replaced (latents and the TE cache are built here on first train);
- ``train`` (GPU) → ``OUT/experiments/kana_reband_cold[_<tag>]``;
- ``read`` (GPU): ``retrain_read``'s grid (4 prompts × 2 seeds, routed) —
  the kana run's 13 ``read`` words on ``en`` and its 14 singles on ``swap`` —
  paired per render against ``retrain_kana``'s own reads of record
  (``retrain_kana/native_r4_{en,swap}/``, never re-rendered), and one sheet
  per key (``retrain_kana`` | this arm). ``--rows hira`` reads the hiragana
  keys only (9 words, 8 singles); C2's eight words also carry ``p1_mix``'s
  cached tally (36 hiragana rows, cold, the seed's bands).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/kana_reband/run_exp.py \\
      --label hira_r0 --rows hira --legs data train read"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.builder import tier_of  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS_0921, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "kana_reband"
SRC_RUN = "retrain_kana"
SRC = OUT / SRC_RUN / "data"
BAND = (0.75, 0.93)
STEPS_PER_ROW = 135  # retrain_kana's: 90 × the mix's 1.5 (retrain_experiments § 5)
# the kana run's record ↔ the trainer's constants: the arm differs by the band
RECORD = {
    "lr_rows": "LR",
    "init_anchor": "INIT_ANCHOR",
    "free_residual": "FREE_RESIDUAL",
    "batch": "BATCH",
    "lr_decay": "LR_DECAY",
    "lr_warmup_ratio": "WARMUP_RATIO",
    "box_share": "BOX_SHARE",
    "box_share_cap": "BOX_SHARE_CAP",
    "box_share_glyphs": "BOX_SHARE_GLYPHS",
    "seed": "SEED",
}


def check_trainer(steps_per_row: int) -> dict:
    from cjk_scale import train as T

    rec = json.loads((OUT / SRC_RUN / "train_record.json").read_text("utf-8"))
    for k, const in RECORD.items():
        assert rec[k] == getattr(T, const), (k, rec[k], getattr(T, const))
    assert rec["grid_box"] == int(T.GRID_BOX), rec["grid_box"]
    assert rec["cold"] and Path(rec["context"]) == SEED_ROWS_0921, rec["context"]
    return {
        "src_steps_per_row": rec["steps_per_row"],
        "src_steps": rec["train_steps"],
        "same_budget": rec["steps_per_row"] == steps_per_row,
    }


def is_hira(text: str) -> bool:
    return all("ぁ" <= c <= "ゖ" for c in text.replace(" ", ""))


def source(rows: str) -> tuple[list, list]:
    """``retrain_kana``'s records and vocabs, or their hiragana part."""
    recs = [
        json.loads(ln) for ln in (SRC / "train.jsonl").read_text("utf-8").splitlines()
    ]
    vocabs = json.loads((SRC / "vocabs.json").read_text("utf-8"))
    if rows == "hira":
        recs = [r for r in recs if is_hira(r["text"])]
        vocabs = [v for v in vocabs if is_hira(v)]
        drawn = {c for r in recs for c in r["text"].replace(" ", "")}
        assert drawn == set(vocabs), set(vocabs) ^ drawn
    return recs, vocabs


def data(dst: Path, band: tuple, rows: str) -> dict:
    recs, vocabs = source(rows)
    old = Counter((tier_of(r), tuple(r["band"])) for r in recs)
    for r in recs:
        r["band"] = list(band)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
    )
    (dst / "vocabs.json").write_text(
        json.dumps(vocabs, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    shutil.copy2(SRC / "eval.json", dst / "eval.json")
    bj = json.loads((SRC / "build.json").read_text("utf-8"))
    (dst / "build.json").write_text(
        json.dumps(
            bj | {"band": list(band), "reband_from": str(SRC), "rows": rows},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return {
        "from": str(SRC),
        "items": len(recs),
        "rows": len(vocabs),
        "tiers": dict(Counter(tier_of(r) for r in recs)),
        "old_bands": {f"{g} {a}-{b}": n for (g, (a, b)), n in sorted(old.items())},
        "band": list(band),
        "glyph_route": bj.get("glyph_route"),
    }


def read(name: str, rows: str) -> dict:
    """The arm on ``retrain_read``'s grid, paired against ``retrain_kana``'s
    reads of record on the same prompts × seeds."""
    from cjk_scale import reads as R
    from cjk_scale.config import load_run
    from common.readers import contact_sheet
    from eval.stage import sheet_row

    RR = load_experiment("retrain_read")  # pins the old seed: the kana run's floor
    rc = load_run(SRC_RUN)  # its data dir and `read` words; the arm is `arm`
    arm = OUT / "experiments" / name
    assert (arm / "trained.pt").exists(), f"{arm}: not trained yet"
    words, singles = list(rc.read), list(RR.HIRA + RR.KATA)
    if rows == "hira":
        words, singles = [w for w in words if is_hira(w)], list(RR.HIRA)
    out: dict = {"words": words, "singles": singles}
    hits, files = {}, {}
    for clause, keys in (("en", words), ("swap", singles)):
        ref = OUT / SRC_RUN / f"native_r4_{clause}" / "native_reads.json"
        assert ref.is_file(), f"{ref}: the kana run's read of record is missing"
        files[clause] = {SRC_RUN: ref, name: RR.ensure(rc, arm, keys, clause)}
        hits[clause] = {
            a: RR.grid_hits(p, keys, clause) for a, p in files[clause].items()
        }
        for a, h in hits[clause].items():
            print(f"===== {a} · {clause}", flush=True)
            out.setdefault(a, {})[clause] = R.tally(h)
    c2 = [w for w in words if w in RR.C2_WORDS]
    print("===== p1_mix · C2 words (cache)", flush=True)
    out["p1_mix"] = {
        "en": R.tally(
            RR.grid_hits(
                RR.P2.EXP / "p1_mix" / RR.FLOOR_CACHE["en"] / "native_reads.json",
                c2,
                "en",
            )
        )
    }
    groups = {
        "words_hira": ("en", [w for w in words if is_hira(w)]),
        "words_kata": ("en", [w for w in words if not is_hira(w)]),
        "singles_hira": ("swap", [c for c in singles if c in RR.HIRA]),
        "singles_kata": ("swap", [c for c in singles if c in RR.KATA]),
    }
    groups = {g: v for g, v in groups.items() if v[1]}
    out["paired_vs_" + SRC_RUN] = {}
    for g, (clause, keys) in groups.items():
        a, b = (
            {k: v for k, v in hits[clause][x].items() if k[0] in keys}
            for x in (name, SRC_RUN)
        )
        out["paired_vs_" + SRC_RUN][g] = pr = R.paired(a, b)
        print(f"  {g}: {name} vs {SRC_RUN} {pr}", flush=True)
    sheets = arm / "sheets_r4"
    sheets.mkdir(exist_ok=True)
    for clause, keys in (("en", words), ("swap", singles)):
        by = {
            a: {
                (m["text"], m["pi"], m["seed"]): m
                for m in json.loads(p.read_text("utf-8"))
                if m["clause"] == clause
            }
            for a, p in files[clause].items()
        }
        for t in keys:
            rows = [
                sheet_row(by[a][(t, pi, s)], f"{a} p{pi} s{s}")
                for pi in range(RR.N_PROMPTS)
                for s in range(RR.SEEDS)
                for a in (SRC_RUN, name)
            ]
            contact_sheet(rows, sheets / f"sheet_{t}_{clause}.png", cols=2)
    out["sheets"] = str(sheets)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "read"]
    )
    p.add_argument("--band", type=float, nargs=2, default=list(BAND))
    p.add_argument("--steps_per_row", type=int, default=STEPS_PER_ROW)
    p.add_argument(
        "--rows",
        default="all",
        choices=["all", "hira"],
        help="hira: the hiragana rows on the all-hiragana items (docstring)",
    )
    p.add_argument("--tag", default="", help="suffix for the data dir and the arm")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    sfx = "".join(
        f"_{x}" for x in (args.rows if args.rows != "all" else "", args.tag) if x
    )
    dst = OUT / f"run1002_{NAME}" / f"data{sfx}"
    name = f"{NAME}_cold{sfx}"

    from cjk_scale.config import load_run

    recs, vocabs = source(args.rows)
    rc = dataclasses.replace(
        load_run(SRC_RUN), name=name, vocabs=["chars:" + "".join(vocabs)]
    )
    n_items = len(recs)
    del recs
    steps = args.steps_per_row * len(vocabs)
    budget = check_trainer(args.steps_per_row)
    print(
        f"{name}: {n_items} {SRC_RUN} items → band {args.band}, {len(vocabs)} rows × "
        f"{args.steps_per_row} = {steps} steps (≈ {steps * 4 / n_items:.1f} epochs at "
        f"batch 4), cold on {SEED_ROWS_0921}; {budget}",
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {
        "src": str(SRC),
        "band": list(args.band),
        "steps_per_row": args.steps_per_row,
        "steps": steps,
        "rows": len(vocabs),
        "row_set": args.rows,
        **budget,
    }
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = data(dst, tuple(args.band), args.rows)
        print(json.dumps(metrics["data"], ensure_ascii=False), flush=True)
    if "train" in args.legs:
        from cjk_scale import train as T

        T.train(
            rc,
            data=dst,
            out=OUT / "experiments" / name,
            cold=True,
            steps_per_row=args.steps_per_row,
            context=SEED_ROWS_0921,
        )
    if "read" in args.legs:
        metrics["read"] = read(name, args.rows)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(dst), str(OUT / "experiments" / name)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
