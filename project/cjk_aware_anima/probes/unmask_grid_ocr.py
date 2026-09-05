#!/usr/bin/env python3
"""Automated non-diegetic-text count for the arm-C unmask eval grids.

The grids (`output/tests/cjk_unmask_eval2/arm<X>_s<seed>/`) render the 8
text-free EN rows of `assets/unmask_eval_prompts.txt` (PNGs sort in row
order). Every CJK line PP-OCRv6 finds in rows 1–7 is text nobody asked for;
row 8 (`comic, 2koma`) legitimately carries SFX and is reported separately.
This replaces the eyeball tally of `reports/0903_rank_armC.md` with a number
that can be re-run on a new arm.

    .venv/bin/python project/cjk_aware_anima/probes/unmask_grid_ocr.py \
        --arms C2 C5 C3 C4 C9 P --out project/cjk_aware_anima/reports/unmask_grid_ocr.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EVAL = REPO / "output" / "tests" / "cjk_unmask_eval2"
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True, help="arm ids (C2 → armC2_s*)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1234])
    ap.add_argument("--eval_dir", type=Path, default=EVAL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min_score", type=float, default=0.5)
    ap.add_argument("--min_chars", type=int, default=2)
    opts = ap.parse_args()

    from anime_tools.ocr import load_ocr

    engine = load_ocr(
        device=opts.device,
        min_score=opts.min_score,
        min_chars=opts.min_chars,
        skip_en=True,
    )
    cells = []
    for arm in opts.arms:
        for seed in opts.seeds:
            d = opts.eval_dir / f"arm{arm}_s{seed}"
            pngs = sorted(d.glob("*.png"))
            if not pngs:
                print(f"missing: {d}")
                continue
            for row, png in enumerate(pngs, 1):
                lines = [
                    (ln.text, round(ln.score, 2))
                    for ln in engine.read(png)
                    if CJK_RE.search(ln.text)
                ]
                cells.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "row": row,
                        "cell": png.name,
                        "lines": lines,
                    }
                )
                if lines:
                    print(f"arm{arm:6s} s{seed:<5d} r{row}  {lines}")

    per_arm = defaultdict(
        lambda: {"cells_1_7": 0, "lines_1_7": 0, "comic_lines": 0, "n_seeds": set()}
    )
    per_arm_seed = defaultdict(
        lambda: {"cells_1_7": 0, "lines_1_7": 0, "comic_lines": 0}
    )
    for c in cells:
        a, s = per_arm[c["arm"]], per_arm_seed[(c["arm"], c["seed"])]
        a["n_seeds"].add(c["seed"])
        if c["row"] < 8:
            a["cells_1_7"] += bool(c["lines"])
            a["lines_1_7"] += len(c["lines"])
            s["cells_1_7"] += bool(c["lines"])
            s["lines_1_7"] += len(c["lines"])
        else:
            a["comic_lines"] += len(c["lines"])
            s["comic_lines"] += len(c["lines"])

    md = [
        "| arm | seeds | text cells r1–7 (of 7×seeds) | OCR lines r1–7 | comic-row lines | per seed (cells r1–7) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    table = {}
    for arm in opts.arms:
        a = per_arm.get(arm)
        if a is None:
            continue
        n = len(a["n_seeds"])
        per_seed = ", ".join(
            f"s{s}:{per_arm_seed[(arm, s)]['cells_1_7']}"
            for s in opts.seeds
            if (arm, s) in per_arm_seed
        )
        table[arm] = {
            "seeds": n,
            "cells_1_7": a["cells_1_7"],
            "lines_1_7": a["lines_1_7"],
            "comic_lines": a["comic_lines"],
            "per_seed_cells_1_7": {
                s: per_arm_seed[(arm, s)]["cells_1_7"]
                for s in opts.seeds
                if (arm, s) in per_arm_seed
            },
        }
        md.append(
            f"| {arm} | {n} | {a['cells_1_7']} / {7 * n} | {a['lines_1_7']} | {a['comic_lines']} | {per_seed} |"
        )
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(
        json.dumps({"table": table, "cells": cells}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    opts.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
