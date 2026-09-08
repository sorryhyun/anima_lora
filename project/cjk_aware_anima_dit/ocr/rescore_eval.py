#!/usr/bin/env python3
"""Re-score stored eval predictions on the **current** `exact_key` (CPU, no model).

    python project/cjk_aware_anima_dit/ocr/rescore_eval.py            # the table
    …/rescore_eval.py --write                                         # + rewrite the reports

`exact_key` gained the ellipsis fold (every dot run → one `…`) in acd41d72,
2026-09-08 00:12 — the eval half of the O4e guard fix. Every
`output/ocr/eval/*.jsonl` written before that, and every
`reports/ocr_eval*.md` generated from one, therefore carries an `exact` on the
older key, which counts `・・・` ≠ `...` ≠ `…` as three different reads. The
effect is not a rounding difference: stock VL-1.6's COO speech row moves
**1623 → 2118 of 2559** (63.4 → 82.8 %), because a manga speech line pauses
and the two sides spell the pause differently.

So a row measured before that commit **cannot be compared to one measured
after**. This script re-derives `exact` for every stored jsonl from its own
`pred_norm` + `text` columns — the same expression `eval_manga109.score` uses
— so the whole table lands on one key without re-running a single model. `sim`
is untouched (it strips punctuation and symbols, so the fold never reached it)
and so is `runaway`.

`--write` regenerates each `reports/ocr_eval*.md` from the re-scored rows; the
default prints the table and leaves the record alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402


def rescore(path: Path) -> pd.DataFrame | None:
    rows = [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not rows or "pred_norm" not in rows[0]:
        return None
    df = pd.DataFrame(rows)
    df["exact_stored"] = df.exact
    df["exact"] = [
        ev.exact_key(pn) == ev.exact_key(t) for pn, t in zip(df.pred_norm, df.text)
    ]
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true", help="rewrite reports/ocr_eval*.md")
    ap.add_argument("--glob", default="*.jsonl")
    a = ap.parse_args()

    print(f"{'row':34s} {'kind':7s} {'n':>5s} {'stored':>7s} {'current':>8s} {'Δ':>5s}")
    for p in sorted(ev.OUT.glob(a.glob)):
        df = rescore(p)
        if df is None:
            continue
        for kind, g in df.groupby("kind"):
            d = int(g.exact.sum()) - int(g.exact_stored.sum())
            print(
                f"{p.name:34s} {kind:7s} {len(g):5d} {int(g.exact_stored.sum()):7d} "
                f"{int(g.exact.sum()):8d} {d:+5d}{'  re-based' if d else ''}"
            )
        if a.write:
            name = p.stem[: -len("_test")] if p.stem.endswith("_test") else p.stem
            split = "test" if p.stem.endswith("_test") else "sincos"
            out = ev.REPORTS / f"ocr_eval_{name}.md"
            if out.is_file():
                out.write_text(
                    ev.summary(df, name, split, float("nan")), encoding="utf-8"
                )
                print(f"  wrote {out}")


if __name__ == "__main__":
    main()
