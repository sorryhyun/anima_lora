#!/usr/bin/env python3
"""Re-score stored eval predictions on the **current** `exact_key` (CPU, no model).

    python project/ocr_reader/ocr/rescore_eval.py            # the table
    …/rescore_eval.py --write                                         # + rewrite the reports

`exact_key` folds every dot run into one `…` since acd41d72 (2026-09-08). An
`output/ocr/eval/*.jsonl` written before that, and a `reports/eval/ocr_eval*.md`
generated from one, carries an `exact` on the older key, which counts `・・・`,
`...` and `…` as different reads, and **does not compare with a later row**. This
script re-derives `exact` for every stored jsonl from its own `pred_norm` + `text`
columns (the expression `scoring.score` uses), so the whole table lands on one key
without re-running a model. `sim` is untouched (it strips punctuation and symbols,
so the fold never reached it) and so is `runaway`.

`--write` regenerates each `reports/eval/ocr_eval*.md` from the re-scored rows; the
default prints the table and leaves the record alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from project.ocr_reader.core.paths import EVAL_OUT, EVAL_REPORTS
from project.ocr_reader.core.scoring import exact_key, summary


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
        exact_key(pn) == exact_key(t) for pn, t in zip(df.pred_norm, df.text)
    ]
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--write", action="store_true", help="rewrite reports/eval/ocr_eval*.md"
    )
    ap.add_argument("--glob", default="*.jsonl")
    a = ap.parse_args()

    print(f"{'row':34s} {'kind':7s} {'n':>5s} {'stored':>7s} {'current':>8s} {'Δ':>5s}")
    for p in sorted(EVAL_OUT.glob(a.glob)):
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
            out = EVAL_REPORTS / f"ocr_eval_{name}.md"
            if out.is_file():
                out.write_text(summary(df, name, split, float("nan")), encoding="utf-8")
                print(f"  wrote {out}")


if __name__ == "__main__":
    main()
