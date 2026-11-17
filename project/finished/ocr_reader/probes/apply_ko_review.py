#!/usr/bin/env python3
"""Apply a ``ko_label_review.py`` export (TSV: source, id, status, orig, new; ``\\n`` /
``\\t`` / ``\\\\`` escaped) to the label manifests: ``edit`` replaces the text, ``drop``
removes the row, ``ok`` changes nothing. Edited text is stored through
``textnorm.collapse_keep_lines`` (no spaces at a line's edges, no blank lines). A row whose current text no longer equals ``orig``
is skipped and reported.

    python project/finished/ocr_reader/probes/apply_ko_review.py <review.tsv> [--dry_run]

Manifests per source: gelnote_ko / gelnote_zh → ``manifest_pseudo_gelnote_{ko,zh}`` +
``manifest_sfxp_kozh``;
swaprelax_v5 → ``manifest_pseudo_swaprelax_v5`` + ``manifest_sfxp_kozh``. Each manifest
is copied to ``<name>.pre_review.parquet`` once, before its first change.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402
from textnorm import collapse_keep_lines  # noqa: E402

TARGETS = {
    "gelnote_ko": ["pseudo_gelnote_ko", "sfxp_kozh"],
    "swaprelax_v5": ["pseudo_swaprelax_v5", "sfxp_kozh"],
    "gelnote_zh": ["pseudo_gelnote_zh", "sfxp_kozh"],
}


def unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"n": "\n", "t": "\t", "\\": "\\"}.get(s[i + 1], s[i : i + 2]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    rv = pd.read_csv(a.tsv, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rv["orig"] = rv.orig.map(unescape)
    # an edit is stored as training reads it: spaces around a line break and blank
    # lines go (``textnorm.collapse_keep_lines``)
    rv["new"] = rv.new.map(unescape).map(collapse_keep_lines)
    d = m109.derived_root()
    for name in sorted({m for s in rv.source.unique() for m in TARGETS[s]}):
        path = d / f"manifest_{name}.parquet"
        df = pd.read_parquet(path)
        srcs = [s for s in rv.source.unique() if name in TARGETS[s]]
        sub = rv[rv.source.isin(srcs) & (rv.status != "ok")]
        key = (
            df.source.astype(str) + ":" + df.id.astype(str)
            if name == "sfxp_kozh"
            else df.id.astype(str)
        )
        pos = {k: i for i, k in enumerate(key)}
        edited, dropped, stale = 0, [], []
        for r in sub.itertuples():
            k = f"{r.source}:{r.id}" if name == "sfxp_kozh" else r.id
            i = pos.get(k)
            if i is None or df.text.iat[i] != r.orig:
                stale.append(r.id)
                continue
            if r.status == "drop":
                dropped.append(df.index[i])
            else:
                df.iat[i, df.columns.get_loc("text")] = r.new
                edited += 1
        df = df.drop(index=dropped)
        print(
            f"[{name}] edit {edited}, drop {len(dropped)}, stale {len(stale)} {stale[:5]}"
        )
        if a.dry_run or not (edited or dropped):
            continue
        bak = path.with_suffix(".pre_review.parquet")
        if not bak.exists():
            shutil.copy2(path, bak)
        df.to_parquet(path, index=False)


if __name__ == "__main__":
    main()
