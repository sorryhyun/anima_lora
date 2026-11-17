#!/usr/bin/env python3
"""Automatic fixes for the KO ``OCR:`` labels from the v6 reads of ``ko_v6_read.py``.

* **Line breaks**: a label with no line break whose characters (whitespace removed) equal
  the read's, where the read has breaks, takes the read's breaks; the label's own
  characters and spacing stay (a space at a break goes). Written as ``edit`` rows of a
  review TSV for ``apply_ko_review.py``.
* **Edge text** (reported, not fixed): the read lies inside the label with label text left
  over at its start or end — text the label holds that the 4 px crop may not show (the
  swaprelax labels were audited on crops cut at 12 % of the long side).

    python project/finished/ocr_reader/probes/ko_label_fix.py <reads.parquet> --tsv <out.tsv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
from textnorm import collapse_keep_lines  # noqa: E402


def bare(t: str) -> str:
    return "".join(t.split())


def transplant(label: str, read: str) -> str:
    """``label`` with a line break after each non-space count where ``read`` breaks."""
    cuts, n = set(), 0
    for ch in read:
        if ch == "\n":
            cuts.add(n)
        elif not ch.isspace():
            n += 1
    out, n = [], 0
    for ch in label:
        if ch.isspace():
            out.append(" ")
            continue
        if n in cuts and n:
            out.append("\n")
            cuts.discard(n)
        out.append(ch)
        n += 1
    return collapse_keep_lines("".join(out).replace(" \n", "\n").replace("\n ", "\n"))


def edge(label: str, read: str) -> str | None:
    """Label text outside the read (``'<head>|<tail>'``) when the read sits inside it."""
    b, r = bare(label), bare(read)
    if not r or b == r or len(r) < 2:
        return None
    i = b.find(r)
    if i < 0:
        return None
    return f"{b[:i]}|{b[i + len(r) :]}"


def esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reads", type=Path)
    ap.add_argument(
        "--tsv", type=Path, required=True, help="line-break edits (review TSV)"
    )
    a = ap.parse_args()
    df = pd.read_parquet(a.reads)
    rows, n_cand = [], 0
    for r in df.itertuples():
        if "\n" in r.text or "\n" not in r.read:
            continue
        n_cand += 1
        if bare(r.text) != bare(r.read):
            continue
        new = transplant(r.text, r.read)
        if new != r.text:
            rows.append((r.source, r.id, "edit", esc(r.text), esc(new)))
    with open(a.tsv, "w", encoding="utf-8") as f:
        f.write("source\tid\tstatus\torig\tnew\n")
        for x in rows:
            f.write("\t".join(x) + "\n")
    no_lf = df[~df.text.str.contains("\n")]
    print(
        f"[lf] labels with no break: {len(no_lf)}; read has breaks: {n_cand}; "
        f"same characters → fixed: {len(rows)} → {a.tsv}"
    )
    print(
        df.assign(fixed=df.id.isin({x[1] for x in rows}))
        .groupby("source")
        .fixed.sum()
        .to_string()
    )
    df["edge"] = [edge(t, r) for t, r in zip(df.text, df.read)]
    e = df[df.edge.notna()]
    print(f"[edge] label longer than the read at its ends: {len(e)}")
    print(e.groupby("source").size().to_string())
    for r in e.head(12).itertuples():
        print(f"  {r.id}  {r.edge!r}  label {r.text[:40]!r}")
    out = a.reads.with_name(a.reads.stem + "_flags.parquet")
    df[["source", "id", "read", "conf", "edge"]].to_parquet(out, index=False)
    print(f"[flags] → {out}")


if __name__ == "__main__":
    main()
