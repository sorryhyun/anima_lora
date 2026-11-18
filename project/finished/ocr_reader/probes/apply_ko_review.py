#!/usr/bin/env python3
"""Apply a ``ko_label_review.py`` export (TSV: source, id, status, orig, new; ``\\n`` /
``\\t`` / ``\\\\`` escaped) to the label manifests: ``edit`` replaces the text, ``drop``
removes the row, ``ok`` (or an empty status: a row with only an SFX mark) changes nothing.
The optional ``sfx`` column (the page's SFX checkbox) goes to ``assets/sfx_marks.tsv``. Edited text is stored through
``textnorm.collapse_keep_lines`` (no spaces at a line's edges, no blank lines). A row whose current text no longer equals ``orig``
is skipped and reported.

    python project/finished/ocr_reader/probes/apply_ko_review.py <review.tsv> [--dry_run]

Manifests per source: gelnote_ko / gelnote_zh → ``manifest_pseudo_gelnote_{ko,zh}`` +
``manifest_sfxp_kozh``;
swaprelax_v5 → ``manifest_pseudo_swaprelax_v5`` + ``manifest_sfxp_kozh``;
hito_v4_ja / hito_v4_ko → ``manifest_pseudo_hito_v4``;
linebreak_suspicious / dots_suspicious → ``manifest_sfxp_speech`` (``orig`` is a proposal,
so ``ok`` writes it too; a row matches when its text equals ``orig`` blind to whitespace and
dot glyphs). Each manifest
is copied to ``<name>.pre_review.parquet`` once, before its first change.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402
from build_sfx_prefix import ellipsis_fold  # noqa: E402
from textnorm import collapse_keep_lines  # noqa: E402

TARGETS = {
    "gelnote_ko": ["pseudo_gelnote_ko", "sfxp_kozh"],
    "swaprelax_v5": ["pseudo_swaprelax_v5", "sfxp_kozh"],
    "gelnote_zh": ["pseudo_gelnote_zh", "sfxp_kozh"],
    "hito_v4_ja": ["pseudo_hito_v4"],
    "hito_v4_ko": ["pseudo_hito_v4"],
    "linebreak_suspicious": ["sfxp_speech"],
    "dots_suspicious": ["sfxp_speech"],
}
APPLY_OK = {"linebreak_suspicious", "dots_suspicious"}
"""Sources whose page text is a proposal, not the manifest's text: ``ok`` applies it."""


def flat_key(t: str) -> str:
    """A proposal source's match key: the text without whitespace or dot glyphs (the
    proposal changes breaks and dot spelling only; a dot-only label keys to "")."""
    return re.sub(r"[\s.．・･…‥]", "", t)


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
        act = rv.status.isin(["edit", "drop"]) | (
            rv.source.isin(APPLY_OK) & (rv.status == "ok")
        )
        sub = rv[rv.source.isin(srcs) & act]
        if name == "sfxp_kozh":
            key = df.source.astype(str) + ":" + df.id.astype(str)
        elif name == "sfxp_speech":  # the review id is <book>_<page:03d>_<text id>
            key = [
                f"{b}_{int(p):03d}_{t}" if s == "sfxp_speech" else None
                for b, p, t, s in zip(df.book, df.page, df.id, df.source)
            ]
        else:
            key = df.id.astype(str)
        pos = {k: i for i, k in enumerate(key) if k is not None}
        edited, dropped, stale = 0, [], []
        for r in sub.itertuples():
            k = f"{r.source}:{r.id}" if name == "sfxp_kozh" else r.id
            i = pos.get(k)
            proposal = r.source in APPLY_OK
            cur = None if i is None else df.text.iat[i]
            if cur is None or (
                flat_key(cur) != flat_key(r.orig) if proposal else cur != r.orig
            ):
                stale.append(r.id)
                continue
            if r.status == "drop":
                dropped.append(df.index[i])
                continue
            new = r.orig if r.status == "ok" else r.new
            if proposal:
                new = ellipsis_fold(new)
            if new != cur:
                df.iat[i, df.columns.get_loc("text")] = new
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
    if "sfx" in rv.columns:
        write_marks(rv, a.dry_run)


def write_marks(rv: pd.DataFrame, dry_run: bool) -> None:
    """The page's SFX checkboxes → ``assets/sfx_marks.tsv`` (source, id, text). The
    export carries every mark in the browser, so each exported source's marks are
    replaced whole; other sources' marks stay. Training does not read the file."""
    path = Path(__file__).resolve().parents[1] / "assets" / "sfx_marks.tsv"
    old = (
        pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        if path.exists()
        else pd.DataFrame(columns=["source", "id", "text"])
    )
    srcs = set(rv.source)
    new = rv[rv.sfx == "1"].assign(
        text=lambda x: x.new.where(x.status == "edit", x.orig).str.replace("\n", "\\n")
    )[["source", "id", "text"]]
    out = pd.concat([old[~old.source.isin(srcs)], new], ignore_index=True)
    print(
        f"[sfx_marks] {len(new)} marked in {sorted(srcs)}; file {len(old)} → {len(out)} rows"
    )
    if not dry_run:
        out.to_csv(path, sep="\t", index=False, quoting=3)


if __name__ == "__main__":
    main()
