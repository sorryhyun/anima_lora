#!/usr/bin/env python3
"""Apply a ``ko_label_review.py`` export (TSV: source, id, status, orig, new; ``\\n`` /
``\\t`` / ``\\\\`` escaped) to the label manifests: ``edit`` replaces the text, ``drop``
removes the row, ``ok`` (or an empty status: a row with only an SFX mark) changes nothing.
The optional ``sfx`` column (the page's SFX checkbox) goes to ``assets/sfx_marks.tsv``, the
optional ``sub`` column (the page's ``+ sub`` box, the ``OCR2:`` text of the same crop; ``-`` =
removed) to ``assets/sub_labels.tsv``. Edited text is stored through
``textnorm.collapse_keep_lines`` (no spaces at a line's edges, no blank lines). A row whose current text no longer equals ``orig``
is skipped and reported.

    python project/finished/ocr_reader/probes/apply_ko_review.py <review.tsv> [--dry_run]

Manifests per source: gelnote_ko / gelnote_zh → ``manifest_pseudo_gelnote_{ko,zh}`` +
``manifest_sfxp_kozh``;
swaprelax_v5 → ``manifest_pseudo_swaprelax_v5`` + ``manifest_sfxp_kozh``;
hito_v4_ja / hito_v4_ko → ``manifest_pseudo_hito_v4``;
linebreak_suspicious / dots_suspicious / odd_dots / furigana → ``manifest_sfxp_speech`` (``orig`` is a proposal,
so ``ok`` writes it too; a row matches when its text equals ``orig`` blind to whitespace and
dot glyphs). Each manifest
is copied to ``<name>.pre_review.parquet`` once, before its first change.

mainsub_sfx_wide / mainsub_det / mainsub_speech (``--mainsub`` sources) → ``assets/m109_mainsub.tsv``
(source, id, status, main, sub; ``\\n`` escaped), one row per crop, replaced whole on each
export: ``ok`` / ``edit`` (or a sub typed without either) → status ``ok`` with the main box
(``orig`` = the proposal for ``ok``) and the sub (the exported one, ``-`` = none); ``drop`` →
status ``drop``. The manifests are not touched; training does not read the file yet.

doujin_sfx → ``assets/sfx_labels_sincos_597.tsv`` (the doujin gate's labels, by ``row``):
``edit`` writes ``text_hand`` (a line break as a space), ``edit`` / ``ok`` set ``status`` to ``checked``, ``drop`` sets
it to ``unchecked`` (``eval_sfx.py`` skips those rows; the row stays in the file). The file is
copied to ``sfx_labels_sincos_597.pre_review.tsv`` once, before its first change.
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
    "odd_dots": ["sfxp_speech"],
    "furigana": ["sfxp_speech"],
}
DOUJIN = Path(__file__).resolve().parents[1] / "assets" / "sfx_labels_sincos_597.tsv"
MAINSUB = {"mainsub_sfx_wide", "mainsub_det", "mainsub_speech"}
APPLY_OK = {"linebreak_suspicious", "dots_suspicious", "odd_dots"}
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
    if (rv.source == "doujin_sfx").any():
        apply_doujin(rv[rv.source == "doujin_sfx"], a.dry_run)
    if rv.source.isin(MAINSUB).any():
        write_mainsub(rv[rv.source.isin(MAINSUB)], a.dry_run)
        rv = rv[~rv.source.isin(MAINSUB)]
    for name in sorted(
        {m for s in rv.source.unique() if s != "doujin_sfx" for m in TARGETS[s]}
    ):
        path = d / f"manifest_{name}.parquet"
        df = pd.read_parquet(path)
        srcs = [s for s in rv.source.unique() if name in TARGETS.get(s, ())]
        act = rv.status.isin(["edit", "drop"]) | (
            rv.source.isin(APPLY_OK) & (rv.status == "ok")
        )
        sub = rv[rv.source.isin(srcs) & act]
        if name == "sfxp_kozh":
            key = df.source.astype(str) + ":" + df.id.astype(str)
        elif name == "sfxp_speech":  # the review id is <book>_<page:03d>_<text id>
            key = [
                f"{b}_{int(p):03d}_{t}" if s in ("sfxp_speech", "m109_lines") else None
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
    if "sub" in rv.columns:
        write_subs(rv, a.dry_run)


def apply_doujin(rv: pd.DataFrame, dry_run: bool) -> None:
    """``doujin_sfx`` rows → the doujin gate's label TSV (see the module docstring)."""
    df = pd.read_csv(DOUJIN, sep="\t", dtype=str, keep_default_na=False)
    pos = {int(r): i for i, r in enumerate(df.row)}
    edited, checked, dropped, stale = 0, 0, 0, []
    for r in rv[rv.status.isin(["ok", "edit", "drop"])].itertuples():
        i = pos.get(int(r.id))
        if i is None or df.text_hand.iat[i] != r.orig:
            stale.append(r.id)
            continue
        st = "unchecked" if r.status == "drop" else "checked"
        # the gate file keeps one row per line: a break is stored as a space (the
        # file's block separator; eval_sfx's exact is whitespace-blind)
        new = r.new.replace("\n", " ")
        if r.status == "edit" and new != r.orig:
            df.iat[i, df.columns.get_loc("text_hand")] = new
            edited += 1
        if df.status.iat[i] != st:
            df.iat[i, df.columns.get_loc("status")] = st
            dropped += st == "unchecked"
            checked += st == "checked"
    print(
        f"[doujin_sfx] edit {edited}, → checked {checked}, → unchecked {dropped}, "
        f"stale {len(stale)} {stale[:5]}"
    )
    if dry_run or not (edited or checked or dropped):
        return
    bak = DOUJIN.with_name(DOUJIN.stem + ".pre_review.tsv")
    if not bak.exists():
        shutil.copy2(DOUJIN, bak)
    df.to_csv(DOUJIN, sep="\t", index=False, lineterminator="\r\n")  # the file is CRLF


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


def write_mainsub(rv: pd.DataFrame, dry_run: bool) -> None:
    """The Manga109 main / sub rows → ``assets/m109_mainsub.tsv`` (see the module
    docstring); rows the export leaves out stay."""
    path = Path(__file__).resolve().parents[1] / "assets" / "m109_mainsub.tsv"
    cols = ["source", "id", "status", "main", "sub"]
    old = (
        pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        if path.exists()
        else pd.DataFrame(columns=cols)
    )
    sub_col = rv["sub"] if "sub" in rv.columns else pd.Series("", index=rv.index)
    rv = rv.assign(sub=sub_col)[(rv.status != "") | (sub_col != "")]
    esc = lambda t: t.replace("\n", "\\n")  # noqa: E731
    new = pd.DataFrame(
        dict(
            source=rv.source,
            id=rv.id,
            status=["drop" if s == "drop" else "ok" for s in rv.status],
            main=[
                "" if s == "drop" else esc(n if s == "edit" else o)
                for s, n, o in zip(rv.status, rv.new, rv.orig)
            ],
            sub=[
                ""
                if s == "drop" or b in ("", "-")
                else esc(collapse_keep_lines(unescape(b)))
                for s, b in zip(rv.status, rv["sub"])
            ],
        )
    )
    key = lambda df: df.source + ":" + df.id  # noqa: E731
    out = pd.concat([old[~key(old).isin(set(key(new)))], new], ignore_index=True)
    n_sub = int((new["sub"] != "").sum())
    print(
        f"[m109_mainsub] {len(new)} rows (drop {int((new.status == 'drop').sum())}, "
        f"with sub {n_sub}, empty main {int(((new.main == '') & (new.status == 'ok')).sum())}); "
        f"file {len(old)} → {len(out)} rows"
    )
    if not dry_run:
        out.to_csv(path, sep="\t", index=False, quoting=3)


def write_subs(rv: pd.DataFrame, dry_run: bool) -> None:
    """The page's sub boxes → ``assets/sub_labels.tsv`` (source, id, text, sub; ``\\n``
    escaped): the ``OCR2:`` label of the row's crop, one item per line in reading order.
    A row exported with ``-`` loses its sub; rows the export leaves out stay. Training
    does not read the file yet."""
    path = Path(__file__).resolve().parents[1] / "assets" / "sub_labels.tsv"
    old = (
        pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        if path.exists()
        else pd.DataFrame(columns=["source", "id", "text", "sub"])
    )
    touched = rv[rv["sub"] != ""].assign(sub=lambda x: x["sub"].map(unescape))
    key = lambda df: df.source + ":" + df.id  # noqa: E731
    keep = old[~key(old).isin(set(key(touched)))]
    new = touched[touched["sub"] != "-"].assign(
        text=lambda x: x.new.where(x.status == "edit", x.orig).str.replace("\n", "\\n"),
        sub=lambda x: x["sub"].map(collapse_keep_lines).str.replace("\n", "\\n"),
    )[["source", "id", "text", "sub"]]
    out = pd.concat([keep, new], ignore_index=True)
    print(
        f"[sub_labels] {len(new)} set, {len(touched) - len(new)} removed; "
        f"file {len(old)} → {len(out)} rows"
    )
    if not dry_run:
        out.to_csv(path, sep="\t", index=False, quoting=3)


if __name__ == "__main__":
    main()
