"""Label text edits shared by the manifest builders and the review page."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DOT_RUN = re.compile(r"\.{3,}|．{3,}|・{3,}|･{3,}")


def ellipsis_fold(t: str) -> str:
    """A run of one dot glyph (``.`` ``．`` ``・`` ``･``) whose length is a multiple of
    three → length / 3 × ``…``; in a label with text, a run of four → one ``…`` (the
    user's review edits, 11-18: ``そりゃ・・・・`` → ``そりゃ…``; dot-only labels are
    spelled per printed column by ``linebreak_auto.py dots``). A run a line break splits
    (``火薬・・\n・・・・・・？``) stays as written. Idempotent."""
    has_text = bool(re.sub(r"[\s.．・･…‥]", "", t))

    def sub(m: re.Match) -> str:
        r, a, b = m[0], m.start(), m.end()
        split = (a >= 2 and t[a - 1] == "\n" and t[a - 2] == r[0]) or (
            b + 1 < len(t) and t[b] == "\n" and t[b + 1] == r[0]
        )
        if split:
            return r
        if len(r) == 4 and has_text:
            return "…"
        return r if len(r) % 3 else "…" * (len(r) // 3)

    return DOT_RUN.sub(sub, t)


def line_broken(raw: str, lengths: list[int]) -> str | None:
    """``raw`` cut at the annotated line lengths, each line stripped; ``None`` when
    the lengths do not sum to the text."""
    flat = raw.replace("\n", "")
    if sum(lengths) != len(flat):
        return None
    out, i = [], 0
    for n in lengths:
        out.append(flat[i : i + n].strip())
        i += n
    return "\n".join(s for s in out if s)


def agent_subs(root: Path | None) -> dict[str, tuple[str, str, str]]:
    """``furigana_find.py batches``' agent outputs under ``root`` (``**/out*.tsv``: n, id,
    sub with ``\\n`` breaks, verdict, note) → id → (sub, verdict, note)."""
    out: dict[str, tuple[str, str, str]] = {}
    if root is None:
        return out
    for f in sorted(root.glob("**/out*.tsv")):
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        df = df.reindex(columns=["n", "id", "sub", "verdict", "note"]).fillna("")
        for r in df.itertuples():
            sub = r.sub.replace("\\n", "\n").strip()
            out[r.id] = (sub, r.verdict.strip(), r.note.strip())
    return out
