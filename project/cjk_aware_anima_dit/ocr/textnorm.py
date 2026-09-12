"""One glyph fold for the OCR line — training target, scoring key, record form.

Until ``TARGET_NORM = 3`` (2026-09-12) the three surfaces disagreed: the trainer
taught the annotators' spelling verbatim (``・・・`` / ``...`` / ``…`` / ``……``
were four targets for one pause, ``♥`` competed with ``♡`` 41 : 296), while
``eval_manga109.exact_key`` and ``anime_tools.ocr.sfx.normalize_read`` folded
every variant away. The model was learning label noise the pipeline discards.

:func:`fold_glyphs` is the single rule. Callers pick the whitespace policy:
``normalize_target`` / ``space_key`` collapse runs to one ASCII space,
``exact_key`` deletes them.

Folds, in order:

1. **Spacing dakuten** ``゛`` / ``゜`` (U+309B / U+309C) → the combining marks
   (U+3099 / U+309A) *before* NFKC. NFKC decomposes the spacing forms to
   ``U+0020 + combining``, so a whitespace-collapsing normalizer turned ``あ゛っ``
   into ``あ ゙っ`` — a spurious space every ``TARGET_NORM = 2`` target carried
   (80 train rows; the sincos hand labels write the combining form).
2. **NFKC** — fullwidth ``！？．～`` → ASCII, ``･`` → ``・``, halfwidth kana
   composed, ``‼`` → ``!!``.
3. **Symbol table** — hearts ``♥`` / ``❤`` (+ U+FE0F) → ``♡``; wave dash ``〜``
   (U+301C) → ``~``; long dashes ``─`` (U+2500 box drawing) / ``—`` (U+2014) →
   ``―`` (U+2015, the JIS dash; 616 vs 412 vs 1 train rows). Dash *runs* keep
   their length — ``だから――――`` is a prolongation, like ``ーー``.
4. **Dot runs** — ``[.・…‥]{2,}`` and a lone ``‥`` → one ``…``.
"""

from __future__ import annotations

import re
import unicodedata

ELLIPSIS = "…"
ELLIPSIS_RE = re.compile(r"[.．・…‥]{2,}|‥")
"""Every dot run (any spelling) and the two-dot leader ``‥`` on its own."""

_DAKUTEN_FOLD = str.maketrans({"゛": "゙", "゜": "゚"})
"""Spacing → combining voiced marks, applied before NFKC (see module doc)."""

HEART_FOLD = str.maketrans(
    {
        "♥": "♡",
        "❤": "♡",
        "\ufe0f": "",  # emoji variation selector riding on ❤
        "〜": "~",
        "─": "―",
        "—": "―",
    }
)
"""Post-NFKC symbol table. Name kept from ``eval_manga109`` — it grew the wave
and dash folds but every consumer imported it under this name."""


def fold_glyphs(s: str) -> str:
    """The canonical spelling of ``s`` with whitespace untouched."""
    s = unicodedata.normalize("NFKC", s.translate(_DAKUTEN_FOLD))
    return ELLIPSIS_RE.sub(ELLIPSIS, s.translate(HEART_FOLD))


def normalize_target(s: str) -> str:
    """Training target / spaced key: folds + whitespace runs → one ASCII space."""
    return " ".join(fold_glyphs(s).split())


def exact_key(s: str) -> str:
    """Scoring key: folds + whitespace deleted."""
    return "".join(fold_glyphs(s).split())
