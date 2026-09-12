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

The **training target** (:func:`training_target`, ``TARGET_NORM = 4``) is the
same fold with the heart written as ``♥``: in the VL-1.6 tokenizer ``♥`` is
one token and ``♡`` is three bytes, and ``TARGET_NORM = 3`` (``♡`` in the
target) deleted the decoder's only cheap heart — raw ``♥`` in sincos
predictions went 144 → 0 and the gate 350 → 319 (plan_vl_respace R5). The
scoring key and the record form keep ``♡``, so the direction is free
downstream.
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


TRAIN_HEART = "♥"
"""The heart the training target spells — the single-token one (R5b)."""

_TRAIN_HEART_FOLD = str.maketrans({"♡": TRAIN_HEART})


def normalize_target(s: str) -> str:
    """Spaced key / record form: folds + whitespace runs → one ASCII space."""
    return " ".join(fold_glyphs(s).split())


def training_target(s: str) -> str:
    """:func:`normalize_target` with the heart as :data:`TRAIN_HEART` — the
    training target only; ``exact_key`` / ``space_key`` stay ``♡``-spelled."""
    return normalize_target(s).translate(_TRAIN_HEART_FOLD)


def exact_key(s: str) -> str:
    """Scoring key: folds + whitespace deleted."""
    return "".join(fold_glyphs(s).split())
