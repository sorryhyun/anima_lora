"""One glyph fold for the OCR line — training target, scoring key, record form.

Until ``TARGET_NORM = 3`` (2026-09-12) the three surfaces disagreed: the trainer
taught the annotators' spelling verbatim (``・・・`` / ``...`` / ``…`` / ``……``
were four targets for one pause, ``♥`` competed with ``♡`` 41 : 296), while
``eval_manga109.exact_key`` and ``anime_tools.ocr.reader.normalize_read`` folded
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
3. **Hangul jamo** (``TARGET_NORM = 6``) — a lone conjoining jamo after NFKC
   (``ᄏ`` U+110F, NFKC's image of the typed ``ㅋ``) → the compatibility jamo
   (``ㅋ`` U+314B), :data:`JAMO_FOLD`. Norm 3–5 taught ``ᄏᄏ`` for ``ㅋㅋ``.
4. **Symbol table** — hearts ``♥`` / ``❤`` (+ U+FE0F) → ``♡``; wave dash ``〜``
   (U+301C) → ``~``; long dashes ``─`` (U+2500 box drawing) / ``—`` (U+2014) →
   ``―`` (U+2015, the JIS dash; 616 vs 412 vs 1 train rows). Dash *runs* keep
   their length — ``だから――――`` is a prolongation, like ``ーー``. Curly
   double quotes ``“`` / ``”`` → ASCII ``"`` (``TARGET_NORM = 6``, 2026-11-17: one
   quote, three spellings across the manifests; swaprelax_v5 had 75 ``”`` rows
   to 3 ``"``).
5. **Dot runs** — ``[.・…‥]{2,}`` and a lone ``‥`` → one ``…``. Scoring keys
   only since ``TARGET_NORM = 7`` (2026-11-18): the training target keeps
   ``...`` / ``・・・`` / ``…`` as labelled (``…`` shielded from NFKC, which
   spells it ``...``); ``‥`` takes NFKC's ``..``.

The **training target** (:func:`training_target`, ``TARGET_NORM = 7``) keeps
the label's line breaks as ``\\n`` (one token, id 23; norm 4 collapsed them to a
space, so v4 answers one line) and writes the heart as ``♥``: in the VL-1.6 tokenizer ``♥`` is
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
        "“": '"',
        "”": '"',
    }
)
"""Post-NFKC symbol table. Name kept from ``eval_manga109`` — it grew the wave,
dash and quote folds but every consumer imported it under this name."""


def _jamo_fold() -> dict[int, int]:
    out = {}
    for cp in range(0x1100, 0x1200):
        name = unicodedata.name(chr(cp), "")
        for part in ("CHOSEONG ", "JUNGSEONG ", "JONGSEONG "):
            if part in name:
                try:
                    out[cp] = ord(
                        unicodedata.lookup("HANGUL LETTER " + name.split(part)[1])
                    )
                except KeyError:
                    pass
    return out


JAMO_FOLD = _jamo_fold()
"""Conjoining jamo (U+1100–11FF) → compatibility jamo (U+3131–318E), by name.
NFKC sends the typed ``ㅋ`` (U+314B) to the conjoining ``ᄏ`` (U+110F); a jamo
left standing after NFKC (syllables are composed by then) goes back to the
typed form."""


_DOT_KEEP = str.maketrans({"…": ""})
_DOT_RESTORE = str.maketrans({"": "…"})
"""``dots=False``: ``…`` rides through NFKC as a private-use code point, since
NFKC spells it ``...``. ``‥`` is left to NFKC (``..``): 99 Manga109 speech rows
from 4 books, no hand label."""


def fold_glyphs(s: str, *, dots: bool = True) -> str:
    """The canonical spelling of ``s`` with whitespace untouched. ``dots=False``
    skips fold 5: ``...``, ``・・・`` and ``…`` stay as written, ``‥`` → ``..``."""
    if not dots:
        s = s.translate(_DOT_KEEP)
    s = unicodedata.normalize("NFKC", s.translate(_DAKUTEN_FOLD)).translate(JAMO_FOLD)
    s = s.translate(HEART_FOLD)
    if not dots:
        return s.translate(_DOT_RESTORE)
    return ELLIPSIS_RE.sub(ELLIPSIS, s)


TRAIN_HEART = "♥"
"""The heart the training target spells — the single-token one (R5b)."""

_TRAIN_HEART_FOLD = str.maketrans({"♡": TRAIN_HEART})


def normalize_target(s: str) -> str:
    """Spaced key / record form: folds + whitespace runs → one ASCII space."""
    return " ".join(fold_glyphs(s).split())


def collapse_keep_lines(s: str) -> str:
    """Whitespace runs inside a line → one ASCII space; line breaks kept, a run
    of them (blank lines) one ``\\n``. U+3000 is a space here (NFKC)."""
    lines = (" ".join(line.split()) for line in s.replace("\r", "\n").split("\n"))
    return "\n".join(line for line in lines if line)


def training_target(s: str) -> str:
    """The training target (``TARGET_NORM = 7``): :func:`fold_glyphs` without
    the dot-run fold (``dots=False``), line breaks kept as ``\\n``
    (:func:`collapse_keep_lines`), the heart as :data:`TRAIN_HEART`.
    ``exact_key`` / ``space_key`` stay ``♡``-spelled, whitespace-blind and
    dot-folded, so a read with or without breaks, and with any dot spelling,
    scores the same."""
    return collapse_keep_lines(fold_glyphs(s, dots=False)).translate(_TRAIN_HEART_FOLD)


def exact_key(s: str) -> str:
    """Scoring key: folds + whitespace deleted."""
    return "".join(fold_glyphs(s).split())
