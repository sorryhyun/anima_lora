"""One glyph fold for the OCR line — training target, scoring key, record form.

:func:`fold_glyphs` is the single rule, so the reader never learns a spelling the scorer and
``anime_tools.ocr.reader`` discard. Callers pick the whitespace policy:
``normalize_target`` / ``space_key`` collapse runs to one ASCII space, ``exact_key`` deletes
them.

Folds, in order:

1. **Spacing dakuten** ``゛`` / ``゜`` (U+309B / U+309C) → the combining marks
   (U+3099 / U+309A) *before* NFKC. NFKC decomposes the spacing forms to
   ``U+0020 + combining``, so a whitespace-collapsing normalizer would turn ``あ゛っ``
   into ``あ ゙っ``.
2. **NFKC** — fullwidth ``！？．～`` → ASCII, ``･`` → ``・``, halfwidth kana
   composed, ``‼`` → ``!!``.
3. **Hangul jamo** — a lone conjoining jamo after NFKC (``ᄏ`` U+110F, NFKC's image of the
   typed ``ㅋ``) → the compatibility jamo (``ㅋ`` U+314B), :data:`JAMO_FOLD`.
4. **Symbol table** (:data:`HEART_FOLD`) — hearts ``♥`` / ``❤`` (+ U+FE0F) → ``♡``; wave
   dashes ``〜`` / ``〰`` → ``~``; long dashes ``─`` (U+2500) / ``—`` (U+2014) → ``―``
   (U+2015), which fold 7 then rewrites; curly double quotes ``“`` / ``”`` → ``"``; circles
   ``◯`` / ``〇`` → ``○`` and crosses ``☓`` / ``✕`` / ``✖`` / ``╳`` / ``✗`` / ``✘`` → ``×``
   (one drawn shape, one spelling). Latin ``O`` / ``x`` / ``X`` stay. ``anime_tools``'
   ``normalize_read`` folds the same (``SHAPE_FOLD``).
5. **Korean dashes** — in a string holding Hangul, every dash (``―`` after fold 4, ``–``,
   ``‒``, ``‐``) → ASCII ``-``, run length kept. Japanese keeps ``―`` for fold 7.
6. **Dot runs** — ``[.・…‥]{2,}`` and a lone ``‥`` → one ``…``, in the scoring keys only.
   The training target keeps ``...`` / ``…`` / ``……`` as labelled (``…`` shielded from
   NFKC, which spells it ``...``; ``‥`` takes NFKC's ``..``) and folds only a ``・`` run, to
   one ``…`` per three dots, at least one (``・・・・・`` → ``……``, :data:`MID_DOT_RUN_RE`).
   ``.`` runs stay: Korean keeps ``..`` / ``...`` apart from ``…``.
7. **Drawn-out lines** — outside a Hangul string, a run of ``ー`` / ``―`` / hyphens (any mix,
   any count, one included) → one ``ー`` (``むうーーーーーーーー~~`` → ``むうー~``,
   ``もう―――`` → ``もうー``, ``―――?`` → ``ー?``, ``二年前──…`` → ``二年前ー…``); a ``~``
   run → one ``~``. The hyphen stays, as one ``-``, in two places:

   * a run of hyphens alone that touches an ASCII letter or digit (``747-400D``, ``G-3!!``,
     ``=-2``);
   * a Hangul string, as fold 5 has it: its dashes and its ``ー`` / ``-`` runs are one ``-``,
     a lone ``ー`` stays (``하ー아``).

   A Chinese string folds as Japanese does (``——全部剪掉吧`` → ``ー全部剪掉吧``): a string of
   ideographs alone does not say which language it is. In vertical lettering ``ー`` and
   ``―`` are one stroke and the labels spell a drawn-out sound either way with any count,
   so the fold gives it one spelling, the long mark the speech lines train it as. In the
   target and every key.

The **training target** (:func:`training_target`) keeps the label's line breaks as ``\\n``
(one token, id 23) and keeps the two hearts apart: ``♥`` (and ``❤``) for the filled one,
``♡`` for the outline, a token of its own (``core/vocab_ext.py``). The scoring key and the
record form fold every heart to ``♡``, so a read is scored heart-blind.
"""

from __future__ import annotations

import re
import unicodedata

ELLIPSIS = "…"
ELLIPSIS_RE = re.compile(r"[.．・…‥]{2,}|‥")
"""Every dot run (any spelling) and the two-dot leader ``‥`` on its own."""
MID_DOT_RUN_RE = re.compile(r"・{2,}")
"""A run of middle dots, the one dot run the training target folds (:func:`_mid_dots`).
A lone ``・`` is a separator (``ジョン・スミス``) and stays."""

_DAKUTEN_FOLD = str.maketrans({"゛": "゙", "゜": "゚"})
"""Spacing → combining voiced marks, applied before NFKC (see module doc)."""

HEART_FOLD = str.maketrans(
    {
        "♥": "♡",
        "❤": "♡",
        "\ufe0f": "",  # emoji variation selector riding on ❤
        "〜": "~",
        "〰": "~",  # the wavy dash
        "─": "―",
        "—": "―",
        "“": '"',
        "”": '"',
        "◯": "○",  # every drawn circle → ○, every cross → ×
        "〇": "○",
        **{c: "×" for c in "☓✕✖╳✗✘"},
    }
)
"""Post-NFKC symbol table. Name kept from ``eval_manga109`` — it grew the wave,
dash and quote folds but every consumer imported it under this name."""


HANGUL_RE = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]")
KO_DASH_FOLD = str.maketrans({c: "-" for c in "―–‒‐"})
"""Fold 5: the dashes of a Hangul string → ``-`` (applied after :data:`HEART_FOLD`, so
``—`` / ``─`` arrive as ``―``)."""

KO_LINE_RUN_RE = re.compile(r"[ー-]{2,}")
"""Fold 7 in a Hangul string (after :data:`KO_DASH_FOLD`): a ``ー`` / ``-`` run → one ``-``."""

HYPHENS = frozenset("-–‒‐")
LINE_RUN_RE = re.compile(r"[ー―\-–‒‐]+")
WAVE_RUN_RE = re.compile(r"~{2,}")
"""Fold 7 outside a Hangul string: a run of drawn lines → one ``ー``
(:func:`_line_run`), and a ``~`` run → one ``~``."""


def _ascii_alnum(c: str) -> bool:
    return c.isascii() and c.isalnum()


def _line_run(m: re.Match) -> str:
    """One drawn-line run → ``ー``; a run of hyphens alone with an ASCII letter or digit
    on either side → ``-``."""
    if HYPHENS.issuperset(m[0]):
        s = m.string
        before = s[m.start() - 1] if m.start() else ""
        after = s[m.end()] if m.end() < len(s) else ""
        if _ascii_alnum(before) or _ascii_alnum(after):
            return "-"
    return "ー"


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
NFKC spells it ``...``. ``‥`` is left to NFKC (``..``)."""


_SHAPE_KEEP_HEARTS = {
    **HEART_FOLD,
    ord("♥"): ord("♥"),
    ord("❤"): ord("♥"),
}
""":data:`HEART_FOLD` with the filled heart kept (``❤`` → ``♥``)."""


def fold_glyphs(s: str, *, dots: bool = True, hearts: bool = True) -> str:
    """The canonical spelling of ``s`` with whitespace untouched. ``dots=False``
    skips fold 6: ``...``, ``・・・`` and ``…`` stay as written, ``‥`` → ``..``.
    ``hearts=False`` keeps ``♥`` apart from ``♡`` (``❤`` → ``♥``)."""
    if not dots:
        s = s.translate(_DOT_KEEP)
    s = unicodedata.normalize("NFKC", s.translate(_DAKUTEN_FOLD)).translate(JAMO_FOLD)
    s = s.translate(HEART_FOLD if hearts else _SHAPE_KEEP_HEARTS)
    if HANGUL_RE.search(s):
        s = KO_LINE_RUN_RE.sub("-", s.translate(KO_DASH_FOLD))
    else:
        s = LINE_RUN_RE.sub(_line_run, s)
    s = WAVE_RUN_RE.sub("~", s)
    if not dots:
        return s.translate(_DOT_RESTORE)
    return ELLIPSIS_RE.sub(ELLIPSIS, s)


def normalize_target(s: str) -> str:
    """Spaced key / record form: folds + whitespace runs → one ASCII space."""
    return " ".join(fold_glyphs(s).split())


def collapse_keep_lines(s: str) -> str:
    """Whitespace runs inside a line → one ASCII space; line breaks kept, a run
    of them (blank lines) one ``\\n``. U+3000 is a space here (NFKC)."""
    lines = (" ".join(line.split()) for line in s.replace("\r", "\n").split("\n"))
    return "\n".join(line for line in lines if line)


def training_target(s: str) -> str:
    """The training target: :func:`fold_glyphs` without
    the dot-run fold (``dots=False``) or the heart fold (``hearts=False``: ``♥`` and
    ``♡`` apart), a ``・`` run → ``…`` per three dots (:data:`MID_DOT_RUN_RE`), line breaks kept
    as ``\\n`` (:func:`collapse_keep_lines`). ``exact_key`` / ``space_key`` stay
    ``♡``-spelled, whitespace-blind and dot-folded, so a read with or without breaks,
    any dot spelling and either heart scores the same."""
    s = fold_glyphs(s, dots=False, hearts=False)
    return collapse_keep_lines(MID_DOT_RUN_RE.sub(_mid_dots, s))


def _mid_dots(m: re.Match) -> str:
    """A ``・`` run → one ``…`` per three dots, rounded, at least one (``・・`` → ``…``,
    ``・・・・・`` → ``……``): the run's length stays, the repeated token goes."""
    return ELLIPSIS * max(1, round(len(m[0]) / 3))


def exact_key(s: str) -> str:
    """Scoring key: folds + whitespace deleted."""
    return "".join(fold_glyphs(s).split())


_WS_RUN = re.compile(r"\s+")


def _gaps(s: str) -> tuple[str, list[str]]:
    """``s`` folded as :func:`exact_key` (its glyphs) and the gap before each glyph after
    the first: ``""`` none, ``" "`` a space, ``"\n"`` a run holding a line break."""
    s = fold_glyphs(s).strip()
    glyphs, gaps, prev = [], [], 0
    for m in _WS_RUN.finditer(s):
        part = s[prev : m.start()]
        glyphs.append(part)
        gaps += [""] * (len(part) - 1) + ["\n" if "\n" in m[0] else " "]
        prev = m.end()
    part = s[prev:]
    glyphs.append(part)
    gaps += [""] * max(0, len(part) - 1)
    return "".join(glyphs), gaps


def linebreak_blind(a: str, b: str) -> bool:
    """``a`` and ``b`` match as :func:`exact_key` and space the same, except where either
    has a line break: there a space and no space both match (a break in one line of the
    crop can stand for 띄어쓰기 or for none). Between :func:`exact_key` (whitespace-blind)
    and :func:`normalize_target` (a break counts as a space)."""
    ga, sa = _gaps(a)
    gb, sb = _gaps(b)
    return ga == gb and all(x == y or "\n" in (x, y) for x, y in zip(sa, sb))
