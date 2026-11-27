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
4. **Symbol table** — hearts ``♥`` / ``❤`` (+ U+FE0F) → ``♡``; wave dashes ``〜`` / ``〰``
   (U+301C) → ``~``; long dashes ``─`` (U+2500 box drawing) / ``—`` (U+2014) →
   ``―`` (U+2015, the JIS dash; 616 vs 412 vs 1 train rows), which fold 7 then
   writes ``-``. Curly
   double quotes ``“`` / ``”`` → ASCII ``"`` (``TARGET_NORM = 6``, 2026-11-17: one
   quote, three spellings across the manifests; swaprelax_v5 had 75 ``”`` rows
   to 3 ``"``). Circles ``◯`` / ``〇`` → ``○`` (``TARGET_NORM = 9``, 2026-11-20: the
   censor mark, the bullet and the kanji zero are one drawn shape, spelled all three
   ways across the labels); crosses ``☓`` / ``✕`` / ``✖`` / ``╳`` / ``✗`` / ``✘`` → ``×``
   (one label wrote ``☓☓☓と☓×☓``). Latin ``O`` / ``x`` / ``X`` stay. ``anime_tools``
   0.13.2's ``normalize_read`` folds the same (``SHAPE_FOLD``).
5. **Korean dashes** (``TARGET_NORM = 8``, 2026-11-20) — in a string holding Hangul,
   every dash (``―`` after fold 4, ``–``, ``‒``, ``‐``) → ASCII ``-``, run length kept.
   The KO labels spell one dash four ways (swaprelax_v5: ``-`` 289, ``―`` 161, ``—`` 38,
   ``–`` 4); the user's call. Japanese keeps ``―``.
6. **Dot runs** — ``[.・…‥]{2,}`` and a lone ``‥`` → one ``…``. Scoring keys
   only since ``TARGET_NORM = 7`` (2026-11-18): the training target keeps
   ``...`` / ``…`` / ``……`` as labelled (``…`` shielded from NFKC, which
   spells it ``...``); ``‥`` takes NFKC's ``..``. A ``・`` run is folded to ``…``, one per
   three dots (``・・・・・`` → ``……``, the user), in the target since ``TARGET_NORM = 11``
   (2026-11-26, :data:`MID_DOT_RUN_RE`): Manga109 spells a pause ``・・`` / ``・・・`` (2,343 train
   main labels against 21,818 with ``…``), and under norm 7 the reader lost where a
   repeated-``・`` run ends (COO speech runaways 11 → 94 at ``sfxpc_1118_2037``). ``.``
   runs stay: the KO reviews keep ``..`` / ``...`` apart from ``…``.
7. **Drawn-out lines** (``TARGET_NORM = 13``, 2026-11-27, the user) — outside a Hangul
   string, a run of ``ー`` / ``―`` / hyphens (any mix, any count, one included) → one ``ー``
   (``むうーーーーーーーー~~`` → ``むうー~``, ``もう―――`` → ``もうー``, ``―――?`` → ``ー?``,
   ``二年前──…`` → ``二年前ー…``); a ``~`` run → one ``~`` (``〰`` is ``~`` by fold 4). The
   hyphen stays, as one ``-``, in two places:

   * a run of hyphens alone that touches an ASCII letter or digit (``747-400D``, ``G-3!!``,
     ``=-2``);
   * a Hangul string, as fold 5 has it: its dashes and its ``ー`` / ``-`` runs are one ``-``,
     a lone ``ー`` stays (``하ー아``).

   A Chinese string folds as Japanese does (``——全部剪掉吧`` → ``ー全部剪掉吧``; 15 dash or
   hyphen runs in ``gelnote_zh``'s 2,127 rows): a string of ideographs alone does not say
   which language it is, and keeping ``-`` there split the Manga109 kanji-only strings 132
   ``-`` against 139 ``ー`` for one stroke. The user, 2026-11-27: the long mark is what the
   speech lines train a drawn-out sound as, and one spelling is worth more than the Chinese
   dash.

   In vertical lettering ``ー`` and ``―`` are one
   stroke, and the labels spell a drawn-out sound either way with any count: on
   ``mainsub_1126_2213``'s val reads 168 of 1,466 misses differed only in the count (140 of
   them short) and 198 read ``ー`` for a labelled ``―``. In the target and every key.
   ``TARGET_NORM = 12`` (one run only, ``mainsub_1127_0028``) wrote the run and the dash as
   ``-`` and kept a lone ``ー``: the reader then chose between two tokens for one stroke and
   read a lone ``ー`` on 276 of the 425 val mains whose target held ``-``
   (``mainsub_plan.md``).

The **training target** (:func:`training_target`, ``TARGET_NORM = 13``) keeps
the label's line breaks as ``\\n`` (one token, id 23; norm 4 collapsed them to a
space, so v4 answers one line) and keeps the two hearts apart: ``♥`` (and ``❤``) for
the filled one, ``♡`` for the outline. Norms 4–9 wrote every heart as ``♥``: in the
stock VL-1.6 tokenizer ``♥`` is one token and ``♡`` is three bytes, and
``TARGET_NORM = 3`` (``♡`` in the target) deleted the decoder's only cheap heart —
raw ``♥`` in sincos predictions went 144 → 0 and the gate 350 → 319 (plan_vl_respace
R5). Norm 10 trains with ``♡`` as a token of its own (``core/vocab_ext.py``). The
scoring key and the record form fold every heart to ``♡``, so a read is scored
heart-blind.
"""

from __future__ import annotations

import re
import unicodedata

ELLIPSIS = "…"
ELLIPSIS_RE = re.compile(r"[.．・…‥]{2,}|‥")
"""Every dot run (any spelling) and the two-dot leader ``‥`` on its own."""
MID_DOT_RUN_RE = re.compile(r"・{2,}")
"""A run of middle dots, the one dot run the training target folds (``TARGET_NORM = 11``,
:func:`_mid_dots`).
A lone ``・`` is a separator (``ジョン・スミス``) and stays."""

_DAKUTEN_FOLD = str.maketrans({"゛": "゙", "゜": "゚"})
"""Spacing → combining voiced marks, applied before NFKC (see module doc)."""

HEART_FOLD = str.maketrans(
    {
        "♥": "♡",
        "❤": "♡",
        "\ufe0f": "",  # emoji variation selector riding on ❤
        "〜": "~",
        "〰": "~",  # the wavy dash, 30 Manga109 speech labels (TARGET_NORM = 13)
        "─": "―",
        "—": "―",
        "“": '"',
        "”": '"',
        "◯": "○",  # every drawn circle → ○, every cross → × (TARGET_NORM = 9)
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
"""Fold 7 (``TARGET_NORM = 13``) outside a Hangul string: a run of drawn lines → one ``ー``
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
NFKC spells it ``...``. ``‥`` is left to NFKC (``..``): 99 Manga109 speech rows
from 4 books, no hand label."""


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
    """The training target (``TARGET_NORM = 13``): :func:`fold_glyphs` without
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
