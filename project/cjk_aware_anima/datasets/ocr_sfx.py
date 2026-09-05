"""Speech vs SFX for an OCR line — torch-free, text-only.

Manga text is either *spoken* (a balloon, a caption box, a whisper) or a
*sound effect* drawn onto the artwork (``ぱんぱん``, ``ばるん``, ``びくっ``).
The ``sentence`` caption format keeps them apart (``Japanese text reads as
"…". Japanese SFX reads as "…".``) so the address a line gets is the kind of
text it is. The split reads the string alone — no geometry — so it applies
to any reader's records (PP-OCRv6, PaddleOCR-VL Spotting, manga-ocr).

Rules, in order, on the kana core of the line (kanji / ASCII / punctuation
stripped):

1. any kanji, an empty core, or more than ``MAX_SFX_KANA`` kana → speech
   (onomatopoeia is short and kana-only);
2. a vowel / h-row / ``ん`` initial → speech: ``あっ…うっ…``, ``はぁ``,
   ``おおおん`` are a mouth, not an object;
3. a repeated unit (``ぱんぱん``, ``カリカリ``), a lexicon onset
   (``ちら``, ``きゅ``), a voiced / semi-voiced initial (``じゃぽ``,
   ``ブルン``, ``でくv``) or a sokuon initial → SFX;
4. katakana-only up to 4 kana (``ウズ``), or a sokuon-final up to 4 kana
   (``きゅっ``) → SFX; everything else → speech.

Known misses (sincos, 2026-09-05): a reader that turns ``ぱ`` into ``は``
(``はんぱん``) lands on rule 2; a voiced-initial 6-kana garble
(``がんばんがば``) lands on rule 3. Both are the reader's, not the rule's.
"""

from __future__ import annotations

import re

MAX_SFX_KANA = 6

_KANJI_RE = re.compile("[一-鿿]")
_KANA_RE = re.compile("[ぁ-んァ-ヶー]")
_KATAKANA_ONLY_RE = re.compile("[ァ-ヶー]+")

VOCAL_INITIAL = frozenset(
    "あいうえおんはひふへほぁぃぅぇぉアイウエオンハヒフヘホァィゥェォ"
)
VOICED_INITIAL = frozenset(
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
    "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴ"
)
SOKUON = frozenset("っッ")

# Onsets of common manga onomatopoeia whose initial is unvoiced (the voiced
# ones are caught by ``VOICED_INITIAL``). Matched as a prefix of the core.
SFX_LEXICON: tuple[str, ...] = (
    "ちら",
    "ちゅ",
    "ちゃぷ",
    "きゅ",
    "くちゅ",
    "くい",
    "くる",
    "こし",
    "ころ",
    "とく",
    "とろ",
    "たゆ",
    "ぷる",
    "ぴく",
    "ぴちゃ",
    "ぺろ",
    "ぺた",
    "ぱく",
    "ふに",
    "ふる",
    "ふわ",
    "むぎゅ",
    "むに",
    "もみ",
    "すり",
    "ずり",
    "しこ",
    "つん",
    "つぷ",
    "かり",
    "こく",
    "きらきら",
    "はむ",
    "れろ",
    "にゅ",
    "チラ",
    "チュ",
    "キュ",
    "クチュ",
    "トク",
    "プル",
    "ピク",
    "ペロ",
    "フワ",
    "スリ",
    "ツン",
    "カリ",
    "コク",
    "ハム",
    "レロ",
    "ニュ",
    "ムニ",
    "モミ",
)


def kana_core(text: str) -> str:
    """The kana of ``text`` in order, everything else dropped."""
    return "".join(_KANA_RE.findall(text))


def _repeated(core: str) -> bool:
    return any(
        core[:k] * 2 == core[: 2 * k]
        for k in (1, 2, 3)
        if core[:k] and 2 * k <= len(core)
    )


def line_kind(text: str, in_bubble: bool | None = None) -> str:
    """``"sfx"`` or ``"speech"`` for one OCR line (rules in the module doc).

    ``in_bubble`` is the geometric veto when the caller has a balloon mask
    (SAM3 ``speech bubble``, ``bubble_kind.py``): a line inside a balloon is
    speech whatever it says (``カリカリ``, ``バスト91``). ``False`` / ``None``
    fall through to the text rules — a line outside every balloon is *not*
    thereby SFX (narration, floating dialogue, UI chrome), and SAM3 misses
    balloons (34 of 97 sincos pages got one, 2026-09-05).
    """
    if in_bubble:
        return "speech"
    core = kana_core(text)
    if not core or _KANJI_RE.search(text) or len(core) > MAX_SFX_KANA:
        return "speech"
    if core[0] in VOCAL_INITIAL:
        return "speech"
    if (
        _repeated(core)
        or core.startswith(SFX_LEXICON)
        or core[0] in VOICED_INITIAL
        or core[0] in SOKUON
    ):
        return "sfx"
    if len(core) <= 4 and (_KATAKANA_ONLY_RE.fullmatch(core) or core[-1] in SOKUON):
        return "sfx"
    return "speech"


def split_lines(
    lines: list[str], in_bubble: list[bool | None] | None = None
) -> tuple[list[str], list[str]]:
    """``(speech, sfx)`` in the input (reading) order; ``in_bubble`` is the
    per-line balloon veto, when known."""
    flags = in_bubble if in_bubble is not None else [None] * len(lines)
    kinds = [line_kind(ln, b) for ln, b in zip(lines, flags, strict=True)]
    speech = [ln for ln, k in zip(lines, kinds, strict=True) if k == "speech"]
    sfx = [ln for ln, k in zip(lines, kinds, strict=True) if k == "sfx"]
    return speech, sfx
