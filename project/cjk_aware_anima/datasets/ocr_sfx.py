"""Speech vs SFX for an OCR line — now :mod:`anime_tools.captions.ocr_sfx`.

The rule (``line_kind`` / ``split_lines``) and the per-sound dedupe
(``sfx_key`` / ``sfx_groups`` / ``dedupe_sfx``) were promoted into the package
on 2026-09-07 — Export's ``--combine_ocr`` builds the ``Japanese text reads as
"…". Japanese SFX reads as "…".`` clauses from them — and this module is the
research tree's alias so ``from ocr_sfx import …`` keeps working in
``cache_te_ext.py`` / ``build_ocr_records.py`` / the probes. The rule's history
(why a vowel initial is a mouth, the sincos misses) lives in the package
module's docstring.
"""

from __future__ import annotations

from anime_tools.captions.ocr_sfx import (  # noqa: F401
    MAX_SFX_KANA,
    SFX_LEXICON,
    SOKUON,
    VOCAL_INITIAL,
    VOICED_INITIAL,
    _KANA_RE,
    _KANJI_RE,
    _KATAKANA_ONLY_RE,
    _KEY_DROP,
    _repeated,
    dedupe_sfx,
    kana_core,
    line_kind,
    sfx_groups,
    sfx_key,
    split_lines,
)

__all__ = [
    "MAX_SFX_KANA",
    "SFX_LEXICON",
    "SOKUON",
    "VOCAL_INITIAL",
    "VOICED_INITIAL",
    "dedupe_sfx",
    "kana_core",
    "line_kind",
    "sfx_groups",
    "sfx_key",
    "split_lines",
]
