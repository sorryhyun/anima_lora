"""Prompt templates, eval groups, the native scene prompts and clauses."""

from __future__ import annotations

from .paths import REPO

EN_WORDS = [
    "HELLO",
    "STOP",
    "YES",
    "NO WAY",
    "WAIT",
    "SORRY",
    "WHAT",
    "OK",
    "RUN",
    "HELP",
    "GO",
    "HEY",
]

TPL_BUBBLE = 'manga, speech bubble, japanese text. Japanese text reads as "{}".'
TPL_PLAIN = (
    'japanese text, white background, simple background. Japanese text reads as "{}".'
)
TPL_EN = 'manga, speech bubble, english text. English text reads as "{}".'
# S-line composite: the scene's own tag caption (``english text`` →
# ``japanese text``) + the trained clause shape
TPL_SCENE_JA = '{tags}. Japanese text reads as "{text}".'

# report / sheet order; ``word`` / ``word_held`` / ``line`` are the 2026-09-14
# word-address groups (``--words``)
EVAL_GROUPS = (
    "single",
    "single_held",
    "single_ext",
    "single_kanji",
    "single_extra",
    "word",
    "word_held",
    "line",
    "flip",
    "str3",
    "phrase",
    "phrase_held",
    "short",
    "short_held",
    "combo",
    "corpus",
    "en",
)

NATIVE_PROMPTS = (
    REPO / "project" / "cjk_aware_anima" / "assets" / "unmask_eval_prompts.txt"
)
# the user's own target captions (ComfyUI, 2026-09-17), rendered verbatim by
# ``--stage target``
TARGET_PROMPTS = (
    REPO / "project" / "cjk_renderable_anima" / "assets" / "target_prompts.txt"
)
NATIVE_CLAUSES = {
    # the trained clause shape, hung off a scene prompt instead of the template
    "en": '{p}, japanese text. Japanese text reads as "{k}".',
    # the user's phrasing: a Japanese-language clause (its own words route to
    # untrained pack rows; only the kana row carries the delta)
    "ja": "{p}. ひらがなの「{k}」という文字がある。",
    # the EN-reference swap (2026-09-15): the *same* caption as the EN
    # reference render with only the quoted word replaced — the pair differs
    # in nothing but the ext row (enref uses "hi"; see eval/enref.py)
    "swap": '{p}, english text. English text reads as "{k}".',
}
