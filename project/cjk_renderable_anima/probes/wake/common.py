"""Paths, inventories, prompt templates, text metrics, shapes and output dirs."""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
# the OCR readers (pseudo_label) stayed in the frozen line
sys.path.insert(0, str(REPO / "project" / "cjk_aware_anima_dit" / "ocr"))

OUT = REPO / "output" / "wake_probe"
CORPUS_TRAIN = REPO / "post_image_dataset" / "render" / "ja" / "resized"
CORPUS_HELD = REPO / "post_image_dataset" / "render" / "ja" / "heldout"

# ----------------------------------------------------------------------------
# inventories

HIRA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
KATA = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
KANA = HIRA + KATA
# P0b (2026-09-14): voiced / handakuten / small kana — each is its own Qwen piece
# and pack row (checked: 68/68); added to the singles inventory by --kana_ext and
# scored as group ``single_ext``, apart from the basic 92
KANA_EXT_HIRA = "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉっゃゅょ"
KANA_EXT_KATA = "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォッャュョ"
KANA_EXT = KANA_EXT_HIRA + KANA_EXT_KATA
# small kana are not a singles concept (P0b: a lone ゃ draws full-size — no
# size reference); the S line shows them only inside words / phrases
KANA_SMALL = "ぁぃぅぇぉっゃゅょァィゥェォッャュョ"
KANA_RE = re.compile(r"^[ぁ-ゟァ-ヿー〜っ・…！？!?]+$")
CJK_RE = re.compile(r"[぀-ヿ぀-ゟ㐀-䶿一-鿿]")
KANJI_RE = re.compile(r"[一-鿿]")
# a word piece: kana / kanji / long vowel only (no ・ … punctuation pieces)
WORD_RE = re.compile(r"^[ぁ-ゟァ-ヺー一-鿿]+$")

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

# ----------------------------------------------------------------------------
# prompt templates and eval groups

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
    "word",
    "word_held",
    "line",
    "flip",
    "str3",
    "phrase_held",
    "combo",
    "corpus",
    "en",
)

NATIVE_PROMPTS = (
    REPO / "project" / "cjk_aware_anima" / "assets" / "unmask_eval_prompts.txt"
)
NATIVE_CLAUSES = {
    # the trained clause shape, hung off a scene prompt instead of the template
    "en": '{p}, japanese text. Japanese text reads as "{k}".',
    # the user's phrasing: a Japanese-language clause (its own words route to
    # untrained pack rows; only the kana row carries the delta)
    "ja": "{p}. ひらがなの「{k}」という文字がある。",
    # the EN-reference swap (2026-09-15): the *same* caption as the EN
    # reference render with only the quoted word replaced — the pair differs
    # in nothing but the ext row (enref uses "hi"; see wake/enref.py)
    "swap": '{p}, english text. English text reads as "{k}".',
}

# ----------------------------------------------------------------------------
# text metrics


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(
        ch
        for ch in s
        if not ch.isspace() and ch not in "「」『』、。,.!?！？…・〜~\"'()（）"
    )


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    r = norm(ref)
    if not r:
        return 1.0
    return min(1.0, lev(norm(hyp), r) / len(r))


# ----------------------------------------------------------------------------
# canvas shapes


def wh(size) -> tuple[int, int]:
    """Canvas ``(W, H)`` from an int side or a ``(W, H)`` pair."""
    if isinstance(size, int):
        return size, size
    W, H = size
    return int(W), int(H)


def parse_shape(tok: str) -> tuple[int, int]:
    """``'512'`` → (512, 512); ``'384x512'`` → (384, 512) as (W, H)."""
    tok = tok.strip().lower()
    if "x" in tok:
        W, H = tok.split("x")
        return int(W), int(H)
    return int(tok), int(tok)


def parse_shapes(spec: str) -> list[tuple[int, int, float]]:
    """``--shapes`` → ``[(W, H, weight)]``. ``'384,448,512:2,384x512'`` draws
    512² twice as often as each other entry. Sides must be multiples of 16
    (VAE 8× and a 2-patch), so every entry is one static token family."""
    out = []
    for tok in spec.split(","):
        if not tok.strip():
            continue
        shp, _, w = tok.partition(":")
        W, H = parse_shape(shp)
        assert W % 16 == 0 and H % 16 == 0, (
            f"shape {tok}: sides must be multiples of 16"
        )
        out.append((W, H, float(w) if w else 1.0))
    return out


# ----------------------------------------------------------------------------
# output dirs (read OUT at call time so a caller can redirect it)


def data_dir(a) -> Path:
    return OUT / ("data" + (f"_{a.data_tag}" if a.data_tag else ""))


def arm_dir(a) -> Path:
    return OUT / (
        a.arm
        + (f"_{a.data_tag}" if a.data_tag else "")
        + (f"_{a.arm_tag}" if a.arm_tag else "")
    )
