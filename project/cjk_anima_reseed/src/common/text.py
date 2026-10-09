"""Kana inventories, the script regexes, and the text metrics (norm / lev / cer)."""

from __future__ import annotations

import re
import unicodedata

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
