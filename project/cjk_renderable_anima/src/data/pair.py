"""ΔFM siblings (plan_synth2): the Latin reference string drawn beside a
composite item by the same fit. One upper-case letter per JA glyph, never a
letter of the target's romaji (す never gets S / U — the row must not see
the pairing as a mapping), from a small pool per (scene, glyph count) so
the reference captions stay few (each caption is ≈ 1.3 MB of text cache).
"""

from __future__ import annotations

import random
import string

# Hepburn letters per kana, hiragana and katakana alike; only the *letters*
# matter (the exclusion set), so ambiguities (si / shi, du / zu) are unioned.
_ROWS = {
    "あいうえお": ["a", "i", "u", "e", "o"],
    "かきくけこ": ["ka", "ki", "ku", "ke", "ko"],
    "さしすせそ": ["sa", "shi", "su", "se", "so"],
    "たちつてと": ["ta", "chi", "tsu", "te", "to"],
    "なにぬねの": ["na", "ni", "nu", "ne", "no"],
    "はひふへほ": ["ha", "hi", "fu", "he", "ho"],
    "まみむめも": ["ma", "mi", "mu", "me", "mo"],
    "やゆよ": ["ya", "yu", "yo"],
    "らりるれろ": ["ra", "ri", "ru", "re", "ro"],
    "わをん": ["wa", "wo", "n"],
    "がぎぐげご": ["ga", "gi", "gu", "ge", "go"],
    "ざじずぜぞ": ["za", "ji", "zu", "ze", "zo"],
    "だぢづでど": ["da", "dji", "dzu", "de", "do"],
    "ばびぶべぼ": ["ba", "bi", "bu", "be", "bo"],
    "ぱぴぷぺぽ": ["pa", "pi", "pu", "pe", "po"],
    "ぁぃぅぇぉ": ["a", "i", "u", "e", "o"],
    "ゃゅょ": ["ya", "yu", "yo"],
    "っ": ["tsu"],
    "ゎ": ["wa"],
    "ゔ": ["vu"],
}
ROMAJI: dict[str, str] = {}
for kana, roma in _ROWS.items():
    for ch, r in zip(kana, roma):
        ROMAJI[ch] = r
        ROMAJI[chr(ord(ch) + 0x60)] = r  # the katakana sibling
ROMAJI["ー"] = ""


def romaji_letters(text: str) -> set[str]:
    """Upper-case letters of the Hepburn reading of every kana in ``text``
    (kanji and punctuation contribute nothing)."""
    return {c.upper() for ch in text for c in ROMAJI.get(ch, "")}


class RefPool:
    """``--pair_ref_pool`` reference strings per (scene, glyph count), filled
    lazily; ``draw`` returns one whose letters avoid the target's romaji,
    adding a fresh one to the pool only when none of the pool fits."""

    def __init__(self, size: int, rng: random.Random):
        self.size, self.rng = size, rng
        self.pool: dict[tuple[int, int], list[str]] = {}

    def _fresh(self, n: int, avoid: set[str]) -> str:
        letters = [c for c in string.ascii_uppercase if c not in avoid]
        return "".join(self.rng.choice(letters) for _ in range(n))

    def draw(self, scene_i: int, text: str) -> str:
        avoid = romaji_letters(text)
        key = (scene_i, len(text))
        pool = self.pool.setdefault(key, [])
        if len(pool) < self.size:
            pool.append(self._fresh(len(text), avoid))
            return pool[-1]
        ok = [s for s in pool if not (set(s) & avoid)]
        if ok:
            return self.rng.choice(ok)
        s = self._fresh(len(text), avoid)
        pool.append(s)
        return s

    def n_strings(self) -> int:
        return sum(len(v) for v in self.pool.values())
