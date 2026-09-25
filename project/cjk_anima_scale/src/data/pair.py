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


class JaRefPool:
    """``--pair_ref ja`` (idea.md, counterfactual input): the sibling is a
    *confusable* JA unit of the same glyph count from the run's own units —
    for strings a permutation (½) or a one-glyph swap within the glyph's
    script (½), for singles another single of the same script; another unit
    of the same length is the fallback. Spaces keep their positions (the
    compositor asserts equal length). Same ``draw`` / ``n_strings`` surface
    as ``RefPool``."""

    def __init__(self, units: list[str], rng: random.Random):
        from collections import defaultdict

        from common.text import KANA, KANJI_RE

        self.rng = rng
        self.by_len: dict[int, list[str]] = defaultdict(list)
        glyphs: dict[str, set[str]] = defaultdict(set)
        for u in units:
            self.by_len[len(u)].append(u)
            for ch in u:
                if ch != " ":
                    glyphs[self._script(ch, KANA, KANJI_RE)].add(ch)
        self.glyphs = {k: sorted(v) for k, v in glyphs.items()}
        self._kana, self._kanji_re = KANA, KANJI_RE
        self.seen: set[str] = set()

    @staticmethod
    def _script(ch: str, kana, kanji_re) -> str:
        if ch in kana:
            return "kana"
        if kanji_re.match(ch):
            return "kanji"
        return "other"

    def _swap_one(self, chars: list[str]) -> str | None:
        pos = [i for i, c in enumerate(chars) if c != " "]
        self.rng.shuffle(pos)
        for i in pos:
            pool = [
                g
                for g in self.glyphs.get(
                    self._script(chars[i], self._kana, self._kanji_re), []
                )
                if g != chars[i]
            ]
            if pool:
                out = list(chars)
                out[i] = self.rng.choice(pool)
                return "".join(out)
        return None

    def _permute(self, chars: list[str]) -> str | None:
        pos = [i for i, c in enumerate(chars) if c != " "]
        if len({chars[i] for i in pos}) < 2:
            return None
        for _ in range(8):
            vals = [chars[i] for i in pos]
            self.rng.shuffle(vals)
            out = list(chars)
            for i, v in zip(pos, vals):
                out[i] = v
            s = "".join(out)
            if s != "".join(chars):
                return s
        return None

    def draw(self, scene_i: int, text: str) -> str:
        chars = list(text)
        s = None
        if len(chars) >= 2 and self.rng.random() < 0.5:
            s = self._permute(chars)
        if s is None:
            s = self._swap_one(chars)
        if s is None:
            others = [u for u in self.by_len.get(len(text), []) if u != text]
            s = self.rng.choice(others) if others else None
        if s is None:
            # a lone glyph of an unseen script: any kana of the same count
            s = "".join(
                c if c == " " else self.rng.choice([k for k in self._kana if k != c])
                for c in chars
            )
        assert len(s) == len(text) and s != text, (text, s)
        self.seen.add(s)
        return s

    def n_strings(self) -> int:
        return len(self.seen)


def en_frame(caption: str) -> str:
    """A sibling caption moved back under the **EN frame** the scene was
    rendered with — the inverse of ``synth.scene_caption``'s swap: the
    ``japanese text`` tag → ``english text``, ``Japanese text / SFX reads
    as`` → ``English …``; frames that name no language (``She is saying
    "…"``) keep their words. ΔFM ``--pair_ref_frame en`` (2026-09-17): under
    the JA frame the base's sibling residual carries its JA pseudo-text
    prior, the paired loss cancels it, and the row never learns to suppress
    it. The tag keeps its sorted position (not re-sorted)."""
    return (
        caption.replace(", japanese text,", ", english text,")
        .replace(", japanese text.", ", english text.")
        .replace("Japanese text reads as", "English text reads as")
        .replace("Japanese SFX reads as", "English SFX reads as")
    )
