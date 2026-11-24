"""Scoring: the comparison keys every eval script reads.

* :func:`exact_key` — :func:`textnorm.exact_key` (NFKC, whitespace-blind, glyph folds);
* :func:`linebreak_blind` — :func:`textnorm.linebreak_blind` (``lb``: spacing counts, line
  breaks do not);
* :func:`norm` / :func:`sim` / :func:`is_runaway` — the A/B's keys, from the pilot
  (``finished/cjk_aware_anima/datasets/build_ocr_records.py``, bodies unchanged);
* :func:`normalize_ja` — the ``ー`` put-back, vendored (see its section);
* :func:`score` / :func:`summary` — a Manga109-manifest frame + reads → scored rows and the
  report table.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

from . import textnorm


def exact_key(s: str) -> str:
    """:func:`textnorm.exact_key` — NFKC + whitespace-blind; heart / wave / dash
    variants folded (the hand labels write ``♡`` and ``〜``, readers emit ``♥``
    / ``~`` for the same glyph) and every dot run one ``…``."""
    return textnorm.exact_key(s)


def linebreak_blind(a: str, b: str) -> bool:
    """:func:`textnorm.linebreak_blind` — :func:`exact_key` equality plus the same
    spacing, a line break on either side matching a space or none."""
    return textnorm.linebreak_blind(a, b)


# --------------------------------------------------------------------------- pilot keys

_STRIP_RE = re.compile(r"[\s。、．，,.・…‥「」『』!！?？~～〜❤♥♡♪☆★()（）\-ー—–|｜]")


def norm(s: str) -> str:
    """The A/B's comparison key: NFKC, then punctuation / symbols / spaces gone."""
    return _STRIP_RE.sub("", unicodedata.normalize("NFKC", s))


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    return SequenceMatcher(None, a, b).ratio() if a or b else 1.0


def is_runaway(text: str, *, ngram_repeats: int = 3, run: int = 8) -> bool:
    """The VL failure class PP does not have: ``ぉぉぉ…``×100, ``ふくっ``×100.

    A glyph run of ``run`` or more, or any 3-gram occurring ``ngram_repeats``
    or more times. ``ぱんぱん`` (one repeat) and ``おおおん`` pass.
    """
    t = "".join(text.split())
    if re.search(r"(.)\1{%d,}" % (run - 1), t):
        return True
    if len(t) >= 9:
        grams = defaultdict(int)
        for i in range(len(t) - 2):
            grams[t[i : i + 3]] += 1
        if max(grams.values()) >= ngram_repeats:
            return True
    return False


# --------------------------------------------------------------------------- ja normaliser
# Vendored verbatim from anime_tools.ocr._text @ 62f6fc3^ (the ``ー`` put-back that
# PP-OCRv6 needed). The package retired it with the CTC recognizer on 2026-09-07
# (anime-tools 62f6fc3); it stays here so every eval row scores on the same
# normaliser as O0–O3 — a re-based scorer would not be comparable.

CHOON = "ー"
"""The prolonged sound mark. PP-OCRv6's vocabulary is Chinese-first: set
vertically the mark is a bare stroke it reads as ``1`` / ``|`` / ``l``, set
horizontally a dash — so ``おちんぼの時間だぞ1`` and ``でるッリ一チ`` are one
misread each, not two words."""

_CHOON_VERTICAL = frozenset("1|lI１｜丨")
"""What a vertical ``ー`` comes back as."""

_CHOON_HORIZONTAL = frozenset("-—–ｰ")
"""What a horizontal ``ー`` comes back as (``ｰ`` is its own halfwidth form)."""

_NI_AS_EQUALS = frozenset("=＝")
"""What ``ニ`` comes back as between katakana — ``メ=ュー`` — two strokes the
vocabulary has as an equals sign."""


def is_kana(ch: str) -> bool:
    """Hiragana or katakana, ``ー`` included — the scripts a ``ー`` follows."""
    return "\u3041" <= ch <= "\u3096" or "\u30a1" <= ch <= "\u30fc"


def _is_katakana(ch: str) -> bool:
    return "\u30a1" <= ch <= "\u30fc"


_KANA_COUNTERS = frozenset("かつヶケ")
"""Counters written in kana — ``1か月`` / ``あと1つ`` — after which a digit is a
digit even though kana precedes it."""


def _counts_something(nxt: str) -> bool:
    """Whether the glyph after a would-be ``ー`` makes it a numeral instead:
    a kanji (``もう1回``), another digit or Latin, or a kana counter."""
    if not nxt:
        return False
    if "\u4e00" <= nxt <= "\u9fff" or nxt.isascii() and nxt.isalnum():
        return True
    return "\uff10" <= nxt <= "\uff19" or nxt in _KANA_COUNTERS


def normalize_ja(text: str, *, vertical: bool) -> str:
    """Put back the ``ー`` the recognizer spelled as a digit or a dash.

    Only after kana, and only when nothing counted follows: ``1`` after a kanji
    is a number, ``-`` after Latin a hyphen, and ``もう1回`` / ``1か月`` keep
    their digit (:func:`_counts_something`). A horizontal ``一`` (the kanji *one*) is a ``ー`` only *between* katakana —
    ``リ一チ`` — since ``一杯`` after a katakana word is a real ``一``. Vertical
    text never confuses the two (the strokes cross), so ``一`` is left alone
    there. The mark chains: ``ーー`` is written as such, and a fixed ``ー`` is
    kana for the next glyph. Likewise ``=`` between katakana is ``ニ``
    (``メ=ュー``), in either orientation.
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        prev = out[-1] if out else ""
        if prev and is_kana(prev):
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if (
                ch in _CHOON_HORIZONTAL or (vertical and ch in _CHOON_VERTICAL)
            ) and not _counts_something(nxt):
                out.append(CHOON)
                continue
            between_katakana = _is_katakana(prev) and bool(nxt) and _is_katakana(nxt)
            if not vertical and ch == "一" and between_katakana:
                out.append(CHOON)
                continue
            if ch in _NI_AS_EQUALS and between_katakana:
                out.append("ニ")
                continue
        out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- scoring


def score(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    rows = []
    for (_, r), p in zip(df.iterrows(), preds):
        pn = normalize_ja(p, vertical=(r.orient == "vertical"))
        rows.append(
            dict(
                pred=p,
                pred_norm=pn,
                exact=exact_key(pn) == exact_key(r.text),
                lb=linebreak_blind(pn, r.text),
                sim=sim(pn, r.text),
                runaway=is_runaway(p),
            )
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summary(scored: pd.DataFrame, name: str, split: str, wall: float) -> str:
    lines = [
        f"# OCR eval — `{name}` on Manga109-s `{split}` (official COO split ∩ Manga109-s)\n",
        f"Reader wall {wall:.0f} s for {len(scored)} crops ({len(scored) / max(wall, 1e-9):.1f} crops/s).\n",
        "| kind | n | exact | exact % | lb | sim (mean) | sim ≥ 0.8 | runaway |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, g in scored.groupby("kind"):
        lines.append(
            f"| {k} | {len(g)} | {int(g.exact.sum())} | {100 * g.exact.mean():.1f} | "
            f"{int(g.lb.sum())} | "
            f"{g.sim.mean():.3f} | {100 * (g.sim >= 0.8).mean():.1f} % | {int(g.runaway.sum())} |"
        )
    sfx = scored[scored.kind == "sfx"]
    if len(sfx):
        lines.append(
            "\n## SFX by orientation\n\n| orient | n | exact % | sim |\n|---|---|---|---|"
        )
        for o, g in sfx.groupby("orient"):
            lines.append(
                f"| {o} | {len(g)} | {100 * g.exact.mean():.1f} | {g.sim.mean():.3f} |"
            )
        lines.append(
            "\n## SFX by length\n\n| len | n | exact % | sim |\n|---|---|---|---|"
        )
        L = sfx.text.str.len().clip(upper=8)
        for n, g in sfx.groupby(L):
            lines.append(
                f"| {n}{'+' if n == 8 else ''} | {len(g)} | {100 * g.exact.mean():.1f} | {g.sim.mean():.3f} |"
            )
        lines.append(
            "\n## Worst 25 SFX (by sim)\n\n| book / page / id | gt | pred | sim |\n|---|---|---|---|"
        )
        for _, r in sfx.sort_values("sim").head(25).iterrows():
            pred = r.pred.replace("|", "\\|")[:40]
            lines.append(
                f"| {r.book} {r.page:03d} {r.id} | {r.text} | {pred} | {r.sim:.2f} |"
            )
    return "\n".join(lines) + "\n"
