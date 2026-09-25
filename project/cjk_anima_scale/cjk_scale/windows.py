"""windows — the vocab band law as code: (kind, px, layout) → the σ band an
item trains in.

The rows below are ``band_experiment_results.md`` (§ 2 ceiling table, § 3
training reads) and ``design.md`` § 2 (the stage table), one row per cell
that has a read, each with its provenance. They are a lookup, not a formula:
a formula off the EN ceiling table cannot reproduce the reads (it puts a
128 px grid cell above 0.9, which C.2 says is dead at 48 px and step1_0921
trained at 0.7–0.9). A new read that changes a row changes it here, with
its report.

Kinds, by the vocab's Qwen tokens and glyphs (``vocab_kind`` / ``kind_of``):

    single   one token, one glyph            あ 日 ！
    piece    one token, two or more glyphs   って 先生 ・・・  (one ext row carries the string)
    multi    two or more tokens              あっ (host + small row), a short line, a sentence

The probe's ``--t_band_multi`` / ``_remap_band`` "multi" meant ≥ 2 glyphs
(piece + multi here); the band reads are on pieces (micro_cf_0922) and on
lines (step 2, A.2 strings), and the rows say which.

Units: ``px`` is the data stage's ink-stat glyph px, √(box area / glyphs) —
the number every "as built" px in the reports is. ``layout``: ``scene`` (a
bubble in a generated scene), ``flat`` (a 1×1 canvas, bare or ellipse — the
two are equal to the second decimal, A.1), ``grid`` (2×2 and up, a position
clause per cell).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

KINDS = ("single", "piece", "multi")
LAYOUTS = ("scene", "flat", "grid")
SIGMA_MAX = 0.9  # C.2: 0.8–0.95 is dead at 48 px for kana and kanji alike


@dataclass(frozen=True)
class Window:
    lo: float
    hi: float
    source: str

    @property
    def band(self) -> tuple[float, float]:
        return (self.lo, self.hi)


@dataclass(frozen=True)
class Row:
    kinds: tuple
    layouts: tuple
    px_lo: float  # inclusive
    px_hi: float | None  # exclusive; None = open
    lo: float
    hi: float
    source: str

    def holds(self, kind: str, px: float, layout: str) -> bool:
        return (
            kind in self.kinds
            and layout in self.layouts
            and px >= self.px_lo
            and (self.px_hi is None or px < self.px_hi)
        )


ROWS = (
    Row(
        ("single",),
        LAYOUTS,
        40,
        None,
        0.7,
        0.9,
        "band_b1_2026_09_23 (B.1): 24 kana on bubble-fit composites (the "
        "fill-0.7 draw: median 51 px, p10–p90 40–73 as built), native 84/55 vs "
        "49/30 of 192 at 0.5–0.7 — the row starts at that draw's p10, not at "
        "the nominal 48; step1_0921: grid cells 85–200 px + the flat half at "
        "0.7–0.9 bought identity; cf_kanji_c1 / band_c2_kanji: kanji take the "
        "kana band, 0.8–0.95 dead at 48 px (hi = 0.9)",
    ),
    Row(
        ("single",),
        LAYOUTS,
        24,
        40,
        0.5,
        0.7,
        "design.md § 2 stage0507 — ceiling only (cf_band_a1 A.1: 24–32 px "
        "letter live 0.35–0.7, peak 0.5); no training read. Where this row "
        "ends and the 0.7–0.9 row begins (32–40 px) is unread",
    ),
    Row(
        ("piece", "multi"),
        LAYOUTS,
        24,
        64,
        0.5,
        0.7,
        "piece: micro_cf_0922 (cf_sense_gate0_2026_09_22), 16 pieces at 35 px, "
        "exact 7 vs 1 of 32, native 6/3 vs 1/0 at 0.7–0.9. multi: no cell of "
        "its own — step 2 trained every ≥ 2-glyph item at 0.35–0.7 "
        "(_remap_band) and design.md § 2 puts 2–5-piece short lines ≈ 32 px "
        "here; small-kana digraphs (あっ) trained inside the singles recipe at "
        "0.7–0.9 in step1_0921, not read against this band. Ceiling: string "
        "32 px 0.5–0.7, 48 px 0.6–0.8 (A.1); grid strings +0.1 from 32 px on "
        "(A.1 runs 4–5), not read in training (plan_band B.2 unrun)",
    ),
    Row(
        ("piece", "multi"),
        LAYOUTS,
        12,
        24,
        0.3,
        0.5,
        "multi: cf_band_a1 A.2, EN two-word strings — 16 px live 0.25–0.6 "
        "centred 0.4, 12 px 0.2–0.6; grid cell 12–16 px 0.3–0.5; design.md § 2 "
        "stage0305. piece: by design, no read at this px. Nothing read under "
        "σ 0.25 (whether a stage0103 exists is open)",
    ),
)


def glyph_count(text: str) -> int:
    return sum(not c.isspace() for c in text)


def vocab_kind(vocab: str, n_tokens: int) -> str:
    """One vocab's kind from its Qwen token count and glyph count."""
    assert n_tokens >= 1, (vocab, n_tokens)
    if n_tokens >= 2:
        return "multi"
    return "single" if glyph_count(vocab) == 1 else "piece"


_RANK = {k: i for i, k in enumerate(KINDS)}


def kind_of(vocabs, n_tokens: Callable[[str], int]) -> str:
    """The item's kind: the heaviest of its vocabs' kinds (single < piece <
    multi). A grid of single glyphs is ``single``; a grid of pieces is
    ``piece``; anything holding a ≥ 2-token vocab — a line, a small-kana
    digraph — is ``multi``. Recipes draw one kind per item; a mixed grid
    (``grid_string`` with ``source = "both"``) takes the heavier band."""
    return max((vocab_kind(u, n_tokens(u)) for u in vocabs), key=_RANK.__getitem__)


def window(kind: str, px: float, layout: str) -> Window | None:
    """The training band for an item, or ``None`` when no row covers the
    cell (the builder counts those as ``no_window`` and re-draws)."""
    assert kind in KINDS, kind
    assert layout in LAYOUTS, layout
    for r in ROWS:
        if r.holds(kind, px, layout):
            assert r.hi <= SIGMA_MAX
            return Window(r.lo, r.hi, r.source)
    return None


def overlap(band, w) -> float:
    """|band ∩ w| / |band|."""
    lo, hi = band
    inter = max(0.0, min(hi, w.hi) - max(lo, w.lo))
    return inter / max(hi - lo, 1e-9)


def covers(band, w: Window | None, min_overlap: float = 0.8) -> bool:
    """The builder's gate (design § 4): keep an item iff the stage band is
    inside its window, or the window holds at least ``min_overlap`` of it."""
    if w is None:
        return False
    lo, hi = band
    if w.lo <= lo + 1e-9 and hi <= w.hi + 1e-9:
        return True
    return overlap(band, w) >= min_overlap - 1e-9


def table() -> str:
    """The rows as a text table (``scale.py windows``)."""
    lines = ["kinds         layouts             px          band       source"]
    for r in ROWS:
        px = f"{r.px_lo:g}–{'' if r.px_hi is None else f'{r.px_hi:g}'}"
        lines.append(
            f"{'/'.join(r.kinds):<13} {'/'.join(r.layouts):<19} {px:<11} "
            f"{r.lo:.1f}–{r.hi:.1f}    {r.source}"
        )
    return "\n".join(lines)
