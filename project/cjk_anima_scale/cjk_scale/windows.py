"""windows — the vocab band law as code: (kind, px, layout) → the σ band an
item trains in.

The rows below are ``band_experiment_results.md`` (§ 2 ceiling table, § 3
training reads) and ``design.md`` § 2 (the stage table), one row per cell
that has a read, each with its provenance. They are a lookup, not a formula:
a formula off the EN ceiling table cannot reproduce the reads (it puts a
128 px grid cell above 0.9, which C.2 says is dead at 48 px and step1_0921
trained at 0.7–0.9). A new read that changes a row changes it here, with
its report.

Units: ``px`` is the data stage's ink-stat glyph px, √(box area / glyphs) —
the number every "as built" px in the reports is. ``kind`` is keyed on the
item's glyph count (``kind_of``), the count ``train/stage.py::_remap_band``
used: one glyph → ``single``, else ``multi``. ``layout``: ``scene`` (a bubble
in a generated scene), ``flat`` (a 1×1 canvas, bare or ellipse — the two are
equal to the second decimal, A.1), ``grid`` (2×2 and up, a position clause
per cell).
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("single", "multi")
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
    kind: str
    layouts: tuple
    px_lo: float  # inclusive
    px_hi: float | None  # exclusive; None = open
    lo: float
    hi: float
    source: str

    def holds(self, kind: str, px: float, layout: str) -> bool:
        return (
            kind == self.kind
            and layout in self.layouts
            and px >= self.px_lo
            and (self.px_hi is None or px < self.px_hi)
        )


ROWS = (
    Row(
        "single",
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
        "single",
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
        "multi",
        LAYOUTS,
        24,
        64,
        0.5,
        0.7,
        "micro_cf_0922 (cf_sense_gate0_2026_09_22): 16 pieces at 35 px, exact "
        "7 vs 1 of 32, native 6/3 vs 1/0 at 0.7–0.9; ceiling string 32 px "
        "0.5–0.7, 48 px 0.6–0.8 (A.1). Grid strings: ceiling +0.1 from 32 px "
        "on (A.1 runs 4–5), not read in training (plan_band B.2 unrun)",
    ),
    Row(
        "multi",
        LAYOUTS,
        12,
        24,
        0.3,
        0.5,
        "cf_band_a1 A.2: 16 px string live 0.25–0.6 centred 0.4, 12 px "
        "0.2–0.6; grid cell 12–16 px 0.3–0.5; design.md § 2 stage0305. "
        "Nothing read under σ 0.25 (whether a stage0103 exists is open)",
    ),
)


def kind_of(units) -> str:
    """``single`` iff every unit of the item is one glyph, else ``multi``.
    A grid of single glyphs is ``single``; a grid of strings, a piece, a
    line, a small-kana digraph (あっ: two glyphs on the canvas) are
    ``multi`` — the count the caption asks the DiT to draw."""
    return "single" if all(glyph_count(u) == 1 for u in units) else "multi"


def glyph_count(text: str) -> int:
    return sum(not c.isspace() for c in text)


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
    lines = ["kind    layouts             px          band       source"]
    for r in ROWS:
        px = f"{r.px_lo:g}–{'' if r.px_hi is None else f'{r.px_hi:g}'}"
        lines.append(
            f"{r.kind:<7} {'/'.join(r.layouts):<19} {px:<11} "
            f"{r.lo:.1f}–{r.hi:.1f}    {r.source}"
        )
    return "\n".join(lines)
