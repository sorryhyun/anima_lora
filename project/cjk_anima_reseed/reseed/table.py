"""The reseed table: one row per tier — what is drawn, at which glyph px, in
which σ band, how much of it.

A tier is ``<form>_<px>``: ``lone`` (one glyph alone on its canvas),
``grid`` (2×2 – 3×3, one glyph per cell), ``bubble1`` (one glyph in a scene
bubble), ``bubbleN`` (a 2–6 glyph window in one, routed per glyph),
``sent`` (a dialogue line in 2–3 columns of one), ``sentko`` (a KO line in
horizontal lines broken at its spaces); the number is the median
ink px (√(box area / glyphs)) the kana build drew.

The values are ``reseed_anchor fit``'s (``run1003_reseed_anchor_fit``):
- **px**: ``grid_44`` / ``grid_29`` / ``grid_16`` and their lone twins from
  ``grid_44``; the bubble tiers ``builder.TABLE``'s, ``bubble1_32`` and
  ``bubbleN_18`` fitted to their bubble (``fit``).
- **band**: ``reseed_recap``'s ``hp`` — upper edge at the gradient's
  identity half point, lower edge at half its peak
  (``_archive/reports/grad_bands_2026_10_03.md``).
- **share**: ``reseed_recap``'s; Σ 1.5 → 100 items per row.

``px_keep`` is the ink px an item must land in (else re-drawn): the band
law's gate cut these tiers there in the builds of record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEED = 0
# items per row at share 1: the scale line's ITEMS_PER_VOCAB (run0925_300f)
ITEMS_PER_ROW = 20_000 / 300

# the dialogue lines' Qwen-piece range and normalisation (scale's ``config.DATA``)
PHRASE_MIN_PIECES, PHRASE_MAX_PIECES, PHRASE_NORM = 2, 10, True

# the scene pools (``output/cjk_anima_scale/scenes_<tag>``)
SCENES = "s1,s1w,sl1w,ja_comic"
ONE_BUBBLE = "ja_comic"
SINGLE_SCENES = "s1,s1w"  # where a lone glyph goes
SINGLE_MAX_AR = 2.0
SMALL_POOL = "s1s"  # small bubbles: only a tier naming it in scene_pools draws it
HORIZONTAL_SCENES = "sl1w"  # left-to-right windows go here only
HORIZONTAL_FRAC = 0.3  # of the windows: a left-to-right line, marked in the caption
MONO_SHARE = 0.1  # of every scene draw: a greyscale / line-art scene (user, 10-03)
COLOR_MIN = 0.08  # under this share of coloured pixels a scene is mono
SHAPES = "448,512:2,448x512,512x448"  # a lone glyph's canvas
# a scene kept: every anchor bubble's interior this far (px) inside the
# canvas (closer, the outline is cut by the edge), and the outline-keeping
# erase taking the letters (at most this share of their ink left: past it the
# ring-median fill is not the bubble's — a dark bubble, a box over its edge)
BUBBLE_EDGE_MIN = 15
ERASE_LEFT_MAX = 0.3
# a scene takes at most this share of a tier's items (of each worker's draw):
# the long column windows fit 30 scenes and one took 151 of 533 (10-03)
SCENE_CAP = 0.01

# a dialogue line in columns (user, 10-05): 2–3 columns, never a square block
SENT_REGION_AR = 1.3  # its region at most this wide for its height
SENT_BLOCK_AR = 1.5  # its block at least this tall for its width: 3×3, 3×4, 2×3 out
SENT_FILL = (0.65, 0.9)  # the share of the region the block's fit takes (0.5 left
# small text afloat in a big bubble, user 10-05)
SENT_MIN_LETTERS = 6  # kana + kanji (the scale line's sentence floor)
SENT_FRAMES_OUT = {"sign"}  # dialogue goes in bubbles, not on a held sign
SENT_H_LINES = 4  # a KO line: at most this many horizontal lines (cjk_ko, 10-11)
SENT_H_LEFT = 0.7  # of the KO lines: left-aligned, the rest centred (user, 10-11)

# mark rows (user, 10-05): their windows hold the seed's trained letters
# (seed_retrain_0930 = preview4's bake); a heart is synthesised into dialogue
# lines (Manga109 letters none) — at the end of a line, or at a phrase break
# inside it (a ！？ / after 〜～…) at HEART_MID, doubled at HEART_DOUBLE
MARK_CONTEXT = (
    "models/vocab_packs/anima_cjk_vocab_pack_preview4/"
    "anima_cjk_vocab_pack_preview4_trained.json"
)
HEARTS = "♡♥"
MARK_NOT_LONE = "、。"  # never alone in a bubble
HEART_LINES = 4000
HEART_MID = 0.3
HEART_DOUBLE = 0.2

GRIDS = "2x2:1,3x3:1,2x3:1,3x2:1"
LONE = "1x1:1"
BUBBLE_FIT = (1.15, 1.7)  # a cell's bubble: its inscribed rectangle × the ink
CELL_JITTER = 0.08  # a grid glyph at its cell's centre ± this share of the cell


@dataclass(frozen=True)
class Tier:
    name: str  # <form>_<px>: file prefix, the record's `tier`, build.json key
    recipe: str  # recipes.RECIPES
    share: float  # items = ITEMS_PER_ROW × rows × share
    band: tuple  # σ the trainer draws in
    params: dict = field(default_factory=dict)
    px_keep: tuple = (None, None)  # ink px kept, inclusive; None = open


def _grid(name, share, band, glyph_px, *, lone=False, px_keep=(None, None)) -> Tier:
    """``glyph_px``: the font px range (the ink px lands ≈ 0.9 of it)."""
    p = {
        "grids": LONE if lone else GRIDS,
        "glyph_px": glyph_px,
        "bubble_frac": 0.5,
        "bubble_fit": BUBBLE_FIT,
    }
    if not lone:  # a lone glyph sits anywhere: its caption names no position
        p["cell_jitter"] = CELL_JITTER
    return Tier(name, "grid", share, band, p, px_keep)


TABLE = (
    _grid("grid_44", 0.1125, (0.55, 0.75), [44, 62], px_keep=(40, None)),
    _grid("grid_29", 0.225, (0.4, 0.7), [28, 42]),
    _grid("grid_16", 0.225, (0.2, 0.6), [14, 26]),
    _grid("lone_44", 0.075, (0.45, 0.8), [46, 64], lone=True),
    _grid("lone_28", 0.075, (0.35, 0.65), [28, 42], lone=True),
    _grid("lone_16", 0.075, (0.2, 0.5), [14, 26], lone=True),
    # the glyph filling 0.7 of a scene bubble's one-glyph fit
    Tier(
        "bubble1_52",
        "bubble1",
        0.1125,
        (0.55, 0.8),
        {"fit": 0.7, "min_glyph": 28},
        px_keep=(40, None),
    ),
    # the glyph at font px in a bubble whose one-glyph fit it fills 0.5–0.8
    Tier(
        "bubble1_32",
        "bubble1",
        0.09,
        (0.35, 0.6),
        {
            "glyph_px": [28, 40],
            "fill": [0.5, 0.8],
            "min_glyph": 12,
            "scene_pools": [SMALL_POOL],
        },
        px_keep=(None, 40),
    ),
    # windows filling 0.7–1.0 of their bubble
    Tier(
        "bubbleN_34",
        "bubbleN",
        0.21,
        (0.45, 0.7),
        {
            "fill": [0.7, 1.0],
            "min_glyph": 28,
            # the pool drew 7 / 18 / 27 / 25 / 22 %; 6 glyphs in a column at
            # 34 px fit 30 scenes (user, 10-03: 3 most, then 2, 4, 5, 6)
            "lengths": {2: 0.22, 3: 0.32, 4: 0.20, 5: 0.16, 6: 0.10},
        },
        px_keep=(None, 64),
    ),
    # windows at font px, in bubbles they fill ≥ 0.5 along and ≥ 0.3 across
    Tier(
        "bubbleN_18",
        "bubbleN",
        0.3,
        (0.2, 0.5),
        {
            "glyph_px": [12, 24],
            "min_glyph": 12,
            "fill": 0.9,
            "fill_min": 0.5,
            "cross_min": 0.3,
            "scene_pools": [SMALL_POOL],
        },
    ),
    # opt-in (share 0: a run's `shares` turns it on): the band read off the
    # gradient at 65 px (`_archive/reports/grid_64_2026_10_03.md`: f's half point 0.82,
    # ‖I‖'s low half 0.63); last, so a run without it draws what it drew
    _grid("grid_64", 0.0, (0.65, 0.8), [66, 92], px_keep=(56, None)),
    # opt-in: a Manga109 dialogue line lettered in 2–3 columns (user, 10-05)
    # at font px 26–36 / 16–22 (10-05: 20–28 / 13–19 read small) — the small
    # one on s1s too; the ink px counts the column gaps; the bands bubbleN's
    # at the same ink px
    Tier(
        "sent_34", "sent", 0.0, (0.45, 0.7), {"glyph_px": [26, 36], "lengths": [8, 10]}
    ),
    Tier(
        "sent_22",
        "sent",
        0.0,
        (0.3, 0.6),
        {"glyph_px": [16, 22], "lengths": [8, 14], "scene_pools": [SMALL_POOL]},
    ),
    # opt-in: a KO dialogue line (a ``lines`` file in ``LINE_LANG``) in 1–4
    # horizontal lines broken at its spaces (cjk_ko, 10-11), at sent_34 /
    # sent_22's font px and bands; lengths in cells (spaces not counted)
    Tier(
        "sentko_34",
        "sent",
        0.0,
        (0.45, 0.7),
        {"glyph_px": [26, 36], "lengths": [4, 12], "horizontal": True},
    ),
    Tier(
        "sentko_22",
        "sent",
        0.0,
        (0.3, 0.6),
        {
            "glyph_px": [16, 22],
            "lengths": [6, 20],
            "horizontal": True,
            "scene_pools": [SMALL_POOL],
        },
    ),
    # opt-in: the partial forms (``jamo.PART_SPELL``: a vowel or a final with
    # its initial left out, in its in-syllable form), alone or in a grid of
    # them on a flat canvas (cjk_ko, 10-12: a bubble fitted to a final's ink is
    # a sliver), at font px 40–72 — the part is a share of that; the band spans
    # grid_29 / grid_44's (not measured)
    Tier(
        "partial",
        "grid",
        0.0,
        (0.3, 0.75),
        {
            "grids": f"{LONE},{GRIDS}",
            "glyph_px": [40, 72],
            "bubble_frac": 0.0,
            "cell_jitter": CELL_JITTER,
            "parts": True,
        },
    ),
    # opt-in: the counterfactual syllables (``counterfactual``: a trained
    # syllable less its initial or its vowel, the rest where it sits), drawn
    # as ``partial`` is (cjk_ko, 10-12)
    Tier(
        "counterfactual",
        "grid",
        0.0,
        (0.3, 0.75),
        {
            "grids": f"{LONE},{GRIDS}",
            "glyph_px": [40, 72],
            "bubble_frac": 0.0,
            "cell_jitter": CELL_JITTER,
            "cf": True,
        },
    ),
    # opt-in: the counterfactual syllables in a scene bubble, as bubble1_52 /
    # bubble1_32 draw a glyph (cjk_ko, 10-13)
    Tier(
        "cf_bubble1_52",
        "bubble1",
        0.0,
        (0.55, 0.8),
        {"fit": 0.7, "min_glyph": 28, "cf": True},
        px_keep=(40, None),
    ),
    Tier(
        "cf_bubble1_32",
        "bubble1",
        0.0,
        (0.35, 0.6),
        {
            "glyph_px": [28, 40],
            "fill": [0.5, 0.8],
            "min_glyph": 12,
            "scene_pools": [SMALL_POOL],
            "cf": True,
        },
        px_keep=(None, 40),
    ),
    # opt-in: the large glyph at the σ where the canvas is decided
    # (L3, `reports/l3_full_2026_10_16.md`, user 10-16), each in place of bubble1_52's share
    # (one glyph an item, so a row's occurrences stay): in a coloured scene
    # bubble (L3a), its ink centred there, or alone on a flat white page (L3b)
    Tier(
        "bubble1_72",
        "bubble1",
        0.0,
        (0.65, 0.9),
        {
            "glyph_px": [64, 96],
            "fill": [0.4, 0.95],
            "min_glyph": 28,
            "mono": False,
            "pose": True,  # the ink centred in its bubble / sign (user 10-16)
        },
        px_keep=(56, None),
    ),
    Tier(
        "lone_72",
        "grid",
        0.0,
        (0.65, 0.9),
        {"grids": LONE, "glyph_px": [64, 96], "bubble_frac": 0.0, "bg": "white"},
        px_keep=(56, None),
    ),
)
