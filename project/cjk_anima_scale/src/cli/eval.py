"""Flags the ``eval`` / ``native`` / ``target`` and ``cf_sense`` stages read."""

from __future__ import annotations

from common.prompts import NATIVE_CLAUSES, NATIVE_PROMPTS, TARGET_PROMPTS


def eval_args(g):
    g.add_argument(
        "--eval_groups",
        default="",
        help="eval: comma list of groups to render (single,combo,corpus,en); default all",
    )
    g.add_argument(
        "--eval_limit",
        type=int,
        default=0,
        help="eval: first N prompts per group (0 = all)",
    )
    g.add_argument(
        "--eval_tag",
        default="",
        help="eval: write img/reads/report/sheets under <arm>/eval_<tag>/ (e.g. a second eval_size)",
    )
    g.add_argument(
        "--no_floor",
        action="store_true",
        help="eval: skip the delta-scale-0 floor renders (identical across arms on the same eval set)",
    )
    g.add_argument(
        "--native_prompts",
        default=str(NATIVE_PROMPTS),
        help="native: scene prompt file (one per line; default the blind-pairs set)",
    )
    g.add_argument(
        "--native_chars",
        default="あ,か,す",
        help="native: comma list of kana to hang off every scene prompt",
    )
    g.add_argument(
        "--native_clauses",
        default=",".join(NATIVE_CLAUSES),
        help="native: clause shapes to append (" + ", ".join(NATIVE_CLAUSES) + ")",
    )
    g.add_argument(
        "--native_limit",
        type=int,
        default=0,
        help="native / target: first N prompts (0 = all)",
    )
    g.add_argument(
        "--target_prompts",
        default=str(TARGET_PROMPTS),
        help="target: file of full captions rendered verbatim, one per line, expected "
        "text = the quoted span (default: the user's ComfyUI prompts of 2026-09-17 — "
        "hoshino ai by @akipeko saying はい / こんにちは). Reads --eval_shape, "
        "--seeds, --no_floor, --eval_tag",
    )
    g.add_argument(
        "--delta_scale",
        type=float,
        default=1.0,
        help="native: ExtDelta scale for the trained cond (scene-survival vs identity probe)",
    )
    g.add_argument(
        "--en_word",
        default="hi",
        help="native: the EN word of the EN-reference render (English text reads "
        'as "<word>", same prompt and seed) — en cos / en cos out / box IoU '
        "score every trained render against it; refs are shared per size/steps/cfg",
    )


def cf_args(g):
    g.add_argument(
        "--cf_lang",
        default="en",
        choices=["en", "ja"],
        help="cf_sense (idea.md Gate 0): en = nonsense Latin words, no ext id, the "
        "ceiling of caption leverage; ja = the arm's trained single kana rows, "
        "delta on (trained) and off (floor)",
    )
    g.add_argument(
        "--cf_rows",
        default="single",
        choices=["single", "piece"],
        help="cf_sense: single = single-glyph units (kana rows / one-word strings); "
        "piece = the table's kana-only multi-glyph rows (id: two of one glyph count, "
        "order: two concatenated vs swapped) / en two-word strings and a three-word "
        "permutation",
    )
    g.add_argument(
        "--cf_units",
        default="kana",
        help="cf_sense --cf_lang ja --cf_rows single (plan_kanji Stage C): which "
        "trained single rows the pairs are drawn from. kana = the trained kana "
        "(the Gate 0 draw); kanji = the trained single kanji in three ink terciles "
        "(ink at 48 px Noto Serif CJK Regular); chars:<g0>/<g1>/… = explicit "
        "strata of trained rows. Pairs never cross a stratum and every item "
        "records its stratum, so the report bins by stratum and by stratum × px",
    )
    g.add_argument(
        "--cf_pairs",
        type=int,
        default=24,
        help="cf_sense: (A, B) pairs per kind (id / order)",
    )
    g.add_argument(
        "--cf_per_pair",
        type=int,
        default=1,
        help="cf_sense: layouts rendered per pair",
    )
    g.add_argument(
        "--cf_t",
        default="0.35,0.5,0.6,0.7,0.8,0.9",
        help="cf_sense: σ grid (DiT-scale) at which B's latent is noised",
    )
    g.add_argument(
        "--cf_layout",
        default="mixed",
        choices=["mixed", "flat", "bubble", "grid"],
        help="cf_sense (plan_band Stage A): mixed = the Gate 0 draw (an ellipse "
        "bubble on 60 %% of pairs, the bubble caption on all); flat = no bubble, "
        "plain caption; bubble = every pair in the ellipse; grid = the pair in "
        "the centre cell of a 3x3 flat grid with the grid caption (EN only)",
    )
    g.add_argument(
        "--cf_glyph_px",
        default="",
        help="cf_sense: glyph px list (e.g. 24,32,48,64,96,128) cycled over the "
        "items and recorded per item — --cf_per_pair = its length gives every "
        "pair every px. Empty = the layout's own draw (110–200 px singles, "
        "320/n–400/n strings)",
    )
    g.add_argument(
        "--cf_font",
        default="",
        help="cf_sense: pin one font file for every pair (the Light-vs-Black ink "
        "axis). Empty = one draw per pair among the fonts that cover the text",
    )
    g.add_argument(
        "--cf_text",
        default="auto",
        choices=["auto", "letter", "word", "string2"],
        help="cf_sense EN pair shape: letter = two capitals (id only); word = "
        "one nonsense word (+ the two-word order pair); string2 = two-word "
        "strings (+ the three-word order pair). auto = word for --cf_rows single, "
        "string2 for piece",
    )
