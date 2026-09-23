"""Flags the ``eval`` / ``native`` and ``classify`` / ``classify_str`` stages read."""

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
        "--with_c_flat",
        type=int,
        default=0,
        help="eval: add the saved c_flat to every trained row (the flat-template "
        "eval with the switch on); native runs without it",
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
        "--seeds, --delta_parts, --no_floor, --eval_tag",
    )
    g.add_argument(
        "--delta_scale",
        type=float,
        default=1.0,
        help="native: ExtDelta scale for the trained cond (scene-survival vs identity probe)",
    )
    g.add_argument(
        "--out_vec",
        default="",
        help="native: .pt with the pretrained quoted-EN adapter-output shift "
        "(quote_dir_save.py: dirs[<frame>], shift_norm[<frame>]); adds conds fq<s> "
        "= rows f + s × that shift at the ext positions of the adapter output",
    )
    g.add_argument(
        "--out_vec_scales",
        default="1.0",
        help="native: comma list of multiples of the EN shift norm for --out_vec",
    )
    g.add_argument(
        "--out_vec_frame",
        default="reads_as",
        help="native: which frame's shift to use from --out_vec (reads_as|bubble_reads|she_says|bare_quotes|avg)",
    )
    g.add_argument(
        "--delta_parts",
        default="full",
        help="native: comma list of table parts to render as separate conds "
        "(encoder arms; raw = g + c + f): full (named `trained`), f (per-row "
        "residual), c (common vector), g (centred encoder part), and sums fg fc gc",
    )
    g.add_argument(
        "--native_floor",
        type=int,
        default=0,
        help="native: also render the delta-off floor cond (needed only for the "
        "old scene-kept margin; the EN-reference ruler replaced it 2026-09-15)",
    )
    g.add_argument(
        "--en_word",
        default="hi",
        help="native: the EN word of the EN-reference render (English text reads "
        'as "<word>", same prompt and seed) — en cos / en cos out / box IoU '
        "score every trained render against it; refs are shared per size/steps/cfg",
    )
    g.add_argument(
        "--kept_ref",
        default="",
        help="native: dir holding floor_*.png of an earlier native run, the "
        "scene-kept reference when this run has --no_floor (default: the arm's "
        "native/img)",
    )
    g.add_argument(
        "--kept_tau",
        type=float,
        default=0.0,
        help="native: scene-kept margin — cos(img, floor) − cos(img, flat "
        "training-canvas prototype) at or above this counts as kept",
    )


def classify_args(g):
    g.add_argument(
        "--cls_t",
        default="0.1,0.2,0.35,0.5,0.65,0.8,0.95",
        help="classify: σ grid (DiT-scale) to score the candidates at",
    )
    g.add_argument(
        "--cls_per_kana",
        type=int,
        default=2,
        help="classify: held-out renders per kana",
    )
    g.add_argument(
        "--cls_batch",
        type=int,
        default=24,
        help="classify: candidate captions per DiT forward",
    )
    g.add_argument(
        "--cls_pairs", type=int, default=24, help="classify_str: 2-kana strings"
    )
    g.add_argument(
        "--cls_triples", type=int, default=8, help="classify_str: 3-kana strings"
    )
    g.add_argument(
        "--cls_lang",
        default="ja",
        choices=["ja", "en"],
        help="classify_str: ja = kana strings of trained rows; en = nonsense two-word Latin strings (base-model order control)",
    )
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
