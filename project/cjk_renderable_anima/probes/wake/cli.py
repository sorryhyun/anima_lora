"""Command line: flags grouped by the stage that reads them."""

from __future__ import annotations

import argparse

from .common import NATIVE_CLAUSES, NATIVE_PROMPTS


def build_parser(stages, description: str | None = None) -> argparse.ArgumentParser:
    """``stages``: the stage names ``--stage`` accepts (besides ``all``)."""
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _run_args(p.add_argument_group("run"), stages)
    _generation_args(p.add_argument_group("generation"))
    _data_args(p.add_argument_group("data"))
    _train_args(p.add_argument_group("train"))
    _encoder_args(p.add_argument_group("train: encoder arm"))
    _eval_args(p.add_argument_group("eval / native"))
    _classify_args(p.add_argument_group("classify / classify_str"))
    _scene_args(p.add_argument_group("scenes"))
    _synth_args(p.add_argument_group("data / train: S line (plan_synth)"))
    return p


# ----------------------------------------------------------------------------


def _run_args(g, stages):
    g.add_argument("--stage", nargs="+", default=["all"], choices=["all", *stages])
    g.add_argument("--arm", default="rows", choices=["rows", "rows_adapter", "encoder"])
    g.add_argument("--device", default="cuda")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--data_tag",
        default="",
        help="suffix for output/wake_probe/data_<tag> and <arm>_<tag>",
    )
    g.add_argument(
        "--arm_tag",
        default="",
        help="suffix for the arm dir only (<arm>_<data_tag>_<arm_tag>): a second "
        "train recipe on the same data without overwriting the first",
    )


def _generation_args(g):
    g.add_argument("--steps", type=int, default=28, help="inference steps")
    g.add_argument("--cfg", type=float, default=4.0)
    g.add_argument(
        "--seeds", type=int, default=2, help="seeds per prompt (salad: 3 recommended)"
    )
    g.add_argument("--salad_size", type=int, default=768)
    g.add_argument("--train_size", type=int, default=512)
    g.add_argument("--eval_size", type=int, default=512)
    g.add_argument(
        "--eval_shape",
        default="",
        help="eval: WxH canvas instead of --eval_size² (e.g. 384x512); pair with --eval_tag",
    )


def _data_args(g):
    g.add_argument(
        "--shapes",
        default="",
        help="data: mixed canvas shapes, comma list of side or WxH with an optional "
        ":weight (e.g. '384,448,512:2,384x512,512x384'); every font item draws one, "
        "corpus crops draw from the squares; train then batches one shape per step "
        "and caches latents per shape. Empty = 512² renders at --train_size (old dirs)",
    )
    g.add_argument("--n_single", type=int, default=6, help="font renders per kana")
    g.add_argument("--n_combo", type=int, default=700)
    g.add_argument("--n_corpus", type=int, default=600)
    g.add_argument(
        "--kanji",
        type=int,
        default=0,
        help="data (P0b): add the N most frequent single-row corpus kanji as singles "
        "×n_single; eval group single_kanji (18 drawn)",
    )
    g.add_argument(
        "--kana_ext",
        action="store_true",
        help="data (P0b): add voiced / handakuten / small kana (68) as singles "
        "×n_single; eval group single_ext (18 drawn)",
    )
    g.add_argument(
        "--balanced",
        type=int,
        default=0,
        help="data: groups of N distinct strings sharing one layout (font, canvas, "
        "bubble, glyph size/position); train then batches one group per step "
        "(W2a; needs --batch N). 0 = shuffled items",
    )
    g.add_argument(
        "--layout",
        default="v1",
        choices=["v1", "jitter"],
        help="data: v1 = pre-W2a renders (big centred dark glyph on a light canvas, "
        "bit-identical rebuilds); jitter = random position / size / ink colour / "
        "outline / dark backgrounds / bubble box (the 2026-09-14 data lever)",
    )
    g.add_argument(
        "--only_chars",
        default="",
        help="restrict the kana inventory (textual-inversion regime: few chars, many exposures)",
    )
    g.add_argument(
        "--words",
        type=int,
        default=0,
        help="data: add the N most frequent single-Qwen-piece words of the training "
        "corpus to the inventory (each is an existing pack row = one address); "
        "corpus lines are then kept only when every piece is a trained row",
    )
    g.add_argument("--word_min_len", type=int, default=2)
    g.add_argument(
        "--held_out_words",
        type=int,
        default=0,
        help="data: K of the words removed from every training item, eval group word_held",
    )
    g.add_argument(
        "--n_word_eval",
        type=int,
        default=16,
        help="data: trained words in eval group word",
    )
    g.add_argument(
        "--n_line_eval",
        type=int,
        default=16,
        help="data: held-out corpus lines of 2-3 pieces, every piece trained (eval group line)",
    )
    g.add_argument(
        "--line_max_len",
        type=int,
        default=8,
        help="data: corpus line length cap in word mode",
    )
    g.add_argument(
        "--strings_only",
        action="store_true",
        help="data: strings arm — no singles; 2–4-piece random-order strings of trained rows only (needs --words)",
    )
    g.add_argument(
        "--n_strings", type=int, default=6000, help="data: --strings_only font items"
    )
    g.add_argument(
        "--single_frac",
        type=float,
        default=0.0,
        help="data: --strings_only fraction of font items that are singles (plan P1 mixed distribution)",
    )
    g.add_argument(
        "--word_frac",
        type=float,
        default=0.25,
        help="data: --strings_only P(slot is a trained word)",
    )
    g.add_argument(
        "--n_flip_eval",
        type=int,
        default=12,
        help="data: --strings_only clean kana pairs, both orders → group flip",
    )
    g.add_argument(
        "--n_str3_eval",
        type=int,
        default=8,
        help="data: --strings_only clean 3-kana strings → group str3",
    )


def _train_args(g):
    g.add_argument("--train_steps", type=int, default=2000)
    g.add_argument("--batch", type=int, default=4)
    g.add_argument(
        "--lr_rows", type=float, default=3e-3, help="in units of the mean pack-row norm"
    )
    g.add_argument("--lr_adapter", type=float, default=1e-4)
    g.add_argument("--adapter_rank", type=int, default=16)
    g.add_argument(
        "--lr_decay",
        default="none",
        choices=["none", "cosine"],
        help="train: lr schedule over --train_steps (cosine to 0; all param groups)",
    )
    g.add_argument("--grad_ckpt", type=int, default=1)
    g.add_argument(
        "--compile",
        type=int,
        default=1,
        help="train: per-block torch.compile of the frozen DiT (the OOM remedy of record)",
    )
    g.add_argument("--activation_memory_budget", type=float, default=0.99)
    g.add_argument(
        "--aggressive_recompute",
        type=int,
        default=1,
        help="compile: partitioner aggressive recomputation (−VRAM, +~12 % s/it); 0 when memory allows",
    )
    g.add_argument(
        "--t_min",
        type=float,
        default=None,
        help="restrict FM timesteps (W2 σ-restriction lever; None = full range)",
    )
    g.add_argument("--t_max", type=float, default=None)


def _encoder_args(g):
    g.add_argument(
        "--lr_enc", type=float, default=3e-4, help="encoder arm: AdamW lr on the CNN"
    )
    g.add_argument(
        "--lr_common",
        type=float,
        default=1e-3,
        help="encoder arm: lr on the shared layout vector c (row-norm units; the rows lr)",
    )
    g.add_argument(
        "--common_cap",
        type=float,
        default=0.75,
        help="encoder arm: ‖c‖ bound in row norms, applied to the parameter after each step",
    )
    g.add_argument(
        "--out_scale",
        type=float,
        default=1.0 / 64,
        help="encoder arm: scale on the zero-init head (raise one notch to 1/16 if spread stays flat)",
    )
    g.add_argument(
        "--kill_spread",
        type=float,
        default=0.05,
        help="encoder arm: abort if rel_spread_ref (fixed font, no shift) is below this once --kill_spread_step is reached",
    )
    g.add_argument(
        "--kill_spread_step",
        type=int,
        default=600,
        help="encoder arm: 0 disables the spread rule (--kill_max_row 0 disables the max-row rule)",
    )
    g.add_argument(
        "--kill_max_row",
        type=float,
        default=2.0,
        help="encoder arm: abort if any row's delta passes this many row norms",
    )
    g.add_argument(
        "--glyph_size", type=int, default=96, help="encoder arm: glyph render side"
    )
    g.add_argument(
        "--head_init",
        default="zero",
        choices=["zero", "random"],
        help="encoder arm: last head layer zero (attempts 1–10) or random full-rank, "
        "rescaled so the step-0 identity spread is --init_spread row norms",
    )
    g.add_argument(
        "--init_spread",
        type=float,
        default=1.0,
        help="encoder arm: --head_init random target spread on the reference render",
    )
    g.add_argument(
        "--decor",
        type=float,
        default=0.0,
        help="encoder arm: λ on mean_{i≠j} cos²(r_i, r_j) over the centred trained "
        "rows of the encoder table (0 = off; history.md Run 1b amended: ≈ 0.02 "
        "against an FM loss of ≈ 0.04)",
    )
    g.add_argument(
        "--free_residual",
        type=float,
        default=0.0,
        help="encoder arm: μ on mean_i ‖f_i‖² for a per-row free residual on the "
        "trained rows (row = g(glyph) + f_i; held-out rows get g only). 0 = off. "
        "history.md Run 1d: the semi-amortised hybrid — f carries the identity "
        "magnitude the shared head cannot, the L2 pushes what g can explain into g",
    )
    g.add_argument(
        "--lr_free",
        type=float,
        default=1e-3,
        help="encoder arm: lr of the free residual, row-norm units (W1 rows: 1e-3; 3e-3 walks off-manifold)",
    )
    g.add_argument(
        "--init_encoder",
        default="",
        help="encoder arm: warm-start the encoder (conv/proj/head + common) from another arm's trained.pt",
    )
    g.add_argument(
        "--init_free",
        default="",
        help="train: warm-start the free residual by ext id from another arm's trained.pt",
    )
    g.add_argument(
        "--enc_pool",
        default="spatial",
        choices=["spatial", "mean"],
        help="encoder arm: feature pooling — spatial keeps the arrangement (glyph), "
        "mean keeps channel statistics only (attempts 4–7 tracked font, not glyph)",
    )
    g.add_argument(
        "--font_mode",
        default="mean",
        choices=["mean", "random"],
        help="encoder arm: input render — mean over every font (font-free) or one random font per row per step",
    )
    g.add_argument(
        "--held_out",
        type=int,
        default=0,
        help="encoder arm: N single chars removed from every training item and "
        "evaluated as group single_held (the generalisation test)",
    )
    g.add_argument(
        "--held_out_chars",
        default="",
        help="encoder arm: explicit held-out chars instead of --held_out's draw "
        "(Run 2: IDS composites whose atoms are trained, e.g. 明休男岩加相困森)",
    )


def _eval_args(g):
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
        "--native_limit", type=int, default=0, help="native: first N prompts (0 = all)"
    )
    g.add_argument(
        "--delta_scale",
        type=float,
        default=1.0,
        help="native: ExtDelta scale for the trained cond (scene-survival vs identity probe)",
    )
    g.add_argument(
        "--delta_parts",
        default="full",
        help="native: comma list of table parts to render as separate conds "
        "(encoder arms; raw = g + c + f): full (named `trained`), f (per-row "
        "residual), c (common vector), g (centred encoder part), and sums fg fc gc",
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


def _synth_args(g):
    g.add_argument(
        "--scenes",
        default="",
        help="data: S-line mix from output/wake_probe/scenes_<tag> — flat singles "
        "(the rest) + scene composites (--scene_frac) + natural phrases on flat "
        "canvases (--natural_frac) + random-order strings (--strings_frac); "
        "replaces the singles×n_single + combos + corpus-crop mix",
    )
    g.add_argument(
        "--n_items", type=int, default=16000, help="data: --scenes total items"
    )
    g.add_argument(
        "--scene_frac",
        type=float,
        default=0.4,
        help="data: --scenes share of items that are scene composites",
    )
    g.add_argument(
        "--natural_frac",
        type=float,
        default=0.2,
        help="data: --scenes share that are covered corpus phrases in a font on a flat canvas",
    )
    g.add_argument(
        "--strings_frac",
        type=float,
        default=0.0,
        help="data: --scenes share that are random-order 2–4-piece strings "
        "(strings-arm recipe; adds the flip / str3 eval groups). S0: 0 (user, 2026-09-14)",
    )
    g.add_argument(
        "--scene_stroke",
        type=float,
        default=0.25,
        help="data: --scenes share of composites whose glyphs get a thin outline "
        "in the fill colour (manga lettering over art)",
    )
    g.add_argument(
        "--scene_min_glyph",
        type=int,
        default=40,
        help="data: --scenes smallest per-glyph cell (px) a composite may draw; "
        "a text that would go smaller is redrawn shorter",
    )
    g.add_argument(
        "--n_phrase_eval",
        type=int,
        default=16,
        help="data: --scenes covered held-out corpus lines never trained → group phrase_held",
    )
    g.add_argument(
        "--box_weight",
        type=float,
        default=1.0,
        help="train: FM loss weight inside a composite item's swapped text box "
        "(1 outside; batch-normalised); 1 = off",
    )
    g.add_argument(
        "--c_flat",
        type=int,
        default=0,
        help="train (rows arm): per-source layout vector added to every trained "
        "row on flat-canvas batches only (Δ_r = f_r + 𝟏[flat]·c_flat)",
    )
    g.add_argument(
        "--lr_c_flat", type=float, default=0.0, help="train: c_flat lr (0 = --lr_rows)"
    )
    g.add_argument(
        "--c_flat_cap",
        type=float,
        default=0.75,
        help="train: ‖c_flat‖ bound in row norms (projected after each step)",
    )
    g.add_argument(
        "--f_orth",
        type=float,
        default=0.0,
        help="train: λ · mean_r cos²(f_r, c_flat) guard (0 = off; read `leak` first)",
    )
    g.add_argument(
        "--with_c_flat",
        type=int,
        default=0,
        help="eval: add the saved c_flat to every trained row (the flat-template "
        "eval with the switch on); native runs without it",
    )


def _classify_args(g):
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


def _scene_args(g):
    g.add_argument(
        "--scene_tag", default="s0", help="scenes: output/wake_probe/scenes_<tag>"
    )
    g.add_argument(
        "--scene_n", type=int, default=1000, help="scenes: prompts to generate"
    )
    g.add_argument(
        "--scene_shapes",
        default="448,512:2,448x512,512x448",
        help="scenes: canvas pool (S0 recipe; --shapes syntax)",
    )
    g.add_argument(
        "--scene_anchors",
        default="",
        help="scenes: comma list of EN anchor words (default the built-in ten)",
    )
    g.add_argument(
        "--scene_bubble_tag",
        default="speech bubble",
        help="scenes: bubble tag in the prompt — recorded so the data stage "
        "spells the composite caption identically",
    )
    g.add_argument(
        "--scene_min_box",
        type=int,
        default=56,
        help="scenes: reject usable regions under this many px on the short side",
    )
    g.add_argument(
        "--scene_char_frac",
        type=float,
        default=0.3,
        help="scenes: share of 1girl prompts naming a dataset character (else `original`)",
    )
    g.add_argument(
        "--scene_artist_frac",
        type=float,
        default=0.8,
        help="scenes: share of prompts carrying one of the dataset's @artist tags",
    )
    g.add_argument(
        "--scene_batch", type=int, default=4, help="scenes: prompts per DiT pass"
    )
    g.add_argument(
        "--scene_negative",
        default="worst quality, lowres, old, bad hands, bad anatomy, sepia, blurry, glitch, jpeg artifacts",
        help="scenes: negative prompt (inference only; never enters a caption)",
    )
    g.add_argument(
        "--scene_artists",
        default="sincos,hews",
        help="scenes: comma list of curated artist names added to the dataset pool at 4× weight",
    )
    g.add_argument(
        "--scene_gen_scale",
        type=float,
        default=1.0,
        help="scenes: render at this multiple of the pool shape, then downsample "
        "(the base draws crude scenes at 512²; 2.0 = its native ~1024)",
    )
    g.add_argument(
        "--scene_rejudge",
        type=int,
        default=0,
        help="scenes: re-apply the filter to scenes_<tag> from its stored reads (CPU, no generation)",
    )
    g.add_argument(
        "--scene_allow_open",
        type=int,
        default=0,
        help="scenes: keep images whose bubble fill runs to the border (open background)",
    )
