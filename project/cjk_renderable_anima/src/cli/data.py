"""Flags the ``data`` stage reads: the unit sources and flat mix, then the S-line scene mix."""

from __future__ import annotations

from data.units import KINDS


def data_args(g):
    g.add_argument(
        "--units",
        action="append",
        metavar="SPEC",
        help="data: what the ext table trains on — repeat once per source, "
        f"`kind[:arg][*weight][/held=K]` over {' | '.join(KINDS)}: e.g. `kana`, "
        "`kana_ext`, `kanji:200`, `words:100/held=8`, `chars:あかす出人日`, "
        "`list:、,。,！！`. Default `kana` (the 92); name no kana source to train "
        "punctuation / kanji / words alone. `*W` sets the draw weight in the "
        "S-line singles pool, `/held=K` holds K units out. Typed order never "
        "changes the data — see data/units.py",
    )
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
        "--word_min_len",
        type=int,
        default=2,
        help="data: shortest piece `--units words:N` will count as a word",
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
        help="data: strings arm — no singles; 2–4-piece random-order strings of trained rows only (needs --units words:N)",
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


def data_synth_args(g):
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
        "--phrase_file",
        default="",
        help="data: TSV `line[\\tbook[\\tn_pieces]]` replacing the corpus as the "
        "phrase source (--natural_frac items and the composites' phrase kind); "
        "held set = --phrase_held_books whole books → eval group phrase_held, "
        "plus group `phrase` (trained lines)",
    )
    g.add_argument(
        "--phrase_pieces",
        type=int,
        default=0,
        help="data: with --phrase_file, add its N most frequent pieces outside the "
        "inventory as rows (trained through the phrases only; no singles / evals)",
    )
    g.add_argument(
        "--phrase_held_books",
        type=int,
        default=6,
        help="data: --phrase_file books held out whole for phrase_held",
    )
    g.add_argument("--phrase_min_pieces", type=int, default=3)
    g.add_argument("--phrase_max_pieces", type=int, default=10)
    g.add_argument(
        "--scene_drop",
        default="",
        help="data: --scenes kept scenes to leave out, `tag:i,i;tag:i` "
        "(sl1w:332,957 — bubble-less tall regions the sentence quota reused)",
    )
    g.add_argument(
        "--scene_mix",
        default="",
        help="data: --scenes hard per-kind quotas for the composites, e.g. "
        "`single=0.1,short=0.5,sentence=0.4` (sentence arm, 2026-09-16): "
        "`single` = one unit of the singles pool, `short` = a --phrase_file "
        "line of --short_pieces pieces below the sentence floor, `sentence` = "
        "a line with >= --sentence_min_letters letters. Texts are drawn "
        "uniformly among the lines that fit the bubble; a kind that fits no "
        "text re-picks the scene and is never demoted. Empty = the seed "
        "behaviour (composites mirror the flat kinds, first fitting line)",
    )
    g.add_argument(
        "--short_pieces",
        default="2-5",
        help="data: --scene_mix piece range (Qwen pieces, inclusive) of the "
        "`short` kind; the phrase file must carry lines that short "
        "(dialogue_2_10.tsv; --phrase_min_pieces 2)",
    )
    g.add_argument(
        "--short_max_lines",
        type=int,
        default=1,
        help="data: --scene_mix columns a `short` item may wrap into (1 = one "
        "vertical line, user 2026-09-16; a bubble too low for it is a miss and "
        "the scene is re-picked). Sentences use --scene_max_lines",
    )
    g.add_argument(
        "--sentence_min_glyph",
        type=int,
        default=0,
        help="data: --scene_mix per-glyph floor (px) for the `sentence` kind; "
        "0 = --scene_min_glyph. The sentence arm runs 20 (user, 2026-09-16: a "
        "6-glyph line as one column; at 28 px two of the 276 sl1w bubbles hold it)",
    )
    g.add_argument(
        "--sentence_fill",
        type=float,
        default=0.0,
        help="data: --scene_mix bubble fill for the `sentence` kind; 0 = "
        "--scene_fill. The sentence arm runs 0.9",
    )
    g.add_argument(
        "--scene_fewest_lines",
        type=int,
        default=1,
        help="data: --scene_mix 1 = a text that fits in fewer columns at the "
        "glyph floor stays there (one column whenever it fits); 0 = the seed "
        "rule (more columns when the glyph gets 1.4x larger)",
    )
    g.add_argument(
        "--short_lexical",
        type=int,
        default=1,
        help="data: --scene_mix 1 = a `short` line must carry a word piece "
        "(a multi-glyph kana Qwen piece or a kanji): あっ / ぎゃああ / せーの "
        "are out, お前 / 待て / 勝先生 stay (user, 2026-09-16: combined glyphs "
        "must make words). Costs the words the tokenizer splits into single "
        "glyphs (きつね), ~25 %% of the 2–5-piece lines. 0 = every line",
    )
    g.add_argument(
        "--sentence_min_letters",
        type=int,
        default=6,
        help="data: --scene_mix floor for the `sentence` kind — kana + kanji "
        "glyphs (at least 4 of them distinct), punctuation / digits not "
        "counted, so ハハハ・・・, 何っ！ and ハハハハハハ are not sentences. "
        "Lines under it with a `short` piece count are `short`; other lines "
        "under it are in no kind",
    )
    g.add_argument(
        "--scene_vertical",
        type=int,
        default=0,
        help="data: --scenes 1 = tategaki only: a multi-glyph text that does "
        "not fit as columns is a miss (re-pick the scene under --scene_mix, "
        "shorter text otherwise), never a horizontal line",
    )
    g.add_argument(
        "--scene_min_tokens",
        type=int,
        default=0,
        help="data: --scenes keep only scenes whose canvas is at least this "
        "many DiT tokens ((W/16)*(H/16)); 900 keeps the 512² family and "
        "drops 448² (784) and 448x512 (896) (user, 2026-09-16)",
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
        default=28,
        help="data: --scenes smallest per-glyph cell (px) a composite may draw; "
        "a text that would go smaller is redrawn shorter. 40 through the seed "
        "(S0 … 53k); 28 since 2026-09-16 for the sentence line — at 40 a "
        "phrase fits 5 %% of sl1w bubbles even wrapped, at 28 + 2 columns 43 %%",
    )
    g.add_argument(
        "--scene_tall_ar",
        type=float,
        default=0.0,
        help="data: --scenes keep only scenes whose headline region is at least "
        "this tall for its width (1.0 = taller than wide; 0 = every kept scene). "
        "Tategaki pool for the sentence line: sl1w 116/276 at 1.0, 61 at 1.3",
    )
    g.add_argument(
        "--scene_max_lines",
        type=int,
        default=3,
        help="data: --scenes columns (tategaki, right-to-left; lines when the "
        "text only fits horizontally) a composite may wrap into, cut at Qwen "
        "piece boundaries only; more columns win only at 1.4x the glyph. "
        "1 = the single-line seed behaviour; the sentence arm runs 2 (user, "
        "2026-09-16: three columns in a bubble read wrong)",
    )
    g.add_argument(
        "--flat_bubble",
        type=float,
        default=0.6,
        help="data: --scenes share of flat-canvas items (font singles / phrases) "
        "drawn inside a speech bubble; the rest are bare-canvas TPL_PLAIN. "
        "1.0 = S0b option (a): one flat layout, so c_flat is one direction "
        "(S0 at 0.6 leaked the plain layout into the rows, leak 0.28)",
    )
    g.add_argument(
        "--scene_fill",
        type=float,
        default=0.9,
        help="data: --scenes fraction of the bubble's usable region the text "
        "block may fill (0.9 = edge to edge, S0; 0.7 leaves manga-like air — "
        "at 0.7 the median single glyph is ≈ 50 px on the s0 scenes, 92 %% ≥ 40 px)",
    )
    g.add_argument(
        "--n_phrase_eval",
        type=int,
        default=16,
        help="data: --scenes covered held-out corpus lines never trained → group phrase_held",
    )
