# recipe — the JA vocab-pack training order (2026-09-23)

The clean line. Code stays in `project/cjk_renderable_anima/src/` (`wake_probe.py`
stages `data` / `train` / `eval` / `native` / `cf_sense`); this file is the
recipe the table of record is built with. Numbers behind each setting:
`project/cjk_renderable_anima/reports/` and `sent_run.md`. Every launch:
raw pack in the submit shell, through the daemon.

```
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack   # sha 7b9fce0b…
make daemon-run ARGS="--label <name> [--queue] project/cjk_renderable_anima/src/wake_probe.py …"
```

## Order

```
step 1a  single-glyph table   (kana · kanji · punctuation)      grid 50 %, σ 0.7–0.9, 80 steps/row
step 1b  multi-glyph table    (kana-bearing pieces)             scene only, σ 0.5–0.7, ≈ 190 steps/row
         merge_tables.py  1a + 1b  →  step-1 table
step 2   sentence run          warm from the step-1 table       band2 μ 0.1, 30 k (the 09-22 run's argv)
         bake_vocab_pack.py    →  pack pair
```

Step 1 gives every row its own unit; step 2 trains real lines on the merged
table so the rows compose. **No cold row inside step 2** — new vocabulary is a
step-1 table, merged first.

Rows are split by glyph count because the caption's leverage lives at
different σ for the two: single glyphs at 0.7–0.9 (peak 0.8, zero below 0.7),
multi-glyph boxes at 0.5–0.7 (`reports/cf_sense_gate0_2026_09_22.md`). They
are two runs, not one run with `--t_band_multi`: that flag classifies an item
by its `text`, and a grid item's `text` is every cell joined, so a grid of
single kanji lands in the multi band (`src/train/stage.py::_remap_band`).

## Step 1a — single-glyph table

`step1_0921`'s recipe, unchanged (`rows_step1_0921_s30k`; the read that set
the budget).

Data — scene half + grid half, the share is the item ratio:

```
--stage data --arm rows --data_tag <tag> \
--scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic --single_scenes s1,s1w --single_max_ar 2 \
--units kana --units 'kana_ext*1' --units small --units 'kanji:200*1' \
--units 'list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・*1' \
--scene_mix single=1.0 --n_items <N> --scene_frac 1.0 --natural_frac 0 --strings_frac 0 --flat_bubble 1.0 \
--scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 1 --scene_vertical 1 \
--shapes 448,512:2,448x512,512x448 \
--grid 2x2,3x3,2x3,3x2 --n_grid <N> --grid_unit_min_glyph 56 --grid_mark_horizontal 1
```

Train — cold, plain FM:

```
--stage train eval --arm rows --data_tag <tag> --shapes 448,512:2,448x512,512x448 \
--train_steps <80 × rows> --save_every 5000 --batch 4 --t_min 0.7 --t_max 0.9 \
--compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
--lr_rows 1e-3 --lr_decay cosine --free_residual 1e-3 --box_share 0.25 --pair_loss 0 --c_flat 0 \
--seeds 2 --no_floor --arm_tag <tag>
```

Kanji beyond the 200 go in as a `list:@<file>` unit source (the ranked cold
list, kanji rows only), same recipe.

## Step 1b — multi-glyph table

Scene composites only, one unit per item, σ 0.5–0.7. Cold, or warm from an
existing table for rows already trained at 0.7–0.9 (`--init_rows <table>
--init_anchor 0 --lr_warmup 500`; warm beat cold on every ruler at equal
draws, `micro_warm_0923`).

Data — the step-1a scene half, units from a piece list:

```
--stage data --arm rows --data_tag <tag> \
--scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic --single_scenes s1,s1w --single_max_ar 2 \
--units 'list:@<pieces>.txt*1' \
--scene_mix single=1.0 --n_items <N> --scene_frac 1.0 --natural_frac 0 --strings_frac 0 --flat_bubble 1.0 \
--scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 1 --scene_vertical 1 \
--shapes 448,512:2,448x512,512x448
```

Train — step 1a's argv with the band and the budget changed:

```
… --t_min 0.5 --t_max 0.7 --train_steps <≈ 190 × rows> [--init_rows <table> --init_anchor 0 --lr_warmup 500] …
```

Budget: exact and native follow scene draws monotonically with no knee
(40 / 80 / 188 steps per row → exact 6 / 8 / 13 of 32, native both-hit
4 / 8 / 17 of 128, warm from the merged table). 190 is the read of record;
80 buys about 60 % of it. No grid: a grid half bought nothing for these rows
at this band — at 40 steps per row it read below scene-only (3 vs 6 of 32),
at 80 level on exact (7 vs 8) and behind on native (both-hit 4 vs 8 of 128),
with the same row movement (`warm_cos` 0.87 vs 0.85). Grid cells are 85–200
px glyphs, a different size regime from the 35 px composites; whether
multi-glyph grids come back at their own band was `plan_band.md` B.2 — not
run (B.1 kept the high band for singles); the ceiling puts a grid cell one
step above a flat glyph of the same px (`band_experiment_results.md` § 2).

Piece list format is `ja_cold_0001_1900.txt`'s (`piece  count  class  glyphs
ext_id  src_rank`, `#` header); the data stage reads the first column.

## Merge

```
.venv/bin/python project/cjk_renderable_anima/src/probe/merge_tables.py <1a> <1b> → rows_<tag>_merge
```

Rows are rescaled to one `row_scale` on the way in. On an ext id present in
both, the base (first) table's row is kept by default; `--on_overlap override`
gives the row to the later table — that is the switch for replacing a
0.7–0.9 multi-glyph row with its 1b retrain.

## Step 2 — the sentence run

`rows_step2_0921m_band2_mu01_30k_0922`'s argv as it ran on 2026-09-22
(job `20260922-164130-5a2661`), warm from the step-1 table.

Data — `data_step2_0921m`, merged inventory + Manga109-s dialogue:

```
--stage data --arm rows --data_tag <tag> \
--scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic --single_scenes s1,s1w --single_max_ar 2 \
--units kana --units 'kana_ext*1' --units small --units 'kanji:200*1' \
--units 'list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・,@<pieces>.txt*1' \
--phrase_file <manga109s>/derived/dialogue_2_10.tsv --phrase_min_pieces 2 --n_phrase_eval 8 \
--scene_mix single=0.1,short=0.5,sentence=0.4 --short_pieces 2-5 --short_max_lines 1 \
--short_lexical 0 --phrase_norm 1 --text_draw balanced \
--sentence_min_letters 6 --sentence_min_glyph 20 --sentence_fill 0.9 --scene_vertical 1 \
--n_items 40000 --scene_frac 1.0 --natural_frac 0 --strings_frac 0 --flat_bubble 1.0 \
--scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 2 --shapes 512
```

Train:

```
--stage train eval native --arm rows --data_tag <tag> --shapes 512 \
--init_rows output/wake_probe/rows_<step-1>_merge/trained.pt --init_anchor 0.1 --lr_warmup 500 \
--train_steps 30000 --save_every 5000 --batch 4 \
--t_min 0.7 --t_max 0.9 --t_band_multi 0.35,0.7 \
--compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
--lr_rows 1e-3 --lr_decay cosine --free_residual 1e-3 \
--box_share 0.05 --box_share_cap 0.25 --c_flat 0 --pair_loss 0 \
--eval_groups single,single_ext,single_small,single_kanji,single_extra,short,short_held,phrase,phrase_held,en \
--native_chars あ,か,す,日 --native_clauses en,swap --seeds 2 --delta_parts full --no_floor --arm_tag <tag>
```

Step 2's data has no grid items, so `--t_band_multi` is safe here. What it
does to the rows: the anchor's cosine tail returns most rows to the seed
(`warm_cos` 0.978 at 2.5 k → 0.996 at 30 k); the rows that move are the
frequent particles and punctuation (148 of 2 271 below cos 0.99). Multi-glyph
rows the sentences rarely draw stay where step 1b left them — which is why
step 1b, not step 2, is where their band matters.

## Reads

- Step 1: `single_*` exact from the run's own eval, split by glyph count;
  `native` (`--native_chars` = trained rows, `--native_clauses en,swap`,
  2 seeds); `cf_sense --cf_rows piece` for multi-glyph tables.
- Step 2: pooled sub-exact lift with the `_held` groups
  (`src/probe/sub_exact.py`), single-glyph native, multi-glyph native
  (`--eval_tag sent`), `row_dose.py` (`sent_run.md` *Reading rules*).
- Any table: `src/probe/table_geometry.py` for row geometry vs its seed.

## Open

- Grid for multi-glyph rows: closed at 0.5–0.7 (40 and 80 steps per row both
  read at or below scene-only). Reopens only through a new band cell if grid
  cells turn out to want their own band.
- Step 1b budget between 80 and 190 steps per row has one point each; the
  curve has no knee in the four reads so far (40 / 80 / 188: 6 / 8 / 13).
- **Settled 2026-09-23** (`band_experiment_results.md`): the band is keyed
  on glyph count — singles 0.7–0.9 (B.1: 24 kana on 48 px composites,
  native 84 vs 49 of 192), pieces 0.5–0.7 — and rendered px sets the floor
  a band may reach (a 16 px string lives at 0.25–0.6), grid cells one step
  higher. Step 1a and 1b stand as written. Single composites are 48–53 px
  as built, not 38.
- Step 2 μ (0.1 vs 0.3) and length are the sentence run's open knobs
  (`project/cjk_renderable_anima/plan.md` § 2).
