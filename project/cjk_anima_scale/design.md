# design — the scale pipeline (rough, 2026-09-23)

The production line that builds the JA pack at scale. It replaces
`recipe.md`'s step 1a / 1b / merge / step 2 with **one loss, one trainer,
and a band schedule** whose stages differ only in their dataset. The probe
code in `project/cjk_renderable_anima/src/` stays the research surface;
this line takes its render primitives and leaves its levers behind.

Status: sketch. Numbers it leans on: `band_experiment_results.md` (the
per-px window table, the count rule), `plan_kanji.md` (deleted; git `f5cd4c0c`; C.1 read in
`reports/cf_kanji_c1_2026_09_23.md`, C.2 in `reports/band_c2_kanji_2026_09_23.md`).

## 1. Principles

- **Plain FM loss only.** No pair loss, no `c_flat`, no CF input, no
  encoder arm, no out-vec, no free residual as a lever (a fixed constant if
  kept at all). Rows + the box-weighted FM loss that every read of record
  used.
- **The band is a property of the stage, not of the item.** A stage trains
  one σ band; the dataset is built so every item in it *wants* that band.
  `--t_band_multi` and `_remap_band` go away — band purity is the data
  builder's job.
- **Simple core, complex data.** Trainer / eval are thin; the data builder
  is where the design lives (§ 4).
- **Warm chain.** Each stage starts from the previous stage's table
  (`init_rows`, anchor μ), so a later band cannot wipe what an earlier one
  bought. Cold rows exist only in the first stage they appear in.

## 2. The schedule

Named by band, `stage<lo><hi>`:

```
stage0709  →  stage0507  →  stage0305  →  stage0309
 (warm from the merged step-1 table)          (warm, consolidation)
```

The chain starts from the merged step-1 table (`step1_0921` + `step1_0921z`,
≈ 2 300 rows, the 1 900 cold ranks already trained at 0.7–0.9), so no stage
is cold: `stage0709` refines rows that already carry identity, and a new
row enters cold only when the inventory grows.

Easy-to-hard: high σ first (the glyph-identity band), then lower bands
(the count / layout / small-text bands), then one accumulate pass over the
whole range so the rows compose (the role `recipe.md`'s step 2 played).
Whether the last stage is 0.3–0.9 accumulate or per-stage tables merged is
`band_experiment_results.md` § 6 item 1 (switch vs accumulate) — unread.

What each band is for, from the window table (§ 2 of the results) and the
count rule (§ 3):

| stage | band | what trains there | why |
|---|---|---|---|
| `stage0709` | 0.7–0.9 | single glyphs ≥ 48 px: scene bubble fit (48–53 px), grid cells (85–200 px), large flat | singles want the top half (B.1); grid cells sit +0.1 (ceiling 0.8–0.9) |
| `stage0507` | 0.5–0.7 | multi-glyph pieces 32–48 px in scenes; single glyphs 24–32 px; 2–5-piece short lines ≈ 32 px | pieces 0.5–0.7 (`micro_cf_0922`); 24–32 px letter window 0.35–0.7; string at 48 px 0.6–0.8 |
| `stage0305` | 0.3–0.5 | 12–24 px text: sentences with `scene_min_glyph 16`, small-bubble dialogue, small grid strings | 16 px string 0.25–0.6 centred 0.4 (A.2); grid at 12–16 px 0.3–0.5 |
| `stage0309` | 0.3–0.9 | everything, mixed, at natural sizes | consolidation; the sentence run's role, low μ anchor |

Open on this table: **dense kanji** (C.2 agrees with C.1: no band of their
own — kanji take the kana band — but at 48 px scene-only they read 0 / 64
native; whether that is px (grid cells in `stage0709`) or exposure (a
density-weighted draw count) is the `band_experiment_results.md` § 6 item-3
cell); **0.8–0.95 is dead at 48 px** (C.2), so no stage goes above 0.9;
whether `stage0305` needs a `stage0103` below it (nothing has been read
under 0.25). Complexity descriptors (ink, straightness) are budget
weights at most, not `windows.py` terms (C.2 report, last section).

## 3. Layout of the line

```
project/cjk_anima_scale/
  design.md            this file
  recipe.md            retired into configs/ once the stages run
  scale.py             front door: --stage <s> --tag <t> --steps data train eval bake [--submit]
  configs/
    stage0709.toml     band, gate, warm chain, recipe mix, trainer surface, eval
    stage0507.toml
    stage0305.toml
    stage0309.toml
  cjk_scale/           one package (see below for why not `src/`)
    paths.py           output dirs: output/wake_probe/{data,rows}_scale_<stage>_<tag>
    windows.py         (kind, px, layout) → training band; the law as a row table with provenance
    config.py          configs/<stage>.toml → StageConfig
    recipes.py         the item generators (§ 4)
    builder.py         stage config → data dir, the band gate, the ± 20 % px gate
    rows.py            the ExtDelta table, warm chain, anchor
    train.py           rows-only plain FM, one band
    eval.py            exact / native / cf_sense via the probe stages + the regression check
    bake.py            table → pack pair
    ledger.py          runs/ledger.jsonl
  tests/               line-local (imports, the law's rows, configs, the chain)
  runs/                ledger.jsonl: stage, tag, steps, argv, job id per submit
```

Render primitives (scene compositor, grid, flat, ink, fonts), the inventory
resolvers, `LatentStore` / `Batcher` / the box-share loss and the eval
stages are imported from `project/cjk_renderable_anima/src/` — not copied.
That `src/` puts its packages on `sys.path` as top-level names (`common`,
`data`, `train`, `eval`), so this line's code is the `cjk_scale` package,
not a second `src/` (a `src/train.py` here would shadow the probe's
`train/` the moment both were on the path). Outputs keep the probe's
`output/wake_probe/` layout with a `scale_` prefix so every probe reader
opens a stage table unchanged.

## 4. The data builder — where the complexity is

A stage dataset is a **mix of recipes**; a recipe is an item generator with
a declared band window per px. The builder's contract:

```
stage band B
for each recipe r in the stage's mix (with its share):
    draw px / layout / unit for an item
    w = window(r.kind, px, r.layout)          # from windows.py
    keep the item iff B ⊆ w (or |B ∩ w| / |B| ≥ 0.8)
    re-draw otherwise (px is the knob that moves w)
```

So a recipe does not need to know the stage; the window table does, and
the same recipe (`scene_single`) feeds `stage0709` at bubble-fit px and
`stage0507` at 24–32 px because the builder re-draws px until the window
contains the band.

Recipes (names are placeholders):

| recipe | source | px control | unit kinds |
|---|---|---|---|
| `scene_single` | bubble pool, one glyph per bubble | bubble fit (48–53) or a `--scene_glyph_px` cap for the small end | kana / kanji / punct rows |
| `grid_single` | **1×1 … 3×3** grid, one glyph per cell; **1×1 is the flat single** (`layout v1` / `jitter` retired into it) | `grid_fill` 0.15–0.8 of the cell short side — 60–400 px at 1×1, 25–85 at 3×3 | same |
| `scene_piece` | bubble pool, one multi-glyph unit | fill 0.7 (≈ 35 px) up to 1.0 in large bubbles (≈ 48) | piece rows |
| `scene_short` | 2–5 pieces, one line | ≈ 32 px | corpus short lines |
| `scene_sentence` | Manga109-s dialogue | `scene_min_glyph` 16–28 | corpus lines |
| `grid_string` | strings in cells (the A.2 cell) | 12–24 px | pieces / short lines |

Caption follows the cell count, not the recipe: 1×1 takes the plain /
bubble template (the W-line flat reads — "identity at σ 0.8", `--t_max 0.6`
losing on flat singles — were made under it), 2×2 and up take
`grid_caption`'s per-cell position clauses. The ellipse (`flat_bubble`) is
a 1×1 option, not a recipe: ellipse = flat to the second decimal at every
px (A.1).

`windows.py` is the § 2 ceiling table + the count rule + the grid shift,
as a function; it is also the thing that has to be *re-read* when the table
changes (a new px, kanji strata, a new layout), so it carries its provenance
per row (which report, which cell).

Per-stage builder outputs a `train.jsonl` whose every row records `kind`,
`px`, `ink`, `layout`, `window`, and a `sheet` per recipe — the ± 20 %
gate on median px / ink per recipe runs before a train job is submitted.

## 5. Trainer and eval (thin)

Train args, the whole list: `stage`, `data_tag`, `train_steps` (or
`steps_per_row`), `batch`, `lr_rows`, `lr_decay`, `lr_warmup`, `t_min`,
`t_max`, `init_rows`, `init_anchor`, `box_share`, `box_share_cap`,
`compile`, `seed`, `arm_tag`. Everything else in
`cjk_renderable_anima/src/cli/train.py` is a probe lever and is not
carried over (pair_*, c_flat*, out_vec*, cf_input, free_residual,
t_band_multi, the encoder arm, kill_*, decor, init_encoder, font_mode,
held_out*, row_boost*).

### Budget

**30 steps per row per stage** (`stage0709` / `stage0507` / `stage0305`:
30 / 30 / 30; `stage0309` to set). Steps per stage = 30 × rows; at ≈ 2 300
rows and batch 4 that is ≈ 70 k steps, ≈ 12 h at `step1_0921`'s rate
(30 k in ≈ 5.5 h). Draws per row = steps × batch ÷ rows = 120 item draws
a row, of which the recipe mix decides how many are scene draws.

This is far under the cold-row knee (`exposure_ledger_2026_09_21.md`:
identity 0.78 at 250 scene draws a row on a small table, ≈ 1 000 at 374
rows; `step1_0921` spent 80 steps a row = 160 scene + ≈ 1 000 cell draws)
and is chosen because the rows are warm — the stage refines a band, it does
not buy identity. Whether 30 is enough per band is read on the first run
of each stage (the per-stage regression check in § 5 and the stage's own
exact / native), not assumed.

Eval per stage: `exact` per unit group, `native` on a fixed row sample
(`en` / `swap`), `cf_sense --cf_lang ja` on the stage's own rows at the
stage's px (does leverage land in the band it trained in). Advancing to
the next stage also re-runs the previous stage's exact groups on the new
table — the warm-chain regression check.

## 6. Open questions (decide before code)

1. **Switch vs accumulate** for the last stage (`stage0309` accumulate vs
   merge of per-band tables) — the § 6 item-1 cell, warm from a 0.7–0.9
   table, 0.5–0.7 vs 0.5–0.9.
2. **Anchor μ per stage** — step 2 used 0.1 (rows return to the seed,
   `warm_cos` 0.996); a lower band needs the rows to *move*, so the early
   stages may want μ 0 and only `stage0309` an anchor.
3. **Dense kanji** — pending C.2 (`bk_mid` / `bk_hi`); the answer sets
   whether `kanji:N` splits by ink into two recipes or one.
4. **px per item vs per recipe** — the builder above draws px per item; a
   simpler builder fixes px per recipe per stage. Per item is what makes
   one recipe serve two stages.
5. **Corpus for `stage0305`** — 16 px text needs long lines in small
   bubbles; Manga109-s dialogue is the pool, the count of usable lines is
   unknown.
6. ~~**Import direction** for the render primitives (§ 3).~~ Decided
   2026-09-23: this line imports the probe's `src/` (§ 3); nothing moves.
7. **Is the grid shift the caption or the context?** The +0.1–0.2 σ a grid
   cell carries at the ceiling (A.1) was read on 3×3 with the grid caption.
   Whether a 1×1 gets it decides if `windows.py` applies the shift by cell
   count or by caption. One EN `cf_sense` cell (1×1 with `grid_caption` vs
   the plain template, ≈ 1.5 min) reads it.
