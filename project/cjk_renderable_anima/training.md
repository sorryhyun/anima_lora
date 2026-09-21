# training — how the table is trained today (2026-09-21)

Index: [`README.md`](README.md). What is planned next is [`plan.md`](plan.md);
why each choice was made is [`findings.md`](findings.md) and
[`reports/`](reports/README.md); the shipped file form is
[`deploy_plan.md`](deploy_plan.md). This file is only the method in use. The
earlier method write-up (`synth.md`: the ΔFM loss, `c_flat`, flat items, the
`step1_0920` / `step2_0919` argv) is archived under
`_archive/cjk_renderable_anima/`.

## What trains

One table of **ext-row deltas** over the vocab pack: `Δ_r = raw_r · row_scale`,
added to the pack row of ext id `r` at lookup by the `ExtDelta` hook on
`llm_adapter.embed`. The DiT, the adapter and the text encoder are frozen; there
is no encoder, no LoRA and no runtime gate. A row is a **Qwen piece** — one
kana, one kanji, or a multi-glyph piece (って, じゃない, ありがとう) — and it is
trained so that the clause `Japanese text reads as "<piece>"` draws it. The
arm is `--arm rows`; its output is `output/wake_probe/rows_<data_tag>_<arm_tag>/trained.pt`
(`delta.ext_ids`, `delta.raw`, `delta.row_scale`).

Everything runs through one entry point, stage by stage:

```
project/cjk_renderable_anima/src/wake_probe.py --stage scenes | data | train | eval | native
```

> **Every launch states its pack:**
> `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` (raw, log sha
> `7b9fce0bb57b` — read it off the job log). `configs/base.toml` defaults to the
> preview pack, which already carries a trained delta. GPU stages go through the
> daemon (`make daemon-run ARGS="--label … --stall-timeout 0 --queue <argv>"`);
> the data stage is CPU.

## 1. Scene pools (`--stage scenes`, built once, reused)

The base model draws its own scene from a dataset-format caption with `speech
bubble`, `english text` and a short EN anchor (`English text reads as "hi"`).
A judge keeps the scene when the detector finds the anchor and nothing else,
flood-fills the bubble and records its usable region. Pools on disk: **`s1`
233, `s1w` 380, `sl1w` 213, `ja_comic` 292** scenes (`output/wake_probe/scenes_<tag>/`,
1.6 GB). No current table renders a new pool.

The 8 native eval prompts are held out of the prompt vocabulary — except
`comic`, which `ja_comic` puts in every caption, so native prompt 8 is read
separately.

## 2. Data (`--stage data`, CPU)

Two kinds of item, 50 : 50 by item count — the share is the item ratio, there
is no flag:

- **Scene singles** — one unit composited into an existing scene's bubble: the
  anchor is erased inside the bubble's flood interior, the unit is drawn with a
  Noto CJK font fitted to the region (`--scene_fill 0.7`, `--scene_min_glyph 28`),
  multi-glyph units as one column (`--scene_vertical 1`, cap 5 glyphs). Caption
  = the scene's own tags with `english text` → `japanese text`, then
  `Japanese text reads as "<unit>"`. The only thing the caption does not
  already explain is the glyph in the bubble. One-glyph units use the `s1,s1w`
  pools (`--single_scenes`, `--single_max_ar 2`).
- **Grid items** — k units drawn in a `2x2 / 3x3 / 2x3 / 3x2` grid (512², 512²,
  416×624, 624×416) on a flat canvas or, for half the items, in bubbles;
  caption = the frame tags + one position clause per cell in reading order
  (`On the top left, Japanese text reads as "…".`). Units are **dealt** from a
  shuffled deck, so every unit is drawn once per pass and never twice in an
  item.

Multi-glyph units in a grid:

- `--grid_unit_min_glyph 56` — the deck's next unit picks a (grid, frame) whose
  cell holds its glyph count (one glyph anywhere, a digraph in 3x3, 3 glyphs in
  2x2 or a flat 2x3 / 3x2, 4 – 5 in a flat 2x2); the other cells take what fits.
- `--grid_mark_horizontal 1` — a cell drawn as a left-to-right line says
  `horizontal Japanese text reads as "…"`; a column keeps the bare clause.

The inventory is `--units`, one source per flag: `kana`, `kana_ext*1`, `small`
(the 18 small kana, each inside up to 6 two-glyph digraphs — a lone ゃ renders
full-size), `kanji:N*1`, and `list:` — literal units and / or `@<file>` under
`assets/units/` (one unit per line, first column). Every `list:` unit is
asserted to be one Qwen piece with a pack row. Weights are uniform (`*1`).

A large build is **sharded**: the same argv with `--seed k` and its own
`--data_tag`, run side by side (`systemd-run --user`, one CPU each), then
`train.jsonl` concatenated into the dir the run names. Records hold absolute
paths, so the joined dir holds no images and the shard dirs must stay (and
travel) with it. The data stage records no argv: write `data_<tag>/args.json`
by hand.

`step1_0921z`, per shard (10 shards, 5 100 + 5 100 items each, ≈ 20 min on 12
cores):

```
--stage data --arm rows --data_tag step1_0921z_p<k> --seed <k> \
--scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic --single_scenes s1,s1w --single_max_ar 2 \
--units 'list:@ja_cold_0001_1900.txt*1' --scene_mix single=1.0 --n_items 5100 --scene_frac 1.0 \
--natural_frac 0 --strings_frac 0 --flat_bubble 1.0 --scene_fill 0.7 --scene_min_glyph 28 \
--scene_max_lines 1 --scene_vertical 1 --shapes 448,512:2,448x512,512x448 \
--grid 2x2,3x3,2x3,3x2 --n_grid 5100 --grid_unit_min_glyph 56 --grid_mark_horizontal 1
```

**Check before training**, off the joined `train.jsonl`: items per unit for the
scene half and the grid half (the scene half is a weighted draw, not a quota);
the grid line of the data log (grid shares, ink height); and that every unit's
own ext id is among its caption's tokens with no other ext row beside it. Then
look at the sheets.

## 3. Train (`--stage train`)

| knob | value | why |
|---|---|---|
| loss | plain flow matching, `--pair_loss 0` | the paired loss (ΔFM) held the scene and lost identity at full inventory |
| in-box weight | `--box_share 0.25` | per item `s·mean_in + (1 − s)·mean_out`, `s = min(0.25 · n_glyphs, 0.75)` — the row's share of the loss does not depend on the box area. Scene items only; grid items carry per-cell boxes and train unweighted |
| σ band | `--t_min 0.7 --t_max 0.9` | identity is decided at σ ≈ 0.8 |
| lr | `--lr_rows 1e-3 --lr_decay cosine` | cosine: a run cannot be stopped early, size it first |
| anchor to zero | `--free_residual 1e-3` | sets the end row norm together with the in-box share |
| batch | 4; a batch is one (shape, source) | compute-bound at 4 on 16 GB and on 96 GB alike |
| budget | **80 steps per row**, cold | `step1_0921`: 374 rows, 30 k |
| speed / safety | `--compile 1 --grad_ckpt 0 --aggressive_recompute 0 --save_every 5000` | `trained_partial.pt` is crash insurance; rename to `trained.pt` to read it |
| off | `--c_flat 0`, no flat items, no encoder arm | closed in `findings.md` |

The first train on a data dir encodes every caption (prompt embeds stream to a
temp file, ≈ 1 MB per caption — `TMPDIR` picks the disk) and caches the latents
as `data_<tag>/latents_*.pt` (≈ 5 GB per 20 k items, held in RAM while
training). `step1_0921` as run:

```
--stage train eval --arm rows --data_tag step1_0921 \
--units kana --units 'kana_ext*1' --units small --units 'kanji:200*1' \
--units 'list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・*1' \
--grid 2x2,3x3,2x3,3x2 --scenes s1,s1w,sl1w,ja_comic --shapes 448,512:2,448x512,512x448 --n_grid 10000 \
--train_steps 30000 --save_every 5000 --batch 4 --t_min 0.7 --t_max 0.9 \
--compile 1 --grad_ckpt 0 --aggressive_recompute 0 \
--lr_rows 1e-3 --lr_decay cosine --free_residual 1e-3 \
--box_share 0.25 --pair_loss 0 --seeds 2 --no_floor --c_flat 0 --arm_tag s30k
```

`train_log.json` carries `in_box` / `out_box` (the mixed `loss` is ≈ 80 %
out-of-box and cannot see a row) and `delta_norm_mean` — the norm peaks early
and settles where the in-box gradient balances the anchor.

## 4. Read (`--stage eval`, `--stage native`)

- **eval** renders each eval string in the bare template (`manga, speech
  bubble, japanese text. Japanese text reads as "…"`), 2 seeds, and reads it
  with both readers (SFX reader + VL16): `report.md` (exact per group:
  `single`, `single_ext`, `single_small`, `single_kanji`, `single_extra`,
  multi-glyph groups, `en`) and one sheet per group. Compare `single_ext` only
  between runs that share an `eval.json`.
- **native** renders the unit inside 8 held-out scene prompts, under the trained
  JA clause (`en`) and the EN reference's caption with the word swapped
  (`swap`): `--native_chars あ,か,す,日 --native_clauses en,swap --seeds 2
  --delta_parts full`. The count is **`both`** (both readers exact, of 64);
  the scene is read by **`en cos`** (PE-Spatial cos to the `English text reads
  as "hi"` render of the same prompt and seed) and placement by `box IoU`.
  `--native_floor` is the old ruler and is not used.
- Multi-glyph groups are ordered by `src/probe/sub_exact.py` pooled lift (exact
  is ≈ 0 everywhere), per-row gains by `src/probe/row_dose.py` against a seed
  read. Always look at the sheets: repeats (は → はは), first-glyph-only reads
  of a multi-glyph piece, and a wiped scene do not show in a count.

`step1_0921`: `single` / `_ext` / `_small` / `_kanji` 28 / 20 / 10 / 19 of 36,
native 34 / 28 of 64, en cos 0.903.

## 5. Merge, sentence run, bake

- **Merge** tables with disjoint ext ids:
  `src/probe/merge_tables.py --base <arm>/trained.pt --add <arm>/trained.pt --out <arm name>`
  — rows are rescaled by `row_scale_src / row_scale_base`, so each row applies
  exactly the delta its own run trained. The result is an ordinary arm dir.
- **Sentence run** — the merged table warm-started on real multi-piece lines:
  `--scene_mix single=0.1,short=0.5,sentence=0.4` from a phrase file
  (`--short_lexical 0 --phrase_norm 1 --text_draw balanced`), a line being
  usable only when every piece has a warm row; train with `--init_rows <merged>
  --init_anchor 0.3 --lr_warmup 500` (without the warmup and the anchor Adam
  erases the warm start within 50 steps), σ 0.5 – 0.9, `--box_share 0.05
  --box_share_cap 0.25`, plain. **No cold row inside a sentence run.** Full argv
  and sizing: `plan.md` § 2; every run so far: [`sent_run.md`](sent_run.md).
- **Bake** the final table into a pack pair:
  `scripts/toolkits/bake_vocab_pack.py <arm dir> --out models/vocab_packs/<name>`
  ([`deploy_plan.md`](deploy_plan.md)).

## Cost (5070 Ti 16 GB)

| stage | cost |
|---|---|
| scenes, 1 000 images | ≈ 62 min, ≈ 20 % kept |
| data | ≈ 23 items/s per process; 102 k items in ≈ 20 min over 10 shards |
| train, grid mix, batch 4 | ≈ 2.35 it/s — 30 k steps in 3.5 h (RTX PRO 6000: 5.6 it/s on the DiT bench, plan on ≈ 5.3) |
| sentence run, 6 k steps | ≈ 45 min |
| eval / native | ≈ 20 min / ≈ 10 min |
