# cjk_anima_scale — the JA vocab pack at scale

Opened 2026-09-23 out of [`../cjk_renderable_anima/`](../cjk_renderable_anima/).
That line is the research surface (probe code, `reports/`, `findings.md`);
this one is the production line that builds the pack on what it settled:
one loss, one trainer, σ per item from the band law. The **singles retrain**
(2026-09-28 – 09-30) produced the current seed; its plan and the plans after
it (`plan.md`, `plan_retrain.md`, `plan_polish.md`, `hypothesis.md`,
`idea.md`, `plan_garble_replace.md`) are under `_archive/` with what came
before — the stage chain, the 300-piece runs, line mode, plan_2900 (see its
README). The open question is [`proposal_seed_synthesis.md`](proposal_seed_synthesis.md)
(`proposal_length.md` ran its step 1 and is superseded).

## Finished (2026-10-09)

Moved from `project/` to `finished/` on 2026-10-09. Cold training continued
in [`../../cjk_anima_reseed/`](../../cjk_anima_reseed/README.md), which ported
the trainer (`cjk_scale/{train,rows,loss}` → `reseed/{trainer,rows,loss}`),
vendored this line's `src/` subset and fonts, and imports nothing from here;
a cold-batch capability reseed lacks is ported into its table / recipes, not
run from this tree. This tree is frozen and runnable by path; its outputs
stay at `output/cjk_anima_scale/`, which reseed reads (the seed rows, the
scene pools, the EN refs).

## The vocab band law

[`band_experiment_results.md`](band_experiment_results.md) is the **vocab
band law**: which σ band a vocab-pack row trains in, keyed on what. It is not
a theory — every line of it is a measured read (EN ceiling on the base model,
JA training arms with two seeds), and it holds only over the sizes, layouts
and units those reads covered. As it stands:

- **The band is keyed on the row's glyph count.** Single-glyph rows train at
  0.7–0.9; multi-glyph one-token rows at 0.5–0.7. The two are two runs (or
  two per-item bands), never one band.
- **Rendered px sets the floor the band may reach**, not the band itself
  (§ 2, the per-px window table: 12–16 px text lives at 0.2–0.6, 48 px at
  0.5–0.7, 128 px at 0.8). A grid cell sits one step higher.
- **Nothing above 0.9** — 0.8–0.95 is dead at 48 px for kana and kanji alike.
- **Ink, stroke density and the bubble ellipse move no band.** Kanji take the
  kana band; density is an exposure / px question (§ 6 item 3, open).
- **16 px glyphs carry more caption leverage than 24 px**, lower in σ — small
  text is a band question, not a capability question.

The plans that produced it (`plan_band.md`, `plan_kanji.md`) closed on
2026-09-23 and were deleted; they are in git history at `f5cd4c0c`. The
probe line's step 1a / 1b / merge / step 2 recipe (`recipe.md`) was retired
the same day — `configs/stage*.toml` carry its settings, the band law § 3–4
its reads (git `ff2f70f9` has the last copy). The
reads themselves are the dated reports under `../cjk_renderable_anima/reports/`
(`cf_rebin_gate0`, `cf_band_a1`, `band_b1`, `cf_kanji_c1`, `band_c2_kanji`).
A new read that changes a row of the law goes into
`band_experiment_results.md`, with its report there.

## Files

| file | what |
|---|---|
| [`proposal_seed_synthesis.md`](proposal_seed_synthesis.md) | **the open question** (2026-10-01, evening): the rows decide what fills the text region, the base decides the region — `span_reband` (the word-length windows at 0.85–0.95 refill the banner with smaller glyphs), `shared_dir` on it (all of it above 0.8), `inject_count` (the slot count commits between σ 0.95 and 0.9, no `cf_sense`); the size-bias objection and the answer to it (no common size in the item set); the proposal: the seed's own renders with the word redrawn to fill the base's region, at 0.85–0.95 |
| [`proposal_length.md`](proposal_length.md) | superseded (2026-10-01): the repeats are leftover slots — the concept so far (attack the base's own confused text region), what the garble arms closed, and the two levers that follow (span at σ ≥ 0.9 first — the seed's word items rebanded — count-CF at 0.8–0.9 second), with a no-training traj read first |
| [`hypothesis.md`](hypothesis.md) | (2026-10-01) a row is one vector at every σ: an item trained above where its glyphs resolve teaches only its layout, which then overrides every caption's — band and px are one choice; three predictions |
| [`reports/band_size_2026_10_02.md`](reports/band_size_2026_10_02.md) | (2026-10-02) band × glyph size: the seed's bands were right for its sizes and no word item could train above 0.7; `kana_reband` (the kana run's hiragana items at 0.75–0.93, cold: nothing reads, confounded), the `--b0305` mirror (the base above 0.8, the rows below: 0–1 / 16), the P1 arms re-read for rendered size, `grid_small` (running); the target is manga-size dialogue |
| [`reports/grad_identity_2026_10_02.md`](reports/grad_identity_2026_10_02.md) | (2026-10-02) the band law read on the training gradient, no training: with a cold row fixed and the glyph drawn in its slot swapped, the glyph-dependent share of the gradient falls with σ, earlier the smaller the glyph — half point 0.62 / 0.72 / 0.76 for 15 / 28 / 44 px grid cells = the EN ceiling's upper edge; identity peaks at σ 0.4 / 0.5 / 0.6; below the peak exposure runs out, not identity; 0.75–0.93 is ≥ 90 % glyph-independent for 15–28 px; a lone 1×1 gets 3–10× less identity per draw than a grid cell. The row-space design (true row vs other rows in the slot) is null by construction |
| [`findings.md`](findings.md) | (2026-10-01) where the text is decided on the trajectory: identity at σ 0.85–0.7 when the model knows the glyphs (small EN on the base), the base banner's ≈ 6 slots, the `japanese text` tag's text area, ！ in the leftover slot, 3 × 3 EN / kana / kanji grids, the seed's b0305 captions with uncond below 0.8 — against b0305 / b0507 |
| [`retrain_experiments.md`](retrain_experiments.md) | the retrain's record (2026-09-28): why the singles re-seed cold (P1 / P1b), per-glyph routing (P2), the windowed word pool, checks C0–C3, `retrain_kana` trained and read, the code that landed |
| [`band_experiment_results.md`](band_experiment_results.md) | **the vocab band law** — the verdict, the per-px window table, the training reads, what is left unrun |
| [`floor_score.md`](floor_score.md) | the floors on sent / target / word / en: the new seed's (`seed_retrain_0930`, what runs read against) and the old seed's (what the retrain was read against) |
| [`product_criteria.md`](product_criteria.md) | what a pack has to do to ship: the text axis and the page axis, dev set vs acceptance set |
| [`colab.md`](colab.md) | running `data` / `train` on a Colab VM (G4 = the kanji batches) |
| [`future.md`](future.md) | not planned: real images do not train rows, an OCR-reward update, token scaling as the last stage |
| `reports/` | the reads the live code cites: [`conflict_joint_2026_09_25.md`](reports/conflict_joint_2026_09_25.md) + [`grid_box_2026_09_25.md`](reports/grid_box_2026_09_25.md) (the trainer constants), [`piece_2026_09_25.md`](reports/piece_2026_09_25.md) (the piece ruler), [`long_b0_2026_09_27.md`](reports/long_b0_2026_09_27.md) (the long-piece budget row), [`stage_i_2026_09_26.md`](reports/stage_i_2026_09_26.md) (`b0709`, the cold-kanji budget), [`row_geometry_2026_09_28.md`](reports/row_geometry_2026_09_28.md) (the retrain rows in row space), [`sigma_split_2026_09_30.md`](reports/sigma_split_2026_09_30.md) (the rows gated by σ, x̂0 per σ: text is decided at 0.9–0.7), [`garble_replace_2026_09_30.md`](reports/garble_replace_2026_09_30.md) + [`delta_scale_2026_10_01.md`](reports/delta_scale_2026_10_01.md) (the garble arms; the repeats are leftover slots) |
| `configs/runs/*.toml` | the runs — `{vocabs, read[, context]}`: `retrain_kana`, `retrain_kanji_b1..b4`; `run0923_micro` / `run0925_300f` stay for the tests |
| `cjk_scale/` | the code (`windows` = the law, `config` = the run file + data pools, `recipes` + `builder` = data, `rows` + `train` = the fixed trainer, `budget`, `eval` = floor + trained on one sheet, `conflict`, `merge`, `ledger`); `scale.py` is the front door |
| `src/` | the stage packages the line runs on (render, readers, scoring, sheets, eval / native / target / cf_sense / scenes), vendored 2026-09-25 byte-faithful and pruned the same day; `src/run_stage.py` runs one by hand |
| `experiments/` | the retrain's experiments (`stage_b` stays as the module they import) — see its README |
| `assets/` | what `src/` reads: `fonts/` (binaries gitignored, `FONTS.md`), `vocabs/` (vocab files), `glyph_ink.json`, `target_prompts.txt` |
| `_archive/` | gitignored: the retired stage code, and the pre-retrain docs / reports / experiments / run files (see its README) |
| `runs/` | `ledger.jsonl` — every submitted job |

## Where it stands (2026-10-02)

`retrain_kana` (174 cold kana rows, routed) composes like `p1_mix` and holds
the singles; its rows are the kana half of the new seed.
`retrain_kanji_b1` (329 kanji on `retrain_kana`'s rows) trained on a Colab
G4, b2 and b3 locally (all unread); b4 (the training set's JA tail, the
first under the encode fold) trained 2026-09-30 and its rows are the new
seed, `output/cjk_anima_scale/seed_retrain_0930/` (`paths.SEED_ROWS`; the
old one is `SEED_ROWS_0921`). Its floor is in `floor_score.md` (acceptance
8 → 37 / 80), and it is baked with routing on and published as
`anima_cjk_vocab_pack_preview4` (Hub `sorryhyun/anima-vocab-pack-cjk`,
ComfyUI Adapter node ≥ 3.13.0). Details and numbers: `retrain_experiments.md`,
`_archive/plan_retrain.md`.

2026-10-01: the polish passes on the seed (`polish_seed`, `garble_replace`)
moved layout but not strings; the repeats are slots the word leaves over
(`proposal_length.md`). Evening: `span_reband` + `inject_count` — the rows
fill the base's region, they do not resize it; the count commits between
σ 0.95 and 0.9 (`proposal_seed_synthesis.md`).

2026-10-02 (`reports/band_size_2026_10_02.md`): the seed's bands were right
for its glyph sizes — every arm that moved a px above its ceiling window
lost identity, `kana_reband` (the kana run's hiragana items at 0.75–0.93,
cold) included — and the target is manga-size dialogue, not the `sent`
banner. Identity does not need large glyphs (`p1_cold`); `grid_small` (small
glyphs only, grids 60 %) is training. A reseed draft sits in
`../../cjk_anima_reseed/_archive/motivation.md` (archived 10-05).

## Running a run

```bash
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack   # MANGA109S comes from .env
.venv/bin/python project/finished/cjk_anima_scale/scale.py retrain_kanji_b2 data              # CPU
.venv/bin/python project/finished/cjk_anima_scale/scale.py retrain_kanji_b2 train --submit --queue
.venv/bin/python project/finished/cjk_anima_scale/scale.py retrain_kanji_b2 eval --submit
.venv/bin/python project/finished/cjk_anima_scale/scale.py retrain_kanji_b2 conflict --submit
.venv/bin/python project/finished/cjk_anima_scale/scale.py windows | runs | ledger
```

A run is `configs/runs/<run>.toml` = `vocabs` (a vocabs file: one vocab per
line) + `read` (the `native_sent` strings). `data` draws the items by the
vocabs' kinds, from the item pools of § Item pools (singles → the six
`lone` / `grid` / `bubble1` / `bubbleN` tiers; pieces → the six `piece_*` /
`line_*` tiers), ≈ 67 items per vocab, every item stamped with its tier and
its band; `train` trains the vocabs' rows from the seed rows
with every other row frozen at them, and saves `trained.pt` as the whole
merged rows — the seed's rows with the run's on top (μ 0, lr 1e-3, batch 4,
cosine, warmup 10 %, grid box, 90 steps/row — `cjk_scale/train.py`); `eval`
renders the floor (the seed rows' dir, a read cache shared by every run —
only the keys it lacks render) and the trained rows
(`trained.pt` in place — no `ctx/`) on
`word` / `single` / `en`, あ / い native, up to 8 trained pieces alone in a
native scene (`native_piece/`), the `read` strings and the target
captions, and writes one `sheet.png` + `reads.json`. Everything lands in
`output/cjk_anima_scale/<run>/`; `--submit` records the job in
`runs/ledger.jsonl`. The stage-shaped runs before 2026-09-25 stay on disk
as `{data,rows}_<stage>_<tag>/` records.

## Item pools

An item pool is a **tier** of `builder.TABLE`, named `<form>_<px>`: what is
drawn, and the median ink px (√(box area / glyphs)) its items were built at
on the kana run. The number is a fixed label (`builder.TIER_PX`), not
recomputed per build — a kanji run draws the large tiers larger
(`retrain_kanji_b4`: `grid_82` → 100, `lone_190` → 235); each build's own
px is `build.json` `tiers.<tier>.px_kept`. The name is the image prefix,
the record's `tier` and the `build.json` key.

Forms: **`lone`** one glyph alone on its canvas (1×1, bare or one bubble);
**`grid`** 2×2 – 3×3 cells, one glyph per cell; **`bubble1`** one glyph in a
bubble of a generated scene; **`bubbleN`** a 2–6 glyph window in one,
routed per glyph. Piece kind: `piece_bubble`, `piece_grid` (word cells),
`line_bubble` (a corpus line).

| tier | px p10 / median / p90 | σ band | recipe | name until 2026-10-02 (group / recipe) |
|---|---|---|---|---|
| `lone_190` | 119 / 191 / 268 | 0.7–0.9 | `grid` (its 1×1 deals) | `b0709` / `grid_single`, 1×1 |
| `grid_82` | 55 / 82 / 118 | 0.7–0.9 | `grid` | `b0709` / `grid_single`, 2×2 – 3×3 |
| `bubble1_52` | 42 / 52 / 75 | 0.7–0.9 | `bubble1` | `b0709` / `scene_single` |
| `grid_44` | 41 / 44 / 49 | 0.7–0.9 | `grid` | — (`grid_44`, 10-02) |
| `lone_44` | 34 / 44 / 52 | 0.7–0.9 | `grid`, 1×1 | — (`grid_44`) |
| `bubbleN_34` | 29 / 34 / 45 | 0.5–0.7 | `bubbleN` | `b0507` / `scene_window` |
| `bubble1_32` | 27 / 32 / 38 | 0.5–0.7 | `bubble1` | `b0507` / `scene_single_small` |
| `bubbleN_18` | 14 / 18 / 22 | 0.3–0.5 | `bubbleN` | `b0305` / `scene_window` |
| `grid_29` | 25 / 29 / 33 | 0.5–0.7 | `grid` | `g0507` / `grid_single` (`grid_small`, `grid_lone`) |
| `lone_28` | 22 / 28 / 35 | 0.5–0.7 | `grid`, 1×1 | `l0507` / `grid_single` (`grid_lone`) |
| `grid_16` | 13 / 16 / 20 | 0.3–0.5 | `grid` | `g0305` / `grid_single` |
| `lone_16` | 12 / 16 / 21 | 0.3–0.5 | `grid`, 1×1 | `l0305` / `grid_single` |
| `piece_bubble_38` | 30 / 38 / 53 | 0.5–0.7 | `scene_piece` | `b0507` / `scene_piece` |
| `line_bubble_32` | 29 / 32 / 39 | 0.5–0.7 | `scene_short` | `b0507` / `scene_short` |
| `piece_grid_29` | 26 / 29 / 32 | 0.5–0.7 | `grid_string` | `b0507` / `grid_string` |
| `piece_bubble_19` | 15 / 19 / 23 | 0.3–0.5 | `scene_piece` | `b0305` / `scene_piece` |
| `line_bubble_19` | 16 / 19 / 23 | 0.3–0.5 | `scene_sentence` | `b0305` / `scene_sentence` |
| `piece_grid_17` | 13 / 17 / 21 | 0.3–0.5 | `grid_string` | `b0305` / `grid_string` |

The six `grid` / `lone` tiers under 50 px are the experiments' (`grid_small`,
`grid_lone`, `grid_44`), not `TABLE`'s. Px sources: `retrain_kana/data`,
`run1002_grid_lone/data`, and `run0925_300f`'s build log for the piece kind.

Until 2026-10-02 an item was named by its band group and its recipe, and
the recipes were `scene_single` + `scene_single_small` (now one,
`bubble1`), `scene_window` (`bubbleN`) and `grid_single` (`grid`). Every
data dir, report and result of record carries those names;
`builder.tier_of(record)` gives a record's tier either way. The band is an
attribute of the group a tier is drawn in (one kind, one band, one rng
restart — the draw unit, unnamed) and of each item; `lone_190` and
`grid_82` are one draw (a deck that deals 1×1 beside the grids), named per
item. The rename left the draws alone: the kana run's table built before
and after it gives the same 17 400 records and pixels.

## Scene pools

The four pools the recipes draw on (`config.DATA["scenes"] = "s1,s1w,sl1w,ja_comic"`)
are grown, not rebuilt: the prompt stream is deterministic in `--seed`, so a
pool's own argv with a larger `--scene_n` keeps every stored row and renders
only the new indices (`src/scenes/stage.py`). `--scene_prune 1` deletes the
rejected renders (rows stay in `scenes_all.jsonl`); the pools were pruned on
2026-09-24 and every grow run prunes its own rejects. The argv per pool —
`S=project/finished/cjk_anima_scale/src/run_stage.py --stage scenes`, raw pack
in the env, through `make daemon-run --stall-timeout 0` (pools land in
`output/cjk_anima_scale/scenes_<pool>/`):

| pool | argv after `--scene_tag <pool>` | grown to |
|---|---|---|
| `s1` | `--scene_frames reads_as,bubble_reads,saying,sign` | 2000 (2026-09-24) |
| `s1w` | `--seed 3 --scene_shapes 576x448,448x576,640x448,448x640,640x384,384x640 --scene_frames reads_as,bubble_reads,saying,sign` | 2600 |
| `sl1w` | `--seed 1 --scene_shapes 576x448,…,384x640 --scene_frames reads_as,bubble_reads,saying --scene_anchors <the 55 EN sentences: `sorted({r["anchor"]})` over the pool's `prompts.jsonl`>` | 2000 |
| `ja_comic` | `--seed 2 --scene_shapes 384x640,448x640,448x576 --scene_frames ja_reads_as,ja_bubble_reads,ja_saying --scene_extra_tags comic --scene_min_box 40` | 4400 |
| `s1s` | `--seed 4 --scene_frames reads_as,bubble_reads,saying,sign --scene_min_box 20` — the small-bubble pool (one-glyph fit p10 38 px vs s1's 57); not in `config.DATA`, only F2a′ adds it (`experiments/f2a_line`) | 1000 (2026-09-27), 390 kept |

The line's code is `cjk_scale/`; the stage packages it runs on are its
own `src/` (nothing is imported from another line); tests:
`.venv/bin/python -m pytest project/finished/cjk_anima_scale/tests`.
