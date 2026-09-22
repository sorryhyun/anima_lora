# plan — covering Japanese, then the sentence run (2026-09-21)

Index: [`README.md`](README.md). Two things are planned here and nothing else:
**how the table comes to cover Japanese**, and **how the sentence run is done
on it**. How the table is trained today is [`training.md`](training.md), every sentence run so
far is [`sent_run.md`](sent_run.md), verdicts are [`findings.md`](findings.md),
numbers are [`reports/`](reports/README.md), tonight's borrowed-box logistics
are [`plan_z8.md`](plan_z8.md), and the shipped file form is
[`deploy_plan.md`](deploy_plan.md).

> **Every launch states its pack:**
> `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` (raw, log sha
> `7b9fce0bb57b`). `configs/base.toml` defaults to the preview pack, which
> already carries a trained delta.

## 1. Covering Japanese

**What "covered" means.** A dialogue line is covered when every Qwen piece in
it has a trained ext row. Rows are Qwen pieces, and a piece is often several
glyphs (って, じゃない, ありがとう are one piece, one row). Over
`dialogue_2_10.tsv` (42 557 lines) multi-glyph pieces are 35 % of piece tokens
and single kanji 16 %, so the inventory is cut from **one joint frequency
ranking** of both
([`reports/piece_coverage_2026_09_21.md`](reports/piece_coverage_2026_09_21.md),
[ranked TSV](reports/piece_coverage_ranked_2026_09_21.tsv)), not from
`kanji:N`.

**The tables.** Disjoint ext ids, one recipe, merged by id.

| table | rows | what | steps | covered lines, cumulative | state |
|---|---|---|---|---|---|
| `step1_0921` | 374 | kana, `kana_ext`, small kana (in digraphs), 13 punctuation units, `kanji:200` | 30 k | 13.5 % | **done** — arm `rows_step1_0921_s30k` |
| `step1_0921z` | 1 900 | cold ranks 1 – 1 900 of the joint list, re-cut against `step1_0921`'s `ext_ids`: 1 317 multi-glyph pieces + 583 kanji; 1 / 2 / 3 / 4 / 5 glyphs = 583 / 796 / 353 / 139 / 29 | 152 k | ≈ 85 % | **done** 2026-09-22 — arm `rows_step1_0921z_s152k` (Z8, [`plan_z8.md`](plan_z8.md)); kanji 15/48, multi-glyph 3/144 as strings — under the gate, kept for coverage; merged → `rows_step1_0921m_merge` (2 274 rows) |
| next | ≈ 1 100 | the following ranks, to ≈ 3 000 (≈ 430 kanji + 670 multi-glyph, 431 of them with kanji) | 88 k | ≈ 95 % | later, here — after § 1b's probe |
| tail | ≈ 1 100 | the rest of the ranking | — | ≈ 98 % | later |

Units of 6 + glyphs are left out of every cut (10 inside the first 1 900:
かもしれない, ありがとうございます, …): no grid cell holds them above ≈ 16 px and
the scene compositor's one-column cap is 5 glyphs, so such a row would train on
renders nobody can read. What frequency misses (こんにちは is rank 1 426) is in
the 1 900; anything a release needs beyond the ranking is pinned into the list
file by hand.

### The recipe — the grid mix

One run holds **scene singles** (one unit composited into a bubble of an
existing scene pool — frame, trigger, "one unit") and **grid items** (k units
in a cols × rows grid, one position clause per cell — k rows per step, the
glyph stays large). Grid alone is not a seed table and a grid share above 50 %
costs native; both are closed
([`reports/grid_s0_2026_09_20.md`](reports/grid_s0_2026_09_20.md),
[`grid_m0`](reports/grid_m0_2026_09_21.md),
[`grid_m0b_s1`](reports/grid_m0b_s1_2026_09_21.md),
[`grid_g1`](reports/grid_g1_2026_09_21.md),
[`exposure_ledger`](reports/exposure_ledger_2026_09_21.md)).

| knob | value |
|---|---|
| loss | plain FM (`--pair_loss 0`), `--lr_rows 1e-3`, cosine, `--free_residual 1e-3` |
| σ band | 0.7 – 0.9 |
| in-box weight | `--box_share 0.25` (scene items) |
| mix | grid 50 % — the share is the item ratio (scene items : grid items), no flag |
| grids | `2x2,3x3,2x3,3x2` (512², 512², 416×624, 624×416), bubble frame on half |
| batch / budget | 4; **80 steps per row**, cold, `--save_every 5000` |
| scene half | `--scene_mix single=1.0`, pools `s1,s1w,sl1w,ja_comic`, `--scene_vertical 1` (every multi-glyph scene unit is a column) |

`step1_0921` is this recipe at 374 rows and it is the read that set the
budget — 30 k plain steps against `step1_0920`'s 53 k ΔFM scene-only:

| | `single` | `_ext` | `_small` | `_kanji` | native `en` / `swap` (both readers, of 64) | en cos |
|---|---|---|---|---|---|---|
| `step1_0920` | 20/36 | 8/36 | 0/36 | 8/36 | 19 / 8 | 0.934 |
| **`step1_0921`** | **28/36** | **20/36** | **10/36** | **19/36** | **34 / 28** | 0.903 |

Multi-glyph groups are still 0 – 1 exact (`line` 1/32, `combo` 0/36): identity
got cheaper, sentence content did not move. That is section 2's job.

**Multi-glyph units in the data** (new for `step1_0921z`, all default-off so
older dirs rebuild unchanged):

- `--units 'list:@<file>'` — the inventory is a file under `assets/units/`
  (`ja_cold_0001_1900.txt`: piece, count, class, glyphs, ext id, source rank).
- `--grid_unit_min_glyph 56` — the deck's next unit picks a (grid, frame) whose
  cell holds its glyph count: one glyph anywhere, a digraph in 3x3, 3 glyphs in
  2x2 or a flat 2x3 / 3x2, 4 – 5 glyphs in a flat 2x2 only; the other cells
  take what fits. Every unit is still dealt once per deck pass. The grid share
  moves toward 2x2 (review build: 343 of 600 items, 5.2 cells per item against
  6.25).
- `--grid_mark_horizontal 1` — a grid cell drawn as a left-to-right line says
  `horizontal Japanese text reads as "…"`; a column keeps the bare clause (the
  manga default, and the only form the scene half draws).
- `--seed k` moves every rng stream, so a big build is K shards (`--seed
  0..K-1`, own `--data_tag`) run side by side with `train.jsonl` concatenated —
  the records hold absolute paths, the joined dir holds no images.

**Before training a table**, off the joined `train.jsonl`: items per row for
the scene half and the grid half separately (a weighted draw is not a quota —
name every unit under ≈ half the mean), the glyph-px histogram, and the caption
check — each unit's own ext id is what its caption tokenizes to and no other
ext row appears (review build: 3 705 of 3 705, 0 stray).

**Reading a cold table.** Its own units, rendered alone, read **as strings**,
split by glyph count and sampled across the rank range (head and tail
separately). Pass = multi-glyph pieces not below `step1_0921`'s kana rate
(28/36) and kanji not below 19/36. A piece that reads as its first glyph only
(会長 → 会) is a unit that did not train as a unit — the known failure, and the
first thing to look for on the sheets. The eval set for this is owed
(`single_extra` is the list's first 18 today).

**Merge.** `src/probe/merge_tables.py --base <step1_0921> --add <step1_0921z>`
(`row_scale` converted, ids disjoint so nothing overlaps) → an ordinary `rows`
arm dir. First read of the merged table: `step1_0921`'s 374 eval rows inside
its rerun floor, and a native read.

### 1b. Mean-init for every row not trained yet (idea, user 2026-09-22 — not run)

**Why.** A caption that touches an untrained ext row breaks the whole picture,
not just the bubble, so every JA piece outside the tables is a hazard until its
rank is reached — and the rest is long and rare: cold ranks 1 901 – 3 000 are
430 kanji + 670 multi-glyph pieces of which 431 contain kanji (78 % of the
table has to draw a kanji; corpus counts 7 → 2), the cells `step1_0921z` read
weakest on (kanji tail 3/24, multi-glyph tail 0/72; the read is in
`rows_step1_0921z_s152k/`, builder `eval_build.py`).

**The idea.** Fill every untrained JA row with the **mean of the trained rows
of its type** (kanji / kana-only piece / piece with kanji / katakana, by glyph
count if the means differ), then train the remaining ranks from that init at
fewer steps per row.

**What is already known** (`findings_seed.md`, merged-table geometry
2026-09-22):

- The mean is the shared direction: m̂ carries 20 – 27 % of row energy, the two
  tables' m̂ have cos 0.855, residuals are near-orthogonal (class-block cos
  0.03 – 0.05 after m̂ is out) so they cancel in a mean. A class mean has norm
  ≈ 67 against a row's ≈ 149 — the scale is a knob of the probe.
- **m̂ alone keeps the scene best of any cond** (transplant: en cos 0.920, IoU
  0.31 against the table's 0.882 / 0.13) **and renders 0/64** — floor garble in
  the bubble, scene intact. That is the failure wanted from an untrained row.
- **It should not be expected to save steps.** `--pin_dir` (m̂ inherited,
  residual trained) overlapped scratch at 500 and 2 000 steps: the budget is
  the per-row identity, and m̂ is what a free row grows first anyway. The pin
  froze the direction where this only initialises it, and pin 500 led `swap`
  20 to 9, so the step question is low-odds, not closed.

Probes that would change this plan (the exposure audit: equal draws, a 152-step
revisit interval, the pull not scaling with table size; the split-gain read;
the effective-embedding init) are in [`suggestions.md`](suggestions.md).

**Order — micro arm first, no training:**

1. Probe (≈ 15 min GPU): 12 – 16 untrained pieces from ranks 1 901 + (kanji,
   piece with kanji, katakana), `native` under (a) no delta, (b) class mean,
   (c) class mean scaled to a trained row's norm. Read en cos / box IoU: does
   the picture survive, and at which norm.
2. If it does: a table holding the init for every remaining JA row, merged
   under the trained tables (`merge_tables.py`, trained rows win). This closes
   the broken-picture hazard by itself and is independent of step 3.
3. Separately, one variable against cold: the 1 100-row table from the init at
   40 and at 80 steps per row.

**How long "every JA row" is.** Corpus side: 4 132 cold pieces in the ranking,
1 900 trained → 2 232 left (1 100 to 95.1 %, 1 132 more to 98.4 %). Pack side
(30 951 Qwen-piece rows): kana-bearing pieces 2 053 (1 282 trained → **771
left**; 1 / 2 / 3 / 4 / 5 / 6 + glyphs = 180 / 792 / 628 / 288 / 103 / 62),
single han 8 502 (783 trained; shared with ZH — the JA share is the ≈ 2 100 –
3 000 jōyō + jinmeiyō), multi-glyph han-only pieces 16 379 (199 trained;
mostly Chinese words), Hangul 3 473. A mean-init costs nothing per row, so step
2 can cover all of single han + kana-bearing (≈ 10 500 rows) without deciding
which han are Japanese.

## 2. The sentence run

**What it is for.** Step 1 gives rows that each draw their own unit. The
sentence run warm-starts the **merged** table and trains real multi-piece lines
so the rows compose. Standing rule: **no cold row inside a sentence run** — new
vocabulary is a step-1 table, merged first.

**Why it runs after coverage, not before.** Every sentence run to date trained
on a pool of 538 sentences / ≈ 925 strings, because a line is dropped the
moment one of its pieces has no warm row. They all landed at pooled lift
+0.09 – +0.15 with 0 exact, the gain sat in ≈ 27 frequent rows, and oversampling
the rare rows of the same pool bought nothing (`sent_run.md` items 7 – 10).
What separated the rows that gained from the rows that did not was the variety
of strings they appeared in. The merged inventory opens the pool to ≈ 30 000
sentences; that is the lever this run tests, and the reason coverage came
first.

**Data** — `step2_0920b`'s build on the merged inventory:

```
--stage data --arm rows --data_tag <tag> \
--scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic --single_scenes s1,s1w --single_max_ar 2 \
--units kana --units 'kana_ext*1' --units small --units 'kanji:200*1' \
--units 'list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・,@ja_cold_0001_1900.txt*1' \
--phrase_file <manga109s>/derived/dialogue_2_10.tsv --phrase_min_pieces 2 --n_phrase_eval 8 \
--scene_mix single=0.1,short=0.5,sentence=0.4 --short_pieces 2-5 --short_max_lines 1 \
--short_lexical 0 --phrase_norm 1 --text_draw balanced \
--sentence_min_letters 6 --sentence_min_glyph 20 --sentence_fill 0.9 --scene_vertical 1 \
--n_items 10000 --scene_frac 1.0 --natural_frac 0 --strings_frac 0 --flat_bubble 1.0 \
--scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 2 --shapes 512
```

A `list:` source mixes literals and `@file` tokens. The build prints
covered lines / sentences / shorts; **the run is sized from that print**, in
looks per string (6 k steps was ≈ 25 looks at each of 925 strings), not from
the old step count.

**Train** — Round 2's argv, one warm source:

```
--stage train eval native --arm rows --data_tag <tag> --shapes 512 \
--init_rows output/wake_probe/<merged arm>/trained.pt --init_anchor 0.3 --lr_warmup 500 \
--train_steps <sized> --batch 4 --t_min 0.7 --t_max 0.9 --t_band_multi 0.35,0.7 --compile 1 --grad_ckpt 0 \
--lr_rows 1e-3 --lr_decay cosine --free_residual 1e-3 \
--box_share 0.05 --box_share_cap 0.25 --c_flat 0 --pair_loss 0 \
--eval_groups single,single_ext,single_small,single_kanji,short,short_held,phrase,phrase_held,en \
--native_chars あ,か,す,日 --native_clauses en,swap --seeds 2 --delta_parts full --no_floor
```

**The σ band is length-conditioned** (2026-09-22, `reports/step2_band2_2026_09_22.md`):
single-glyph items at 0.7–0.9, multi-glyph at 0.35–0.7 — where `cf_sense` says
the caption has leverage on each (`reports/cf_sense_gate0_2026_09_22.md`). The
band alone, at 5 k steps, beat the 30 k run at 0.5–0.9 on every eval group and
returned most of the single-glyph native it had lost; the multi rows move
orthogonal to m̂, so step1's high-σ behaviour is inherited. Size the run at
≈ 5 k, not 30 k.

The anchor μ is the one trade knob: μ 0.3 → 0.1 bought +0.054 lift and halved
single-glyph native each time — on the 0.5–0.9 band, where the string draws
at 0.7–0.9 were also what cost the singles' native. **Running now: μ 0.1 on
the new band, 5 k, otherwise band2's argv** (`20260922-143706-78a340`, arm
`rows_step2_0921m_band2_mu01_0922`; its `--eval_tag sent` native is a
separate launch after). **The decision on what the sentence run of record
is — band2 at μ 0.3, μ 0.1, and at which length — waits on that read**: if
μ 0.1 keeps the singles' native (en / swap ≥ band2's 22 / 16) it is the
recipe and the run is sized from its `warm_cos` curve (30 k with a cosine
tail moves the rows further than 5 k; the old 30 k run's "nothing after 5 k"
was on the wrong band); if it halves them again, band2 at μ 0.3 is the
recipe and length is the only open question. A 30 k band2 run was queued
and withdrawn (`20260922-143615-70ac94`) for this read.

**Reading it** (`sent_run.md` *Reading rules*): `src/probe/sub_exact.py`
pooled lift with the `_held` groups, never exact match alone; single-glyph
native (あ か す 日 × `en,swap`) beside it, because that is where a sentence run
pays; multi-glyph native (`--eval_tag sent`); `src/probe/row_dose.py` against a
seed read of the merged table made with the same eval strings — the rows under
400 multi-glyph items read +0.000 on every run so far and are the target.

**Optional, cheap, comparable:** the same 6 k run on `step1_0921` alone with
`data_step2_0920b` as it is on disk — one variable against `step2_0920b` (the
seed table), ≈ 45 min.

## 3. preview2 — what v2.0.0.beta2 ships

```
step1_0921 ─┐
            ├─ merge_tables ─→ sentence run (section 2) ─→ bake ─→ anima_cjk_vocab_pack_preview2
step1_0921z ┘
```

`preview2` is the merged JA table **after** the sentence run, baked with
`scripts/toolkits/bake_vocab_pack.py` into the pack pair
([`deploy_plan.md`](deploy_plan.md)). It replaces today's
`anima_cjk_vocab_pack_preview` (`sent_s24k_a1_s05`, 503 rows, preview-pack era).
In v2.0.0.beta2 the trainer's default moves to it: `VOCAB_PACK_STEM` in
`library/downloads.py` and `vocab_pack` in `configs/base.toml`. A new pack is a
new digest, so the release notes carry `make preprocess-te ARGS=--overwrite`
for CJK captions and the stamp-mismatch warning on LoRAs trained against the
old preview. If the later rank tables are done by then they are merged in
before the sentence run, never after it.

## Order

1. `step1_0921z` data: join the shards, the pre-training checks, the eval set.
2. Z8 night: train `step1_0921z`, pull everything back ([`plan_z8.md`](plan_z8.md)).
3. Read the cold table; merge; read the merged table.
4. Sentence data on the merged inventory → size → the sentence run → its reads.
   **Where it stands (2026-09-22):** the 30 k run at 0.5–0.9 is read; band2
   (length-conditioned band, 5 k) beat it on every group; μ 0.1 on band2 is
   running. Next decision — μ and run length — is made on the μ 0.1 read,
   then one run of record.
5. Bake `preview2`; v2.0.0.beta2.
6. § 1b's mean-init probe (no training), then ranks 1 901 – 3 000 here, whenever the GPU is free; KO / ZH stay parked
   (`plan_z8.md` *Parked*).

## Closed — do not re-propose

Grid alone as a seed table; a grid share above 50 %; ΔFM on grid items or on
sentences; row oversampling of an unchanged string pool (`--row_boost`); the
in-box weight as a sentence lever; a better seed by itself as a sentence lever;
cold rows inside a sentence run; random multi-glyph strings in a cell (real
pieces only); `kanji:N` alone as the coverage route. Reasons: `findings.md`,
`sent_run.md`, `reports/grid_*.md`.

## Where the older plans went

`_archive/cjk_renderable_anima/`: `plan_2026_09_20.md` (step 1 / step 2 / V /
K as planned on 09-20 — source docstrings citing `plan.md` S1a / S1b / S1c / K
resolve there), `plan_grid.md` (the grid gates S0 → G1 and the exposure
arithmetic; docstrings citing `plan_grid S1`), `plan_step1.md` (dropped when
its boost gate read ≈ 0), `plan_synth*.md`, and the full
`deploy_plan_2026_09_17.md`.
