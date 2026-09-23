# plan_micro_chain — the stage chain at 30 / 30 / 30 on 24 rows (plan only, 2026-09-23)

`design.md` § 5 sets **30 steps per row per stage** and says whether that
is enough is read on the first run of each stage. At 2 300 rows a stage is
≈ 12 h, so the first run of the chain should be a micro one: the same three
band stages, the same configs except the inventory, on 24 warm rows —
≈ 25 min a stage. This plan reads that; nothing here has run.

The question is not "which band" (the law answers that) but **does the
chain as built move warm rows the right way at the production budget**:
each stage buys what it is for, and keeps what the previous stage bought.

## 0. What is known

- Every row of the law was read on micro tables of 16–24 rows
  (`band_b1`: 24 kana; `micro_cf_0922` / `micro_warm_0923`: 16 pieces), at
  40–190 steps per row, scene-only. No read has run the **chain** — warm
  from one band stage into the next — and none has run the scale line's
  own trainer / builder end to end.
- The one warm budget curve (`micro_warm_0923`, pieces at 0.5–0.7 warm
  from the merged table): 40 / 80 / 188 steps per row read exact
  6 / 8 / 13 of 32, monotone, no knee. 30 sits under its lowest point, so
  the piece stage is expected to move pieces *a little*; the read is
  whether it moves them at all and whether the singles hold.
- Draws per row is what makes 24 rows stand in for 2 300. At batch 4 and
  30 steps per row, 24 rows are 720 steps = 2 880 item draws a stage. Per
  row that lands near where production lands (production draws only from
  the rows its recipes can place — singles at `stage0709`, mostly pieces at
  `stage0507`; the micro shares are the stage's after the corpus recipes
  drop out, § 2):

  | stage | micro draws / row | production draws / row (2 300-row table) |
  |---|---|---|
  | 0709 (16 singles only) | 180 | ≈ 300 (935 singles share 280 k draws) |
  | 0507 pieces / singles (0.71 / 0.29) | 257 / 51 | ≈ 106 / 60 |
  | 0305 pieces only | 360 | ≈ 100–200 (depends on the string pool) |

  Singles sit at 0.6–0.85 × production, pieces at 2–2.5 × (8 pieces take
  the whole piece budget of a 24-row table). So a single-row read here is
  slightly conservative and a piece-row read is an upper bound; a flat
  piece read is a flat read at scale.

## 1. Rows

Twenty-four rows, **all warm in `rows_step1_0921_merged`** (the trainer's
`rows warm start: 24/24` line at step 1 is the check; a row that comes up
cold gets swapped for one that is not), three groups so every stage has
work and the open kanji question gets a free look:

- **8 kana**, from B.1's 24 (`band_b1_2026_09_23.md`), none of its dead
  rows (マ メ ロ の イ): `あ い お な ア ナ ラ ル`.
- **8 kanji**, from C.2's strata (`band_c2_kanji_2026_09_23.md`), all in
  `kanji:200`: 4 simple K_lo `人 日 口 女` + 4 dense K_hi `精 聞 動 願`.
  Kanji take the kana band (C.1 / C.2), so they are singles to the law;
  dense kanji read 0 / 64 native at 48 px scene-only, and whether grid
  cells fix that (design § 6 item 3) is what `stage0709`'s grid half reads
  on them here.
- **8 pieces**, from `assets/units/ja_micro_warm_0923.txt` (2–3 glyphs,
  one Qwen token each): `それを はじ やはり すご メン アン プロ ファ`.

Checked (CPU, `build_pools` on the run file): 16 singles, 8 pieces, every
unit one Qwen token; `single` eval = the 16 chars, `word` = the 8 pieces.

Per stage, by the law (`cjk_scale/windows.py`):

| stage | band | singles | pieces |
|---|---|---|---|
| stage0709 | 0.7–0.9 | (kana + kanji) scene 48–53 px + grid cells 60–180 px | **no window** (no row above 64 px, 24–64 is 0.5–0.7) — zero draws, rows untouched |
| stage0507 | 0.5–0.7 | scene 24–32 px | scene 35–48 px |
| stage0305 | 0.3–0.5 | **no window** (no single row under 24 px) — zero draws | scene 12–24 px (`scene_piece`, added 2026-09-23) + word-cell grid 12–24 px (`grid_string`; the stage's `scene_sentence` has no lines on this inventory — § 2) |

So the chain, on these rows, is: singles move at stage0709, both move at
stage0507, pieces move at stage0305. "Untouched" is exact — no draw, no gradient,
and with `init_anchor` the anchor term is zero on a zero delta — which
makes the untouched groups a free determinism check on the eval.

## 2. Run file vs stage file

A stage file (`configs/stage*.toml`) is the **band recipe**: band, gate,
recipe mix, trainer surface, its place in the chain (`warm_from`,
`regress`). It does not say which rows. A **run file**
(`configs/runs/<run>.toml`) says the rest: the inventory, the seed table,
the steps per row per stage, and the eval rows — the things that differ
between a 24-row read and the 2 300-row production run, and nothing else.
The run's name is the chain's tag.

```toml
# configs/runs/run0923_micro.toml — the chain at 30 / 30 / 30 on 24 warm rows
run = "run0923_micro"                       # = --tag; every stage dir carries it
seed_table = "output/cjk_anima_scale/rows_step1_0921_merged/trained.pt"   # the first stage warms from here
seed = 0

[data]                                      # over every stage's [data]
units = ["chars:あいおなアナラル人日口女精聞動願", "list:それを,はじ,やはり,すご,メン,アン,プロ,ファ"]
pieces = ""                                 # the 8 pieces ride the list: source
phrase_file = ""                            # no corpus line is made of these rows only
n_items = 4000                              # ≥ 2 880 draws a stage

[budget]                                    # steps per row, per stage
stage0709 = 30
stage0507 = 30
stage0305 = 30

[eval]                                      # over every stage's [eval]
groups = "single,word,en"                   # single = the 16 chars (chars: base → the base is the eval),
                                            # word = the 8 pieces (n_piece_eval ≥ 8), en = the control
native_chars = "あ,い,日,願"                 # 2 kana + a simple + a dense kanji
```

Precedence: **run over stage** on every key both name (the run is the
specific thing; a stage never overrides a run). `[budget].<stage>` lands
in that stage's `steps_per_row`. The production chain is the second run
file, `run_full.toml` (the whole inventory, the merged seed table,
30 / 30 / 30). Warmup is `lr_warmup_ratio` in the stage file (0.1 of the
stage's steps: 72 here, 7 k at scale), so the run file has no `[train]`.
**Done 2026-09-23** (`cjk_scale/config.py`); the stage files carry no row
keys.

Two consequences for the stage mixes under a small inventory:

- **A recipe with no source under the run's inventory is dropped, and the
  remaining shares renormalise** — logged in `build.json` as
  `dropped: {scene_short: "no short lines"}`. On this run that removes
  `scene_short` from `stage0507` (0.3 → the other two) and
  `scene_sentence` from `stage0305` (0.6 → `grid_string`). The stage
  file is not edited; the production run keeps all of them.
- **`stage0305`'s `grid_string` runs `source = "both"`** with an empty
  short pool, i.e. pieces only. Its `build_pools` assert on `phrase_file`
  goes conditional on the source actually needing lines.

Everything else — band, gate, batch 4, lr 1e-3 cosine, box-share curve,
compile, anchors 0 / 0.01 / 0.01 — is the stage value. The `px_target`
gates (± 20 % on the drawn median) stay on; a micro build that fails one
fails the same way at scale.

## 3. Steps

```bash
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
S=project/cjk_anima_scale/scale.py

# 0. the seed baseline: the same eval strings on the untrained-here table
.venv/bin/python $S --run run0923_micro --stage stage0709 --steps data            # CPU, ≈ 5 min
.venv/bin/python $S --run run0923_micro --stage stage0709 --steps eval --seed_only --submit   # § 4 item 4

# 1–3. the chain, one stage at a time — read each before the next
.venv/bin/python $S --run run0923_micro --stage stage0709 --steps train eval --submit
.venv/bin/python $S --run run0923_micro --stage stage0507 --steps data train eval --submit
.venv/bin/python $S --run run0923_micro --stage stage0305 --steps data train eval --submit
```

One stage at a time, not a chained submit: a stage that reads wrong
(warm start short of 24/24, a `px_target` gate miss, a regression) stops
the chain there. Each `build.json` is checked before its train: `dropped`
and `rejects` per recipe, `px_kept` median inside the window the stage
wants, `horizontal` ≈ 0.3 of multi-glyph items / cells (the orientation
draw, design § 4), and the untouched kind absent from the stage's
`train.jsonl`.

## 4. Tool changes

1. ~~**Run files**~~ — done 2026-09-23: `configs/runs/`, `config.load(stage,
   run=…)`, `scale.py --run <name>` (tag = run name; `--tag` stays for a
   run-less stage), `lr_warmup` → `lr_warmup_ratio`; `build.json` /
   `train_record.json` record the run file.
2. ~~**Recipe drop on an empty source**~~ — done 2026-09-23
   (`recipes.missing_source`, `builder.build`): shares renormalised,
   `build.json` carries `dropped` + the live `shares`; `grid_single` skips
   grid sizes the inventory cannot fill (`_fillable_grids`). The three
   micro builds: 0709 nothing dropped; 0507 `scene_short` ("no short
   lines") → 0.715 / 0.285; 0305 `scene_sentence` → `scene_piece` /
   `grid_string` 0.5 / 0.5 at 12–24 px. The first 0305 build was
   `grid_string` alone: word cells are grid-sized (256 px on a 2×2) and only
   the font shrinks to the px, so every image was an empty canvas with
   eight 12–20 px specks; `scene_piece` at `glyph_px 12–24` went into the
   stage file (0.4 / 0.3 / 0.3) so the band also sees a piece in a bubble,
   with `fill_min = 0.5` (`recipes._draw_scene`): a scene recipe with a
   `glyph_px` target keeps only the scenes whose bubble the text fills to
   that share (`target_px / _fit_px`), so 12–24 px pieces land in small
   bubbles (micro build: fill median 0.58, 562 distinct scenes over 2 000
   items) instead of floating in a big one.
3. ~~**`grid_string` needs a phrase file only when its source needs lines**~~
   — done with item 2: the `phrase_file` assert is gone; an empty source
   drops the recipe instead. Same day: the builder forks the draw loop over
   `--workers` (default cpu − 2) processes, each on a stream seeded from
   the build seed — 4 000 items in ≈ 25 s of rendering vs 2.4 min serial
   (`build.json` records `workers`; a build is deterministic per
   seed × workers).
4. **Seed baseline** — `--seed_only` on `eval`: evaluate the run's
   `seed_table` under `rows_scale_<stage>_<run>_seed/` with the stage's
   data dir, no training. The probe's eval readers may open
   `trained.pt["args"]` (the merged table has `delta` / `arm` /
   `merged_from` / `killed`, no `args`); if so the step writes a wrapper
   table with a synthetic `args` rather than training a 1-step stand-in.
5. Nothing in the trainer or the law.

## 5. Reads and verdict rules

Rulers, all on the probe's stages through `cjk_scale/eval.py`:
`single` exact 16 rows × 2 seeds (8 kana + 8 kanji, read per script),
`word` exact 8 × 2, `en` 12 × 2, native on 4 chars (`en` / `swap`), `cf_sense --cf_lang ja` on the stage's
`cf_rows`, and `regress.json` (the previous stages' groups on the new
table, same strings by construction).

| read | rule | what a miss means |
|---|---|---|
| stage0709 `word` == seed `word`, exactly | determinism (pieces got no draws) | eval noise floor is not zero — every other row of this table gets a ± band before it is read |
| stage0709 `single` / native vs seed | ≥ seed | 30 steps at 1e-3 on warm rows churns them: read `warm_cos` (`train_log.json`); < 0.9 says the seed's identity was overwritten, not refined — the budget or lr is wrong for a *refinement* stage |
| stage0507 `word` vs seed | > seed by ≥ 2 of 16 on either seed, native piece both-hit up | the piece stage buys nothing at 30 / row (the `micro_warm` curve says 40 buys 6 / 32 from cold-ish); then 30 / 30 / 30 is under budget for pieces and § 5 of design.md gets a number |
| stage0507 `regress stage0709/single` | within 1 of stage0709 | the 24–32 px single draw at 0.5–0.7 undoes 0709's identity — the anchor 0.01 is too light or the small-single recipe does not belong in 0507 (design § 6 item 2) |
| stage0305 `regress stage0709/single` | == stage0507's (no draws) | determinism again |
| stage0305 `word` vs stage0507 | ≥ stage0507, cf_sense peak in 0.3–0.5 | small-px pieces do not read on the 48 px `word` ruler even if they trained — then the read is cf_sense alone, and a `word` drop says the small band overwrote the 0507 state |
| stage0709 kanji `single` (dense 精 聞 動 願 vs simple 人 日 口 女) | dense ≥ seed; native on 願 up | grid cells did not wake dense kanji — design § 6 item 3 leans to exposure, not px, and the dense stratum needs a density-weighted draw count |
| chain end vs seed, all groups | singles ≥ seed, pieces > seed | the chain at 30 / 30 / 30 is a net gain on warm rows — go to scale with the production configs as they are |

Two seeds are the whole replication; the ± band from rule 1 is read
before any "≥" above is called. A result that is flat everywhere is a
budget statement, not a law statement — the next cell is the same chain at
60 / 60 / 60 (`--steps_per_row 60`), not a band change.

## 6. Cost

| step | wall |
|---|---|
| data, 3 000 items × 3 stages (CPU) | ≈ 5 min each |
| train, 720 steps + load + compile | ≈ 10 min each (B.1: 4 500 steps in 37 min; smoke: 0.9 it/s at step 25 under compile warm-up) |
| eval: exact + native + cf_sense | ≈ 10 min each (B.1 native on 24 rows: 14 min) |
| seed baseline eval | ≈ 10 min |

≈ 30 min a stage (720 steps), ≈ 100 min the chain with the baseline, read one stage at
a time. GPU steps through the daemon (`--submit`), raw pack in the submit
shell.
