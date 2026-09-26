# cjk_anima_scale — the JA vocab pack at scale

Opened 2026-09-23 out of [`../cjk_renderable_anima/`](../cjk_renderable_anima/).
That line is the research surface (probe code, `reports/`, `findings.md`);
this one is the production line that builds the pack on what it settled:
one loss, one trainer, and a σ-band schedule whose stages differ only in
their data. Nothing here has trained yet — the plans are plans.

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
| [`band_experiment_results.md`](band_experiment_results.md) | **the vocab band law** — the verdict, the per-px window table, the training reads, what is left unrun |
| [`design.md`](design.md) | the scale pipeline: stage schedule, data builder, thin trainer, open questions (§ 6) |
| [`micro_chain_result.md`](reports/micro_chain_result.md) | **the first run of the chain** — the three band stages at 30 / 30 / 30 on 24 warm rows (8 kana + 8 kanji + 8 pieces), read rule by rule; the seed-baseline and table-membership tool changes it forced |
| `reports/` | dated reads: [`conflict_joint_2026_09_25.md`](reports/conflict_joint_2026_09_25.md) — the gradient conflict probe + ten chain / joint arms on one data set: chain ≡ joint, steps/row is not the budget, singles and pieces want opposite regimes; [`grid_box_2026_09_25.md`](reports/grid_box_2026_09_25.md) — grid cells as the loss box: the `grid_string` price ×8–19, warm singles unmoved; [`piece_2026_09_25.md`](reports/piece_2026_09_25.md) — where 300f actually failed, by layer; [`piece_only_2026_09_26.md`](reports/piece_only_2026_09_26.md) — the one-file parity by replay + the scene_piece-only arm (not the doubling lever); [`spell_2026_09_26.md`](reports/spell_2026_09_26.md) — singles spelled in a line (layout, not identity, blocked composition; line-trained singles double), the piece row vs its glyphs in row space, the paired re-read vs the seed floor; [`transplant_line_2026_09_26.md`](reports/transplant_line_2026_09_26.md) — spell_b's Δ split into its shared "line" component and the per-row rest (strip / shared-only / a transplant of 300f_sp's direction): composition splits, doubling is the shared part; [`transplant_piece_2026_09_26.md`](reports/transplant_piece_2026_09_26.md) — proposal Stage A: a piece direction out of sample, contained 27 → 52 / 256; [`stage_b_2026_09_26.md`](reports/stage_b_2026_09_26.md) — proposal Stage B: a 36-kana donor's direction on 10 held-out kana, composition transfers and doubling with it; [`floor_score.md`](floor_score.md) — **the seed floor of record** on sent / target / word / en |
| [`idea.md`](idea.md) | not scheduled — a per-cell gradient bank + validation-influence price in place of `‖ḡ‖·coh` and trained arms, the matched-σ sweep that would tell band from weighting; from the 2026-09-25 outside review |
| [`proposal.md`](proposal.md) | **the open items of the line-mode work** (merged 2026-09-26 from the line-mode and factorized-rows proposals; what ran is in `reports/`): in-word doubling (B), trained rows rendering a line alone (F1b, the count twin), Stage I, the mode for pieces, shipping the mode, modes beyond line |
| [`opinion_factorizedrows.md`](opinion_factorizedrows.md) | an outside advisory (2026-09-26): adapter replay / distillation; parked (`proposal.md` § 2.7) |
| [`plan.md`](plan.md) | the collapse spec: a run is one file, everything else is a rule (§§ 1–5; § 6 = the order, 1–2 done) |
| `configs/runs/*.toml` | the runs — `{vocabs, read}` and nothing else: `run0925_300f` (300 pieces, the freeze arm; its re-run on this shape is plan.md § 6-3) |
| `cjk_scale/` | the code (`windows` = the law, `config` = the run file + data pools, `recipes` + `builder` = data and the recipe table by kind, `rows` + `train` = the fixed trainer, `eval` = floor + trained on one sheet, `conflict`, `bake`, `ledger`); `scale.py` is the front door |
| `src/` | the stage packages the line runs on (render, readers, scoring, sheets, eval / native / target / cf_sense / scenes), vendored 2026-09-25 byte-faithful and pruned the same day to the code the line runs; `src/run_stage.py` runs one by hand |
| `assets/` | what `src/` reads: `fonts/` (binaries gitignored, `FONTS.md`), `vocabs/` (vocab files), `target_prompts.txt` |
| `_archive/` | the retired stage surface — `configs/stage*|joint*.toml`, the stage-shaped run files, `joint.py`, `boxprobe.py` (see its README) |
| `runs/` | `ledger.jsonl` — every submitted job |

## Where it stands (2026-09-26)

**The singles' line mode transfers to held-out kana, and doubling travels
with it** ([`reports/stage_b_2026_09_26.md`](reports/stage_b_2026_09_26.md),
proposal Stage B). The shared Δ direction of 36 donor kana, trained on
in-line words with a count tier, was added at one coefficient to the seed
rows of 10 kana no donor word contains. On spelled held-out words
(ひまわり さくら みどり くもり まくら) ≤ 1 edit goes 11 → 66 / 160
(norm-matched random 9), exact 0 → 12. Held-out singles repeat 25 → 53 /
320 and official drops 149 → 109. The count tier at 0.3 split neither in
the donor's own rows nor in the transplant. Stage A had shown the same
transfer for pieces ([`reports/transplant_piece_2026_09_26.md`](reports/transplant_piece_2026_09_26.md)).
The α sweep (report § 7) puts the operating dose at α 1 (words ≤ 1 edit
11 / 40 / 66 / 34 at floor / 0.5 / 1 / 2). No dose bounds doubling. Alone
(repeat 25 → 60) it can be gated by context; in words (`dup` 26 → 44–73)
it cannot. F0 (adapter only, [`reports/f0_interaction_2026_09_26.md`](reports/f0_interaction_2026_09_26.md))
found the adapter context-blind to a row's change (out cos 0.95–0.97 alone
vs in a word, line-trained rows like a random step), so the gate belongs at
the embed hook. F1 ([`reports/f1_line_2026_09_26.md`](reports/f1_line_2026_09_26.md))
trained the donor with a gated `v_line`. The seed rows + `v_line` compose
the held-out words like post-hoc `u_S` (≤ 1 edit 64 vs 66 / 160) with the
singles at the floor, but in-word `dup` rises to 77 (u1 44). The trained
rows still render a line alone (94 / 144, floor 47). At half dose
(report § 5), seed + 0.5 · `v_line` reaches ≤ 1 edit 80 / 160 (u1 66) with
`dup` 54 and the singles at the floor: the best held-out composition so
far. The open items (in-word doubling first) are
[`proposal.md`](proposal.md) § 3.

**The shared "line" Δ is half the composition and most of the doubling**
([`reports/transplant_line_2026_09_26.md`](reports/transplant_line_2026_09_26.md)):
split spell_b's five rows into their shared Δ component and the per-row
rest — held-out `あ り が と う` ≤ 2 edits: per-row only 5, shared only 7,
both 13 / 32 (floor 1); repeats: per-row only 22 (= floor 19), shared only
37, both 53 / 160. A line mode transplanted by row arithmetic carries the
doubling with it (singles official 90 → 60); 300f_sp's piece direction
added to the seed singles renders the seed. Count has to come from data (B2).

**Spelled singles compose once trained in lines**
([`reports/spell_2026_09_26.md`](reports/spell_2026_09_26.md)):
`"あ り が と う"` (five single ids — a half-width space is dropped) renders one
big あ on the seed; the five rows trained on in-line words at the piece
bands spell the held-out ありがとう within 2 edits 13 / 32 (seed 1, the
piece row alone 4) and double when rendered alone (あ → ああ 16 / 32). Row
geometry does not show it (piece ↔ glyph R² ≈ 0.03 everywhere). Paired
against the seed floor, the only gain on record is piece-alone native;
`word` exact moved nowhere. The floor is now one cache in the seed dir.


**The one-file shape reproduces run0925_300f**, and **the scene_piece-only
mix is not the doubling lever** ([`reports/piece_only_2026_09_26.md`](reports/piece_only_2026_09_26.md)).
Parity by replay instead of the retrain (merge max |Δ| 0, step 1 identical,
eval within render noise). `run0926_300f_sp` (same 300 pieces + freeze,
`scene_piece` only) ties 300f on piece identity, buys `word` exact 4 → 9 / 36,
loses sentence strings (sent contained 11 → 4). Doubling predates training
and survives both mixes — the next read is doubling itself, not a mix. Every
eval now carries the piece ruler (`native_piece/`, `eval.piece_vocabs`).

### 2026-09-25

**Grid items now train under the box-share loss**
([`reports/grid_box_2026_09_25.md`](reports/grid_box_2026_09_25.md)): every
grid draw before this date was the plain canvas mean, a 24–32 px cell ≈ 0.3–0.8 %
of it — the price table's 10–60 × `grid_string` gap was the loss form.
`grid_box = 1` (all stage files) takes the cells' union as the box:
`grid_string` buys a piece row a quarter to a third of `scene_piece` per
draw (parity per item), and matches it at 0.7–0.9; warm singles' grid
gradient is unchanged (no in-box residual to weight). The `reports/next_2026_09_25.md` § 0
`grid_string` → 0 decision is superseded; the shares and the grid-piece
band are open.

**The chain question is closed** ([`reports/conflict_joint_2026_09_25.md`](reports/conflict_joint_2026_09_25.md)):
the band stages' gradients agree per row (a training-free read,
`scale.py --steps conflict`), and ten arms on the same 15 000 renders —
30 / 30 / 30, 100 / 100 / 100, joint 90 / 300, μ 0.1 / 0.01 / 0, lr 1e-3 / 1e-4
— land within ±2 of each other whenever the rows stay near the seed (drift
≤ 0.1), and lose native when they leave it (μ 0 / lr 1e-3, drift 0.55). What
moves pieces is displacement ≈ 1.0 at μ 0 / lr 1e-3 (`micro_warm_0923`),
which the seed's singles cannot ride. **Re-read 2026-09-25 (report § 6)**: the
report's drift column is per warm-from table, so the chain's pieces are at
0.39 vs the seed (not 0.07) and the μ 0 / lr 1e-3 joint's pieces at 0.89
(not 0.55) — that arm bought the comparator's displacement without its hits,
so "more steps/row" is not the open branch; the freeze arm is. Row exposure
is grid-dominated (a piece row: 313 `scene_piece` vs 622–1 664 `grid_string`
items per stage dir). Pieces trained alone with the singles frozen (`run0925_300f`, 300 pieces,
`reports/next_2026_09_25.md` § 4a): the acceptance rulers read **nothing bought** at drift 1.4
or 1.7, but piece (§ 4b) showed 2-glyph piece identity + native trigger
WAS bought — the failure list is doubling, sentence assembly, 3+-glyph
pieces. The seed floor is now nailed
([`floor_score.md`](floor_score.md),
full-seed floor arm): sent floor はい 3 / おしい 5 (the rest 0), target 0/14,
word exact 6/36 contained 11/36 — so the run's はい = floor exactly (freeze
control ✓), **おしい went below floor** (5 → 2, piece identity up while the
string fell), and **doubling predates the run** (the seed's own misses
double). Next: `scene_piece`-only data on the same 300 vocabs, judged on
piece + whether containment rises without exact falling.
`product_criteria.md` now splits a dev set (choose arms) from the acceptance
set (accept one). The paragraph below is the state before these reads.


The law is written and encoded (`cjk_scale/windows.py`); the four stage
configs, the data builder with the band gate, the thin trainer, the eval
delegation and the front door exist. **The chain has run once, on 24 warm
rows** (reports/micro_chain_result.md): at μ = 0.1 on
every stage it passes the chain-end rule (singles 24 → 26 / 32, pieces
2 → 5 / 16, EN held), which set `init_anchor = 0.1` in the stage files; at
μ ≤ 0.01 stage0507 undid stage0709's singles. The piece read is a budget
statement (the next cell is 60 / 60 / 60), and the piece `cf_sense` ruler
renders too large to read the 0.3–0.5 band. **No stage has trained at
scale.** Seed rows for the chain:
`output/cjk_anima_scale/rows_step1_0921_merged` (2 274 rows, the probe line's
`step1_0921` + `step1_0921z` merge). Every launch states
the raw pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`,
sha `7b9fce0b…`) and goes through the daemon.

## Running a run

```bash
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack   # MANGA109S comes from .env
.venv/bin/python project/cjk_anima_scale/scale.py run0925_300f data              # CPU
.venv/bin/python project/cjk_anima_scale/scale.py run0925_300f train --submit --queue
.venv/bin/python project/cjk_anima_scale/scale.py run0925_300f eval --submit
.venv/bin/python project/cjk_anima_scale/scale.py run0925_300f conflict --submit
.venv/bin/python project/cjk_anima_scale/scale.py windows | runs | ledger
```

A run is `configs/runs/<run>.toml` = `vocabs` (a vocabs file: one vocab per
line) + `read` (the `native_sent` strings). `data` draws the items by the
vocabs' kinds (singles → `scene_single` + `grid_single` at 0.7–0.9; pieces →
`scene_piece` at two px tiers, `grid_string`, `scene_short`,
`scene_sentence` at 0.5–0.7 / 0.3–0.5), ≈ 67 items per vocab, every item
stamped with its band; `train` trains the vocabs' rows from the seed rows
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

## Scene pools

The four pools the recipes draw on (`config.DATA["scenes"] = "s1,s1w,sl1w,ja_comic"`)
are grown, not rebuilt: the prompt stream is deterministic in `--seed`, so a
pool's own argv with a larger `--scene_n` keeps every stored row and renders
only the new indices (`src/scenes/stage.py`). `--scene_prune 1` deletes the
rejected renders (rows stay in `scenes_all.jsonl`); the pools were pruned on
2026-09-24 and every grow run prunes its own rejects. The argv per pool —
`S=project/cjk_anima_scale/src/run_stage.py --stage scenes`, raw pack
in the env, through `make daemon-run --stall-timeout 0` (pools land in
`output/cjk_anima_scale/scenes_<pool>/`):

| pool | argv after `--scene_tag <pool>` | grown to |
|---|---|---|
| `s1` | `--scene_frames reads_as,bubble_reads,saying,sign` | 2000 (2026-09-24) |
| `s1w` | `--seed 3 --scene_shapes 576x448,448x576,640x448,448x640,640x384,384x640 --scene_frames reads_as,bubble_reads,saying,sign` | 2600 |
| `sl1w` | `--seed 1 --scene_shapes 576x448,…,384x640 --scene_frames reads_as,bubble_reads,saying --scene_anchors <the 55 EN sentences: `sorted({r["anchor"]})` over the pool's `prompts.jsonl`>` | 2000 |
| `ja_comic` | `--seed 2 --scene_shapes 384x640,448x640,448x576 --scene_frames ja_reads_as,ja_bubble_reads,ja_saying --scene_extra_tags comic --scene_min_box 40` | 4400 |

The line's code is `cjk_scale/`; the stage packages it runs on are its
own `src/` (nothing is imported from another line); tests:
`.venv/bin/python -m pytest project/cjk_anima_scale/tests`.
