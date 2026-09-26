# piece_only — run0925_300f's pieces on scene_piece items only; the one-file shape's parity (2026-09-26)

Two reads. § 1 is the refactor's test (plan.md § 6-3) done without the 3.2 h
retrain: the one-file code replays run0925_300f and matches it. § 2 is
`next_2026_09_25.md` § 4a step (2): the same 300 pieces and freeze, trained on
`scene_piece` items only. **Verdict: the mix was not the cause of doubling.**
Dropping `scene_sentence` / `scene_short` / `grid_string` buys a little `word`
exact (4 → 9 / 36), holds piece identity (official tied, contained −12),
and costs sentence strings (sent contained 11 → 4, おしい 2 → 0). `builder.TABLE`
is unchanged.

## 1. Parity — `experiments/parity_300f/`

Envelopes `experiments/parity_300f/results/20260926-0043-plan/` (CPU) and
`…/20260926-0044-gpu1/` (job `20260926-004435-b563ff`, 60 min). The run is
`output/cjk_anima_scale/parity_300f_gpu1/`: its `data/` symlinks the old
`data_joint0507_0305_run0925_300f/` + a `vocabs.json`; nothing is retrained
past step 50.

- **plan (CPU)**: the vocabs file's 300 = the old `words.json`; `train.plan`
  on the old data vs the old `train_record.json` — 19 keys equal (n_rows 300,
  n_touched 300, n_context 1659, warmup 2700, the two bands, every trainer
  constant), 27 000 steps both; `rows.merge_seed` on the old vocabs-only
  `trained.pt` vs the old ctx overlay — 2 274 rows, same ids, max |Δ| **0**.
- **steps (GPU, 50 steps)**: step 1 identical on every logged field (loss
  0.08428, in_box 0.13131, out_box 0.058297, lr); step 50 loss 0.11182 vs
  0.11233, warm_drift 0.0091326 vs 0.0091346 (bf16 + compile); lr exact at
  every log step.
- **eval (GPU)**: the old final rows, merged, under the new `eval.run`,
  against the old ctx arm's reads (trained) and the seed's reads of record
  (floor), render by render on the shared prompts × seeds:

  | ruler | floor new / old (official · contained) | trained new / old |
  |---|---|---|
  | eval (word + en, 60) | 29 / 30 · 35 / 35 | 28 / 28 · 44 / 42 |
  | native あ い (64) | – (no old floor) | 47 / 44 · 61 / 61 |
  | piece (4 shared pieces, 128) | 3 / 3 · 12 / 13 | 4 / 7 · 37 / 33 |
  | sent (80) | 8 / 8 · 10 / 10 | 5 / 5 · 11 / 10 |
  | target (14) | 0 / 0 | 0 / 0 |

  Renders are not bit-repeatable across jobs: on the floor — the same rows
  in both jobs — only 37/60, 32/128, 39/80 renders read identically. So a
  60–128-render cell moves ±3–4 on noise; the script's ±2 flag is too tight
  for those. The two cells past it go opposite ways (native +3; piece
  official −3, contained +4). **Parity holds**; § 6-3's retrain is not run.

What parity does not cover: the full 27 000-step trajectory (the old drift
curve 0.79 → 1.39 → 1.74 → 1.66 is the comparator any new run gets) and the
data builder, which was verified record- and pixel-identical on 2026-09-25.

## 2. The arm — `experiments/piece_only/`, `run0926_300f_sp`

Job `20260926-014508-c997ac` (267 min: data 4 min, train 192 min, eval
~60 min). Everything as run0925_300f except the mix: each piece band group
cut to its `scene_piece` tier — b0507 10 000 items at px 38 (30–53), σ
0.5–0.7; b0305 10 000 items at px 19 (15–23) in small bubbles, σ 0.3–0.5.
Same 300 vocabs, singles frozen at the seed, fixed trainer (μ 0, lr 1e-3,
grid_box, 90 steps/row). Drift 0.74 (2 500) → 1.33 (5 000) → 1.72 (12 500)
→ 1.62 (end), cos 0.49 — run0925_300f's curve, so displacement is the same;
only what it bought differs.

Comparator: run0925_300f's rows under this same eval (the parity run's
trained arm), so every cell is the same ruler, strings and render seeds.
The piece ruler is `eval.piece_vocabs` (new, § 4): the pieces the `read`
strings tokenize into (しい った ちょっと こんにちは) then the `word`
group's (あと きて こう こと).

| ruler | seed floor | 300f | **scene_piece only** |
|---|---|---|---|
| piece (8 × en/swap × 16) | 3 · c27 | 16 · c88 | **16 · c76** |
| word exact · contained (36) | 6 · c11 | 4 · c20 | **9 · c17** |
| sent (5 × 16) | 8 · c10 | 5 · c11 | **3 · c4** |
| target (14) | 0 | 0 | 0 |
| en (24) / native あ い (64) | 24 / 47 | 24 / 47 | 24 / 47 |

Per piece (official · contained, of 16), where they move:

| piece | 300f en | sp en | 300f swap | sp swap |
|---|---|---|---|---|
| しい | 1 · 10 | 0 · 9 | 2 · 9 | 0 · **0** |
| った | 1 · 13 | 1 · **8** | 0 · 0 | 0 · 0 |
| ちょっと | 0 · 5 | 0 · **0** | 0 · 0 | 0 · 0 |
| きて | 1 · 4 | 1 · **10** | 2 · 3 | 0 · 3 |
| あと / こう / こと | 1 / 2 / 0 · 6 / 8 / 13 | 2 / 3 / 2 · 6 / 10 / 12 | 3 / 3 / 0 · 9 / 6 / 2 | 5 / 2 / 0 · 8 / 7 / 2 |

- **Piece identity: level.** Official tied at 16. Contained −12, all of it
  on the pieces the `read` strings carry (しい swap 9 → 0, った 13 → 8,
  ちょっと 5 → 0); the `word`-group pieces hold or rise (きて en 4 → 10).
  A piece trained only alone in a bubble reads worse where it sits inside
  a line.
- **Doubling: down a little, not gone.** `word` exact 4 → 9 / 36 (contained
  20 → 17). The misses are still doubling-dominated: ああとと, ここと, ことと,
  ききいて, はなない, 下ささい, これれら. The sentence / short / grid share
  was not the doubling source — as `floor_score.md` said, the seed's own
  misses double before any training.
- **Sentence assembly: worse, as predicted.** sent 5 → 3, contained 11 → 4;
  おしい 5 (floor) → 2 (300f) → **0**. はい (two frozen singles) 3 = floor —
  the freeze control holds.
- **target**: 0 / 14 on all three.

One seed each; read ±3–4 per 64–128-render cell as noise (§ 1). The `word`
gain (+5 / 36) and the sent loss (−7 contained) are past that; the piece
official tie is not a difference. *(Re-read paired 2026-09-26,
`spell_2026_09_26.md` § 8: the `word` gain is not significant — 300f → sp
+8 / −3 renders, p 0.23; the sent loss is, per render, but sits in おしい.)*

## 3. What it closes, what stays open

- **Closed**: "the sentence / short / grid share causes the doubling / wipe"
  (`next_2026_09_25.md` § 4a's suspect). The mix trades sentence strings for a few
  `word` exacts; it is not the lever. `builder.TABLE` keeps its piece mix.
- **Open**: doubling itself. It predates training (seed misses double) and
  survives both mixes at the same displacement, so it looks like a property
  of the rows' starting point or the base model's rendering of a repeated
  kana (ああ / ここ / きき / なな / ささ), not of the data. Next read, before
  a new arm: the seed's `word` misses (`floor_score.md` reads) and both
  runs' misses by glyph count and by which glyph doubles (first vs last,
  hiragana vs kanji), to tell a length / stop problem from a per-glyph one.
- **Open**: 3+-glyph pieces (ちょっと, こんにちは) — 0 on every arm.

## 4. Code this read added

- `cjk_scale/rows.py::merge_seed` — the save-time merge, shared by
  `Rows.state_dict` and the parity plan / eval legs.
- `cjk_scale/train.py::plan` — the rows split + schedule + record without a
  model; `train(rc, *, data, out, max_steps)` for experiments (`scale.py`
  passes none).
- `cjk_scale/eval.py` — the **piece ruler** (`native_piece/`, `en` + `swap`,
  `PIECE_N` 8, `piece_vocabs`), on by default for any run whose vocabs hold a
  multi-glyph vocab: reports/piece_2026_09_25.md's rule ("read piece before
  calling a piece run dead") as code.
- `cjk_scale/builder.py::build(…, table=TABLE)` — an experiment can build
  on another recipe table; `scale.py` never does.
- Tests: `test_piece_ruler`, `test_merge_seed`.

Repro:
```
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
.venv/bin/python project/cjk_anima_scale/experiments/parity_300f/run_exp.py --label plan --dry_run
make daemon-run ARGS="--queue --stall-timeout 0 project/cjk_anima_scale/experiments/parity_300f/run_exp.py --label gpu1 --legs steps eval --steps 50"
make daemon-run ARGS="--queue --stall-timeout 0 project/cjk_anima_scale/experiments/piece_only/run_exp.py --label sp1 --legs data train eval --workers 10"
```
