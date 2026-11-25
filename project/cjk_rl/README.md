# cjk_rl — OCR-reward RL on the JA vocab-pack rows

Opened 2026-11-13 out of the reseed line (`../cjk_anima_reseed/`). GRPO on jp_v1's rows with
the 4-step turbo stack (merged14 + Hub turbo v3, cfg 1) as the policy and a reward read off
its own renders by the tuned VL reader. The target is sentence fidelity: the asked string,
in order, at its length, with no invented dialogue around it.

- [proposal.md](proposal.md) — design, reward, arms R / A (GRPO), D (reader-NLL reward
  gradient) and R-credit, the budget against Flow-GRPO, phases P1–P3 / D0–D1 and their gates,
  the SFX track.
- [plan.md](plan.md) — the glyph stage on G-D (the reader's loss carried back to the rows):
  the path, G0d and G1's micro arm done; next the sixteen rows to their ceiling, then the
  hiragana.
- [proposal_glyph.md](proposal_glyph.md) — the glyph-first stage (user, 11-25) after the
  hiragana: the katakana, the frequent kanji and words (G2), transfer and ship reads (G3).
- [reports/](reports/) — finished probes:
  - [ruler_turbo_2026_11_13.md](reports/ruler_turbo_2026_11_13.md) — the 64-string ruler on
    base / merged14 / merged14 + v3: the soup loses the string, the turbo is a fair proxy.
  - [p0_2026_11_13.md](reports/p0_2026_11_13.md) — P0: sampler (SDE a = 0.7), group spread
    (passes), cost of one update.
  - [p1_2026_11_13.md](reports/p1_2026_11_13.md) — P1: 300 GRPO groups on the rows change
    nothing on the ruler; arm R fails.
  - [reward_reader_2026_11_13.md](reports/reward_reader_2026_11_13.md) — the reader beside the
    update: int8 / fp8 dropped, group-batched read without the whole page, tuned reader only.
  - [position_read_2026_11_25.md](reports/position_read_2026_11_25.md) — the ruler's reads of
    record aligned per bubble: hits by position, substitutions, and the EN references' own
    text on the base, merged14 and the turbo.
  - [turbo_read_2026_11_25.md](reports/turbo_read_2026_11_25.md) — fresh renders on merged15 +
    the turbo v1.1 delta: the ruler at four seeds with its EN references, and a kanji grid.
  - [glyph_g0_2026_11_25.md](reports/glyph_g0_2026_11_25.md) — G0a – c of the glyph proposal:
    the tagless frame with and without `speech bubble` against one-token EN words, the reward
    terms' spread inside a group, the pixel head against the VAE.
  - [glyph_scenes_2026_11_25.md](reports/glyph_scenes_2026_11_25.md) — the glyph stage's scene
    pool picked by render (128 text-free captions in two rounds, by artist folder; 25
    selected), and every glyph on the same four scenes.
  - [gd_g0d_2026_11_25.md](reports/gd_g0d_2026_11_25.md) — G0d: G-D's path built and checked,
    its memory and seconds, and the size and direction of the gradient it puts on a row.
  - [g1micro_2026_11_25.md](reports/g1micro_2026_11_25.md) — G1's micro arm: 16 rows, 8 steps
    each on the reader's loss; solo on held-out scenes 59 → 132 of 256.
  - [reader_pair_2026_11_25.md](reports/reader_pair_2026_11_25.md) — hayai v2.7 Supernova
    beside the tuned reader on g1micro's read pages and the crossed grid: hayai counts
    g1micro's solo 60 → 125; the crops the two differ on.
  - [hayai_g0_2026_11_25.md](reports/hayai_g0_2026_11_25.md) — G0h: hayai's cross-entropy
    beside the tuned reader's on G1's path; the two row gradients' sizes and cosine.
- `src/pool.py` — P1's prompt pool: the training captions with each quoted OCR string swapped
  for another OCR string of its length (Manga109-s, or another caption's), nothing else
  changed; → `output/cjk_rl/pool/pool.json`.
- `src/reward.py` — the read terms: r_seq over every asked string (one bubble each), g_p over
  all of them.
- `src/grpo.py` — the GRPO loop (arm R).
- `src/gd.py` — G-D's path: the policy with a step-by-step backward to the rows, and the
  reader's cross-entropy and the detector's score on a page tensor.
- `src/g1.py` — G1's loop on that path (a row steps along the momentum of its renders' mean
  gradient) and its read on held-out scenes with blind sheets.
- `probes/` — `grpo_p0.py` (P0), `reader_batch.py` (the reward reader). Both run on the
  reseed line's package, ruler and readers (`../cjk_anima_reseed/{reseed,src}`).
  `position_read.py` (CPU) reads the ruler results' `renders.json`. `turbo_read.py` renders
  and reads the ruler and a kanji grid on `m15v11` (pixel-head decode by default); its render
  (every caption's text prepared before the loop, 4 images a forward) and read (16 pages a
  call) are the ones `glyph_g0.py` runs on.
  `glyph_g0.py` and `glyph_scenes.py` are the glyph stage's forward probes; `gd_g0d.py`
  measures the row gradient of `src/gd.py`.
  `reader_pair.py` reads pages on disk with the tuned reader and hayai v2.7 on the same
  crops; `hayai_g0.py` measures hayai's loss beside the tuned reader's on G1's path.
- `configs/` — one file a run, its question and gate in the header.
- `assets/glyph_scenes.json` (gitignored) — the selected scene prompts, written by
  `glyph_scenes.py screen`.
- `results/` (gitignored) — one directory per probe run; renders under `output/cjk_rl/`.

Every read is the tuned PaddleOCR-VL-1.6 reader (`anime_tools.ocr.reader.OcrReader`); stock
VL16 is not a reader. hayai v2.7 Supernova was read beside it once (`hayai_1125_1234`). The
user's call (11-25): from the next run on the loss carries hayai's cross-entropy at weight 0.2
(`src/gd.py` `Hayai`, `hayai_weight` in `src/g1.py`, off unless the config sets it).

State (11-25): P1 arm R failed (`grpo_1113_1940`, rows `rl_p1`). On merged15 + the turbo v1.1
delta the rows read near their null and the EN reference at cer 0.83
(`reports/turbo_read_2026_11_25.md`). G0a – c of `proposal_glyph.md` ran
(`reports/glyph_g0_2026_11_25.md`), then the scene screen and the crossed grid
(`reports/glyph_scenes_2026_11_25.md`; selected prompts in `assets/glyph_scenes.json`, local).
The user's call (11-25): G-D is the arm, G-R is out. G-D's path is built and G0d ran
(`g0d_1125_1119`, `reports/gd_g0d_2026_11_25.md`). G1's micro arm ran (`g1micro_1125_1148`,
`reports/g1micro_2026_11_25.md`): solo on held-out scenes 59 → 132 of 256; the user read the
sheets as a success. Next: `plan.md` item 3 (the sixteen rows to their ceiling, x_3 cache on, hayai's
term at 0.2), then item 4 (the 81 hiragana); neither queued.
Arm A and the D0 probes (proposal § 4) wait behind the glyph stage.

## Rules

The user's calls (2026-11-13).

- The goal is the intended sentence (order, length, no invented text), not crisper glyphs.
  Rewards and reads are sequence-level. SFX is in scope and needs a style guard.
- Sampler: SDE a = 0.7.
- Prompt pool: every caption with a JA text / SFX clause, only the quoted strings swapped. It
  holds out the ruler's images and 5-grams and the `$CHARACTER_REF_EXCLUDE` tags. No rating
  filter.
- Rows can learn order (proposal § 2), so arm A is a sample-count fallback, not an order fix.
- Arm A starts with the `src/grpo.py` fixes: summed log π, G 16–24.
- Renders go 4 images a forward (user, 11-25; `probes/turbo_read.py` `BATCH`).
