# cjk_rl — OCR-reward RL on the JA vocab-pack rows

Opened 2026-11-13 out of the reseed line (`../cjk_anima_reseed/`). GRPO on jp_v1's rows with
the 4-step turbo stack (merged14 + Hub turbo v3, cfg 1) as the policy and a reward read off
its own renders by the tuned VL reader. The target is sentence fidelity: the asked string,
in order, at its length, with no invented dialogue around it.

- [proposal.md](proposal.md) — design, reward, arms R / A (GRPO), D (reader-NLL reward
  gradient) and R-credit, the budget against Flow-GRPO, phases P1–P3 / D0–D1 and their gates,
  the SFX track.
- [proposal_glyph.md](proposal_glyph.md) — the glyph-first stage (user, 11-25): a tagless
  `Text reads as "<glyph>"` caption on the turbo, pixel-head decode, the reader's score on the
  row's own glyph and against lettering that was not asked for; arms G-D / G-R, phases G0 – G3.
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
- `src/pool.py` — P1's prompt pool: the training captions with each quoted OCR string swapped
  for another OCR string of its length (Manga109-s, or another caption's), nothing else
  changed; → `output/cjk_rl/pool/pool.json`.
- `src/reward.py` — the read terms: r_seq over every asked string (one bubble each), g_p over
  all of them.
- `src/grpo.py` — the GRPO loop (arm R).
- `probes/` — `grpo_p0.py` (P0), `reader_batch.py` (the reward reader). Both run on the
  reseed line's package, ruler and readers (`../cjk_anima_reseed/{reseed,src}`).
  `position_read.py` (CPU) reads the ruler results' `renders.json`. `turbo_read.py` renders
  and reads the ruler and a kanji grid on `m15v11` (pixel-head decode by default).
- `results/` (gitignored) — one directory per probe run; renders under `output/cjk_rl/`.

Every read is the tuned PaddleOCR-VL-1.6 reader (`anime_tools.ocr.reader.OcrReader`); stock
VL16 is not a reader.

State (11-25): P1 arm R failed (`grpo_1113_1940`, rows `rl_p1`). On merged15 + the turbo v1.1
delta the rows read near their null and the EN reference at cer 0.83
(`reports/turbo_read_2026_11_25.md`). Next: `proposal_glyph.md` G0, on the user's word; arm A
and the D0 probes (proposal § 4) wait behind it.

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
