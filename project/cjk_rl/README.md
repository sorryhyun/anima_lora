# cjk_rl — OCR-reward RL on the JA vocab-pack rows

Opened 2026-11-13 out of the reseed line (`../cjk_anima_reseed/`). GRPO on jp_v1's rows with
the 4-step turbo stack (merged14 + Hub turbo v3, cfg 1) as the policy and a reward read off
its own renders by the tuned VL reader. The target is sentence fidelity: the asked string,
in order, at its length, with no invented dialogue around it.

- [proposal.md](proposal.md) — design, reward, arms R / A (GRPO), D (reader-NLL reward
  gradient) and R-credit, the budget against Flow-GRPO, phases P1–P3 / D0–D1 and their gates,
  the SFX track.
- [reports/](reports/) — finished probes:
  - [ruler_turbo_2026_11_13.md](reports/ruler_turbo_2026_11_13.md) — the 64-string ruler on
    base / merged14 / merged14 + v3: the soup loses the string, the turbo is a fair proxy.
  - [p0_2026_11_13.md](reports/p0_2026_11_13.md) — P0: sampler (SDE a = 0.7), group spread
    (passes), cost of one update.
  - [p1_2026_11_13.md](reports/p1_2026_11_13.md) — P1: 300 GRPO groups on the rows change
    nothing on the ruler; arm R fails.
  - [reward_reader_2026_11_13.md](reports/reward_reader_2026_11_13.md) — the reader beside the
    update: int8 / fp8 dropped, group-batched read without the whole page, tuned reader only.
- `src/pool.py` — P1's prompt pool: the training captions with each quoted OCR string swapped
  for another OCR string of its length (Manga109-s, or another caption's), nothing else
  changed; → `output/cjk_rl/pool/pool.json`.
- `src/reward.py` — the read terms: r_seq over every asked string (one bubble each), g_p over
  all of them.
- `src/grpo.py` — the GRPO loop (arm R).
- `probes/` — `grpo_p0.py` (P0), `reader_batch.py` (the reward reader). Both run on the
  reseed line's package, ruler and readers (`../cjk_anima_reseed/{reseed,src}`).
- `results/` (gitignored) — one directory per probe run; renders under `output/cjk_rl/`.

Every read is the tuned PaddleOCR-VL-1.6 reader (`anime_tools.ocr.reader.OcrReader`); stock
VL16 is not a reader.

State (11-13): P1 arm R failed (`grpo_1113_1940`, rows `rl_p1`). Next: arm A and the D0
probes (proposal § 4).
