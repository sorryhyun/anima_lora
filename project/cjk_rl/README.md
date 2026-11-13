# cjk_rl — OCR-reward RL on the JA vocab-pack rows

Opened 2026-11-13 out of the reseed line (`../cjk_anima_reseed/`). GRPO on jp_v1's rows with
the 4-step turbo stack (merged14 + Hub turbo v3, cfg 1) as the policy and a reward read off
its own renders by the tuned VL reader. The target is sentence fidelity: the asked string,
in order, at its length, with no invented dialogue around it.

- [proposal.md](proposal.md) — design, reward, arms R / R-credit / A, phases P1–P3 and their
  gates, the SFX track.
- [reports/](reports/) — finished probes:
  - [ruler_turbo_2026_11_13.md](reports/ruler_turbo_2026_11_13.md) — the 64-string ruler on
    base / merged14 / merged14 + v3: the soup loses the string, the turbo is a fair proxy.
  - [p0_2026_11_13.md](reports/p0_2026_11_13.md) — P0: sampler (SDE a = 0.7), group spread
    (passes), cost of one update.
  - [reward_reader_2026_11_13.md](reports/reward_reader_2026_11_13.md) — the reader beside the
    update: int8 / fp8 dropped, group-batched read without the whole page, tuned reader only.
- `src/pool.py` — P1's prompt pool: the training captions with each quoted OCR string swapped
  for another OCR string of its length (Manga109-s, or another caption's), nothing else
  changed; → `output/cjk_rl/pool/pool.json`.
- `src/reward.py` — the read terms: r_seq over every asked string (one bubble each), g_p over
  all of them.
- `probes/` — `grpo_p0.py` (P0), `reader_batch.py` (the reward reader). Both run on the
  reseed line's package, ruler and readers (`../cjk_anima_reseed/{reseed,src}`).
- `results/` (gitignored) — one directory per probe run; renders under `output/cjk_rl/`.

Every read is the tuned PaddleOCR-VL-1.6 reader (`anime_tools.ocr.sfx.SfxReader`); stock
VL16 is not a reader.

State (11-13): nothing trained. The prompt pool (§ 3.5) and the multi-target r_seq (§ 3.3)
are built; next is the P1 loop.
