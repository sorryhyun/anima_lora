# Pages against VAE(x̂0) on the ×1 base, `x1pxh_1127_2135` (2026-11-27)

A 3,000-rollout finetune of `x1dll_1123_0821` with the pages' target at the base points changed
from the page to the VAE's decode of x̂0, read against `x1dll_1123_0821` as it is. Settings,
question, read and bars: `../configs/x1pxh_1127_2135.toml`.

## Facts

- Job `20261127-213400-f37545`, rc 0, 36.4 min, 0.68 s per rollout, peak 10.1 GB; 1,008 of the
  3,000 rollouts were pages. Head (local): `output/pixel_head/x1pxh_1127_2135/head.safetensors`.
  Code: `page_target` in `scripts/pixel_head/train/_x1_run.py` (default `real`, the old path).
- Held-out, teacher-forced, means; `x1dll` → `x1pxh`, with VAE(x̂0)'s own column last. DINO
  distance × 10⁴. One seed, no twin on the DINO recipe: spread unknown.

  | σ, set | dB against VAE(x̂0) | dB against the original | DINO to the original | VAE(x̂0): dB / DINO to the original |
  |---|---|---|---|---|
  | 0.0577 test 30 | 37.69 → 37.83 | 34.00 → 34.02 | 1.09 → 1.11 | 35.10 / 1.35 |
  | 0.0577 text 8 | 32.36 → 32.56 | 29.00 → 29.03 | 0.85 → 0.86 | 30.25 / 0.83 |
  | 0.0577 manga 10 | 30.93 → 31.46 | 26.23 → 26.29 | 1.28 → 1.49 | 26.95 / 1.50 |
  | 0.1364 test 30 | 37.83 → 38.06 | 31.18 → 31.16 | 1.59 → 1.61 | 31.31 / 2.04 |
  | 0.1364 manga 10 | 30.89 → 31.59 | 24.02 → 23.98 | 1.78 → 2.01 | 23.93 / 2.10 |
  | turbo 0.5 manga 10 | 30.58 → 30.62 | 17.41 → 17.40 | 5.27 → 5.32 | 17.22 / 5.23 |

  Per image on manga: against VAE(x̂0) up on 10 / 10 at both σ (+0.20 … +1.10 dB at 0.0577);
  DINO distance to the original up on 10 / 10. On test at σ 0.0577: against VAE(x̂0) up on
  27 / 30 (−0.05 … +0.31).
- Generated, P3's 16 prompts × seeds 0, 1, base 28 steps CFG 4 (σ_last 0.1), both heads on the
  same renders: `x1pxhb28_1127_2135` (job `20261127-213400-bc6015`, rc 0), texture read
  `tex_1127_2220`. Local: `bench/pixel_readout/results/20261127-2210-x1pxhb28_1127_2135/`,
  `…/20261127-2217-tex_1127_2220/` (pair sheets in `pairs_x1dll_1123_0821_x1pxh_1127_2135/`).

  | read, colour 26 / mono 6 | VAE decode | `x1dll_1123_0821` | `x1pxh_1127_2135` |
  |---|---|---|---|
  | dB against the VAE decode (mean) | | 33.76 / 25.69 | 33.92 / 26.45 |
  | high-pass amplitude, share of the VAE's (median) | 1 / 1 | 0.909 / 0.882 | 0.922 / 0.917 |
  | stamp at 16 px, levels (median) | 0.426 / 1.102 | 0.565 / 1.451 | 0.559 / 1.176 |
  | stamp at 8 px (median) | 0.307 / 0.751 | 0.476 / 1.241 | 0.474 / 0.861 |

  dB up on 26 / 26 colour renders (+0.04 … +0.36) and 6 / 6 mono (+0.25 … +1.42). `manga_page`
  s0 / s1: dB 24.89 → 26.31 / 25.13 → 26.48; 16-px stamp 4.66 → 2.72 / 4.94 → 3.47, the VAE's
  1.49 / 2.77. Amplitude above `x1dll` on 31 / 32 renders, 16-px stamp on 10 / 32, 8-px on 6 / 32.
- Claude's read, labels visible, the zoom rows of `sheet_base_13_manga_page_s0.png` and `_s1.png`
  (VAE decode | `x1dll` | `x1pxh`). Light tone on cloth (s0): the VAE has sparse dots of varying
  density; `x1dll` a regular fine grid over the whole tone; `x1pxh` a fainter, finer screen,
  nearer flat grey, without the VAE's separate dots. Dark tone on hair (s0): the VAE a dense
  halftone; `x1dll` a coarse irregular weave; `x1pxh` an even fine screen at the VAE's tone. Light
  tone (s1): the VAE a diagonal screen; `x1dll` a dot grid; `x1pxh` a faint diagonal screen.
  Lines are not told apart between the two heads.
- No read by the user yet. Sheets opened for it: `~/x1pxh_manga_page_s0.png`, `_s1.png` (local).

## Interpretation

- By the config's rule the read sits between its first and third outcomes: the grid is weaker and
  every number moved toward the VAE decode, while the VAE's separate dots are still not written.
  The user's read of the sheets decides which.
- The DINO distance to the original on pages rose to VAE(x̂0)'s own (1.49 against 1.50 at σ
  0.0577) with dB against the original level: the head moved from the original's texture to the
  VAE path's on pages, which is what the target asks for.
- The colour rows also rose (26 / 26 renders, with no change to their target), so part of the
  mono gain is the 3,000 extra rollouts. The stamp fell on mono only (colour 0.565 → 0.559), which
  the extra rollouts alone would not single out; the same finetune with `page_target = "real"`
  (about 36 min) would separate the two.
- About 500 of the rollouts were pages at a base point. The manga rows against VAE(x̂0) were
  still rising at the last read (30.88 / 31.25 / 31.46 at 1,000 / 2,000 / 3,000), under a cosine
  that ends there; a full-length base with `page_target = "xhat0"` is the arm that says how far
  the dots come back.
