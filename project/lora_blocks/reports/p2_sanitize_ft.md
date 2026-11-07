# P2: sanitize fine-tuned from C against from kaiming, 300 steps

Date: 2026-11-07. Phase P2 of [proposal.md](../proposal.md), step 4 of [plan.md](../plan.md).

Result in one line: **at 300 steps the run from C removes text on the hard manga pages about as
well as the shipped v3 (2,216 steps), and the run from kaiming does not; neither run holds the
tone: with EMA off the page brightness swings by 2–4 L between saves in both arms.** `cut` on top of either fine-tune adds nothing
on boxed marks and invents content on the large masks.

## Setup

- **Init** (`TrainRequest.init_lora`, new): the adapters load a saved LoRA in place of their
  init; algo, rank, alpha, targets and state keys must match.
- **Runs**: [`configs/sanitize_ft_c_s300.toml`](../configs/sanitize_ft_c_s300.toml) (from
  `c_fill_s400`) and [`configs/sanitize_ft_k_s300.toml`](../configs/sanitize_ft_k_s300.toml)
  (kaiming). The sanitize_copy_s300 recipe (277 pairs, muon, lr 5e-5, rank 32, int8 plan, EMA
  off, no gate in training) with v3's tone term (0.003 from step 150). Swap 36, budget 0.15:
  2.19 s/step, peak 12.3 GB. `output/ckpt_moss_i2i/sanitize_ft_{c,k}_s300[/…_sN]`.
- **Eval**: `ref_gate.py --part sanitize --arms ctrl,cut --lora …`, the five sanitize eval
  pages, seeds 0–2, the sanitize prompt and generate settings, AnimeText masks.
  `output/tests/ref_gate/20261107-2202-p2/{v3,kft,cft}`; `cmp_<page>_s0.png` puts the three
  models side by side at seed 0 (top row `ctrl`, bottom `cut`).

## Removal, `ctrl` (no gate, the normal use), seed 0 by eye

| page | v3 | K-ft | C-ft |
|---|---|---|---|
| usual_case1 (watermark) | kept | removed | removed |
| usual_case2 (faint URL) | — | — | — (none visible at sheet scale) |
| hard_to_sanitize1 (stamps, bubbles) | both stamps removed, anger bubble kept | stamps and both bubbles removed, a faint smudge at the collar | stamps and bubbles removed, a small scribble where the anger bubble was |
| hard_to_sanitize2 (manga, red bubbles) | clean | red bubble shapes left as red blocks, smears at top left | clean; one red bubble turned into red cloth |
| hard_to_sanitize3 (lettering, signature) | clean | lettering gone, ring marks on the sleeve, a red blob by the hand | clean, as v3 |

## Outside the mask, `ctrl`, mean over seeds 0–2

| page | ΔL v3 / K / C | Δb v3 / K / C | L1 v3 / K / C |
|---|---|---|---|
| usual_case1 | −0.16 / −1.36 / +0.75 | −0.62 / +0.17 / +0.04 | 5.59 / 7.43 / 6.66 |
| usual_case2 | +0.33 / −1.52 / +1.76 | −0.33 / +1.05 / −0.25 | 6.32 / 8.81 / 8.42 |
| hard_to_sanitize1 | −0.12 / −0.51 / +3.04 | +0.10 / +0.79 / +0.33 | 6.18 / 7.77 / 10.96 |
| hard_to_sanitize2 | −0.00 / −2.38 / +1.30 | −0.37 / +0.01 / −0.94 | 16.45 / 18.03 / 16.56 |
| hard_to_sanitize3 | −0.47 / −1.81 / +1.23 | −0.31 / +0.31 / −0.84 | 5.81 / 8.20 / 6.69 |

## Claims

1. **C buys removal quality at 300 steps.** On the two manga pages K-ft leaves bubble-shaped
   blocks and stray marks, and C-ft is as clean as v3. On `usual_case1` both remove the watermark
   v3 keeps.
2. **The tone is not reached by either 300-step run, and the sign at step 300 is a snapshot of
   a swing, not a property of the init.** C-ft ends brighter on every page and K-ft darker, but
   the intermediate saves (`steps/`, `usual_case1` / `hard_to_sanitize1`, seeds 0–2) swing by
   2–4 L between saves 100 steps apart, in both arms, with the seeds tight inside each save:

   | ΔL out, usual_case1 / hard_to_sanitize1 | s100 | s200 | s300 |
   |---|---|---|---|
   | C-ft | −0.11 / +1.49 | −2.17 / −0.87 | +0.75 / +3.04 |
   | K-ft | +0.23 / +1.75 | — | −1.36 / −0.51 |
   | C alone (sanitize prompt) | −0.66 / −0.09 | | |
   | v3 | −0.16 / −0.12 | | |

   The tone term cannot push the wrong way (a squared norm of the rest-region mean offset), and
   the swing is there at s100, before it switches on. Both arms ran with EMA off; v3's
   `ema_window 0.1` averages such a swing away. L1 outside: C-ft below K-ft on four pages of
   five, both above v3 everywhere.
3. **`cut` is not a sanitize-LoRA add-on.** On the boxed marks the fine-tunes already remove, it
   changes nothing (ΔL / L1 within the seed spread). On the large masks (`hard_to_sanitize2` /
   `3`, ~1,550–1,600 cells) it makes every model invent: v3 green socks and pasted photo
   rectangles, K-ft dark smears, C-ft the most plausible fill but new hands and redrawn anatomy.
   P1's `cut` result holds for C alone on small boxes.

## What it means

- The copy block's win shows up as removal quality at equal short steps, as findings/copy.md
  predicted (a fine-tune that starts past the phase change spends its steps on the edit).
- The open part is the tone, and the first thing to fix is the confound: rerun both arms with
  v3's EMA (`ema_window 0.1`) before reading anything into the sign.

## Not tested

- One run per arm, one hole-draw seed for C; seeds 1–2 read through the numbers, not by eye.
- The bf16 plan (v3 is bf16 with EMA; both arms here int8 without EMA).
- The 50-pair arm (C-ft50) was dropped for this pass.
