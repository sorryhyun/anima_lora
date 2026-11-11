# C v2: the copy / fill block with the tone term, EMA and B&W pairs

Date: 2026-11-08. Follow-up to [p1_c_fill.md](p1_c_fill.md) and [p2_sanitize_ft.md](p2_sanitize_ft.md).

Result in one line: **C v2 copies as well as C and fixes C's darkening on `usual_case1`, but
under `cut` it letters the large masks (pseudo-English in the balloons, a pasted manga panel)
where C left blank ground, and it brightens three pages of five by about 1 L; four things
changed at once, so none of this is attributed yet.**

## Setup

- **Data** (`src/build_c_data.py --bw`, `post_image_dataset/moss/c_fill_illu_bw`): C's 1,667
  colour identity pairs plus a `bw__` pair per illustration on its colorize B&W condition
  (latent = reference). The `manga__` conditions are left out. 481 of the 1,667
  illustrations carry a text tag (`text` 318, `speech bubble` 106, `artist name` 76, …), so
  the B&W half doubles the text-bearing pages.
- **Tone term** (`library/moss/train.py`): an identity pair has no edit region, so the tone rest
  is the whole page; the step's hole rows are now removed from it.
- **Run**: [`_archive/configs/lora_blocks/c_fill_v2_s800.toml`](../configs/c_fill_v2_s800.toml). C's recipe, plus
  `tone_weight 0.003` from half-way, `ema_window 0.1`, 800 steps, swap 34, budget 0.15.
  2.36 s/step, 31.5 min, peak 13.28 GB, a hole on 718 steps (cover mean 0.148).
  `output/ckpt_moss_i2i/c_fill_v2_s800[/…]`.
- **Eval**: `ref_gate.py --part fill`, the five sanitize pages, seeds 0–2, `ctrl` / `cut`
  (`output/tests/ref_gate/20261108-c_fill_v2/v2`). C on `hard_to_sanitize2` / `3`, the two
  pages P1 did not render (`…/v1`); its other three pages are P1's `20261107-1911-fill`.

## Outside the mask, ranges over seeds 0–2

| page (mask cells) | model | arm | ΔL out | Δb out | L1 out | L1 out vs ctrl |
|---|---|---|---|---|---|---|
| usual_case1 (52) | C | ctrl | −0.79 … −0.63 | −0.25 … −0.18 | 5.90–6.16 | |
| | C v2 | ctrl | −0.28 … −0.13 | −0.34 … −0.28 | 5.61–5.69 | |
| | C v2 | cut | −0.25 … −0.14 | −0.36 … −0.26 | 5.62–5.71 | 0.38–0.42 |
| usual_case2 (47) | C | ctrl | −0.03 … +0.24 | −0.48 … −0.38 | 6.31–6.37 | |
| | C v2 | ctrl | +0.80 … +1.05 | −0.81 … −0.75 | 6.61–6.95 | |
| hard_to_sanitize1 (366) | C | ctrl | +0.01 … +0.08 | −0.32 … −0.27 | 6.14–6.19 | |
| | C v2 | ctrl | −0.30 … −0.19 | −0.39 … −0.35 | 6.31–6.37 | |
| | C v2 | cut | −0.46 … −0.28 | −0.17 … −0.02 | 6.35–6.47 | 0.79–0.86 |
| hard_to_sanitize2 (1,607) | C | ctrl | −0.54 … −0.39 | −0.21 … −0.14 | 12.95–13.21 | |
| | C | cut | −1.57 … −1.14 | −0.57 … +0.21 | 15.01–15.91 | 4.51–5.64 |
| | C v2 | ctrl | +1.07 … +1.13 | −0.12 … −0.03 | 12.87–12.95 | |
| | C v2 | cut | +0.37 … +1.80 | −0.61 … +0.04 | 13.28–14.15 | 2.07–4.17 |
| hard_to_sanitize3 (1,549) | C | ctrl | +0.05 … +0.27 | −0.10 … −0.02 | 5.60–5.67 | |
| | C | cut | −0.88 … +0.99 | −1.57 … −0.71 | 7.42–8.75 | 3.82–5.16 |
| | C v2 | ctrl | +1.05 … +1.15 | −0.44 … −0.38 | 6.30–6.48 | |
| | C v2 | cut | +1.13 … +2.56 | −1.48 … −1.05 | 6.98–9.47 | 2.02–4.21 |

sanitize v3 alone, for scale: L1 out 5.55–5.60 on `usual_case1`.

## The tone term in training

Logged every 10 steps (10 rows per bin):

| steps | tone, median (min–max) | g_tone / g_fm, median |
|---|---|---|
| 1–100 | 3.89 (2.40–23.55) | 2.03 |
| 101–200 | 3.64 (0.29–13.29) | 1.81 |
| 201–300 | 2.00 (0.72–2.74) | 2.34 |
| 301–400 | 1.55 (0.32–4.65) | 2.41 |
| 401–500 | 2.71 (0.92–11.29) | 3.11 |
| 501–600 | 2.48 (1.14–7.33) | 2.77 |
| 601–700 | 1.75 (0.37–5.06) | 3.42 |
| 701–800 | 1.44 (0.45–2.71) | 2.24 |

At weight 0.003 the tone gradient in output space is 2–3× the flow-matching one, and the two
are near orthogonal (g_cos ≈ 0.02). The term is on x̂₀ − x₀ = σ·(target − pred), so it
swings with σ step to step; the median drifts down only slowly after it is switched on at 400.

## Claims

1. **The copy holds, and C's darkening is gone.** `usual_case1` ΔL −0.7 → −0.2, L1 out
   6.0 → 5.65, at v3's level. Small boxed marks still go under `cut` (`hard_to_sanitize1`,
   both stamps, 3 / 3; a headband appears on seed 2).
2. **C v2 fills large text-shaped masks with lettering.** On `hard_to_sanitize2` it writes
   pseudo-English into every masked balloon; on `hard_to_sanitize3` it pastes a manga panel
   with Japanese text and a lettered box. C fills the same masks with blank balloons and
   background. The likely source is the text-bearing illustrations: a text-layout hole that
   lands on their lettering is trained to letter, and v2 has twice those pages and twice the
   steps.
3. **The tone term does not settle the tone.** It fixes `usual_case1` and leaves three pages
   1 L brighter than C (`usual_case2` also 0.4 cooler in b).

## Not tested

- Which change caused what: B&W pairs, tone, EMA and 800 steps moved together. C v2's s200–s600
  saves are not rendered (s400 against C's s400 would separate steps from data).
- A text-free data set (tag filter, or AnimeText boxes kept out of the hole and the loss).
- The tone term's σ dependence (a σ cut-off or σ-normalised error).
- C v2 as the init for sanitize / uncensor.
