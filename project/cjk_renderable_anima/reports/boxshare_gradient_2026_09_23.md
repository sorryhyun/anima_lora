# boxshare_gradient — what the box share does to a row's gradient (2026-09-23)

> **The in-box term owns the row gradient at any share above ≈ 0.1.** On
> stage0507's data (240 scene items: 48 singles, 120 pieces, 72 short lines,
> 31–36 px, seed table `rows_step1_0921m_merge`, σ 0.5–0.7, 3 draws each),
> ‖g_in‖ is 25–100 × ‖g_out‖ for every glyph count, and the two are
> near-orthogonal (cos +0.01 … +0.11). The in-box fraction φ of the resolved
> row gradient is 0.89–0.99 for every curve between 0.25 and 0.75 — flat,
> the probe's linear `min(0.25 n, 0.75)`, or the log 0.25 → 0.5 — so the
> **shape of the curve over the glyph count is close to inert** on this
> measure. The share starts to bite below 0.1: at stage0309's 0.05 floor a
> single's row gradient is 22 % background, a 5–7-glyph line's 43 %; at
> 0.02 both are about half. No training was run; this is the gradient
> composition only, read by `cjk_anima_scale/cjk_scale/boxprobe.py`.

Job `20260923-183317-81fd08`; reads and report under
`output/cjk_anima_scale/boxprobe_scale_stage0507_bp1/`.

## The read

Per scene item and σ draw, `mean_in` (cells under the text box) and
`mean_out` (the rest) are backpropagated separately onto the item's ext
rows; the row gradient under a share `s` is `s·g_in + (1 − s)·g_out`.

| glyphs | reads | px | rows | ‖g_in‖ | ‖g_out‖ | ratio | cos | box cells |
|---|---|---|---|---|---|---|---|---|
| 1 | 141 | 31 | 1 | 0.645 | 0.0099 | 0.01 | +0.11 | 25 |
| 2 | 201 | 36 | 1 | 0.449 | 0.0096 | 0.02 | +0.06 | 60 |
| 3 | 138 | 36 | 1 | 0.382 | 0.0106 | 0.03 | +0.01 | 70 |
| 4 | 117 | 31 | 3 | 0.577 | 0.0141 | 0.02 | +0.06 | 80 |
| 5–7 | 123 | 33 | 4 | 0.382 | 0.0159 | 0.04 | +0.03 | 110 |

Medians. `ratio` = ‖g_out‖ / ‖g_in‖ (p10–p90: 0.006–0.12 across bins).
The 8+ bin is empty: stage0507's short lines top out at 7 glyphs.

In-box fraction φ of the row gradient by share (medians):

| glyphs | s 0.02 | s 0.05 | s 0.10 | s 0.25 | linear (0.25 n, cap 0.75) | log (0.25 → 0.5 at 8) |
|---|---|---|---|---|---|---|
| 1 | 0.58 | 0.78 | 0.88 | 0.96 | 0.96 | 0.96 |
| 2 | 0.48 | 0.71 | 0.83 | 0.94 | 0.98 | 0.96 |
| 3 | 0.41 | 0.64 | 0.79 | 0.92 | 0.99 | 0.95 |
| 4 | 0.46 | 0.68 | 0.82 | 0.93 | 0.99 | 0.97 |
| 5–7 | 0.34 | 0.57 | 0.74 | 0.89 | 0.99 | 0.95 |

σ halves inside the band agree (s* 0.01–0.04 on both sides).

## Reading

- **Why the background is so small**: `mean_out` averages ≈ 4 000 cells
  the row barely touches, so its gradient on the row is 1/50 of the box's.
  With `cos ≈ 0` it is noise on the row, not a pull against the glyph.
- **What the share therefore controls** is the *noise fraction* of the row
  step, and only below ≈ 0.1. Under AdamW the magnitude `s·‖g_in‖` is
  normalised away per row; what survives is how coherent the step is. This
  matches the 09-20 in-box weight smoke (`step2_0920_box_weight_smoke`):
  drift rose monotonically from share 2 % to 48 % — a more coherent step
  moves the row further — while sentences gained nothing and singles paid.
- **Glyph count**: ‖g_out‖ grows mildly with the box (0.010 → 0.016 over
  1 → 5–7 glyphs) and ‖g_in‖ falls (0.645 → 0.382), so the ratio does rise
  with glyph count, four-fold from 1 to 5–7. A curve that rises with the
  glyph count is the right *sign*; at the 0.25–0.5 level it changes φ by
  two or three points, which is not a cell.

## What this settles and what it does not

- The linear-vs-log question for the band stages is closed as inert at
  these values: keep the log 0.25 → 0.5 (a 2-glyph piece 0.33) — it does
  not pay a piece row three times a single's *magnitude* while leaving
  φ at 0.95. No micro arm on this axis.
- The one place the share is a lever is the low end: `stage0309`'s 0.05
  floor leaves singles at φ 0.78 and lines at 0.57. Whether that noise
  fraction is what protected the singles in the 09-20 smoke (a less
  coherent step drifts less) or merely wasted steps is the open cell —
  0.05 vs 0.15 at that stage, if `stage0309`'s first run shows singles
  paying.
- Not read: grid / flat items (the trainer gives them no box share), and
  glyphs beyond 7.
