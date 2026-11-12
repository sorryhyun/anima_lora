# Arm 0: control `1112_0833` and null twin `1112_0834` (2026-11-12)

The recipe in `../plan.md`, 4-step, 250 iterations, constant LR 5e-5, EMA window 0.2, save every
125. The twin differs in `seed` only (43 against 42). Configs: `../configs/1112_083{3,4}.toml`.
Each run took about 27 min at 5.7–5.8 s/iteration after a 200-step fake warmup. Logs:
`output/logs/turbo/20261112-083405/` (control) and `20261112-090127/` (twin).

This report holds the measurements. The render read (judge_hard, 768 × 1344, beside the teacher)
is pending. Renders go to `output/tests/turbo4/{teacher_m14,1112_0833,1112_0834}/`.

## ΔW: both runs random-walk from step 125

`bench/turbo/dw_overlap.py` and the cosine between consecutive moves (dense ΔW, CPU).

| | control | twin |
|---|---|---|
| \|ΔW\| raw 125 / 250 | 7.34 / 10.33 | 7.49 / 10.54 |
| √2 · \|ΔW\|₁₂₅ | 10.37 | 10.59 |
| \|move\| 0–125 / 125–250 | 7.34 / 7.14 | 7.49 / 7.21 |
| cos(move 0–125, move 125–250) | 0.018 | 0.028 |
| \|ΔW\| EMA 250 | 8.73 | 8.98 |
| cos(EMA 250, raw 250) | 0.961 | 0.965 |

- |ΔW| grows as √t and the two 125-step moves are orthogonal and equally long. These are the 2-step
  numbers again.
- EMA 0.2 (about 50 updates) keeps 96 % of the raw direction. It shortens ΔW by 15 % and does not
  remove the walk inside one 125-step move.

## Control against twin: they share 9–10 % of ΔW

| Pair | cos |
|---|---|
| raw 125 | 0.087 |
| raw 250 | 0.093 |
| EMA 250 | 0.100 |

- A seed change alone leaves 0.09–0.10 in common. The 2026-11-12 pairs that differed in objective
  or LR read 0.09–0.11, and the one pair at 0.14 (`m14_dpdmd_svd_500` against `_tafake`) shared
  seed and data order. The 0.14 likely includes the shared data order.
- So ΔW overlap cannot separate arms. An arm with a different objective would read the same as a
  reseed. Arms are judged on renders and `moment_gap`, as the plan says.

## Scalars: the twin sits on the control

Means over steps 0–125 / 125–220, control then twin: `dm_cos` 0.953 / 0.966 and 0.960 / 0.961,
`dm_rel_gap` 0.269 / 0.245 and 0.263 / 0.268, `x_pred_std` 0.60 / 0.64 and 0.60 / 0.66,
`gan_disc_margin` 0.11 / 0.14 and 0.10 / 0.17. No GAN runaway. The validation probe at 125:
`div_ac_sim` 0.68 / 0.59, `div_xpred_ac_sim` 0.95 / 0.95, so there is no collapse signature. The
twin's TB means of `grad_norm_*` and `fake_loss` are higher late. A few spikes (max 242) drive
this; the medians below match.

## Gradient norm against σ_a: adaptive clip alone

`grad_steps.jsonl`, pre-clip `gn_student` median per σ_a quarter (count in brackets):

| | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | top / bottom | median | p95 | max |
|---|---|---|---|---|---|---|---|---|
| control | 2.67 (64) | 2.50 (66) | 2.34 (57) | 4.41 (63) | 1.65 | 2.86 | 18.6 | 244 |
| twin | 2.80 (66) | 2.54 (68) | 3.05 (56) | 3.57 (60) | 1.27 | 2.91 | 12.9 | 242 |
| pooled | 2.76 (130) | 2.52 (134) | 2.62 (113) | 3.94 (123) | **1.43** | 2.88 | 15.0 | 244 |

- No iteration is below 1.0 in either run, so the clip at 1.0 acts on every step, as on 2 steps.
- The norm is flat over σ_a < 0.75 and rises only in the top quarter. The 4-step rise is milder
  than the 2-step one (4.9 / 1.2 there), consistent with the missing rollout path.
- The rule fixed in the plan (pooled top / bottom ≤ 1.5) gives **adaptive clip alone** for arm 3.
  The per-run ratios (1.65, 1.27) straddle the line, so the decision is close. The spikes (p95
  13–19, max ~240) are the steps an adaptive clip caps either way.

## For the GAN-off probe `1112_0917`

Its criterion (GAN drives the walk if consecutive-move cos ≥ 0.1 and |ΔW|₂₅₀ clearly above
√2 · |ΔW|₁₂₅) now has a two-run reference: move cos 0.018 / 0.028, |ΔW|₂₅₀ / (√2 · |ΔW|₁₂₅) 0.996 /
0.995.
