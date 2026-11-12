# Arm 0: control `1112_0833` and null twin `1112_0834` (2026-11-12)

The baseline in `../1112_v1.md`, 4-step, 250 iterations, constant LR 5e-5, EMA window 0.2, save every
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

## Divergence bursts: the control's raw 250 is caught inside one

The control's raw 250 renders a flat noise texture on all 12 judge images. The file is sound
(same keys, finite, equal to `output/ckpt/1112_0833.safetensors`). The run diverged in its last
steps:

- `gn_student` 5.5 / 14 / 6 / 16 at steps 242–245, then **231 / 244 / 106 / 72** at 246–249.
  `gn_fake` rises with it (25 → 111 / 69 / 145 / 85), so student and fake blow up together.
- σ_a over those steps is 0.03–0.96, so the burst is not a σ_a tail.
- `x_pred_std` 0.74 → 1.98 at 250; validation at 250: `fm_mse` 0.29 → 0.88 and `div_xpred_ac_sim`
  0.948 → 0.999. That is the collapse signature in `mfd.md`.
- The twin had a burst of the same kind at steps 153–162 (37 / 29 / 58 / **242** / 131) and
  recovered. Its validation at 250 is normal (`fm_mse` 0.27).
- EMA 0.2 dilutes the burst: the control's EMA 250 renders. The 125 + 250 average carries part of
  it (blocky crowd on the guitarist prompt).

So bursts happen about once per 250 iterations at this recipe, and a raw checkpoint can sit
inside one. The EMA shipping rule holds. The adaptive clip in arm 3 is the knob aimed at bursts,
and arm 3 should cap the fake as well as the student.

## Renders, control only (rough read, 2026-11-12)

Grids: `output/tests/turbo4/cmp/0833_{ba5,cafe,bocchi}.jpg`. Rows are teacher (28 steps, cfg 4),
EMA 250, avg2, raw 250, EMA 125. Columns are seeds 1–4.

- **EMA 125 reads best, EMA 250 is more uniform.** At 250 the café gives the same pink aprons and
  layout on all four seeds, and the five-girl prompt gives one fixed grid of figures on grey
  paving with the sky gone. EMA 125 still varies the apron colours, keeps the sky, halos and
  confetti, and draws crowd silhouettes.
- **No student reaches the teacher's composition.** The teacher draws the jump from below and the
  tumbling slapstick group. The students draw frontal standing poses on every seed. The teacher
  spells `CAFE MILLENNIUM`; the students garble it.
## Renders, control against twin: the noise floor

Grids: `output/tests/turbo4/cmp/arm0_{ba5,cafe,bocchi}.jpg`. Rows are teacher, control EMA 125,
twin EMA 125, control EMA 250, twin EMA 250, twin raw 250.

- **The control's "250 more uniform than 125" is its seed.** The twin's EMA 250 sits at its EMA 125
  level: the aprons still vary (blue, white, lilac), the sky and halos stay on the five-girl
  prompt, and the stage keeps its lights. The twin's raw 250 is over-saturated with soft crowd heads
  and is not collapsed.
- **The render noise floor is large.** Between the two seeds, palette, apron colour, background
  density and sign legibility all change about as much as they change between 125 and 250 of one
  run. The twin's EMA 125 spells `CAFE MILLENNIUM` on seed 1, while the control's EMA 125 does not
  spell it on any seed. An arm has to beat differences of this size to pass.
- **Composition is shared by both seeds, so it comes from the objective, not the seed.** Both runs draw
  frontal standing figures on every seed. Neither draws the teacher's jump from below or the
  tumbling group. An arm that moves composition shows above this floor; one that moves palette or
  sharpness does not.

## GAN-off probe `1112_0917`: GAN does not drive the walk

The control with `gan.weight_gen = 0`, seed 42 (config `../configs/1112_0917.toml`). The criterion
was fixed before the run: GAN drives the walk if consecutive-move cos ≥ 0.1 and |ΔW|₂₅₀ clearly
exceeds √2 · |ΔW|₁₂₅.

| | control | twin | GAN off |
|---|---|---|---|
| \|ΔW\| raw 125 / 250 | 7.34 / 10.33 | 7.49 / 10.54 | 7.38 / 10.27 |
| \|ΔW\|₂₅₀ / (√2 · \|ΔW\|₁₂₅) | 0.996 | 0.995 | 0.985 |
| cos(move 0–125, move 125–250) | 0.018 | 0.028 | 0.016 |

- **Verdict: GAN is not the cause.** The walk without GAN is the same walk. GAN stays in the recipe.
- **The GAN-off run still spikes, but less.** `gn_student` above 20 at steps 14–16 (43 / **133** /
  38), 37, 39 and 169 (49), with no divergence. The control's burst (244, with `gn_disc` up to 7)
  was the largest of the three runs. That is one run per arm, so whether GAN enlarges bursts is
  open.
- cos(GAN-off EMA 250, control EMA 250) = 0.136, above the 0.10 of the reseed pair. The two runs
  share seed and data order, which fits the 0.14 of the earlier same-seed pair.
