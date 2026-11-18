# Plan: the 4-step line after `v4_preview`

Rewritten 2026-11-17. This file holds what is open; finished runs and probes are in `reports/` and
the run table (`reports/runs.md`). The closed 2-step plan is
`91d476e0:scripts/distill_turbo/plan.md`.

## Where the line stands

Facts:

- `v4_preview` (`1112_1512`, 1k EMA on the `1112_v1.md` recipe) is shipped. The recipe in
  `configs/methods/turbo.toml` has since taken the static grid and the cond-only L_CDM
  (`reports/grid_cdm.md`).
- Every student of the DM recipe on merged14 / merged15 draws one layout per prompt
  (`reports/warm_v3.md`, `reports/m15_1k.md`). The line since then starts from the official turbo
  v1.1 delta, which has the best steps 1–3 on merged15 with no training, and fits step 0 alone on
  the anchor (`reports/v11_init.md`, `anchor_only_recipe.md`).
- With the anchor target itself as step 0 the student's steps 1–3 draw the teacher's image (cos
  0.966). Removing a quarter of step 0's residual to the anchor target clears most of the
  guitarist's breaks; half of it reads like the teacher's own step 0 (`reports/step0_depth.md`).
- The full-dataset cache (`turbo_1117_0823`, 6484 pairs, one pass) takes the held-out anchor MSE
  0.078 → 0.070 and `S` cos 0.760 → 0.801, with steps 1–3 and the seed variety held. It is the best
  anchor student and the start of the arms below.
- Rank 96 at the √r alpha (`turbo_1117_0918`) ties `0823` on the probe and in the renders.
- DM on steps 1–3 from the teacher-forced handoff (`2221`) is not a win (`reports/teacher_handoff.md`,
  `reports/step0_depth.md`).
- Paired feature matching at block 27 (`turbo_1117_1442`) and the paired GAN on top of it
  (`turbo_1118_1028`) are in `reports/paired_fm_gan.md`. The line uses the paired GAN (the user's
  call, 2026-11-18); its open problem is the disc margin (item 2).

Interpretation:

- The breaks left in the renders are step 0's residual to the anchor target
  (`reports/step0_depth.md`).
- More pairs is a shallow slope.
- Rank is not the limit.
- DM on steps 1–3 from the teacher-forced handoff trains them where they are already clean.

## Target: keep v1.1's steps 1–3, fit step 0 to a third of its residual

One LoRA. Every run is read on the handoff probe beside the renders, against its starting point:

| | `0823` | target |
|---|---|---|
| held-out anchor MSE (14 of 28, judge prompts) | 0.070 | 0.02–0.04 (the `M0.5`–`M0.25` levels of the mix probe) |
| `S` cos | 0.801 | 0.88–0.94 |
| `T>S@1` cos | 0.944 | held at 0.94 |
| sameness `S` | 0.300 | near the teacher's 0.314 |

Renders: guitarist seed 2 and the group's lower halves, where `0823` still breaks.

The anchor stays on 14 of 28. Its caches are built (`m15_k14of28_{p64,p512,pall}`), and its chord
ends at σ 0.75, the first boundary of the 2-step grid. On the 2-step grid the residual enters the
state at 0.25 instead of 0.1.

A rank change takes alpha by the √r law (alpha / √rank at 22.5: 180 @ r64, 220 @ r96, adaln 90 @
r16; `docs/anima/methods/adaln.md`, `docs/anima/findings/lora_recipe.md`).

Interpretation: a step 0 that meets this target is also the 2-step student's step 0 (`idea.md` 3),
and the depth matters more for 2 steps than for 4.

## What is open, in order

### 1. Student LR 2e-5 on the full cache

Facts:

- `turbo_1117_1054`: rank 64, `0823` with only the LR changed, 42 min, read against `0823`.
- The short A/B at rank 64 (`turbo_1117_1020` / `1021`, `reports/step0_depth.md`) did not pass on
  the probe: held-out MSE level, `S` cos and sameness down at 2e-5. The user's read of its renders
  (2026-11-17) sees promise in 2e-5, so the LR is not closed until the full-length run is compared
  with `0823`.
- Ran 2026-11-17: not passed on the probe (held-out MSE 0.0695 → 0.0674, `S` cos 0.801 →
  0.792, `T>S@1` 0.944 → 0.937, sameness 0.300 → 0.284), renders close to `0823`; the LR closes or
  stays open on the user's read of `output/tests/turbo4/cmp/lr2e5_full_p{0,1,2}.jpg`.
- The rank 96 form planned before follows for the record. Rank 96 at student LR 2e-5: `0918` with
  `optim.student_lr` 5e-5 → 2e-5, nothing else changed; the cosine then ends at 2e-6. One pass,
  about 43 min, read against `0918`. Rank 96 on the user's call.
- Held-out anchor MSE by LR (`_archive/anima_turbo/reports/step0.md`, merged14 S-b): a constant
  2e-4 reads 0.277, a constant 5e-5 0.14–0.16, an annealed tail 0.11–0.13. No anchor run has had a
  peak under 5e-5, and the cosine has never gone under 5e-6.
- `0918` trails `0823` on the train loss by 0.003–0.004 at the high end of the cosine and by 0.001
  at its end, on the same pairs in the same order (`reports/step0_depth.md`). The arm was set for a
  `0918` that ends above `0823`; it ended level on the held-out number. Rank 64 ties it.
- `0823`'s fall over its first half: 0.088 → 0.070.

Interpretation:

- The regression's fit is set by the LR at the end of training, not by length, draws or an
  average of checkpoints.
- The case for the rank 96 arm rests on the gap closing with the LR and on the archive's result.
  Since rank 64 ties rank 96, the same arm on `0823` would read as well.
- The risk is travel: a lower LR moves the student less far from the v1.1 start, and `0823`'s fall
  over its first half is not split into travel and level.

### 2. Paired GAN on the σ 0.75 state: fix the disc margin

Facts:

- Feature matching (`1105`, `1123`, `1403`, `1442`) and the paired GAN (`0953`, `1028`) are in
  [reports/paired_fm_gan.md](reports/paired_fm_gan.md).
- The line uses the paired GAN, on the user's read of `1028` against `1442` (2026-11-18): the GAN
  has a clear effect, and with the margin issue fixed it should be usable.
- Starting point: `turbo_1118_1028` (`1442` + paired GAN at block 27, `gan.paired_ratio` 0.5, token
  head, `disc_lr` 5e-5, 200-step delay, 200-step ramp).
- `1028`'s disc margin: 0.44 → 1.41 → 2.15 → 2.83 → 2.98 → 3.06 over steps 0–200 … 1600–2048;
  disc loss 1.23 → 0.79; GAN share 0.43 from step 400 on; anchor MSE level with `1442` (0.0705).
- `0953` (no normalisation, GAN share 0.99), steps 1600–2048: disc margin 0.11, disc loss 1.36,
  training anchor MSE 0.131, FM loss 0.0104. The student closed the margin there with a worse fit.
- `1028`'s disc is on the student's cosine schedule (`disc_lr` 2.9e-5 → 6.8e-6 over steps
  800–2048), and the student gradient is clipped every iteration (pre-clip norm about 3 against
  `grad_clip` 1.0).
- `turbo_1118_1122` (`1028` with `gan.paired_ratio` 1.0): not passed. The user's blind picks on
  the student sheets: `1028` 6, `1122` 2, no pick on the 4 café rows. The user's read: the GAN does
  not try to break things, but the perspective of objects is ambiguous and it is comparatively
  monotonous. Disc margin 2.57 against `1028`'s 3.05; training anchor MSE 0.0732 against 0.0703,
  held-out 0.0748 against 0.0739, spread unknown. 0.5 stays the parent.
- Claude's read of the student sheets: `1028` keeps `1442`'s layout on every seed; the guitarist's
  guitar is whole on seeds 1–4 (`1442` breaks it on seed 2); the group's seed 3 draws one
  grey-haired girl (`1442`: two) and seed 4 draws the lower faces apart (`1442`: merged).

Interpretation:

- The GAN term changes local structure (whole objects, one figure per character) and leaves layout
  and the anchor fit where `1442` has them.
- The margin keeps rising with no step at epoch 2, so the disc is winning on student / teacher
  separability rather than memorising prompts; near saturation its gradient says less about how
  far a state is from the teacher's. Fixing the margin means keeping the disc in its informative
  range.
- Candidate arms, each one change from `1028`, read against `1028` on the same probe sheets
  (`probe_handoff.py`, `judge_hard.txt` seeds 1–4) with the margin trajectory beside them:
  - Renoise augmentation: the disc sees `x_t` / `x_s` renoised to σ' ∈ [0.75, σ_max] with shared
    noise; real and fake overlap more and the margin falls. Costs a second teacher forward per side
    unless the generator term moves to the renoised state as well.
  - A slower or regularised disc: `disc_lr` 5e-5 → 1e-5 (the smoke's rate, margin 0.05 at step 80)
    or `gan.r1_weight` > 0 (one extra no-grad teacher forward).
  - A higher `paired_ratio` (0.5 → 1.0): run as `1122`, not passed.
- `1122` lowered the margin and read worse, so a lower margin is not by itself the aim of the next
  arm; the arms above are read on the renders with the margin as a number beside them.

Disc head (2026-11-18), facts:

- The user's call: disc variants are read at `gan.paired_ratio` 1.0, where the GAN's effect is
  plain to see, against `1122`; whether to go back to 0.5 is decided after.
- The user's proposal: pool the tokens 2×2 before the head. The heads in the code are `pooled`
  (one logit per tap) and `token`; neither folds the tokens back to the grid.
- `probe_gap_scale` (step-1024 students, 16 pairs, `reports/paired_fm_gan.md`): a 2×2 mean pool
  keeps 60% of the student / teacher gap at block 27 and 4×4 keeps 38–40% (equal-norm random
  residual: 29%, 9%; the teacher's own features: 39%, 20%); student over random residual goes
  32 → 68 → 142.

Disc head, interpretation:

- A 2×2 pooled head has a gap to read and drops the token-scale 40% of it.

Disc head, runs:

- `turbo_1118_1215` (`1122` with `gan.disc_pool` 2): finished. Against `1122`, steps 1600–2048:
  training anchor MSE 0.0729 / 0.0732, FM loss 0.0081 / 0.0081, disc margin 2.77 / 2.57; probe:
  held-out anchor MSE 0.0758 / 0.0748, `S` cos 0.771 / 0.779, sameness `S` 0.311 / 0.279; spread
  unknown. Blind sheets `output/blind/turbo/1215_vs_1122` (local); the user's picks `1215` 2, `1122` 2 on the
  4 group rows, no pick on the other 8; the user's read: it does not look meaningful. Level.
- FastGen's heads (commit `898e757`, `reports/paired_fm_gan.md`): none is per token; the image DiT
  head is a 4×4 stride-2 conv then a global average, and the Cosmos Predict2.5-2B recipe (the
  backbone family of Anima) uses a multi-scale head (scales 1–16) on taps 13, 20, 27. The port
  took a pooled-token MLP, then the per-token head, on one tap.
- The user's call after `1215` (2026-11-18): the GAN should be a proper implementation before
  arms are compared, FastGen's 4×4 stride-2 conv head on taps 13, 20, 27 (not DMD2's).
- `turbo_1118_1302` (`1122` with `gan.disc_head = "conv"`, `gan.feature_blocks` 13 / 20 / 27,
  `disc_hidden` 256; head and taps together): finished. Against `1122`, steps 1600–2048: training
  anchor MSE 0.0697 / 0.0732, FM loss 0.0081 / 0.0081, disc margin 11.15 / 2.57, disc loss
  0.15 / 0.86; probe: held-out anchor MSE 0.0764 / 0.0748, `S` cos 0.777 / 0.779, sameness `S`
  0.296 / 0.279; spread unknown. FastGen's width (hidden 1024, 100.7M parameters) does not fit
  the card. Blind sheets `output/blind/turbo/1302_vs_1122` (local); the user's picks pending.
- Interpretation: the conv disc saturates under `1122`'s disc settings (`disc_lr` 5e-5, no R1, no
  renoise). If the renders show little of the GAN, the next arm is on the disc's strength with
  FastGen's own values as the source (`disc_lr` 1e-5, renoised inputs), not on the head.

### 3. (merged into 2)

The paired GAN arm is item 2 since 2026-11-18.

### 4. Steps 1–3 on the student's own handoff, if step 0 cannot reach the depth

Facts:

- `2221` with `dpdmd.teacher_handoff` off, from the best anchor student: the DM terms see the
  inputs the student's step 0 hands them. 250 iterations, about 20 min. Never run in this form.

Interpretation:

- The risk is steps 1–3 that redraw the layout; it reads as a fall in `T>S@1` cos.

### 5. Two passes over the full cache

Facts:

- 12968 iterations, about 87 min. Length was flat from 250 to 1000 iterations, and a second pass
  re-reads pairs (`reports/step0_depth.md`, `2019`).

Interpretation:

- Low expectation.

Parked:

- The 7 of 28 target. Ceiling 0.944 against 0.967 (`reports/step0_depth.md`). The user's call:
  keep 14 of 28.
  - Interpretation: nearer in MSE and worth less per unit of it.
- DM on step 0 (`idea.md` 1 and 2). It brings the fake network back (4.5 s an iteration). Behind
  the paired forms above.
  - Interpretation: the step-0 problem is the precision of a paired map, and DM carries no
    pairing: it removes a residual by moving to the nearest mode, which is the one layout per
    prompt of S-c.1, `1355` and `1552`.

## A/B protocol

1. Anchor arms run one pass of the full cache (`m15_k14of28_pall`, 6484 iterations,
   `grad_accum 2`, about 43 min) and are read on the last EMA. DM / GAN arms run 250 iterations;
   one that wins at 250 needs a 1k check before it ships (`reports/m15_1k.md`).
2. **Renders are the gate, and the user's read decides.** The probe's `S` column pairs with every
   other probe of the day (CPU-drawn noise, `judge_hard.txt` seeds 1–4, 768 × 1344, merged15 int8);
   the judge renders from `inference.py` draw on CUDA and do not pair with it. Sheets go to
   `output/tests/turbo4/cmp/`.
3. Probe beside the renders (`bench/turbo/probe_handoff.py --dit
   models/diffusion_models/anima_merged15.safetensors --ts_handoffs 1 --st_handoffs , --anchor_k 14
   --anchor_steps 28`, about 3 min per student): held-out anchor MSE, `S` and `T>S@1` cos, and the
   sameness across seeds (m15 teacher 0.314). `--step0_mix` adds the depth arms.
4. An arm is read against its starting point; the read and the source of any bar go in the config
   header before the run (`experiment-planning` skill).
5. One seed per arm so far.

## Open

- The recipe on the base teacher. Every student of the DM recipe collapsed on a merged teacher
  and Hub v3 was trained on base; the split is in `reports/m15_1k.md`. It matters again if arm 4
  brings the DM terms back.
- Which term draws the text boxes of `2221` (DM, CDM or the GAN). Not separated.
  - Interpretation: it clouds the finish read of any arm that turns those terms on.
- Fake-lag probe, before anything that touches the fake (student : fake 1 : 10, the VA-TDM
  teacher-regularised fake): freeze the student at a resume point, run about 40 more fake steps,
  and read how far the DM signal turns on fixed draws. Not built.
- Café colour drift under cond-only L_CDM. A seed-43 twin of `0958` and a weight-matched arm
  (`cdm.weight` ≈ 1.7) would tell seed luck, magnitude and direction apart. Not run.
- 2 steps. Reopening it waits on a step 0 that meets the target above.

## Dropped without a run

- A relativistic paired GAN loss (`D(f_t) − D(f_s)` per token): the token logits of the teacher
  and the student state at the same position correlate at 0.12 (`1028`) and 0.18 (`1122`), so
  there is little shared offset for it to cancel (`probe_gap_scale`, `reports/paired_fm_gan.md`).
- Anchor-only and DM-only iterations drawn at random: the clip it was meant to unshare does not
  bind on `2221` (`reports/step0_depth.md`).
- Adaptive clip: on v1's recipe the clip at 1.0 already acts on 98 % of the steps under AdamW.
  - Interpretation: so every update has unit norm; a threshold at twice the median would give a
    spike twice the weight it has now.
- A longer run from the `weight_svd` init: v3 shows where it ends, at 16 h; a `weight_svd`
  student also stays inside its init subspace, and the working turbo ΔW sits outside it
  (`reports/warm_v3.md`).
