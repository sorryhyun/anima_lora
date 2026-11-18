# Plan: the 4-step line after `v4_preview`

Rewritten 2026-11-17, re-ordered 2026-11-18. This file holds what is open; finished runs and probes
are in `reports/` and the run table (`reports/runs.md`). The 2-step plan closed on 2026-11-12 is
`91d476e0:scripts/distill_turbo/plan.md`.

Order (the user's call, 2026-11-18): step 0 stops here as the first line of work and becomes the
second priority. First come the 2-step student (superturbo, item A) and steps 1–3 (item B); the
user's reasoning is that with step 0 fitted, the rest of the image may come from one step.

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
  call, 2026-11-18). With FastGen's conv head on taps 13 / 20 / 27 and R1 100 (`turbo_1118_1353`)
  the disc margin holds near 5.5 and the user picks it over the token head 6 / 2 (item 2).

Interpretation:

- The breaks left in the renders are step 0's residual to the anchor target
  (`reports/step0_depth.md`).
- More pairs is a shallow slope.
- Rank is not the limit.
- DM on steps 1–3 from the teacher-forced handoff trains them where they are already clean.

## A. Superturbo: two steps on a fitted step 0

Facts:

- The 2-step static grid at shift 3 is σ 1 → 0.75 → 0. The 14 of 28 anchor is the teacher's mean
  velocity over σ [0.75, 1], so on the 2-step grid a step 0 at zero anchor loss lands on the
  teacher's own σ 0.75 state. On 4 steps the same velocity is applied over σ 1 → 0.9 only.
- With the anchor target as step 0, v1.1's steps 1–3 draw the teacher's image at cos 0.967
  (`reports/v11_init.md`); the step-0 students sit at `S` cos 0.80 (`0823`) with held-out anchor
  MSE 0.070–0.075.
- On the 4-step students of 2026-11-12 a handoff at σ 0.75 kept 0.86–0.93 of the teacher's
  layout, against 0.70–0.80 at σ 0.9 (`reports/handoff.md`).
- The 2-step line closed on 2026-11-12 at `1111_1115` step 250: merged14, cold `weight_svd` init,
  the 6 of 12 anchor, DM through the rollout (`dpdmd.rollout_grad`,
  `_archive/proposals/superturbo.md`). None of today's parts were in it: the v1.1 init, the 14 of
  28 anchor, the cache, the paired GAN.
- No handoff probe has been run at 2 steps. `probe_handoff.py --student_steps 2` exists.
- Nothing has trained the jump σ 0.75 → 0. The 4-step students take σ 0.75 → 0.5 there.
- `dpdmd.teacher_handoff` with `dmd.student_steps 2` starts the DM rollout on the teacher's exact
  σ 0.75 state (`x_1 = ε − 0.25 · v_target`); the config accepts the pair. On 4 steps (`2221`) it
  started on the chord's point at σ 0.9.

A1, the probe (forward only, about 3 min a student): `bench/turbo/probe_handoff.py --student_steps 2
--ts_handoffs 1 --st_handoffs , --anchor_k 14 --anchor_steps 28` on the v1.1 delta untrained,
`0823` and `1353` EMA 2048 (`1425` when it ends). Arms `S` (2 steps), `T>S@1` (teacher to σ 0.75,
the student's one jump), `A>S@1` (the anchor target, then the jump). The read and its bars go in
the job's header before it runs (`experiment-planning` skill); the 4-step `T>S@1` of each student
(0.94) is the reference.

A1 as queued (2026-11-18, the user's go; label `handoff_2step_a1`, behind `1425`'s probe):

- Question: at 2 steps, is the loss in step 0 or in the one jump σ 0.75 → 0? `A>S@1` near the
  4-step `A>S@1` (0.966 on `0823`) → the jump holds and the 2-step student is a step-0 problem,
  which item 2 is on. `A>S@1` well under it → the jump is what a 2-step arm trains
  (`teacher_handoff` on the 2-step grid).
- Students: the v1.1 delta untrained, `0823` EMA 6484, `1353` EMA 2048. None trained the jump.
- Records it rests on: 4-step `T>S@1` cos 0.932 (v1.1, `reports/v11_init.md`), 0.944 (`0823`),
  0.942 (`1353`); 4-step `A>S@1` 0.966 (`0823`, `reports/step0_depth.md`).
- Read: the probe grids (group / café / guitarist, seeds 1–4, arms `T`, `T>S@1`, `A>S@1`, `S`),
  labelled, beside the cos and sameness columns. The user's read decides.
- Bars: spread unknown (no 2-step twin); the numbers go beside the 4-step ones of the same
  students.
- Cost: forward only, about 3 min a student, about 10 min.

Interpretation:

- A high `A>S@1` with a low `S` puts the 2-step loss in step 0, which item 2 is fitting already.
  A low `A>S@1` puts it in the one jump, and the jump is what a 2-step arm trains.
- The training form that follows is the jump on the teacher-forced handoff: `teacher_handoff` on
  the 2-step grid from the best step-0 student, a 2-step LoRA of its own. Its terms (DM, CDM, the
  GAN) and length are set after A1.

## B. Steps 1–3

Facts:

- With the anchor target or the teacher's step 0 in front of them, the step-0 students' steps 1–3
  finish at 0.94–0.97 (`reports/step0_depth.md`).
- `2221` (DM, CDM and a GAN on steps 1–3 from the teacher-forced handoff, 250 iterations): step 0
  and the layouts held; the finish moved both ways (café aprons and the group's lying girl toward
  the teacher, subtitle boxes of garbled text and a guitar without its neck added); the probe sits
  slightly under `2209` on every number (`reports/teacher_handoff.md`).
- Hub v3 untrained keeps the cleanest finish and the only legible café sign of the 4-step rows;
  v1.1 untrained is hazy and low in contrast (`reports/v11_init.md`).
- `2221` with `dpdmd.teacher_handoff` off, from the best step-0 student, puts the DM terms on the
  inputs the student's own step 0 hands them. 250 iterations, about 20 min. Never run.

Interpretation:

- Steps 1–3 have two things to gain: holding up under step 0's residual (the inputs they are
  handed in use), and the finish (contrast, text) where v1.1's steps are soft.
- On the teacher-forced handoff the DM terms train them where they are already clean; on the
  student's own handoff they see the residual.
- The risk of the own-handoff form is steps 1–3 that redraw the layout; it reads as a fall in
  `T>S@1` cos.
- Which term draws `2221`'s text boxes (DM, CDM or the GAN) is not separated, and it clouds the
  finish read of any arm that turns those terms on.

## Step 0 (second priority): keep v1.1's steps 1–3, fit step 0 to a third of its residual

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

Interpretation: a step 0 that meets this target is also the 2-step student's step 0 (item A),
and the depth matters more for 2 steps than for 4.

## Step 0 arms (second priority)

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
  the card. Blind sheets `output/blind/turbo/1302_vs_1122` (local); the user knows `1122`'s renders on
  sight (the read is labelled in effect). Read skipped (the user, 2026-11-18): with the disc
  saturated it is not a proper run of the conv head, so it carries no verdict.
- The user's call (2026-11-18): keep the disc from saturating; renoising to a higher τ is
  rejected (the disc would tell noised states apart, not what it should).
- `turbo_1118_1353` (`1302` with `gan.r1_weight` 100): queued 13:53, probe job behind it. R1 term
  on `1302`'s step-1024 disc 0.013 at α 0.1; smokes' margin at step 60: 1.01 / 0.92 / 0.70 at
  weight 0 / 100 / 1000.
  At step 768–896 against `1302`: disc margin 4.80 / 8.64, disc loss 0.51 / 0.31. Finished: margin
  flat at 5.4–5.9 from step 1280 (1920–2048: 5.88 / 11.56), disc loss 0.40 / 0.10; probe against
  `1302` / `1122`: held-out anchor MSE 0.0751 / 0.0764 / 0.0748, `S` cos 0.770 / 0.777 / 0.779,
  `T>S@1` 0.942 / 0.943 / 0.942, sameness `S` 0.277 / 0.296 / 0.279 (teacher 0.314). Sheets
  `output/blind/turbo/1353_vs_1122`, `1353_vs_1302`. The user's read against `1122` (2026-11-18,
  labelled in effect): `1353` 6, `1122` 2, no pick 4 (group 4 / 0, café 2 / 2, the guitarist rows not
  picked). Passed against `1122`: R1 stays in the recipe; `1425` keeps running.
- `turbo_1118_1425` (`1353` with `student_lr` and `gan.disc_lr` 5e-5 → 2e-5, full cache `pall`, 6484
  iterations, about 2 h): queued 14:21 behind `1353`'s probe on the user's call, before `1353`'s
  read; stopped if that read comes out wrong. Read against `1054` (same LR and cache, no GAN). Probe
  job `handoff_pgan_lr2e5_pall_1425` behind it.
- Interpretation: the conv disc saturates under `1122`'s disc settings (`disc_lr` 5e-5, no R1).
  If `1353`'s margin still runs away, the next values are R1 1000 or `disc_lr` 1e-5 (FastGen's).

- Open on the GAN: `gan.paired_ratio` 0.5 with the conv head and R1 (the user's order: 1.0 first,
  0.5 decided after).

### 3. Two passes over the full cache

Facts:

- 12968 iterations, about 87 min. Length was flat from 250 to 1000 iterations, and a second pass
  re-reads pairs (`reports/step0_depth.md`, `2019`).

Interpretation:

- Low expectation.

Parked:

- The 7 of 28 target. Ceiling 0.944 against 0.967 (`reports/step0_depth.md`). The user's call:
  keep 14 of 28.
  - Interpretation: nearer in MSE and worth less per unit of it.
- DM on step 0 (`dm_first_step`, with a higher `div_weight` or σ_a drawn from the upper range only).
  It brings the fake network back (4.5 s an iteration). Behind the paired forms above.
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
  and Hub v3 was trained on base; the split is in `reports/m15_1k.md`. It matters again if item A or B
  brings the DM terms back.
- Fake-lag probe, before anything that touches the fake (student : fake 1 : 10, the VA-TDM
  teacher-regularised fake): freeze the student at a resume point, run about 40 more fake steps,
  and read how far the DM signal turns on fixed draws. Not built.
- Café colour drift under cond-only L_CDM. A seed-43 twin of `0958` and a weight-matched arm
  (`cdm.weight` ≈ 1.7) would tell seed luck, magnitude and direction apart. Not run.
- DAVE on the student's step 0 at inference (`--dave auto`, strength 0.3 / 0.8 on an existing
  EMA): a render-only check, parked on the user's call. The mask was calibrated on base Anima at
  CFG 4, and on 4 steps its default window covers step 0.
- A lower-resolution anchor (a smaller ε for the regression only) would save 8–15 % of an
  iteration; with the anchor cache the anchor costs 0.4 s an iteration and throughput is not the
  limit.

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
