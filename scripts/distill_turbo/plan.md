# Plan: average the DM signal over several (σ_a, ε_dm) draws

Written 2026-11-11. `dmd.dm_draws` is built (`steps.dm_slot`, test in `tests/test_turbo_mfd.py`);
the run is `1111_1740`. It follows from two measurements on the superturbo
runs (2-step student, `dpdmd.rollout_grad`, proposal `docs/anima/proposal/superturbo.md`), whose
renders were best at step 250 and worse at 375 and 500, at LR 5e-5 (`1111_1005`) and again at
2e-5 (`1111_1115`).

## What was measured

### The student weights random-walk

ΔW = up·down of the saved student checkpoints, Frobenius norm over all 364 modules.

| Run | LR, schedule | \|ΔW\| at 125 / 250 / 375 / 500 | cos between consecutive 125-step moves |
|---|---|---|---|
| `1111_1115` (GAN) | 2e-5, constant | 2.77 / 3.98 / 4.85 / 5.50 | +0.018, −0.001 |
| `1111_1005` (GAN) | 5e-5, constant | 6.54 / 9.74 / 12.37 / 14.77 | +0.012, +0.012 |
| `..._k3_nogan_..._const_ema10` | 5e-5, constant | 6.66 / 9.27 / 11.23 / 12.77 | – |
| `..._k6_nogan_nocdm_500_tafake` | 5e-5, cosine | 5.71 / 7.74 / 8.21 / 8.26 | – |

- Under a constant LR, |ΔW| follows √t (`1111_1115`: 2.77·√(t/125) = 3.92 / 4.80 / 5.54) and
  each 125-step move is orthogonal to the next. Nothing pulls the weights back and nothing
  pushes them one way.
- Two runs of the same objective from the same init share cos 0.08–0.17 of their ΔW.
- Lowering the LR 2.5× scaled each move by 2.56× and changed nothing else. Adam scales the
  drift and the noise by the same factor, so the step at which noise overtakes learning does
  not move.
- The walk is there without the GAN. Only the cosine run stops (8.21 → 8.26), and it is the
  only run whose `div_loss`, `grad_signal_rms` and `dm_cos` improve in the last 150 steps.
- The EMA of window 0.1 (50 steps) does not remove it (`1111_1115` at 500: 5.12 EMA, 5.50 raw).

### The gradient noise is the DM query draw

`bench/turbo/probe_grad_noise.py` on `output/ckpt/1111_1115/1111_1115_resume.pt` (step 375):
16 samples (prompt, rollout ε), 3 draws of (σ_a, ε_dm) each, no optimizer step. Result:
`bench/turbo/results/20261111-1312-grad_noise/result.json`. Numbers below are in Adam's metric
(gradient divided by √v̂ from the bundle); the raw gradient gives the same picture.

Direction energy of the per-step student gradient (unit vectors, which is what grad clip 1.0
leaves of a step):

| Part | Share |
|---|---|
| Between (σ_a, ε_dm) draws of one sample | 77 % |
| Between samples | 20 % |
| Mean over everything | 3 % |

Squared-norm share by term: DM through the rollout into step 0 56 %, DM on step 1 13 %,
anchor 7 %, GAN 7 % (the rest is the cross term of the two DM paths).

- The rollout path is the largest term and the noisiest (mean share 2 %, against 7 % for the
  step-1 path).
- DM and the anchor agree on average (cos of the mean directions +0.55, step-0 path alone
  +0.54). The GAN's mean is mildly opposite DM (−0.19).
- The DM gradient norm grows with σ_a: median 1.24 below 0.25, 2.03, 2.36, and 4.93 above
  0.75, with 21.9 at σ_a 0.96 and 30.5 at 0.99, nearly all of it through the rollout path.
  Under the clip, such a step counts as much as any other.

Limits: one checkpoint, 3 draws, and all 16 samples came from the largest token bucket
(4154–4200 tokens; the batch sampler serves it first). The cos between mean directions rests on
16 samples.

Reading that joins the two: the mean share of 3 % at frozen weights does not show up as a
persistent direction in ΔW (≤ 2 % of a 125-step move), so the objective holds a few directions
near an equilibrium while the draw noise diffuses through the rest. This is an interpretation,
not a measurement.

## Change

One knob, `dmd.dm_draws` (default 1), also on the CLI.

- `distill.run_loop` calls the DM-slot surrogate (`steps.mfd_surrogate` or `dmd_surrogate`)
  `dm_draws` times on the same `x_pred` and averages `grad_signal` (each draw keeps its own
  `dm_x0_norm` division). The surrogate loss is linear in `grad_signal`, so the student
  backward stays a single one.
- The first draw makes the same RNG calls as now, and its `tau_dm` / `eps_dm` feed the GAN
  generator renoise, so `dm_draws = 1` is the current loop bit for bit (test beside
  `tests/test_turbo_mfd.py`).
- Metrics (`delta_dm`, `v_real_cond_dm`, `v_fake_cond_dm`, `tau_dm_e`) read the first draw;
  `grad_signal_rms` reads the averaged signal.
- `fake_update`, the anchor, `fake_transition` and L_CDM do not change.

Cost: one more draw is K × 3 no-grad forwards (K averages 3.4 at dtau 0.1 / max 4). Estimated
6.5 → about 10 s/step at 4 draws, to be read off a smoke. No VRAM change expected (no graph).

Expected effect, from the probe's shares: 4 draws cut the draw part to a quarter, which takes
the mean share from 3 % to about 7 %. Draws alone cap near 13 %; past that the between-sample
part needs a larger batch.

## Run

`1111_1740`: `1111_1115` with `dm_draws = 4`, nothing else changed (config: its
`output/ckpt/1111_1740.snapshot.toml`); `1111_1115` itself is the control.

Read, with the criteria fixed here:

1. Renders at 2 steps, cfg 1, the ANIMA-sign and nozomi prompts, seeds as in
   `output/tests/superturbo/1111_1115/`, checkpoints 250 / 375 / 500 (raw and EMA). It passes
   if 375 and 500 are no worse than its own 250 and its 250 is no worse than `1111_1115`'s.
2. |ΔW| at 125 / 250 / 375 / 500 and the cos between consecutive moves, as in the table above.
3. The probe on its step-375 bundle: mean share against 3 %.

## Result (2026-11-11): fails criterion 1; the walk is unchanged

Ran 500 steps at 9.5 s/step (control 6.5). Grids side by side with the control:
`output/tests/superturbo/1111_1740/cmp_{anima,nozomi}.jpg` (the control's 375 raw was rendered in
the same sweep; it had none).

1. Renders. 250 is on par with the control's (one exact "ANIMA" in the EMA row, none in the
   control's). 375 is worse than its own 250: the ANIMA sign breaks into scattered boards (raw
   and EMA), nozomi tilts with flying cans and scratch lines in the sky. 500 recovers, and both
   500 rows beat the control's 500 on ANIMA; nozomi 500 raw keeps the scratch lines, 500 EMA is
   clean. The style drifts (brown hair, cap, school buildings, harder lines). Fails as written.
   Better, then worse, then better again is what a walk does, not a slower decline.
2. |ΔW| 2.80 / 3.92 / 4.70 / 5.35 (control 2.77 / 3.98 / 4.85 / 5.50), cos between moves +0.008,
   +0.008. Still √t and orthogonal.
3. Probe (`bench/turbo/results/20261111-1913-grad_noise_1111_1740/`, per-draw as before): mean
   share 2.9 % against 3.0 %. The averaged signal's rms is 0.65× the control's all run, and the
   DM-term metrics (`dm`, `cos`, `rel_gap`, first draw) lead the control over steps 300–400 only.

Why 4 draws buy less than 4: the DM norm's σ_a tail. Bin medians 1.24 / 2.03 / 2.36 / 4.93 with
21.9 and 30.5 above σ_a 0.95 (1111_1115 bundle; 1111_1740's top is 11.1). Norm-weighted the DM
mean share is 1.6 %, unit-weighted 4.3 %: the large draws are mostly noise. Resampling the 96
probe draws (independent draw noise assumed), the effective number of draws out of 4 is 2.23
iid, 1.93 stratified by σ_a (one draw per quarter forces a tail draw into every step), 2.48 with
σ_a capped at 0.9, 2.84 with each draw divided by its bin's median norm, 4 with a per-draw
backward normalised to unit norm.

## Next: logit-normal σ_a (not run; config `output/ckpt/next_logit_normal.config.toml`)

`1111_1740` with `dmd.dm_tau_distribution = "logit_normal"`, `dm_tau_scale = 1.0` (σ_a =
sigmoid(N(0,1)) on the DM / MFD query; the GAN generator renoise reuses it; CDM, fake update
unchanged). Control: `1111_1740`. On the 96 probe draws it lifts the effective draws to 2.62 of
4 (P(σ_a > 0.9) 1.4 % against 10 %). It changes the objective, not only its variance: the
σ_a > 0.75 mass drops from 25 % to 14 %. Read as `1111_1740`, criterion 1 against its renders.

## Checkpoint averages (2026-11-11): taking the walk out does not beat 250

Exact ΔW averages of saved checkpoints (`merge_loras.py --normalize off --adaln sum`, equal
weights), rendered as the sweeps above (2 steps, cfg 1, same prompts and seeds). Grids:
`output/tests/superturbo/avg/cmp_{anima,nozomi}.jpg`.

| Average | ANIMA sign | nozomi |
|---|---|---|
| `avg3_1115`: `1111_1115` at 250 / 375 / 500 | letters still broken and doubled, level with its 250 | stray dark shapes, a blue can; below its 250 |
| `avg2_500`: `1111_1115` and `1111_1740` at 500 | bolder, cleaner strokes; spelling still wrong | two cans on seeds 43, 44 |
| `avg6_both`: both runs at 250 / 375 / 500 | the calmest row, seed 3 nearly "ANIM.A" | two cans on seeds 43, 44 |

- The averages undo what `1111_1115` drew at 500 (benches, vehicles, brown hair): the scene is
  trees and sky again. The walk is what degrades the late checkpoints.
- None of them is above `1111_1115` at 250, which stays the cleanest nozomi row. With the walk
  removed after the fact, the result is the 250 level.
- Read by Claude on the 448 px grids, not per glyph at full size.

What this does not test: whether a cleaner gradient during training moves the point the student
settles at. The fake tracks a student that jitters every step, and every step is a unit-norm
vote under the clip.

## Next: gradient accumulation (`1111_2056`, queued 2026-11-11)

`optim.grad_accum` (default 1; `steps.ClippedGradMean`, test in `tests/test_turbo_mfd.py`): the
student optimizer steps once per `grad_accum` loop iterations on the mean of the per-iteration
clipped gradients. Clipping before the sum keeps a high-σ_a draw at one vote; clipping the sum
would repeat the 2.23-of-4 loss measured for `dm_draws`. The fake and the disc update every
iteration. `iterations`, `save_every`, the GAN delay and the logs count loop iterations; the
student LR schedule and the EMA count student updates.

From the probe's shares, 4 independent (sample, draw) gradients take the mean share from 3 % to
about 11 % (8: about 20 %); `dm_draws` caps near 13 % because it leaves the between-sample 20 %.

`1111_2056` (config `output/ckpt/1111_2056.config.toml`): `1111_1115` with `grad_accum = 4`,
1000 iterations = 250 student updates, `fake_steps_per_student_step = 1` (4 fake and disc
updates per student update, as in the control, each on a different x_θ), GAN delay / ramp
400 / 400, `save_every = 500`, `ema.window = 0.2` (the control's 50-update span). Smoke: 8
iterations ran clean, peak 13.21 GB.

Read, fixed here: the final checkpoint (250 student updates, raw and EMA), same prompts and
seeds, against `1111_1115` at 250 and `avg6_both`. It passes if it beats both. Level with
`avg6_both` means the 250 level is this objective's 2-step ceiling, and the 2-step line closes.

## Later, each on its own

- `lr_schedule = "cosine"`: the one setting seen to stop the walk.
- The σ_a ≥ 0.75 tail: a cap or a logit-normal σ_a. The probe does not say how much of the
  high-noise signal a 2-step student needs.
- A larger student batch for the between-sample 20 %.
