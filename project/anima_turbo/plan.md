# Plan: the 4-step line after `v4_preview`

Rewritten 2026-11-17. This file holds what is open; finished runs and probes are in `reports/` and
the run table (`reports/runs.md`). The closed 2-step plan is
`91d476e0:scripts/distill_turbo/plan.md`.

## Where the line stands

- `v4_preview` (`1112_1512`, 1k EMA on the `1112_v1.md` recipe) is shipped. The recipe in
  `configs/methods/turbo.toml` has since taken the static grid and the cond-only L_CDM
  (`reports/grid_cdm.md`).
- Every student of the DM recipe on merged14 / merged15 draws one layout per prompt
  (`reports/warm_v3.md`, `reports/m15_1k.md`). The line since then starts from the official turbo
  v1.1 delta, which has the best steps 1–3 on merged15 with no training, and fits step 0 alone on
  the anchor (`reports/v11_init.md`, `anchor_only_recipe.md`).
- **The breaks left in the renders are step 0's residual to the anchor target**
  (`reports/step0_depth.md`). With the target itself as step 0 the student's steps 1–3 draw the
  teacher's image (cos 0.966). Removing a quarter of the residual clears most of the guitarist's
  breaks; half of it reads like the teacher's own step 0.
- **More pairs is a shallow slope.** The full-dataset cache (`turbo_1117_0823`, 6484 pairs, one
  pass) takes the held-out anchor MSE 0.078 → 0.070 and `S` cos 0.760 → 0.801, with steps 1–3 and
  the seed variety held. It is the best anchor student and the start of the arms below.
- **Rank is not the limit.** Rank 96 at the √r alpha (`turbo_1117_0918`) ties `0823` on the probe
  and in the renders.
- DM on steps 1–3 from the teacher-forced handoff (`2221`) is not a win: it trains them where they
  are already clean (`reports/teacher_handoff.md`, `reports/step0_depth.md`).

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
ends at σ 0.75, the first boundary of the 2-step grid: a step 0 that meets this target is also the
2-step student's step 0 (`idea.md` 3). There the residual enters the state at 0.25 instead of 0.1,
so the depth matters more for 2 steps than for 4.

A rank change takes alpha by the √r law (alpha / √rank at 22.5: 180 @ r64, 220 @ r96, adaln 90 @
r16; `docs/anima/methods/adaln.md`, `docs/anima/findings/lora_recipe.md`).

## What is open, in order

1. **Rank 96 at student LR 2e-5.** `0918` with `optim.student_lr` 5e-5 → 2e-5, nothing else
   changed; the cosine then ends at 2e-6. One pass, about 43 min, read against `0918`.
   - The regression's fit is set by the LR at the end of training, not by length, draws or an
     average of checkpoints: a constant 2e-4 reads a held-out anchor MSE of 0.277, a constant 5e-5
     0.14–0.16, an annealed tail 0.11–0.13 (`_archive/anima_turbo/reports/step0.md`, merged14 S-b).
     No anchor run has had a peak under 5e-5, and the cosine has never gone under 5e-6.
   - `0918` trails `0823` on the train loss by 0.003–0.004 at the high end of the cosine and by
     0.001 at its end, on the same pairs in the same order (`reports/step0_depth.md`). The arm was
     set for a `0918` that ends above `0823`; it ended level on the held-out number, so the case
     for it rests on the gap closing with the LR and on the archive's result.
   - Rank 96 on the user's call. Rank 64 ties it, so the same arm on `0823` would read as well.
   - The risk is travel: a lower LR moves the student less far from the v1.1 start, and `0823`'s
     fall over its first half (0.088 → 0.070) is not split into travel and level.

2. **Paired feature matching on the σ 0.75 state.** The regression weighs every element of the
   velocity alike; what breaks the image is the part of the residual that steps 1–3 read.
   - Real: the cached target's state, `x_t = ε − 0.25 · v_target`, the teacher's own state at
     σ 0.75. Student: `x_s = ε − 0.25 · v_student`, with gradient.
   - Loss: the L2 between the frozen teacher's features of `x_s` and of `x_t` at σ 0.75 (the
     early-exit feature path the GAN generator term already runs), added to the anchor MSE.
   - No discriminator, no fake network, paired by ε, so nothing pulls a seed toward another seed's
     layout. Estimated 1.2–1.5 s an iteration against the anchor's 0.4; not measured.
   - Needs code: the state and feature path, one weight.

3. **A paired GAN on the same path**, if 2 moves the probe and leaves the finish short.
   - The discriminator head on those teacher features: real = `x_t` from the cache, fake = `x_s`.
     The anchor MSE holds the pairing; the adversarial term asks for a state that looks like a
     teacher state. Teacher samples as the real set is LADD's arrangement.
   - It is not the loop's GAN, whose real set is dataset latents against the student's x̂₀: on
     step 0 that separates a blurred x̂₀ from a sharp image and pulls toward the dataset rather
     than the soup teacher (`findings/turbo_gan_dm_grad_orthogonal.md`: texture, not diversity).
   - Watch: the GAN gradient on this loop ran about 6× the DM signal with a heavy tail (weight and
     clip), and 6484 fixed reals can be memorised (the discriminator loss; renoising the state to a
     higher σ is the augmentation).

4. **Steps 1–3 on the student's own handoff**, if step 0 cannot reach the depth. `2221` with
   `dpdmd.teacher_handoff` off, from the best anchor student: the DM terms see the inputs the
   student's step 0 hands them. The risk is steps 1–3 that redraw the layout; it reads as a fall in
   `T>S@1` cos. 250 iterations, about 20 min. Never run in this form.

5. **Two passes over the full cache** (12968 iterations, about 87 min). Low expectation: length was
   flat from 250 to 1000 iterations, and a second pass re-reads pairs (`reports/step0_depth.md`,
   `2019`).

Parked:

- **The 7 of 28 target.** Nearer in MSE and worth less per unit of it; ceiling 0.944 against 0.967
  (`reports/step0_depth.md`). The user's call: keep 14 of 28.
- **DM on step 0** (`idea.md` 1 and 2). The step-0 problem is the precision of a paired map, and DM
  carries no pairing: it removes a residual by moving to the nearest mode, which is the one layout
  per prompt of S-c.1, `1355` and `1552`. It also brings the fake network back (4.5 s an
  iteration). Behind the paired forms above.

## A/B protocol

1. **Anchor arms run one pass of the full cache** (`m15_k14of28_pall`, 6484 iterations,
   `grad_accum 2`, about 43 min) and are read on the last EMA. **DM / GAN arms run 250 iterations**;
   one that wins at 250 needs a 1k check before it ships (`reports/m15_1k.md`).
2. **Renders are the gate, and the user's read decides.** The probe's `S` column pairs with every
   other probe of the day (CPU-drawn noise, `judge_hard.txt` seeds 1–4, 768 × 1344, merged15 int8);
   the judge renders from `inference.py` draw on CUDA and do not pair with it. Sheets go to
   `output/tests/turbo4/cmp/`.
3. **Probe beside the renders** (`bench/turbo/probe_handoff.py --dit
   models/diffusion_models/anima_merged15.safetensors --ts_handoffs 1 --st_handoffs , --anchor_k 14
   --anchor_steps 28`, about 3 min per student): held-out anchor MSE, `S` and `T>S@1` cos, and the
   sameness across seeds (m15 teacher 0.314). `--step0_mix` adds the depth arms.
4. **An arm is read against its starting point**, and the read is fixed in the config header before
   the run.
5. One seed per arm so far.

## Open

- **The recipe on the base teacher.** Every student of the DM recipe collapsed on a merged teacher
  and Hub v3 was trained on base; the split is in `reports/m15_1k.md`. It matters again if arm 4
  brings the DM terms back.
- **Which term draws the text boxes of `2221`** (DM, CDM or the GAN). Not separated; it clouds the
  finish read of any arm that turns those terms on.
- **Fake-lag probe**, before anything that touches the fake (student : fake 1 : 10, the VA-TDM
  teacher-regularised fake): freeze the student at a resume point, run about 40 more fake steps,
  and read how far the DM signal turns on fixed draws. Not built.
- **Café colour drift under cond-only L_CDM.** A seed-43 twin of `0958` and a weight-matched arm
  (`cdm.weight` ≈ 1.7) would tell seed luck, magnitude and direction apart. Not run.
- **2 steps.** Reopening it waits on a step 0 that meets the target above.

## Dropped without a run

- **Anchor-only and DM-only iterations drawn at random:** the clip it was meant to unshare does not
  bind on `2221` (`reports/step0_depth.md`).
- **Adaptive clip:** on v1's recipe the clip at 1.0 already acts on 98 % of the steps under AdamW, so every
  update has unit norm; a threshold at twice the median would give a spike twice the
  weight it has now.
- **A longer run from the `weight_svd` init:** v3 shows where it ends, at 16 h; a `weight_svd`
  student also stays inside its init subspace, and the working turbo ΔW sits outside it
  (`reports/warm_v3.md`).
