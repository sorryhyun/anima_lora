# Plan: the 4-step arms still to run

Rewritten 2026-11-12. The recipe and what goes into or out of it is `1112_v1.md`. Finished work is
in `reports/`:

- `reports/walk.md`: what the 2-step line measured, the 4-step ΔW overlap, the clip as a normaliser.
- `reports/arm0.md`: control and null twin, the 4-step norm profile, bursts, the GAN-off probe.
- `reports/handoff.md`: step 0 is the largest single loss of the teacher's layout.
- `reports/step0.md`: S-a, S-b, S-b2, S-c.1, S-c.2 and the 500-iteration probe.

The closed 2-step plan is `91d476e0:scripts/distill_turbo/plan.md`.

## A/B protocol

A single raw checkpoint against another reads mostly run noise. Every A/B here:

1. **250 iterations**, `save_every 125`, EMA on. Arm 0 took 27 min per run (5.7–5.8 s per
   iteration after a 200-step fake warmup).
2. **Compare EMAs at 250.** The exact average of the raw 125 and 250 checkpoints
   (`merge_loras.py --normalize off --adaln sum`) is the cross-check.
3. **The null twin is `1112_0834`** (the control with `seed 43`). Its difference to the control
   `1112_0833` is the noise floor for every arm. A passing arm becomes the new control, with no new
   twin unless the step time or the loss terms change.
4. **Renders are the gate.** 4 steps, cfg 1, `bench/turbo/judge_hard.txt` (seeds 1–4 each) at
   768 × 1344 (W × H), beside the teacher (merged14, int8) at 28 steps, cfg 4
   (`output/tests/turbo4/teacher_m14/`). The three prompts: five named Blue Archive students with a
   position clause each and slapstick props; a two-girl café with Japanese menu text and an English
   sign; a solo guitarist jumping over a crowd, from below with a dutch angle.
5. **A distribution number beside the renders.** `bench/turbo/moment_gap.py`: S–T4 nW2² and the
   pooled spread ratio against the merged14 teacher at cfg 4. The script is built for the v1.1
   delta on the base teacher. It has to take the merged14 teacher and an arbitrary student LoRA
   first. The T4 set is rendered once and reused. Not run on arm 0. Its features are mean-pooled,
   so it does not read layout and does not gate the step-0 arms.
6. **The handoff probe beside the renders.** `bench/turbo/probe_handoff.py`, about 7 min for two
   students: the `S` row's low-passed cosine to the teacher's image of the same noise, with the
   `T>S@1` row as the level a teacher-quality step 0 gives, and the held-out anchor MSE. It pairs
   each image with the teacher's, so it does not pay the seed-to-seed render floor.

An arm passes if its EMA renders beat the control's on both prompts by more than the control
and the twin differ, and its S–T4 nW2² sits below the control's by more than |control − twin|.
For the step-0 arms the number is the probe's `S` cosine against the control's (0.54 at EMA 125)
and the twin's (0.46 at EMA 250), with the renders as the gate.

## Next: the v1 run

The first 1000-iteration LoRA, on the setup in `1112_v1.md`: the baseline with a cosine LR, the EMA
span counted in summed LR, `dm_first_step` on and `save_every 250`. About 100 min. Its read is fixed
in `1112_v1.md`: renders of the EMA at each save as the gate, diversity across seeds and
`val/div_ac_sim`, the handoff probe beside them.

- Renders gain against the control and diversity holds at S-c.1's level: v1 becomes the control for
  the arms below.
- Diversity falls with the length: the second run is v1 with `dm_first_step` off.

What the step-0 work before it settled (`reports/step0.md`): the anchor target is right; the fit of
the regression follows the LR at the end of training (0.11–0.13 annealed, 0.14–0.18 at a constant
5e-5), not the length, the draws or the competing terms; at 0.12 the trained step 0 still draws the
students' own layout. The two-stage plan (a long anchor-only run, then a warm-started joint run) is
dropped: the anchor-only run does not fit step 0 better than a joint cosine run does on its own.

## Arms not yet run

They change the DM term or the optimiser, which reach steps 1–3. Step 1 is the second loss of the
teacher's layout (0.13–0.16, `reports/handoff.md`). They follow the v1 run, each changing one
thing against the control of that time.

1. **MFD dtau 0.1 / max 4** in the DM slot. MFD's T2I result is a 4-step student (SANA, 3000
   steps), so 4 steps is the setting it was shown on. On 2 steps it sharpened stray content into
   objects as K grew; CDM held the sky (`reports/mfd.md`).
2. **Static grid** (`dynamic_schedule` off). The dynamic grid adds a draw (N, interior σ) to an
   already noise-dominated step, and inference runs the static grid.
3. **Adaptive clip** (`optim.grad_clip_mode = "adaptive"`, not built): threshold = k × a running
   median of the pre-clip norm (EMA in log space, k ≈ 2), per net, on the student and the fake.
   Ordinary steps pass at their own norm, spikes are capped, and the threshold follows the norm's
   drift over the run. It is the knob aimed at the divergence bursts (`reports/arm0.md`). σ_a
   normalisation (`dmd.dm_sigma_weight`: the DM loss times 1 / the running median norm of its σ_a
   bin) is not part of the arm: arm 0's profile came out under the 1.5× line.
4. **Teacher-regularised fake + corrected critic** (VA-TDM idea 2, `vatdm.md`, λ_r 1). Not
   built. It costs a teacher forward per fake step.

## Open

- **A/B length and schedule after v1.** The protocol above is 250 iterations on a constant LR
  against `1112_0833` / `1112_0834`. If v1 becomes the control, the arms need a length and a schedule
  that keep the step-0 fit an anneal gives; a twin of v1 is then the noise floor.
- **How low the anchor MSE has to go.** The gap to the anchor ceiling (`A>S@1` − `P>S@1`) is
  0.14–0.18 at a held-out MSE of 0.127 and 0.19–0.20 at 0.157. If it keeps shrinking in proportion,
  the regression has to go far below anything reached here, and nothing measured gets it there.
- **2 steps.** The 2-step grid's first step ends at σ 0.75, where the anchor lands exactly, and a
  handoff at σ 0.75 keeps 0.86–0.93 of the teacher's layout (`reports/handoff.md`). No handoff probe
  was run on a 2-step student. Reopening it waits on a step 0 that carries the layout on 4 steps.
- **Seed averaging.** With a shared share c ≈ 0.14, an exact ΔW average of k seeds raises it to
  kc / (1 + (k − 1)c): 0.25 at k = 2, 0.39 at k = 4. On 2 steps `avg2_500` (two runs) did not
  beat the 250 level, so this is cleanup, not a quality lever.
