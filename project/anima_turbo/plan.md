# Plan: the 4-step line after `v4_preview`

Rewritten 2026-11-13. The recipe is `1112_v1.md`; finished work is in `reports/`. The closed 2-step
plan is `91d476e0:scripts/distill_turbo/plan.md`.

## Where the line stands

- `v4_preview` (`1112_1512`, 1k EMA) is shipped. Its gap to the teacher is composition: every
  student trained on merged14 draws one frontal, centred layout per prompt where the teacher varies
  scale, pose and camera (`output/tests/turbo4/cmp/sc1_bocchi.jpg`, `mfd_bocchi.jpg`).
- The gap sits in step 0, and the layout comes back in proportion to the step-0 anchor fit
  (`reports/handoff.md`, `reports/step0_fit.md`). The 1k students sit at a held-out anchor MSE of
  0.146–0.149, 0.85 of the teacher's own cond velocity.
- Hub v3 (base teacher, 10k iterations) sits at 0.30 of its reference and draws the view from
  below. Its step 0 carries over to merged14 with no training: 0.063 against the merged14 target.
- So the fit comes with length, and 10k iterations on merged14 is about 16 h. The next run asks
  whether a student can start from v3's step 0 instead.

## A/B protocol

1. **250 iterations on the v1 recipe**, compare EMA 250. Control: `turbo_1113_1105` (27 min).
2. **Renders are the gate.** 4 steps, cfg 1, `bench/turbo/judge_hard.txt` seeds 1–4 at 768 × 1344
   on the merged14 DiT, beside the teacher (`output/tests/turbo4/teacher_m14/`). Grids go to
   `output/tests/turbo4/cmp/`.
3. **Probe beside the renders** (`bench/turbo/probe_handoff.py`, about 3 min per student):
   held-out anchor MSE, the `S` cosine, and the sameness across seeds (teacher 0.328, control
   0.496).
4. The control has no twin. The noise floor is still arm 0's (`reports/arm0.md`), from a constant
   LR without `dm_first_step`. A twin (seed 43) is run when a read is close.

## Next: warm start from Hub v3

The control with `network.student_init_weights` set to
`models/loras/turbo_hub/anima_turbo_4step_v3.safetensors`. Same rank 64 / alpha 180 / adaln 16;
250 iterations, about 27 min. The fake starts as in the control (`weight_svd`, 200 warmup steps on
the student's outputs): v3's own fake was not kept, only its student LoRA is on disk. The one fake
of this line that survives is in `output/ckpt/1112_1512/1112_1512_resume.pt` (v1 at step 750).

Read, fixed before the run. It passes if all four hold:

- the guitarist renders keep the view from below on the seeds where the control draws the frontal
  figure;
- the held-out anchor MSE at EMA 250 is below the control's 0.149 (v3 untrained: 0.063);
- the sameness across seeds is below the control's 0.496;
- the five-girl prompt is one scene. v3 split multi-character prompts into panels and `v4_preview`
  does not.

What follows:

- **Pass:** the 1000-iteration run from the same start on the v1 recipe, judged against
  `v4_preview` and Hub v3. Its EMA is the next release.
- **The fit is lost by 250** (the MSE climbs back toward 0.15): the same run with `dm_first_step`
  off, which is how v3 was trained. The DM term on step 0 was there to move composition, and v3's
  step 0 already has it.
- **The fit holds but the panel split returns:** read the 125 and 250 saves for its direction
  before choosing a length.

## After it

- **Static grid** (`dmd.dynamic_schedule` off), config only. The dynamic grid draws its interior σ
  uniformly, so the inference transitions (σ 0.9 → 0.75 → 0.5 → 0) are never trained as such and
  the last step took the DM gradient on 76 of v1's 1000 iterations. It gives up other step counts.
- **Fake-lag probe** before anything that touches the fake (student : fake 1 : 10, the VA-TDM
  teacher-regularised fake): freeze the student at a resume point, run about 40 more fake steps,
  and read how far the DM signal turns on fixed draws. Not built.
- **Anchor target cache.** The target depends only on (prompt, ε) and the frozen teacher. Caching
  it takes about 2 s off a 5.75 s iteration and lets the regression run on a real batch. It is the
  fallback if the warm start does not hold its fit.

## Dropped

- **MFD in the DM slot** (`turbo_1113_1104`): the seeds of a prompt draw one image, and its steps
  1–3 carry most of that, so restricting it to steps ≥ 1 does not help (`reports/mfd.md`).
- **Adaptive clip:** the clip at 1.0 already acts on 98 % of the steps under AdamW, so every update
  has unit norm; a threshold at twice the median would give a spike twice the weight it has now.
  Under the cosine the one coupled burst of v1 was at steps 152–162.
- **Low-frequency weighting of the anchor loss:** the low band is already most of the error
  (`reports/step0_fit.md`).
- **A longer run from the `weight_svd` init:** v3 shows where it ends, at 16 h.

## Open

- **Sameness in steps 1–3.** With the teacher drawing step 0 the control still reads 0.405 against
  the teacher's 0.328: the plain DM term pulls the seeds together too. No arm aims at it yet.
- **2 steps.** Reopening it waited on a step 0 that carries the layout on 4 steps
  (`idea.md` 3). A warm-started student that passes is that step 0.
