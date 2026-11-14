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
- Hub v3 on merged14 untrained also draws seeds as varied as the teacher's (sameness 0.326 against
  0.328), and 250 iterations of the v1 recipe from that start undo it (`reports/warm_v3.md`).

## A/B protocol

1. **250 iterations on the v1 recipe**, compare EMA 250. Control: `turbo_1113_1105` (27 min).
2. **Renders are the gate.** 4 steps, cfg 1, `bench/turbo/judge_hard.txt` seeds 1–4 at 768 × 1344
   on the merged14 DiT, beside the teacher (`output/tests/turbo4/teacher_m14/`). Grids go to
   `output/tests/turbo4/cmp/`.
3. **Probe beside the renders** (`bench/turbo/probe_handoff.py`, about 3 min per student):
   held-out anchor MSE, the `S` cosine, and the sameness across seeds (teacher 0.328, control
   0.496). A warm-started arm is also compared with its starting point, untrained.
4. The control's twin is `turbo_1114_1745` (seed 43). On the dynamic grid the two seeds are far
   apart: sameness 0.496 / 0.675, `S` cos 0.547 / 0.453, `T>S@1` cos 0.779 / 0.745, anchor MSE
   0.149 / 0.188. Arms on the static grid are read against `turbo_1114_1709`, which has no twin.

## Warm start from Hub v3: done, regressed (`reports/warm_v3.md`)

`turbo_1113_1221` passed the gate written here, which compared it with the control only. Hub v3
on merged14 with no training beats it on every number: sameness 0.326 (teacher 0.328) against
0.465, `S` cos 0.639 against 0.593, anchor MSE 0.063 against 0.070. The user read the renders as a
regression from v3. The v1 recipe on the m14 teacher pulls any start toward the control, so the
1k run from this start is dropped. The best 4-step result on merged14 is now merged14 + Hub v3,
untrained. `turbo_1113_1309` (`dm_first_step` off) regresses the same way: step 0 still makes the
seeds alike at v3's anchor MSE, and steps 1–3 lose teacher likeness. `turbo_1113_1438` (the fake
seeded from v3 as well, as v3's own run seeded both stacks from one file) regresses the same way:
sameness 0.483, on `1221`'s side of the read fixed in its config, so the cold fake is not the cause.
By that read the line closes on merged14 + Hub v3 untrained. What comes next is not decided.

## After it

- **Static grid** (`dmd.dynamic_schedule` off): done 11-14, adopted on the user's read of the
  renders (`turbo_1114_1709`, grids `output/tests/turbo4/cmp/static_twin_*.jpg`). Sameness 0.445
  against 0.496 / 0.675 for the two dynamic seeds, `S` cos 0.567 against 0.547 / 0.453, `T>S@1` cos
  0.823 against 0.779 / 0.745. Anchor MSE 0.171 sits between the two (0.149 / 0.188), and the café
  sign text is worse than the control's. One static seed. It gives up other step counts.
- **Cond-only L_CDM.** Our L_CDM takes the CFG'd teacher; the paper (arXiv:2605.06376 Eq. 9) and
  its code take the cond-only one, with CFG as a separate on-trajectory loss. On the static arm's
  step-248 bundle the cond-only signal is 0.58 of the CFG'd one in rms (cos 0.71), and the CFG'd
  signal at τ̂ > 0.67 is mostly the CFG part (`bench/turbo/probe_cdm_signal.py`, result
  `20261114-1739-cdm_signal_turbo_1114_1709`). A distinct arm, on the static grid. Not built.
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
