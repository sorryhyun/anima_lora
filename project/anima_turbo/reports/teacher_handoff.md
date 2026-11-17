# DM on steps 1–3 from a teacher-forced handoff (2026-11-16)

The anchor alone fits step 0 and the renders stop improving: a bigger target cache gives the same
breaks. This report holds the cache-scale run that showed it (`turbo_1116_2209`) and the arm that
follows from it (`turbo_1116_2221`): the DM terms on steps 1–3 only, rolled from the input a fitted
step 0 hands them rather than from the student's own step 0. 

## Where the anchor alone stops: `turbo_1116_2209`

`turbo_1116_2019` on the 512-prompt cache (`post_image_dataset/turbo_anchor_cache/m15_k14of28_p512`,
512 prompts × 2 ε = 1024 pairs, against 2019's 64 × 2), `grad_accum 2`, 1024 iterations: 512
student updates, one pass over the pairs. Config `../configs/turbo_1116_2209.toml`.

- **Speed.** 2.5 it/s (0.4 s an iteration, one student forward and backward); the live 14 of 28
  anchor (`1810`) ran 3.4 s an iteration. 1024 iterations take about 7 minutes.
- **Train anchor MSE**, last 10 % of the run: 0.065. Every pair is seen once, so each logged loss is
  on a pair the student has not trained on. 2019's 0.036 is on 128 pairs seen about four times and
  does not compare.
- **Renders** (`output/tests/turbo4/cmp/p512_{ba5,cafe,bocchi}.jpg`; rows teacher, `1810` live 250,
  `2019` p64 500, `2209` p512 1024): each seed draws 2019's layout and breaks where 2019 breaks.
  Small moves toward the teacher: ba5 seed 1 gets the lying pink-haired girl back in the lower left
  (crowded, faces merged), the café seed-1 apron turns from brown to the teacher's blue, the bocchi
  seed-1 guitar is one piece. The ba5 lower halves and the bocchi seed-4 neck through the mic stand
  stay broken.
- The in-run diversity val (`AC sim` 0.55 against 2019's 0.35) is on the last prompt of each cache,
  a different prompt per cache, and does not compare.

So 8× the prompts and a full pass leave the finish where the 1k runs on the live anchor left it
(`v11_init.md`, `1653`). The fit of step 0 has a ceiling on what it gives the renders.

## The idea

DP-DMD splits the student by step: step 0 regresses on the teacher's trajectory (the anchor,
diversity), the later steps take the DM signal (the finish), and the gradient stops at the step
boundary (`detach_after_first`, `dm_first_step` off). `plan.md` candidate 4 is that split from an
anchor student.

The new part is the input the DM steps train on. The student's σ grid is 1, 0.9, 0.75, 0.5, 0, and
the 14 of 28 anchor is the teacher's mean velocity over σ [0.75, 1]. A step 0 that meets the
target puts x at

    x_1 = ε − (1 − σ_1) · v_target = ε − 0.1 · v_target,

on the chord from ε to the teacher's state at σ 0.75. With `dpdmd.teacher_handoff` the rollout to
the grad step g ∈ {1, 2, 3} starts there instead of from the student's own step 0. Steps 1–3 then
train on the distribution a fitted step 0 hands them, which is the handoff where the target as step
0 already finishes cleanly (`anchor_only_recipe.md`: the 14 of 28 target as step 0 under v1.1's steps
1–3, cos 0.967 to the teacher's image, sameness 0.249).

- **Step 0 is held by the cached anchor in the same iteration** (`div_weight` 0.1, backwarded before
  the DM term). The gradient split is in the graph only: one LoRA serves every σ, so the DM signal
  on steps 1–3 can still move step 0 through the shared weights (`1355` drew one layout per prompt;
  `1552`, joint with `dm_first_step` on, fell behind the anchor alone on every probe number).
- **What the cache saves.** The 29 teacher forwards of the anchor rollout. The DM terms (the CFG
  teacher at the renoised x_θ, the fake, four fake updates, CDM, the GAN) depend on the current
  student and fake and stay per iteration.

Code: `dpdmd.teacher_handoff` (`scripts/distill_turbo/config.py`, `steps.py::student_rollout`); the
anchor cache now also feeds the DM path.

## The arm: `turbo_1116_2221`

Config `../configs/turbo_1116_2221.toml`. `1552`'s joint recipe with:

| Field | Value |
|---|---|
| student init | `turbo_1116_2209` EMA 1024 (step 0 fit on the same cache) |
| fake init | the v1.1 delta, `fake_warmup_steps` 200 |
| `dpdmd` | `teacher_handoff` on, `dm_first_step` off, `div_weight` 0.1, anchor from the p512 cache (14 of 28) |
| terms | DM, `cdm.weight` 1.0 (cond), `gan.weight_gen` 0.03 (1552's) |
| length | 250 iterations, `grad_accum` 1, cosine at 5e-5 |

**Read, fixed before the run**, against `2209` EMA 1024 untrained:

- renders (`judge_hard.txt`, 4 steps, cfg 1, merged15 int8): the finish at 2209's breaks (ba5 lower
  half seeds 1–2, bocchi guitar / mic seeds 1 and 4, the café aprons) gets cleaner, with each seed's
  layout kept from 2209;
- a layout that leaves 2209's / the teacher's, or one layout across seeds, means the DM signal
  reached step 0 through the shared weights;
- the train anchor MSE on the cache stays near 2209's 0.06.

## Results

Grids `output/tests/turbo4/cmp/handoff_{ba5,cafe,bocchi}.jpg` (rows teacher, `2209` EMA 1024, `2221`
EMA 250). Claude's read; the user's read is the gate.

| | s / it | train anchor MSE (last 10 %) | val `AC sim` (same prompt) | wall |
|---|---|---|---|---|
| `2209` EMA 1024 (start) | 0.4 | 0.065 | 0.555 | 7 min |
| `2221` EMA 250 | 4.5 | 0.065 (0.061 at the start) | 0.556 | 21 min with the head-start and compile |

DM terms over the run: `dm_cos` 0.979 → 0.988, `delta_dm_rms` 0.21 → 0.16, `fake_loss` 0.12 → 0.13.

- **Step 0 held.** Every seed keeps 2209's layout, the seeds stay apart, and the anchor MSE and the
  diversity val do not move. Through the shared weights the DM signal on steps 1–3 did not reach the
  layout in 250 iterations.
- **The finish moved, both ways.**
  - Toward the teacher: café seed 1 (the white-haired girl's light-blue apron with its pocket, black
    tights, the tray with coffee and cake), café seed 4 (she holds the tablet as in the teacher's),
    ba5 seed 2 lower half (the lying girl and the girl beside her come apart instead of merging).
  - New breaks: subtitle-like boxes of garbled Japanese text over the café renders (seeds 1–3), the
    bocchi seed-1 guitar loses its neck (body at the waist, headstock in the raised hand; 2209's was
    one piece), denim legs under one café apron.
  - Unchanged: the ba5 lower halves of seeds 1 and 3 stay crowded, the bocchi seed-4 neck still runs
    through the mic.
  - Overall higher contrast and more high-frequency detail than 2209.

Verdict: the mechanism passes (DM on steps 1–3 from the teacher-forced handoff leaves step 0 and the
layout alone); the finish is mixed at 250, with as many new breaks as repairs. Not a win over 2209
yet. Which term brings the text boxes (DM, CDM or the GAN) is not separated.

## After this report

The handoff probe of `2221` and the arm this report proposed next (anchor-only and DM-only
iterations drawn at random) are in `step0_depth.md`: `2221` sits slightly below `2209` on every
probe number, and the random draw is closed (the clip it was meant to unshare does not bind).
