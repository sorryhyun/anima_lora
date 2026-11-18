# Plan: the 4-step line, restarted 2026-11-18

Restarted on the user's call. Every student of this line up to 2026-11-18 was trained on an anchor
whose chord ended at σ 0.75 (6 of 12, then 14 of 28) while its step 0 covered σ 1 → 0.9 only, so
their reads do not carry over. The earlier plan, reports, run configs and proposals are in
`_archive/anima_turbo/` and `_archive/configs/anima_turbo/` (local). This file holds what is
carried, the two open items, and what is to be measured again.

## Carried

Facts:

- Grid: the static 4-step grid at shift 3, σ 1 → 0.9 → 0.75 → 0.5 → 0 (the user's call).
- Anchor rule: the step-0 target is the teacher's mean velocity over σ [σ_k, 1], and step 0 applies
  it over [σ_1, 1], so the chord ends on the student's first boundary. On this grid that is 7 of
  28. `setup.build_run` warns when the two differ.
- The anchor is rolled on the teacher's 28-step schedule. The 12-step rollout ends on a different
  image of the same noise (as step 0 it reads cos 0.818 to the teacher's image).
- Teacher: merged15 (the user's call: merged14 against merged15 is not reopened), CFG 4, 28 steps,
  shift 3, int8 `stream` with the fp32 stream. Its sameness on the judge set is 0.314.
- Init: the official turbo v1.1 delta (rank 96 ASVD, adaln included) truncated to rank 64, alpha
  180, adaln rank 16 / alpha 90 (alpha / √rank 22.5). Rank stays at 64 (the user's call).
- Step-0 fitting runs at `student_lr` 2e-5 and `grad_accum` 2 (the user's call).
- Judge set: `bench/turbo/judge_hard.txt`, 3 prompts (group, café, guitarist) × seeds 1–4. The
  user's read of the renders decides; probe numbers are reported beside them.
- Tools: `scripts/distill_turbo/anchor_cache.py` (ε is seeded by (seed, key, draw), so a cache at
  another depth reuses the same noise), `bench/turbo/probe_handoff.py` (`--student_sigmas` for a
  grid of its own), `bench/turbo/blind_sheets.py`, `deploy/`.
- The loop (`make turbo`) is the base DP-DMD loop: the anchored step 0, the DMD gradient on steps
  1..N−1 against a LoRA fake, and `dpdmd.anchor_only` for step 0 alone. L_CDM, MFD, the GAN, paired
  feature matching, the teacher-forced handoff and the dynamic grid were removed from the code on
  2026-11-18; they are in the git history.

## 1. Step 0 on 7 of 28 without losing steps 1–3

Facts:

- Done so far, in [reports/step0_k7of28.md](reports/step0_k7of28.md): the v1.1 delta untrained
  finishes from the teacher's σ 0.9 state (the user's read 8 / 4, near tie, taken as yes);
  `turbo_1118_1818` (step 0 alone on the 7 of 28 cache `m15_k7of28_p500`, one pass) won every
  probe number and lost the user's blind reads, S 0 / 12 and `T>S@1` 2 / 10.
- `1818`'s step 0 on the untouched v1.1 delta (`P>S@1`) also lost the user's blind read to v1.1
  S, 10 / 2 (report section 3).
- The v1.1 delta untrained, `T>S@1` against its own S: the user's blind read 6 / 6, level
  (report section 4).
- Not read: teacher T against v1.1 S.
- Reference only, not a basis: `turbo_1117_0823` (fitted to 14 of 28) read MSE 0.037 to the 7 of
  28 target without training on it, and its velocity mixed toward that target read cos 0.863 at
  MSE 0.021, 0.906 at 0.009 and 0.944 at 0 (`_archive/anima_turbo/reports/step0_depth.md`, local).

Interpretation:

- The teacher's real σ 0.9 state does not finish better than v1.1's own step 0, so the 7 of 28
  target is not the step-0 objective for v1.1's steps 1–3. What this line distills toward is open.

## 2. Superturbo (σ 0.75 → 0 in one jump): parked

Facts:

- Parked until step 0 is fitted (the user's call). The 2-step grid σ 1 → 0.75 → 0 needs a step 0
  on the 14 of 28 chord.
- The 14 of 28 caches stay: `post_image_dataset/turbo_anchor_cache/m15_k14of28_{p64,p512,p1024,pall}`
  (local). They load only under `k_anchor 14`.
- With a step 0 covering σ 1 → 0.75, the 14 of 28 students break more than on the shift-3 grid:
  `turbo_1117_1054` on σ 1 → 0.75 → 0.5 → 0.25 → 0 reads `S` cos 0.803 and sameness 0.406 against
  0.791 and 0.283 on shift 3 (`bench/turbo/results/20261118-1738-handoff_u4_075_1054/`, local;
  Claude's read with labels visible: extra hands, merged figures, a haze over the café), and
  `turbo_1117_0823` on the 2-step grid reads sameness 0.442
  (`bench/turbo/results/20261118-1634-handoff_2step_a1/`, local). In both, `T>S@1` reads 0.990.
- The jump σ 0.75 → 0 has never been trained.

Interpretation:

- On a grid whose step 0 covers a quarter of σ, the step-0 residual enters the state at 0.25, and
  an anchor MSE near 0.07 is not enough for it.

## To measure again

Nothing below carries a verdict from the archive.

- The depth of the step-0 fit on 7 of 28 (held-out anchor MSE against renders), and what moves it:
  the number of pairs, the schedule.
- The objective for steps 1–3 if item 1 says they need training: DMD as it stands in the loop, MFD,
  APEX, VA-TDM.
- The GAN (its form, its discriminator, where it reads), feature matching, L_CDM.
- Whether the handoff into steps 1–3 is the student's own step 0 or the teacher's.
