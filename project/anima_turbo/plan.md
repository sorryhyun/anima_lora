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

## 1. Can v1.1 finish from the teacher's σ 0.9 state?

Facts:

- Question: handed the teacher's own state at σ 0.9 (7 of 28), does the v1.1 delta, untrained,
  draw the teacher's image through σ 0.9 → 0.75 → 0.5 → 0? Yes → fitting step 0 to 7 of 28 is the
  whole job, and the cache is rebuilt at 7 of 28. No → steps 1–3 need training on this grid, and
  the objective for them is chosen first (under "To measure again").
- The arm is `T>S@1` of the handoff probe on the shift-3 grid. It was run on 2026-11-16 on the
  v1.1 delta untrained, merged15: cos to the teacher's image 0.932 (group 0.914, café 0.946,
  guitarist 0.936), sameness 0.301. Results and renders:
  `bench/turbo/results/20261116-1550-handoff_m15_v11_untrained/` (local).
- The user's read of those renders is not recorded. Claude's read (guitarist grid only, arm labels
  visible): hard to tell from the teacher's on all four seeds.
- Next: the user's read of the 12 `T` / `T>S@1` pairs. No run is needed for it.
- If yes, the cache: `python -m scripts.distill_turbo.anchor_cache --out
  post_image_dataset/turbo_anchor_cache/m15_k7of28_<set> --k_anchor 7 --teacher_anchor_steps 28`
  with the 14 of 28 caches' seed and prompts. A pair costs 14 teacher forwards against 28; the
  build time is not measured.
- Reference only, not a basis: `turbo_1117_0823` (fitted to 14 of 28) read MSE 0.037 to the 7 of
  28 target without training on it, and its velocity mixed toward that target read cos 0.863 at
  MSE 0.021, 0.906 at 0.009 and 0.944 at 0 (`_archive/anima_turbo/reports/step0_depth.md`, local).

Interpretation:

- `T>S@1` uses the teacher's real σ 0.9 state and no training of ours, so it is not touched by the
  anchor mismatch.
- How deep the step-0 fit has to go on 7 of 28 is not known; the mix numbers come from a student
  fitted to another target.

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
