# Plan: the 4-step line, restarted 2026-11-18

Restarted on the user's call. Every student of this line up to 2026-11-18 was trained on an anchor
whose chord ended at σ 0.75 (6 of 12, then 14 of 28) while its step 0 covered σ 1 → 0.9 only, so
their reads do not carry over. The earlier plan, reports, run configs and proposals are in
`_archive/anima_turbo/` and `_archive/configs/anima_turbo/` (local). This file holds what is
carried, the open items (preference RL first), and what is to be measured again.

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

## 1. Preference RL on the warm v1.1 delta

Replaces the step-0-on-7-of-28 item (the user's call, 2026-11-18). That item's record, with the
user's blind reads (`turbo_1118_1818` S 0 / 12; teacher T against v1.1 S, T 8 / S 16 over 24
rows), is [reports/step0_k7of28.md](reports/step0_k7of28.md). Not started.

Facts:

- The user's call (2026-11-18): the rest of this line is preference RL, trained on the warm v1.1
  delta on merged15 (the policy of item 1a), not on merged14 + Hub v3.
- Design and gates: [pref_rl/proposal.md](pref_rl/proposal.md). The student follows the
  gradient of a preference head fitted on the user's picks, through its own 4-step Euler rollout;
  step 0 trains first, with an anchor to the policy's own step 0.
- P0 and P1 ([pref_rl/p1.md](pref_rl/p1.md)), on merged14 + Hub v3, session `test`
  (25 rounds, 400 renders, 158 picked): P1 not passed. No head (DINOv3 CLS, PE-Spatial pooled,
  tagger logits, tagger embedding) ranks picked above unpicked within a round at p < 0.05; 10 of 25
  rounds sit at the 8-pick cap. The DiT-head basis is not built.
- Built: the round window and render server (`make exp-turbo-pref`, blind A/B mode),
  `pref_rl/features.py`, `pref_rl/head.py`. Not built: the P2 reward-gradient loop.
- v1.1 S sameness: 0.397 on `judge_hard` and 0.572 on `judge_random_1118` against the teacher's
  0.314 / 0.504 (report sections 5 and 6).

Steps, in order:

1. DiT-head basis (frozen merged-DiT block features of the renoised latent, caption-conditioned),
   read on session `test`'s 25 rounds against the P1 gate. GPU, minutes.
2. The round server's policy set to merged15 (its int8 GPTQ) + the v1.1 delta; new rounds on it,
   with a pick cap below 8 (4 proposed).
3. P1 gate read again on the v1.1 rounds, every basis.
4. If it passes: the P2 loop (step 0 only, anchor to v1.1's own step 0, head refreshed on new
   rounds), gated as in the proposal § 4 P2.

Interpretation:

- The P1 picks were made on another policy, so they rank bases but do not train a head for v1.1.
- v1.1 already sits above the teacher on sameness, so P2's "sameness not up" condition leaves
  less room than it did on Hub v3.

## 1a. Release of the 4-step as it stands

Facts:

- The user's call (2026-11-18): the untrained v1.1 delta on merged15 ships as
  `anima_turbo_4step_v4_longer` on `huggingface.co/sorryhyun/anima-turbo-4step`. Steps, checks and
  the open calls: [deploy/release_v4_longer.md](deploy/release_v4_longer.md). Not built or uploaded yet.

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
