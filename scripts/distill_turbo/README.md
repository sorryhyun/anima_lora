# scripts/distill_turbo

The `make turbo` distiller (DP-DMD student, bespoke loop). User doc and knobs:
`docs/anima/methods/turbo.md`; structure: `docs/anima/structure/turbo.md`. `plan.md` here is
the open plan (averaging the DM signal over several query draws, from the superturbo walk and
gradient-noise measurements); `vatdm.md` compares VA-TDM (arXiv:2611.06796) with the loop and MFD. Elsewhere: data-space iteration
`docs/anima/proposal/turbo_data_space.md`; dataset / CFG 1 / CFG 4 / student moment gaps
`docs/anima/findings/turbo_moment_gaps.md`; the removed DMAD path's plan and its comparison with
the official code `_archive/turbo_dmad/`.

## Mean Flow Distillation in the DM slot (2026-11-04, branch `mfd_test`)

MFD (arXiv:2606.11155, ICML 2026) replaces the DM term's point delta `τ·(v_real − v_fake)` with
the gap between the CFG'd teacher's and the fake's endpoints after integrating both from σ_a to
σ_b in K Euler steps. The ODE Jacobian is treated as identity, so the signal still enters through
`(grad · x_θ)`. The paper distills a plain flow-matching teacher; no MeanFlow-trained model is
needed. It is a DMD-family method (the auxiliary flow model is our fake), not a consistency one.

- Code: `steps.mfd_surrogate`, `steps._mean_velocity`; dispatched in `distill.py` when
  `mfd_dtau > 0`. L_CDM keeps the point delta.
- Knobs (`[mfd]` / CLI): `dtau` (0 = off), `max_steps` (cap on K, default 4), `interval`
  (`to_zero`: σ_b 0; `uniform`: σ_b ~ U(0, σ_a)). K = ceil(max span / dtau), shared by the batch.
- `to_zero` with K = 1 is `dmd_surrogate` bit for bit (`tests/test_turbo_mfd.py`), so DP-DMD is
  the K = 1 end of one axis.
- The integrals are no-grad: no grad ckpt is needed and peak VRAM did not move (12.35 GB in the
  2-step smokes with and without MFD).
- At batch 1 and σ_a ~ U(0,1): dtau 0.1 / max 4 gives mean K 3.4; the paper's dtau 0.03 needs
  `max_steps 34` and gives mean K 17.2 (max 34).

### 100-step arms

Common to every arm: 2-step student, init `models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln`,
GAN off, fake warmup 200, int8 ConvRot (GPTQ) base, fp32 stream, batch 1, 100 iterations.
Render: `bench/turbo/real_prompts.txt` line 14 (the Monster can prompt), seeds 42–45, 2 steps,
euler, cfg 1.0, bf16 base.

| Arm (`output/ckpt/`) | CDM | dtau | mean K | s/step | `dm` mean ± sd | `rel_gap` sd | `cos` sd |
|---|---|---|---|---|---|---|---|
| `anima_turbo_dpdmd2_100` | 1.0 | – | 1 | 5.33 | 0.194 ± 0.057 | 0.046 | 0.012 |
| `anima_turbo_mfd2_100` | 1.0 | 0.1 | 3.4 | 6.32 | 0.167 ± 0.041 | 0.037 | 0.008 |
| `anima_turbo_dpdmd2_100_nocdm` | 0 | – | 1 | 4.33 | 0.207 ± 0.056 | 0.040 | 0.010 |
| `anima_turbo_mfd2_100_nocdm` | 0 | 0.1 | 3.4 | 5.34 | 0.172 ± 0.036 | 0.029 | 0.006 |
| `anima_turbo_mfd2_100_dt003_nocdm` | 0 | 0.03 | 17.2 | 11.30 | 0.185 ± 0.035 | 0.027 | 0.006 |

The sd columns are over the 20 rows of `output/logs/<arm>.progress.jsonl` (one row per 5 steps).
No arm had a `grad` row above 1.5× its median except the two CDM-on arms (2 each). `xpred` std
stayed at 0.58–0.59 everywhere.

Renders (untrained init for reference):

- **Untrained init:** clean but soft, one pose for all four seeds.
- **All trained arms:** sharper faces, more pose spread, and stray content the init never drew:
  dark shapes in the sky, a second can or hand, background structures.
- **The stray shapes sharpen with K.** At K = 1 they are shapeless smudges, at K 3.4 partial
  bird/arrow forms, at K 17 clearly drawn spade-tipped demon tails (seeds 44, 45). The
  pointed ears and fangs are the likely source. K 17 also fills the sky with floating cans and
  towers.
- **CDM (MFD dtau 0.1, on vs off):** on keeps the sky as sky; off adds water-like ripples and a
  concrete railing. Same seed and init, so DP-DMD off and MFD 0.1 off share most compositions.

### Reading (the user's, 2026-11-04)

100 steps is too few to learn to draw the noise region cleanly. MFD, more so at larger K, turns
it into recognisable objects; CDM helps separate background / noise regions; DP-DMD learns to
draw meaningful objects less well. The no-CDM DP-DMD arm was run after this reading as its test:
on the K axis it drew the least legible shapes, as the reading predicts.

Two caveats on the mechanism:

- **Variance:** the DM signal's step-to-step sd drops about 35% from K 1 to K 3.4 and not
  further at K 17. That fits the paper's variance claim, with the K samples along one trajectory
  too correlated for its 1/K.
- **Bias:** a larger K also integrates the teacher's ODE more accurately, so the x0 the
  student chases is a sharper endpoint. Legible shapes at K 17 fit that as well as they fit lower
  variance. Neither reading makes the stray content wanted.

Limits: one training run per arm, one prompt, four seeds, 100 steps from a 4-step init. Whether
MFD changes the end state (rather than the first 100 steps) needs runs of 500+ steps; the first
one is below.

### 500 steps, dtau 0.03, no CDM

`anima_turbo_mfd2_500_dt003_nocdm`: the 100-step arm's settings, 500 iterations, 10.7 s/step,
checkpoints at 250 and 500 (only 500 rendered). Same render as above.

| steps | `dm` | `rel_gap` | `cos` | `grad` |
|---|---|---|---|---|
| 5–100 | 0.199 | 0.188 | 0.980 | 0.228 |
| 205–300 | 0.182 | 0.173 | 0.983 | 0.205 |
| 405–500 | 0.165 | 0.158 | 0.985 | 0.182 |

- The floating cans and towers of the 100-step arm are gone; the sky is mostly sky again and the
  compositions sit near the init's, with more pose spread.
- Stray content remains: demon tails on seeds 42, 44 and 45 (some attached, some floating), a dark
  bird-like silhouette (44) and dark blobs (43, 45), a grey pillar or tower in the background (43,
  45), and a smeared raised hand on 42.
- No CDM-on 500-step twin, so how much of the sky CDM would keep is not known.

Decision (the user's, 2026-11-04): keep training MFD, with longer runs.

## Anchor-free MFD with DM on step 0 collapses (2026-11-10)

`dm_first_step = true` with `div_weight = 0` (the MFD paper's anchor-free loop) collapses the
student within 250 steps. Step 0's x_pred becomes one flat colour, the same for every seed and
prompt; the 4-step render is a flat noise texture with no figure. Cause not yet isolated.

Common settings: teacher merged14 (int8 GPTQ base, fp32 stream), student and fake LoRA r64 / α180
from `weight_svd` (no warm start), adaln r16, `fake_transition` on, 4-step student with
`dynamic_schedule` (N ~ U{2..4}, so g = 0 on ~36% of student updates), MFD dtau 0.1 / max 4 /
`to_zero`, GAN and CDM off, lr 5e-5 for both, fake 4:1, grad clip 1.0, batch 1.

| Run (`output/ckpt/`) | fake warmup | `val/div_xpred_ac_sim` @250 | `val/div_ac_sim` @250 | `val/fm_mse` @250 |
|---|---|---|---|---|
| `anima_turbo_m14_dpdmd_svd_500_tafake` (reference: anchor on, DP-DMD) | 200 | 0.940 | 0.678 | 0.357 |
| `anima_turbo_m14_mfd_noanchor_nogan_nocdm_500_tafake` | 200 | 0.9994 | 0.077 | 0.371 |
| `anima_turbo_m14_mfd_noanchor_nogan_nocdm_fwu0_500_tafake` (stopped at 250) | 0 | 0.9996 | 0.155 | 0.452 |

The reference also differs in CDM 1.0 and GAN 0.03; the two MFD runs differ only in fake warmup.

- **Detect it with the validation probe, not the training scalars.** `rel_gap`, `cos`, `dm` and
  the grad norms of the collapsed runs sit in the reference run's range, because ε dominates
  every velocity. `div_xpred_ac_sim` near 1 (step-0 output identical across seeds) and
  `div_ac_sim` near 0 (final outputs uncorrelated noise) are the signature.
- **Renders** (4 steps, cfg 1; plus 1 step to see step 0): the ANIMA-sign and nozomi prompts both
  give a flat purple-grey (warmup 200) or grey (warmup 0) field at 1 step, and a uniform noise
  texture at 4 steps, at 250 and 500. The reference draws the figure even at 1 step.
- **Ruled out:** the g = 0 code path (`steps.student_rollout`: same `x_g − σ_g·v_g` form, sign
  and normalisation as g ≥ 1); `fake_transition` (a scalar σ embedding, so the fake still models
  the marginal, not the paper's App. C.4 conditional-velocity failure, whose point-mass collapse
  looks the same); fake LR warmup (second run). LoRA itself is not the problem: the paper's SANA
  run is LoRA r64 on student and fake (App. A.3.2).
- **Differences from the paper still open:** fake : student 10:1 there vs 4:1 here, grad clip
  0.5, logit-normal (s, t) with an adaptive ODE (max step 0.02) vs uniform σ_a, `to_zero` and
  K ≤ 4, and a 1-step student there vs one LoRA shared by 2–4 steps here.
- **Next split:** `dm_first_step = false` with `div_weight = 0` (step 0 frozen at init, no
  anchor) tells DM-on-step-0 from the anchor-free loop as a whole; TTUR 10:1 is the fake-tracking
  check.
