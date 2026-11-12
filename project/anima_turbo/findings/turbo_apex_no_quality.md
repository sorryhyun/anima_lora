# Turbo APEX on Anima: unstable as DMD, blurred as regression — CLOSED

Status: CLOSED 2026-11-01. One LoRA as generator, reference and fake
([proposal](../proposal/turbo_apex.md)) did not reach DP-DMD's 4-step quality in either form we
ran. The step is 3–5× cheaper; the student explodes in the proposal's form and loses its
sharpness in the paper's. Phase 0a (endpoint weighting on DP-DMD) has no APEX code and stays open.

Sources: APEX, arXiv:2604.12322; the public code at `LINs-lab/APEX`
(`src/methodes/apex/apex.py`, `qwenimage_lora_1order.yaml`). Dataset: `post_image_dataset/lora`,
batch 1, warm start from the v1.1 delta, int8 stream base, fp32 stream, LR 5e-5.

## Results

![4-step grids at CFG 1.0, seed 42: v1.1 delta, the arm at 500 / 1000 / 1500, v3_longer, p0b_dpdmd](assets/turbo_apex_4step_grid.jpg)

| Arm | Form | Outcome | s/iteration | Peak GB |
|---|---|---|---|---|
| A | `F − d` on rollout points, `L_sup` apart | `x_pred` std 0.77 → 7.6 from step 55, killed at 110 | 2.08 | 11.56 (30-step smoke) |
| B | `L_mix` + `L_fake` on real `x_t`, head start 200, 150 steps | stable, `roll_std` 0.38–0.75 | 1.33 | 11.85 |
| C | as B, no head start, 150 steps | stable, `roll_std` 0.40–0.72 | 1.31 | 11.78 |
| D | as C, 1500 steps | stable, `roll_std` 0.33–0.80, **grid fails** | 1.33 | 12.01 |

The shipped recipe runs at 6.7–7.0 s/iteration.

On arm D's grid every checkpoint is worse than its own warm start: blurred, dark, backgrounds
gone, and posterized from 1000 on. `WELCOME` reads at 500 and is gone at 1000. The v1.1 delta,
`anima_turbo_v3_longer` and `anima_turbo_p0b_dpdmd` are all sharp on the same seed. Six
`general_prompts` lines at one seed, on the default bf16 base at inference, for every arm. The
proposal's prompts 0 / 1 / 8 / 13 were not rendered. The gap is too wide for the sample size to
change the verdict.

The LoRA did move, so the failure is not a dead run. Distance from the v1.1 delta in merged
`ΔW`, relative to the delta's norm:

| Checkpoint | rel. dist. | cos | norm |
|---|---|---|---|
| 150 (arm C) | 0.45 | 0.90 | 3.18 |
| 500 | 1.08 | 0.63 | 4.38 |
| 1500 | 1.15 | 0.60 | 4.55 |
| `v3_longer` | 3.82 | 0.21 | 12.36 |

The delta's own norm is 3.17. Between 1000 and 1500 the distance is 0.09. Arm D ends at cos 0.14
to `v3_longer`.

## What the training scalars do not show

Arm D's scalars are flat and healthy while the grid degrades: `mix_v` 0.03–0.05, `d` rms about
0.10, `x_pred` std 0.62, `roll_std` mean 0.50–0.57. `roll_std` catches a runaway and nothing
else: a blurred sample has a normal latent std. Render a grid at the first checkpoint.

## Why (inferred, not measured)

- **Arm A.** `v* = F − d` holds `F` relative to itself, and the rollout point has no data term.
  `L_sup` stayed near 0.2 on real points during the blow-up. `x_pred` left its range while `d` was
  still 0.13, and `d` grew afterwards (0.96 at step 90, 1.75 at 105), with `cos(d, d')` going from
  −0.03 to 0.96.
- **Arms B–D.** With `v_fake ≈ F`, the fixed point of `(1−λ)·v_sup + λ·v_fake` is `v_sup`, a guided
  velocity field. Euler at 4 steps on a velocity field gives the conditional mean. This is the
  proposal's Open question 1: `L_sup` pulls a warm-started student back to a velocity field.

## Paper and code differ

The paper's form is arm B's. The public LoRA config runs something else, and arm A's `F − d` is in
it (`dist_match`):

- `F − d` on a one-step sample from pure noise, weight `dist_match_cof = 0.5`, on the same point
  an RCGM target supervises. No multi-step rollout.
- RCGM needs the target-time input `tt`, which Anima does not have.
- `barron_reweighting` with alpha 1 is close to per-sample RMSE, so a spike does not dominate.
- The fake is split by time sign, LR is 1e-4, `lora_target_modules` includes `timestep_embedder`.
- Neither the paper nor `training_step` has a head start or ramp.

Not tried: that combination without `tt`. It would drop the term that anchors the point.

## `−c` separation (Phase 0b)

Bare `−c` is a usable branch, so the failure is not in the condition shift. Over 12 items and 10 τ
bands (`bench/turbo/apex_cond_shift.py`), in units of `rms(v_c − v_u)`:

| | `rms(v(−c) − v_c)` | `rms(v(−c) − v_u)` | `cos(v(−c) − v_u, v_c − v_u)` |
|---|---|---|---|
| base | 1.2–7.5 | 0.75–7.2 | about 0.05 |
| v1.1 delta | 0.82–6.1 | 0.74–6.1 | 0.08–0.59 |

The head start made no visible difference at 150 steps (arms B and C).

## History

The repo carried an earlier APEX for plain LoRA, in the paper's form, with `λ = 0.5` and an
FM-first ramp (`_archive/apex/`). It was stable over 200–500 steps and was archived on 2026-05-11.
No reason for that was recorded.
