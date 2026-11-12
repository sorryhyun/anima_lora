# Step-0 arms: S-a ceiling and S-b anchor alone (2026-11-12)

`plan.md`, "Step-0 arms". The handoff probe (`handoff.md`) put the largest single loss of the
teacher's layout at the student's step 0 (σ 1 → 0.9), which the recipe trains only through the
anchor regression at `div_weight 0.1`. These arms ask whether the anchor target is the right
target (S-a) and how close a rank-64 LoRA gets to it when nothing else competes (S-b).

**The render reads below are Claude's, on downscaled grids. The user's read is pending.**

## S-a: step 0 taken with the anchor target itself

Probe arm `A>S@1` (`bench/turbo/probe_handoff.py`, on by default since this run): step 0 is
`ε − (1 − 0.9)·v_target`, where `v_target` is what `steps.teacher_anchor` computes in training (the
CFG 4 teacher's mean velocity ε → z_tk over 6 of 12 Euler steps, t_k = 0.75), and the student
finishes steps 1–3. This is what the step-0 regression gives at zero loss. The probe also reports
the held-out anchor MSE (the training's `div_loss`: student velocity at σ 1 against `v_target`) on
the judge prompts.

Run: `bench/turbo/results/20261112-1049-handoff_anchor/` (job `20261112-104354-e78614`). Same
students, prompts, seeds and noise as `handoff.md`; the `T`, `T>S` and `S` numbers reproduce that run
to the third decimal.

Low-passed cosine to the teacher's image, mean of 12 (per prompt: group / café / guitarist):

| Arm | twin EMA 250 | control EMA 125 |
|---|---|---|
| `T>S@1` (teacher's path to σ 0.9) | 0.695 (0.684 / 0.740 / 0.662) | 0.803 (0.783 / 0.843 / 0.784) |
| `A>S@1` (anchor chord to σ 0.9) | **0.672** (0.700 / 0.704 / 0.611) | **0.729** (0.798 / 0.776 / 0.612) |
| `S` (student's own step 0) | 0.455 (0.524 / 0.512 / 0.328) | 0.537 (0.618 / 0.537 / 0.456) |

Held-out anchor MSE (judge prompts, 768 × 1344): twin EMA 250 **0.163** (0.090 / 0.134 / 0.264),
control EMA 125 **0.180** (0.081 / 0.128 / 0.330). In training `div_loss` sat at 0.16–0.17 by
step 250, so the judge prompts are not out of the training range on average; the guitarist is
the hardest prompt for both.

Renders (guitarist grids): with the anchor's step 0, both students keep the view from below, the
silhouetted crowd in the foreground and a smaller figure; the control's EMA 125 keeps the jump on
seeds 1–3. The camera tilt is weaker than the teacher's own path gives. The `S` column on the same
noise is the large centred frontal figure.

Read, against the decision fixed in `plan.md`:

- **The target is right.** `A>S@1` sits nearer `T>S@1` than `S` on both students (0.672 against
  0.695 / 0.455; 0.729 against 0.803 / 0.537), and the renders keep the view from below. On the
  control's guitarist it is at the midpoint (0.612 against 0.784 / 0.456).
- **The chord costs little.** The anchor is a mean velocity to σ 0.75 applied over 1 → 0.9; against
  the teacher's own path to 0.9 it loses 0.02 (twin) and 0.07 (control). The anchor does not need to
  end at the student's first boundary first.
- **The loss is the fit.** The students' step 0 is far from the target they regress on (MSE 0.16–0.18,
  still falling in training). S-b goes ahead as planned.

## S-b: the anchor regression alone

Run `1112_1044` (`configs/1112_1044.toml`): arm 0's control with `dpdmd.anchor_only` on (one student
forward at σ 1 per iteration, MSE to the anchor, its own clip 1.0; no DM, CDM, GAN or fake update),
250 iterations, EMA 0.2. Read: held-out anchor MSE, and the probe's `P>S@1` arm (this LoRA draws step
0, an arm 0 student draws steps 1–3).

2.0 s per iteration, 8.5 min. Training `div_loss` by fifths: 0.220 / 0.172 / 0.156 / 0.155 / 0.141;
pre-clip student norm 18.7 / 9.5 / 8.4 / 6.9 / 6.8 (the clip at 1.0 acts on every step).

Probe runs: `bench/turbo/results/20261112-1104-handoff_step0lora/` (`P>S@1` on both arm 0 students,
job `20261112-105850-e35a76`) and `20261112-1109-handoff_anchor_ref/` (the teacher's own velocity at
σ 1 against the anchor target, job `20261112-110625-7d9245`).

| Arm | twin EMA 250 | control EMA 125 |
|---|---|---|
| `A>S@1` (anchor target) | 0.672 | 0.729 |
| `P>S@1` (S-b LoRA's step 0) | **0.472** (0.555 / 0.516 / 0.345) | **0.543** (0.638 / 0.520 / 0.472) |
| `S` | 0.455 | 0.537 |

Held-out anchor MSE (judge prompts; per prompt group / café / guitarist):

| Velocity at σ 1 | MSE to the anchor target |
|---|---|
| teacher, cond only, no LoRA | 0.171 (0.097 / 0.135 / 0.281) |
| teacher, CFG 4, no LoRA | 0.281 (0.233 / 0.198 / 0.413) |
| arm 0 control EMA 125 | 0.180 |
| arm 0 twin EMA 250 | 0.163 |
| S-b EMA 250 (anchor alone) | 0.157 (0.078 / 0.140 / 0.253) |

Read:

- **The anchor alone does not fit step 0 better than the full recipe.** S-b's step 0 moves the
  probe by +0.02 / +0.01 over the students' own, inside what separates the twin from the control.
  Step 0 is not losing to the DM / CDM / GAN terms in the joint step; a warm start from S-b would add
  nothing to S-c.
- **Every trained step 0 sits at the teacher's instantaneous velocity.** The base teacher's cond
  velocity at σ 1, with no LoRA, is 0.171 from the target; 250 iterations of the regression alone
  take that to 0.157, and the arm 0 control ends above it (0.180). The CFG'd instantaneous velocity is
  farther (0.281): the 6-step mean is not the first CFG step either.
- Whether this is a floor for a one-forward step 0 or only slow fitting at LR 5e-5 is open: training
  `div_loss` was still falling. S-b2 (`1112_1110`, LR 2e-4, nothing else changed) asks that.

## S-b2: the anchor alone at LR 2e-4

Run `1112_1110` (`configs/1112_1110.toml`): S-b with student LR 2e-4, same seed and data order.
Training `div_loss` by fifths: 0.264 / 0.248 / 0.245 / 0.245 / 0.230 (S-b: 0.220 → 0.141); pre-clip
norm 10.6 / 3.3 / 2.5 / 2.3 / 2.6. Probe: `bench/turbo/results/20261112-1124-handoff_step0lora_lr2e4/`
(job `20261112-111818-82ee1c`).

| | twin EMA 250 | control EMA 125 |
|---|---|---|
| `A>S@1` | 0.672 | 0.729 |
| `P>S@1`, S-b (LR 5e-5) | 0.472 | 0.543 |
| `P>S@1`, S-b2 (LR 2e-4) | **0.429** (0.559 / 0.462 / 0.266) | **0.487** (0.626 / 0.444 / 0.390) |
| `S` | 0.455 | 0.537 |

Held-out anchor MSE of S-b2's step 0: **0.277** (0.145 / 0.194 / 0.492), against 0.157 for S-b and
0.171 for the base teacher's cond velocity.

- **A higher LR fits worse.** Four times the LR raises the training loss from the first fifth on and
  ends the held-out MSE at the CFG'd teacher's level (0.28); its step 0 draws below the students' own.
  The regression is not step-limited at 5e-5.
- With the batch-1 regression, the per-step target moves with every (prompt, noise) draw; the larger
  step follows the draw. The same reading as the 2-step gradient probe (most of a step is the draw).

## Where this leaves step 0

- The anchor target is the right one (S-a). The trained step 0 does not reach it: alone or in the
  recipe, at either LR, it stays at the teacher's own instantaneous velocity or farther.
- Neither the competition with the DM terms (S-b) nor the LR (S-b2) is the lever.
- S-c candidates (`plan.md`): step 0 also takes the DM signal with the anchor kept as the diversity
  term (`dm_first_step` on), first; lowering the anchor's draw noise (several noises per iteration)
  second; a step-0-only adapter last, since it breaks the one-LoRA ship.
