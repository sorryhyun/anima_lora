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

## S-c.1: step 0 also takes the DM signal

Run `1112_1143` (`configs/1112_1143.toml`): arm 0's control with `dpdmd.dm_first_step` on, the
anchor kept as the diversity term. 250 iterations, 24 min (5.75 s/it), peak 13.7 GB. Step 0 drew the
DM gradient on 98 of 250 iterations (g = 0 / 1 / 2 / 3: 98 / 83 / 53 / 16).

Training: `dm_cos` 0.962 / 0.960 and `dm_rel_gap` 0.249 / 0.260 (steps 0–125 / 125–250), the arm 0
level; `gan_disc_margin` 0.21 / 0.26 against 0.10–0.17. No divergence burst: pre-clip `gn_student`
median 3.23, p95 11.2, max 28.5 (step 4), against max 240+ in both arm 0 runs. Validation:
`div_ac_sim` 0.68 / 0.55 and `div_xpred_ac_sim` 0.97 / 0.92 at 125 / 250, so no collapse signature.

Probe: `bench/turbo/results/20261112-1218-handoff_sc1/` (job `20261112-121134-58d01e`).

| | S-c.1 EMA 250 | S-c.1 EMA 125 | twin EMA 250 | control EMA 125 |
|---|---|---|---|---|
| `S` | **0.445** (0.525 / 0.424 / 0.387) | **0.441** (0.507 / 0.450 / 0.367) | 0.455 | 0.537 |
| `T>S@1` | 0.711 | 0.706 | 0.695 | 0.803 |
| `A>S@1` | 0.654 | 0.672 | 0.672 | 0.729 |
| held-out anchor MSE | 0.181 | 0.182 | 0.163 | 0.180 |

Renders (judge_hard, job `20261112-121134-388248`; grids `output/tests/turbo4/cmp/sc1_{ba5,cafe,bocchi}.jpg`,
rows teacher, control EMA 125, twin EMA 250, S-c.1 EMA 125 / EMA 250 / raw 250). Claude's read on
downscaled grids:

- **Guitarist: the pose moves, the camera does not.** Both S-c.1 EMAs draw the jump (bent knees,
  arms up, confetti) on all four seeds, where both arm 0 seeds stand. The figure stays large and
  centred, seen from the front; the teacher's view from below with the crowd silhouettes in the
  foreground is not there.
- **Five-girl prompt:** EMA 250 sprawls the group on grass with figures lying down, looser than arm 0's
  standing grid but not the teacher's close group from above.
- **Café:** EMA 125 draws blue aprons near the teacher's and spells `CAFE MILLENNIUM` on seed 2; EMA
  250 gives the same navy aprons on all four seeds and garbles the sign.
- **Raw 250 is broken** with no gradient burst: melted figures on the five-girl prompt, clothing
  dropped on the café (bare torsos on three seeds), extra hands on the guitarist. The EMA does not
  carry it.

Read, against the criterion fixed in `plan.md` (the probe's `S` against the control's 0.54 and the
twin's 0.46, renders as the gate):

- **S-c.1 does not pass.** `S` sits at the twin's level on both EMAs, and the anchor MSE did not move:
  the DM signal on step 0 does not bring step 0 nearer the anchor target or the teacher's image of
  the same noise.
- The renders move one thing above arm 0's seed floor, the guitarist's pose, on every seed. It does
  not show in the probe, which pairs each image with the teacher's on the same noise.
- Taking the DM gradient on step 0 removes about 40 % of the iterations from steps 1–3 and leaves
  the raw checkpoint worse at 250.

## S-c.2 probe: the anchor alone with four ε draws per iteration

Run `1112_1230` (`configs/1112_1230.toml`): S-b with `dpdmd.anchor_draws 4` (four fresh ε for the
same prompt per iteration, a teacher anchor on each, the gradients averaged before the clip), nothing
else changed. 6.9 s per iteration against S-b's 2.0 (each draw costs about one S-b iteration), 29 min.
Training `div_loss` by fifths (mean of the four draws): 0.187 / 0.164 / 0.169 / 0.147 / 0.135
(S-b 0.220 / 0.176 / 0.165 / 0.150 / 0.140). Probe: `bench/turbo/results/20261112-1304-handoff_step0lora_draws4/`
(job `20261112-123022-70d064`).

Criterion, fixed in the config header before the run: held-out anchor MSE below 0.140 (S-b 0.157
minus the 0.017 control–twin gap) and `P>S@1` above S-b by more than 0.05 on both students.

| | twin EMA 250 | control EMA 125 | held-out anchor MSE |
|---|---|---|---|
| `A>S@1` | 0.672 | 0.729 | |
| `P>S@1`, S-c.2 (draws 4) | **0.483** (0.565 / 0.523 / 0.361) | **0.569** (0.645 / 0.556 / 0.505) | **0.147** (0.074 / 0.132 / 0.235) |
| `P>S@1`, S-b (draws 1) | 0.472 | 0.543 | 0.157 |
| `S` | 0.455 | 0.537 | |

- **A small, consistent gain.** Every paired number favours S-c.2: anchor MSE per prompt 0.078 / 0.140
  / 0.253 → 0.074 / 0.132 / 0.235, `P>S@1` per prompt 0.555 / 0.516 / 0.345 → 0.565 / 0.523 / 0.361
  (twin) and 0.638 / 0.520 / 0.472 → 0.645 / 0.556 / 0.505 (control). The probe pairs each image with
  the teacher's on the same noise, so these differences do not carry the render seed floor. Under
  the repo's gate rule (every metric favours the arm → it passes, however small) this is a pass;
  the config header's size thresholds (MSE < 0.140, +0.05) were not met, and they should not have
  been the gate. No renders: a step-0 LoRA alone does not render.
- The gain is small for its cost: 5–10 % of the distance to `A>S@1`, at about 1.6 s per extra draw.
  Four times the samples per update bought 0.010 of anchor MSE, and the training curve tracks S-b
  from the second fifth on. Draw noise is a minor part of the step-0 limit, not the main part.
- **Not visible in the images.** Probe grid `1112_0833_125_ema_p2.jpg` (guitarist, control EMA 125
  drawing steps 1–3): `P>S@1` is nearly the same image as `S` on all four seeds (large centred
  frontal figure; seed 1 bends the knees slightly more), while `T>S@1` and `A>S@1` keep the view from
  below, the smaller figure and the crowd silhouettes. S-c.1's change was visible (the pose moved);
  this one moves the numbers only. User's call (2026-11-12): no S-c.1 + draws stacked run.
- **S-c.3 is already measured.** A step-0-only adapter trained on the anchor is what S-b and S-c.2 are:
  their LoRA only ever sees σ 1, and `P>S@1` uses it for step 0 alone. Building it as a separate
  adapter would rerun S-b at the cost of the one-LoRA ship.

## Length: the 500-iteration students (probe only)

Every arm above ran 250 iterations. The two 4-step students of 2026-11-10
(`anima_turbo_m14_dpdmd_svd_500` and `..._tafake`: same teacher, same anchor settings, LR 5e-5, the
joint recipe at `div_weight 0.1`) ran 500. Probe: `bench/turbo/results/20261112-1329-handoff_m14_500/`
(job `20261112-132721-ab44f2`, `T>S@1` / `A>S@1` / `S` only).

| | held-out anchor MSE (group / café / guitarist) | `S` | `A>S@1` | `T>S@1` | `A>S@1` − `S` |
|---|---|---|---|---|---|
| teacher, cond only, no LoRA | 0.171 (0.097 / 0.135 / 0.281) | | | | |
| arm 0 control EMA 125 | 0.180 (0.081 / 0.128 / 0.330) | 0.537 | 0.729 | 0.803 | 0.19 |
| arm 0 twin EMA 250 | 0.163 (0.090 / 0.134 / 0.264) | 0.455 | 0.672 | 0.695 | 0.22 |
| S-c.2 EMA 250 (anchor alone, draws 4) | 0.147 (0.074 / 0.132 / 0.235) | | | | |
| `m14_dpdmd_svd_500` | **0.118** (0.074 / 0.103 / 0.177) | 0.562 | 0.688 | 0.737 | 0.13 |
| `m14_dpdmd_svd_500_tafake` | **0.109** (0.073 / 0.098 / 0.156) | 0.492 | 0.637 | 0.697 | 0.15 |

- **500 iterations get past the instantaneous level.** Both students sit at 0.11–0.12, below every
  250-iteration LoRA (0.147–0.182) and the untrained level (0.171). Twice the length moved the MSE
  about five times as far as four draws did, inside the joint recipe.
- **The gap to the anchor ceiling shrinks with it**, from 0.19–0.22 to 0.13–0.15.
- **Renders move partly** (`_tafake`, guitarist grid, Claude's read on 320 px cells): the student
  alone has crowd hands in the foreground and a lower angle; the figure still fills the frame.
- **Confound.** Those runs used the cosine LR (the constant schedule came on 2026-11-11), are raw
  finals with no EMA, and share one seed. Length and the anneal are not separated.
- So S-b's "every trained step 0 sits at the teacher's instantaneous velocity" holds at 250
  iterations only. S-b2 showed that 4× the LR is too large a step for the noise; it did not test
  more steps at 5e-5, and S-b's training loss was still falling at 250.

## Length against the LR: `1112_1336`, `1112_1410`

`1112_1336` (`configs/1112_1336.toml`): S-b resumed to 1000 iterations on the constant LR, one ε per
iteration, EMA span kept at 50 updates. Its bundle is from step 125, so its 250 checkpoint is a
replicate of S-b's. 1.8 s per iteration. `1112_1410` (`configs/1112_1410.toml`): `1112_1336` resumed
from its step-750 bundle to 1000 with `lr_schedule cosine`, so only the last 250 iterations differ
(about 1.2e-5 falling to 5e-6 instead of 5e-5).

Training `div_loss` of `1112_1336` per 125 iterations from 125 on: 0.141 / 0.149 / 0.149 / 0.149 /
0.152 / 0.140 / 0.140. Pre-clip norm 6.8 → 4.9.

Probes: `bench/turbo/results/20261112-1405-handoff_sblong_mse/` (the four EMAs),
`20261112-1408-handoff_step0lora_sblong1k/`, `20261112-1420-handoff_step0lora_anneal/`,
`20261112-1425-handoff_step0lora_avg3/`.

| Step-0 LoRA | held-out anchor MSE (group / café / guitarist) | `P>S@1` twin EMA 250 | `P>S@1` control EMA 125 |
|---|---|---|---|
| S-b EMA 250 | 0.157 (0.078 / 0.140 / 0.253) | 0.472 | 0.543 |
| S-c.2 EMA 250 (draws 4) | 0.147 (0.074 / 0.132 / 0.235) | 0.483 | 0.569 |
| `1112_1336` EMA 250 | 0.165 (0.081 / 0.125 / 0.289) | | |
| `1112_1336` EMA 500 | 0.143 (0.080 / 0.146 / 0.204) | | |
| `1112_1336` EMA 750 | 0.155 (0.081 / 0.132 / 0.252) | | |
| `1112_1336` EMA 1k | 0.140 (0.072 / 0.128 / 0.220) | 0.481 (0.579 / 0.518 / 0.344) | 0.576 (0.671 / 0.590 / 0.466) |
| exact average of raw 500 / 750 / 1k | 0.156 (0.078 / 0.134 / 0.255) | 0.470 | 0.561 |
| `1112_1410` raw 1k (annealed tail) | **0.123** (0.081 / 0.109 / 0.178) | | |
| `1112_1410` EMA 1k (annealed tail) | **0.127** (0.080 / 0.113 / 0.188) | 0.495 (0.550 / 0.562 / 0.374) | 0.587 (0.616 / 0.623 / 0.521) |
| `A>S@1` (anchor target) | 0 | 0.672 | 0.729 |
| `S` (the students' own step 0) | 0.163 / 0.180 | 0.455 | 0.537 |

- **Length on the constant LR is not the lever.** The EMA's held-out MSE does not fall at each save
  (0.165 / 0.143 / 0.155 / 0.140) and the training loss is flat from 125 on. The 250 replicate reads
  0.165 against S-b's 0.157, so differences of 0.01 between saves are run noise. By the rule fixed
  in the config header this is not "length".
- **1000 pairs land in one place however they are spent.** 1000 iterations at one draw (0.140,
  `P>S@1` 0.481 / 0.576) and 250 iterations at four draws (0.147, 0.483 / 0.569) agree.
- **The anneal is what the 500-iteration students had.** Lowering the LR over the last 250
  iterations takes the MSE to 0.123 (raw) / 0.127 (EMA) from the same step-750 state that reaches
  0.140 at the constant LR. The cosine students sit at 0.109–0.118. On the constant LR the weights
  stay at a level the LR sets: S-b2 at 2e-4 reads 0.277, 5e-5 reads 0.14–0.16, the annealed runs
  0.11–0.13.
- **Averaging does not replace it.** The exact average of the raw 500 / 750 / 1k checkpoints reads
  0.156, no better than its members. Three points 250 iterations apart are a coarse stand-in for a
  long EMA.
- **0.12 does not give the layout.** `P>S@1` of the annealed step 0 is +0.04 / +0.05 over `S` and
  0.14–0.18 under `A>S@1`. Against the constant-LR 1k it is +0.011 / +0.014 in the mean and mixed
  per prompt (café and guitarist up, group down). The guitarist grid of the constant-LR 1k
  (`20261112-1408-.../1112_0833_125_ema_p2.jpg`, Claude's read on 320 px cells) shows `P>S@1` as the
  same large frontal figure as `S`; the annealed grid was not opened.
- Not read: the cosine between consecutive 250-step moves of `1112_1336`, which its header listed.

## Where this leaves step 0

- The anchor target is the right one (S-a).
- The regression's fit is set by the LR at the end of training, not by its length, its draws, the
  competing terms or an average of checkpoints. An annealed LR reaches a held-out anchor MSE of
  0.11–0.13; a constant 5e-5 stays at 0.14–0.18.
- At 0.12 the trained step 0 still draws the students' own layout. The regression has not been
  taken lower, and nothing here says how low it has to go.
- The one change that moved composition in the renders is the DM signal on step 0 (S-c.1), at a
  cost in diversity.
