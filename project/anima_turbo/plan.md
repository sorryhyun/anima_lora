# Plan: a 4-step recipe from the 2-step measurements

Written 2026-11-12. The 2-step line (superturbo, `dpdmd.rollout_grad`) is closed; its plan and
results are this file at commit `91d476e0`. From here the work is the 4-step student, on the
merged14 soup teacher (`output/ckpt/anima_merged14.safetensors`, shipped as one LoRA with the
soup, see the memory note on soup turbo). Arm 0 and the handoff probe have run (`reports/arm0.md`,
`reports/handoff.md`); the step-0 arms and arms 1–4 have not.

## What carries over from the 2-step line

Measured on the 2-step runs `1111_1005`, `1111_1115`, `1111_1740`, `1111_2056` (all batch 1):

- **The per-step gradient is mostly noise.** Gradient probe (`bench/turbo/probe_grad_noise.py`,
  `1111_1115` step 375): of the direction energy of a student step, 77 % is between DM query draws
  (σ_a, ε_dm) of one sample, 20 % between samples, 3 % is the mean that accumulates.
- **The weights random-walk.** Under a constant LR, |ΔW| grows as √t and consecutive 125-step
  moves are orthogonal (cos ≤ 0.02) from the first 125 steps on.
- **LR scales the walk, not its onset.** 5e-5 → 2e-5 shrank every move by 2.56× and moved
  nothing else (Adam scales drift and noise together).
- **Lowering the noise does not raise the level.** `dm_draws 4`, `grad_accum 4` and exact
  checkpoint averages all landed on the level of `1111_1115` at step 250.
- **Removing the walk after the fact undoes the late drift.** Exact ΔW averages of the
  250 / 375 / 500 checkpoints, and the EMA of `1111_2056` (window 0.2), took the late
  benches / vehicles / brown hair back out. EMA window 0.1 (`1111_1115`) did not.
- Reading: the objective sets the level; the batch-1 walk only subtracts from it.

## The 4-step loop is in the same regime (measured 2026-11-12)

`bench/turbo/dw_overlap.py`, cosine between the ΔW of two runs (weight_svd init, so ΔW starts
at 0). For two runs of one objective from one init it reads as the share of ΔW energy they
have in common.

| Pair | Steps | cos |
|---|---|---|
| `anima_turbo_m14_dpdmd_svd_500` vs `..._tafake` (4-step; same seed, init, data order; differ in `fake_transition` only, which left the losses identical) | 500 | **0.14** |
| `1111_1115` vs `1111_1740` (2-step; differ in `dm_draws`) | 125 / 250 / 375 / 500 | 0.19 / 0.17 / 0.16 / 0.17 |
| any other pair (different LR, or 2-step vs 4-step) | 500 | 0.09–0.11 |

- About 85 % of a 4-step student's ΔW is specific to its run, as in the 2-step line.
- The shared part is not concentrated: per module kind (adaln, self-attn, cross-attn, mlp) the
  4-step pair's cos is 0.11–0.19; per block group 0.10–0.21, highest in blocks 21–27.
- Short runs are no more deterministic: the 2-step pair already shares only 0.19 at step 125.
- Consequence for past reads: the 2026-11-10 `fake_transition` A/B (tafake sharper, plain
  drifting to brown hair and one centred sign; `vatdm.md`) compared two such draws. Its render
  differences may be mostly run noise.

No 4-step run has intermediate checkpoints or a resume bundle, so the 4-step |ΔW| trajectory
and a 4-step gradient probe are not measured (the probe refuses anything but the 2-step
superturbo config).

## Recipe: fixed by evidence, not A/B'd again

| Part | Setting | Evidence |
|---|---|---|
| Teacher, base | merged14, int8 GPTQ (`anima_merged14_int8_gptq`), fp32 stream (code default since `44d3023f`) | soup turbo note; fp32 stream settled |
| Diversity anchor | DP-DMD on: `div_weight 0.1`, `k_anchor 6 / 12`, `detach_after_first`, `dm_first_step` off | anchor-free MFD collapsed by step 250 (`reports/mfd.md`) |
| Fake | `fake_transition` on, `fake_warmup_steps 200`, 4 fake steps per student step | no harm at 500 steps (`vatdm.md`) |
| CDM | 1.0 | keeps the sky as sky in the 100-step arms (`reports/mfd.md`) |
| GAN | token head, `weight_gen 0.03` | runaway is an NFE-2 failure; flat at NFE 4 (`docs/anima/methods/turbo.md`) |
| LR | 5e-5 student and fake, constant | LR does not move the walk onset; the shape is not the lever |
| Walk removal | EMA `window 0.2`, the EMA is what ships | 2-step EMA 0.2 = checkpoint-average level |

Not to be A/B'd again on 4 steps: `dm_draws`, `grad_accum`, LR scale. On the 2-step line all
three were null against the level.

## A/B protocol

A single raw checkpoint against another reads mostly run noise. Every A/B here:

1. **250 iterations**, `save_every 125`, EMA on. About 40 min per arm (estimate: ~7 s per
   iteration plus fake warmup and compile; read the first run's real time).
2. **Compare EMAs at 250.** The exact average of the raw 125 and 250 checkpoints
   (`merge_loras.py --normalize off --adaln sum`) is the cross-check.
3. **One null twin, once.** The control with `seed 43` instead of 42. Its difference to the
   control is the noise floor for every later arm.
4. **Renders are the gate.** 4 steps, cfg 1, `bench/turbo/judge_hard.txt` (seeds 1–4 each) at
   768 × 1344 (W × H), beside the teacher (merged14, int8) at 28 steps, cfg 4
   (`output/tests/turbo4/teacher_m14/`). The ANIMA-sign and nozomi prompts are too easy to
   separate arms. The three prompts, in the spirit of MOSS's Space set
   (`bench/moss_ab/space/`): five named Blue Archive students with a position clause each and
   slapstick props; a two-girl café with Japanese menu text and an English sign; a solo
   guitarist jumping over a crowd, from below with a dutch angle.
5. **A distribution number beside the renders.** `bench/turbo/moment_gap.py`: S–T4 nW2² and the
   pooled spread ratio against the merged14 teacher at cfg 4. The script is built for the v1.1
   delta on the base teacher. It has to take the merged14 teacher and an arbitrary student LoRA
   first. The T4 set is rendered once and reused. Not run on arm 0. Its features are mean-pooled,
   so it does not read layout and does not gate the step-0 arms.
6. **The handoff probe beside the renders** (added 2026-11-12). `bench/turbo/probe_handoff.py`,
   about 7 min for two students: the `S` row's low-passed cosine to the teacher's image of the same
   noise, with the `T>S@1` row as the level a teacher-quality step 0 gives. It pairs each image with
   the teacher's, so it does not pay the seed-to-seed render floor.

An arm passes if its EMA renders beat the control's on both prompts by more than the control
and the twin differ, and its S–T4 nW2² sits below the control's by more than |control − twin|.
For the step-0 arms the number is the probe's `S` cosine against the control's (0.54 at EMA 125)
and the twin's (0.46 at EMA 250), with the renders as the gate.

## Gradient norm: the clip is a normaliser

`grad_clip 1.0` acts on almost every step. Pre-clip student norms (TB `train/grad_norm_student`,
5-step interval means) on the 2-step runs:

| Run | p5 | median | p95 | intervals above 1.0 |
|---|---|---|---|---|
| `1111_1005` | 1.74 | 3.20 | 10.1 | 100 % |
| `1111_1115` | 2.52 | 4.01 | 8.8 | 100 % |
| `1111_1740` | 2.42 | 3.74 | 7.4 | 100 % |
| `1111_2056` | 3.44 | 5.00 | 8.4 | 100 % |

The fake is the same (median 2.2–2.9, every interval above 1.0). The 4-step m14 runs predate
the norm logging.

- Adam ignores a constant gradient scale, so a higher clip does not make steps larger; LR does.
  What the clip sets is how steps weigh against each other. At 1.0 every step enters at unit
  norm; without it a step weighs by its norm.
- On 2 steps the large norms were the noise. The DM norm grew with σ_a (median 1.24 below 0.25,
  4.93 above 0.75, 22–30 near 0.96–0.99), nearly all through the rollout path, and norm-weighting
  took the DM mean share from 4.3 % (unit-weighted) to 1.6 %.
- 4 steps have no rollout path, so the σ_a profile of the norm may be flat there. Unmeasured.

Two knobs, neither built:

- **Adaptive clip** (`optim.grad_clip_mode = "adaptive"`): threshold = k × a running median of
  the pre-clip norm (EMA in log space, k ≈ 2), per net. Ordinary steps pass at their own norm,
  spikes are capped, and the threshold follows the norm's drift over the run (2.5 → 5 in
  `1111_2056`).
- **σ_a normalisation** (`dmd.dm_sigma_weight`): the DM loss times 1 / (running median norm of
  its σ_a bin, 8 bins), so the σ_a-driven part of the norm is removed before the clip. One
  backward, as now.

Logging (built 2026-11-12): `<run log dir>/grad_steps.jsonl`, one row per iteration with
`gn_student`, `gn_fake`, `gn_disc`, `sigma_a` (DM query level, first draw), `s_g`, `g`, `n`
(`metrics.StepGradLog`). Arm 0 gives the 4-step profile from it at no cost.

Decision, fixed here, read on arm 0's rows (both runs pooled): bin `gn_student` by `sigma_a`
into quarters. If the top quarter's median is within 1.5× the bottom quarter's, the arm is the
adaptive clip alone. If it is above 1.5×, the arm is adaptive clip plus σ_a normalisation.

## Where the layout is lost (measured 2026-11-12)

Arm 0's renders leave one gap above the seed-to-seed floor: both seeds draw a frontal standing
figure where the teacher draws the jump from below and the tumbling group. The handoff probe
(`reports/handoff.md`) swaps teacher and student at a step boundary, on one noise per image:

| The student takes over from | σ range it adds | Loss in low-passed cosine to the teacher's image (twin EMA 250 / control EMA 125) |
|---|---|---|
| step 3 | 0.5 → 0 | 0.05 / 0.02 |
| step 2 | 0.75 → 0.5 | 0.09 / 0.05 |
| step 1 | 0.9 → 0.75 | 0.16 / 0.13 |
| step 0 | 1 → 0.9 | 0.24 / 0.27 |

- Step 0 is the largest single loss. With the teacher drawing only σ 1 → 0.9, the student's steps
  1–3 keep the view from below and the figure's scale.
- Step 0 is trained by the anchor regression alone, at `div_weight 0.1`, inside a gradient that is
  clipped together with DM / CDM / GAN. Its `div_loss` fell from about 0.20 to 0.16 in 250
  iterations and was still falling.
- Arms 1, 2 and 4 change the DM term, which reaches steps 1–3 only. They cannot move step 0. Step 1
  is the second loss, so they keep a target, after step 0.

## Arms, in order

0. **Control + null twin** (~80 min, done). The recipe above, 4-step, `dynamic_schedule` on, DP-DMD
   point delta. They give the 250 level and the noise floor. A 4-step |ΔW| trajectory comes
   with them (125 / 250), and the per-iteration norm profile (`grad_steps.jsonl`).

**Step-0 arms**, before arm 1. None is built.

- **S-a. Ceiling of a perfectly fit step 0** (forward only, about 7 min). A new arm in the probe:
  step 0 is taken with the anchor target itself (`steps.teacher_anchor`, the teacher's mean
  velocity over its 6-of-12-step anchor), then the student draws steps 1–3. This is what the
  regression gives at zero loss. Decision, fixed here: if its cosine sits nearer `T>S@1` than `S`
  on both students and the renders keep the view from below, the target is right and S-b fits it.
  If not, the anchor is changed first, so that it ends at the student's first boundary
  (`k_anchor / teacher_anchor_steps`), and S-a is read again.
- **S-b. The anchor alone** (micro arm; estimate 10 min for 250 iterations, 12 teacher forwards and
  one student backward each). A switch that runs the loop with the anchor term only, with its own
  clip: no DM, CDM, GAN or fake. It is a regression on a fixed target, so its loss is a real one.
  Reads: `div_loss` on held-out noise (whether a rank-64 LoRA has a floor above the target), and
  the probe with this LoRA drawing step 0 and an arm 0 student drawing steps 1–3 (the probe takes
  one LoRA for all steps today). The LoRA is shared across σ, so this arm says what step 0 can
  reach, not what survives beside the DM terms.
- **S-c. The recipe with the stronger step 0**, against the control and the twin under the protocol
  above. S-b picks the form: a warm start from the S-b LoRA (`student_init_weights`, which excludes
  the `weight_svd` down init), or the anchor gradient clipped and weighted by itself inside the
  joint step.

Arms 1–4 follow on the step-0 result:

1. **MFD dtau 0.1 / max 4** in the DM slot. MFD's T2I result is a 4-step student (SANA, 3000
   steps), so 4 steps is the setting it was shown on. On 2 steps it sharpened stray content into
   objects as K grew; CDM held the sky.
2. **Static grid** (`dynamic_schedule` off). The dynamic grid adds a draw (N, interior σ) to an
   already noise-dominated step, and inference runs the static grid.
3. **Gradient norm** (section above): adaptive clip, with σ_a normalisation if arm 0's profile
   calls for it. Built after arm 0.
4. **Teacher-regularised fake + corrected critic** (VA-TDM idea 2, `vatdm.md`, λ_r 1). Not
   built. It costs a teacher forward per fake step.

Each arm changes one thing against its control: arm 0 for the step-0 arms, the step-0 result for
arms 1–4 if S-c passes. A passing arm becomes the new control, with no new twin unless the step
time or the loss terms change.

## Open

- Whether 250 iterations is where the 4-step level is reached. The shipped `anima_turbo_v3_longer`
  (warm start from the v1.1 delta, base teacher) kept improving to 2k+ and ended at |ΔW| 12.4.
  Arm 0's 125 / 250 renders and |ΔW| say whether 250 is already flat. If not, the A/B length goes
  up before arm 1.
- Seed averaging. With a shared share c ≈ 0.14, an exact ΔW average of k seeds raises it to
  kc / (1 + (k − 1)c): 0.25 at k = 2, 0.39 at k = 4. On 2 steps `avg2_500` (two runs) did not
  beat the 250 level, so this is cleanup, not a quality lever.
