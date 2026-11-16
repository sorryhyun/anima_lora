# Anchor-only recipe: fit step 0 on the teacher's 28-step chord, leave steps 1–3 alone

Written 2026-11-16. The recipe that came out of `reports/v11_init.md`. The form below with the
28-step anchor has one 250-iteration run (`turbo_1116_1810`, `reports/runs.md`); the runs
behind the rest used the 12-step anchor.

## Motivation

- On merged14 / merged15 every student of the DM recipe draws one layout per prompt, and the loss
  sits in step 0 (`reports/handoff.md`, `reports/m15_1k.md`).
- The official turbo v1.1 delta already has steps 1–3 that keep the teacher's layout when they are
  handed the teacher's step 0 (`T>S@1` cos 0.932 on merged15). Its step 0 is the unfitted part.
- So the student starts from that delta and trains step 0 only, on a regression that needs no DM,
  CDM, GAN or fake network. From the same init the anchor alone is ahead of the joint recipe on
  every probe number, and steps 1–3 hold through 1000 iterations (`T>S@1` cos 0.945).

## What one iteration does

1. Draw one prompt and one noise ε.
2. Roll the frozen teacher (merged15, CFG 4) from ε along its own inference schedule to σ 0.75 and
   take the mean velocity of that stretch, `v_target = (ε − z) / (1 − 0.75)`. No gradient.
3. One student forward at σ 1 on ε. Loss: MSE between its velocity and `v_target`.

Steps 1–3 receive no loss and move only through the weights they share with step 0.

## The recipe

`_archive/configs/anima_turbo/turbo_1116_1558.toml` with the anchor on the teacher's 28-step schedule:

| Field | Value | Note |
|---|---|---|
| `network.student_init_weights` | the turbo v1.1 delta, Kaiming downs | r96 file, truncated to rank 64 by the loop |
| `dpdmd.anchor_only` | on | `dm_first_step` off; `cdm.weight`, `gan.weight_gen`, `fake_warmup_steps` at 0 |
| `dpdmd.k_anchor` / `teacher_anchor_steps` | **14 / 28** | the runs so far: 6 / 12 |
| `optim.student_lr`, `lr_schedule` | 5e-5, cosine | the fit follows the anneal (`reports/step0.md`); 2e-4 fits worse |
| `optim.grad_clip` | 1.0 | |
| `ema.window` | 0.2 | the EMA is what is read |
| `iterations` | 1000 | 250 leaves the fit shorter (anchor MSE 0.082 against 0.068) |
| `optim.grad_accum` | 1 | 4 ties the same compute spent on length (below) |
| batch, base | 1; merged15 int8 `stream`, fp32 stream | |

**Why 14 of 28.** The teacher is rendered, and the student is judged, at 28 steps, shift 3, CFG 4.
An anchor on the first 14 of those 28 steps ends on the teacher's own state at σ 0.75. The 12-step
anchor ends on a coarser rollout's state, which is a different image of the same noise and draws
seeds more alike. With the target itself as step 0 (`reports/v11_init.md`, no training):

| anchor | cos to the teacher's image | sameness (teacher 0.315) |
|---|---|---|
| 6 of 12 | 0.818 | 0.346 |
| 14 of 28 | 0.967 | 0.249 |
| 7 of 28 (the teacher's own state at σ 0.9) | 0.945 | 0.313 |

## Gradient accumulation

`turbo_1116_1730`: `1558` with `optim.grad_accum = 4` at 1000 iterations (250 updates on `1558`'s
cosine, each the mean of four per-iteration clipped gradients), on the 6 of 12 anchor. EMA, group /
café / guitarist in brackets (`reports/v11_init.md`):

| | updates × pairs each | anchor MSE | `S` cos | sameness `S` | `T>S@1` cos |
|---|---|---|---|---|---|
| `1558` | 250 × 1 | 0.082 (0.060 / 0.070 / 0.115) | 0.766 | 0.350 | 0.946 |
| `1653` | 1000 × 1 | 0.068 (0.057 / 0.065 / 0.083) | 0.775 | 0.365 | 0.945 |
| `1730` | 250 × 4 | 0.068 (0.055 / 0.065 / 0.084) | 0.770 | 0.373 | 0.944 |

- **The draw noise does limit the fit at equal updates:** four pairs an update take the anchor MSE
  from 0.082 to 0.068.
- **At equal compute it is a tie with length.** 1000 pairs give the same anchor MSE as four per
  update or one per update; `S` cos and sameness are slightly behind `1653`. The fit follows the
  number of pairs seen, so accumulation buys nothing while a pair costs a teacher rollout.
- The renders break in the same places as `1558` and `1653` (Claude's read); the café aprons keep
  the teacher's blue where `1653` turned two seeds green.
- So the recipe stays at `grad_accum 1`. A batch becomes worth it with a target cache, where a pair
  costs one student forward and backward.

## Cost

Measured on the 6 of 12 anchor (RTX 5070 Ti, int8 base, compiled): 1.7 s per iteration
(13 forwards: 12 teacher, 1 student with its backward). The anchor is two teacher forwards per Euler
step, so 14 of 28 is 29 forwards. The 14 of 28 figures are estimates from that count, not measured.

| | 6 of 12 (measured) | 14 of 28 (estimate) |
|---|---|---|
| per iteration | 1.7 s | about 3.6 s |
| 250 iterations | 8 min | about 16 min |
| 1000 iterations | 28.5 min | about 60 min |
| 250 updates at `grad_accum 4` (1000 iterations) | 28.3 min | about 60 min |
| 10k iterations | about 4.7 h | about 10 h |

The probe and the judge renders after a run add about 3 min. The target depends only on
(prompt, ε) and the frozen teacher, so a target cache would leave one student forward and backward
per iteration (`plan.md`, candidate 3): `dpdmd.anchor_cache`, built by
`python -m scripts.distill_turbo.anchor_cache --config <run>.toml [--draws K]` (smoke-tested; the
full cache is not built). `turbo_1116_1810` measured 3.4 s per iteration on the 14 of 28 anchor.

## What is open

- **The finish.** At anchor MSE 0.068 on the 6 of 12 target the student draws the teacher's layouts
  with broken objects (merged figures, a guitar cut from its neck), and 1000 iterations did not fix
  what 250 left. The target itself as step 0 finishes cleanly, so the question is how deep the fit
  has to go, and whether the 28-step target changes it.
- **Reading a run.** `plan.md`, "A/B protocol": the handoff probe (anchor MSE, `S` and `T>S@1` cos,
  sameness) beside the judge renders. A run on the 14 of 28 target is probed with
  `--anchor_k 14 --anchor_steps 28`, or its anchor MSE is measured against the wrong target.
