# Phase 1b — real-gradient error (2026-10-30)

**Verdict: passed on a revised gate, with a larger error than bf16.** The gate is now
"distance from the fp32 gradient at most 2x bf16's", and int8 `full` is at 1.3–1.5x.
The int8 gradient error is real and larger than bf16's at every σ and depth group
except blocks 0–6 at σ=0.9. Phase 3's A/B, with samples generated on the bf16 base,
decides whether it matters for LoRA quality.

**Against fp32, int8 is about 1.4x as far off as bf16, and the gate as written is one
bf16 itself fails** (see [fp32 reference](#fp32-reference) below). Measured from the
fp32 gradient, bf16 sits at 0.15–0.20 of minibatch noise and int8 `full` at 0.18–0.27.
The int8 error adds in quadrature to bf16's own rounding error and is about the same
size. The early-block error predates int8: bf16 is already 0.8–1.1 of noise away from
fp32 in blocks 0–6.

## First pass, against bf16

**The gate fails.** Median over modules, the int8 `full` gradient error is 0.18–0.22 of
minibatch noise (gate: < 0.1) at every σ. The error comes from the int8 **forward**:
mode `fwd` (bf16 dX) is as far off as `full`. Quantizing the backward adds nothing
measurable. The error is concentrated in the early blocks. Blocks 14–27 sit at the
gate (0.08–0.12), and blocks 0–6 are at 0.9–1.4: there the int8 gradient differs from
the bf16 one by more than one image's gradient differs from another's.

Script: [`src/grad_error.py`](../src/grad_error.py), run as
`make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/grad_error.py"`
(job `20261030-095523-c2007d`, 6 min). Raw numbers: [`grad_error.json`](grad_error.json).

## Setup

- `anima-base-v1.0`, base DiT only, no LoRA applied, eager, torch gradient
  checkpointing (non-reentrant). 168 linears converted (6 per block × 28).
- Six cached images from `post_image_dataset/lora` (4056–4200 tokens), B=1, one fixed
  ε per image, σ ∈ {0.1, 0.5, 0.9}, plain flow-matching MSE.
- Per LoRA-target linear, `G = dYᵀX` sketched as `S = (dY Ω)ᵀ X`, with Ω Gaussian and
  q = 64. Every LoRA gradient (up `G Aᵀ`, down `Bᵀ G`) is a linear map of `G`.
- Four passes per image: bf16, bf16 again (the run-to-run floor), `fwd`, `full`.
  `err = RMS_i ‖S_mode − S_bf16‖`, `noise = RMS_i ‖S_bf16,i − S_bf16,i−1‖`.
- 170 of 314 LoRA targets carry a gradient here. The adaLN linears take only the
  timestep embedding, which has no grad path from the noisy input once the base is
  frozen, so they do not appear.

## Results

Median over modules, error / noise:

| σ | bf16 repeat | `fwd` | `full` | `full` rel. to ‖S‖ | loss bf16 → `full` |
|---|---|---|---|---|---|
| 0.1 | 0.008 | 0.193 | 0.190 | 0.25 | 0.2120 → 0.2123 |
| 0.5 | 0.007 | 0.179 | 0.181 | 0.27 | 0.0665 → 0.0666 |
| 0.9 | 0.007 | 0.216 | 0.217 | 0.30 | 0.0976 → 0.0978 |

`full` error / noise by depth (median per group of seven blocks):

| σ | blocks 0–6 | 7–13 | 14–20 | 21–27 |
|---|---|---|---|---|
| 0.1 | 1.38 | 0.82 | 0.11 | 0.11 |
| 0.5 | 0.89 | 0.56 | 0.10 | 0.08 |
| 0.9 | 1.02 | 0.81 | 0.12 | 0.09 |

Worst modules: `blocks.4.cross_attn.output_proj` (2.1 at σ=0.1), `blocks.3.cross_attn.q_proj`
(2.0), and `cross_attn.q_proj` in blocks 0–7 at every σ.

## Reading

- The measurement is sound. The bf16 repeat is below 1% of noise, so the int8 gap is not
  nondeterminism.
- The loss moves 0.1–0.2%, while the gradient moves 25–30%. The loss gradient is driven
  by the residual `pred − target`, which is small next to `pred`, so a small relative
  error in `pred` becomes a large relative error in `dpred`. Not yet measured directly.
- The depth pattern means an early block's gradient goes back through every later
  block's Jacobian, and each Jacobian is built from int8-perturbed activations (softmax,
  GELU, norms). Late blocks see few of them.
- Mode `fwd` ≈ `full` rules out the dX GEMM as the problem. The int8 dX is as good as
  bf16 once the forward is int8.
- Not measured: whether the forward error comes from the weight or the activation
  quantization (a W8A16 arm).

## fp32 reference

Script: the same [`src/grad_error.py`](../src/grad_error.py) with an fp32 phase first
(params cast to fp32, SDPA, no autocast, TF32 off; cast back before any int8 copy),
job `20261030-101108-72571d`, 11 min. Same images, ε and Ω as the first pass; its
bf16-referenced numbers reproduce the first pass to ±0.003. Raw numbers:
[`grad_error_fp32.json`](grad_error_fp32.json). Noise is now measured between fp32
gradients.

Median over modules, distance from the fp32 gradient / noise:

| σ | bf16 | int8 `fwd` | int8 `full` | int8 / bf16 (per module, median) | `full` vs bf16 (old gate) |
|---|---|---|---|---|---|
| 0.1 | 0.198 | 0.272 | 0.269 | 1.41 | 0.193 |
| 0.5 | 0.155 | 0.174 | 0.176 | 1.28 | 0.181 |
| 0.9 | 0.147 | 0.226 | 0.224 | 1.50 | 0.214 |

By depth, distance from fp32 / noise:

| σ | | blocks 0–6 | 7–13 | 14–20 | 21–27 |
|---|---|---|---|---|---|
| 0.1 | bf16 | 0.89 | 0.67 | 0.10 | 0.07 |
| | int8 `full` | 1.16 | 0.82 | 0.14 | 0.13 |
| 0.5 | bf16 | 0.77 | 0.46 | 0.06 | 0.06 |
| | int8 `full` | 0.88 | 0.58 | 0.10 | 0.09 |
| 0.9 | bf16 | 1.09 | 0.51 | 0.07 | 0.06 |
| | int8 `full` | 1.02 | 0.94 | 0.12 | 0.09 |

Prediction error against fp32, mean over images, relative to ‖pred‖ and to the residual
‖pred − target‖:

| σ | bf16 / ‖pred‖ | bf16 / residual | int8 / ‖pred‖ | int8 / residual |
|---|---|---|---|---|
| 0.1 | 1.7% | 4.6% | 2.4% | 6.4% |
| 0.5 | 0.8% | 3.9% | 1.0% | 4.9% |
| 0.9 | 0.9% | 3.5% | 1.1% | 4.2% |

Reading:

- The residual does amplify, 2.7–4.7x from ‖pred‖ to residual. It does so for bf16 and
  int8 alike.
- The gradient moves more than the residual does (15–20% of noise from a 4% residual
  error), because the backward Jacobians are built from rounded activations too. This
  is also true of bf16.
- `0.198² + 0.193² ≈ 0.269²`: the int8 error is roughly independent of bf16's rounding
  and about as large. Adding int8 is like adding a second bf16-sized rounding error.
- Early-block gradients are mostly rounding even in bf16 (0.8–1.1 of noise in blocks
  0–6). int8 makes that 1.0–1.2. bf16 LoRA training works with it.
- The original gate (int8 within 0.1 of noise from bf16) asks for more than bf16
  delivers against fp32 (0.15–0.20), so it cannot separate a usable base from an
  unusable one. A gate relative to bf16's own error would; int8 sits at 1.3–1.5x.
