# MOSS-Image real-gradient error (2026-10-30)

**Verdict: passes.** Median over the 252 block linears, the int8 `full` LoRA gradient
sits at 0.53x, 1.17x and 0.98x bf16's distance from the fp32 gradient at σ = 0.1, 0.5
and 0.9 (gate: at most 2x). Anima measured 1.28–1.50x. The ratio is lower on MOSS
because MOSS's bf16 gradient is already far from fp32, not because int8 adds less: the
prediction error grows 1.24–1.46x, as on Anima.

Script: [`src/grad_error_moss.py`](../src/grad_error_moss.py), job
`20261030-160517-b900bd`, 11.5 min, peak 14.0 GB. Raw numbers:
[`moss_grad_error.json`](../assets/moss_grad_error.json).

## Setup

As [grad_error.md](grad_error.md), with these differences:

- MOSS-Image Base, 252 linears (`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`), eager,
  torch non-reentrant checkpointing per block, loss as `training_loss`.
- Six images from `channel_(caststation)_layout` (4056–4154 image + 184–383 text
  tokens), the full caption, σ ∈ {0.1, 0.5, 0.9}.
- fp32 reference: every linear runs an fp32 GEMM on its bf16 weight, norms and biases
  in fp32, SDPA, TF32 off.
- The int8 copies are quantized on every call (round to nearest), since bf16 plus both
  int8 copies does not fit.
- Two-sided sketch `S = (dY Ω)ᵀ (X Ψ)`, q = 64 by 256. Absolute values are not
  comparable to Anima's one-sided sketch, but ratios are.

## Results

Median over modules. Distance from the fp32 gradient / minibatch noise, and int8 over
bf16:

| σ | bf16 | bf16 repeat vs bf16 | int8 `full` | `full` / bf16 | `w8a16` / bf16 | `w16a8` / bf16 |
|---|---|---|---|---|---|---|
| 0.1 | 1.65 | 0.02 | 0.89 | 0.53 | 0.51 | 0.58 |
| 0.5 | 0.77 | 0.01 | 0.86 | 1.17 | 1.02 | 1.21 |
| 0.9 | 0.85 | 0.01 | 0.79 | 0.98 | 1.31 | 1.01 |

`full` / bf16 by depth (median per group of seven blocks), with bf16's own distance
from fp32 / noise in parentheses:

| σ | 0–6 | 7–13 | 14–20 | 21–27 | 28–34 | 35–41 |
|---|---|---|---|---|---|---|
| 0.1 | 0.51 (1.88) | 0.52 (1.75) | 0.50 (1.74) | 0.35 (2.45) | 0.86 (0.57) | 3.06 (0.22) |
| 0.5 | 1.06 (0.93) | 1.17 (0.88) | 1.16 (0.81) | 1.07 (0.71) | 1.49 (0.10) | 2.35 (0.09) |
| 0.9 | 0.93 (0.94) | 0.92 (0.97) | 0.94 (0.90) | 1.07 (0.48) | 1.31 (0.16) | 1.32 (0.10) |

Prediction error against fp32, `‖v − v_fp32‖ / ‖v_fp32‖`, mean over images:

| σ | bf16 | `full` | `w8a16` | `w16a8` | `full` / residual | loss bf16 → `full` |
|---|---|---|---|---|---|---|
| 0.1 | 4.81% | 6.99% | 6.21% | 5.77% | 10.5% | 0.5635 → 0.5666 |
| 0.5 | 1.93% | 2.82% | 2.48% | 2.36% | 7.6% | 0.2253 → 0.2259 |
| 0.9 | 2.08% | 2.58% | 2.44% | 2.34% | 4.9% | 0.3819 → 0.3824 |

`fwd` matches `full` in every row, to ±0.01.

## Reading

- **The int8 dX GEMM is free on MOSS too.** `fwd` ≈ `full` everywhere, and the loss
  moves 0.1–0.5%. The 13% step-10 loss gap in the swap-0 training run
  ([moss_scoping.md](moss_scoping.md)) is not a forward error; the LoRA had already
  diverged by step 10.
- **MOSS's bf16 gradient is mostly rounding in blocks 0–27** (0.7–2.5 of noise from
  fp32; Anima's blocks 0–6 were 0.8–1.1, the rest under 0.7). The likely cause is the
  bf16 residual stream: MOSS trains without autocast, so `x` and the modulation are
  bf16 through 42 blocks. int8 adds a smaller error than that, so the ratio stays near 1.
- **At σ = 0.1 int8 is closer to fp32 than bf16 is**, in all four int8 arms. Each
  differs from bf16 only in how the forward GEMM is computed: int32 accumulation, or fp32
  in the split arms. That points at the bf16 GEMM's own reduction, since
  `allow_bf16_reduced_precision_reduction` is on by default. Not tested.
- **The late blocks (28–41) are where int8 costs the most**: 1.3–3.1x bf16, though
  in absolute terms 0.14–0.51 of noise, where bf16 is very accurate (0.09–0.22). There,
  `w8a16` carries the ratio and `w16a8` stays at 1.1–1.2, so the excess is the weight
  term.
- **Weight against activation, at the prediction:** subtracting bf16 in quadrature,
  the `w8a16` and `w16a8` excesses add up to `full`'s within 10%, and the weight term
  is 57–60% of the excess energy at every σ (Anima: 35–50%). GPTQ removed 81% of the
  weight term on Anima, so on MOSS it should cut int8's excess prediction error by
  about a quarter, and the late-block gradient excess by most of it. SmoothQuant would
  address the remaining ~40%.

## bf16 precision arms

Same script, images, ε and sketches, `--modes bf16,bf16_rep,bf16_f32s,bf16_noreduce`
(job `20261030-162202-f75fd1`; raw numbers:
[`moss_grad_error_bf16arms.json`](../assets/moss_grad_error_bf16arms.json)). Both test
an explanation from the reading above.

- `bf16_f32s`: every parameter except the block linears' weights in fp32, so the residual
  stream, modulation, gates and norms are fp32 around the bf16 block GEMMs (and FA2).
- `bf16_noreduce`: `allow_bf16_reduced_precision_reduction = False` for the pass.

Prediction error against fp32 and loss (fp32 loss in parentheses):

| σ | bf16 | `bf16_f32s` | `bf16_noreduce` | loss bf16 / `f32s` (fp32) |
|---|---|---|---|---|
| 0.1 | 4.81% | **1.64%** | 4.80% | 0.5635 / 0.5612 (0.5609) |
| 0.5 | 1.93% | **0.81%** | 1.94% | 0.2253 / 0.2249 (0.2249) |
| 0.9 | 2.08% | **0.59%** | 2.12% | 0.3819 / 0.3814 (0.3814) |

Gradient, median over modules, as a multiple of bf16's distance from fp32, overall and
for blocks 0–27 and 28–41 (group medians):

| σ | `f32s` | `f32s` 0–27 | `f32s` 28–41 | `noreduce` | `noreduce` 0–27 | `noreduce` 28–41 |
|---|---|---|---|---|---|---|
| 0.1 | 0.57 | 0.54–0.62 | 0.49–0.89 | 0.39 | 0.24–0.41 | 0.38–0.96 |
| 0.5 | 0.79 | 0.75–1.00 | 0.62–0.65 | 0.99 | 0.84–1.10 | 0.98–1.00 |
| 0.9 | 0.96 | 1.00–1.37 | 0.36–0.49 | 1.12 | 1.07–1.60 | 1.02–1.03 |

Reading:

- **The bf16 residual stream accounts for most of bf16's prediction error.** An fp32
  stream around the same bf16 GEMMs cuts the prediction error 2.4–3.5x, and the loss
  matches fp32 to the fourth decimal.
- **The early-block gradient error does not come from the stream.** With an fp32
  stream, blocks 0–27 stay at 0.75–1.37x bf16 at σ 0.5 and 0.9. Those gradients pass
  back through every later block's bf16 dX GEMMs and the FA2 backward, which this arm
  leaves as they were. Late blocks improve to 0.36–0.65x.
- **Reduced-precision reduction is not a fix.** Turning it off cuts the σ = 0.1
  gradient error to 0.39x, but it does nothing at 0.5 and is worse at 0.9 (blocks
  0–20 at 1.4–1.6x). The prediction does not move. cuBLAS picks different kernels, and none of
  them is accurate at every shape. The σ = 0.1 anomaly in the int8 arms has the same
  kind of cause: any other forward GEMM beats the default bf16 kernel there.
- Not measured: int8 with an fp32 stream. In quadrature, int8's own prediction error
  (~2.1% at σ = 0.5) on top of `f32s` (0.81%) would land near today's bf16 (1.93%).
