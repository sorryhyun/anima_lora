# Anima's bf16 stream and the kernel's input dtype (2026-10-31)

**Verdict: Anima has MOSS's bf16 stream error, at about half the size. An fp32 stream cuts
prediction error 1.5–2.1x around bf16 GEMMs and 21–36% around the int8 base, which then sits
below bf16 at every σ. The LoRA gradient improves in blocks 14–27 around bf16 GEMMs and is
mixed around int8, with an untraced tail in blocks 0–13. Anima keeps its bf16 stream: the
arm's step cost and any LoRA trained on it have not been measured.**

Script: [`src/anima_stream_error.py`](../src/anima_stream_error.py), job
`20261031-201403-548105`, 11.2 min. Raw numbers: `assets/anima_stream_error.json` (local,
`project/**/assets/*.json` is gitignored).

## Setup

Six cached images of 4116–4158 tokens, σ ∈ {0.1, 0.5, 0.9}, one fixed ε each, eager, gradient
checkpointing on. The reference is an fp32 forward and backward (SDPA, TF32 off). int8 arms are
mode `full` with the catalog's GPTQ forward copies. No LoRA is applied.

| arm | linears | kernel input | stream |
|---|---|---|---|
| `bf16` | bf16 | – | bf16 |
| `bf16_rep` | bf16 | – | bf16, the same pass again (floor) |
| `bf16_s32` | bf16 | – | fp32 |
| `int8` | int8 | fp32 (as trained today) | bf16 |
| `int8_in16` | int8 | bf16 (before the fp32 cast) | bf16 |
| `int8_s32` | int8 | fp32 | fp32 |

The block input `x` is bf16 and training runs under bf16 autocast, confirmed on the trainer path
(job `20261031-195951-1375ec`). So each block's three `x + gate·result` adds round to bf16, over
28 blocks. The linears are not as uniform as MOSS's: `LayerNorm` returns fp32 under autocast, so
`qkv_proj`, `q_proj` and `layer1` get fp32 computed from the rounded stream, while `output_proj`
and `layer2` get bf16.

**The fp32 arms are narrower than MOSS's.** Anima has no `--stream`. The arm casts `x` to fp32 at
each block and the `adaln_up_*` outputs to fp32, so the adds and the gate products run in fp32.
Autocast stays on, so those linears still multiply in bf16, and nothing else is cast. MOSS's arm
casts every parameter outside the block linears, with no autocast. Read the gains here as a
floor.

## Prediction

`‖v − v_fp32‖ / ‖v_fp32‖`, %, mean over images:

| σ | bf16 | `bf16_s32` | `int8_in16` | `int8` | `int8_s32` |
|---|---|---|---|---|---|
| 0.1 | 1.719 | 1.118 | 2.277 | 2.093 | **1.646** |
| 0.5 | 0.837 | 0.421 | 0.991 | 0.933 | **0.600** |
| 0.9 | 0.922 | 0.434 | 1.069 | 1.000 | **0.645** |

- The fp32 stream wins on 18 of 18 (image, σ) pairs in both comparisons: 1.5–2.1x around bf16
  GEMMs, 21–36% around int8.
- `int8_s32` beats `bf16` on 18 of 18. `int8` never does.
- The fp32 kernel input wins on 18 of 18, by 6–8%. MOSS measured 5–10%.
- `bf16_rep` matches `bf16` to all digits, so the eager forward carries no run-to-run noise here.

Against MOSS at σ = 0.5, bf16 reads 0.84 against 1.93 and `bf16_s32` 0.42 against 0.81. Why
Anima's error is smaller was not traced.

Mean loss:

| σ | fp32 | bf16 | `bf16_s32` | `int8` | `int8_s32` |
|---|---|---|---|---|---|
| 0.1 | 0.21168 | 0.21197 | 0.21181 | 0.21216 | 0.21199 |
| 0.5 | 0.06645 | 0.06651 | 0.06647 | 0.06655 | 0.06648 |
| 0.9 | 0.09757 | 0.09761 | 0.09755 | 0.09765 | 0.09739 |

## Gradient

Each image's distance from the fp32 gradient over bf16's, on the sketch `S = (dY Ω)ᵀ X` of
`grad_error.py`, pooled over the targets of seven blocks. Median over images [p90]. Six images
make the p90 close to the maximum. The pool is not the median over modules of `grad_error.md`,
so these figures do not compare with its gate. `kv_proj` and adaLN have no sketch: nothing
upstream of them requires grad here.

| σ | arm | 0–6 | 7–13 | 14–20 | 21–27 |
|---|---|---|---|---|---|
| 0.1 | `bf16_s32` | 1.01 [2.06] | 1.00 [1.57] | 0.75 [0.84] | 0.71 [0.76] |
| 0.1 | `int8_in16` | 0.91 [1.05] | 0.97 [1.07] | 1.40 [1.49] | 1.20 [1.26] |
| 0.1 | `int8` | 1.60 [3.00] | 1.88 [2.84] | 1.42 [1.47] | 1.18 [1.21] |
| 0.1 | `int8_s32` | 0.98 [1.37] | 1.08 [1.28] | 1.31 [1.38] | 0.99 [1.12] |
| 0.5 | `bf16_s32` | 1.08 [1.45] | 1.02 [2.26] | 0.74 [0.75] | 0.64 [0.69] |
| 0.5 | `int8_in16` | 1.05 [1.08] | 1.16 [1.30] | 1.45 [1.49] | 1.24 [1.34] |
| 0.5 | `int8` | 0.91 [2.35] | 1.10 [1.93] | 1.39 [1.44] | 1.17 [1.25] |
| 0.5 | `int8_s32` | 1.33 [1.77] | 1.73 [2.05] | 1.24 [1.29] | 0.95 [1.07] |
| 0.9 | `bf16_s32` | 0.78 [0.90] | 0.81 [0.85] | 0.73 [0.81] | 0.76 [0.83] |
| 0.9 | `int8_in16` | 1.11 [1.38] | 1.13 [1.37] | 1.53 [1.74] | 1.50 [1.57] |
| 0.9 | `int8` | 0.91 [1.42] | 1.21 [1.49] | 1.42 [1.65] | 1.25 [1.40] |
| 0.9 | `int8_s32` | 0.84 [1.22] | 0.92 [1.08] | 1.40 [1.47] | 1.29 [1.33] |

`bf16_rep` reads 1.00 [1.00–1.01] everywhere.

- **Blocks 14–27 around bf16 GEMMs:** the fp32 stream moves every σ closer to fp32, at
  0.64–0.76x. MOSS saw the same split: late blocks gain, early ones barely move.
- **Blocks 14–27 around int8:** `int8_s32` improves on `int8` at σ 0.1 and 0.5, and is level or
  slightly worse at 0.9.
- **Blocks 0–13 are a tail, not a shift.** The large ratios sit on single images. At σ = 0.1,
  blocks 0–6, `int8` reads 1.23, 1.80, 1.40, 3.09, 3.00, 0.50.

The tail does not belong to one arm:

| case | arm | blocks 0–6, per image |
|---|---|---|
| σ 0.1 | `bf16_s32` | 1.04, 0.89, 0.98, **2.88**, 2.06, 0.58 |
| σ 0.1 | `int8` | 1.23, 1.80, 1.40, **3.09**, 3.00, 0.50 |
| σ 0.1 | `int8_in16` | 1.11, 0.69, 1.05, 0.99, 0.83, 0.48 |
| σ 0.1 | `int8_s32` | 1.04, 1.37, 1.65, 0.92, 0.76, 0.45 |
| σ 0.5 | `bf16_s32` | 1.34, 0.83, **3.13**, 0.51, 0.60, 1.45 |
| σ 0.5 | `int8_s32` | 1.18, **2.61**, 1.43, 1.77, 0.65, 1.23 |

`bf16_s32` has no int8 at all and spikes on the same images as `int8` at σ 0.1. So the fp32 kernel
input is not shown to cause the rise in `int8`'s median, even though `int8_in16` stays flat on
these six images at σ 0.1. Which image spikes changes with the arm and with σ, as with the
single-token gradient outliers on MOSS ([int8.md](../../../docs/moss/optimization/int8.md#single-token-gradient-outliers)).
Whether this is the same mechanism was not traced.

## Not done

- **Cost.** No step time or peak for an fp32 stream on Anima. The saved set would grow with the
  stream, so the budget sweep in `docs/anima/optimizations/int8.md` does not carry over.
- **A full arm**: every non-block parameter in fp32, with those GEMMs out of autocast.
- **The tail**: the hot token's share of `‖dY‖²` per arm, as `src/grad_spike.py` did on MOSS, and
  30 images, about an hour.
- **Anything trained.** No LoRA has run on an fp32 stream on Anima, so the effect on samples is
  unknown. MOSS found no consistent quality gain, and closer to fp32 is not a target in itself.
- **The backward's `dY`** stays bf16 in every int8 arm and goes through the kernel's rounding
  (comfy-kitchen#289).
