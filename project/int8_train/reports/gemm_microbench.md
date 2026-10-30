# Phase 0 — GEMM microbench (2026-10-30)

On the RTX 5070 Ti, the ConvRot W8A8 int8 GEMM is 2.6x faster than bf16 on Anima's block
shapes, both for the forward and for the backward input-gradient GEMM. Raw int8 throughput
is 3.3x bf16. fp8 reaches only 1.9x, because GeForce cards run fp32 accumulation at half
rate and int32 accumulation is not throttled.

Script: [`src/gemm_bench.py`](../src/gemm_bench.py), run as
`make daemon-run ARGS="project/int8_train/src/gemm_bench.py 4096"` (job
`20261030-090111-0ab793`, 0.2 min).

## Setup

- RTX 5070 Ti (sm120, 16 GB), torch 2.14.1+cu132, comfy_kitchen 0.2.40.
- Real weights of `anima-base-v1.0`, block 10. q/k/v and cross-attn k/v concatenated as the
  runtime fuses them.
- M = 4096 image tokens (1024², B=1); the cross-attn kv projection runs on 512 context tokens.
- Forward: `Y = X Wᵀ`. Backward: `dX = dY W`. The int8 backward calls the same kernel
  (`torch.ops.comfy_kitchen.int8_linear`, `convrot=True`, group 256) on a second int8 copy
  of `Wᵀ` rotated along N, so the online rotation acts on dY.
- Activations: Gaussian with 8 outlier channels ×30. Gradients: Student-t (ν=3) × 1e-3.
  Both synthetic: the error column is a smoke check, not a verdict.
- Error: relative Frobenius error against an fp32 reference.

## Results

Per block, timings in ms (mean of 50 after 10 warmup):

| layer | M×K×N | bf16 fwd | int8 fwd | bf16 dX | int8 dX | err bf16 | err int8 fwd | err int8 dX |
|---|---|---|---|---|---|---|---|---|
| self qkv | 4096×2048×6144 | 1.012 | 0.338 | 1.036 | 0.418 | 0.17% | 1.28% | 1.40% |
| self out | 4096×2048×2048 | 0.360 | 0.145 | 0.341 | 0.145 | 0.17% | 1.27% | 1.25% |
| cross q | 4096×2048×2048 | 0.356 | 0.144 | 0.340 | 0.145 | 0.17% | 1.29% | 1.26% |
| cross kv | 512×1024×4096 | 0.048 | 0.072 | — | — | 0.17% | 1.13% | — |
| cross out | 4096×2048×2048 | 0.355 | 0.145 | 0.341 | 0.146 | 0.17% | 1.20% | 1.25% |
| mlp 1 | 4096×2048×8192 | 1.383 | 0.445 | 1.417 | 0.573 | 0.17% | 1.19% | 1.40% |
| mlp 2 | 4096×8192×2048 | 1.367 | 0.573 | 1.405 | 0.470 | 0.17% | 1.54% | 1.25% |
| **total** | | **4.881** | **1.861** | **4.879** | **1.897** | | | |

The cross-attn kv projection needs no dX, because the cached context does not require grad.

Raw GEMM on pre-quantized operands, no scale or rotation work:

| dtype | TFLOPS |
|---|---|
| bf16 | 98 |
| fp8 e4m3 `_scaled_mm` | 190 |
| int8 `_int_mm` | 322 |

## Reading

- The fused rotate-and-quantize costs about 20% over the raw int8 GEMM (1.86 vs 1.51 ms).
- At M=512 int8 is slower than bf16. The cross-attn kv projection stays bf16.
- The int8 error is about 7x bf16's rounding on these synthetic inputs. Whether that is
  small next to minibatch gradient noise is the phase 1 question.
- These numbers bound the GEMM share only. The step-level speedup depends on how much of a
  real step these GEMMs take (phase 1a).
