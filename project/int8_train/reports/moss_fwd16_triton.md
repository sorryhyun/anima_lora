# MOSS: fp16 accumulation in the attention forward, in Triton (2026-11-03)

**Status: parked, no gate decided it.** On FA2 the estimates are about 1.2–1.5% of a step for PV in fp16
accumulation and about 2.5–3% for QKᵀ + PV + S. Neither arm shows a clear numerics cost on six images. PV
reads flat on prediction error. With S it reads 1–7% higher, but a move of that size at σ 0.9 also appears
in the opposite direction on PV, so this may be sample spread. The LoRA gradient cannot separate the arms.
The FA2 port was not built, mainly because the gain looked small for a new wheel.

Scripts: [`src/moss_fwd16_triton.py`](../src/moss_fwd16_triton.py),
[`src/moss_block_dtype_probe.py`](../src/moss_block_dtype_probe.py). Jobs
`20261103-105522-a3475c` (arms), `20261103-104334-65bec9` (dtypes). Raw numbers:
[`moss_fwd16_triton.json`](../assets/moss_fwd16_triton.json). Follows
[attn_bwd_fp16acc.md](attn_bwd_fp16acc.md), whose Anima call-level arms were not re-run, and
[fa2_fp16acc.md](fa2_fp16acc.md).

## Setup

`moss_fwd16_triton.py` replaces `flash_attn_func` in `library/moss/model.py` with the Triton pair of
`int8_attn.py`, then runs `grad_error_moss.py` on the production numerics: `gptq_fused_f32s`, six t2i
images, σ ∈ {0.1, 0.5, 0.9}, eager, checkpointed per block. Every arm is on that plan, and `bf16` runs
first because the bench requires it.

| arm | forward QKᵀ / PV | backward S | dP, dV, dQ, dK |
|---|---|---|---|
| `tri_nos` | bf16 / bf16, fp32 accumulation | fp32 accumulation | fp16acc, dV gain |
| `tri_nos_rep` | the same, repeated | | |
| `tri_pv` | bf16 / fp16acc | fp32 accumulation | fp16acc, dV gain |
| `tri_all` | fp16acc / fp16acc | fp16acc | fp16acc, dV gain |

`tri_nos` stands for what FA2 runs today. `tri_all` is `all_fwd16` of the earlier report with dV's P × 256,
which fixes block 0's tail there. Time and gradients are read against `tri_nos`, not against FA2.

## Speed

ms per call, `[1, 4600, 24, 128]`, random input, median of 30:

| arm | forward | backward | sum | gain |
|---|---|---|---|---|
| `tri_nos` | 2.882 | 7.771 | 10.653 | |
| `tri_pv` | 2.386 | 7.842 | 10.229 | 0.42 ms, 4.0% |
| `tri_all` | 2.281 | 7.445 | 9.726 | 0.93 ms, 8.7% |

Over 42 blocks, against a 1022 ms step: 18 ms (1.7%) and 39 ms (3.8%), as measured in Triton.

Two things lower that on FA2. These are estimates, not measurements:

- Triton's backward computes S in both of its kernels and FA2 once, so FA2 keeps about half of the
  backward's 0.33 ms.
- FA2 needs Q, K and V in fp16, three `to(half)` per forward that Triton's load does not pay. With them
  the gain is about 12–15 ms for PV (1.2–1.5%) and 25–30 ms for all three (2.5–3%).

## Prediction error

`‖v − v_fp32‖ / ‖v_fp32‖`, %, mean over images:

| σ | FA2 | `tri_nos` | `tri_pv` | `tri_all` |
|---|---|---|---|---|
| 0.1 | 3.647 | 3.626 | 3.642 | 3.697 |
| 0.5 | 1.623 | 1.606 | 1.607 | 1.627 |
| 0.9 | 1.215 | 1.204 | 1.131 | 1.283 |

- `tri_all` reads 2.0%, 1.3% and 6.6% above `tri_nos`, `tri_pv` +0.4%, +0.1% and −6.1%.
- Neither σ 0.9 move is explained, and they are about the same size in opposite directions. The bench
  keeps only means per σ, so no paired per-image test was possible, and a real cost for S is not
  established. If `tri_all`'s move is real, it takes back about 60% of the 0.135 points the fp32 kernel
  input gained there ([int8.md](../../../docs/moss/optimization/int8.md)).
- `tri_nos` sits 0.6–1.0% below FA2, so the two are not rounding-identical.

Mean loss stays within 2.2e-4 of `tri_nos` in every arm at every σ.

## Gradient

`‖S − S_fp32‖` of each arm over `tri_nos`, pooled per image, median [max] over the six images:

| σ | FA2 | `tri_pv` | `tri_all` |
|---|---|---|---|
| 0.1 | 1.099 [1.126] | 1.009 [1.246] | 1.105 [1.342] |
| 0.5 | 1.334 [5.377] | 0.861 [1.333] | 0.854 [2.294] |
| 0.9 | 1.040 [2.034] | 1.228 [1.823] | 0.999 [1.370] |

By depth, over all (image, σ) pairs, median [max]:

| over `tri_nos` | 0–6 | 7–13 | 14–20 | 21–27 | 28–34 | 35–41 |
|---|---|---|---|---|---|---|
| `tri_nos_rep` | 1.000 [1.00] | 1.000 [1.00] | 1.000 [1.00] | 1.000 [1.00] | 1.000 [1.00] | 1.000 [1.00] |
| FA2 | 1.285 [2.97] | 1.111 [4.96] | 1.052 [5.70] | 1.159 [6.28] | 1.039 [1.29] | 1.024 [1.19] |
| `tri_pv` | 0.987 [2.05] | 1.019 [1.91] | 0.953 [1.95] | 0.997 [1.56] | 0.950 [1.09] | 0.976 [1.17] |
| `tri_all` | 1.012 [2.58] | 1.056 [2.31] | 1.122 [2.39] | 1.000 [1.93] | 0.977 [1.53] | 1.024 [1.44] |

- Triton repeats exactly, so the repeat row is no floor. FA2 against `tri_nos`, the same precision
  apart from a 128×32 tile and FA2's non-determinism, is the nearer one, and it is wider than either
  fp16acc arm in most cells.
- The 1.05x gate proposed before this run cannot judge at this size, so it neither passed nor killed
  an arm. The 30-image run might separate them; it was not run.
- Blocks 28–41 are the readable ones: `tri_all` stays within 0.98–1.02x in the median. Its worst
  images, 1.53 and 1.44, sit above FA2's 1.29 and 1.19.

## Range

S peaked at 5373 on `tri_all`, 12x under fp16's 65504. That samples head 0 on one call in seven, and it
is the final sum, not the partial sums inside the MMA. Across 6048 Triton calls no gradient held a
NaN or an infinity.

## Where FA2's output goes

`moss_block_dtype_probe.py`, one block at real size (3072 / 8192 / 24 heads, 4608 tokens), random
weights, no LoRA, `full` mode, fp32 stream. Share of each int8 kernel input that a bf16 value could
hold:

| kernel input | eager | compiled |
|---|---|---|
| `y` (`block.qkv`) | 1.00 | 0.00002 |
| FA2's O (`proj.o`) | 1.00 | 1.00 |
| FFN `y` (`ffn_up_gate`) | 1.00 | 0.00002 |
| `_situ` output (`ffn_down`) | 1.00 | 0.00003 |

- In a compiled block Inductor elides `y.to(bf16)` + `x.float()`, so only O reaches the kernel in bf16.
  In eager every input holds bf16 digits, though its dtype is fp32. The bench above runs eager;
  whether the compiled step's prediction error differs was not measured.
- O goes to `proj.o` and to the LoRA's `down`; their sum meets the fp32 gate, then the stream. O's own
  rounding is small next to the int8 quantization it enters: 0.17% against about 1.3% in
  [gemm_microbench.md](gemm_microbench.md), on synthetic input.
- All backward kernel inputs are bf16 in both paths.

fp16 outside attention, same job, `[4608, 3072]`, rank 32, synthetic input with eight loud channels:

| | bf16 | fp16, fp32 accumulation | fp16acc |
|---|---|---|---|
| LoRA `down` + `up` forward, ms | 0.111 | 0.098 | 0.081 |
| error against fp32, `down` / `up` | 0.0036 / 0.0029 | 0.00045 / 0.00036 | 0.0026 / 0.00044 |

The two forward GEMMs would save about 8 ms over 252 linears, under 1%, and a plain fp16 node would
underflow the ~1e-7 cotangent, so they would need a custom backward. The bf16 backward GEMMs measured
here add up to 0.22 ms per linear and stay out of fp16's range. `comfy_kitchen.int8_linear` with
`out_dtype` fp16 gains nothing on `block.qkv`: 0.895 ms against bf16's 0.883.

## Not done

- **Any gate on these arms.** A paired per-image prediction test, the 30-image gradient, a compiled step.
- **The FA2 port.** [`src/fa2_fwd_fp16acc.patch`](../src/fa2_fwd_fp16acc.patch), on top of the
  three patches in `flash-attention-fp16`: `FLASH_ATTN_FP16ACC_FWD=pv|all`, off when unset. Its two
  fp16acc translation units compile; the wheel was stopped before it finished and has never run. Another wheel build would
  be a cheap moment to try PV on a real step.
- Anima on the same swap, cross-attention, i2i, a compiled step, any LoRA or render.
