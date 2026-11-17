# QKᵀ off fp32 accumulation: three ways tried, none kept (2026-11-17)

**Verdict: QKᵀ stays on fp32 accumulation for training.** The error of fp16-accumulate QKᵀ
(`FLASH_ATTN_FP16ACC_FWD=all`) is the fp16 accumulator rounding a growing partial sum, not the
logsumexp mismatch. Handing the partial sum to fp32 more often removes it, and so do two int8
planes per operand, but each costs more time than fp16 QKᵀ saves. Removing the key mean changes
nothing. `all` stays an opt-in for generation, which has no backward.

All errors are against fp32 SDPA on the 24 self-attention calls of the Anima capture
(`output/int8_train/attn_capture.pt`), as multiples of the shipped arm, median [max]. Triton
arms are the kernels of [`src/int8_attn.py`](../src/int8_attn.py); `port` is the shipped FA2's
stand-in there (fp32-accumulate QKᵀ, fp16acc elsewhere), and Triton's `all` reproduces FA2's
(dK 1.53 [2.09] against 1.53 [2.12], dV 1.72 [2.79] on both).

## Where QKᵀ sits

On the shipped build, MOSS self shape `[1, 4600, 24, 128]`: of the five backward GEMMs only QKᵀ
still accumulates in fp32, about 1.5 ms of the 5.81 ms kernel
([fa2_fp16acc.md](fa2_fp16acc.md#where-the-2x-went-2026-11-04)). FA2 `pv` → `all`, measured here
([`attn_s_split_speed.py`](../src/attn_s_split_speed.py)):

| shape | `pv` fwd + bwd | `all` fwd + bwd | gain |
|---|---|---|---|
| Anima self `[1, 4131, 16, 128]` | 1.278 + 2.809 ms | 1.082 + 2.565 ms | 0.44 ms |
| MOSS self, random input | 2.316 + 5.279 ms | 1.942 + 4.890 ms | 0.76 ms |

That is about 12 ms of an Anima step and 32 ms of a MOSS step, 3% each. The first build's
table in [fa2_fwd_fp16acc.md](fa2_fwd_fp16acc.md) showed no backward gain on MOSS; this one does.

`all` already shares the logsumexp: the forward's fp16 S makes it and the backward recomputes
the same S. Against stock FA2 it reads O 1.10, dQ 1.01, dK 1.24, dV 1.32; against the shipped
build, whose fp16acc backward sits under stock, dK 1.53 and dV 1.72. Block 0, where the
P-weighted logit is 0.15, pays nothing; blocks 9, 18 and 27 (logits 4.9, 7.0, 15.0) pay
1.3–2.1x.

## Key mean removed

[`fa2_kcenter_bench.py`](../src/fa2_kcenter_bench.py), on FA2's own kernels through
`flash_attn_f16in_func`. Softmax ignores a constant added to a row, so K − mean gives the same
outputs and gradients.

| arm | O | dQ | dK | dV |
|---|---|---|---|---|
| `all` | 1.12 [1.44] | 1.07 [1.43] | 1.53 [2.12] | 1.72 [2.79] |
| `all`, K − mean in fp16 | 1.13 [1.41] | 1.06 [1.33] | 1.51 [2.47] | 1.73 [2.15] |
| `all`, K − mean rounded to bf16 | 1.27 [1.73] | 1.17 [1.73] | 2.03 [2.95] | 2.35 [3.75] |
| `pv`, K − mean in fp16 | 1.00 | 0.99 | 1.01 | 1.01 |

The P-weighted |q·k| stays at 62.5 → 62.9: the large logits belong to particular query–key
pairs, not to a component shared by all keys. The subtraction alone takes 0.19 ms, the forward
gain of `all`.

## Partial sum handed to fp32 inside the head dim

[`attn_s_split_bench.py`](../src/attn_s_split_bench.py): one fp16 GEMM per chunk of the head
dim, summed in fp32, the same split on the forward and on the backward's S.

| chunk | O | dQ | dK | dV |
|---|---|---|---|---|
| 128 (`all`) | 1.12 [1.44] | 1.06 [1.44] | 1.53 [2.09] | 1.72 [2.79] |
| 64 | 1.04 [1.14] | 1.02 [1.09] | 1.18 [1.57] | 1.28 [1.56] |
| 32 | 1.01 [1.05] | 1.01 [1.24] | 1.08 [1.33] | 1.10 [1.20] |
| 16 | 1.01 [1.04] | 1.00 [1.21] | 1.04 [1.13] | 1.06 [1.12] |

Criterion, set before the run: dK and dV at most 1.10x by median and 1.21x on the worst call.
Only the 16-dim chunk passes. Those arms reload Q and K per chunk, so their time does not count.

Time comes from a four-way split cut in registers (`split_reg`, dims grouped by index mod 4;
error dK 1.08 [1.23], dV 1.10 [1.21], the same as the contiguous 32-dim chunk). ms per call,
median of 7 interleaved rounds:

| shape | arm | forward | backward | sum |
|---|---|---|---|---|
| Anima self | `port` | 1.307 | 4.128 | 5.436 |
| | `all` | 1.231 | 3.769 | 5.000 |
| | four-way split | 1.300 | 4.225 | 5.525 |
| MOSS self, random | `port` | 2.328 | 8.113 | 10.441 |
| | `all` | 2.211 | 7.374 | 9.586 |
| | four-way split | 2.349 | 8.335 | 10.684 |

The split is slower than `port`. Triton computes S in both backward kernels and FA2 in one, and
Triton emits one GEMM per chunk; taking the split's cost per S from here off FA2's gain leaves
about 0.14 ms a call on either line, 1% of an Anima step and 0.6% of a MOSS step, at an error
that only just meets the criterion. Not ported.

## Two int8 planes per operand

[`attn_int8x2_bench.py`](../src/attn_int8x2_bench.py). Q and K as 16-bit fixed point per token,
`scale · (a + b / 254)`; q·k ≈ a·a + (a·b + b·a) / 254 is three int8 GEMMs with exact int32
sums, and the backward recomputes S from the same planes.

| arm | O | dQ | dK | dV |
|---|---|---|---|---|
| one int8 plane, rotated (5a) | 1.71 [2.59] | 1.29 [2.30] | 2.99 [4.36] | 3.50 [5.30] |
| two planes | 1.00 [1.00] | 1.00 [1.05] | 1.00 [1.03] | 1.00 [1.00] |

| shape | arm | forward | backward | sum | quantization |
|---|---|---|---|---|---|
| Anima self | `port` | 1.291 | 4.131 | 5.422 | |
| | one plane | 0.953 | 3.887 | 4.840 | 0.108 |
| | two planes | 1.419 | 4.772 | 6.191 | 1.189 (torch ops) |
| MOSS self, random | `port` | 2.254 | 8.092 | 10.346 | |
| | one plane | 1.706 | 7.431 | 9.137 | 0.190 |
| | two planes | 2.595 | 9.259 | 11.854 | 2.532 (torch ops) |

S is exact, and the kernels are 14–15% slower than `port` before any quantization. Inside the
attention kernel an int8 GEMM is about 2.5x the fp32-accumulate one (0.23 against 0.57 ms on
the Anima forward), so three of them cost more than the GEMM they replace. A mix of one int8
GEMM and fp16 residual terms needs two fp16 GEMMs and costs more still; it was not built.

## Raw numbers

`assets/fa2_kcenter.json`, `attn_s_split.json`, `attn_s_split_speed.json`, `attn_int8x2.json`
(not tracked). Jobs `20261117-121829-c5c169`, `-122425-6bede0`, `-123044-bc52f5`,
`-124052-72f5c3`.
