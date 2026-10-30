# Phase 5a′ — forward variants judged on gradients through FA2's backward (2026-10-30)

**Verdict: int8 QKᵀ is out; fp16-accumulate QKᵀ passes, narrowly.** Rounding either Q
or K alone to per-token int8 (Hadamard-rotated, as the kernel does) already puts the
gradients at 1.8–2.6x FA2's distance from fp32; rounding both gives 2.5–3.4x. Q
smoothing does not help. QKᵀ and PV in fp16 with fp16 accumulation, no quantization,
stay at **1.03–1.40x FA2's gradient error** and run **1.31x FA2's forward**, just over
both gates. The step gain is about 28 × (1.626 − 1.246) ≈ 11 ms of the 433 ms int8
step (2.5%).

Script: [`src/attn_fwd_grad_bench.py`](../src/attn_fwd_grad_bench.py) (job
`20261030-133447-233ad7`). Raw numbers: [`attn_fwd_grad.json`](../assets/attn_fwd_grad.json).

## Setup

The 24 self-attention calls of the 5a capture. Each arm produces O and a natural-log
logsumexp; FA2's CUDA backward (`_flash_attn_backward`) takes them with the captured
dO. dQ, dK, dV and O are compared with fp32 SDPA autograd (TF32 off) on the same bf16
inputs. Gate (proposal.md, 5a′): each gradient's median per σ at most 2x FA2's, and a
kernel arm at least 1.3x FA2's forward on q `[1, 4131, 16, 128]`.

The `emu_*` arms are fp32 torch emulations of the kernel's rounding (centering, bf16
cast, Hadamard, per-token scale, round-half-away) with exact fp32 attention after it.
`emu_int8` reproduces the int8 kernel's errors to the second decimal in every cell, so
the emulation is faithful and fp16acc PV adds nothing measurable.

## Results

Median error as a multiple of FA2's, O / dQ / dK / dV:

| arm | σ=0.1 | σ=0.5 | σ=0.9 |
|---|---|---|---|
| FA2 (absolute) | 0.0017 / 0.0035 / 0.0027 / 0.0023 | 0.0017 / 0.0029 / 0.0025 / 0.0022 | 0.0017 / 0.0031 / 0.0023 / 0.0022 |
| `emu_exact`, fp32 forward | 0.98 / 1.01 / 0.99 / 1.00 | 0.98 / 0.98 / 1.00 / 1.00 | 0.98 / 1.02 / 0.99 / 1.00 |
| Triton bf16 QKᵀ, fp16acc PV | 0.99 / 0.99 / 1.00 / 1.00 | 0.99 / 0.98 / 1.00 / 1.00 | 0.99 / 1.04 / 0.99 / 1.00 |
| **fp16acc QKᵀ and PV** | 1.13 / 1.11 / 1.03 / **1.40** | 1.10 / 1.17 / 1.16 / **1.40** | 1.08 / 1.20 / 1.25 / 1.23 |
| `emu_q`, Q rounded | 1.22 / 2.46 / 1.86 / 2.23 | 1.19 / 2.44 / 2.17 / 2.14 | 1.21 / 2.13 / 2.45 / 2.39 |
| `emu_k`, K rounded | 1.59 / 2.29 / 1.95 / 1.84 | 1.41 / 2.04 / 1.89 / 1.75 | 1.31 / 2.26 / 2.60 / 2.36 |
| `emu_int8`, both | 1.80 / 3.09 / 3.07 / 2.85 | 1.56 / 2.90 / 2.53 / 2.67 | 1.48 / 2.73 / 3.41 / 3.15 |
| int8 kernel (5a) | 1.81 / 3.10 / 3.06 / 2.85 | 1.57 / 2.89 / 2.51 / 2.66 | 1.48 / 2.73 / 3.41 / 3.15 |
| `emu_qsmooth`, both, Q mean removed | 1.83 / 2.88 / 2.54 / 2.87 | 1.58 / 3.11 / 2.88 / 2.86 | 1.49 / 2.74 / 3.37 / 2.76 |

Worst single call (dK / dV against FA2): int8 5.6 / 9.7, Q smoothing 6.3 / 11.4,
fp16acc 2.9 / 4.0.

Forward time, ms (against FA2's 1.626): int8 1.154 (1.41x), Triton bf16 QKᵀ with
fp16acc PV 1.331 (1.22x), **fp16acc QKᵀ and PV 1.246 (1.31x)**.

Q's token mean is 0.22–0.52 of the RMS token norm (K's, already removed in the kernel:
0.13–0.41).

## Reading

- **Both operands carry the int8 error, about equally.** Each alone is ~2x, together
  ~3x, so no single-operand fix (Q smoothing, a residual int8 term on one side) can
  bring int8 QKᵀ under 2x. Q's mean is large, but after the Hadamard rotation and a
  per-token scale it no longer sets the rounding step, so removing it changes nothing.
- **fp16 accumulation is nearly free in accuracy.** fp16 keeps 10 mantissa bits against
  int8's 7, and accumulating the 128-term dot in fp16 costs less than rounding the
  inputs. GeForce runs fp16 accumulation at twice fp32's rate; that is all the 1.31x.
- **The margin is thin on both gates.** 1.31x against 1.3x is within run-to-run noise,
  and dV's 1.40x median hides a 4x worst call.

## Next

The candidate for 5c is the fp16acc forward with FA2's backward: about 11 ms per step,
1.37x → about 1.40x against bf16, and cross-attention could take the same kernel (no
quantization, so the padding-key failure does not apply). Whether that gain is worth
5c's integration (the kernel behind `dispatch_attention` under `torch.compile`, phase
1b's gradient gate, a 25-minute A/B) is an open decision.
