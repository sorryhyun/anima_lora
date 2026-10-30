# Phase 5b — int8 attention backward in Triton (2026-10-30)

**Verdict: fails, on both accuracy and speed. The int8 backward is dropped; the
backward stays FA2.** Recomputing P from the forward's int8 Q and K
alone puts dK and dV at 2.5–3.3x FA2's distance from fp32, over the 2x gate, before any
backward GEMM is int8. Running dV = Pᵀ·dO in int8 raises dV to 5–8x. The fastest arm
(every GEMM int8) is 1.17x FA2's backward, under the 1.3x gate. Feeding the int8
forward's O and logsumexp into FA2's own backward is also 2.5–3.4x, so 5a's forward
cannot ship alone either: the gradient error comes from the int8 forward, whichever
backward follows it. Forward arms are now judged on these gradients (proposal.md, 5a′).

Script: [`src/attn_bwd_bench.py`](../src/attn_bwd_bench.py), kernels in
[`src/int8_attn.py`](../src/int8_attn.py) (`attn_fwd_train`, `attn_bwd`). Raw numbers:
[`attn_bwd.json`](attn_bwd.json) (job `20261030-131458-007ef8`); the first run without
the FA2-backward arm: `attn_bwd_run1.json` (`20261030-131136-122e62`), same numbers.

## Setup

- Data: the 24 self-attention calls of 5a's capture (2 images, 3500–4608 tokens,
  σ ∈ {0.1, 0.5, 0.9}, blocks 0, 9, 18, 27), with their real dO.
- Error: `‖g − g_fp32‖ / ‖g_fp32‖` for dQ, dK, dV against fp32 SDPA autograd (TF32 off)
  on the same bf16 inputs. Gate: median per σ at most 2x FA2's, for each gradient.
- Speed: image 0's shape, q `[1, 4131, 16, 128]`, CUDA events over 50 calls after
  autotune. Every arm's backward includes its preparation kernels (dO scaling and Δ,
  the transposed int8 copies); its forward includes the Q and K quantization.

## Kernel

FA2's backward as two Triton kernels (the layout of Triton's fused-attention tutorial):
one program per K/V block accumulates dK and dV over all Q blocks, one per Q block
accumulates dQ. Both recompute P from the forward's stored logsumexp. With an int8
forward, the forward now reads a Q copy made by a prep kernel (`quant_tok`, token +
Hadamard, the same rounding as 5a's in-kernel path), so the backward's int8 Qᵀ·K is
bit-identical to the forward's.

- **dO scaling.** Captured dO has mean |dO| ≈ 1.4e-7, under fp16's normal range. dO is
  divided by its max per (batch, head) and held in fp16; results are multiplied back at
  the store.
- **Per-GEMM modes.** dP = dO·Vᵀ: fp16 (fp32 accumulate), fp16acc, or int8 (dO and V per
  token, rotated). dV = Pᵀ·dO, dK = dSᵀ·Q, dQ = dS·K: fp16, or int8 with the register
  operand (P or dS) rounded per tile row and the other a transposed int8 copy, tokens
  contiguous, one scale per 64 tokens and channel.
- **Triton fp16 baseline** (bf16 forward, every GEMM fp16 with fp32 accumulation):
  0.7–0.95x FA2's error, so the kernel is correct.

## Results

Median error as a multiple of FA2's, dQ / dK / dV:

| arm | forward | dP | dV | dQ, dK | σ=0.1 | σ=0.5 | σ=0.9 |
|---|---|---|---|---|---|---|---|
| FA2 (absolute) | bf16 | | | | 0.0035 / 0.0027 / 0.0023 | 0.0029 / 0.0025 / 0.0022 | 0.0031 / 0.0023 / 0.0022 |
| Triton fp16 | bf16 | fp16 | fp16 | fp16 | 0.94 / 0.70 / 0.73 | 0.92 / 0.75 / 0.76 | 0.95 / 0.79 / 0.78 |
| int8 QKᵀ only | int8 | fp16 | fp16 | fp16 | 1.66 / 2.54 / 2.25 | 1.99 / 2.96 / 2.49 | 1.54 / 2.96 / 3.28 |
| int8 QKᵀ, fp16acc | int8 | fp16acc | fp16acc | fp16 | 1.70 / 2.61 / 2.39 | 1.99 / 2.98 / 2.64 | 1.55 / 2.96 / 3.27 |
| proposal's 5b | int8 | fp16acc | int8 | int8 | 4.73 / 4.05 / 8.18 | 4.31 / 4.41 / 7.84 | 4.22 / 3.79 / 5.33 |
| dP fp32 acc | int8 | fp16 | int8 | int8 | 4.72 / 4.03 / 8.18 | 4.29 / 4.40 / 7.84 | 4.23 / 3.72 / 5.33 |
| all int8 | int8 | int8 | int8 | int8 | 5.64 / 4.65 / 8.18 | 5.43 / 5.61 / 7.84 | 5.68 / 4.88 / 5.33 |
| int8 fwd → FA2 bwd | int8 | FA2 | FA2 | FA2 | 3.10 / 3.06 / 2.85 | 2.89 / 2.51 / 2.66 | 2.73 / 3.41 / 3.15 |

Time per call, ms, and the saving over 28 blocks against FA2's 1.628 + 3.930 ms:

| arm | forward | backward (vs FA2) | fwd + bwd (vs FA2) | saved per step |
|---|---|---|---|---|
| Triton fp16 | 1.609 | 5.575 (0.70x) | 0.77x | −45 ms |
| int8 QKᵀ only | 1.070 | 4.236 (0.93x) | 1.05x | 7 ms |
| int8 QKᵀ, fp16acc | 1.086 | 3.630 (1.08x) | 1.18x | 24 ms |
| proposal's 5b | 1.098 | 3.871 (1.02x) | 1.12x | 16 ms |
| dP fp32 acc | 1.100 | 4.229 (0.93x) | 1.04x | 6 ms |
| all int8 | 1.101 | 3.354 (1.17x) | 1.25x | 31 ms |
| int8 fwd → FA2 bwd | 1.099 | 3.993 (0.98x) | 1.09x | 13 ms |

Preparation kernels: `quant_tok` 0.045 ms, `quant_t` 0.021 ms per tensor.

## Reading

- **The int8 forward sets the floor.** The arm with only Qᵀ·K in int8 is already 2.5–3.3x
  on dK and dV, and FA2's backward, given the int8 forward's O and logsumexp (corrected
  for the removed K mean, `lse_natural`), lands at the same 2.5–3.4x. dS = P ∘ (dP − Δ)
  is a difference of near-equal terms. A logit error that moves O by 1.5–1.8x FA2's
  (5a) moves dS, and with it dK and dQ, by about twice that. dQ fares best (1.5–2.0x)
  because it is a sum over all keys, where the per-key errors partly cancel.
- **int8 dV repeats 5a's PV failure.** P rounded per row of a 32–64-query tile still
  rounds the small entries to 0, and over 4k queries that tail carries real mass: 5–8x.
- **The two-kernel layout costs 30% before int8.** It recomputes S and dP in both
  kernels, 7 GEMMs against FA2's 5, so Triton fp16 is 0.70x FA2. FA2's single pass with
  an atomic dQ would close part of that, but not the accuracy gap, which no backward
  layout changes.
- **The step gain would be small even if accuracy passed:** the best arm saves 31 ms of
  the 433 ms int8 step (7%), at 5–8x FA2's gradient error.

## Next

The backward stays FA2. 5a′ re-measures forward variants by the gradients they yield
through FA2's backward (the `int8fwd_fa2bwd` arm here), at most 2x FA2's per gradient.
Until one passes, the int8 step stays at phase 3's 1.37x with FA2 for all attention.
