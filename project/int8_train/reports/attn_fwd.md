# Phase 5a — int8 attention forward in Triton (2026-10-30)

> **Re-judged after 5b ([attn_bwd.md](attn_bwd.md)): fails.** The gate below scores the
> output O only. Through FA2's backward, this forward's O and logsumexp put dQ, dK, dV
> at 2.5–3.4x FA2's distance from fp32, because dS = P ∘ (dP − Δ) amplifies the logit
> error about twice. The gate is now on those gradients (proposal.md, 5a′).

**Verdict: passes for self-attention, fails for cross-attention.** With int8 QKᵀ
(per-token scales, Q and K Hadamard-rotated along D, K mean-removed) and PV in fp16
with fp16 accumulation, the self-attention forward runs **1.44x FA2** with
quantization included. Its output error is **1.5–1.8x FA2's** distance from fp32, under
the 2x gate. Cross-attention stays at 2.1x or more in every variant, because 400–470
of its 512 keys are one repeated padding key and one sink key has about 30x the norm of
the others. Cross-attention stays on FA2. It is 12% of the attention forward time.

Script: [`src/attn_bench.py`](../src/attn_bench.py), kernel
[`src/int8_attn.py`](../src/int8_attn.py). Raw numbers of the final run:
[`attn_fwd.json`](../assets/attn_fwd.json); earlier runs: `attn_fwd_run{1,2,3}.json`.

## Setup

- Capture: 2 images (3500–4608 tokens), σ ∈ {0.1, 0.5, 0.9}, blocks 0, 9, 18, 27, self
  and cross: 48 real calls with q, k, v and dO, in `output/int8_train/attn_capture.pt`
  (dO is for 5b).
- Error: `‖o − o_fp32‖ / ‖o_fp32‖` against fp32 SDPA (TF32 off) on the same bf16 inputs.
  Gate: median per kind and σ at most 2x FA2's.
- Speed: image 0's shapes, q `[1, 4131, 16, 128]`, Lk 4131 (self) or 512 (cross), CUDA
  events over 50 calls after autotune. Arms include the K/V quantization kernels.

## Runs

| run | job | arms | notes |
|---|---|---|---|
| 1 | `20261030-123855-73488c` | PV bf16, fp16, fp16acc, int8; per-block scales | PV int8 is 20–80x FA2's error |
| 2 | `20261030-124331-70b381` | per-block and per-token scales | per-token cuts the error by 25–45% at the same speed |
| 3 | `20261030-124825-76d640` | Hadamard rotation | accuracy valid; timings invalid (Hadamard rebuilt on the host per call, fp32 `ieee` dot in the K prep) |
| 4 | `20261030-125249-c1fd7b` | rotation, both fixed | the final numbers below |

## Results (run 4)

Median output error, PV fp16acc in every int8 arm:

| kind | σ | FA2 | Triton bf16 | int8 block | int8 token | int8 block + rot | **int8 token + rot** |
|---|---|---|---|---|---|---|---|
| self | 0.1 | 0.0017 | 0.0017 | 0.0051 | 0.0039 | 0.0037 | **0.0031** (1.81x) |
| self | 0.5 | 0.0017 | 0.0017 | 0.0044 | 0.0034 | 0.0031 | **0.0027** (1.57x) |
| self | 0.9 | 0.0017 | 0.0017 | 0.0040 | 0.0032 | 0.0029 | **0.0025** (1.48x) |
| cross | 0.1 | 0.0018 | 0.0020 | 0.0101 | 0.0058 | 0.0066 | 0.0051 (2.85x) |
| cross | 0.5 | 0.0018 | 0.0019 | 0.0103 | 0.0058 | 0.0060 | 0.0050 |
| cross | 0.9 | 0.0018 | 0.0019 | 0.0101 | 0.0057 | 0.0059 | 0.0051 |

The worst single self call is block 18, σ=0.9: 0.0043, 2.56x FA2.

Forward time, ms (speedup against FA2):

| arm | self | cross |
|---|---|---|
| FA2 | 1.611 | 0.217 |
| Triton bf16 | 1.591 (1.01x) | 0.209 (1.04x) |
| int8 block | 1.022 (1.58x) | 0.138 (1.58x) |
| int8 token | 1.003 (1.61x) | 0.141 (1.54x) |
| int8 block + rot | 1.091 (1.48x) | 0.161 (1.35x) |
| **int8 token + rot** | **1.120 (1.44x)** | 0.166 (1.31x) |
| K prep alone (token + rot) | 0.060 | 0.037 |

Run 1 also measured PV variants with per-block int8 QKᵀ. PV bf16 and PV fp16 with fp32
accumulation have the same error as fp16acc and are 7–16% slower on the self shape. PV
int8 (P rounded to 1/127, V per channel) is 1.36x and 20–80x FA2's error: P below 1/254
of its row maximum rounds to 0, and over 4k keys that tail carries real mass.

## Reading

- **Triton matches FA2's CUDA kernel** in bf16 (1.01x, same error). The gains come
  from int8, not from Triton.
- **Granularity and rotation do most of the work.** Going from per-block to per-token
  scales and adding the rotation cuts self error from 3x FA2's to 1.5–1.8x. That costs
  0.1 ms, the rotation of each Q block in the kernel and of K in the prep.
- **Cross-attention fails for structural reasons.** The TE context is max-padded, so
  400–470 of its 512 keys are the same padding key. That key's rounding error repeats
  in every copy and moves the padding's whole attention mass the same way. Run 3's
  cross error (0.0029 with an fp32 prep) against run 4's (0.0051 with a bf16 prep) is
  this one vector's rounding landing differently. Centering K on the padding key
  instead of the mean (`quant_k(center="last")`) makes it exactly 0 and brings the
  median to 0.0038, still 2.1x. The rest is one sink key with |k| ≈ 9.7 against
  0.26–0.37 for the others, where Q's rounding error is multiplied by the large key.
- **Expected step gain from self-attention only:** 28 × (1.61 − 1.12) ≈ 14 ms in the
  forward. If 5b's backward gains the same ratio, it adds about 34 ms, so roughly
  433 → 385 ms per step, about 1.54x against bf16 in total.

## Next

5b, the backward for self-attention, on the captured dO: recompute QKᵀ with the
forward's rotated int8 Q and K, int8 dV, dQ and dK, fp16 dO·Vᵀ.
