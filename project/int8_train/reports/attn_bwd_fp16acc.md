# fp16-accumulate attention backward (2026-11-02)

**Verdict: passes the gate, with QKᵀ left in bf16.** dP, dV, dQ and dK in fp16 accumulation, plus
PV on the forward, run 1.340x the Triton baseline's backward and 1.304x its forward + backward, at
1.02–1.10x its gradient error (worst single call 1.21x). fp16 accumulation of QKᵀ is the one part
that does not hold: it alone puts dK and dV at 2.06–2.18x. On Triton's two-kernel layout this is
only 1.01x FA2. The port to FA2's CUDA backward is [fa2_fp16acc.md](fa2_fp16acc.md).

Candidate 2 of [step_headroom.md](../../../docs/moss/proposal/step_headroom.md). Script:
[`src/attn_bwd_fp16acc_bench.py`](../src/attn_bwd_fp16acc_bench.py), kernels in
[`src/int8_attn.py`](../src/int8_attn.py). Raw numbers:
[`attn_bwd_fp16acc.json`](../assets/attn_bwd_fp16acc.json) (job `20261102-133015-5cab14`; two earlier
runs with fewer arms, `20261102-131920-8f9d9e` and `20261102-132314-48282a`, agree within 1%).

## Setup

As [attn_bwd.md](attn_bwd.md): the 24 self-attention calls of the 5a capture (Anima, 2 images,
σ ∈ {0.1, 0.5, 0.9}, blocks 0, 9, 18, 27), their real dO, error against fp32 SDPA autograd, speed on
q `[1, 4131, 16, 128]`. Gate from [reports.md](../reports.md), against `tri_fp16` (bf16 forward, fp16
inputs with fp32 accumulation): dQ, dK, dV each at most 2x, median per σ, and at least 1.3x the
backward's time. Baseline here: forward 1.557 ms, backward 5.422 ms, error 0.0026–0.0033 / 0.0019 /
0.0017. FA2 on the same run: 1.614 + 3.799 ms.

New modes: `dqk` `fp16acc`, `s` `fp16acc`, `fp16acc_row` on `dv` and `dqk` (left operand scaled to
unit max per tile row) and `fp16acc_gain` on `dv` (P × 256, no reduction).

## Results

Worst median over σ and worst single call, as multiples of the baseline, dQ / dK / dV:

| arm | forward | backward modes | fwd ms | bwd ms | bwd | fwd + bwd | worst median | worst call |
|---|---|---|---|---|---|---|---|---|
| `dp` | bf16 | dP | 1.572 | 4.646 | 1.167x | 1.122x | 1.06 / 1.08 / 1.00 | 1.08 / 1.18 / 1.00 |
| `dv` | bf16 | dV | 1.557 | 5.103 | 1.063x | 1.048x | 1.00 / 1.00 / 1.04 | 1.00 / 1.00 / 3.74 |
| `dv_row` | bf16 | dV, row scale | 1.557 | 5.318 | 1.020x | 1.015x | 1.00 / 1.00 / 1.01 | 1.00 / 1.00 / 1.02 |
| `dqk` | bf16 | dQ, dK | 1.557 | 5.003 | 1.084x | 1.064x | 1.00 / 1.01 / 1.00 | 1.02 / 1.03 / 1.00 |
| `dqk_row` | bf16 | dQ, dK, row scale | 1.557 | 5.225 | 1.038x | 1.029x | 1.03 / 1.01 / 1.00 | 1.06 / 1.03 / 1.00 |
| `s` | bf16 | S | 1.557 | 4.764 | 1.138x | 1.104x | 1.50 / 2.06 / 2.18 | 2.07 / 2.76 / 5.92 |
| `all` | bf16 | all | 1.557 | 3.690 | 1.469x | 1.330x | 1.55 / 2.08 / 2.80 | 2.15 / 2.99 / 5.92 |
| `all_fwd16` | fp16acc | all | 1.221 | 3.762 | 1.441x | 1.401x | 1.14 / 1.69 / 1.91 | 1.44 / 2.19 / 3.74 |
| `no_s` | bf16 | dP, dV, dQ, dK | 1.557 | 4.024 | 1.348x | 1.251x | 1.06 / 1.10 / 1.04 | 1.07 / 1.17 / 3.74 |
| `no_s_dvgain` | bf16 | same, dV gain | 1.557 | 4.020 | 1.349x | 1.251x | 1.06 / 1.10 / 1.02 | 1.07 / 1.17 / 1.06 |
| **`no_s_dvgain_fwdpv`** | bf16 QKᵀ, fp16acc PV | same, dV gain | 1.304 | 4.046 | 1.340x | 1.304x | 1.04 / 1.10 / 1.02 | 1.10 / 1.21 / 1.06 |

`no_s_dvgain_fwdpv` against FA2's own error: 0.91–0.98 / 0.75–0.88 / 0.74–0.79x.

## Reading

- **The feared part is free.** dQ = Σ dS·K and dK = Σ dSᵀ·Q are signed, near-cancelling sums, and
  fp16 accumulation moves them by at most 1%. Row scaling buys nothing there and costs 0.22 ms.
- **QKᵀ is the error source.** The bf16 forward's logsumexp no longer matches the fp16acc logits,
  and dS is a difference of near-equal terms. With an fp16acc forward the two share part of the
  rounding, which takes dK and dV from about 2.1x to about 1.7–1.9x (`all_fwd16`). That arm passes
  and is the fastest, but `no_s_dvgain_fwdpv` gives up 7% of speed for a third of its error.
- **dV's tail is underflow in block 0.** The five worst `dv` calls are all block 0, where P·dO is
  small enough that fp16 loses it (median 1.04x, worst call 3.74x). A row scale fixes it for 0.21 ms.
  A fixed P × 256 fixes it for nothing: P ≤ 1 and |dO| ≤ 1 keep a 64-query tile under 16384.
- **dP takes most of the saving** (0.78 ms), but it runs in both kernels here, and FA2 runs it once.
