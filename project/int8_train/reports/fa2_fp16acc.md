# fp16 accumulation in FA2's backward (2026-11-02)

**Verdict: works, and repo jobs on sm_120 run it by default.** A real MOSS step goes from 1071–1082 ms to
1022 ms (4.6–5.5%) at the same 12.88 GB peak, and a real Anima int8 step from 426.5 to 404.7 ms (5.1%). FA2's backward runs 1.274x on Anima self-attention and 1.242x
on MOSS's self shape, 0.98x on cross-attention, with dQ, dK, dV at 0.74–0.97x stock FA2's distance
from fp32. MOSS's LoRA gradient keeps its distance from fp32 (1.000–1.005x).

The CUDA port of [attn_bwd_fp16acc.md](attn_bwd_fp16acc.md). Patch: `patches/fp16acc.patch` of
`~/anima/flash-attention-fp16`, on top of `sm120.patch`. Script:
[`src/fa2_fp16acc_bench.py`](../src/fa2_fp16acc_bench.py). Raw numbers:
[`fa2_fp16acc.json`](../assets/fa2_fp16acc.json), `moss_grad_error_fa2_{stock,fp16acc}.json`.
Wheel from a clean sdist + `sm120`, `fp16acc`: job `20261102-141315-5aa7a6`. Released with `hdim64.patch` as
`v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14`.

## Selection

The wheel holds sm_120 kernels and head dims 64 and 128, and sits in a private repo, so it cannot be the
`pyproject.toml` pin. `make fa2-sm120` unpacks it under `models/fa2_sm120/<tag>`, and
`library/runtime/fa2.py` puts that folder first on `sys.path` before `flash_attn` is imported, when the
folder exists, the card reports compute capability 12, torch is 2.14 and Python 3.13. Anywhere else the
pinned wheel loads, with a warning if the folder exists. ComfyUI, in the same venv, keeps the pin. Head dim
64 is there for Anima's `llm_adapter` (1024 / 16); 128 is the only one with the fp16acc backward. Qwen's
optional `flash_attention_2` text encoder does not go through the selector.

## Kernel

A separate translation unit, `flash_bwd_hdim128_fp16acc_sm80.cu`, with `Flash_bwd_fp16acc_kernel_traits`
on the fp16 kernel. `mha_bwd` takes it for bf16 input, head dim 128, no dropout, causal, window, alibi,
softcap or `deterministic`, and `num_heads_k == num_heads`. Anything else takes the stock path, and
`FLASH_ATTN_FP16ACC=0` forces it.

- Q, K, V run as fp16 (`to(half)`). S stays `F32F16F16F32`; Triton shows fp16 inputs cost nothing
  against bf16 here, on the 64-term tile too (arm `port`, `INT8_ATTN_BWD_L=64`).
- dP, dV, dK, dQ use `F16F16F16F16` into a per-block fragment, added to fp32 accumulators. Its C
  layout is the fp32 atom's.
- dO is divided by its max per (batch, head) in `dot_do_o`'s pre-kernel, which also stores the fp16
  copy. Doing this with torch ops cost 0.30 ms on Anima's self shape, 0.74 ms on MOSS's, and put
  cross-attention at 0.80x. In the pre-kernel only `abs` + `amax` and three `to(half)` remain.
- P is stored × 256 (block 0's underflow); dS and dV divide it back.
- Results leave as bf16 through the same 16-bit smem and gmem paths.

## Results

Wheel, ms per call. Forward is the 128×32 tile of `sm120.patch`.

| | q | forward | stock bwd | fp16acc bwd | bwd | fwd + bwd |
|---|---|---|---|---|---|---|
| Anima self | `[1, 4131, 16, 128]` | 1.490 | 3.787 | 2.973 | 1.274x | 1.182x |
| Anima cross | `[1, 4131, 16, 128]`, Lk 512 | 0.190 | 0.607 | 0.619 | 0.980x | 0.985x |
| MOSS self, random input | `[1, 4600, 24, 128]` | 2.667 | 6.992 | 5.628 | 1.242x | 1.164x |

Median error as a multiple of stock FA2's, dQ / dK / dV, 48 captured calls:

| | σ=0.1 | σ=0.5 | σ=0.9 |
|---|---|---|---|
| self | 0.94 / 0.75 / 0.74 | 0.95 / 0.80 / 0.77 | 0.97 / 0.90 / 0.79 |
| cross | 0.85 / 0.92 / 0.78 | 0.82 / 0.83 / 0.81 | 0.78 / 0.91 / 0.84 |

Worst single call: self 1.05 / 1.05 / 0.86, cross 0.96 / 1.01 / 1.00.

Real MOSS step, `phase_timer_moss.py`, `configs/moss/lora.toml` (int8 stream, fp32 stream), `hews_layout`,
40 steps, median of steps 11+, the same build with the flag off and on:

| | forward | backward | sum | peak |
|---|---|---|---|---|
| stock, run 1 | 351.8 | 711.0 | 1081.6 ms | 12.88 GB |
| stock, run 2 | | | 1071.3 ms | |
| fp16acc | 341.4 | 661.8 | 1022.2 ms | 12.88 GB |

Real Anima step, `profile_step.py --int8 full`, `--method lora --preset default`, 25 timed steps, the same
build with the flag off and on: 426.5 → 404.7 ms, 5.1%, which the per-call numbers predicted (22 ms).

No NaN. Loss tracks to the third digit. `|g|` at step 40 was 0.145 and 0.178, but stock alone repeats at
0.145 and 0.105, so that spread is FA2's existing non-determinism.

`grad_error_moss.py --modes bf16` with each path, on the bf16 plan: `e32` over noise 0.6866 → 0.6883,
0.9087 → 0.9133, 0.7173 → 0.7174 at σ 0.1 / 0.5 / 0.9. Per image and depth, 0.98–1.04x. The prediction
is unchanged, since the forward is.

Edge cases, all finite and under stock's error: B up to 4, Lq and Lk multiples of 64 and not, Lq ≠ Lk,
1 to 24 heads, an all-zero dO, one zero head, permuted dO strides. Stock's path still runs unchanged.

## Range

fp16 inputs overflow past 65504. The capture peaks at |V| 95, and Σ|V| per row at 2697, which bounds |dP|
24x under the limit. MOSS was not bounded this way; the real run stayed finite for 40 steps. An
overflow gives a NaN gradient, not a quiet error.

## Not done

- **Forward PV.** Triton's 0.25 ms would need Q, K, V in fp16 on the forward too, a net of about 0.17
  ms, 1% of a step, and it changes generation's output. Skipped.
- **The remaining prep.** `abs` + `amax` and `to(half)` are about 0.12 ms on self. Cross-attention
  lands at parity with them; a kernel of their own would gain about 2% of its time.
- **A/B.** One LoRA pair trained to completion: `caststation_fp16acc` and `caststation_fa2stock`, 240
  steps, 1.067 against 1.119 s/step, same 13.21 GB peak. Their renders were not scored, and there is no
  stock-against-stock floor.
