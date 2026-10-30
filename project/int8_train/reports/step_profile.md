# Phase 1a — step split (2026-10-30)

The six convertible linears take **47% of a real compiled training step** (280 of
594 ms). With phase 0's 2.6x, int8 would save 172 ms per step, a predicted **1.41x**
step speedup; at a pessimistic 2.0x it is still 1.31x. The 1.25x gate passes.

Script: [`src/profile_step.py`](../src/profile_step.py), run as
`make daemon-run ARGS="--stall-timeout 1200 project/int8_train/src/profile_step.py --warmup 15 --timed 20 --profiled 8 -- --method lora --preset default"`
(job `20261030-094716-a9566b`). Raw numbers: [`step_profile.json`](step_profile.json).

## Setup

- `make lora` defaults: method `lora` (rank 32, T-LoRA mask, REPA), preset `default`
  (no block swap, no gradient checkpointing, `activation_memory_budget = 0.99`),
  `torch_compile` with `compile_dynamic_seq`, flash attention, B=1.
- The stock trainer, patched at the loop's per-step profiler hooks. Steps 1–15 warm up
  (compile), 16–35 are wall-timed with a sync on each side, 36–43 run under
  `torch.profiler`. Batches held 4056–4200 image tokens (the 1024 tier).
- GEMMs are `aten::mm` calls classified by their second operand: a converted layer
  shape (2048 / 6144 / 8192 pairs) with M ≥ 1024. Per-shape call counts match
  28 blocks × (forward + dX) exactly.

## Results

Per step, ms:

| part | ms | share |
|---|---|---|
| step wall time (median of 20) | 594 | |
| GPU kernel time (profiled) | 589 | 99% of step |
| converted GEMMs, forward + dX | 280 | 47% |
| flash attention, forward + backward | 172 | 29% |
| other GEMMs (cross kv, LoRA rank, adaLN, …) | 38 | 6% |
| everything else (fused elementwise, norms, optimizer) | ~99 | 17% |

Predicted step speedup, `594 / (594 − 280 · (1 − 1/r))`:

| GEMM ratio r | saved ms | step speedup |
|---|---|---|
| 2.0 | 140 | 1.31x |
| 2.3 | 158 | 1.36x |
| 2.60 (phase 0, M=4096) | 172 | 1.41x |

## Reading

- The step is GPU-bound: kernel time is 99% of wall time, so the GEMM saving carries
  over to the step one for one.
- 280 ms matches phase 0's per-block bf16 numbers × 28 blocks (272 ms at M=4096), so the
  microbench shapes are the right ones.
- The bf16 dX GEMMs run on a 64×64-tile CUTLASS kernel (143 ms) and the forwards on
  larger tiles; phase 0's bf16 dX timings already include this.
- After conversion, attention becomes the largest part (about 41% of a 422 ms step).
  That is out of scope here (no int8 attention backward).
- Not measured here: the gradient error (phase 1b), and whether the custom op costs
  inductor fusions around the linears (phase 3 measures the real step).
