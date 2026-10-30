# int8_train — int8 GEMMs for Anima LoRA training

Opened 2026-10-30. The frozen DiT's large linears run as ConvRot int8 × int8 → int32, in
the forward and in the backward input-gradient GEMM, through comfy_kitchen's existing
kernel. Anima first. The design, phases and gates are in [proposal.md](proposal.md).

All code lives in `src/`. The line patches the trainer from its own entry point and does
not change `library/` or `networks/`.

## Status

| phase | state |
|---|---|
| 0, GEMM microbench | done: int8 2.6x bf16 for forward and dX ([report](reports/gemm_microbench.md)) |
| 1a, step split | done: converted GEMMs are 47% of a 594 ms step, predicted 1.41x ([report](reports/step_profile.md)) |
| 1b, real-gradient error | passed on the revised gate (≤ 2x bf16's distance from fp32): int8 is 1.3–1.5x, a real but bf16-sized extra error ([report](reports/grad_error.md)) |
| 2, implement in `src/` | next (`src/int8_linear.py` exists) |
| 3, step time and 25-min A/B | — |

## Layout

- `proposal.md`: the design and gates.
- `src/gemm_bench.py`: phase 0 microbench, run as
  `make daemon-run ARGS="project/int8_train/src/gemm_bench.py 4096"`.
- `src/profile_step.py`: phase 1a step profile of the stock trainer (argv after `--`).
- `src/int8_linear.py`: the int8 custom ops and the in-place `convert_`.
- `src/grad_error.py`: phase 1b sketched-gradient error, bf16 vs `fwd` vs `full`.
- `reports/`: one report per phase.
