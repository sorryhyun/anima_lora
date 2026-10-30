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
| 2, implement in `src/` | done: `train_int8.py`, 7 unit tests pass ([report](reports/phase3_ab.md)) |
| 3, step time and A/B | passed: 1.37x step, loss and bf16-base samples match on ama mitsuki, 8 epochs ([report](reports/phase3_ab.md)) |

## Layout

- `proposal.md`: the design and gates.
- `gptq_offline.md`: deferred plan to GPTQ-calibrate the int8 weights if phase 3 fails
  on quality.
- `src/gemm_bench.py`: phase 0 microbench, run as
  `make daemon-run ARGS="project/int8_train/src/gemm_bench.py 4096"`.
- `src/profile_step.py`: phase 1a step profile of the stock trainer (argv after `--`).
- `src/int8_linear.py`: the int8 custom ops and the in-place `convert_`.
- `src/grad_error.py`: phase 1b sketched-gradient error, bf16 vs `fwd` vs `full`.
- `src/train_int8.py`: the trainer on an int8 base, run as
  `make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/train_int8.py --int8 full --method lora --preset default"`.
- `src/test_int8_linear.py`: phase 2 tests (`.venv/bin/python -m pytest project/int8_train/src/test_int8_linear.py`).
- `src/ab_prompts.txt`: phase 3 sample prompts.
- `reports/`: one report per phase.
