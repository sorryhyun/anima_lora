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
| error split, int8-base inference | weight term 35–50% of the forward error; int8 base moves samples 4/255, the LoRA swap 26/255 ([report](reports/split_and_infer.md)) |
| offline GPTQ base | done: W8A8 per-linear error −36%, predictions −10%, CFG velocity −2–7%; same on trajectories and held-out real ([report](reports/gptq.md)) |
| 5a, int8 attention forward (Triton) | self-attention passes: per-token + Hadamard int8 QKᵀ, fp16acc PV, 1.44x FA2 at 1.5–1.8x its error; cross-attention fails (repeated pad key, sink key) and stays FA2 ([report](reports/attn_fwd.md)). **Re-judged after 5b: fails**, dQ/dK/dV through FA2's backward are 2.5–3.4x FA2's error |
| 5b, int8 attention backward (Triton) | fails: the int8 forward alone puts dK and dV at 2.5–3.4x FA2's error with any backward (FA2's included), int8 dV 5–8x, fastest arm 1.17x; the backward stays FA2 ([report](reports/attn_bwd.md)) |
| 5a′, forward re-measured on gradients | open: forward arms through FA2's backward, dQ/dK/dV ≤ 2x FA2's distance from fp32 (fp16-accumulate QKᵀ, Q smoothing, Q/K rounding split) |

## Layout

- `proposal.md`: the design and gates.
- `gptq_offline.md`: the GPTQ plan; carried out in [reports/gptq.md](reports/gptq.md).
- `src/gemm_bench.py`: phase 0 microbench, run as
  `make daemon-run ARGS="project/int8_train/src/gemm_bench.py 4096"`.
- `src/profile_step.py`: phase 1a step profile of the stock trainer (argv after `--`).
- `src/int8_linear.py`: the int8 custom ops and the in-place `convert_`.
- `src/grad_error.py`: phase 1b sketched-gradient error, bf16 vs `fwd` vs `full`.
- `src/train_int8.py`: the trainer on an int8 base, run as
  `make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/train_int8.py --int8 full --method lora --preset default"`.
- `src/infer_int8.py`: `inference.py` on the int8 base, LoRAs as hooks, run as
  `make daemon-run ARGS="project/int8_train/src/infer_int8.py --int8 on <inference args>"`.
- `src/dit_gptq.py`: offline GPTQ of the forward copies, writes a ComfyUI-layout int8 DiT
  (`models/diffusion_models/anima-base-v1.0_int8_gptq.safetensors`, on the NVMe). `train_int8.py` and
  `infer_int8.py` take it as `--int8_file`.
- `src/traj_eval.py`: RTN against a GPTQ file on bf16 sampling trajectories and held-out
  real images.
- `src/int8_attn.py`: phase 5 Triton attention, int8 QKᵀ and the PV variants, and the
  5b backward (`attn_fwd_train`, `attn_bwd`).
- `src/attn_bench.py`: phase 5a capture (q, k, v, dO from real calls, saved to
  `output/int8_train/attn_capture.pt`) and the forward error and speed against FA2.
- `src/attn_bwd_bench.py`: phase 5b, dQ, dK, dV error and backward speed against FA2 on
  the 5a capture.
- `src/test_int8_attn.py`: phase 5 tests (`INT8_ATTN_FIXED=1` skips the autotune sweep).
- `src/test_int8_linear.py`: phase 2 tests (`.venv/bin/python -m pytest project/int8_train/src/test_int8_linear.py`).
- `src/ab_prompts.txt`: phase 3 sample prompts.
- `reports/`: one report per phase.
