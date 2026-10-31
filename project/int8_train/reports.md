# int8_train — completed work (2026-10-30)

What has been measured and decided, in the order it was done. Each entry links its
report under `reports/`. The reports and the `src/` docstrings keep the phase labels
they were written with; each link below names the label its report is titled with.
The labels are history only: the current task list starts again at 1 ([tasks.md](tasks.md)).

## int8 linears

**Complete.** The int8 base is implemented, tested and has trained a LoRA end to end:
848 steps on ama mitsuki at 1.37x, loss and bf16-base samples matching the bf16 run.
That A/B ran on round-to-nearest weights; the GPTQ base has been built and scored but
not trained on.

| work | result |
|---|---|
| GEMM microbench | int8 2.6x bf16 for forward and dX on Anima's block shapes; raw int8 3.3x, fp8 only 1.9x (GeForce halves fp32 accumulation, not int32) ([report](reports/gemm_microbench.md), titled *Phase 0*) |
| step split | converted GEMMs are 47% of a 594 ms step (forward ~137 ms, dX ~143 ms), predicted 1.41x ([report](reports/step_profile.md), titled *Phase 1a*) |
| real-gradient error | int8 `full` is 1.3–1.5x bf16's distance from the fp32 LoRA gradient, a bf16-sized extra error; it comes from the int8 forward (`fwd` ≈ `full`) ([report](reports/grad_error.md), titled *Phase 1b*) |
| implementation | `src/int8_linear.py`, `src/train_int8.py`, 7 unit tests pass ([report](reports/phase3_ab.md), titled *Phase 2–3*) |
| step time and A/B | 1.37x step (433 against 593 ms); loss and bf16-base samples match on ama mitsuki, 8 epochs ([report](reports/phase3_ab.md), titled *Phase 2–3*) |
| error split, int8-base inference | weight term 35–50% of the forward error; the int8 base moves samples 4/255, the LoRA swap 26/255 ([report](reports/split_and_infer.md)) |
| offline GPTQ base | W8A8 per-linear error −36%, predictions −10%, CFG velocity −2–7%; same on trajectories and held-out real ([plan](gptq_offline.md), [report](reports/gptq.md)) |

Criteria used:

- **Gradient:** the int8 LoRA gradient's distance from the fp32 gradient at most 2x
  bf16's, median over modules, at every σ. The first criterion (int8 within 0.1 of
  minibatch noise from bf16) was dropped because bf16 itself sits at 0.15–0.20 of noise
  from fp32.
- **End to end:** step speedup ≥ 1.2x and samples, generated on the bf16 base, not
  visibly worse than bf16's.

Settled:

- **The int8 dX GEMM is free in accuracy; the forward carries the error.** The residual
  `pred − target` amplifies the prediction error 2.7–4.7x, and the backward Jacobians
  are built from the perturbed activations. Early blocks (0–13) take most of it, as they
  already do in bf16.

## int8 attention in Triton

All on the same capture: q, k, v and dO from 24 self-attention and 24 cross-attention
calls of real training (2 images, 3500–4608 tokens, blocks 0, 9, 18, 27,
σ ∈ {0.1, 0.5, 0.9}), scored against fp32 SDPA autograd (TF32 off) on the same bf16
inputs.

| work | result |
|---|---|
| int8 QKᵀ forward | on the output O: 1.44x FA2 at 1.5–1.8x its error for self-attention; cross-attention fails (repeated pad key, sink key). On gradients through a 16-bit backward: dQ, dK, dV at 2.5–3.4x, fails ([report](reports/attn_fwd.md), titled *Phase 5a*) |
| int8 backward | fails on accuracy: on the int8 forward every backward is 2.5–3.4x FA2's error, int8 dV 5–8x ([report](reports/attn_bwd.md), titled *Phase 5b*) |
| forward arms on gradients | int8 QKᵀ out: Q or K rounding alone ~2x, both ~3x, Q smoothing no help. fp16-accumulate QKᵀ and PV: 1.03–1.40x error ([report](reports/attn_fwd_grad.md), titled *Phase 5a′*) |

Criterion, as it stands now:

- **Baseline: the Triton 16-bit arm on the same loop**, the bf16 forward
  (`attn_fwd(..., "bf16", "bf16")`) and the fp16 backward (`tri_fp16`; the backward has
  no bf16 mode, dO is scaled into fp16 range). Its distance from fp32 is the tolerance
  unit and its time the speed unit. FA2 is not a comparison target for a Triton arm.
- **Pass:** dQ, dK, dV each at most 2x the baseline's error, median per σ, and at least
  1.3x the baseline's time. A forward arm is judged on the gradients it yields through a
  16-bit backward, not on O: dS = P ∘ (dP − Δ) is a difference of near-equal terms and
  moves the gradients about twice as far as O.

The reports above were first judged against FA2. Restated on the Triton baseline:

| | against FA2 | against Triton |
|---|---|---|
| Triton bf16 forward | 1.01x, same error | 1.0 |
| int8 QKᵀ forward, time | 1.44x | 1.42x |
| fp16acc forward, time | 1.31x | 1.28x (1.246 against 1.591 ms, two separate runs) |
| Triton fp16 backward | 0.70x time, 0.70–0.95x error | 1.0 |
| all-int8 backward, time | 1.17x | 1.66x (3.354 against 5.575 ms) |
| int8 QKᵀ + fp16acc backward, time | 1.08x | 1.54x |
| backward error multiples | as reported | larger by 1/0.95–1/0.70 |

Settled:

- **No public int8 attention has a backward** (comfy_kitchen and SageAttention 1–3 are
  forward-only; SageBwd is unreleased), which is why this work is in Triton.
- **int8 QKᵀ is out at any backward.** Both operands carry the error about equally, so no
  single-operand fix brings it under 2x.
- **In the backward, int8 speeds up only the GEMMs that reduce over D** (S = QKᵀ,
  dP = dO·Vᵀ). For dV, dK, dQ the reduction runs over tokens in 32–64-token tiles, so P
  and dS are quantized in the kernel and the int32 tile is dequantized every iteration:
  switching those three to int8 saved 0.007 ms of 4.236.
- **The two-kernel backward layout** recomputes S and dP in both kernels, 7 GEMMs against
  FA2's 5.

## MOSS-Image

**Ported (`src/train_int8_moss.py`), gradient gate passed; A/B not run.** On MOSS's block shapes int8 runs 2.85x forward and 2.72x dX; the
converted linears are 47% of the default compiled step (895 of 1902 ms, swap 36, budget
0.5), a predicted 1.43x if the H2D block swap stays hidden. After int8 a swapped block's
forward (~7.6 ms) is about one 8.5 ms block move, so the swap/budget knee moves down. The
offloader moves `.weight` only ([report](reports/moss_scoping.md)).

| work | result |
|---|---|
| int8 base trainer, swap 0 + checkpointing | 1.44–1.49 s/step, 1.57x bf16 on the same plan, 1.28x the bf16 default ([report](reports/moss_scoping.md)) |
| real-gradient error | int8 `full` at 0.53x / 1.17x / 0.98x bf16's distance from fp32 at σ 0.1 / 0.5 / 0.9; MOSS's bf16 gradient is itself far from fp32 in blocks 0–27. Prediction error 1.24–1.46x bf16's, weight term 57–60% of the excess; late blocks 28–41 at 1.3–3.1x, all weight term ([report](reports/moss_grad_error.md)) |
| bf16 precision arms | an fp32 residual stream around bf16 GEMMs cuts bf16's prediction error 2.4–3.5x (loss = fp32) but leaves the early-block gradient error; turning off reduced-precision reduction helps at σ 0.1 and hurts at 0.9 ([report](reports/moss_grad_error.md#bf16-precision-arms)) |
| offline GPTQ base, t2i + i2i | held-out excess prediction error −28–35% (t2i), −31–32% (i2i); blocks 35–41 at σ 0.5 from 2.59x to 1.20x bf16; with an fp32 stream, below today's bf16 prediction error at every σ tested. No LoRA trained on it, no A/B ([report](reports/moss_gptq.md)) |
| `wq_t` streaming (`--int8 stream`) | same gradients, peak 13.58 → 9.36 GB with checkpointing, step cost ≤ 0.03 s; budgets 0.05–0.4 all 1.33–1.37 s/step, so raising the budget buys no speed; the same-seed gradient norm moves up to 5x in bf16 too ([report](reports/moss_wqt_stream.md)) |
| int8 generation (`src/infer_int8_moss.py`) | 1024² 50 steps at 11–15 s, 6.3 GiB (bf16: 21.6 s, 10.7 GiB); a full-size i2i reference on an fp32 stream fits resident at 11.2 GiB ([report](reports/moss_gptq.md#inference)) |
