# HydraLoRA + ReFT — nsys-driven optimization, 2026-05-03

Source artifacts: `output/nsys/`
- `profile.nsys-rep` — open with Nsight Systems GUI for the full timeline
- `profile.sqlite` — queryable kernel/API tables
- `profile_cuda_gpu_kern_sum.txt` — per-kernel rollup (most numbers below)
- `profile_cuda_api_sum.txt`, `profile_cuda_kern_exec_sum.txt`, `profile_nvtx_kern_sum.txt`,
  `profile_cuda_gpu_mem_size_sum.txt`, `profile_cuda_gpu_mem_time_sum.txt`

## What was profiled

`make lora` default stack (HydraLoRA + OrthoLoRA + T-LoRA + ReFT) under
`PROFILE_STEPS=...`, captured 31 training steps. Config in effect:
`network_dim=48`, `num_experts=12`, `reft_dim=48`, `reft_layers=last_8`.

Total GPU kernel time across the capture: **33.7 s** (~1.09 s/step).

## Where the time goes (baseline)

| Bucket | GPU time | % | Source |
|---|---|---|---|
| Small bf16 GEMMs (`64×64×32` cutlass `_relu_bf16`, nn/tn/nt) | 16.5 s | **49 %** | LoRA / Ortho / Hydra / ReFT projections |
| Flash-attn fwd+bwd (incl. `dot_do_o`, `convert_dq`) | 10.0 s | 30 % | base DiT — expected |
| Triton fused layer-norm + GELU | 2.7 s | 8 % | base DiT epilogues |
| LU + TRSM (Cayley `solve`) | 1.0 s | 3 % | also breaks the bf16 path |
| Other (router, einsum, copies) | 3.5 s | 10 % | |

API-side: `cudaStreamSynchronize` accounts for 17 s in 489 calls — median
3.4 µs (incidental), but **max 632 ms / stddev 136 ms**, so a small number
of huge syncs dominate. Likely the warm-up `torch.cuda.synchronize()` in
`_profiler_step_begin` plus a dataloader stall; confirm in the GUI by
looking for GPU-idle gaps inside `:step=N` ranges.

## Implemented

### 1. `_eye_r` buffer — `networks/lora_modules/ortho.py`

`OrthoLoRAExpModule.__init__` and `OrthoHydraLoRAExpModule.__init__`
register a non-persistent `_eye_r` buffer (`lora_dim × lora_dim`, fp32).
The forward path reads it directly; broadcasting handles the
`(r,r) + (E,r,r)` addition, so the prior `eye.unsqueeze(0).expand_as(A)`
materialisation step is also gone.

Replaces ~336 `torch.eye` allocations per training step (one per Cayley
call) with zero, and removes the Inductor-fused
`triton_poi_fused_add_eye_permute_sub_2` kernel (10,416 instances in the
capture).

### 2. `_FREQS_CACHE` for sinusoidal σ features — `networks/lora_modules/hydra.py`

Module-global `dict[(half_dim, device), Tensor]` caches the frequency
vector that `_sigma_sinusoidal_features` previously rebuilt on every
call. Exercised per training step from `set_sigma` /
`_set_sigma_feature_cache`. The function runs outside the compiled
forward, so a Python-level dict cache is safe under `compile_mode=full`.

Bit-equivalent to a fresh recompute (verified `0.0` max diff).

### 3. Batched Cayley within OrthoLoRA / OrthoHydra — `ortho.py`

Both forwards now do a single `torch.linalg.solve` per call by
concatenating the skew matrices:

- `OrthoLoRAExpModule.forward`: `stack([S_q, S_p])` → `(2, r, r)` →
  one solve → split into `R_q`, `R_p`.
- `OrthoHydraLoRAExpModule.forward`:
  `cat([S_q.unsqueeze(0), S_p_eff])` → `(E+1, r, r)` → one solve →
  `R_q = R[0]`, `R_p = R[1:]`. The expert-warmup mask
  (`S_p_eff = S_p * mask + S_p.detach() * (1 − mask)`) is hoisted to the
  top of the forward so its output flows into the same batched solve.

Halves the LU / TRSM launch count per OrthoHydra module per fwd: one
batched `getrf_pivot` + `batch_trsm_left` per module instead of two
(per-module + per-expert). Per-step `getrf_pivot` calls drop from ~336
to ~168 (capture-wide: 10,416 → ~5,200).

Verified parity vs. the previous separate `_cayley(S_q)` /
`_cayley(S_p)` path to ~1e-7 (bf16-clean); gradients still reach `S_q`
and `S_p`.

`OrthoLoRAExpModule._cayley` and `OrthoHydraLoRAExpModule._cayley` are
kept as static methods because `networks/lora_save.py` calls them at
save-time SVD distillation.

## Deferred

### A. Cayley → `matrix_exp(-2A)`

Cayley is the [1, 1] Padé approximant of `exp(-2A)` for skew-symmetric
A; expansions agree to second order and diverge at A³. Both are valid
skew → orthogonal maps but **`S_p` / `S_q` parameterise a different
rotation under each**. Replacing the solve with `matrix_exp` is a
parameterisation change, not a numerical equivalence — existing
checkpoints would need re-mapping or re-training.
`torch.linalg.matrix_exp` does run in bf16 (verified empirically), and
the ~1.0 s LU/TRSM bucket would collapse into a couple of GEMMs. Worth
revisiting if a parameterisation switch is otherwise acceptable.

### B. Cross-module Cayley orchestration

The original audit suggestion was "concat A across modules" into a
single network-wide solve. Conflicts with the per-module compile
boundary: caching R as a buffer kills autograd; passing fresh R tensors
through Python attrs forces cudagraph re-record on every step (same
failure mode that `set_sigma` works around with in-place buffer copies,
which is no help here because we need autograd through R).
Within-module batching captures most of the launch-count win;
cross-module would require either an in-graph pre-step (orchestrated
network walk inside the compiled DiT forward) or moving Cayley
computation off the autograd graph. Re-evaluate after the within-module
fix is benched.

### C. ReFT output-projection layout — `networks/lora_modules/reft.py:120`

```python
edit = torch.nn.functional.linear(delta, self.rotate_layer.weight.T)
```

`weight.T` is a non-contiguous view; cublas picks a transposed-input
variant. Replacing with `delta @ self.rotate_layer.weight` is
bit-equivalent (`F.linear(x, w) = x @ w.T`, so
`F.linear(δ, R.T) = δ @ R`). × 8 ReFT blocks × every step. Drop-in;
not yet applied.

### D. OrthoHydra expert-warmup mask outside warmup — `ortho.py`

When `_expert_grad_mask` is all-ones (the default outside warmup, i.e.
most of training) `S_p * 1 + S_p.detach() * 0` is autograd-equivalent
to `S_p`. Three fp32 elementwise ops per Ortho module per fwd. Small
in absolute terms (~168 modules × 31 steps × 3 ops in this capture);
a short-circuit on a known-warmup-inactive flag would remove them.
Note that the mask now feeds the batched Cayley solve (§Implemented 3),
so any short-circuit needs to keep `S_p_eff` shape-stable for the cat.

### E. Small-tile cutlass dominance — structural

49 % of GPU time is `64×64×32` bf16 tiles. Each `K=48` GEMM has low
arithmetic intensity. Two avenues:

- For HydraLoRA's per-expert P (`ortho.py:408-413`), `P_bases @ R_p`
  is `E=12` separate `(out, r) × (r, r)` GEMMs per fwd. Folding into
  a precomputed `P_eff` is straightforward at inference; **not free
  under autograd during training** because `R_p` depends on `S_p`, and
  a cached `R_p` across microbatches breaks the backward graph. A
  correct training variant would need a manual gradient path through
  cached `R_p`.
- Stop stacking everything by default — bench each adapter family's
  marginal win and drop the ones that don't pull weight. The current
  default composes LoRA + OrthoLoRA + T-LoRA + ReFT on every block.

### F. cudaStreamSynchronize — 17 s in 489 calls

Median 3.4 µs, max 632 ms, stddev 136 ms — a few huge syncs dominate.
Likely the warm-up `torch.cuda.synchronize()` in `_profiler_step_begin`
plus a dataloader stall. Confirm in the GUI: look for "GPU idle" gaps
inside `:step=N` ranges and correlate with the Python-sampling track
(`--python-sampling=true` was on). Not yet re-profiled.

## Status

| Item | State |
|---|---|
| `_eye_r` buffer (§Impl 1) | done |
| `_FREQS_CACHE` (§Impl 2) | done |
| Within-module batched Cayley (§Impl 3) | done |
| `matrix_exp(-2A)` swap (§A) | deferred — parameterisation change |
| Cross-module Cayley (§B) | deferred — autograd vs. cudagraph |
| ReFT `weight.T` (§C) | deferred — drop-in, not yet applied |
| Expert-warmup mask short-circuit (§D) | deferred — small impact |
| Adapter stack pruning (§E) | deferred — bench-driven |
| cudaStreamSynchronize hunt (§F) | deferred — needs GUI dive |

Re-profile after the implemented fixes to refresh the bucket table.
