# Nsys analysis — HydraLoRA + ReFT, 2026-05-03

Source artifacts: `output/nsys/`
- `profile.nsys-rep` — open with Nsight Systems GUI for the full timeline
- `profile.sqlite` — queryable kernel/API tables
- `profile_cuda_gpu_kern_sum.txt` — per-kernel rollup (this is what most of the numbers below come from)
- `profile_cuda_api_sum.txt`, `profile_cuda_kern_exec_sum.txt`, `profile_nvtx_kern_sum.txt`,
  `profile_cuda_gpu_mem_size_sum.txt`, `profile_cuda_gpu_mem_time_sum.txt`

## What was profiled

`make lora` default stack (HydraLoRA + OrthoLoRA + T-LoRA + ReFT) under
`PROFILE_STEPS=...`, captured 31 training steps. Config in effect:
`network_dim=48`, `num_experts=12`, `reft_dim=48`, `reft_layers=last_8`.

Total GPU kernel time across the capture: **33.7 s** (~1.09 s/step).

## Where the time goes

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
`_profiler_step_begin` plus a dataloader stall; worth confirming in the
GUI by looking for GPU-idle gaps inside `:step=N` ranges.

## Concrete inefficiencies

### 1. Cayley via `torch.linalg.solve` — `networks/lora_modules/ortho.py:107` and `:318`

```python
return torch.linalg.solve(eye + A, eye - A)
```

10,416 `getrf_pivot` + 36,456 `batch_trsm_left` + 24,304 `unpack_pivots` +
lots of `ipiv_*` / `create_pivot_v2` / `xxtrf4_set_info_ker` machinery,
all in **fp32** on `r=48` (and batched `(E=12, 48, 48)`) matrices, fired
on every fwd of every Ortho module (~336 LU/step).

For skew-symmetric `A`, `(I-A)(I+A)⁻¹` is a Padé-1 approximation of
`exp(-2A)`. Replacing with `torch.linalg.matrix_exp(-2*A)` (Padé +
scale/square, all GEMMs) eliminates the entire LU/TRSM pipeline, runs in
bf16, and stays inside cudagraphs. Near-drop-in for rotations of this size.

Even without changing the math, batching all module Cayleys into one big
solve (concat `A` across modules) would amortize the launch overhead.

### 2. `torch.eye(...)` allocated inside `_cayley` every forward — `ortho.py:106` and `:315`

~336 fresh small fp32 tensors per step, plus `expand_as(A)` for the
batched 3D path forces a non-contiguous materialization before the
solve. Register a per-rank `_eye_r` buffer at `__init__`; for the
batched path skip `expand_as` (cuBLAS broadcasts identity).

### 3. `arange` kernels: 60,760 instances in 31 steps (~2 K/step)

Sources:
- `_sigma_sinusoidal_features` — `hydra.py:25` constructs
  `torch.arange(half_dim, ...)` every call.
- The repeated `torch.eye` construction in `_cayley`.

`freqs = exp(-log(10000) * arange / half_dim)` never changes; register it
as a buffer at module init.

### 4. ReFT output projection layout — `reft.py:120`

```python
edit = torch.nn.functional.linear(delta, self.rotate_layer.weight.T)
```

`weight.T` is a non-contiguous view; cublas picks a transposed-input
variant. Either `delta @ self.rotate_layer.weight` or storing `R`
transposed in the parameter lets cublas pick the natively contiguous
layout. × 8 ReFT blocks × every step.

### 5. OrthoHydra expert-warmup mask runs unconditionally — `ortho.py:404-405`

```python
expert_mask = self._expert_grad_mask.to(self.S_p.dtype).view(-1, 1, 1)
S_p_eff = self.S_p * expert_mask + self.S_p.detach() * (1.0 - expert_mask)
```

When the mask is all-ones (the default outside warmup, i.e. most of
training) this is 3 fp32 elementwise ops per Ortho module per fwd that
are autograd-equivalent to `S_p`. Either short-circuit on
`_warmup_active` or accept it — at 168 modules × 31 steps it shows up.

### 6. Small-tile cutlass dominates because of the rank stack

49 % of GPU time is `64×64×32` bf16 tiles — a structural consequence of
`network_dim=48` + `num_experts=12` + `reft_dim=48` stacked on every
block. Each contributes a K=48 GEMM, so even big M·N matmuls run with
low compute intensity.

Two avenues:
- For HydraLoRA's per-expert P (`ortho.py:408-413`), `P_bases @ R_p` is
  E=12 separate `(out, r)×(r, r)` GEMMs every fwd. Folding `R_p` into a
  precomputed-per-step `P_eff` buffer (recomputed once per optimizer
  step, not per micro-batch) eliminates one batched GEMM tier per module.
- Stop stacking everything by default — measure each adapter's marginal
  win and drop the ones that aren't pulling weight. The current default
  composes all four families on every block.

### 7. cudaStreamSynchronize — 17 s in 489 calls

Median 3.4 µs, max 632 ms, stddev 136 ms — a few huge syncs dominate.
Likely:
- the warm-up `torch.cuda.synchronize()` in `_profiler_step_begin`
- a dataloader stall

Confirm in the GUI: look for "GPU idle" gaps inside `:step=N` ranges and
correlate with the Python-sampling track (`--python-sampling=true` was
on). If there's a recurring per-step gap, that's the lever.

## Suggested order of attack

1. **Replace `_cayley` with `matrix_exp(-2*A)`** — biggest cleanup,
   removes the entire LU/TRSM path and unsticks bf16 through the rotation
   parameterization. Verify with the existing OrthoLoRA bench.
2. **Cache `eye` and sinusoidal `freqs` as buffers** — trivial, kills
   ~2 K arange launches per step.
3. **Profile a leaner stack** (Ortho + ReFT only, or Hydra with fewer
   experts) for comparison — quantifies the small-tile-GEMM cost of the
   12-expert config.
4. **Open `output/nsys/profile.nsys-rep` in the GUI** and look for
   GPU-idle gaps inside `:step=N` to localize the 17 s of sync.
