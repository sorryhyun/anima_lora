# Proposal: SM120 Attention Optimization for Anima LoRA Training

## Summary

Optimize SM120 flash attention for Anima LoRA training on RTX 5060 Ti.

Phase 0 profiling reveals **backward attention is the dominant bottleneck** (76% of
attention time, 3x slower than forward). The priority is improving the backward path.
MXF4 forward-only work is deprioritized — it can only touch 24% of attention time.

## Phase 0 results

Measured with `bench/phase0_profile.py` on RTX 5060 Ti (SM 12.0, 36 SMs).
Shapes: B=2, H=32, D=64. FA4 CuTeDSL backend, `num_stages=1`.

### Per-layer timing

| | Forward | Backward | Total | Bwd/Fwd |
|---|---|---|---|---|
| Self-attn (4096×4096) | 6.05 ms | 18.44 ms | 24.49 ms | 3.05x |
| Cross-attn (4096×256) | 0.52 ms | 2.16 ms | 2.69 ms | 4.15x |

### Per training step (24 self-attn + 24 cross-attn layers)

| Component | Time | Fraction |
|---|---|---|
| Self-attn total | 587.7 ms | 90.1% |
| Cross-attn total | 64.5 ms | 9.9% |
| **Total attention** | **652.2 ms** | ~22% of 3 s/step |
| Forward only | 157.7 ms | 24.2% of attention |
| **Backward only** | **494.5 ms** | **75.8% of attention** |

### Forward `num_stages` sweep

| num_stages | Self-attn fwd | Cross-attn fwd | Notes |
|---|---|---|---|
| **1 (current)** | **6.06 ms** | **0.39 ms** | Best |
| 2 | 7.87 ms (+30%) | 0.52 ms (+33%) | Occupancy loss from SMEM |
| 3 | CUDA error | — | Exceeds 99 KB SMEM |

**Conclusion:** `num_stages=1` is already optimal for SM120's 99 KB SMEM. More
pipelining stages increase SMEM usage and kill occupancy.

### Bugs found

- **`atomicrmw` API mismatch**: FA4 backward called `nvvm.atomicrmw(res=T.f32(), ...)`
  but CUTLASS DSL 4.4.2 expects positional args `(op, ptr, a)`. Fixed in `utils.py`.

## Hardware constraints

**GPU**: RTX 5060 Ti (SM 12.0), 36 SMs, 3120 MHz boost, 16 GB GDDR7

**SMEM**: 99 KB per SM (vs 163 KB on SM80, 227 KB on SM90). Binding constraint.
No TMEM (SM100+ only). TMA hardware available for loads.

**MMA**: SM80-era `mma.sync.aligned` (BF16) for current kernels. Native SM120
`mma.kind::mxf4` available but unexploited.

**CUDA**: 13.0 installed, 13.1 at `/usr/local/cuda-13.1/`.
CUTLASS DSL 4.4.2.

## Strategy: backward-first optimization

### Phase 1: Backward profiling & tuning

The SM120 backward kernel (`FlashAttentionBackwardSm120`, subclass of SM80) has
several conservative choices that may have headroom:

| Parameter | Current | Possible |
|---|---|---|
| Block size | 64×64 (fixed) | 128×128 for D≤64 (SMEM allows) |
| num_stages_Q | 2 (D≤64) | Sweep 1-3 |
| num_stages_dO | 2 (D≤64) | Sweep 1-3 |
| AtomLayout{MSdP,NdKV,MdQ} | All 4 | Sweep 1, 2, 4 |
| V_in_regs | False | Try True |
| dQ_single_wg | False | Try True |
| 2-CTA instructions | Disabled | SM120 supports it |

Work items:
- [ ] Extend `bench/phase0_profile.py` to sweep backward configs
- [ ] Add `FA4_SM120_BWD_*` env var overrides to `interface.py` (like we did for fwd stages)
- [ ] Profile backward SMEM usage at each config
- [ ] Identify which backward GEMM (SdP, dKV, dQ) dominates

### Phase 1.5: Backward kernel improvements

Based on Phase 1 profiling:
- If block size is the bottleneck: test 128×128 tiles
- If SMEM is the bottleneck: optimize buffer reuse (similar to TMA kernel's `alias_sO_with_sQ`)
- If compute is the bottleneck: investigate whether MXF4 backward GEMMs help (but this is hard)

### Phase 2: TMA-optimized backward

A TMA-optimized forward kernel exists (`flash_fwd_sm120_tma_optimized.py`) but was
reverted. The forward gains were modest because forward isn't the bottleneck.
A TMA-optimized **backward** kernel with dedicated DMA warps could have much more
impact since backward is 3x slower and more memory-bound.

## MXF4 forward (deprioritized)

Phase 0 shows a forward-only MXF4 kernel saves at most ~105 ms/step (3x speedup)
or ~3.5% of training time. This doesn't justify the engineering cost unless backward
is also addressed.

MXF4 forward remains a valid research track with explicit gates:

1. **Gate 1**: Standalone MXF4 GEMM — does packing overhead beat BF16 by ≥1.5x?
2. **Gate 2**: QK-only prototype — real speedup in attention control flow?
3. **Gate 3**: Hybrid kernel (MXF4 QK + BF16 PV) — end-to-end forward ≥1.5x?
4. **Gate 4**: Full MXF4 — only if Gate 3 passes and P quantization is safe

This track is worth revisiting **after** backward optimization, when forward may
become a larger fraction of the remaining time.

### MXF4 design notes

MXF4 uses `e2m1` (4-bit) with `ue8m0` block scaling (1 scale per 32 elements).
FP32 accumulator. Representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.

SMEM budget: ~43 KB single-buffered (tile 64×64, D=64), ~59 KB double-buffered.
Fits 99 KB.

Key risk: quantizing softmax output P to MXF4 for P·V (high numerical risk,
deferred to Gate 4).

## Prior art

- **TMA-optimized forward kernel**: Implemented, correctness-verified (max diff < 0.004),
  then reverted. Used 160 threads (1 DMA + 4 MMA warps), triple-buffered TMA.
  Revert reason undocumented — likely modest gains since forward isn't the bottleneck.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Backward tuning yields marginal gains | Medium | High | Profile first, kill early |
| SMEM limits backward tile size | Medium | Medium | 99 KB fits 128×128 for D=64 |
| MXF4 packing overhead negates MMA gain | Medium | Low (deprioritized) | Gate 1 measures this |
| CUTLASS DSL API instability | Medium | Medium | Pin version, patch as needed |

## Expected outcomes

| Scenario | Attention speedup | Step time impact |
|---|---|---|
| Backward tuning (tile/stage sweep) | 1.1-1.3x | 2-7% faster |
| Backward TMA optimization | 1.3-1.7x | 7-15% faster |
| + MXF4 forward (post-backward) | 1.5-2.0x | 10-20% faster |
