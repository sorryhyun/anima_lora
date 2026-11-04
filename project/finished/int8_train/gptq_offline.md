# Offline GPTQ for the int8 base: a later correction plan (2026-10-30)

**Status: deferred.** Phases 2–3 run on round-to-nearest (RTN) weights as planned. This
plan comes into play only if phase 3's A/B shows samples visibly worse than bf16.
Phase 1b already passed its revised gate with RTN, so this plan does not block the line.

## Question

Should the DiT's int8 weights be GPTQ-calibrated offline instead of rounded to nearest at
load? This is the same approach that built the MOSS TE int8 file.

## What the MOSS TE measured

Metric: TE-embedding relative L2 against bf16, all prompt categories. Sources:
`_archive/bench/moss_te_fp8/results/20261021-1258-gptq_text/` and
`20261022-1300-int8_convrot_text/`.

| arm | RTN | GPTQ | change |
|---|---|---|---|
| fp8 | 0.0967 | 0.0467 | −52% |
| int8 ConvRot, weight-only (W8) | 0.0551 | 0.0351 | −36% |
| int8 ConvRot, W8A8 | 0.0672 | 0.0579 | −14% |

- GPTQ fits weights to bf16-activation statistics, so it removes only the weight term.
  With RTN, the weight term dominates (W8 0.0551 of W8A8 0.0672). After GPTQ, the
  activation term dominates (W8 0.0351 against W8A8 0.0579).
- fp8's large gain comes from its 3-bit mantissa, which gives a large weight error.
  Rotated per-row int8 already starts at a step of about 4σ/127.
- Caveat: the 1022 run's `bf16/all` row is 0.0285 rather than 0, and it does not record
  which GPTQ flags built `int8_gptq`. Its absolute numbers are not strictly comparable
  to the 1021 run's.
- Arms `int8_smooth`, `int8_act` and `int8_smooth_act` are defined in
  `_archive/bench/moss_te_fp8/te_int8_act_compare.py` but have no results, so the
  activation-side options are unmeasured.

Applied to this line, and assuming the DiT gains the TE's W8A8 −14%, phase 1b's "int8
gradient error 1.3–1.5x bf16's" would move to about 1.3x.

## Calibrating to the int8 activations (`--act_aware`)

The int32 accumulation is exact. The information is lost earlier, when the activation is
rounded per token to int8 (X → X̃). Calibrating "to the int32 result" therefore means
fitting against X̃. `project/finished/moss_lora/src/te_gptq.py --act_aware` already does
this:

- Objective: `‖X̃ Ŵᵀ − X Wᵀ‖²`. The input is the int8-rounded X̃, and the target is the
  bf16 output.
- Least-squares weight `W* = W·Gᵀ·(H̃ + λI)⁻¹`, with `G = X̃ᵀX` and `H̃ = X̃ᵀX̃`. GPTQ
  then rounds W* against H̃.
- λ is `--ls_damp` (1e-4). At 1e-2 the ridge over-shrinks W*, and the W8A8 error comes
  out about 6x RTN's.

**Limit.** The rounding error e = X − X̃ is close to noise uncorrelated with X, so
`G ≈ H` and `H̃ ≈ H + D`, where D is the rounding-noise variance. That gives
`W* ≈ W·H(H + D)⁻¹`, a ridge shrinkage (Wiener filter). It corrects only directions whose
activation variance is comparable to the rounding noise. In high-energy directions about
`W·D·Wᵀ` of error remains, because the weight sees only X̃ and cannot recover what the
rounding lost.

Reducing e itself takes one of these:

- **SmoothQuant**: move activation range into the weight. The MOSS TE folds the scales
  into its RMSNorms. In Anima the block input passes through adaLN modulation first, so
  where to fold them is still open.
- **Outlier channels in bf16** (LLM.int8 style): this splits the kernel path.
- **Finer activation scales**: not possible with the current kernel, since comfy_kitchen
  scales per token by amax.

## Plan

1. **W8A16 arm** in `src/grad_error.py`: int8 weights with bf16 activations. This splits
   phase 1b's forward error into a weight term and an activation term. About 10 min.
2. **Weight term dominates**: GPTQ the forward copy `wq` only. Take calibration
   activations over several σ, and reuse `te_gptq.py`'s `int8_convrot` grid, which runs
   GPTQ in the group-Hadamard basis (`W·R`, `R·H·R`) that the kernel uses. Leave `wq_t`
   on RTN: phase 1b found `fwd` ≈ `full`, so the dX quantization adds nothing measurable.
3. **Activation term dominates**: compare per-linear W8A8 output error over the 168
   converted linears with four arms: RTN, GPTQ, act-aware, and smooth + act-aware.
   `w8a8_err` computes it from the H matrices alone, so no training is needed.
4. Only if step 2 or 3 cuts the per-layer error by a meaningful amount: re-run phase 1b's
   gradient error, and then phase 3's A/B with the calibrated weights. That run needs a
   quantized cache file, which v0 does not write.

Calibration has to cover several σ, because DiT activation distributions shift strongly
with σ.
