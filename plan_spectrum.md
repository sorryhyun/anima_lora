# Spectrum Integration Plan for Anima

## Status

Two fixes applied to `library/spectrum.py`:

1. **`_taus` domain mapping** — Changed from hardcoded `[0, 50]` affine to
   `2*(step/total_steps) - 1`, giving full `[-1, 1]` Chebyshev domain coverage
   (was 58% for 30 steps). Matches the ComfyUI-Spectrum-sdxl approach.

2. **Warmup window inflation** — `curr_ws` now only increments post-warmup,
   matching the reference Spectrum repo. Previously all 5 warmup steps inflated
   `curr_ws` from 2.0 to 5.75, causing the first post-warmup prediction to
   extrapolate across 4 steps with zero DOF.

Schedule comparison (window_size=2.0, flex_window=0.75, warmup=5, 30 steps):
```
Before: actual at {0,1,2,3,4,9,15,22}        =  8/30 (3.75x, first skip=4)
After:  actual at {0,1,2,3,4,6,8,11,15,20,25} = 11/30 (2.73x, first skip=1)
```

### Key finding from reference repo

The paper describes `g(t) = 2t - 1` on diffusion timestep `t in [0, 1]`, but
the actual Spectrum repo uses **step indices** as the independent variable, not
timestep values. Both the official repo and ComfyUI-Spectrum-sdxl pass step
index `i` to `update(i, feat)` / `predict(i)`. The `[0, 50]` domain was
designed for step index ranges (typical num_steps is 20-50).

---

## Remaining improvements (not yet implemented)

### Community-recommended settings

From ComfyUI-Spectrum-sdxl (tuned for speed + sharpness, 25-step baseline):
```
w: 0.30          # heavier Taylor, less Chebyshev
m: 3             # lower degree — fewer coefficients, less overshoot
lam: 0.1
window_size: 2
flex_window: 0.25   # much slower window growth than default 0.75
warmup_steps: 6     # DiT models may need 8-10
stop_caching_step: total_steps - 3   # force actual forwards for last ~3 steps
```
Key differences from defaults: lower `w` (0.3 vs 0.5), lower `m` (3 vs 4),
much slower `flex_window` (0.25 vs 0.75), and explicit stop-caching near the
end. Currently applied in `Makefile` `test-spectrum` target (except
stop_caching_step, not yet wired).

---

### Taylor blending weight tuning

Current `w=0.5` blends 50% Chebyshev + 50% first-order Taylor. The reference
repo code defaults to `w=0.5`, but the official config (`configs/algo/spectrum.yaml`)
uses `w=1.0` (pure Chebyshev). The ComfyUI impl defaults to `w=0.6`. Worth A/B
testing `w` in {0.3, 0.5, 0.6, 0.8, 1.0} to find the sweet spot for Anima.

### ~~Residual calibration~~ ✅ Implemented

Implemented as `--spectrum_calibration` (default 0.0, disabled). See `spectrum.md`.

### ~~Delta prediction~~ ❌ Tested — not beneficial

Predict `feat(i) - feat(i_last_actual)` instead of absolute features, then
reconstruct: `feat_pred = feat_last_actual + delta_pred`.

**Result:** Implemented and A/B tested (2026-04-03). Tried both raw deltas and
per-step rate normalization, with Taylor blending fixed to operate in absolute
space. In all variants, delta mode produced slightly worse quality than absolute
mode (visible as subtle artifacts similar to reducing warmup by ~1 step), while
absolute mode already achieved near-normal-inference quality on the test prompt.

**Why it didn't help:** The absolute feature trajectory is already smooth enough
for degree-3 Chebyshev to fit well. The theoretical concern about AdaLN gating
causing scale-envelope issues didn't materialize — the polynomial handles the
trajectory without needing delta factoring. Additional issues:
- First observation must be skipped (no prior), losing one data point from warmup
- Post-warmup deltas span variable step gaps, requiring rate normalization
- Compounding reconstruction error (even if bounded per skip window)

---

## Validation plan

1. **A/B test:** Generate with seed 42 using `make test` (baseline) and
   `make test-spectrum`. Visual comparison of same prompt, same seed.
2. **NFE sweep:** Test different `window_size` / `flex_window` combos and
   compare quality vs speedup.
3. **`w` sweep:** Test `w` in {0.5, 0.6, 0.8, 1.0}.
4. **Delta mode:** If implemented, compare absolute vs delta prediction at
   same schedule settings.
