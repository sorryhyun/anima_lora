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

### Residual calibration

From ComfyUI-Spectrum-sdxl's calibrated node: on each actual forward, compute
`residual = actual_output - last_prediction` and store it. On cached steps, add
`residual * calibration_strength` to the Chebyshev+Taylor prediction. A simple
bias correction that captures systematic prediction error.

Default `calibration_strength=0.5`. Could be added as `--spectrum_calibration`.

### Delta prediction

Predict `feat(i) - feat(i_last_actual)` instead of absolute features, then
reconstruct: `feat_pred = feat_last_actual + delta_pred`.

**Why this matters for Anima specifically:**

Anima's 28 blocks each apply timestep-dependent **multiplicative gating**:
```
x = x + gate_sa * self_attn(norm(x) * (1+scale) + shift)
x = gate_ca * cross_attn(...)  + x
x = x + gate_mlp * mlp(norm(x) * (1+scale) + shift)
```
where `(shift, scale, gate)` come from `adaln_modulation(emb)`. The gate values
change with timestep, so absolute feature magnitude varies significantly across
the denoising trajectory — early steps (high noise, large gates) produce features
with different scale than late steps (refinement, small gates).

This timestep-correlated variance makes absolute features harder for Chebyshev
to fit: the polynomial must simultaneously capture the smooth *signal evolution*
and the scale *envelope*. Deltas factor out the envelope — they represent
incremental changes between adjacent actual forwards, which should be smoother
and have smaller dynamic range.

**Implementation sketch:**

```python
class ChebyshevForecaster:
    def __init__(self, ..., use_delta: bool = False):
        ...
        self.use_delta = use_delta
        self._last_h_flat: Optional[torch.Tensor] = None  # last actual feature

    def update(self, t: float, h: torch.Tensor) -> None:
        h_flat, shape = _flatten(h)
        if self.use_delta and self._last_h_flat is not None:
            store = h_flat - self._last_h_flat  # delta
        else:
            store = h_flat
        self._last_h_flat = h_flat.clone()
        # ... append store to H_buf as before ...

    def predict(self, t_star) -> torch.Tensor:
        # ... Chebyshev predict as before → gives delta_pred ...
        if self.use_delta:
            return _unflatten(self._last_h_flat + h_flat, self._shape)
        return _unflatten(h_flat, self._shape)
```

**Trade-offs:**

| | Absolute (current) | Delta |
|---|---|---|
| What polynomial fits | Full feature trajectory | Incremental changes |
| Dynamic range | Large (gate-scaled) | Smaller (differences) |
| Error behavior | Independent per step | Compounds from last actual |
| Memory overhead | None | +1 tensor per forecaster |
| Failure mode | Over-smoothed features | Drift from accumulation |

The compounding risk is bounded by the schedule: each actual forward resets the
base feature, so drift only accumulates within one skip window. With the fixed
schedule (first skip = 1 step), this is minimal.

**When to try:** After validating the current fixes produce acceptable quality.
Add as `--spectrum_delta` flag, A/B test against absolute mode.

---

## Validation plan

1. **A/B test:** Generate with seed 42 using `make test` (baseline) and
   `make test-spectrum`. Visual comparison of same prompt, same seed.
2. **NFE sweep:** Test different `window_size` / `flex_window` combos and
   compare quality vs speedup.
3. **`w` sweep:** Test `w` in {0.5, 0.6, 0.8, 1.0}.
4. **Delta mode:** If implemented, compare absolute vs delta prediction at
   same schedule settings.
