# Spectrum — Inference Acceleration

Training-free diffusion sampling acceleration via **Chebyshev polynomial feature forecasting**.

Paper: [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) (Han et al., CVPR 2026, Stanford/ByteDance)

Reference implementation: `Spectrum/` (cloned from upstream repo)

## Quick start

```bash
make test-spectrum   # same as make test but with Spectrum enabled
```

Or add `--spectrum` to any `inference.py` invocation:

```bash
python inference.py --spectrum \
    --spectrum_window_size 2.0 --spectrum_flex_window 0.25 \
    --spectrum_w 0.3 --spectrum_m 3 --spectrum_lam 0.1 \
    ...  # other inference args
```

## How it works

Standard diffusion runs the full DiT (28 transformer blocks) at every denoising step. Spectrum observes that block outputs are smooth functions of the timestep, so most steps can be **predicted** instead of computed.

### Per-step decision

```
step i
├─ actual forward?
│  YES → run full DiT, capture block output via hook, update Chebyshev fit
│  NO  → predict block output from polynomial, run only final_layer + unpatchify
```

### Adaptive window schedule

The window size N starts at `window_size` and grows by `flex_window` after each actual forward:

1. **Warmup** (steps `0 .. warmup-1`): always run full forward to seed the forecaster.
2. **Adaptive**: actual forward every `floor(N)` cached steps; N += α after each forward.

With 30 steps and defaults (N=2, α=0.25, warmup=6): more actual forwards for quality, moderate speedup.

### Chebyshev forecasting

For each cached step, features are predicted by:

1. Mapping step indices to τ ∈ [-1, 1].
2. Building Chebyshev basis [T₀(τ), T₁(τ), ..., Tₘ(τ)] via recurrence.
3. Ridge regression: (XᵀX + λI)C = XᵀH (solved via Cholesky).
4. Prediction: h* = [T₀(τ*), ..., Tₘ(τ*)] · C.

Optionally blended with first-order Taylor (Newton forward difference) via weight `w`.

### Cached step fast path

On a cached step, only these layers run:

- `t_embedder` — timestep MLP (tiny)
- `final_layer` — LayerNorm + AdaLN + linear projection
- `unpatchify` — reshape to pixel space

All 28 transformer blocks (self-attn, cross-attn, MLP × 28) are skipped.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--spectrum` | off | Enable Spectrum acceleration |
| `--spectrum_window_size` | 2.0 | Initial window N |
| `--spectrum_flex_window` | 0.25 | Window growth α per forward |
| `--spectrum_warmup` | 6 | Steps that always run full forward |
| `--spectrum_w` | 0.3 | Chebyshev/Taylor blend (1.0 = pure Chebyshev) |
| `--spectrum_m` | 3 | Number of Chebyshev basis functions |
| `--spectrum_lam` | 0.1 | Ridge regression regularization λ |
| `--spectrum_stop_caching_step` | -1 | Force actual forwards from this step onward (-1 = auto: total_steps - 3) |
| `--spectrum_calibration` | 0.0 | Residual calibration strength (0.0 = disabled) |

### Residual calibration

On each actual forward, the forecaster's prediction error is measured: `residual = actual_output - predicted_output`. On subsequent cached steps, this residual is added back as a bias correction: `prediction + residual * calibration_strength`.

This captures systematic prediction error that the polynomial fit misses. Inspired by ComfyUI-Spectrum-sdxl's calibrated node. Try `--spectrum_calibration 0.5` as a starting point.

### Tuning for more aggressive speedup

```bash
# ~5x speedup (fewer forwards, may reduce quality)
--spectrum_window_size 2.0 --spectrum_flex_window 3.0 --spectrum_w 0.7
```

Higher `flex_window` → faster window growth → fewer forwards. Increase `w` toward 1.0 when pushing more aggressive acceleration to rely more on the Chebyshev fit.

## Implementation

| File | Role |
|------|------|
| `networks/spectrum.py` | Anima integration: `SpectrumPredictor`, `spectrum_denoise()`, `_spectrum_fast_forward()` |
| [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) | ComfyUI custom node: drop-in KSampler replacement |
| `Spectrum/src/utils/basis_utils.py` | Core algorithm: `ChebyshevForecaster`, ridge regression, polynomial evaluation |

The integration uses `register_forward_pre_hook` on `Anima.final_layer` to capture block outputs without modifying the model class. Separate forecasters are maintained for conditional and unconditional (CFG) passes.

### ComfyUI custom node

The `SpectrumKSampler` node (`KSampler (Spectrum)`) is a drop-in KSampler replacement. It works with any ComfyUI sampler (er_sde, euler, dpm, etc.) because the caching logic is transparent to the sampling loop.

**Wiring:** The node installs a `model_function_wrapper` on a cloned model. ComfyUI's sampling pipeline calls this wrapper once per step with both cond and uncond batched together (via `calc_cond_batch`). The wrapper decides actual vs cached per step by tracking sigma changes:

```
ComfyUI sampling loop (any sampler)
  └─ model(x, sigma)
       └─ sampling_function()  — handles CFG
            └─ calc_cond_batch()  — batches cond+uncond into one forward
                 └─ model_function_wrapper(apply_model, args)  ← our hook
                      ├─ actual step: apply_model() → hook captures features → update forecasters
                      └─ cached step: predict features → t_embedder + final_layer + unpatchify → calculate_denoised
```

- `args["cond_or_uncond"]` tells us which batch elements are cond (0) vs uncond (1) — separate forecasters per type.
- Step transitions are detected by sigma value changes.
- `calculate_denoised(sigma, v_pred, x)` converts the fast-forward velocity output to denoised x, matching the normal model path.
- Chains with existing wrappers (FlashAttention4, TorchCompile) since those operate at different levels (`transformer_options` and `WrappersMP.APPLY_MODEL` respectively).
