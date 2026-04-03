# Spectrum for ComfyUI

Training-free diffusion sampling acceleration via **Chebyshev polynomial feature forecasting** ([Han et al., CVPR 2026](https://arxiv.org/abs/2603.01623)). Drop-in KSampler replacement that skips transformer blocks on predicted steps for ~2-3x speedup.

## How it works

Standard diffusion runs the full DiT (all transformer blocks) at every denoising step. Spectrum observes that block outputs are smooth functions of the timestep, so most steps can be **predicted** instead of computed.

On "actual" steps the full model runs and block outputs are captured. On "cached" steps all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` execute, using features predicted from a Chebyshev ridge-regression fit.

### Adaptive window schedule

The window size N starts at `window_size` and grows by `flex_window` after each actual forward:

1. **Warmup** (first N steps): always run full forward to seed the forecaster
2. **Adaptive**: actual forward every `floor(N)` cached steps; N grows after each forward

With 28 steps and defaults: ~**8 actual forwards** out of 28 total steps.

## Usage

Place the **KSampler (Spectrum)** node where you'd normally use a KSampler. It has the same inputs (model, seed, steps, cfg, sampler, scheduler, conditioning, latent) plus Spectrum-specific parameters.

Works with any ComfyUI sampler (Euler, DPM, er_sde, etc.) because caching is handled transparently inside a model function wrapper. Chains with other model wrappers (Flex Attention, Flash Attention 4, etc.).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 2.0 | Initial caching window N |
| `flex_window` | 0.25 | Window growth rate per actual forward |
| `warmup_steps` | 6 | Steps that always run full forward |
| `blend_w` | 0.3 | Chebyshev/Taylor blend weight (1.0 = pure Chebyshev) |
| `cheby_degree` | 3 | Number of Chebyshev basis functions |
| `ridge_lambda` | 0.1 | Ridge regression regularization strength |

### Tuning tips

- **More speedup**: increase `flex_window` (faster window growth = fewer forwards)
- **Better quality**: increase `warmup_steps`, decrease `flex_window`
- **Aggressive acceleration**: `flex_window=1.0`, `blend_w=0.7` (~3-4x speedup)
