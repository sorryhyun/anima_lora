# Spectrum sampler bench

Two complementary tools for debugging Spectrum × sampler × scheduler interactions.

## `analyze_drift.py` — analytical drift simulator

Measures how a Spectrum-sized `denoised` perturbation on cached steps propagates through each sampler's update equations, across different scheduler shapes. **No DiT required** — uses a closed-form Tweedie posterior denoiser for a toy Gaussian data prior so results run in seconds.

```bash
uv run python bench/spectrum/analyze_drift.py                 # full 7×7 sweep with curves + heatmap
uv run python bench/spectrum/analyze_drift.py --auto_stop     # schedule-aware stop_caching_step (recommended)
uv run python bench/spectrum/analyze_drift.py --error_mag 0.05  # larger cache errors
uv run python bench/spectrum/analyze_drift.py --prior_var 0   # legacy x-independent oracle
```

Outputs `drift.png` (per-step curves), `drift_heatmap.png` (final-drift matrix, best cell boxed in red), `drift_summary.csv`, and `drift_curves.json` per run. **`final_drift_mean` is the headline number** — rank cells by it to find best (sampler, scheduler) pair. The heatmap is the fastest way to spot it.

**Closed-loop denoiser.** `denoised(x, σ) = (σ²·target + prior_var·x)/(σ²+prior_var)`. Jacobian w.r.t. x is `prior_var/(σ²+prior_var)` — near 0 at high σ (matches an oracle that ignores x) and → 1 at low σ. That's exactly when er_sde+karras (tiny late-step Δλ) and euler_a (random-walk noise riding feedback) blow up empirically. `--prior_var 0` recovers the old x-independent oracle for sanity checks.

### Headline findings (defaults, prior_var=1, ε=0.02, --auto_stop)

| rank | sampler | scheduler | final_drift | notes |
|------|---------|-----------|-------------|-------|
| 1 (bench) | `dpmpp_2s_a` | `linear_quadratic` | 0.001 | bench-suggested, **not yet validated in real images** |
| 2 (bench) | `dpmpp_2s_a` | `simple` | 0.003 | robust runner-up, all-sampler-friendly scheduler |
| 3 (bench) | `heun` | `linear_quadratic` | 0.013 | corrector self-corrects single-step cache errors |
| **validated** | **`er_sde` (`er_sde_3` in bench)** | **`simple`** | **0.028** | **shipped recommendation in the spectrum node README** |

The bench's top suggestion (`dpmpp_2s_a + linear_quadratic`) is 28× lower drift than the validated combo, but `linear_quadratic` is approximated here as 25%/75% linear-then-quadratic — ComfyUI's actual implementation may differ, and we haven't run the empirical image bench on it. Stick with `er_sde + simple` until that's confirmed.

### Other useful knobs

- `dpmpp_2s_a` row dominates the heatmap — single-step ancestral with corrector. Worth empirical validation.
- `linear_quadratic` column wins for most samplers **except `er_sde_3`** (0.185 vs simple's 0.028) — its quadratic tail has small Δλ that hits er_sde's FD weakness. Scheduler choice depends on sampler.
- `dpmpp_2m_sde` is the worst sampler overall (0.094 even on simple). Combines FD amplification with stochastic compounding. In the FRAGILE set.
- `karras` / `karras_rho3` / `exponential` / `kl_optimal` / `polyexp` all underperform simple by 4–9× on every sampler — uniform-tail schedules don't self-correct.

**What it is good for:**
- Quickly ranking samplers by sensitivity to a given cache-error magnitude.
- Distinguishing scheduler shapes — the heatmap surfaces best/worst combos in one glance.
- Visualizing the finite-difference sawtooth pattern from multistep samplers after each cached step (per-step curves).

**What it does NOT capture:**
- The toy data prior is isotropic Gaussian — there's no thin "natural-image manifold" to fall off, so endpoint error saturates and only paired-trajectory drift discriminates cells.
- Sampler / scheduler implementations are k-diffusion-flavored simplifications, not bit-exact copies of comfy's code (especially `linear_quadratic` and `er_sde_3`). Qualitative ranking holds; absolute numbers should not be compared across runs with different implementations.

For absolute corruption diagnosis you still need the empirical image bench below.

## `bench.py` — empirical image bench (ComfyUI)

Drives ComfyUI's HTTP API to sweep `(sampler, scheduler, spectrum on/off)` over a prompt list and dump image pairs + a CSV for visual/perceptual comparison.

## Prerequisites

- ComfyUI running locally with the `comfyui-spectrum-ksampler` custom node loaded.
- A workflow JSON in **API format** (File → Save (API Format) in ComfyUI) that contains **one** `SpectrumKSamplerAdvanced` node. `workflows/modonly.json` works out of the box.
- Note the **node IDs** for:
  - the SpectrumKSamplerAdvanced (default `19` in modonly.json)
  - the positive `CLIPTextEncode` (default `11`)
  - the negative `CLIPTextEncode` (default `12`)

## Run

```bash
# Default sweep (7 sampler/scheduler cells × 4 prompts × {ref, spec} = 56 generations)
python bench/spectrum/bench.py

# Custom cell list, fewer prompts
python bench/spectrum/bench.py \
    --prompts bench/spectrum/prompts.example.txt \
    --cells "euler:simple,euler_a:simple,er_sde:karras,dpmpp_2m_sde_gpu:exponential" \
    --steps 28 --seed 42

# Only generate Spectrum (skip reference) — e.g. when iterating on Spectrum params
python bench/spectrum/bench.py --skip_reference

# Different workflow / node IDs
python bench/spectrum/bench.py \
    --template workflows/modhydra.json \
    --sampler_node 19 --pos_node 11 --neg_node 12
```

## How the reference baseline works

Instead of swapping to stock `KSampler`, the reference pass runs the **same** `SpectrumKSamplerAdvanced` node with `warmup_steps = steps`. That forces every step through the full forward path while keeping the rest of the graph (text encoding, VAE, any model patches) byte-identical to the Spectrum pass. Direct apples-to-apples comparison with no wrapper-side variance.

## Output layout

```
bench/spectrum/results/20260420_161234/
├── config.json                                    # CLI args + cell list
├── results.csv                                    # wall-clock, filenames, prompts
├── p00_euler_simple_ref.png
├── p00_euler_simple_spec.png
├── p00_euler_a_simple_ref.png
├── p00_euler_a_simple_spec.png
└── ...
```

Filenames sort so that paired `*_ref.png` / `*_spec.png` land next to each other — easy to flip through in any image viewer.

## Suggested follow-up

Eyeball the ref/spec pairs for each cell. Failure patterns to watch:
- **Color collapse / posterization** → ancestral noise compounding on cached-step error (euler_a family).
- **Hollowed faces / neon rim-bleed** → same cause, more severe form.
- **Soft detail loss only** → cache prediction error that the sampler tolerates gracefully. Expected on euler, acceptable on er_sde.
- **Geometric drift** → log-sigma trigger too loose; try smaller `--window_size` or larger `--warmup_steps`.

For quantitative comparison (LPIPS / PSNR), read the PNGs back into a notebook — pairs share filenames differing only by the `_ref`/`_spec` suffix.
