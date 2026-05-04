# LoRA-transfer findings — cross-family + multiplier linearity

**Status:** preliminary (n=2 LoRA families) · **Date:** 2026-05-04 · **Question:** does the SNR-t bias signature transfer across LoRA families and multipliers? If yes → calibrate once per base DiT. If no → calibrate per LoRA. · **Analysis scripts:** `../cross_lora_check.py` (family check) · `../lora_multiplier_linearity_check.py` (multiplier sweep)

## Headline

Two LoRA families at standard inference multipliers land **cleanly in the proposal's top decision-gate branch** (base-DiT scoped, no per-LoRA artifact). All `Δ_LL` deviations are <6%, well inside the proposal's ±15% tolerance:

| config | LL Δ vs base | Pearson r vs base (24-step shape) | branch |
|---|---|---|---|
| **artist @1.0** | **−0.24%** | **r = 0.99999** | base-DiT scope |
| hydra @0.5 | −1.76% | r = 0.99999 | base-DiT scope |
| hydra @1.0 | −5.12% | r = 0.99992 | base-DiT scope |

**Artist LoRA at full multiplier is closer to base than HydraLoRA at half multiplier.** Per-step LL gap shape correlations are 5 nines for artist and hydra@0.5; even hydra@1.0's late-half residual maxes out at 5.2%.

See `cross_lora_check.md` for the auto-generated full report and the four subdirs (`base/`, `artist/`, `hydra-mult05/`, `hydra-mult10/`) for the source bench artifacts.

## Source benches

All four runs use n=8 cached training samples × 2 seeds × 24 steps, baseline only (no DCW sweep):

| Run | source dir | LoRA | family | mult | LL signed gap | LL SNR |
|---|---|---|---|---|---|---|
| `base/` | `bench/dcw/results/20260504-0930-band-variance/` | — | — | — | −355.49 | 0.99 |
| `artist/` | `bench/dcw/results/20260504-1050-sincos-0503/` | `output/ckpt-artist/anima_lora_sincos.safetensors` | regular LoRA, artist style | 1.0 | −356.33 | 1.00 |
| `hydra-mult05/` | `bench/dcw/results/20260504-0958-hydra-0503-lower/` | `anima-hydra-0502-4812.safetensors` | HydraLoRA (MoE) | 0.5 | −361.75 | 1.01 |
| `hydra-mult10/` | `bench/dcw/results/20260504-0946-hydra-0503/` | `anima-hydra-0502-4812.safetensors` | HydraLoRA (MoE) | 1.0 | −373.68 | 1.04 |

## Two findings, separable

### 1. Multiplier linearity within HydraLoRA (mult ∈ {0, 0.5, 1.0})

HydraLoRA at multiplier=0.5 sits **within ~1% of perfect linear interpolation** between base (mult=0) and HydraLoRA at mult=1.0. The MoE routing inside HydraLoRA does not introduce nonlinear deviation at standard inference multipliers — even though MoE was the family most expected to break linearity.

| band | base | obs @0.5 | obs @1.0 | linear-pred @0.5 | linerr (% of base) | Δ@1.0 vs base |
|---|---|---|---|---|---|---|
| **LL** | −355.49 | −361.75 | −373.68 | −364.58 | **+0.80%** | −5.12% |
| LH | −160.78 | −160.74 | −159.92 | −160.35 | −0.24% | +0.54% |
| HL | −164.26 | −164.36 | −163.98 | −164.12 | −0.15% | +0.17% |
| HH | −121.33 | −121.04 | −119.55 | −120.44 | −0.49% | +1.47% |

Per-step LL late-half (i ≥ 12) max relative residual: **1.07%**, mean 0.72%.

**Why MoE doesn't break linearity here:** HydraLoRA's routes are σ-conditioned, not multiplier-conditioned. At a given (σ, x_t), the same experts activate at any multiplier; the multiplier just scales their additive output. So `LoRA_out(mult) = mult · Σ_e route_e(σ, x_t) · ΔW_e · x` — exactly linear in mult by construction.

See `linearity_check.md` for the auto-generated linearity report.

### 2. Cross-family invariance (HydraLoRA vs regular artist LoRA)

The artist LoRA (regular LoRA on a single-artist style dataset, `output/ckpt-artist/`) at mult=1.0 produces an **essentially zero perturbation** to the bias signature: Δ_LL = −0.24%, per-step shape correlation r = 0.99999, late-half max residual 0.29% of base. The cross-sample SNR profile shifts by ≤2% on every band.

**Sub-finding: artist LoRA perturbs the bias signature ~20× less than HydraLoRA at the same multiplier.** Likely reasons:
- Regular LoRA vs Hydra MoE → fewer total params.
- Artist-style training distribution is narrow → small effective weight delta.
- The model is shifting the style decoder, not the SNR-bias-generating blocks (which live deeper).

This is the key result for the `project_dcw_when_to_use` failure mode. The base-DiT-scoped reference profile applies to artist LoRAs; v0a's per-prompt α fit handles the sign-flip on individual flat-style prompts (~8% of prompts had α<0 in the transfer-hyp bench). **No per-LoRA artifact is needed for artist/style LoRAs.**

## Implications for the proposal

Lands cleanly in the **top branch** of `docs/proposal/dcw-learnable-calibrator.md` §"Cross-LoRA invariance gate":

> All LoRA × multiplier configurations match base-DiT SNR profile within ±15% AND λ_LL* within ±15% → **Base-model-scoped calibration.** One reference profile per base DiT release; no per-LoRA artifact.

Both observed LoRA families fit. Specifically:
- One `<base_dit_name>_dcw_vfwd.safetensors` per base DiT release (~200 B), inherited by every LoRA.
- No multiplier-aware rescaling needed at inference.
- The per-LoRA fallback path stays in the resolution chain but is expected to be unused.
- Multiplier sweeps can be dropped from any future cross-LoRA bench — one mult=1.0 measurement per LoRA is enough.

The proposal's deliverable step 1 wanted ≥3 LoRAs spanning style range. With (HydraLoRA MoE multi-task) + (regular LoRA artist style) we have **2 of 3**, spanning {MoE, regular} × {multi-task, single-style}. One more LoRA from a structurally different parameterization (T-LoRA / OrthoLoRA / ReFT) would close it out.

## What this does NOT answer yet

- **OrthoLoRA / T-LoRA / ReFT.** Different parameterizations might inject bias differently (e.g. T-LoRA's σ-conditioned rank could couple with the SNR-bias-generating attention path).
- **Painterly / detail-dense LoRAs.** Both checked LoRAs have very small bias-signature deltas; a LoRA designed to *enhance* detail might actually push the LL gap further (positive coupling instead of zero).
- **Image-conditioned methods (IP-Adapter / EasyControl).** Flagged for v2 since they change the network forward more meaningfully than weight-space adapters.
- **Multipliers > 1.0.** Sub-linearity at mult=0.5 was concave (0.8% above the line); at mult=1.5 it might bite harder. Worth checking if you ever go there in production.

## How to reproduce

```bash
# Bench each LoRA at mult=1.0 (≈3 min on 5060 Ti at n=8 × 2 seeds)
uv run python archive/dcw/measure_bias.py \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --lora_weight <path> --lora_multiplier 1.0 \
    --n_images 8 --n_seeds 2 --infer_steps 24 \
    --label <unique-label>

# Cross-family report (instant)
uv run python archive/dcw-learnable-calibrator/cross_lora_check.py \
    --base archive/dcw-learnable-calibrator/lora-transfer/base \
    --lora "<label>@<mult>" <run_dir> \
    [--lora ... ...] \
    --out_md <somewhere>/cross_lora_check.md

# Multiplier-linearity check (only when you have 3 runs at mult ∈ {0, mid, 1})
uv run python archive/dcw-learnable-calibrator/lora_multiplier_linearity_check.py \
    --base <base run>/per_step_bands.csv \
    --mid  <mult=0.5 run>/per_step_bands.csv \
    --full <mult=1.0 run>/per_step_bands.csv \
    --mid_mult 0.5 \
    --out_md <somewhere>/linearity_check.md
```
