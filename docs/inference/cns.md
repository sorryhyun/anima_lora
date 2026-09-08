# CNS — Colored Noise Sampling

Training-free SDE sampler plug-in. Replaces the white noise that `ERSDESampler` injects each step with frequency-colored, RMS-normalized noise that dumps the fixed stochastic-energy budget into the radial frequency bands the network has *not yet resolved* at that step. A zero-sum reallocation of a fixed variance budget — not a global noise scale-up.

Paper: [Colored Noise Diffusion Sampling](https://hadardavidson.github.io/CNS/) (Davidson, Issachar, Benaim — Hebrew U., arXiv [2605.30332](https://arxiv.org/abs/2605.30332)). Local PDF at repo root `2605.30332v1.pdf`.

Read first: `_archive/bench/cns/plan.md` (precondition + Phase-0 staircase results + composition tensions). The γ premise is independently corroborated by `project_sigma_signal_resolves_by_045` (base resolves x0 by σ≈0.45; e_low triples in the σ<0.45 tail — that finding is the CNS γ-matrix viewed from the σ axis).

## The mechanism (Algorithm 1, paper p.7)

Precompute a `[T, F]` completion matrix γ(f, t) once per model (Eq. 8) from deterministic ODE trajectories:

```
γ(f, t) = 1 − |X₀(f) − X_pred(f, t)|² / |X₀(f)|²        # ∈ [0, 1], 1 = band resolved
```

Then per SDE step, recolor that step's injected noise instead of drawing white:

```
scale(f) = sqrt(1 − γ(f, σ))          # Eq. 11 numerator: energy → unresolved bands
W        = fft2(white) · scale[bin(f)]
w_c      = ifft2(W).real
w_c     /= std(w_c)                    # RMS-renormalize → conserve total variance
```

`w_c` is a unit-variance colored draw; the sampler then scales it by its own per-step noise coefficient exactly as it would the white draw. Because total injected variance is unchanged, this is a pure spectral *reallocation* of the SDE's fixed stochastic budget (Eq. 10): less energy into already-resolved low-freq bands early, more into the high-freq bands still being built late. RMS/variance conservation (paper §A) is load-bearing — over-inject pushes off-manifold, under-inject collapses modes.

## Why it works on Anima — precondition verified

CNS feeds on spectral bias (low-freq structure resolves early, high-freq detail late). Phase 0 measured this directly on Anima (`_archive/bench/cns/plan.md`, 2026-05-31, GO):

| config | t50 spread (low→high freq) | aggregate σ50 | linear-target MAE |
|---|---|---|---|
| base 1024² cfg1 | +0.410 | 0.486 | 0.101 |
| LoRA 1024² cfg1 | +0.409 | 0.490 | 0.100 |
| base portrait cfg4 | +0.521 | 0.394 | 0.144 |
| LoRA portrait cfg4 | +0.521 | 0.398 | 0.142 |

The staircase is sharp, the aggregate-γ crossing re-derives `project_sigma_signal` (≈0.45), and the γ heatmap reproduces paper Fig. 3 on Anima. Two operational takeaways: γ is LoRA-transparent (reusable with the adapter on or off), and CFG×aspect sharpens the staircase (so γ is calibrated at the deploy config, where the bias is strongest). The shipped γ at cfg=4 with LoRA has been A/B-validated as well-working at production aspects.

## The precondition that must be met: run er_sde

CNS only acts on the stochastic path. Anima's CLI default `--sampler euler` is a deterministic ODE — it injects no noise, so CNS is a literal no-op there (and on `lcm`). Production inference uses `--sampler er_sde`, which is where CNS has a surface. **The fair A/B baseline for any CNS result is er_sde white-noise, never the euler default.**

## Quick start

```bash
python inference.py --sampler er_sde --cns auto   ...   # shipped γ, full strength
```

`--cns auto` resolves the shipped completion matrix at `networks/calibration/cns_gamma.npz`. Pass an explicit path to use a custom one. There is no dedicated `make` target — CNS rides on any `er_sde` inference invocation and composes with the existing `SPECTRUM=1` / `MOD=1` test flags.

## CLI

| Flag | Default | Notes |
|------|---------|-------|
| `--cns` | unset | Path to a `cns_gamma.npz` completion matrix, or `auto` for the shipped default. No-op on `--sampler euler/lcm` (warns if set there). |
| `--cns_strength` | `1.0` | Linear blend white↔recolored, then RMS-renormalize. `1.0` = full CNS, `0.0` = pass-through. Safety knob only — the paper's ablation shows partial white-noise corruption is strictly inferior (see below); leave at 1.0 unless a checkpoint goes off-manifold. |

## The completion matrix artifact

`networks/calibration/cns_gamma.npz` (loaded by `--cns auto`):

| key | shape | meaning |
|---|---|---|
| `gamma` | `(A, T, F)` | completion matrix per aspect — shipped as `(1, 28, 32)` (single averaged γ) |
| `aspects` | `(A, 2)` | pixel `(H, W)` each γ row was calibrated at |
| `sigmas` | `(T+1,)` | calibration σ schedule (runtime σ-interpolates onto this) |
| `radial_centers` | `(F,)` | radial-freq bin centers in [0, 1] |
| `cfg`, `flow_shift`, `steps`, `source_aspects`, `averaged` | scalar/meta | calibration provenance |

Cross-aspect variation is cosmetic (β MAD ~0.01, same conclusion as `project_dcw_bucket_prior_cosmetic`), so one averaged `(1, T, F)` γ ships and serves any resolution — the recolorer's nearest-aspect select degrades to index 0. Radial bins are normalized to [0, 1], so a γ calibrated at one grid maps onto any other shape by bin index. Caveat: all calibrated aspects are 4200-token, AR 0.6–1.34; extreme shapes are unmeasured extrapolation.

## Calibration: `scripts/calibration/cns_calibrate.py`

```bash
python scripts/calibration/cns_calibrate.py --cfg 4.0 --n_aspects 3            # compiled (default)
python scripts/calibration/cns_calibrate.py --cfg 4.0 --no-compile            # eager
# probe the adapter instead of base (γ is LoRA-transparent per Phase 0):
python scripts/calibration/cns_calibrate.py --cfg 4.0 --extra --lora_weight output/ckpt/<x>.safetensors
```

Drives the Phase-0 euler capture across the deploy config — `--cfg 4.0` at the top-`--n_aspects` `DCW_ASPECT_BUCKETS` — and bundles per-aspect γ into the shipped npz (with `--average_aspects`, default on). γ is measured from the deterministic euler ODE (Eq. 8); the recoloring then applies on er_sde. Prompts are real captions (richest/most-tags across distinct artists, from preprocessed stems → `image_dataset/*.txt`), not synthetic — they matter: real-vs-synthetic β MAD 0.036, ~3–5× the cross-aspect MAD. GPU-heavy (3 aspects × prompts × seeds at cfg=4); loads the DiT once, precomputes all text, frees the TE before the loop (TE→free→DiT invariant), and compiles by default (the 3 default aspects are all token-count 4200 → one native-flatten graph reused across them). `gamma_probe.py` is the single-config read-only Phase-0 staircase check; `calibrate.py` reuses its helpers.

## Composition

CNS lives at the per-step noise-injection seam (`ERSDESampler._sample_noise`, `library/inference/sampling.py`), so it is orthogonal to anything that touches the model forward or the drift.

| Composes with | Interaction |
|---|---|
| `--sampler er_sde` | The only live surface. CNS swaps the white draw for a unit-variance colored draw; the surrounding `alpha_t · s_noise · noise_coeff` scaling is untouched, so total injected energy is conserved. |
| `--spectrum` | Structurally clean (different seam) but a real tension: Spectrum forecasts a smooth feature trajectory, and CNS concentrates energy in high-freq *late* — exactly where the Chebyshev forecaster is weakest. CNS conserves total variance so expect ≤ the er_sde-white perturbation Spectrum already tolerates, but the cached-step error must be A/B-checked, not assumed (cf. `project_spectrum_er_sde_forecastability`: er_sde forecasts ~1.6× worse than euler). |
| SMC-CFG | Independent — SMC-CFG corrects the velocity/drift CFG combine (pre-`denoised`); CNS is the noise term. Different seams. |
| mod-guidance | Orthogonal seam (AdaLN t-embedding), but it changes the v-trajectory → γ should be recalibrated with mod-guidance ON if co-deployed (2nd-order; the shipped γ was calibrated mod-guidance OFF). |
| LoRA / OrthoLoRA / T-LoRA / Hydra | Orthogonal — no module patching, no extra weights. γ is LoRA-transparent (Phase 0). |

## Faithfulness to the paper, and knobs deliberately omitted

Our `library/inference/corrections/cns.py` is a faithful transcription of Algorithm 1 + Eq. 11: `sqrt(1−γ)` scaling, empirical-std RMS renorm (Alg. 1's literal `w_c/std(w_c)`), dynamic γ(f, t), and strict variance conservation. We additionally σ-interpolate γ each step (robust to step-count / flow_shift mismatch vs the calibration schedule, vs Alg. 1's raw `gamma[i]` indexing) and select the nearest calibrated aspect — both engineering improvements over the reference pseudocode.

The authors' reference repo (`colored-noise-sampling/transport/integrators.py::cns_sde`) carries extra knobs used for SiT/ImageNet FID-chasing — `gamma_matrix_divider`, `power_gamma`, `alpha_tilting` (time-varying frequency tilt), `energy_scale`. None are part of the paper's method, and the paper's own ablation (Table 6, FID-10K, CNS = 9.61) argues against their analogs, so we deliberately do not implement them:

| Omitted repo knob | Closest paper ablation | Verdict |
|---|---|---|
| `energy_scale` ≠ 1.0 | Scale 0.90 → 16.17, Scale 1.05 → 20.46 | violating the normalization constraint hurts; principled value is 1.0 (what we use). |
| `cns_strength` < 1.0 (white blend) | 50% White Noise → 10.64 | "partial white-noise corruption consistently yields inferior results"; our `strength` < 1 is a safety escape hatch, not a tuning lever. |
| `alpha_tilting` (parametric tilt) | mBm (time-varying Hurst) → 11.88 | the data-driven γ matrix beats a parametric colored schedule. |
| `gamma_matrix_divider` / `power_gamma` | (not in paper) | reshape the residual curve; the repo's `divider=25` "gentle" guided config is FID-squeezing on a specific SiT+CFG, not a principle. |

One implementation divergence worth noting: Alg. 1 renormalizes by a single global scalar `std(w_c)`; we renormalize per-channel over the spatial plane (`dim=(-2,-1)`), so each latent channel keeps exactly unit variance like the white noise it replaces. Faithful-or-better, not a gap.

## Limitations / open questions

- er_sde-only. No-op on the euler/ODE default and on `lcm`. The paper validates CNS across higher-order stochastic solvers (Heun, SRK2/SRK2S, Table 2); we only hook ER-SDE. Porting to a Heun-style stochastic path is unimplemented.
- γ provenance. Shipped γ is calibrated at cfg=4, mod-guidance OFF, across 3 aspects (all 4200-token, AR 0.6–1.34). Extreme aspect ratios and mod-guidance-ON deployment are extrapolation / 2nd-order recalibration territory.
- Spectrum compose not yet A/B-confirmed clean — see Composition table.
- (Optional refinement) Eq. 11's analytic denominator `sqrt(mean_f(1−γ))` is the deterministic equal-in-expectation form of our empirical `std(w_c)`; switching would give bit-stable per-step normalization at the cost of diverging from Alg. 1. Not pursued.

## Related code

| File | Role |
|---|---|
| `library/inference/corrections/cns.py` | `CNSRecolorer` — loads γ, selects aspect, σ-interpolates, fft2 × `sqrt(1−γ)`, ifft2, per-channel RMS-renorm. |
| `library/inference/sampling.py` | `ERSDESampler._sample_noise` — the recolor seam (white draw → `cns.recolor(white, σ_s)`); injection at `step()`. |
| `library/inference/generation.py` | `_build_cns_recolorer` — builds from `--cns` / `--cns_strength`, warns if `--sampler` injects no noise. |
| `inference.py` | `--cns` / `--cns_strength` CLI surface. |
| `scripts/calibration/cns_calibrate.py` | Phase-1 completion-matrix calibration → the shipped npz + per-aspect γ heatmaps next to it. |
| `scripts/calibration/gamma_probe.py` | Phase-0 read-only staircase check (one config). |
| `_archive/bench/cns/plan.md` | Precondition, phase log, composition tensions. |
| `networks/calibration/cns_gamma.npz` | Shipped completion matrix (`--cns auto`). |
