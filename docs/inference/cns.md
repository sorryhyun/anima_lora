# CNS — Colored Noise Sampling

Training-free SDE sampler plug-in. Replaces the white noise that `ERSDESampler` injects each step with frequency-colored, RMS-normalized noise that dumps the fixed stochastic-energy budget into the radial frequency bands the network has *not yet resolved* at that step. A zero-sum reallocation of a fixed variance budget — not a global noise scale-up.

Paper: [Colored Noise Diffusion Sampling](https://hadardavidson.github.io/CNS/) (Davidson, Issachar, Benaim — Hebrew U., arXiv [2605.30332](https://arxiv.org/abs/2605.30332)). Reference code: [HadarDavidson/colored-noise-sampling](https://github.com/HadarDavidson/colored-noise-sampling) (SiT/ImageNet only — no FLUX/JiT code).

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
| LoRA / T-LoRA / Hydra | Orthogonal — no module patching, no extra weights. γ is LoRA-transparent (Phase 0). |

## Faithfulness to the paper, and the schedule relaxations we leave out

Our `library/inference/corrections/cns_core.py` is a faithful transcription of Algorithm 1 + Eq. 11: `sqrt(1−γ)` scaling, empirical-std RMS renorm (Alg. 1's literal `w_c/std(w_c)`), dynamic γ(f, t), and strict variance conservation. We additionally σ-interpolate γ each step (robust to step-count / flow_shift mismatch vs the calibration schedule, vs Alg. 1's raw `gamma[i]` indexing) and select the nearest calibrated aspect — both engineering improvements over the reference pseudocode.

The paper's App. C.2 ("Empirical Schedule Relaxations") adds tuning knobs on top of Eq. 11, and **every headline number uses them** (Tab. 7). The reference repo (`transport/integrators.py`, `__generate_matrix_scaled_noise`) implements them as:

| Repo knob | What it does | Authors' settings (Tab. 7 / `run_cns_best_results.sh`) |
|---|---|---|
| `gamma_matrix_divider` c | progression scaling γ̃ = γ/c → `scale = (1 − γ/c)^power` | SiT unguided 1.73; every CFG run 5–25 (SiT CFG 25) |
| `power_gamma` / `sqrt_gamma` | exponent on the residual (Eq. 11 = 0.5) | 0.5, except SiT unguided 0.75 |
| `alpha_tilting` | time-varying exponential tilt `exp(α(t)·f_norm)` on the residual | SiT only; unguided 0.15 → −0.5 (late high-freq is *damped*, against the paper's narrative) |
| `energy_scale` | multiplies the unit-std draw | 0.98–0.999 |

The authors' own logs (`evaluation_logs/`, 250-step Euler-Maruyama, FID-50K) show where the gain comes from. Unguided: white-SDE 8.79 → CNS 6.32, using the full knob set. Guided: 2.04 → 1.97 with divider 25. γ is in [0, 1] (the repo min-max-normalizes it per bin, in `scale_gamma_matrix.ipynb`; ours already runs from ≤0.06 to 1 per bin), so γ/25 ≤ 0.04: the guided draw is within ~2% of white noise in every band.

**We ship Eq. 11 as-is (c = 1, power 0.5), with no tilt and no energy scale.** CLI `--cns auto` and the ComfyUI `er_sde_cns` sampler both run it with nothing to tune. A/B on 2026-10-15 (base DiT, er_sde, cfg 4, 28 steps, 6 prompts × 2 seeds, 1024² / 832×1248, paired noise draws):

- **divider.** With our γ, c ≥ 2 flattens the per-band amplitude to within ±10% of white, and c = 25 is within 0.2%. Any non-white draw re-rolls the er_sde trajectory: c = 5 still moves pixels by a mean |Δ| of ~8/255 versus white, so pixel distance does not measure the effect. Judged by eye, c = 1 beats white: fewer stray props, cleaner hands and SFX kana. c = 2 / 5 behave like white.
- **power** (at c = 1). Power 0.75 is indistinguishable from 0.5. Power 1.0 drifts further from white and its changes are mixed: some hands and can labels improve, while SFX kana and the artist signature get worse. Nothing broke off-manifold.
- **energy_scale ≠ 1 and `cns_strength` < 1.** Not adopted. The paper's Tab. 10 / Tab. 6 ablations show large departures hurt (scale 0.90 → 16.17, 1.05 → 20.46; 50% white blend → 10.64 vs CNS 9.61). `cns_strength` is a safety escape hatch, not a tuning lever.
- **divider < 1 (stronger reallocation).** Not tried. Every band with γ ≥ c is zeroed, which piles the whole budget into the top few bins. That is the single-band "bang-bang" injection App. B.2.3 argues against.

One implementation divergence worth noting: Alg. 1 renormalizes by a single global scalar `std(w_c)`; we renormalize per-channel over the spatial plane (`dim=(-2,-1)`), so each latent channel keeps exactly unit variance like the white noise it replaces. Faithful-or-better, not a gap.

## Limitations / open questions

- er_sde-only. No-op on the euler/ODE default and on `lcm`. The paper validates CNS across higher-order stochastic solvers (Heun, SRK2/SRK2S, Table 2); we only hook ER-SDE. Porting to a Heun-style stochastic path is unimplemented.
- γ provenance. Shipped γ is calibrated at cfg=4, mod-guidance OFF, across 3 aspects (all 4200-token, AR 0.6–1.34). Extreme aspect ratios and mod-guidance-ON deployment are extrapolation / 2nd-order recalibration territory.
- Spectrum compose not yet A/B-confirmed clean — see Composition table.
- (Optional refinement) Eq. 11's analytic denominator `sqrt(mean_f(1−γ))` is the deterministic equal-in-expectation form of our empirical `std(w_c)`; switching would give bit-stable per-step normalization at the cost of diverging from Alg. 1. Not pursued.

## Related code

| File | Role |
|---|---|
| `library/inference/corrections/cns_core.py` | `CNSRecolorer` numerics — loads γ, selects aspect, σ-interpolates, fft2 × `sqrt(1−γ)`, ifft2, per-channel RMS-renorm. Vendored verbatim into the ComfyUI Spectrum node (`er_sde_cns`). |
| `library/inference/corrections/cns.py` | Thin subclass: `--cns auto` / path resolution under the repo home. |
| `library/inference/sampling.py` | `ERSDESampler._sample_noise` — the recolor seam (white draw → `cns.recolor(white, σ_s)`); injection at `step()`. |
| `library/inference/generation.py` | `_build_cns_recolorer` — builds from `--cns` / `--cns_strength`, warns if `--sampler` injects no noise. |
| `inference.py` | `--cns` / `--cns_strength` CLI surface. |
| `scripts/calibration/cns_calibrate.py` | Phase-1 completion-matrix calibration → the shipped npz + per-aspect γ heatmaps next to it. |
| `scripts/calibration/gamma_probe.py` | Phase-0 read-only staircase check (one config). |
| `_archive/bench/cns/plan.md` | Precondition, phase log, composition tensions. |
| `networks/calibration/cns_gamma.npz` | Shipped completion matrix (`--cns auto`). |
| `networks/calibration/moss_cns_gamma*.npz` | MOSS-Image γ (`scripts/calibration/moss_cns_calibrate.py`, 50 steps). `library.moss.cns.GAMMA_PATHS` maps (variant, cfg) → file; `MossCNS.from_spec("auto", variant, cfg)` takes the nearest calibrated cfg. Table below. |

MOSS-Image γ files (aggregate σ50 / low−high spread at 1344×768; the Base cfg-3 file at 1024², its only size). `scripts/moss/generate.py` has no CNS flag: the files are read by `library.moss.cns` (`lora_ab.py`'s `cns` arms) and by the ComfyUI Spectrum node's `er_sde_cns`, which picks one by variant and cfg.

| File | Variant, cfg | Sizes | σ50 | spread |
|---|---|---|---|---|
| `moss_cns_gamma.npz` | Base, 3 | 1024² | 0.61 | +0.53 |
| `moss_cns_gamma_cfg4.npz` | Base, 4 | 7 aspects, AR 0.57–1.75 | 0.554 | +0.579 |
| `moss_cns_gamma_cfg5.npz` | Base, 5 | 1344×768, 768×1344 | 0.508 | +0.620 |
| `moss_cns_gamma_sft_cfg3.npz` | SFT, 3 | 1344×768, 768×1344 | 0.606 | +0.493 |
| `moss_cns_gamma_sft_cfg4.npz` | SFT, 4 | 1344×768, 768×1344 | 0.575 | +0.534 |
| `moss_cns_gamma_sft_cfg5.npz` | SFT, 5 | 1344×768, 768×1344 | 0.539 | +0.553 |

The staircase holds in every file, and cross-size σ50 MAD stays at 0.005–0.01. Each step up in cfg lowers σ50 by 0.03–0.05. At the same cfg, SFT sits about 0.02 above Base, with a flatter staircase.
