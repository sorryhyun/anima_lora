# E17 — Gaussian-closure test of the posterior residual identity

| | |
|---|---|
| **Status** | **DONE 2026-08-01** |
| **Verdict** | **All three closures FAIL the pre-registered amplitude bar** (RMSE 0.17–0.24 vs ≤ 0.05) → the measured ‖Δr̄(σ)‖ is certified as the minimal honest closure; manuscript stance unchanged. But the failure is *structured*: shape, route-uniformity, reenc≈0 control, and the σ=1 endpoint are all reproduced — the miss is concentrated at low σ |
| **Runs** | `bench/results/20260801-1636/` (fit 20 / holdout 16, holdout = E11's probe set) |
| **Question** | Is the measured route-uniform mismatch curve ‖Δr̄(σ)‖ *derivable* from paired clean latents alone through a second-order (Gaussian) closure of the exact posterior identity r̄\*(σ;x) = σ·Q(σ)⁻¹(x−μ) — or is higher-order/non-Gaussian posterior structure (or the frozen model's approximation error q_θ) essential? |
| **Depends on** | [E11](../e11/) (the measured curves + probe images: `bench/results/20260730-2054-resvec/`), the paper's posterior-operator proposition (`paper_suggestion/sec_theory.tex` Eq. posterior; appendix "mean residual as a posterior operator"). Distinct from [E12](../e12/): E12 tested a *scalar* posterior-uncertainty budget (√tr-Cov, refuted); E17 tests the posterior **mean operator** with paired cross-band covariance structure |
| **Instrument** | `bench/run_posterior_closure.py` — VAE-only GPU (one-time demoted/reenc latent encode, cached under `probe1280_cache/closure_latents/`), then pure CPU. No DiT forward |
| **In the paper** | Either way one sentence in the appendix section: pass → part of the measured curve becomes a data-only derivation; fail → the measured curve is certified as the minimal honest closure (the current manuscript stance, unchanged) |

## Design

Three nested closures of `x|c ~ N(μ, C)` per arm (native / demote1024 / 896
/ 768 / 512 / reenc), fitted on the ~20 complete tier-1280 images NOT in the
probe set (band pooling + shrinkage), predictions scored on the 16 held-out
probe images — the same images behind the measured curves:

1. **diag** — per-(radial-band, channel) variance, diagonal in DCT modes =
   the destroyed-band Wiener null (already falsified at the measured level;
   gets its honest quantitative shot here).
2. **block** — + within-band 16×16 channel covariance (shrunk toward I,
   λ=0.2).
3. **octave** — + parent–child coupling between DCT modes (i,j) and (2i,2j)
   per channel (clipped |r|≤0.6) — the cheapest wavelet-style cross-band
   structure, solved exactly along disjoint octave chains (solver verified
   to machine precision against a dense Bayes solve).

Estimand replicated exactly: per image and adjacent pair, hi-grid predicted
residual area-downsampled to the lo grid, relative L2 (no split-half floor —
predictions are analytic means), mean over holdout images; compared against
the measured `excess` curves. Stated approximations: caption-**marginal**
clean-latent model (true posterior is caption-conditional); band-pooled
stationary covariance (free-fit shapes force a shape-invariant
parameterization).

## Pre-registered reads (frozen before the run)

- **Shape**: per route, Pearson r over σ between predicted and measured
  excess (measured declines 0.89 → 0.36, monotone).
- **Route-uniformity**: measured curves are route-uniform (±0.02 per-σ
  spread); destroyed-band ordering violates this.
- **Amplitude**: RMSE (predicted vs measured excess over σ), held out.
- **Verdict rule**: a closure **PASSES** iff mean held-out RMSE ≤ 0.05 AND
  max route spread ≤ 2× measured max spread. Expected outcome (recorded
  honestly): all three fail — two nested special cases already died (E12
  scalar, diagonal Wiener), and E11's image-specific mismatch directions sit
  awkwardly with any deterministic linear-operator response. A fail is still
  informative: it certifies the measured ‖Δr̄(σ)‖ as the minimal honest
  closure and justifies the manuscript's "measured ingredient" stance.

## Results — FAIL on amplitude, everything else reproduced

Held-out predicted vs measured excess (route-averaged pattern; full curves in
the run's `result.json`):

| σ | measured | diag | block | octave |
|---|---|---|---|---|
| 0.125 | 0.86–0.90 | 1.22–1.26 | 1.32–1.37 | 1.24–1.30 |
| 0.625 | 0.71–0.77 | 0.69–0.73 | 0.80–0.85 | 0.80–0.85 |
| 1.0 | 0.36–0.44 | 0.33–0.40 | (same) | (same) |

- **Shape**: Pearson r = 0.94 (diag) / 0.96 (block) / 0.97 (octave) —
  monotone decline reproduced.
- **Route-uniformity**: predicted per-σ route spread 0.068 vs measured
  0.075, with the same mild 768→512 elevation. Notably even `diag`
  is route-uniform *in this estimand* — the destroyed-band ordering washes
  out of the normalized relative-L2 at these band statistics, so
  route-uniformity is a weaker discriminator against the diagonal closure
  than the paper's band-confinement read (which stands separately).
- **Endpoint σ=1**: predicted 0.33–0.40 vs measured 0.36–0.44 — the
  closure nails the endpoint, as it must (r̄\*(1;x) = x−μ is almost
  model-free).
- **reenc control**: predicted 0.002–0.004 ≈ 0, matching the measured
  control at zero-within-CI.
- **The miss is low-σ**: predicted ≈ 1.2–1.4 vs measured excess ≈ 0.86–0.90
  at σ=0.125 (~40% over-prediction). Caveat recorded: at σ=0.125 the
  measured yardstick's split-half floor is large (floor ≈ 0.45 against
  d ≈ 1.33) and the `excess = d − floor` debias is first-order, so the
  measured low-σ point is itself the least certain — the discrepancy is
  real but its exact size is bounded, not pinned, by this instrument.
- Nesting did **not** monotonically help (block/octave worse than diag at
  low σ): richer second-order structure raises predicted fine-band
  reconstruction bias, the opposite of what closing the gap needs.

**Interpretation.** The trained network's cross-grid residual mismatch at
low σ is *smaller* than the caption-marginal Gaussian Bayes field predicts.
The missing ingredients are exactly the ones the identity names:
caption-conditioning (the true posterior is sharper than the marginal
Gaussian, most binding at low σ), non-Gaussian posterior structure, and/or
the frozen model's approximation error q_θ. Per the pre-registered rule the
measured curve stays a measured ingredient — now with a quantified statement
of *how far* second-order data-only structure gets (shape and endpoint: all
the way; low-σ amplitude: no).
