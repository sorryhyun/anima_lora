# SeaCache's SEA filter is a better cache-decision metric on Anima — but not a Spectrum replacement

We evaluated **SeaCache** (Chung et al., "SeaCache: Spectral-Evolution-Aware
Cache for Accelerating Diffusion Models", arXiv:2602.18993v2) as an alternative to
the shipped **Spectrum** inference accelerator. Two separable conclusions, plus a
reusable methodology trap that produced a confidently-wrong first answer.

Bench: `_archive/bench/spectrum_sea/` (`run_bench.py` = Phase 0,
`phase1_counterfactual.py` = Phase 1, `report.md`, `sea.py` = the SEA filter;
archived after the metric shipped). Cross-refs:
`spectral_fraction_metric_inverts.md` (sibling trap), `sigma_signal_where_anima_resolves.md`,
`ctcal_premise_inverted_on_anima.md` (the prior-inversion record this had to clear).

## TL;DR

- **SeaCache is NOT a drop-in for Spectrum.** They solve different halves of the
  problem. Spectrum *forecasts* the block-stack feature and recomputes the head
  every step, so it caches **below** the CFG/plugin boundary and composes cleanly
  with SMC-CFG and mod-guidance. SeaCache's wholesale output-reuse caches **above**
  that boundary and would desync them. The portable idea is SeaCache's **decision
  metric**, not its reuse policy.
- **The SEA filter genuinely improves the cache *decision* on Anima.** Validated
  against the true counterfactual skip-cost: SEA-filtered input distance predicts
  it (ρ +0.51), raw L2 distance **anti**-predicts it (−0.36).
- **Spectrum's own forecast residual loses** — both in feature space and as the
  deployable lagging signal it would actually decide with. The recommendation is
  to graft SeaCache's SEA-filtered input distance onto Spectrum's scheduler.

## Why SeaCache can't just replace Spectrum (composition)

Spectrum keeps two per-branch Chebyshev forecasters over the **pre-`final_layer`
block feature** and, on a skipped step, reconstructs a fresh per-step `noise_pred`
via `t_embedder → final_layer → unpatchify` (`networks/spectrum.py`; verified in
the shipped node `ComfyUI-Spectrum-KSampler/spectrum.py`, which forecasts cond and
uncond features *separately* and lets ComfyUI's CFG combine downstream).

That design preserves the per-step `noise_pred` contract that the sampler-boundary
plug-ins depend on:

- **SMC-CFG** is a stateful finite-difference controller (`s = (e − e_prev) + λ·e_prev`,
  `library/inference/corrections/smc_cfg.py`). Spectrum calls `combine()` every
  step on freshly-forecast cond/uncond, so `e_prev` advances one step at a time.
  SeaCache reusing the combined output would freeze `combine()` across the skip
  window → the `(e − e_prev)` derivative becomes a multi-step staircase.
- **mod-guidance** is injected per-block + final-layer (`library/anima/models.py`).
  Spectrum re-applies the final-layer term each step; SeaCache freezes it.

"Fixing" SeaCache to cache cond/uncond separately and re-run the head reconstructs
Spectrum's architecture — at which point the only difference is copy-vs-forecast,
which forecasting wins. So the only thing worth borrowing is the *metric*.

## The methodology trap (reusable)

Spectrum's scheduler is a content-blind growing window. SeaCache's pitch is a
content-adaptive metric: measure cache distance after a **Spectral-Evolution-Aware
(SEA) filter** (a σ-dependent Wiener-like low-pass, `sea.py`) that downweights the
high-freq noise component so the distance tracks content, not stochastic detail.
To test "is it a better decision metric," correlate each candidate metric against
a skip-cost ground truth across a denoising trajectory.

**Trap: the pooled correlation is dominated by the monotone σ-trend.** The first
run reported "SEA helps, Δρ = +1.5" — an artifact. Both the SEA metric and the GT
share the monotone denoising envelope, so the pooled Spearman just measures
*monotone-in-σ-ness*. But the blind growing window is **already** monotone in σ —
it caches more as σ drops — so a metric being monotone in σ adds **nothing** a
fixed schedule can't do. The only thing a content-adaptive metric can contribute
is the **within-σ** signal: at matched denoising progress, does it flag the
*prompt/step-specific* costly steps?

**The fix — step-stratification.** Under the shared sampler schedule, σ at step
*i* is identical across prompts. Group rows by σ-value (= by step), rank
metric-vs-GT *within* each group, pool the ranks. σ is then exactly constant
inside a group → the trend is fully removed → the correlation isolates "does the
metric correctly rank *which prompt* is costlier to skip at this step."

**Meta-lesson:** the first detrend attempt (σ-quantile bins) was *also* wrong —
bins were wide enough that the trend survived inside them. A monotone-only sanity
input returned ρ = −0.998 instead of ~0, which is the only reason it was caught.
Step-stratification passes the sanity check (monotone → 0.04, injected signal →
1.0). **Always sanity-check a detrending estimator on monotone-only synthetic
inputs before trusting it.**

## Results (step-detrended)

**Phase 0** — metrics vs an x̂₀-motion *proxy* skip-cost (12 prompts, 24 steps,
CFG 4, 1024², compiled):

| metric | detrended ρ |
|---|---|
| `raw_input` ‖Δx_t‖ | −0.44 |
| `sea_input` ‖Δ SEA(x_t)‖ | **+0.57** |
| `resid_feat` Chebyshev feature residual | +0.30 |

**Phase 1** — metrics vs the **true counterfactual** skip-cost. At each step we
actually cache it: Chebyshev-forecast both branches, run only the head
(`_spectrum_fast_forward`), CFG-combine, and measure how far the resulting x̂₀
lands from the real full-compute x̂₀. Non-circular by construction — the metrics
under test are the ones a scheduler genuinely has *before* computing the step
(`sea_input`/`raw_input` are leading; `lag_resid` = the last actual-step residual
Spectrum carries, lagging):

| metric | ρ vs true counterfactual | vs unfiltered GT |
|---|---|---|
| `sea_input` (SeaCache, leading) | **+0.51** | +0.65 |
| `raw_input` (baseline) | −0.36 | −0.46 |
| `lag_resid` (Spectrum has, lagging) | +0.16 | — |
| **proxy(x̂₀) vs true GT** | **+0.82** | (pooled +0.73) |

Reading:

- **SEA filtering helps, on the real objective.** It flips a metric that
  *anti*-predicts injected cost (raw −0.36) into one that predicts it (+0.51).
  Mechanism is exactly the paper's thesis: raw distance is dominated by high-freq
  noise that doesn't move the low-freq content the filter isolates. This survives
  Anima's prior-inversion record (CTCal, σ-reshape).
- **The Phase-0 proxy was faithful** (+0.82 detrended agreement with the true
  counterfactual) — so the cheap x̂₀-motion proxy is a valid stand-in for future
  iteration, not a mirage.
- **Spectrum's own residual loses, non-circularly.** The lagging residual it
  actually has when deciding scores +0.16, beaten by leading SEA-input by +0.35.
  This refutes the prior intuition that "Spectrum can measure its reuse error, so
  its residual beats SeaCache's input proxy" — true in principle, but the residual
  is only available *lagging*, and a leading input signal predicts the upcoming
  step's cost better.

## Conclusion & what's next

**Graft SeaCache's SEA-filtered input distance as Spectrum's scheduling decision
metric**, replacing the content-blind growing window. It is validated against the
true counterfactual, beats raw distance (which is actively misleading), and beats
the lagging residual Spectrum currently has. Keep Spectrum's Chebyshev forecasting
for the *reuse* — only the *when-to-skip* changes, which is orthogonal to the
plugin boundary, so SMC-CFG / mod-guidance composition is unaffected.

**Shipped — library + ComfyUI node.** The SEA-distance trigger is wired into
`spectrum_denoise` (`networks/spectrum.py` + the `networks/spectrum_sea.py`
helpers) as the opt-in `--spectrum_schedule sea` mode with auto-δ matched-compute
calibration — see `docs/inference/spectrum.md` §"SEA schedule". The growing
window remains the default. The **node mirror is done too**: `SpectrumKSampler`
in `~/ComfyUI-Spectrum-KSampler` exposes the `schedule='sea'` mode
(`spectrum.py` + vendored `spectrum_sea.py`). The P2 CMMD A/B ship gate was
resolved by **eyeball A/B** — a near-tie as predicted (SEA at matched compute is
~85% the same schedule as the window), so the node mirror was not held back.

Not yet settled:

- **β sensitivity.** All runs used the natural-image power-law β = 2; untuned.
- The σ<0.45 tail carries ≈0% of SEA-weighted skip-cost (consistent with x̂₀
  resolving by σ≈0.45) yet the blind schedule force-computes the last 3 steps —
  a concrete reallocation the adaptive trigger could exploit.
  **[SETTLED — twice refuted, 2026-06-22 / 2026-07-24.]** The SEA P1
  matched-compute measurement showed the tail actual-rate is governed by
  `stop_caching_step` (identical in both arms; the apparent SEA reallocation was
  a stop-mismatch artifact), and the DPCache Phase-0 bench
  (`_archive/bench/spectrum_dp/`, `docs/findings/dpcache_dp_phase0_refuted.md`) showed a
  planner *freed* from the tail forcing at matched total compute voluntarily
  re-places 3 forwards below σ=0.45 — the forced tail is where a
  budget-constrained schedule wants them anyway. Do not re-propose tail
  reallocation.
