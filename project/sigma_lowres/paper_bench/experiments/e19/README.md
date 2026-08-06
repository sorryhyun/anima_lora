# E19 — why are B and C anti-aligned? Locating the birth of the near-cancellation

| | |
|---|---|
| **Status** | **19.0 + 19.1 DONE 2026-08-06** — completeness/decoherence verdicts + frozen ρ_r prediction below; 19.2 is the next step (scores the frozen prediction), 19.3/19.4 pending the L1 verdict |
| **Question** | [E14](../e14/)'s headline turned the measured gap curve into the *residual of a near-cancellation*: at every σ the data-branch and graph-branch perturbations are strongly anti-aligned (ρ ≈ −0.7…−0.9), both individually far larger than the realized gap, and the σ≈0.4–0.7 cliffs are the \|B⊥\|/\|C⊥\| = 1 crossings. This is why the spectral account misses by ~4× — it transports amplitude without the interference structure. So the theory-facing question is no longer "a law per branch" but: **where is the anti-alignment born — in the residual field r (a property of the trained denoiser + data statistics) or in the pull-back through Jᵀ (a property of the graph/Jacobian)? And is it derivable?** |
| **Depends on** | [E14](../e14/) (the headline + the validated B/C instrument `vector_ledger.py`; unit-honesty rule), [E9](../e9/) (crossing↔window-center localization on 768), [E17](../e17/) (the Gaussian-closure machinery `run_posterior_closure.py` — failed on low-σ *amplitude*, reproduced *shape*/route-uniformity/endpoint), [E11](../e11/) (residual direction structure: norm-only, image-specific directions; `--save_residuals` harness), Q2/G10/G11 in `record/questions.md` (depth band 3–8; RoPE_e origin; PI off-manifold qualifier), [E7](../e7/)+Q3 (adapter axis), [E18](../e18/proposal.md) (the per-draw projection hook — a shared instrument delta, see 19.3) |
| **Instrument** | `vector_ledger.py` re-reads (19.0); `run_posterior_closure.py` + a paired-B/C emitter (19.1); `run_prior_distance.py` + a repromote arm (19.2); `run_sigma_probe.py --repromote --keep_arm_sums` sliced per param group, or the E18 per-draw hook with block-keyed projections (19.3); `--pi_align --repromote` in one process (19.4) |
| **In the paper** | If 19.1/19.2 land coherent: the interaction term I stops being a measured-only entry — §4.6's "reduction's domain" paragraph gains a mechanism, and the cancellation account becomes the theoretical spine of the follow-up paper. If not: the decision tree localizes which layer owns the interaction, which is itself the paper-2 §3 opening |

## The reframe

E14 established *that* the curve is a cancellation residual; nothing yet
says *why* the two interventions oppose. One observation orders the
hypothesis space: near-cancellation is exactly what the safe-route
phenomenon **requires**. If demoting data+graph *together* is
quasi-equivalent (small realized gap) while each half-intervention is
large, then B ≈ −C is forced wherever the route is safe — the joint
demotion direction is a near-flat direction of g(data, graph) even
though neither axis direction is. That reading — approximate **scale
covariance of the trained model along the demotion diagonal** — makes
an immediate, already-answerable prediction: cancellation completeness
should track the safety map (best on 896, degraded on 512), which 19.0
scores from committed artifacts at zero cost.

The discriminator ladder, cheap→expensive:

- **L0** (19.0): what does the existing ledger already pin — ρ(σ) ×
  route profile, completeness ordering, bias checks.
- **L1** (19.1 + 19.2): is the anti-alignment already present at the
  **residual level** (then it is a data-statistics + denoiser property,
  and the closure program is its theory home), or does it **emerge
  through Jᵀ** (then it is graph geometry — two large perturbations
  projected into a narrow shared Jacobian range with opposite sign)?
- **L2** (19.3): if Jacobian-side — is it depth-localized in the
  Q2 early-block band (graph-geometry mediation) or depth-uniform?
- **L3** (19.4): is the graph side RoPE-causal — does PI-aligning the
  demoted grid's phase geometry rotate C?

## 19.0 — re-reads from the committed ledgers (0 GPU, no store needed)

`ledger.json` / `ledger_native.json` ship with the E14 run (the raw
`arm_sums/` store never does), and already carry per-(route, bin): ρ,
S/F/I cross-set debiased, `I_sameset_biascheck`, `rel_cos_B/C`,
h(B)/h(C)/h(B+C), κ∥, both data refs.

1. **ρ(σ) × route table** {896, 768, 512} + a completeness read
   (h(B+C) against h(B), h(C); crossing location per route).
   Pre-registered: the scale-covariance account predicts completeness
   ordered by the map — 512 least complete. A ρ that stays ≈ −0.9 on
   512 with only the *amplitude ratio* broken refutes the "unsafe =
   decoherence" version and keeps only the "unsafe = mismatched
   magnitudes" version.
2. **Validity paragraph, written once**: B and C share the repromote
   arm with opposite signs, so shared-arm draw noise biases a naive ρ
   negative. The instrument already handles this — cross-set products
   (B₁·C₂ + B₂·C₁), ref-noise subtraction, and the same-set bias check
   column — so the deliverable is the explicit comparison
   `I_sameset_biascheck` vs cross-set I per bin, closing the "is the
   headline an artifact?" question from committed numbers alone.
3. **ρ_768 vs E9's window**: E9 located the 768 reduction-failure
   window center at the crossing (σ ≈ 0.688); tabulate ρ and the
   amplitude ratio across both routes' windows so 19.1's prediction has
   a frozen target curve.

## 19.0 — RESULTS (DONE 2026-08-06; `reads_190.py` → `frozen_target_190.json`)

All numbers from the committed E14 ledgers, reliability-gated at
rel_cos ≥ 0.5, both data refs (reenc = primary, matching `ledger.json`).

1. **Completeness ordering CONFIRMED** under both refs. Mean cancellation
   fraction 1 − h(B+C)/(h(B)+h(C)) over reliable bins, 896/768/512:
   **0.770 / 0.725 / 0.631** (reenc ref); 0.638 / 0.593 / 0.343 (native
   ref). 512 least complete, exactly the safety-map ordering the
   scale-covariance account predicts.
2. **The decoherence version is REFUTED.** 512 keeps mean |ρ| = 0.864
   (reenc) / 0.824 (native) — indistinguishable from 896's 0.860/0.727
   and 768's 0.891/0.802 — while its amplitude ratio |B⊥|/|C⊥| = √(S/F)
   never crosses 1 (0.63–0.95 across all bins). The anti-alignment
   *angle* is route-uniform; what breaks on the unsafe route is the
   *magnitude matching*. Per the decision tree: **unsafe = mismatched
   magnitudes, not decoherence** — the open half is the amplitude law
   (why |B|, |C| diverge on 512).
3. **Shared-arm validity closed.** Mean |I_same − I_cross| = 0.062; at a
   few low/mid-σ bins the naive same-set read inflates |I| up to ~2.3×
   (768 @ σ=0.0625: −1.33 vs −0.59), so the artifact is real and the
   cross-set debias is load-bearing — but the headline uses cross-set
   only, and cross-set ρ stays −0.7…−0.9 throughout. **The headline is
   not a shared-arm artifact.**
4. **Crossings vs E9.** 896: crossing at σ ≈ 0.47 (reenc) / 0.53
   (native) — consistent with E9's ≈ 0.5. 768: **no in-window crossing
   on E14's probe-matched grid** — the reenc-ref ratio peaks ≈ 0.99 at
   σ = 0.0875 and sits at 0.76–0.88 through the E9 window (E9's N=24
   run had 0.98 at σ ≈ 0.69; probe sets bound the claim). So the
   crossing↔window-center localization survives cleanly on 896 only; on
   768 the window sits where the ratio is *closest to* 1 from below.
   The frozen target for 19.1 is E14's table, not E9's.
5. Endpoint σ=1 and gate-failed cells are non-verdict-bearing
   (ratio-of-small-numbers; 896's endpoint fails both rel gates).

## 19.1 — closure-predicted ρ_r (CPU + one VAE encode pass; theory first)

Extend E17's fitted machinery to emit **paired branch predictions**:
with per-arm Gaussian closures r̄\*(σ; x) = σ Q⁻¹(x − μ) and the
fitted paired covariance blocks (C_ds is load-bearing and already
estimated), compute

    B_r = r̄_rp − r̄_nat        (data intervention, native grid)
    C_r = A_e r̄_dem − r̄_rp    (graph intervention, demoted data)

and the predicted ρ_r(σ) = corr(B_r⊥, C_r⊥) at the population level,
per route. The repromote arm needs one extra one-time latent encode
into `closure_latents/` (same VAE-only pattern as E17); everything
else is CPU on stored latents.

**The wager, stated up front**: E17's verdict was fail-on-amplitude
(~40% low-σ over-prediction) with shape, route-uniformity, reenc
control, and endpoint all reproduced. ρ is a **scale-free direction
statistic** — the closure gets a second, sharper shot exactly where its
recorded failure mode (level) doesn't bite. The predicted sign and
σ-profile of ρ_r are **frozen in this record before 19.2 runs** —
theory predicts, then the instrument scores, per the line's
full-map-as-confirmation-only principle.

If the closure predicts ρ_r < 0 with roughly the measured profile, the
anti-alignment is *derivable from second-order data statistics* — the
interaction term graduates from measured to derived, which is the
single highest-value theoretical outcome available to this line.

**Instrument (`bench/run_closure_rho.py`, 2026-08-06) — the A_e pin.**
The comparison grid is pinned to the **demoted (lo) grid**, with
native-grid arms brought down by the instrument's area-downsample D_e
(the same operator as `run_prior_distance.py`'s scalar read and E11's
`resid_structure.py`) — i.e. B_r = D_e r̂_rp − D_e r̂_ref,
C_r = r̂_dem − D_e r̂_rp, so **B_r + C_r = r̂_dem − D_e r̂_ref exactly**,
the measured mismatch object — mirroring the g-ledger's exactness. This
replaces the proposal's original up-sampling A_e formulation: an
upsampled C would carry the promote operator's smoothing signature in
one branch only. ⊥ projects out D_e r̂_native per image; pooling is by
summed inner products across images (per-image cosines are secondary —
E11's norm-only verdict). Grid is fully probe-matched to E14: tier-1024
native, routes {896, 768, 512}, E14's 40-image list as holdout, a
disjoint stratified 40-image tier-1024 fit split, σ = E14's 15 centers.
Controls: (a) **cross-fit** — B and C share the repromote arm's fitted
model, whose estimation error enters with opposite signs and biases a
naive ρ_r negative (the residual-level analog of `I_sameset`); the fit
split is halved and B/C use different halves' rp models, symmetrized;
(b) **operator sensitivity** — the ledger re-pooled with bicubic
instead of area; (c) both data refs, matching the E14 ledger pair.
Coherence criterion, pre-registered: the prediction counts as coherent
iff all three closures (diag/block/octave) agree in ρ_r sign per bin at
σ ≥ 0.3 (low-σ reported, not verdict-bearing — E17's caution).

## 19.1 — FROZEN PREDICTION (committed 2026-08-06, before any 19.2 run)

Run `bench/results/20260806-2223-e19-closure-rho` (holdout = E14's 40
probe images; fit = 40 disjoint tier-1024 images); summary in
`prediction_191.json`. The wager resolves as a **qualified yes with a
committed magnitude**:

1. **Sign: ρ_r < 0 at every (route, bin), all three closures, both
   refs** — the pre-registered coherence criterion passes on all three
   routes. The Gaussian closure *does* place the anti-alignment at the
   residual level.
2. **Magnitude: weak.** At the verdict bins (σ ≥ 0.3, area, reenc ref)
   ρ_r ≈ **−0.06…−0.14** (block/diag; octave within 0.02), deepening to
   −0.22/−0.31/−0.39 (896/768/512) only at σ = 1. Far shallower than
   the measured g-level ρ ≈ −0.7…−0.9. Per-image cosines are equally
   weak (−0.10 ± 0.06 at σ=0.57) — the pooled read is not hiding strong
   per-image anti-alignment.
3. **Predicted amplitude ratio |B_r⊥|/|C_r⊥| ≈ 0.21–0.61**, rising with
   σ, **no crossings on any route** — misses the g-ledger's measured
   896 crossing at σ ≈ 0.47. σ-profile: |ρ_r| is U-shaped (largest at
   the σ extremes, weakest at σ ≈ 0.17–0.43), unlike the measured
   g-level profile (deep and flat through mid σ).
4. **Controls.** Cross-fit ≈ same-fit (Δρ ≤ 0.01) — the negative sign
   is not the shared-rp-model artifact. Bicubic pooling is 2–3× more
   negative than area (sign stable) — the pinned area operator reads as
   the conservative magnitude; operator choice owns magnitude, not
   sign. Holdout ≈ fit split (no overfitting).

**Pre-registered reading rule for 19.2** (fixing the "strongly negative"
threshold my proposal left loose): score measured ρ_r against this
band at the verdict bins, reenc ref, area operator, same estimand.

- Measured ρ_r within/below the closure band (|ρ_r| ≲ 0.35): the
  r-level seed is **derivable and weak** → the *depth* of the g-level
  anti-alignment is created through Jᵀ; L2/L3 take over with the
  closure as the seed account.
- Measured ρ_r deep (|ρ_r| ≥ 0.5) with the g-level profile: the
  residual level owns the cancellation and the second-order closure
  under-predicts it → beyond-Gaussian/caption-conditional closure is
  the theory home (E17's named missing ingredients).
- Measured ρ_r ≥ 0 at verdict bins: the closure's sign prediction is
  falsified at the trained model; the interaction is Jᵀ-born outright.

## 19.2 — the r-level ledger (forward-only GPU; the L1 discriminator)

Measure ρ_r on the trained model: `run_prior_distance.py` gains a
repromote arm (instrument delta — the repromote pixel path already
exists in the probe's latent prep) and runs with `--save_residuals` on
a probe-matched image list, emitting measured B_r/C_r per bin with the
instrument's split-half correction. Forward-only — no backward, and
latent-sized vectors instead of 311 MB parameter vectors — several
times cheaper per bin than the E14 run.

Estimand discipline: the g-ledger's B/C are contrasts of **aggregate
mean** gradients, so ρ_r must be computed on the aggregate mean
residual contrast (pooled over images), not per-image — E11's
norm-only verdict says per-image directions are idiosyncratic, and a
per-image ρ_r would measure a different (and noisier) object.

**Decision**: measured ρ_r strongly negative → anti-alignment is born
at the residual level; 19.1's closure comparison decides whether it is
*derivable* there, and the graph branch is demoted to an amplitude
modifier. ρ_r ≈ 0 (or weakly negative) while ρ_g ≈ −0.9 → the
Jacobian creates it; L2/L3 take over and the theory target becomes the
shared low-dimensional range of the early-block gradient operator.

## 19.3 — depth-resolved ρ_ℓ (re-analysis if the store survives; else one reduced-grid rerun)

`--per_group` is bookkeeping over the same flat gradient vectors, so a
**full-vector arm-sum store already contains the depth ledger**: slice
each stored arm sum by the parameter layout (module type × 28 blocks,
the Q2 J-decomposition) and run `bc_ledger` per slice → B_ℓ/C_ℓ/ρ_ℓ.

- If E14's `arm_sums/` still exists on the training box (75 GB,
  never shipped; absent from a fresh clone), this is **pure
  re-analysis** — an afternoon of CPU/GPU dot products, zero forwards.
- If it was reclaimed: one reduced-grid rerun (crossing ±1 bin +
  endpoint, routes {896, 768}, `--repromote --keep_arm_sums
  --self_floor --deterministic`, probe-matched) — a few GPU-hours and
  a ~15–20 GB store instead of E14's 75.

Pre-registered: Q2/G10 predict the anti-alignment concentrates in the
early-block band (~3–8) if the graph side is PE-originated propagation;
depth-uniform ρ_ℓ instead corroborates the r-level account (consistency
cross-check with 19.2 — the two must agree on which branch owns it).

**Shared delta with E18**: E18's kernel hook (per-draw k-dim
projections in `grad_estimate_binned`) makes this storage question
moot if it lands first — block-keyed projections give ρ_ℓ at MB scale.
One implementation should serve both records.

## 19.4 — the PI-align causal arm (few bins, one route pair)

`--pi_align --repromote` in ONE process (kernel-path rule): the
`<e>pi` arm re-runs the demoted-graph forward with PI-stretched RoPE
phases, so C_pi = ḡ_dem,pi − ḡ_rp isolates "graph minus its phase
geometry". If the graph branch's anti-aligned component is
RoPE-mediated, C must **rotate** (ρ toward 0) and the crossing must
move, not merely shrink.

Sign pre-registered with G11's qualifier built in: the clean read is
**noise-dominated bins only** (top of the window + endpoint) — at mid
σ PI is off-manifold with content (768pi measured *worse* through
σ 0.56–0.81), so a mid-σ "worse" result is expected and is not a
falsification. Route 768 primary (largest RoPE_e), 896 as the
small-floor control. Cost: 2 routes × ~4 bins, one process, hours.

## Decision tree

| observation | conclusion | theory home / next |
|---|---|---|
| 19.1 predicts ρ_r < 0 AND 19.2 measures it | interaction derivable from second-order data statistics | closure extension = paper-2 §3; graph branch is an amplitude modifier |
| 19.2: ρ_r < 0, but 19.1's closure misses it | residual-level but beyond-Gaussian | non-Gaussian/caption-conditional closure next (E17's named missing ingredients; E11 `--uncond` rerun feeds it) |
| 19.2: ρ_r ≈ 0 while ρ_g ≈ −0.9 | Jᵀ creates it | 19.3/19.4 localize; theory target = early-block operator's shared range |
| 19.3: ρ_ℓ concentrated in blocks ~3–8 + 19.4 rotates C | graph side is PE-geometry-mediated | mechanism paragraph writes itself; connects G10's origin-side RoPE_e |
| 19.0: 512 keeps ρ ≈ −0.9 with broken amplitude ratio | unsafe = mismatched magnitudes, not decoherence | amplitude law (why \|B\|,\|C\| diverge) becomes the open half |
| 19.0: `I_sameset` − cross-set I large | part of the headline is shared-arm noise | re-score E14's ρ before anything else proceeds |

## Kill switches / honesty

- **Unit-honesty inherited from E9/E14**: at plateau magnitudes
  κ ≈ 0.7–0.9 the S/F/I quadratic is out of its truncation domain —
  magnitudes read only via h(·); ρ, signs, and localization are the
  licensed reads (ρ is an angle statistic and stays in-domain).
- **Reliability gate**: any (route, bin) cell with rel_cos_B/C < 0.5
  does not read (E14's reproducibility floor).
- **Low-σ caution transfers**: E17 recorded that the measured low-σ
  excess is itself the least certain point (split-half floor ≈ 0.45
  against d ≈ 1.33) — 19.2's ρ_r verdict leans on mid/high-σ bins
  first, with low-σ reported but not verdict-bearing.
- **Conventions**: probe-matched lists when levels are compared
  (probe sets bound the claim); every cross-arm cosine inside one
  process (kernel-path chaos); bin-width weights on any WLS over a
  segmented grid.
- 19.1's prediction must be committed (this file + the run record)
  **before** 19.2 is submitted, or the theory-first claim is void.

## Groundings

- **E14** (`runs/20260801-2304-e14-ledger-probematched`): ρ ≈
  −0.7…−0.9 at every σ, cross-set debiased, relB/relC mostly 0.6–0.99;
  \|B⊥\|/\|C⊥\| crossings 896 ≈ 0.5, 768 = 0.688 (E9); native-ref
  h(B) ≈ 104–112% of the plateau, immediately half-cancelled; H-d:
  assumption (iii) fails below σ ≈ 0.45.
- **E17**: all three second-order closures fail the amplitude bar but
  reproduce shape (Pearson 0.94–0.97), route-uniformity, reenc ≈ 0,
  and the endpoint — the basis of 19.1's scale-free wager.
- **E11**: mismatch directions are image-specific (norm-only) — fixes
  19.2's estimand at the aggregate level; `--save_residuals` is the
  harness precedent.
- **E15**: per-sample ‖Δ‖ ≈ 0.7–1.6‖g‖ vs aggregate 0.15–0.35 — the
  cancellation is an *aggregate* phenomenon, consistent with 19.2's
  pooled estimand.
- **Q2/G10/G11**: depth band 3–8 ≈ 3×; RoPE_e erased by PI at the
  endpoint on 768; PI off-manifold with content at mid σ — 19.3's
  localization prior and 19.4's regime restriction.
- **E7 + Q3**: map adapter-agnostic, floor level checkpoint-dependent;
  Q3's mixed-res-trained adapter remains the designated
  training-distribution falsifier for whichever account survives —
  scale covariance along the demotion diagonal is *trained*, so an
  adapter trained off-diagonal should degrade the cancellation.
- **E18**: E14's store is per-bin arm sums only (no per-draw vectors);
  the per-draw projection hook is specified there — 19.3 shares it.

## Cost ladder

| item | GPU | note |
|---|---|---|
| 19.0 | none | committed JSON only |
| 19.1 | one VAE encode pass | then CPU on stored latents (E17 pattern) |
| 19.2 | few hours | forward-only, latent-sized vectors |
| 19.3 | none **or** few hours | re-analysis if the store survives; else reduced grid, ~15–20 GB store |
| 19.4 | hours | 2 routes × ~4 bins, one process |

No verdict-grid-scale spend anywhere; the full σ-map stays reserved
for confirming whatever theory 19.1/19.2 leave standing.
