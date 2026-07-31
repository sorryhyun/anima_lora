# action.md — open items from the review correspondence

*2026-07-30, reduced 2026-07-31. Input: the correspondence in
[`review/`](review/) — `question.md` (our circulated open-questions
note), `response.md` (the review it drew; citations verified real and
on-topic), `additional_question.md` (our follow-up). This file tracks
only what is still OPEN plus the record of what was applied to the
manuscript. Experiment verdicts and full numbers live one place only:
`paper_bench/experiments/<eN>/README.md`.*

## Discharged since (verdicts live in paper_bench)

| item | verdict | record |
|---|---|---|
| §4.6 — the 768 anomaly | **branch (i), negative interference**; I_768 < 0 at every bin, window center σ ≈ 0.69; interference erases ~70–80% of the additively-predicted gap; quote magnitudes via h(·), never S/F/I | `paper_bench/experiments/e9/` |
| §4.5 — reenc-proxy caveat | **CLEARED** — |B⊥| vs reenc within ~4% of vs native at the signal-carrying bins | `paper_bench/experiments/e9/` |
| §4.3 — target-content share | **parallel landing** (κ∥ ≫ κ⊥) | `paper_bench/experiments/e10/` |
| §4.4 — Δr̄ structure | **norm-only** → universal amplitude law | `paper_bench/experiments/e11/` |
| §4.4/Q5 — posterior-budget (lossy-code saturation) | **REFUTED on both probes**; native-only pre-screening void. Do not re-propose without new evidence | `paper_bench/experiments/e12/` |
| §4.6 — spectral account → gap bridge | **fails both directions at every δ**; δ inert at the curve level | `paper_bench/experiments/e8/` |

## Open items

### §4.5 — the 512-banded leg [runs later]

When it runs, it is an **intervention-effect measurement**, not a
falsification test — the "banded can erase at most RoPE_512 ≈ 0.10"
bound is void through cross-terms (retracted in
`review/additional_question.md`). Design: the three-piece report
F_rope / F_rest / I_rope,rest via C_plain/C_rest vector differences
(kept in `review/response.md` §2).

### §3.3/§4.6 — floor-law discriminator [ladder run, not scheduled]

The manuscript now presents {e^(−N/τ), e^(−√N/ℓ), N^(−p)} as
unidentified at our operating points (secant correction within
calibration slack — checked). The discriminating run is the
fixed-target/varied-source x-zero token ladder + per-head
entropy-matched temperature at σ=1, registered as the E-series
follow-on, not yet scheduled. Cheaper-discriminator ask on record as
AQ3.

### §4.4 — residual arm on hold

The `--uncond` rerun (T5("") sidecar conditioning) closes the
caption-conditioning caveat — full captions may pin composition so
hard the model never falls back to a canvas-size prior. Cross-image
consistency appearing at high σ there would partially rehabilitate the
composition-prior story (as a caption-suppressed mechanism); another
null retires it. Implemented; first launch stopped mid-run 2026-07-30
(user hold) — relaunch pending. (`paper_bench/experiments/e11/`)

### §4.1/§6 estimands + E3 — batch aggregation [phrasing APPLIED; E3 open]

*Phrasing fix APPLIED 2026-07-31: every "batch-size-1 lower
bound"/"bounds every batch size from below" claim is deleted from the
manuscript; §3.1 now states the decomposition
E[d_B] ≈ |P⊥b|²/2‖μ‖² + tr(P⊥Σ_η P⊥)/2B‖μ‖² (intercept = coherent
drift that never averages out, plus a 1/B term; monotone improvement
with B needs an unverified iid zero-mean disagreement model), and the
§6 aggregation paragraph now reads the pooled collapse through it and
names the a + b/B fit as the batch-aggregate grid's headline readout
(still [pending] the run).*

Still open with [E3](../paper_bench/experiments/e3/): the pooled-arm
run gains the **a + b/B batch-size fit** as its headline readout, and
the pre-A/B instrument for the long-horizon question becomes a paired
shadow-Adam replay (frozen optimizer state, real batch/accum —
validity-horizon questions on record as AQ5).

### Unregistered lever (salvaged from review/response.md §4; not planned)

Because the target dependence is linear, Fourier-masking x on the same
instrument gives the exact target observability spectrum
t_b = E[J^T P_b x] — a cheap extension of E10 if the parallel-landing
story ever needs a per-band read.

## What does NOT change

- The (route, σ) safety map, E1's debiased floors, E2's α-slope ≈ 0
  *scalar* result (E10 reinterprets it, doesn't contradict it), E4's
  throughput numbers, E5's qualified PASS, E7's factorial reads.
- The endpoint floor decomposition Floor_e = RoPE_e + Resid_e as a
  **σ=1 measurement** — what is retracted is only the implication that
  the split extends in σ as two flat constants, which
  `review/question.md` already flagged.
- The yarnsig recipe and its measured in-window improvement; the
  off-manifold verdict on uniform PI in-window.
- The response's λ_b(σ) band-gate formula is NOT adopted: plausible
  direction, but asserted rather than derived (see
  `review/additional_question.md`); the partial-alignment dial remains our
  planned estimator for RoPE_e(σ).
- The mixed derivative ∂ḡ/∂λ_b via autograd is noted but not planned:
  double-backward through the compiled DiT is impractical on our
  hardware; the paired finite-difference version rides the existing
  yarn-arm machinery if we run it.

## Applied 2026-07-31 (record)

PR #79 (route-explicit d notation, gap-only excess, derived
subtraction; spectral null → "spectral account" reframe) merged into
`sigma-lowres-yarnsig`. On top of it, in `main.tex`/`appendix.tex`:
the §4.6 branch-(i) resolution ("The probe, run" + "How much of the
gap the interference erases" paragraphs, with the S/F/I-vs-h(·)
unit-honesty caveat), the appendix B/C-instrument paragraph + full
ledger Table `tab:e9ledger`, the §4.5 reenc-proxy closure numbers,
the §3.1 batch-aggregation decomposition replacing the batch-1
lower-bound claim (echoed in the §6 map paragraph with the a + b/B
readout named), and the limitations update (768 window mechanism now
measured). AQ4 scored in `review/additional_question.md` (center
CONFIRMED, edge rule did not fire as stated, route disjuncts
half-right); its E9 registry entry flipped to LANDED.

Same day, E8.3 closed (`paper_bench/experiments/e8/e83_bridge.py`,
CPU-only over existing rows): the spec→gap overlay + t\*(δ) sweep are
in §4.6 as Fig. `fig:e83` and the last spectral-account [pending] is
stripped.

## Applied 2026-07-30 (record)

All unblocked manuscript changes are in `main.tex`/`appendix.tex`
(rebuilt; citations resolve): the §4.3 parallel-landing mechanism
paragraph + "angular estimand" qualifiers, the §4.4 direction-resolved
(norm-only) paragraph with the Divergence-is-Uncertainty cite and
rank-one exclusion, "route-uniform in amplitude" phrasing everywhere,
the §4.6 two-branch pre-registration + vector-resolved probe design,
the floor-law "calibrated interpolation" paragraph (κ_eff² refit
τ ≈ 860 tok; family predictions +0.096/+0.085/+0.075 at n=2160,
computed from the e5 fitted floors 512 +0.264 @1016 tok / 896 +0.039
@3012 tok), the Eq. 5 validity domain (κ_eff ≈ 1.02 at 512), the
limitations additions, and the four citations (li2025infoscale,
liu2026tide, calvello2024continuum, xing2026divunc).

Data note recorded here only: the reviewer-facing notes quoted the 512
endpoint |t|/G as "~0.7"; `bench/results/20260730-2116-e10-kappa/
ledger.json` has 0.52/0.57 across the two draw sets, and the
manuscript says ≈ 0.5.
