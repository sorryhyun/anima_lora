# action.md — open items after the response to question.md

*2026-07-30. Input: `question.md` (our circulated open-questions note)
and `response.md` (the review it drew; citations verified real and
on-topic). This file now tracks only the OPEN items. Everything already
applied to the manuscript was folded in 2026-07-30 (record at the
bottom); the E10/E11 verdicts + result paths live in
`paper_bench/required_experiments.md` and the manuscript (§4.3 / §4.4),
and the response's fully-absorbed sections (§1 interventional split,
§3 Goldilocks, §4 exact-α, §6 floor law) were pruned from `response.md`
the same day — their content survives in the manuscript, this file,
and `additional_question.md` AQ4.*

## Open items

### §4.6 — the 768 anomaly [GATED ON E9]

- **E9 relaunch pending** (stopped before start 2026-07-30, user hold).
  When it lands: publish the vector-resolved ledger — per (route,
  σ-bin) S = |B⊥|²/2G², F = |C⊥|²/2G², I = ⟨B⊥,C⊥⟩/G², with
  cross-draw-set debiasing and exact counterfactual angles
  h(u) = 1 − cos(ḡ, ḡ+u) — and report which pre-registered branch the
  data picks: (i) I_768(σ) < 0 with the amplitude-matching
  localization (window center where |B⊥(σ)| ≈ |C⊥(σ)|), or (ii)
  F_768(σ) collapsing below its endpoint value in-window (which would
  also rewrite Q1's "σ-flat Floor_e" reduction). The fork + the
  scalar-probe conflation + the B/C design are already written into
  §4.6; Goldilocks scoring spec is AQ4 in `additional_question.md`.
- **Reenc-proxy caveat** (§4.5 ledger [pending]): E9's repromote arm
  carries the demote arm's own down+up resize, so B measured against
  reenc vs against native bounds the pipeline-cost share empirically.
  Clear the [pending] with E9 numbers.

### §4.5 — the 512-banded leg [runs later]

When it runs, it is an **intervention-effect measurement**, not a
falsification test — the "banded can erase at most RoPE_512 ≈ 0.10"
bound is void through cross-terms (retracted in
`additional_question.md`). Design: the three-piece report
F_rope / F_rest / I_rope,rest via C_plain/C_rest vector differences
(kept in `response.md` §2).

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
(user hold) — relaunch pending.

### §4.4 / Q5 — posterior-budget (quantization-regime) hypothesis [E12, proposed]

*Added 2026-07-30 after reading Lottery Prior (ICML 2026,
26571_Lottery_Prior_Randomized): in their Thm 3.1 the operator enters
the error bound only through codebook distortion — error scale is a
property of the code, not of which measurements were destroyed; and
high-rate quantization error has concentrated magnitude with isotropic,
input-decorrelated direction. That is E11's exact signature.*

- **Hypothesis**: the denoiser at σ acts as a lossy code; a demotion
  perturbation above the code's resolution "re-rolls" posterior detail,
  so the response amplitude saturates at the posterior-uncertainty
  scale (scalar in (model, σ) → route-uniform m(σ)) while the landing
  direction is a fresh draw (image- and route-specific, near-orthogonal
  in high dim). Post-dicts all four E11 facts: norm-only uniformity;
  reenc ≤ 0.02 vs routes 0.36–0.89 (sub-cell vs on-plateau); the weak
  shared component only at high σ (posterior → prior mean shift — same
  object the `--uncond` rerun probes); low-frequency migration. What
  does NOT transfer: RD water-filling itself (distortion ordered by
  discarded energy = the refuted diagonal null); only the saturation
  regime does. Directly answers AQ6's sharpened question ("cheapest
  statistic of D_z v̂ testing amplitude-universality").
- **Probe A — posterior-trace curve** (~1–1.5 h GPU full, ~20 min
  pilot): Hutchinson trace of Cov(x|z,c) via the Divergence-is-
  Uncertainty identity on D_z v̂ (native grid only, forward-only,
  FD-JVP: baseline + k=8 probes per (image, σ, draw); N=40 × 5 σ ×
  D=4 ≈ 7.2k forwards). **Prediction**: m(σ) ∝ √tr-Cov profile,
  route-independently. Shape mismatch kills the hypothesis; absolute-
  scale match would make the data term parameter-free (low prior —
  subset ambiguity, test shape first).
- **Probe B — ε-sweep saturation** (~15–20 min GPU): response amplitude
  ‖v̂(z+εu) − v̂(z)‖ vs ε at matched norms. **Prediction**: plateau at
  m(σ) with reenc below the knee, all routes on the plateau; plateau
  departure predicts the account's validity domain (candidate story
  for 512's rotation/κ_eff ≈ 1).
- **Gotchas**: `torch.func.jvp` likely won't compose with the compiled
  DiT — use finite differences with 2 ε points for a linearity check;
  paired FD REQUIRES `--deterministic` (chaos floor 0.41 swallows small
  ε signal). Forward-only → rides `run_prior_distance.py` machinery
  (~half-day implementation). Daemon-queued; sequence against E9 +
  `--uncond` relaunches.
- **Payoff if confirmed**: native-only pre-screening — the data term's
  σ-shape without demotion arms, i.e. 2/3 of a safety map for a new
  checkpoint/model before any demotion run (upgrades the conclusion's
  "cheap general test"); plus a predicted breakdown locus. Existing
  1024-tier map numbers do not move either way (m is already measured
  there).

### §4.1/§6 estimands + E3 — batch aggregation [FIX with E3]

Delete every "per-example map is the batch-size-1 lower bound"
phrasing. The correct statement is the decomposition
E[d_B] ≈ |P⊥b|²/2‖μ‖² + tr(P⊥Σ_η P⊥)/2B‖μ‖²: an intercept (coherent
drift that never averages out) plus a 1/B term; monotone improvement
with B needs an iid zero-mean disagreement model we have not verified.
E3's pooled-arm run gains an **a + b/B batch-size fit** as its
headline readout, and the pre-A/B instrument for the long-horizon
question becomes a paired shadow-Adam replay (frozen optimizer state,
real batch/accum — validity-horizon questions on record as AQ5).

### Unregistered lever (salvaged from response.md §4; not planned)

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
  the split extends in σ as two flat constants, which question.md
  already flagged.
- The yarnsig recipe and its measured in-window improvement; the
  off-manifold verdict on uniform PI in-window.
- The response's λ_b(σ) band-gate formula is NOT adopted: plausible
  direction, but asserted rather than derived (see
  `additional_question.md`); the partial-alignment dial remains our
  planned estimator for RoPE_e(σ).
- The mixed derivative ∂ḡ/∂λ_b via autograd is noted but not planned:
  double-backward through the compiled DiT is impractical on our
  hardware; the paired finite-difference version rides the existing
  yarn-arm machinery if we run it.

## Experiment registry (details in `paper_bench/required_experiments.md`)

| id | run | reads | status 2026-07-30 |
|---|---|---|---|
| E9 | `run_sigma_probe.py --repromote --keep_arm_sums --self_floor`, routes 896/768/512, window 0.5–1.0 + endpoint | B/C/S/F/I per bin, F(σ) directly, Goldilocks localization, reenc-proxy closure | stopped before start (user hold) — relaunch pending |
| E10 | `--target_alpha 0,1 --target_kappa`, endpoint-only | exact t-vectors, κ∥/κ⊥ | **DONE — parallel landing**; in manuscript §4.3 + `required_experiments.md` |
| E11 | `run_prior_distance.py --save_residuals` + `resid_structure.py` | Δr̄ direction structure | **DONE — norm-only**; in manuscript §4.4 + `required_experiments.md`; `--uncond` rerun pending (above) |
| E12 | posterior-trace (Hutchinson on D_z v̂) + ε-sweep saturation, native-only, FD, `--deterministic` | m(σ) ∝ √tr-Cov profile; saturation plateau + validity domain | proposed (above) — not scheduled; ~1.5 h GPU full / ~20 min pilot |

Analysis lands via `paper_bench/vector_ledger.py` (E9/E10) and
`paper_bench/resid_structure.py` (E11); both are CPU reanalyses of the
runs' saved vectors, so re-reads don't cost GPU time.

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
