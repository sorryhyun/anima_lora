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

### §4.6 — the 768 anomaly [E9 DONE — BRANCH (i): NEGATIVE INTERFERENCE]

*Verdict 2026-07-31 (`bench/results/20260731-0721/` + `ledger.json`;
full numbers in `paper_bench/required_experiments.md` E9): the data
picks branch (i). I_768(σ) < 0 at every bin (−0.31 → −0.014) with B, C
near anti-parallel (ρ ≈ −0.93 in-window), and the amplitude-matching
localization puts the 768 window center at σ ≈ 0.69 (|B⊥|/|C⊥| = 0.98
at the 0.6875 bin). Branch (ii) did not fire: F_768 decreases
monotonically to its endpoint value (0.200 → 0.0036) and never drops
below it in-window, so Q1's σ=1 endpoint reduction is untouched. Note
the anti-parallel cancellation is universal across routes
(ρ −0.62…−1.24); what distinguishes them is amplitude matching — 896
matched but low-amplitude (net S+F+I ≤ +0.012), 768 matched at
mid-window (≤ +0.029), 512 mismatched at low σ (|C⊥|/|B⊥| ≈ 1.8, net
up to +0.197). **Manuscript: APPLIED 2026-07-31** — ledger +
branch-(i) resolution written into §4.6 (two new paragraphs after the
fork text), the full ledger + instrument details added as an appendix
table (`tab:e9ledger`), the Goldilocks localization scored per AQ4 in
`additional_question.md`, and the §4.6 [pending] marker stripped (the
limitations mention updated too).*

- **How the interference account narrows the predicted gap**
  (quantified 2026-07-31 from `ledger.json`; APPLIED to §4.6 as the
  "How much of the gap the interference erases" paragraph, incl. the
  honest caveat). The actual demote gap is exactly h(B+C) (ḡ_dem = ḡ₀+B+C by
  construction), so each account's predicted/actual ratio is directly
  computable per (route, bin):
  - **No-interference scalar account (S+F)**: overpredicts the
    in-window gap **2.4–3.8× at 768** (2.4/3.3/3.8 across the three
    in-window bins), 1.6–5.8× at 896, 1.2–3.0× at 512.
  - **Fully additive counterfactuals (h_B + h_C)**: overpredicts
    3.5–8× — the realized gap is only **~20–30% of the additive sum**
    in-window; i.e. ~70–80% of the additively-predicted gap is erased
    by the interference. That is the quantitative content of
    branch (i).
  - **Vector account (B+C)**: exact by construction — including I is
    what removes the overprediction.
  - **Honest caveat for the manuscript**: the quadratic scalar ledger
    S+F+I itself *under*predicts in-window magnitudes ~3–4× (median
    ratio 0.24×) because |B⊥|, |C⊥| ≈ 0.5–1.0 G there — the
    small-perturbation truncation is out of its domain. So in §4.6:
    use S/F/I for **sign, decomposition, and window localization**;
    quote **magnitudes via the exact h(·) counterfactuals** (h_B, h_C,
    h(B+C) are all in `ledger.json`). Do not cite S/F/I values as gap
    magnitudes.
- **Reenc-proxy caveat — CLEARED** (§4.5 ledger [pending] → strip):
  |B⊥| measured against reenc agrees with |B⊥| against native to ~4%
  at the signal-carrying low-σ bins (worst ±23% only where B is
  already small), so the down+up+encode pipeline cost is a minor share
  of the data intervention. §4.5 text APPLIED 2026-07-31 (pending
  stripped, numbers in, pointer to `tab:e9ledger`).

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

### §4.4 / Q5 — posterior-budget (quantization-regime) hypothesis [E12 DONE — REFUTED]

*Verdict 2026-07-31 (`bench/results/20260731-0023/`, full design +
numbers in `paper_bench/required_experiments.md` E12): both
pre-registered kill criteria fired. Probe A: √tr-Cov(σ) rises ~2.8×
over σ 0.375→0.875 while the absolute demotion response falls ~0.85×
— shape mismatch; no universal dabs/√tr-Cov constant (route-dependent,
CV ≈ 0.5, per-image corr ≈ −0.25). Probe B: response near-linear in ε
(slope 0.84–0.90) with no plateau through the routes' natural
amplitudes; route amplitudes track perturbation size (~1.7× apart).
So E11's route-uniform m(σ) is NOT saturation — it must come from
route-uniform delivered perturbation × a direction/σ gain (demotion
directions are ~1.5× softer than random at matched ε — a lead worth
keeping). Native-only pre-screening payoff void. AQ6's
amplitude-universality question: answered negative. Do not re-propose
the lossy-code saturation account without new evidence; the Probe A
posterior-trace instrument itself is sound and reusable
(`run_posterior_budget.py`, FD linearity 0.96–1.0).*

Original proposal (kept for the record):

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

Still open with E3: the pooled-arm run gains the **a + b/B batch-size
fit** as its headline readout, and the pre-A/B instrument for the
long-horizon question becomes a paired shadow-Adam replay (frozen
optimizer state, real batch/accum — validity-horizon questions on
record as AQ5).

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
| E9 | `run_sigma_probe.py --repromote --keep_arm_sums --self_floor`, routes 896/768/512, window 0.5–1.0 + endpoint | B/C/S/F/I per bin, F(σ) directly, Goldilocks localization, reenc-proxy closure | **DONE 2026-07-31 — branch (i), I_768 < 0, window center σ ≈ 0.69; F-collapse dead; reenc-proxy cleared** (`bench/results/20260731-0721/`); verdict above + `required_experiments.md` |
| E10 | `--target_alpha 0,1 --target_kappa`, endpoint-only | exact t-vectors, κ∥/κ⊥ | **DONE — parallel landing**; in manuscript §4.3 + `required_experiments.md` |
| E11 | `run_prior_distance.py --save_residuals` + `resid_structure.py` | Δr̄ direction structure | **DONE — norm-only**; in manuscript §4.4 + `required_experiments.md`; `--uncond` rerun pending (above) |
| E12 | posterior-trace (Hutchinson on D_z v̂) + ε-sweep saturation, native-only, FD, `--deterministic` | m(σ) ∝ √tr-Cov profile; saturation plateau + validity domain | **DONE 2026-07-31 — REFUTED both probes** (`bench/results/20260731-0023/`); verdict above + `required_experiments.md` |

Analysis lands via `paper_bench/vector_ledger.py` (E9/E10) and
`paper_bench/resid_structure.py` (E11); both are CPU reanalyses of the
runs' saved vectors, so re-reads don't cost GPU time.

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
measured). AQ4 scored in `additional_question.md` (center CONFIRMED,
edge rule did not fire as stated, route disjuncts half-right); its E9
registry entry flipped to LANDED.

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
