# Follow-up — reply and re-ask against response.md

*2026-07-30. Companion to `question.md` and `response.md`. Short version:
we accept the two structural criticisms (the scalar probe's
identifiability confound; the invalid "≤ 0.10" erasure bound), we have
implemented the interventional vector split and the exact affine target
read, and the three cheap measurements are queued (specs at the end —
predictions welcome before the data lands). The questions below are
where we either need the derivation the response gestured at, or where
our implementation made a statistical choice the response left open.*

## What we accepted and already changed

- The pre-registered I_768 < 0 scalar test is retired as verdict object;
  the B/C vector ledger replaces it, with the two-branch pre-registration
  (I < 0 vs F(σ) collapse) recorded in `action.md` before any data.
- "Banded alignment can erase at most RoPE_512 ≈ 0.10" is retracted as a
  falsification bound (cross-terms void it); the 512-banded leg will run
  as an intervention-effect measurement with the three-piece
  F_rope/F_rest/I_rope,rest vector report.
- The α-sweep reanalysis runs as exact vector differences
  t = ḡ(1) − ḡ(0) at shared seeds; κ∥/κ⊥ + observability norms replace
  the scalar-slope read.
- Normalization-recalibration is dropped from the Q2 probe list
  (per-token norms; no explicit N statistic).
- Floor-law refits move to κ_eff² units with the three-family comparison.

## AQ1 — is our noise treatment of the B/C split right?

Every arm runs with two independent draw sets. Our estimator: second
moments by **cross-set inner products** (⟨B₁⊥, B₂⊥⟩ as unbiased |B⊥|²),
the interaction only from **set-crossed pairs** ⟨B₁⊥, C₂⊥⟩ + ⟨B₂⊥, C₁⊥⟩ —
because B₁ = ḡ_rp,1 − ḡ₀ and C₁ = ḡ_dem,1 − ḡ_rp,1 share the repromote
set-1 draws with opposite sign, and that shared noise would *manufacture*
a negative interaction (the exact artifact the probe exists to detect).
The reference ḡ₀ is shared across both B replicates, so S is additionally
corrected by the reference-noise power estimated from the native set
difference (|(a−b)⊥|²/4). Same-set I is reported as a bias check.
Question: is there a strictly better estimator for I at fixed forwards —
e.g. a four-way draw-set split trading variance for the cross-term, or a
jackknife over images — and is our shared-reference correction for S the
one you would use, or would you re-reference B to the reenc arm and eat
its (smaller-N) noise instead?

## AQ2 — the band gate: derivation or dial?

λ_b(σ) ≈ σ²/(σ² + (1−σ)² P_b^sens) is stated as "plausible". Two asks:
(a) is there a derivation sketch — even heuristic — that fixes the
functional form, or is it a posterior-shaped ansatz to be fit? (b) The
proposed P_b^sens lives in the RoPE-gradient tangent ∂ḡ/∂λ_b. Full
autograd for that mixed derivative means double-backward through a
compiled DiT — impractical on our hardware. For the paired
finite-difference version riding our existing banded-alignment machinery:
how many λ points per (band, σ-bin) would you consider the minimum for a
usable RoPE_e(σ) estimate, given our per-estimate draw noise is the
dominant cost? A 2-point secant per band at 4 σ-bins is affordable; a
5-point local fit per band is not.

## AQ3 — discriminating the floor-law family

We checked your secant form against the existing anchors: at our
operating points e^(−N₀/τ) is small enough that target-only and
spectral-tail-secant fits agree within calibration slack — the current
data cannot discriminate. The clean run is your fixed-target,
varied-source ladder (x-zero, so source content is irrelevant). Before we
spend that GPU: is there any cheaper discriminator — e.g. does the
spectral-tail account make a sign or curvature prediction for
*non-nested* grids (source and target grids where neither divides the
other), or a prediction about the per-block distribution of the floor
(our --per_group machinery already splits the gap by block) that
target-only capacity cannot mimic?

## AQ4 — Goldilocks, pre-registered — and its fallback

From the E9 ledger we will read, per σ-bin on 896/768/512:
|B⊥|, |C⊥|, ρ = cos(B⊥, C⊥). Pre-registered against your account:
(i) the 768 zero-gap window center sits at the σ where
|B⊥(σ)| ≈ |C⊥(σ)|; (ii) the window edges sit where ρ crosses −b/2c;
(iii) 896 shows no window because c is too small everywhere, 512 none
because b ≪ c and/or ρ is route-differently directed. One question: if
the data instead lands on your alternative branch — F_768(σ) collapsing
below its endpoint value in-window with I ≈ 0 — is there an
amplitude-matching *story for the collapse itself* (e.g. the positional
component of C rotating toward ḡ₀ as content vanishes), or does that
branch simply hand the question back to a σ-resolved RoPE theory (AQ2)?

## AQ5 — shadow-Adam replay: validity horizon

For the frozen-state replay you propose (same optimizer state +
preconditioner applied to both gradient streams): over how many steps do
you trust it before state divergence (v̂ updated by different g²
histories) makes the paired read unfaithful — is this a one-step
instrument repeated at checkpoints along a real trajectory, or do you
mean to integrate it? And which scalar do you rank on: ‖Δupdate‖/‖update‖
mean drift, or a cosine in update space (which our whole instrument
suite would suggest), and with what normalization for the covariance
term so intercept and 1/B slope are comparable across routes?

## AQ6 — the transfer-matrix probe, sized

For M_σ[a,b] = E‖P_a D_z v̂ P_b‖²_F via Fourier-band JVPs + Hutchinson:
with ~4k tokens × 16 channels per grid, how many Hutchinson probes per
(band-pair, σ) did you have in mind for a stable low-rank read — and is
there a cheaper first pass that only tests *rank-one-ness* of the
destroyed-band columns (e.g. power iteration on P_a D_z v̂ P_b for the
top mode and a trace estimate for the remainder) before paying for the
full matrix? **Update: E11 already landed (below) and the rank-one
target is implausible** — so the sharper version of this question is:
does the posterior-covariance account have a natural regime that
produces *uniform amplitude with image- and route-specific directions*
(e.g. Cov(x|z,c) with route-independent trace but image-dependent
eigenvectors), and if so, what is the cheapest statistic of D_z v̂ that
tests amplitude-universality directly rather than mode-sharing?

## What will exist shortly (predict now if inclined)

- **E9** (queued): routes {896, 768, 512}, σ-window 0.5–1.0 in 4 bins +
  endpoint, D=8, two draw sets/arm, repromote arms, full arm-mean
  vectors retained. Emits S/F/I per (route, bin) + h() exact angles.
- **E10** (LANDED 2026-07-30): endpoint-only, α ∈ {0,1}, exact
  per-image t-vectors + aggregate; κ∥/κ⊥ per route with a-vs-b null.
  **Verdict: parallel landing, confirmed with sign and order.**
  Aggregate: |t_src|/G ≈ 2.23; δt has κ∥ = −0.75/−1.18/−1.86 on
  896/768/512 vs κ⊥ = 0.09/0.14/0.20 (8–9× parallel dominance,
  rel_cos_dt ≥ 0.995 across independent draw sets; reenc control at
  the noise floor with irreproducible direction). Demotion shortens the
  target-content gradient along ĝ_src, rotating it only on the
  harshest route (cos(t_512, t_src) = 0.74). Your three-way readout
  resolved to option one — "large parallel, small orthogonal" — and
  the α-sweep's scalar null is thereby explained rather than
  contradicted (κ⊥ enters the cosine at second order, 0.004–0.02,
  below every floor it would need to move).
- **E11** (LANDED 2026-07-30): Δr̄ per route pair at 5 σ, split-half
  corrected pairwise cosines + normalized stacked SVD top-mode share.
  **Verdict: norm-only.** Non-adjacent-pair directions near-orthogonal
  at low σ (corrected cos ≈ 0–0.08), weakly shared at high σ
  (+0.2–0.33); SVD top share 0.33–0.36 vs 0.25 uniform. Follow-ups on
  the same vectors: cross-image direction consistency ≈ 0 everywhere
  (the mismatch direction is image-specific — a grid-conditional
  composition prior is refuted as carrier under full captions; an
  uncond-conditioning rerun is queued to close that caveat), and Δr̄
  energy shifts to low spatial frequencies as σ rises (low-third share
  ≈ 0.40 → 0.67 by σ = 0.875).

Falsifiable statements on record: our account predicted E11 would land
rank-one-ish (m is one mode) — **it did not**; we log the miss and
weaken the paper's "universal m" to a universal *amplitude* law. The
standing puzzle handed to Q5 is now sharper: one m(σ) amplitude across
routes whose mismatch directions are image- and route-specific. E10
also landed (above): κ reads grew with severity and were κ∥-heavy —
the parallel-landing branch of your three-way readout, consistent with
the target-only floor law's endpoint equivalences. Still pending: your
Goldilocks predicts E9's window localization (AQ4).
