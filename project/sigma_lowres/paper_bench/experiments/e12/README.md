# E12 — posterior-budget (quantization-regime) probes

| | |
|---|---|
| **Status** | **DONE 2026-07-31** |
| **Verdict** | **REFUTED on both probes.** No √tr-Cov shape match, no saturation plateau — the "native-only pre-screening" payoff is void. AQ6 answered: amplitude-universality fails at the first (cheapest) test. |
| **Runs** | `bench/results/20260731-0023/` (36 images; FD linearity 0.96–1.0) |
| **Instrument** | `bench/run_posterior_budget.py` — native-only, forward-only, deterministic by default |
| **Origin** | design registered in `paper/action.md` §4.4/Q5 |
| **In the paper** | §4.4 / Q5 |

## Origin — the lossy-code hypothesis (kept for the record)

*Added 2026-07-30 after reading Lottery Prior (ICML 2026,
26571_Lottery_Prior_Randomized): in their Thm 3.1 the operator enters
the error bound only through codebook distortion — error scale is a
property of the code, not of which measurements were destroyed; and
high-rate quantization error has concentrated magnitude with isotropic,
input-decorrelated direction. That is [E11](../e11/)'s exact
signature.*

**Hypothesis.** The denoiser at σ acts as a lossy code; a demotion
perturbation above the code's resolution "re-rolls" posterior detail,
so the response amplitude saturates at the posterior-uncertainty scale
(a scalar in (model, σ) → route-uniform m(σ)) while the landing
direction is a fresh draw (image- and route-specific, near-orthogonal
in high dim). It post-dicts all four E11 facts: norm-only uniformity;
reenc ≤ 0.02 vs routes 0.36–0.89 (sub-cell vs on-plateau); the weak
shared component only at high σ (posterior → prior mean shift — the
same object the `--uncond` rerun probes); low-frequency migration.
What does NOT transfer: RD water-filling itself (distortion ordered by
discarded energy = the refuted diagonal null); only the saturation
regime does.

**Instrument gotchas (still true, instrument is reusable):**
`torch.func.jvp` does not compose with the compiled DiT — use finite
differences with 2 ε points for a linearity check; paired FD REQUIRES
`--deterministic` (the chaos floor 0.41 swallows small-ε signal).
Forward-only, so it rides `run_prior_distance.py` machinery.

## Design

- **Probe A** — Hutchinson trace of Cov(x|z,c) via the
  Divergence-is-Uncertainty identity
  tr Cov = σ²/(1−σ)·tr(I − σ·D_z v̂), Rademacher FD-JVP (δ = 2⁻⁵, 2δ
  linearity check, σ < 1 grid — the identity degenerates at σ=1).
- **Probe B** — ε-sweep saturation along repromote / reenc / random unit
  directions at matched norms plus each direction's natural amplitude,
  response in [E11](../e11/)'s rel-L2 mean-residual units.

**Predictions.** m(σ) ∝ √tr-Cov route-independently (shape mismatch
kills the hypothesis); plateau at m(σ) with reenc below the knee;
plateau departure locates the validity domain (512 rotation candidate).
Payoff if confirmed: native-only pre-screening — the data term's σ-shape
without demotion arms.

## Results — refuted on both probes

- **(A) shape mismatch** — √tr-Cov rises 1 → 1.66 → 2.81 over σ
  0.375→0.875 while the absolute nat-amplitude response *falls*
  1 → 0.85 → 0.82 on both routes; per-image dabs/√tr-Cov corr ≈ −0.25,
  ratio CV ≈ 0.5 and route-dependent (0.87 vs 1.50) — no universal
  constant.
- **(B) no saturation plateau** — response is near-linear in ε (log-log
  slope 0.84–0.90) through every route's natural amplitude; routes at
  nat differ ~1.7×, tracking their perturbation sizes, so amplitude is
  set by input size × a direction/σ gain, NOT a σ-only posterior budget.
  Demotion directions are ~1.5× *softer* than random at matched ε
  (0.13 vs 0.20 rel @ ε=0.05) — the opposite of a re-roll story.

**Side finding (load-bearing elsewhere):** the deterministic VAE
re-encode reproduces the cached latent **bit-exactly** for most images
(a reenc direction exists for only 4/12) — [E11](../e11/)'s reenc ≤ 0.02
control was carried by draw noise, not an encode-chain perturbation.

## Consequences

E11's route-uniform m(σ) is **not** saturation — it must come from
route-uniform delivered perturbation × a direction/σ gain (demotion
directions being ~1.5× softer than random at matched ε is a lead worth
keeping). The native-only pre-screening payoff is void.

**Guard: do not re-propose the lossy-code saturation account without
new evidence.** The Probe A posterior-trace instrument itself is sound
and reusable (`bench/run_posterior_budget.py`, FD linearity 0.96–1.0).
