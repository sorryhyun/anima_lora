# σ-conditional low-res gradient equivalence — Phase 0 (observability)

**Status: Phase 0 DONE (2026-07-24) — spectral mechanism REFUTED; verdict in
`project/sigma_lowres/record/report.md`.** σ-dependence is real and tier-ordered, but
the collapse sits at σ ≈ 0.5 (not the RAPSD-predicted ≈ 0.14), 512 is never
safe at any σ, and the practical residue (σ>0.5 → 896, ~14% wall-clock
ceiling) is likely below the Phase-1 bar. Bench: `project/sigma_lowres/bench/`.

## Motivation

SwD (Scale-wise Distillation, arXiv:2503.16397, §3) shows for SD3.5/Wan VAE
latents that the forward noising process progressively masks high spatial
frequencies: above some noise level, a downscaled latent carries the full
surviving signal. They use this for a progressive-resolution *inference*
pipeline; we ask the *training-side* question — is the LoRA gradient from a
demoted-tier latent equivalent to the native-tier gradient **conditional on
σ**, with the crossover predicted by the latent spectrum?

## Relation to closed lines (why this is not a re-proposal)

- **tier_routing Phase 3a (CLOSED)** asked whether *per-image content*
  (redundancy, tags) predicts demotion cost. Its estimator **marginalized σ
  out** (one accumulated gradient over the whole stratified σ grid, then one
  cosine). Per-σ-band gaps are explicitly listed under "Not retried / out of
  scope" in `project/sigma_lowres/bench/tier_routing/report.md`. The report's guard — "any new
  predictor needs a fresh probe set and a pre-registered hypothesis" — is
  satisfied: the predictor here is σ (a controlled variable, not per-image
  fishing) and the hypothesis is pre-registered by SwD's spectral analysis.
- **autoscale (CLOSED)** killed *blanket* σ-unconditional demotion at matched
  FLOPs. σ-conditional demotion was never tested.
- Verdicts read off **per-σ-bin means across images** — the estimator class
  that was reliable in 3a (SEM ~0.015–0.02), not the per-image ranking that
  failed (split-half ρ ≈ 0). Split-half reliability check on the bin-mean
  curves is mandatory (`project_tier_routing_phase3a_failed`).

Supporting priors: 3a noted estimator variance "dominated by a few
high-magnitude σ-draws" (consistent with a σ-concentrated gap smeared flat by
marginalization), and `docs/findings/sigma_signal_where_anima_resolves.md`
places Anima's high-frequency content in the σ < 0.45 tail.

## Hypotheses (pre-registered)

- **H1 (spectral)**: Anima/Qwen-VAE latents under flow-matching noising
  `(1−σ)x₀ + σε` have a σ-dependent frequency crossover f\*(σ) above which
  signal power < noise power; f\*(σ) decreases with σ. Closed form per radial
  bin: σ_eq(f) = √P(f) / (1 + √P(f)) where P is the clean-latent RAPSD
  (unit-variance noise).
- **H2 (gradient)**: the demotion gap gap_e(σ) = floor(σ) − cos(native,
  demote_e | σ) is **not flat in σ**: it collapses toward 0 for σ above the
  tier's predicted σ\*(e) (the σ_eq at the demoted grid's Nyquist) and
  concentrates below it.
- **H3 (correlation)**: the observed gap-collapse σ tracks the RAPSD-predicted
  σ\*(e) per tier (dose-response: σ\*(896) < σ\*(768) ordering as predicted by
  the spectrum).

## Design

**Measurement A — `project/sigma_lowres/bench/rapsd.py`** (no DiT, minutes).
RAPSD of the probe set's cached latents in the DiT's spatial grid, normalized
frequency r ∈ (0, 0.5] cycles/latent-pixel. Noise PSD is analytically 1
(unit-variance white). Outputs: P(f) mean curve, σ_eq(f), predicted σ\*(e)
for e ∈ {896, 768} (demoted Nyquist = 0.5·e/1024), above-Nyquist SNR A(σ, e),
Fig-1-style plot. Same probe-set selection as Measurement B.

**Result (`results/20260724-1202-phase0/`)**: Anima latents are quiet at high
frequency (P(f) < 1 above f ≈ 0.16; latent var 0.42). Predicted
**σ\*(896) = 0.136, σ\*(768) = 0.146**, and by the same curve
**σ\*(512) ≈ 0.20** (P(0.25) = 0.065) — per-image spread tight (p10–p90 ≈
0.11–0.16), so the crossover is image-generic: selection headroom is on the σ
axis, not the image axis. Measurement B therefore runs demote arms
**896, 768, 512**; pre-registered prediction is gap collapse to floor above
bin 1 (σ ≳ 0.15–0.25) with tier-ordered onset (H3), 512 included.

**Measurement B — `project/sigma_lowres/bench/run_sigma_probe.py`** (~2 h GPU).
The 3a instrument (redraw-floor null, re-encode confound control, demote
arms with pixel-space downscale → VAE re-encode → noise — SwD's validated
"strategy B" ordering) extended with **per-σ-bin gradient accumulators**:
B uniform σ bins × D stratified draws per bin per arm. Per image × bin:
cos_floor, cos_reenc, cos_e, gap_e, grad norms. Bins uniform in σ (mechanism
axis; per-bin means make the training marginal density irrelevant).

## Verdict criteria

- **Instrument valid**: gap_reenc(σ) ≈ 0 in every bin; split-half (over
  images) bin-mean curve correlation high.
- **H2 pass**: monotone decreasing gap_e(σ) with high-σ bins statistically
  indistinguishable from floor AND low-σ bins carrying ≥2× the pooled mean.
  Flat gap(σ) ⇒ mechanism refuted at gradient level → whole low-res family
  closes with the physics ruled out.
- **H3 pass**: observed collapse-σ within ~1 bin of RAPSD σ\*(e), correct
  tier ordering.

## Payoff if validated (NOT Phase 0 scope)

Training-side σ-conditional resolution routing: sample σ first, use the
low-tier cache when σ > σ\* (4200 → ~2100 tokens on that slice). No inference
pipeline change. Scale-wise inference distillation (SwD proper) is a
separate, later decision.

## Cost

Measurement A: CPU-scale. Measurement B: ~2 h on 16 GB (block-compile
dynamic-seq path, same as 3a). No training runs spent.

---

# Phase 1 — one-tier-down-at-high-σ routing (post-Phase-0 reframe)

Phase 0 refuted the *spectral* governor but measured a real, reliable rule:
**one tier down (~0.87× edge) is gradient-safe at σ > 0.5** (gap within the
reenc-control band), while two tiers down is never safe. Reframed per-tier —
every image demotes to *its own* next tier on high-σ draws — the rule is
data-level and dataset-general: ~50% of draws (sigmoid shift-1 mass above 0.5)
at ~0.65–0.77× per-draw cost → **~14% wall-clock at fixed steps**, more on
high-tier-heavy datasets (1280→1024 is 0.65×/draw). Loss budget: residual gap
0.03–0.05 ≈ reenc band — small but nonzero; the fixed-steps CMMD gate is
mandatory, and the pitch is wall-clock only (never "more steps" — autoscale
lesson).

## Phase 1a — ratio-transfer probes (observability, cheap)

Is "one tier down at σ>0.5" ratio-general or 1024-specific?

- **896→768** (ratio 0.857; 107 native-896 records available): same probe,
  `--tier 896 --demote_edges 768,512`. Pre-registered: gap_768 collapses to
  the reenc band for σ ≥ 0.5 (rule transfers); gap_512 (ratio 0.57, the
  two-tiers-down control) stays elevated at all σ. If 896→512 instead looks
  safe, ratio is not the invariant and the rule needs re-derivation.
  **RESULT (2026-07-24): FAIL** — high-σ residual 0.06–0.12, ~2× the 1024→896
  plateau, outside the reenc band; control as predicted. "One tier down" is
  not the invariant; safe set = {1024→896 @ σ>0.5} only. The ratio-vs-capacity
  ambiguity makes 1280→1024 the discriminating probe (see report).
- **1280→1024** (ratio 0.8, 0.65× tokens): no cached 1280-tier images —
  requires a small `--target_res` re-preprocess of high-res sources first,
  plus a VRAM check at 6300 tokens (may need `--grad_ckpt`). Optional arm;
  the biggest per-draw payoff if it passes.

## Phase 1b — trainer wiring + the gate

σ drawn at batch-assembly time; σ > σ\*(=0.5) → fetch the one-tier-down cache
(stem-suffixed sibling caches, autoscale-emit pattern as design reference —
runtime was stripped 2026-06-28, do not resurrect blindly). Gate: fixed-steps
A/B on ≥1 artist set — CMMD non-inferior + rendered spot-check + realized
wall-clock logged. Ship only as opt-in.

## Phase 1c — bespoke compute-heavy loops (separate probes REQUIRED)

- **EasyControl**: structurally clean (frozen DiT, cached paired latents,
  same draw structure) but the gradient lives in the cond-LoRA stream — the
  Phase-0 equivalence does not automatically transfer. Needs the probe re-run
  at an EC operating point (harness must drive the cond stream) before any
  wiring.
- **turbo**: NOT clean. Rollout latents are generated (no pixel-space demote
  path; latent-space downsampling is untested here and SwD found it inferior),
  and resolution changes inside the rollout change what the student *is*
  (that's SwD's scale-wise pipeline, a different product). Only the fake/critic
  σ-draw forwards are even candidates. Treat as its own research question;
  do not promise savings here.

## Honest bounds

- ~14% ceiling on the current corpus (896+1024 tiers, fixed steps); scales up
  only with high-tier content. If the Phase-1b A/B shows any CMMD regression,
  the line closes — the residual gap is the suspect, not the harness.
- Phase-0 caveat carries: equivalence measured at a native-res-trained
  operating point; mixed-res training could shift it (either direction).
