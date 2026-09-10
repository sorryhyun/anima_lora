# Spectral Guidance — no low-rank guidable subspace on Anima (Phase 0 NO-GO)

<!-- check-docs: ignore-flags (historical record of a NO-GO probe; the flags below never shipped) -->

This records why Spectral Guidance (Moreira et al., Spectral Guidance for
Flexible and Efficient Control of Diffusion Models, arXiv:2605.28900) was not
promoted past its Phase-0 falsification gate on Anima. The short version: the
paper's load-bearing premise — that the posterior-mean operator `T_t` collapses,
midway through the reverse process, onto a few surviving directions `φ_{t,k}`
that form a low-dimensional guidable/preserving subspace — does not hold on
Anima's latents. The round-trip operator stays near-full-rank through the
entire editing band and only collapses in the far σ>0.85 tail, where guidance is
already hopeless. There is no low-dim `span{φ_{t,k}}` to anchor at the σ where
DirectEdit actually edits, so the proposed spectral-anchoring arm has no lever.

The single most reusable lesson: **"the signal resolves by σ≈0.45" (accuracy) and
"the guidable subspace becomes low-dimensional" (operator rank) are different
σ-axes, and on Anima they do not coincide.** The proposal assumed they would; they
don't. See the reconciliation with [[project_sigma_signal_resolves_by_045]] below.

Proposal: `_archive/proposals/spectral_guidance_anima.md` (header flipped to SHELVED).
The Phase-0 probe and its runs (`-2001-cond`, `-2010-quarter`, `-2034-uncond`) are no
longer in-tree.

## The property under test

Two separable claims (paper §3, Fig 6–7):

- (P1) As noise grows, only a few features survive — the leading left singular
  functions `φ_{t,k}` of the conditional-expectation operator `T_t` (clean→noisy
  posterior mean), i.e. the principal modes of the round-trip covariance
  `T_t T_t*`. Guidance can only act along `span{φ_{t,k}}`.
- (P2) That collapse is a transition over the schedule, and guidance is
  effective only inside it (early = redundant, operator≈identity; late = hopeless,
  all info erased).

The proposal additionally assumed the P2 window lines up with σ≈0.45 (where
Anima's `x0_pred` resolves, [[project_sigma_signal_resolves_by_045]]), giving a
principled DirectEdit schedule. Phase 0.1 tests P1+P2 directly; this assumption is
what fails hardest.

## What the probe does

`probe_spectrum.py` estimates `T_t T_t*` from the frozen DiT, no `f_φ` trained.
Faithful to P1's definition, it measures the round-trip covariance over noise:
for one clean latent, draw `M` noise realizations, run the model to its posterior
mean `x̂₀ = x_t − σ·v` for each, and take the covariance of `{x̂₀}` across the `M`
draws (a tiny `M×M` Gram → eigenvalues). Effective rank / top-K energy of that
spectrum is exactly "along how many directions does `E[x₀|x_t]` vary as the noise
varies" — the object P1 names. The curve is averaged over 8 artists (one clean
latent each; one-per-artist controls content), bsz=1 throughout, `--compile` on.

Why per-noise and not the proposal's literal across-batch covariance: with one
latent per artist the samples are different resolutions (no shared feature dim),
and an across-content rank caps at #artists−1. The round-trip covariance caps at
`M` instead, so a small artist set resolves the spectrum. (`M=32` → rank ceiling
31.) FM algebra makes this clean: with a perfect velocity `x̂₀ = x₀` exactly, so
the across-noise covariance is `σ²·Cov(v-prediction-error)` — a real property of
the denoiser, not leaked input noise.

Metric to read: top-4 energy fraction (top-4 eigenvalues / total). PR is
ceiling-saturated in the low/mid band and blunt for "do a few modes dominate";
top-4 energy is M-robust at the collapsed end and directly answers P1.

## Result — NO-GO, triangulated

| run | cond | n | σ=0.45 PR | top-4 @ 0.45 | top-4 @ 0.91 | tk4=0.5 crossing |
|---|---|---|---|---|---|---|
| `2001-cond`    | cond   | 8 | 30.7 | (PR-only) | — | — |
| `2010-quarter` | cond   | 2 | 30.7 | 0.151 | 0.318 | never (nan) |
| `2034-uncond`  | uncond | 8 | 30.7 | 0.149 | 0.277 | never (nan) |

- At the editing σ≈0.45 the top-4 modes carry only ~15% of the round-trip
  variance — essentially the isotropic floor (4/31 ≈ 0.13). The variation is
  spread near-uniformly across all ~31 measurable directions. No low-dim
  subspace. (P1 fails.)
- top-4 energy never crosses 0.5 at any σ. The only concentration is a mild
  rise in the σ>0.85 tail (still only 0.28–0.32 at σ=0.91) — P2's "late, hopeless,
  all-information-erased" regime, not a mid-schedule window. There is no
  transition near 0.45. (P2-window assumption fails.)
- Conditional vs unconditional agree, and uncond is even higher-rank in the
  tail (PR 23.5 vs 20.5 at σ=0.91) — conditioning tightens the posterior, so the
  conditional operator (the one DirectEdit actually uses) is the more favorable
  direction for finding collapse, and it still fails. Uncond can confirm, not
  rescue.

The feared killer ([[project_pe_feature_diagnostics]], PR≈6.2 collapsed manifold)
did not fire: the operator is high-rank, not degenerate. The proposal dies
for the opposite reason — too high-rank, with the transition too late.

## Reconciliation: this does NOT falsify σ≈0.75/0.45

[[project_sigma_signal_resolves_by_045]] measures reconstruction accuracy —
how close `x̂₀` lands to the true `x₀` (a bias curve; normalized lat-MSE crosses
20% at σ≈0.55). This probe measures the rank of how `x̂₀` varies under noise (a
variance-structure curve). Different axes; one cannot falsify the other, and they
are consistent at every σ: at σ=0.45 `x̂₀` is *accurate* (9% error) and its
small residual is spread *high-rank* across ~30 directions. Three distinct
staircases now coexist — (1) the model *knows* the answer by σ≈0.45, (2) the
latent gains structure only in the σ<0.45 tail, (3) the answer stops depending
high-dimensionally on the noise only at σ>0.85 (this probe).

## Why the DirectEdit arm specifically dies

Arm A's premise was: decompose the latent into the preserved low-dim `span{φ_{t,k}}`
(anchor that to source) and its complement (leave free for the edit), replacing the
blunt full-Δz anchor / scalar `t_inj`. If `span{φ}` is ~full-rank at the edit σ,
"anchor `span{φ}`" ≈ "anchor everything" = the full-Δz anchor we were trying to
beat. No low-dim bottleneck to exploit where editing happens → no lever.

## What survives

- Arm C (spectrum-as-diagnostic) — the `T_t T_t*` rank/top-K curve is a cheap,
  model-intrinsic, denoiser-only signal that fell out of this probe for free. Keep
  `probe_spectrum.py` as a diagnostic for "does this adapter shift the operator's
  rank / transition." Not pursued further here, but not killed.
- Editing-direction literature (Chen et al. 2024; Park et al. 2023) — if the
  goal shifts from preservation to sharper edit directions, those are the
  on-target references, not this (preservation-oriented) paper.

## Reproduce

```bash
uv run python -m bench.spectral_guidance.probe_spectrum \
  --n_artists 8 --noise_draws 32 --n_bins 24 --compile --label cond
# paper-faithful unconditional operator:
uv run python -m bench.spectral_guidance.probe_spectrum \
  --n_artists 8 --noise_draws 32 --n_bins 24 --compile --uncond --label uncond
```

Date: 2026-05-30.
