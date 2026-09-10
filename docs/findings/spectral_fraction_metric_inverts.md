# Diagnosing representation edits: a fraction-of-Δ spectral metric inverts, and per-block ablation can't see a cumulative artifact

Building the DAVE "analytic dot-predictor" (`bench/dave/probe_patch_imprint.py`)
surfaced two reusable traps in how we diagnose representation-level /
sampler-boundary interventions (DAVE, Spectrum, CNS, SMC-CFG, mod-guidance —
anything that perturbs intermediate features and is judged by the spectrum of the
resulting change). Both are easy to walk into and both produced a confidently
wrong answer here. Keep them in mind before trusting a spectral probe's verdict.

The DAVE-specific conclusions (the `≤18` block cap, the flat pool) live in
`bench/dave/README.md` Phase 2d/3 — this doc is only the methodology that ports.

## The setup that fooled us

DAVE attenuates a Transformer block's spatial-mean (DC). Late blocks produce a
patch-grid dot artifact (the worst DAVE degradation). We tried to make the
"which blocks dot" cap measured instead of hand-set: attenuate one block at a
time, diff the final latent (`Δ_ℓ = lat_ℓ − lat_base`), and score each block by

    imprint(ℓ) = power(Δ_ℓ) on the patch-grid harmonics / total power(Δ_ℓ)

i.e. "what fraction of the change this block causes is patch-periodic." It passed
a synthetic unit test cleanly (smooth Δ → 0.00, pure checkerboard → 1.00). On the
real model it was **anti-correlated with the actual degradation.**

## Trap 1 — a *fraction-of-Δ* metric rewards the do-nothing intervention

`results/20260609-1613-imprint/` (28 blocks, 3 seeds, the production α=0.5/τ=0.15
dose). The decodes vs the metric, for three representative blocks:

| block | imprint (rank) | what the decode actually shows |
|---|---|---|
| 17 | 0.008 — highest ("dot-causer") | cleanest — near-baseline, hands fine, sign text legible |
| 9  | 0.004 — lowest ("safe") | heavy recomposition, hands + sign text garbled |
| 2  | 0.004 — lowest ("safe") | big restyle/reframe, sign → empty frame |

The ranking is upside down. The mechanism is the normalization, not the
numerator: `imprint` is `band_energy / total_Δ_energy`. A block that barely
changes the image (17) has a tiny denominator, so its small absolute patch noise
reads as a large fraction. A block that recomposes hard (9) has a huge low-freq
numerator-and-denominator, so the patch energy is diluted to near-zero. The
fraction is high precisely when the intervention does little, and low when it
does a lot — so it ranks interventions by inverse-magnitude, not by artifact.

> Rule: never normalize an artifact score by the magnitude of the same
> perturbation. Normalize by a fixed reference (baseline image energy, or the
> band energy of the baseline), or report absolute band energy alongside a
> total-Δ-energy column so the reader can see a near-no-op masquerading as "high
> fraction." Any "fraction of this change that is in band X" proxy for "does this
> change cause artifact X" has the same inversion — it applies verbatim to
> Spectrum/CNS spectral diagnostics, not just DAVE.

## Trap 2 — a per-block ablation cannot see a cumulative artifact

Every single block scored `imprint < 0.01` (the whole field was 0.004–0.008).
There was no dot to find one-block-at-a-time, because the dots are a
full-schedule, multi-block collective effect (Phase 2c: 9 late blocks attenuated
across the whole σ schedule). One τ-gated block in isolation never enters that
regime. The probe faithfully measured single-block behavior and concluded
"nothing dots" — true, and useless, because the failure is compositional.

> Rule: if the artifact you're hunting is produced by the interaction of many
> components or accumulates over the trajectory, a one-component / one-step ablation
> will under-read it to zero. Reproduce the artifact in its real regime first
> (here: the full mask on the full schedule), then ablate within that regime —
> don't ablate from baseline and assume superposition.

## Corollary — the degradation that matters here is structural, and a spectrum can't see it

With the flat-pool mask + `≤18` cap + τ-gate, dots are gone; what actually breaks
at α=0.5 is hands and rendered text — a semantic/structural failure with no
spectral signature (broken fingers occupy the same frequency band as good ones).
This is why the DAVE paper scores block effects with a VLM judge (Gemini, their
Fig 9 / App F.1), not an FFT. Lesson: a spectral probe is the right tool for
*texture/aliasing* artifacts and the wrong tool for *structure/identity* damage —
for the latter, reach for a perceptual or VLM judge from the start.

## What the probe was still good for

Its `--decode` contact sheet — one image per single-block intervention — is a
clean per-block diversity-vs-damage catalog (block 17 = near-no-op, block 9 =
recompose-and-break, block 2 = restyle). That shape of tool is exactly right;
it just needs a perceptual/VLM score on top, not a 2D-FFT scalar. If a future
"analytic block-wise" pass is wanted, build it as a VLM-scored decode sheet (the
paper's recipe), and keep any spectral number as an absolute, baseline-normalized
side-channel only.

## Pointers

- Probe + raw run: `bench/dave/probe_patch_imprint.py`, `results/20260609-1613-imprint/`.
- DAVE-specific conclusions (the cap is justified by eyeball + the paper, not by
  this probe): `bench/dave/README.md` Phase 2d / Phase 3.
- Related "we measured the wrong thing" findings: `turbo_fei_band_deficit_falsified.md`,
  `mod_guidance_quality_tag_axis.md` §3 (pointwise MSE leaves the text
  direction unconstrained — another normalization/projection blind spot).
