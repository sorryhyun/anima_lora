# Adaptive detailer — cfgdelta-localized second-pass region refinement

Status: PROPOSAL, no GPU work done. Reuse inventory verified by code-reading
(cited inline). The localizer is already shipped and validated; the new work is
the second pass and its composite.

## TL;DR

An ADetailer-style detail pass, with the YOLO detector replaced by our own
cfgdelta subject localizer — the one validated, deliberately-kept artifact of
the archived foveated line. Pipeline:

1. Base pass (normal generation) accumulates the per-cell localizer score
   for free during the high-σ steps.
2. At the end, threshold + morphology turns the score into compact subject
   blobs; each blob's bounding box is cropped, re-encoded at a higher
   free-fit tier, re-noised with fresh white noise to σ_start, and re-denoised
   with the same prompt.
3. The refined crop is downscaled and feather-composited back in pixel
   space using the (dilated) blob mask.

Unlike whole-image SR (ResShift ×2), the crop is re-denoised at higher token
density, so the model can synthesize semantic detail (eyes, hands, small
faces) it never had the tokens to express in the base pass — not just sharpen
what's there. Unlike the archived foveated line, nothing is degraded anywhere:
the base image is the floor, the detail pass only adds.

## Why this respects the archived findings (not a re-open)

The foveated line died because periphery blur was constitutive and P4t proved
detail cannot be injected late in the same trajectory (blur is a fixed
point of the flow at low σ). This proposal is the structurally different move
those findings point at:

- Separate second pass, not a tail treatment. The crop is re-noised to
  mid-σ and denoised across the whole remaining band — detail comes from HF
  that is being denoised all along, which is exactly the P4t-compliant route.
- Fresh white noise is on-manifold (established in the same line; the
  falsified thing was group-constant/structured ε).
- Identity locks above σ≈0.75 (deferred σ-gate finding) — so a σ_start
  below the authority window preserves the crop's composition while letting
  HF re-form. Sweep σ_start ∈ {0.4, 0.5, 0.6}.
- Nothing here merges tokens, reallocates refresh budget, or touches the
  Spectrum forecast — none of the closed sub-lines (P3 compose, 3b partial
  recompute, fovea SEA triggers, P4t un-merge) are re-proposed.

## Reuse inventory (all live in-tree)

| Piece | Where | State |
|---|---|---|
| Score sources `combo`/`cfgdelta`/`x0var` + accumulation | `networks/foveated.py` (inside `foveated_denoise`) | shipped; needs extraction into a standalone accumulator usable from the normal denoise loop |
| Score → compact blobs (bisected threshold, open/close/dilate-1, target fraction) | `networks/foveated.py::score_to_cells` | shipped, validated P2b |
| cfg=1 fallback (`combo`/`cfgdelta` → warn + fallback) | `networks/foveated.py:322` | shipped — covers turbo (cfg 1.0): fall back to `x0var`, or spend a few extra uncond forwards at high σ |
| Hard-prompt bench set w/ hand-annotated face boxes | `_archive/bench/foveated/hard_prompts.json` | reusable as-is |
| Free-fit tiers for the crop render | `library/.../buckets.py` | crop box snaps to the nearest tier band; no new resize mode |

Main new code: (a) a small `SubjectLocalizer` hook for the standard generation
loop (accumulate per-cell |v_c − v_u| over pre-σ_c steps — the spatial version
of what `CfgDeltaProbe` logs as norms); (b) the crop → re-encode → re-noise →
denoise-tail → composite driver; (c) the feathered pixel-space paste.

## Design decisions (v0)

- Composite in pixel space (decode crop, downscale, feathered alpha from
  the dilated mask). Latent-space pasting risks VAE seams; pixel-space is the
  ADetailer-proven route. The existing dilate-1 margin in `score_to_cells` is
  the feather seed.
- Per-blob boxes, not per-cell masks, drive the crop. Blob → bbox → pad to
  the nearest free-fit tier aspect/band. The cell mask is only used again at
  composite time.
- Same full prompt for the detail pass (v0). Region-specific captions via
  the tagger are a v2 lever, not needed to prove the line.
- Cost model: one detail pass ≈ (tail fraction of steps) × (crop tier
  tokens). E.g. 1 blob at the 1024 tier with σ_start 0.5 ≈ +0.5× a base-pass
  cost. Cap blobs at 2 by score mass.

## Phases

Phase 0 — kill-shot (bench-local, oracle masks, no new production wiring).
Use the annotated boxes from `hard_prompts.json` directly (skip the localizer).
Crop → re-encode at 1024 tier → fresh-noise to σ_start → denoise tail →
composite. Eyeball + face-crop RMSE/LPIPS vs: (a) base render, (b) base +
ResShift ×2, (c) base with the detail-pass NFE spent on extra base steps.
**Kill if**: no visible detail win over (b) and (c), OR identity drift (the
re-denoised face is a different face) at every σ_start, OR seams survive
feathering. Per the line's own lesson: calibrate on hard prompts from step one;
RMSE certifies change, eyeballs certify improvement.

Phase 1 — localizer replaces the oracle. Extract the accumulator, run
end-to-end auto: base pass → blobs → boxes → detail pass. Gate: blob boxes
cover ≥ the P2b face-cover numbers (95% on channel6-class prompts) and the
Phase-0 quality result survives the swap. Also settle the turbo path here
(x0var-only vs a few uncond probes).

Phase 2 — ship. `--detail_pass` (+ `--detail_sigma_start`,
`--detail_frac`, `--detail_max_blobs`) in `inference.py`, composing with the
standard `test-*`/`gen` surface like other inference plug-ins. Per
CONTRIBUTING Tier 2: bench script under `bench/detailer/` sharing
`bench/_common.py` + an invariant test (composite outside the mask is
bit-identical to the base render).

## Risks / open questions

- Identity drift is the main risk knob: σ_start too high changes the face,
  too low adds nothing. The sweep is cheap (Phase 0, one afternoon).
- Crop context starvation: a tight face crop loses body/scene context and
  may re-denoise inconsistently (lighting, angle). Mitigation: pad the bbox
  generously (context ring rendered but not composited) — measured in Phase 0.
- Localizer targets "where the prompt acts", not "what needs detail". A
  large, already-crisp subject can win the mask over a small degraded face.
  If Phase 1 shows this, add a cheap size prior (down-weight blobs whose bbox
  already exceeds ~40% of the frame) before reaching for anything learned.
- cfgdelta needs CFG > 1 — turbo falls back to x0var, which has a confirmed
  texture trap (morphology mostly contains it). Acceptable for v0; measure.
