# sigma_lowres — open questions

## Q1 — Ratio or absolute capacity: what governs demotion safety? **[ANSWERED 2026-07-26]**

1024→896 (ratio 0.875) passes; 896→768 (ratio 0.857) fails. Two candidate
governors that the existing data cannot separate:

- **Ratio**: safety is a function of edge ratio → 1280→1024 (0.80) should FAIL.
- **Absolute target capacity**: safety needs enough tokens at the demoted grid
  (~4k?) → 1280→1024 (4116 tok target vs the passing 3012) should PASS.

**Answer** (report.md "1280→1024 probe", `results/20260726-2017/`): **ratio
is refuted** — 1280→1024 floors into the reenc band at σ ≥ 0.875 (per-image
and pooled) despite being more aggressive by ratio than the never-flooring
896→768; capacity predicts the ordering correctly. But the capacity
prediction's *threshold* also missed: the 0.625 bin stays elevated (0.096),
so the safe gate is **route-dependent** — σ\*(1024→896) ≈ 0.5,
σ\*(1280→1024) ∈ (0.625, 0.875), σ\*(896→768) > 0.95 or absent. The map is
a boundary σ\*(route), not a binary pass/fail; a `--sigma_window` refinement
localizes the 1280→1024 crossover. Run cheaply via the probe-local 1280
cache (`prep_1280_probe.py` + `--data_root`) — no corpus re-preprocess.

## Q2 — Where in the network does the gap live? **[ANSWERED 2026-07-25]**

Phase 0 established the gap is a network-function property (persists when the
latent is ~pure noise), but not *which* mechanism: attention structure, RoPE
geometry, seq-length-dependent normalization? A per-block / per-param-group
gap decomposition (same probe, gradient split by module) would localize it —
and might reveal a subset of parameters for which demotion IS safe.

**Answer** (`groundings.md` G4 + report.md Phase Q2; runs `*-endpoint-pg` /
`*-xzero-pg`): the Floor localizes in **depth, not module type** — early
blocks (~0–9, peak 3–8) carry ~3× the late-block gap, uniformly across every
param type within a block; RoPE is refuted as a concentrated mechanism
(self-attn up_q/up_k show zero excess over up_v). The content share is a
late-block minority effect; early-block sensitivity is pure graph. The "safe
subset" is a **depth band**: late-half-only updates at 768 read gap
0.03–0.09 vs 0.12 full — a lever, not yet a pass. Remaining mechanism
question (why blocks 3–8 specifically) belongs to the paper phase (Q6), not
to safety mapping.

**Revised 2026-07-27 (G10, origin-side)**: the RoPE refutation above was a
*landing-side* inference and is overturned for the mild route — a
PI-aligned-RoPE arm (`--pi_align`, DyPE-motivated) erases the 768 Floor
entirely (+0.080 → −0.001) and ~30% of 512's (+0.320 → +0.224).
`Floor_e = RoPE_e + Resid_e`: the q/k-vs-v landing uniformity was
propagation of a PE-originated perturbation. Resid_e keeps the depth
localization and the capacity governor. **G11 qualifier (same day)**:
rope alignment is NOT a practical lever — the stretch is off-manifold
with content in the input (768pi *worse* than 768 through σ 0.56–0.81),
so RoPE_e is removable only in the noise-dominated regime. Mechanism
finding; no safe-subset lever came out of it.

## Q3 — Does mixed-res training equalize its own gradients?

All measurements are at a native-res-trained operating point
(`anima_soup_sincos`). An adapter trained on mixed-res batches might close
(or widen) the gap. Any reopen of the broader low-res family should probe a
mixed-res-trained checkpoint **first** — it bounds whether the Phase-0 map is
a property of the base model or of the adapter's training distribution.

## Q4 — Is the 896 high-σ residual (~0.03) actually harmless at training scale?

"Within the reenc band at N=40" is a gradient-level statement; a full run
integrates thousands of demoted steps. Only the Phase-1b fixed-steps CMMD
non-inferiority A/B answers this. If CMMD regresses, the residual gap is the
suspect — the line closes (proposal's pre-commitment).

**Strengthened 2026-07-25** (report.md pooled addendum, `20260725-2155-pool4`):
in the batch-aggregate gradient (per-image gradients summed before cosines —
the object SGD follows), gap_896 is ≈ 0 at every bin σ ≥ 0.625 — the
per-image residual averages out across images rather than accumulating. Also:
stratifying by redundancy shows a null trend (Spearman ≈ −0.07), so there is
no per-image demotion-targeting lever. Still gradient-level; the CMMD A/B
remains the closer.

## Q5 — Do the bespoke loops inherit anything?

- **EasyControl**: structurally clean (frozen DiT, cached paired latents) but
  the gradient lives in the cond-LoRA stream — equivalence does not transfer
  automatically. Needs the probe re-run at an EC operating point with the
  cond stream driven.
- **turbo**: NOT clean — rollout latents are generated (no pixel-space demote
  path), and changing rollout resolution changes what the student *is* (that
  is SwD's scale-wise pipeline, a different product). Only fake/critic σ-draw
  forwards are even candidates. Own research question; no savings promised.

## Q6 — Paper question

Is "spectral sufficiency ≠ gradient equivalence" + the (route, σ) safety map
enough for a workshop paper? Needs Q1 answered (the map needs ≥3 routes) and
ideally Q2 (a mechanism sketch, not just a refutation).
