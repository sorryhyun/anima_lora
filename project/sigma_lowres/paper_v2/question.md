# Deep-research questions — prior work on B/C gradient decomposition, cross-adapter replication, and conditioning-freeze probes

Purpose: find prior/related work for (a) the related-work sections of a
paper in revision, and (b) the mechanism follow-up paper. For each
question below, bring back the closest 5–10 papers, the terminology
that community uses, whether the *exact* object exists anywhere, and
anything we would be scooped by or are obligated to cite.

## Background (self-contained — read this first)

We study training a diffusion transformer (a rectified-flow /
flow-matching DiT for images) in which some training steps run on a
**downscaled latent at high noise** ("low-resolution substitution":
e.g. the 1024-tier latent is replaced by an 896- or 768-tier sibling
of the same image when the sampled noise level σ is high). The claim
family is about the **training gradient**, not samples.

For a frozen operating point (a LoRA adapter on a frozen base model)
we measure how the pooled training gradient g = Jᵀr changes when the
input grid is downscaled, and decompose that change by the product
rule into two exactly-defined branches in the shared
adapter-parameter space:

- **B — "data branch"**: the change contributed by the residual /
  regression target r being evaluated on the downscaled input, with
  the backward operator (Jacobian) held at native.
- **C — "graph branch"**: the change contributed by the network
  Jacobian Jᵀ (the compute graph itself — attention over fewer
  tokens, rotary coordinates, normalization statistics…) being
  evaluated on the downscaled input, with the residual held at
  native.

Measured facts we want situated in the literature:

1. **Near-cancellation.** At every measured (downscale route, σ), B
   and C are each far larger than the realized gradient change and
   strongly anti-aligned (correlation ρ ≈ −0.63…−0.96, typically
   ≈ −0.9). The realized deviation is the small residual of a
   near-cancellation between two large opposing terms.
2. **Where it is born.** The anti-alignment arises in the backward
   pass (Jᵀ), is uniform across depth and parameter type, and its
   magnitude scales with branch energy. It is partially mediated by
   the rotary-phase pathway at noise-dominated σ.
3. **Axis field.** The residual R = B + C has (approximately) one
   direction per σ — shared across downscale routes and across runs
   within one numerical environment — rotating smoothly as σ moves.
   The rotation is "matched-angle, not planar": no fixed 2-plane or
   2-parameter rate law fits it; spherical interpolation between
   measured directions is the best available description.
4. **Conditioning-freeze probe.** Pinning the network's
   noise-conditioning input (the adaLN timestep embedding held at
   σ_cond = 0.7 while the actual noising σ sweeps) neither freezes
   the rotation nor leaves it unchanged: the field reorganizes into
   two internally-coherent σ-blocks split at the conditioning value,
   while the cancellation survives at every bin (shallower). So the
   σ-rotation is carried neither purely by the conditioning pathway
   nor purely by the input noise statistics — the two interact.
5. **Cross-adapter replication.** The cancellation (large opposing
   branches, deep negative ρ, small residual) replicates across
   different LoRA adapters of the same base model trained on
   disjoint data, with cancellation depth ordering by perturbation
   size. Raw cross-adapter direction comparisons are frame-confounded
   (gradients w.r.t. different adapters' parameters live in
   non-overlapping frames), so only frame-free statements are made.

## Questions

### Q1 — The decomposition itself
Has anyone decomposed the *change in a training gradient under an
input perturbation* into a data-term (residual/target) contribution
vs a Jacobian/compute-graph contribution — the product-rule split of
δ(Jᵀr)? Any name for this object? Look in: backward error analysis
of deep nets, influence functions, NTK perturbation theory, gradient
sensitivity analyses, "gradient decomposition" literature. We believe
the exact object is unnamed; confirm or refute.

### Q2 — Near-cancellation of large opposing gradient components
Is there prior work reporting that a measured gradient (or gradient
difference) is the small residual of two large, strongly
anti-aligned components — in any setting? Nearest families we know
and want mapped precisely:
- multi-task gradient conflict / gradient surgery (PCGrad, CAGrad,
  etc.) — conflict *between tasks*, not within one gradient's
  decomposition;
- per-timestep negative transfer in diffusion training (e.g.
  "Addressing Negative Transfer in Diffusion Models", Min-SNR
  weighting) — conflict between timesteps;
- cancellation/interference effects in SGD noise or in Hessian–
  gradient alignment.
Anything closer than these?

### Q3 — σ/timestep-indexed gradient *direction* structure
Diffusion training is often described as multi-task across noise
levels. Who has measured per-noise-level gradient **directions**
(not just magnitudes or losses) and their geometry — e.g. smooth
rotation with σ, clustering of timesteps by gradient similarity,
per-timestep task vectors? Related: timestep-interval training,
noise-level curricula, per-σ loss weighting justified by gradient
geometry.

### Q4 — What the noise-conditioning input causally carries
Interventions that *mismatch* the conditioning timestep/σ embedding
against the actual noise level of the input (our "frozen
conditioning" probe): does prior work exist on robustness to
timestep misspecification, on what adaLN/FiLM conditioning causally
controls, or on freezing/ablating the conditioning pathway to
localize behavior? Any analysis of adaLN as defining a σ-indexed
"frame" for features or gradients?

### Q5 — Low-/mixed-resolution diffusion training at the gradient level
The efficiency literature justifies low-res or coarse-to-fine
training spectrally (at high noise, a downscaled latent keeps almost
all surviving signal): simple diffusion, cascaded models,
progressive/patch-wise training, relay-style pipelines, spectral
"coarse-to-fine" analyses of diffusion. Has *any* of it validated or
analyzed the substitution at the level of the **training gradient**
(rather than signal content, sample quality, or FID)? We claim the
gradient-level test is new — confirm or find the exception.

### Q6 — Cross-adapter geometry and the frame problem
For multiple LoRA adapters of one base model: prior work on shared
low-dimensional structure across adapters (task-vector arithmetic,
subspace overlap, "LoRAs live in a common subspace" claims), and on
the **gauge/frame problem** when comparing directions defined in
different parameter subspaces — including function-space or
induced-ΔW comparisons as the frame-free alternative.

### Q7 — Token-count-dependent compute-graph effects
Analyses of how a vision transformer's *backward pass / Jacobian*
changes with input resolution or token count: resolution
generalization of ViTs/DiTs, rotary-embedding interpolation for
vision (PI/NTK/YaRN-style, applied to images), attention
normalization or entropy shifts with sequence length, and any claim
that part of a resolution-transfer gap is attributable to the
compute graph rather than the data content (we call this a
σ-independent "floor" set by absolute token count).

## Notes for the searcher

- Terminology mapping: we say σ for the flow-matching noising level
  (σ ∈ [0,1], rectified-flow convention z = (1−σ)x + σε); "route"
  for a downscale pair like 1024→768; "demotion" for the low-res
  substitution.
- Negative results welcome: "no prior work measures X" is a useful
  answer if the survey was genuinely broad — say what was searched.
- Distinguish carefully between sample-time (inference) findings and
  train-time gradient findings; we only care about the latter except
  as framing.
