# image → crossattn_emb amortized encoder — proposal

Replace per-image embedding inversion with a single-shot image encoder that
produces `crossattn_emb` directly. The inversion bench (`bench/inversion/`,
`bench/inversionv2/`) is not wasted — it defines the target manifold's
structure and gives us prototype supervision for high-signal slots.

Location for future artifacts: `bench/img2emb/`.

## Motivation

### What's wrong with per-image inversion

- **Cost.** 100–500 DiT forwards per image (`scripts/inversion/invert_embedding.py`
  defaults to 100 steps × 4 grad_accum = 400 forwards). Unusable as a
  conditioning path at serving time.
- **Jitter / drift.** The optimizer roams in directions T5+LLM-adapter never
  spans. `bench/inversion/results/slot_subspace/` found per-slot rank at
  k@95 ≈ 300 / 1024 — ~70 % of the parameter budget is unconstrained noise.
- **Stochastic.** Different seeds give different solutions that require
  Hungarian token-alignment to aggregate. Amortizing over the dataset
  averages out that noise for free.

### Why an encoder is the right shape of answer

From `bench/inversionv2/results/tag_slot/summary.md`:

- Prefix classes (rating / 1girl / solo / artist) live in **k@95 ≤ 5** per
  slot. Effectively prototype-addressable.
- Suffix-tag **position invariance cos ≈ 0.99** across the 8 shuffle
  variants for most tags. Tag identity dominates slot position.
- Only ~55 % of typical slot energy lies in the top-64 shared pooled
  directions — so subspaces are position-specific, but not so specific that
  a learned query-per-slot decoder can't cover them.

That target manifold is small enough for a thin encoder to fit, and
deterministic enough that supervised regression on cached `crossattn_emb_v*`
is a well-posed problem. Once trained:

- O(1) per image (one encoder forward + one cross-attn layer) vs O(N_steps)
  inversion.
- Deterministic, on-manifold — no jitter, no alignment step.
- Generalizes to unseen images; inversion doesn't.
- Reusable for GRAFT candidate conditioning, embedding-space
  analysis, and anywhere we currently need a "what does the DiT see in this
  image" vector.

Inversion stays as a research/probe tool — not deprecated, not the workhorse.

## Target specification

Ground truth: `crossattn_emb_v{0..7}` stored in each image's
`*_anima_te.safetensors` under `post_image_dataset/`.

Shape: `(512, 1024)` float (bf16 on disk). These are post-T5-tokenizer,
post-LLM-adapter, pre-DiT embeddings — the exact tensor the DiT consumes.

Each image has 8 caption-shuffle variants; they share the same rating /
count / character / artist prefix and only re-order the suffix alphabetical
tags (see `library/anima/training.py:30` `anima_smart_shuffle_caption`).

**Keep all variants separate on disk.** Storage is bf16 (N × 8 × 512 ×
1024 × 2 ≈ 8 GB at N=1000, acceptable for a one-off bench). Consumers pick
their own policy:

- **Training** uses per-sample uniform variant sampling (each image in a
  batch draws one of its 8 variants independently). 8× effective data and
  the cos-0.99 spread acts as implicit augmentation of the queries — the
  model learns that several slightly-different embeddings are all correct.
- **Pool linear probe** averages internally. Closed-form OLS on identical
  X rows collapses to predicting the mean target anyway, so keeping
  variants separate adds no information for this probe.
- **Eval / baselines** use the variant-mean. It's what the encoder will
  approximately serve downstream, and it's a stable deterministic target
  for metric comparison.

Active length: the non-zero prefix of the 512-length sequence (p50 = 150,
p95 = 247 from the tag bench). Loss should zero-weight padded tail. The
shuffle doesn't change token count, so L is effectively the same across
variants per image (we store max-over-variants for safety).

## Architecture

BLIP-2 / IP-Adapter family. Minimal form:

```
image (H×W RGB)
  → encoder (frozen)                    → patch tokens [N_patches, D_enc]
  → Perceiver Resampler                 → 512 learned queries
    · K cross-attn layers (queries ← patches)
    · optional FFN per layer
  → output_proj (Linear D_q → 1024)     → crossattn_emb [512, 1024]
```

Encoder choices (phase 0 bench):

- **DINOv3 ViT** — `models/dino/` (DINOv3ViTModel, hidden 1280, 32 layers,
  224×224, patch 16 → 14×14 = 196 patches + CLS + 4 register = 201 tokens).
- **SigLIP2 large** — `google/siglip2-large-patch16-384` (hidden 1024, 24
  layers, 384×384, patch 16 → 24×24 = 576 patches). Pulled from HF hub on
  first use.

Both produce per-patch tokens. The bench decides which encoder goes into
the real training run; we don't commit to either yet.

Resampler config for phase 1 prototype:

- Queries: 512, initialized random-normal scaled to crossattn_emb's measured
  per-element std (≈ 0.15).
- Layers: 4 cross-attn blocks (queries ← patches), each with self-attn over
  queries, MLP, and layer-norm. Dim 1024. Head count 8.
- Output projection: identity if resampler already outputs 1024, else Linear.

Parameter budget at 4 layers / dim 1024: ~50 M trainable. Encoder frozen.

### Optional: prototype-aware prefix head

Split the 512 queries into **prefix queries** (first ~8) and **content
queries** (remaining 504):

- Prefix queries are decoded through a classifier over the known prototype
  library (`phase2_class_prototypes.safetensors` +
  `phase3_artist_prototypes.safetensors`). DINOv3-CLS → softmax over
  prototype IDs → weighted sum of prototype vectors. Enforces the ~5-dim
  structure the bench found for those slots.
- Content queries use the standard resampler path.

This is optional — the baseline path omits it and lets the resampler learn
prototypes implicitly. Adds it only if phase-1 held-out MSE on the prefix
slots stalls above the content slots.

## Training objective

Primary: **MSE on active slots**, cosine-loss supplement.

```
loss = w_mse · MSE(pred[:, :L_i], target[:, :L_i])
     + w_cos · (1 − cos(pred[:, :L_i], target[:, :L_i]))
     + w_zero · MSE(pred[:, L_i:], 0)
```

- `L_i` is per-image active length (first index where `target.abs().amax(-1)
  > 1e-6` drops to 0 — same heuristic as `invert_embedding.py`).
- Cosine term matters because the bench showed direction dominates norm for
  most slot/tag combos.
- Zero-pad term keeps the model from leaking content into slots the DiT was
  trained to treat as attention sinks. Small weight (0.01 · w_mse) so it
  doesn't dominate.

Optional secondary: **flow-matching loss through frozen DiT**. One DiT
forward per step on `(image_latent, pred_emb, random_sigma)`. Expensive; add
only if the MSE path overfits T5-specific noise that doesn't matter to the
DiT. Gated on phase-1 results.

Target construction:
- Load all 8 variants, zero-clamp each variant's padded tail, stack to
  ``(N, 8, 512, 1024)``. No averaging on the target itself.
- At training time, pick a random variant per image per step. At eval,
  compare predictions to the variant-mean (stable deterministic target).

## Phase 0 — encoder fit bench (DINOv3 vs SigLIP2)

**Goal.** Before building the resampler, confirm the encoder has enough
signal about the caption-level semantics that live in `crossattn_emb`.
If per-slot linear R² is too low, swap encoder before any training effort.

**Protocol.**

1. Subsample 1000 images from `post_image_dataset/` (cap cost; dataset has
   1987 usable images).
2. For each image:
   - Load the image file.
   - Encode via DINOv3 at 224×224 and SigLIP2 at 384×384, both with their
     own preprocessor configs.
   - Extract **two features** for comparison: pooled (CLS or mean-over-
     patches) + raw patch sequence.
   - Load `crossattn_emb_v0` from cache as target.
3. Train two linear probes per encoder:
   - **Global-pool probe.** Linear `D_enc → 512 × 1024 = 524k` from pooled
     feature to flat crossattn. Reduced: Linear `D_enc → 1024 · K` for
     K = {1, 8, 32, 64, 128, 512} slot budgets — tells us how much slot
     structure the encoder's pooled representation supports.
   - **Resampler probe.** 1-layer cross-attn resampler with 512 queries over
     patch features. Smallest plausible architecture; a negative result here
     is a real gating signal. Train 2–5k steps.
4. Held-out split: 80/20 by image. Metrics on held-out:
   - Per-slot R² (averaged over active slots only).
   - Per-slot cosine (mean, median, p10).
   - Per-slot MSE against the target variance (normalized).
   - Same metrics stratified by slot position: first 8 (prefix prototype
     slots) vs slot 8–64 (mid-caption content) vs slot 64+ (suffix tail).
5. Sanity checks:
   - Baseline: predict the **dataset-mean crossattn_emb**. Any encoder with
     R² not meaningfully above 0 is worse than this prior.
   - Baseline: predict from the **cached crossattn mean over the 8 variants
     of the same image** — upper bound of "what MSE could possibly be" if
     the encoder had perfect information (answers "how much of the
     embedding is actually caption-shared vs variant-noise").

**Decision rule.**
- If best encoder's held-out **per-slot cosine > 0.6 median** on prefix
  slots and > 0.4 on content slots → proceed to phase 1 with that encoder.
- If < 0.3 everywhere → try a different encoder (WD-Tagger backbone,
  EVA-CLIP, anime-ViT) before committing to phase 1.
- Middle band → proceed with full resampler (phase 1) and re-measure; the
  linear probe underestimates what a 4-layer resampler can do.

**Expected output files.**

```
bench/img2emb/results/phase0/
├── dinov3_pool_probe.json          # R² / cosine per slot budget
├── dinov3_resampler_probe.json
├── siglip2_pool_probe.json
├── siglip2_resampler_probe.json
├── baseline_mean.json              # dataset-mean target baseline
├── baseline_variant_mean.json      # per-image v0..v7 mean baseline
└── summary.md                      # head-to-head table + recommendation
```

**Rough cost.** Encoder forward + linear/resampler fit on 1000 images ×
2 encoders. ~30 min on a single GPU.

## Phase 1 — minimal resampler prototype

Only start after phase 0 picks an encoder.

**Goal.** Confirm the full resampler can fit the target well enough to use
downstream (e.g., as a conditioning path for DiT inference, or GRAFT
candidate generator).

**Scope.**
- Frozen encoder (phase-0 winner).
- 4-layer Perceiver Resampler, dim 1024, 8 heads.
- Train 10k steps on full `post_image_dataset/` with v0..v7 averaged target.
- Loss: MSE + cosine + zero-pad (weights above).
- Eval: per-slot R² / cosine on 10 % held-out images.
- Downstream test: plug predicted `crossattn_emb` into a frozen DiT, run
  inference, VAE-decode, compare side-by-side to (a) real caption inference,
  (b) inversion-result inference. No CLIP/DINO score gatekeeping — human
  eyeball on ~20 images is the first pass.

**Success.** Held-out per-slot cosine > 0.85 median on active slots AND
inference-from-predicted-emb is visually coherent (same subject, same rough
composition) on held-out images. Not pixel-identical — that's not the goal.

## Phase 2 — prototype-aware decomposition (conditional)

Only if phase-1 held-out R² on the **prefix slots** (first 8) is notably
worse than the content slots. The bench predicts the opposite (prefix is
easier because it's low-rank), but if the resampler can't exploit that,
add the explicit prototype head from the Architecture section.

Output artifact: `phase2_prototype_head.json` with prefix-slot metric
deltas and a checkpoint of the classifier.

## Risks / open questions

1. **Domain shift.** DINOv3 and SigLIP2 are both trained on natural images;
   `post_image_dataset/` is booru-derived anime/manga. Neither is
   in-distribution. This is the single biggest risk. Phase-0 gating protocol
   explicitly surfaces it — if both encoders score similarly low, the fix
   is a domain-specific encoder (WD-Tagger ViT, EVA-anime, Kohya CLIP), not
   a bigger resampler.
2. **Artist slot unpredictability.** The cached target includes `@artist`,
   which the image often under-determines (one-off collabs, unsigned
   pieces, borrowed styles). Two options: (a) zero-weight the artist slot
   in the loss, (b) accept that the encoder will predict the style manifold
   projection regardless of the literal @tag — i.e., embed "looks like
   @sincos" for sincos-style images even when the caption said something
   else. Option (b) is probably right; the embedding is functional, not
   labelled.
3. **Variant averaging would collapse positional nuance** — so we don't do
   it on the target. Training uses per-step variant sampling (each image ×
   random variant), eval compares to the variant-mean (stable metric).
   Original risk text resolved at the architecture level; if downstream
   inference later shows the model needs more aggressive positional
   diversity, revisit (e.g., sample from neighbour-image variants too).
4. **Padded-tail leakage.** Even with the zero-pad loss term, the model
   might learn to emit near-zero-but-nonzero values in padded slots. The
   DiT's cross-attn has no text-side bias so those act as attention sinks
   only if exactly zero. Phase-1 eval should include `mean(|pred|)` in the
   padded region as a diagnostic; threshold < 1e-3 before calling it done.
5. **Token-count mismatch across encoders.** DINOv3 at 224 = 201 patch
   tokens, SigLIP2 at 384 = 577 patch tokens. The resampler handles this
   natively (queries don't care about patch count) but it means higher
   compute for SigLIP2. Factor into the phase-0 wall-clock comparison.
6. **This is not a replacement for the inversion bench.** If a future
   analysis needs "what would the DiT ideally want to see at this latent",
   that's still a per-image optimization problem. The encoder approximates
   the inversion's *average behavior*, not its per-image optimum.

## Success criteria (end-to-end)

- Phase 0: one encoder passes the decision rule.
- Phase 1: median held-out slot cosine > 0.85, inference from predicted
  emb visually coherent on ≥ 15/20 held-out images.
- End-state: a script `scripts/encode_image_to_emb.py` that takes an image,
  returns a `crossattn_emb.safetensors` in the same format as
  `scripts/inversion/invert_embedding.py` does today, in < 1 s on a 4090 per image.

## Non-goals

- Not training the DiT end-to-end. The DiT stays fully frozen.
- Not replacing text conditioning in general. This is a parallel path, not
  a replacement — captioned training and inference stay as the primary
  route. The encoder fills the gap where we have images and no captions,
  or need inversion-like behavior at serving speed.
- Not building a captioner. The output is the post-T5 embedding, not
  readable text. If we want readable captions, that's a separate (and
  well-trodden) problem — image-to-text is a different training regime.

## Open calls needed before phase 0

- Confirm encoder short-list (DINOv3 + SigLIP2 only, or add a third?).
- Pick v0-only vs variant-averaged target for phase 0 (recommendation:
  variant-averaged; it's cheap and the bench justifies it).
- Confirm bench output location (`bench/img2emb/results/phase0/` proposed).
