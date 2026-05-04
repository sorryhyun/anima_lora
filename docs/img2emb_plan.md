# img2emb plan — revival under EOSTok lessons

Source paper: arXiv 2605.00503v1 (EOSTok, ByteDance Seed). The mechanically
relevant findings for img2emb are **implicit alignment** (don't constrain the
K query outputs directly; align *intermediate* encoder features to a frozen
VFM instead) and **end-to-end-only supervision** (a separately-trained
representation phase chases targets that the downstream task doesn't actually
care about).

## Goal

Image → K continuous embedding tokens in (post-T5, pre-DiT) cross-attention
space, used to replace or augment text conditioning. Same downstream
contract as the archived `archive/img2emb/` design; different training story.

## What changes vs the archived design

`archive/img2emb/` trains in two stages:

1. `pretrain.py` — positional MSE on K resampler outputs against the cached
   T5 `crossattn_emb` targets, plus anchor classifier heads + InfoNCE.
2. `finetune.py` — flow-matching MSE through the frozen DiT, warm-started
   from the pretrain checkpoint.

The paper's Table 2 (1D row, "+ Direct alignment" vs "+ Implicit alignment":
gFID 5.98 → 3.32) says supervising the queries against a fixed positional
target *worsens* generative quality versus letting them be free. Our
pretrain stage is structurally the same as their "direct alignment" — it
forces the K resampler outputs to match a positionally-aligned T5 target
with a per-position MSE/cosine loss. That's the ablation they ran and lost.

Concrete changes:

- **Drop the dedicated pretrain stage.** No more positional MSE against
  `crossattn_emb_v*`. The K outputs are supervised only by FM loss through
  the frozen DiT.
- **Add implicit alignment.** The resampler's intermediate block output
  (mid-stack, see "decisions" below) gets an aux cosine loss against the
  cached PE-Core (or TIPSv2) layer-L features for the same image.
- **Keep anchor injection optional.** The classifier heads + prototype
  splice are an Anima-specific structural prior orthogonal to the paper.
  Useful for steerability at inference (rating, people_count overrides) but
  not load-bearing for quality. Default off in the new design; opt-in flag
  for users who want the override surface.
- **Drop the T5 variant alignment machinery for training.** With no
  positional MSE, Hungarian alignment of `crossattn_emb_v*` becomes dead
  weight. The shuffled variants are still useful as *input augmentation*
  (the resampler sees one variant per step), but no positional matching
  between them is needed. Keep the alignment code paths gated by a flag
  for the legacy pretrain path; default off.

## Architecture

Two viable shapes. Pick (A) for the cheaper revival; (B) if (A) hits a
manifold ceiling like IP-Adapter has.

### (A) Resampler-only, frozen vision tower (default)

```
image → frozen PE-Core / TIPSv2 → patch tokens
                                       ↓
              learned queries (K) → PerceiverResampler → K output tokens
                                       ↓                        ↓
              implicit-alignment loss              FM loss through frozen DiT
              against frozen VFM patches          (only supervision on outputs)
```

Reuse `library/vision/resampler.py::PerceiverResampler` as-is. The implicit
alignment target is the resampler's *cross-attention KV* (i.e. the projected
patch tokens) — those are already aligned to the frozen VFM patches by
construction, so the alignment "cost" is just adding a small MLP projector
and a cosine loss between `kv` and `f(image)` at the same patch positions.
This is honestly redundant in (A) — the KV is already a linear projection
of frozen VFM features. The point of (A) is to *not* do direct alignment on
the K outputs and rely on FM-only supervision; the implicit-alignment
auxiliary is a safety net that becomes meaningful in (B).

### (B) Trainable mini-ViT with learnable queries (full-fat, paper-faithful)

```
image patches + L learnable queries → trainable causal ViT → discard patch outputs, keep K queries
                                            ↓                              ↓
                            implicit-alignment loss            FM loss through frozen DiT
                            on intermediate hidden patches
                            against frozen PE-Core layer-L
```

This matches the paper's Figure 2 most closely. Higher capacity, breaks
pre-caching of vision features, and is a noticeably bigger build. Defer
unless (A) plateaus.

## Training

Single stage. Loss:

```
L = L_FM + λ_align · L_implicit
```

- `L_FM`: standard flow-matching MSE through frozen DiT. The K resampler
  outputs replace (or are concatenated with — see "decisions") the T5
  cross-attention path.
- `L_implicit`: cosine similarity between an intermediate resampler block's
  output (after a small projector MLP) and the frozen VFM features at
  matching patch positions. In design (A) this is operating on the
  resampler's KV path; in (B) it's operating on the trainable ViT's hidden
  patch embeddings.
- `λ_align`: start at 0.1, sweep [0.01, 0.5]. The paper's Table 2 effect
  size is large but their λ is for a different loss landscape.

Drop:

- Pretrain stage — gone.
- InfoNCE over caption variants — paper provides no support for this and
  it was a phase-1 hack to compensate for the broken pretrain target.
  Remove unless ablation shows it still helps in the FM-only regime.
- Anchor classifier CE/BCE — moved behind opt-in flag.

Keep:

- Variant shuffling as input augmentation (one variant per step).
- VAE/text-emb/PE caching pipeline (we still consume cached T5 targets for
  the *DiT*, not for the resampler — the DiT cross-attention runs against
  the resampler output during finetune, and FM supervision is unchanged).
- Bucketed image preprocessing.

## Implementation steps

1. **Resurrect skeleton.** Copy `archive/img2emb/{anchors,resampler,buckets,encoders,data}.py` paths
   that aren't already in `library/vision/` into `scripts/img2emb/` (or
   straight into `networks/methods/` if we're treating it as a first-class
   adapter family — see "decisions"). Drop `align_variants.py`,
   `pretrain.py`, `rebuild_anchor_artifacts.py` from the live tree;
   leave them in `archive/img2emb/` for reference.

2. **Add implicit-alignment hook.** In `library/vision/resampler.py`,
   expose an intermediate block output (e.g. after block N/2). Add a small
   projector MLP `h_align: d_model → d_pe` and a cosine loss helper in
   `scripts/img2emb/loss.py`.

3. **Single-stage trainer.** New `scripts/img2emb/train.py` that:
   - Loads the resampler + projector.
   - Runs FM loss through frozen DiT using K resampler outputs as the
     cross-attention context.
   - Adds `λ_align · L_implicit` against cached PE features.
   - One variant per step (shuffle augmentation only).

4. **Inference.** Reuse `archive/img2emb/infer.py` shape. Default to no
   anchor overrides; surface `--slot_override` only when anchor heads are
   trained.

5. **Bench harness.** `bench/img2emb/` (new) with the standard
   `_common.py` envelope. Metrics: rFID-style reconstruction-from-image
   (image → resampler → DiT generate → compare), CLIP score against
   GT caption.

## Decisions to make

- **Output contract.** Replace text cross-attention or concatenate? The
  archived design replaced (K=256 zero-padded to 512). Concatenation
  preserves text steerability at inference cost. Pick replace for
  simplicity; revisit if user feedback wants both.
- **Anchor heads on/off by default.** Off — the paper's "free queries"
  argument and our slot-collapse history both push that way. Keep the
  classifier code gated.
- **Encoder choice.** PE-Core-L14-336 (already used by IP-Adapter; cache
  shared) is the obvious default. TIPSv2 stays as an option for ablation.
  PE-G is overkill until (A) saturates.
- **Live adapter or scripts-only?** The archived design was scripts-only
  (no `networks/methods/` entry). For revival, scripts-only is faster
  and matches img2emb's "replaces text path" semantic — it's not a
  per-block adapter. Stay in `scripts/img2emb/`.
- **`λ_align` scaling.** Auxiliary loss scaling is the most likely
  hyperparameter to need tuning. Sweep before declaring (A) done.

## Why no quantizer

The paper's IBQ + APR machinery exists to defend against codebook collapse
in AR next-token prediction. img2emb has no AR head; the K outputs go into
DiT cross-attention as continuous tokens. There is no vocabulary, no NTP
loss, and no codebook-collapse failure mode to fix. Continuous K tokens
are the right primitive here.

The collapses we *do* care about (postfix slot collapse; IP-Adapter
manifold collapse) are structural/representational, not entropic — a
codebook would either be neutral or actively harmful. Ignore the IBQ
section of the paper.
