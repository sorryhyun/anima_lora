# Phase 2 — flow-matching supervision through the frozen DiT

Phase 1.5 hit a wall that isn't architectural: held-out `content_all` cos plateaus
around 0.60 regardless of resampler depth, anchored prefix classes, or encoder
choice. The wall is the supervision target, not the model. Phase 2 keeps the
resampler and the anchored prefix head unchanged and swaps the loss: stop
regressing to cached `crossattn_emb`, start minimizing flow-matching loss on
the real image latent through the frozen DiT with the resampler output as
cross-attn context.

## Why the target is the ceiling

The cached `crossattn_emb` target is `LLM_adapter(T5(caption))`. Captions encode
what booru chose to tag — identity, counts, color, rating, some pose. They do
not encode per-instance layout, pose articulation, camera angle, lighting
direction, micro-expression, composition specifics. Any mapper trained to
regress into `crossattn_emb` is forced into the caption-embedding manifold,
which is a strict subset of the `(512, 1024)` tensors the DiT's cross-attention
will consume productively. From the inversion benches:

- Per-slot rank at k@95 ≈ 300 / 1024: ~70 % of slot dimensions are directions
  the DiT is invariant to *along the caption manifold*. They aren't noise in
  an absolute sense — they're just unobserved by captions.
- Suffix-tag position-invariance cos ≈ 0.99 and prefix k@95 ≤ 5 confirm the
  caption manifold is low-rank and largely prototype-addressable. Phase 1.5
  exploited this for the prefix slots (cos 0.936 ✓) and hit the ceiling on
  content slots where free-form-but-captionable overlaps with
  image-content-not-in-captions.
- Classifier heads at phase 1.5 hit rating 0.85 / count 0.83 / artist 0.84 —
  the encoder is not the bottleneck for *captionable* content. So adding a
  wider classifier on top of the same regression loss just trades resampler
  residual for lookup residual; it does not move the wall.

The DiT itself provides a stronger loss signal: "does this cross-attn input
make the frozen DiT reconstruct this image?" That signal spans the full
cross-attn-consumable manifold, not just the caption slice, and it costs
exactly one extra DiT forward per step.

## Proposed change in one line

Replace the phase 1.5 loss
```
w_mse · MSE(pred, target) + w_cos · (1 − cos(pred, target)) + w_zero · MSE(pred[L:], 0)
```
with a flow-matching loss through the frozen DiT, conditioned on the
resampler's output in place of `crossattn_emb`:
```
sigma ~ p_sigma
x_sigma = sigma · noise + (1 − sigma) · latent
v_pred = DiT(x_sigma, sigma, crossattn_emb = resampler(image), pooled=...)
v_target = noise − latent                   # OT velocity
loss = weight(sigma) · MSE(v_pred, v_target)
```
All DiT parameters are frozen. Gradients flow back through `crossattn_emb`
into the resampler and anchored classifier heads.

The anchored-prefix classifier stays — but its CE auxiliary loss stays too, to
prevent prefix slots from drifting off the prototype library (and to keep the
pipeline interpretable).

## Architecture (unchanged from phase 1.5)

- Encoder: frozen siglip2-large-patch16-384 (phase 0 winner).
- Classifier heads (rating / count / artist) + frozen prototype library from
  `bench/inversionv2/results/tag_slot/phase2_*` and `phase3_*` → prefix slots.
- 4-layer Perceiver Resampler, dim 1024, 8 heads, 512 queries → content slots.
- Output shape matches DiT cross-attn input: `(B, 512, 1024)` bf16.

Total trainable params: ~71 M, same as phase 1.5.

## Data

- Source: `post_image_dataset/` with cached VAE latents (`*_anima_vae.safetensors`)
  and cached `crossattn_emb_v{0..7}` (still used for warm-start and metrics,
  not loss).
- New requirement: per-image cached VAE latent is the supervision target, not
  the cached text embedding. Latents already live on disk from the standard
  preprocess pipeline — verify before training.
- Same 80 / 20 image-level split as phase 1.5 (`results/phase0/split.json`),
  stratified over the artist/rating distribution.

## Training loop

Reuse `library/anima/training.py` and `AnimaTrainer.get_noise_pred_and_target`
as much as possible — the flow-matching loss path is already battle-tested for
LoRA training. Minimal shape of the new loop:

```
for batch in dataloader:
    image, vae_latent = batch          # latent shape (B, C, 1, H, W)
    with torch.no_grad():
        patches = siglip2(image)       # (B, N_patch, D_enc)

    # cond: anchor head + resampler
    prefix_logits = classifier(patches.mean(1))
    prefix_slots  = prototype_lookup(prefix_logits)          # (B, 8, 1024)
    content_slots = resampler(patches)                       # (B, 504, 1024)
    ctx = torch.cat([prefix_slots, content_slots], dim=1)    # (B, 512, 1024)

    # flow-matching loss
    sigma = sample_sigmas(batch_size)
    noise = randn_like(vae_latent)
    x_sigma = sigma * noise + (1 - sigma) * vae_latent
    with torch.set_grad_enabled(True):
        v_pred = dit(x_sigma, sigma, crossattn_emb=ctx, pooled=None)
    v_target = noise - vae_latent
    fm_loss = weight(sigma) * F.mse_loss(v_pred, v_target)

    # auxiliary CE on prefix classifiers (keeps anchors interpretable)
    ce_loss = ce(prefix_logits_r, rating_gt) + ce(prefix_logits_c, count_gt)
             + ce(prefix_logits_a, artist_gt)

    loss = fm_loss + 0.1 * ce_loss
    loss.backward()
```

Notes / invariants:

- The DiT expects max-padded cross-attn (attention-sink invariant, see
  `CLAUDE.md` §"Text encoder padding"). `ctx` is already `(B, 512, 1024)`
  max-padded; no change needed. Zero-pad slots past active length in the
  resampler output — keep a small regularizer (`w_zero · MSE(ctx[L:], 0)`
  with `w_zero = 0.001`) so the model preserves the attention-sink structure.
- Constant-token bucketing stays; VAE latents already live at the 4096-patch
  target. Reuse the dataset's bucket sampler.
- Block-swapping is fine (one DiT forward, no second pass); use the `default`
  or `low_vram` preset.
- Gradient through the DiT crosses into the resampler via cross-attn K/V
  projections — memory footprint is roughly the same as LoRA training with
  `network_dim = 0` (no DiT-side trainables, but activations still stored for
  backward through the K/V projections). Expect ~16 GB at `batch_size = 2`.
- Mixed precision bf16, AdamW, `lr = 1e-4` on resampler, `lr = 3e-5` on
  classifier heads (same ratio as phase 1.5 when it was working).
- Train steps: start with 20 k; re-evaluate at 5 k / 10 k / 20 k checkpoints.

## Evaluation

Phase 2 needs a new primary metric because `crossattn_emb` regression cos is
no longer the loss. Three tiers:

1. **Flow-matching held-out loss.** Same sigma distribution as training, held-
   out image split. Primary number.
2. **Reconstruction FID / LPIPS (held out).** Run `inference.py` with the
   resampler output in place of the text path, generate an image, compare to
   the held-out ground truth. ~200 images, 20-step sampler, `guidance = 1.0`
   (no CFG needed — we're measuring reference fidelity, not prompt
   adherence). Metrics: FID against held-out set, LPIPS per-image against the
   paired ground truth.
3. **Visual eyeball on ~20 held-out images.** Side-by-side grid: (a) ground
   truth, (b) phase 1.5 prediction → inference, (c) phase 2 prediction →
   inference, (d) real-caption inference, (e) inversion-embedding inference
   (upper bound, expensive). Human pass.

Keep the old per-slot cos metric *as a diagnostic only*, computed against the
variant-mean `crossattn_emb`. Expectation: content cos will likely **drop**
below phase 1.5's 0.60 as the resampler escapes the caption manifold. That is
not a regression — it's the hypothesis. The diagnostic tells us by how much
the output has moved off-caption; reconstruction quality tells us whether the
move was in a useful direction.

## Phase 2a — cheap warm-start ablation

Before committing to the 20 k-step run, do a 2 k-step warm-start from the
phase 1.5 checkpoint with flow-matching loss. Two possible outcomes:

- **FM loss drops, content-cos drops, reconstruction FID improves** → the
  supervision-is-the-ceiling hypothesis is right. Commit to the full run.
- **FM loss drops, content-cos drops, reconstruction FID degrades** → the
  resampler is escaping the caption manifold in a direction the DiT can't
  use. Either the anchor classifier is leaking errors, or the resampler is
  finding an adversarial shortcut. Stop and diagnose before the full run.
- **FM loss doesn't drop** → setup bug (frozen grads, wrong context shape,
  CE loss dominating). Fix and retry.

This cheap ablation is a hard gate on the full run. ~1 hour on a 4090.

## Risks / open questions

1. **Attention-sink leakage.** The DiT was trained expecting zero-padded tail
   slots. If FM loss pushes the resampler to emit non-zero values there,
   cross-attn softmax re-normalizes in a way the DiT hasn't seen. The small
   `w_zero` regularizer should hold; monitor `mean(|ctx[L:]|)` as a
   diagnostic and raise the weight if it drifts above 1e-3.
2. **Identifiability collapse.** Without regression to `crossattn_emb`, the
   resampler could converge to a solution that reconstructs held-out images
   but doesn't generalize — e.g., memorizing siglip2 patches as keys. Guard
   with (a) held-out FM loss as the primary metric, (b) the CE auxiliary loss
   keeping prefix slots on the prototype manifold, (c) optional small
   retention term `0.001 · MSE(content_slots, variant_mean_crossattn_emb)`
   on a subset of slots — use only if identifiability is visibly broken.
3. **Classifier-error amplification.** The prefix anchor feeds classifier
   predictions into a loss that now grades reconstruction, not slot values.
   A mis-rated image (classifier says "general" on an explicit image) pushes
   the resampler to compensate on content slots, producing a weird
   crossattn_emb. Fix option: straight-through soft anchor (use the full
   classifier softmax mixed against prototype library instead of hard
   argmax). Defer unless ablation shows it matters.
4. **Compute cost.** ~2–3× phase 1.5 per step (one extra DiT forward + larger
   backward). A 20 k-step run is ~5–7 hours on one 4090 at bs=2. Acceptable
   for a bench; verify `blocks_to_swap` + gradient checkpointing fit before
   committing.
5. **Artist-slot ambiguity is still real.** The FM loss will learn
   "style-that-looks-like-sincos" rather than the literal `@sincos` tag when
   the image under-determines authorship. That's correct behavior, same
   framing as in `proposal.md` risk #2. Don't try to fix it.
6. **This does not by itself reach option C.** If phase 2 still hits a
   ceiling — specifically, if held-out LPIPS against ground truth saturates
   above what one would expect from a real IP-Adapter baseline — the
   remaining gap is per-block architectural: the K/V projections weren't
   designed for image-derived features. Phase 3 (parallel cross-attn /
   IP-Adapter shape) becomes the next step, and phase 2's resampler +
   anchored classifier drop in unchanged as the image-feature producer.

## Success criteria

- Phase 2a ablation passes: FM loss drops on held-out, reconstruction FID
  improves vs phase 1.5 predictions piped through inference.
- Phase 2 full run: held-out FM loss ≥ 15 % below phase 1.5's FM loss
  (measured by running the phase 1.5 checkpoint under the same eval harness),
  AND visual eyeball on 20 images shows at least 14 / 20 phase 2 outputs
  preferred over phase 1.5 on "resembles the reference image".
- End state: the same `scripts/encode_image_to_emb.py` deliverable as
  `proposal.md` success criteria, but now trained with supervision that
  actually corresponds to reference fidelity — not caption reconstruction.

## Non-goals

- Not adding new architecture. Resampler + anchors stay exactly as in
  phase 1.5. Any structural escalation is deferred to phase 3.
- Not using the cached `crossattn_emb` as a loss target. It survives only
  as a diagnostic and as a source of prefix-slot prototypes.
- Not training the DiT. Fully frozen, as in all prior phases.
- Not training the siglip2 encoder. Frozen, as in all prior phases.
- Not introducing a secondary image-branch cross-attn inside the DiT. That
  is phase 3, and only if phase 2's ceiling turns out to be per-block, not
  supervision-side.

## Open calls needed before phase 2a

- Confirm `post_image_dataset/` VAE latents are cached and loadable by the
  existing dataset code (likely yes, same pipeline as LoRA training).
- Confirm reuse plan for `AnimaTrainer.get_noise_pred_and_target` — either
  import the function into a standalone `phase2_flow.py`, or write a slimmer
  reimplementation that drops LoRA-specific bookkeeping.
- Pick sigma distribution: training uses the same `--weighting_scheme` as
  the standard LoRA configs (default: logit-normal). Confirm before the run.
- Confirm eval FID reference set: probably the full 397-image held-out split
  at their native resolution, VAE-decoded from ground-truth latents.
