# IP-Adapter PE feature analysis — 2026-05-02

Working note pairing with `bench/ip_adapter/pe_feature_analysis.py`. Records
what we measured, what it means for the IP-Adapter wall hit in
`docs/methods/ip-adapter-0502.md`, and what changed in the codebase as a
result.

## The decision question

Run #1 of IP-Adapter (`20260502164913`) was the first run in this codebase's
history with positive `epoch_baseline_no_ip_delta`, but the gate barely
opened (`abs_max ~0.004` peak vs init 0). Run #2 has bumped `gate_lr` to
1e-3 / 1e-2, but before spending more training budget we wanted to know:

> Is the IP-Adapter wall a **data problem** (PE features fingerprint the
> source pixels, self-paired training reduces to lookup) or an
> **optimization problem** (gate / LR / budget)?

The answer determines whether to deprecate IP-Adapter for EasyControl. If
it's a data problem and the data is self-paired, EasyControl will hit the
same wall — different architecture, same pathology. If it's an optimization
problem, IP-Adapter just needs more knobs tuned.

## Methodology

Three feature-only measurements over the 2407 cached PE-Core features in
`post_image_dataset/lora/`. No IP-Adapter training needed — these probe the
PE features themselves.

1. **Aug-invariance histogram.** For 200 random images, compute mean-pooled
   PE for each + hflip / center-crop / color-jitter variants + a random
   *other* image. Cosine similarities. Tells you whether aug-pairs cluster
   near 1.0 or collapse toward the cross-pair distribution. Gap = the
   "aug-invariant signal" the resampler can in principle extract.

2. **Crop retrieval rank.** From a 60% random crop of a query image, find
   the rank of the source in the full cache by pooled cosine sim.
   `recall@1 ≈ 100%` means PE features fingerprint source pixels — IP path
   can always recover the target, so self-paired training has nothing
   left to learn beyond memorization.

3. **Effective rank of pooled feature distribution.** SVD on the (N × D)
   pooled-feature matrix. `K=16` resampler tokens × D dims is the IP path's
   capacity; if dataset's pooled features live in a sub-space much smaller
   than that, capacity isn't the bottleneck.

Mean-pool drops CLS, averages over patch tokens. The resampler attends to
all tokens, not the mean — so mean-pool similarity is a *lower bound* on
the resampler's distinguishability. Strong collapse at the mean-pool level
is a strong signal; high mean-pool separation says "the global signal is
fine" but doesn't rule out finer-grained issues.

## Results

```
=== Step 1: aug-invariance histogram (N=200) ===
  pair                  mean    std    p10    p50    p90    min    max
  self_hflip           0.992  0.006  0.984  0.993  0.997  0.957  0.999
  self_crop(0.60)      0.877  0.076  0.775  0.888  0.956  0.555  0.982
  self_jitter(0.20)    0.996  0.004  0.991  0.997  0.999  0.973  1.000
  cross_random         0.686  0.151  0.474  0.707  0.856  0.199  0.945

  Gap (self-aug - cross): hflip +0.306 / crop +0.191 / jitter +0.309

=== Step 2: crop retrieval rank (Q=50, index size=2407, frac=0.60) ===
  recall@1   = 0.220  (11/50)
  recall@10  = 0.360
  recall@100 = 0.700
  median rank = 26

=== Step 3: effective rank ===
  Matrix: (2407, 1024)  (centered)
  95% energy in top 46 dims
  99% energy in top 250 dims
  participation ratio: 6.2
  → 95%-energy rank 46 is above K=16 resampler-tokens
```

## Interpretation

### 1. PE features are aug-invariant for hflip and color jitter, less so for crop

hflip and jitter@0.2 both sit at ~0.99 cos vs 0.69 cross — gap > 0.30, way
above any reasonable "decision threshold." For the IP path's purposes those
augs are near-identity. crop(0.60) drifts to 0.88 with bottom-decile at
0.78, and 10% of samples land *below* the cross-pair p90 (0.856) — i.e.
crop is sometimes more disruptive than swapping to a different image
entirely. Crop is risky.

### 2. Memorization is NOT the failure mode

Crop retrieval recall@1 = 0.22 says PE features cannot reliably look up
the source from a 60% crop. Median rank 26 / 2407 puts source in the 99th
percentile but not at the top. PE features carry real identity signal,
not pixel-fingerprint. **This kills the "self-paired training reduces to
lookup" hypothesis** — the IP path can't shortcut.

### 3. The actual wall: narrow signal on a collapsed manifold

Cross-pair mean = 0.69. Participation ratio = 6.2. 95% energy in 46 dims.
The whole dataset's pooled PE features sit on a ~6-dimensional sub-space,
and any two random images share ~0.69 cos. The IP path's discriminative
signal is a **small per-image delta on top of a strong common-mode
background**.

This is the regime where:
- A slow-opening gate doesn't get enough gradient through to find the
  delta direction quickly.
- The resampler's K=16 tokens have *plenty* of capacity (95% energy needs
  46 dims, but K×D = 16×2048 ≫ 46), so capacity isn't the bottleneck.
- Run #1's "gate barely opens, val Δ still positive" is consistent: the
  signal exists, the optimizer just can't extract it fast.

### Verdict

**Don't deprecate IP-Adapter on data grounds.** EasyControl is also
self-paired and would trade manifold collapse for a different pathology
(VAE latents are pixel-faithful → less aug-invariance, but more raw
discrimination). The decision should be made on architectural fit, not
on the assumption that EasyControl avoids this wall.

## What changed

### `bench/ip_adapter/pe_feature_analysis.py` (new)

Three-step diagnostic. Runs in ~1 minute on the 2407 cached PE features.
Loads PE-Core for steps 1 and 2 (live encoding of augs / crops); step 3
is feature-only. Outputs text report + optional JSON.

## Why we considered, then dropped, reference-image augmentation

Initial reading of the data was: hflip and jitter preserve PE pool to
~0.99 cos, so they're "essentially free aug" — break pixel memorization
without damaging supervision. Wiring was implemented, then reverted on a
second look. The reasoning matters because future-me may be tempted to
revisit:

1. **The bench measured pool-level invariance.** The resampler eats the
   full `[T, D]` token sequence, not the pool. At the per-token level
   hflip + position embeddings produce a content-position recombination
   that the resampler does see. But the bench gives no quantitative
   handle on how *much* per-token variation remains; we only know the
   pool collapses to identity.

2. **Memorization isn't our failure mode anyway.** Crop recall@1 = 0.22
   (results §2) already says the resampler can't shortcut via feature
   lookup. If memorization isn't happening, "aug as anti-memorization"
   solves a problem we don't have.

3. **Aug doesn't address the real wall.** The actual wall is narrow
   signal on a collapsed manifold (results §3 — cross-pair sim 0.69,
   participation ratio 6.2). Adding feature-space noise to a small
   discriminative signal *makes it harder* to extract, not easier.

If you ever want to revisit this: first re-bench at the per-token level
(cosine sim PE(x)[i] vs PE(aug(x))[i] for each spatial slot i, not just
pool sim) to quantify the actual aug effect on the resampler's input.
If per-token sim is also ~0.99, aug really is a no-op for this
architecture and there's nothing to recover.

## Next steps

In priority order, given the narrow-signal-on-collapsed-manifold
diagnosis:

1. **Dataset-mean centering before resampler** (the recommended next
   move). The participation-ratio-6 collapse says most per-feature
   variance is shared across the dataset. Subtract `mean_pool`
   (computed once over the cache, or as an EMA at training start) from
   every PE feature before the resampler — the resampler then sees only
   the per-image delta, which lives in the high-dim sub-space the
   discriminative signal actually occupies. Cheap (one extra subtract,
   one buffer); no new training mechanics. ~30 LoC in
   `IPAdapterNetwork`.

2. **Shuffled-reference validation baseline** (option D from
   ip-adapter-0502.md). Confirms whether the IP path is binding to the
   reference at all — matched-ref ≫ shuffled-ref means binding works
   but is weak in absolute terms (consistent with narrow-signal); they
   close means binding itself is broken.

3. **Paired-but-different references**, if available in the corpus
   (multi-panel manga of the same character). The narrowness of the
   discriminative signal under self-paired training is partly a
   function of the data — a real "different view, same character" pair
   gives the resampler a stronger learning gradient than self-paired
   variants of the same image ever can.

## Reference points

- Bench: `bench/ip_adapter/pe_feature_analysis.py`
- Status doc: `docs/methods/ip-adapter-0502.md`
- Architecture: `docs/methods/ip-adapter.md`
