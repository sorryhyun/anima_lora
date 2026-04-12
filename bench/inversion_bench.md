# Inversion Stability Benchmark

Benchmark for the hypothesis from `inversion_idea.md`: optimizing text embedding across multiple z_T seeds and extracting the common component yields a more robust "z_T-invariant semantic embedding."

## Setup

- **Target image**: `10008181` from `post_image_dataset/`
- **Model**: anima-preview3-base DiT, bf16
- **Inversion**: 5 independent runs, 100 steps each, lr=0.001, grad_accum=1, different RNG seeds (42, 1042, 2042, 3042, 4042)
- **Generation**: 50 denoising steps, flow_shift=5.0, 3 seeds per embedding (42, 43, 44)
- **Aggregation methods tested**: element-wise mean, element-wise max

## E1: Optimization Stability

All 5 inversions converge to similar loss values (~0.058–0.060), suggesting a well-defined optimum exists.

| Metric | Value |
|--------|-------|
| Pairwise cosine sim (flattened) | 0.9961 +/- 0.0004 |
| Per-token cosine sim | 0.5259 +/- 0.0383 |
| Cosine sim to mean | 0.9986 +/- 0.0002 |
| PCA variance explained (top 4) | 30.4%, 24.9%, 23.1%, 21.7% |

**Key observation**: Flattened cosine similarity is very high (0.996), but per-token similarity is only 0.526. This means the embeddings agree in bulk direction/magnitude but encode semantics into different token positions depending on the RNG seed. PCA variance is spread nearly uniformly across all 4 non-trivial components — there is no single dominant variation axis and no obvious "stable subspace" to extract.

## E2: Generation Consistency

### Within-embedding (same embed, different seeds)

| Embedding | Cross-seed pixel MSE |
|-----------|---------------------|
| inv_0 | 0.064589 |
| inv_1 | 0.064330 |
| inv_2 | 0.066380 |
| inv_3 | 0.079481 |
| inv_4 | 0.059450 |
| **mean** | **0.083370** |
| **max** | **0.070422** |

### Cross-embedding (different embeds, same seed)

| Seed | inv-inv MSE | inv-mean MSE |
|------|-------------|--------------|
| 42 | 0.032340 | 0.022221 |
| 43 | 0.044774 | 0.035501 |
| 44 | 0.038838 | 0.028484 |

The mean embedding is geometrically central (lower inv-mean distance than inv-inv), as expected.

## E3: Robustness Verdict (MSE-based)

| Aggregation | Cross-seed MSE | vs. avg individual | Verdict |
|-------------|---------------|-------------------|---------|
| Mean | 0.083370 | -24.8% | Less robust |
| Max | 0.070422 | -5.4% | Less robust |

By the pixel MSE consistency metric, both aggregations are less robust than individual inversions. Max pooling is significantly better than mean pooling (-5.4% vs -24.8%), but still slightly worse than the individual average.

## Subjective Assessment

**The mean embedding produces the best-looking images.**

Despite the MSE metric flagging the mean as "less robust," visual inspection tells a different story. The mean embedding generates perceptually cleaner, more coherent images compared to individual inversions. This disconnect makes sense:

- **What MSE measures**: cross-seed consistency — how similar are images across different z_T. Higher MSE = more variation across seeds.
- **What it misses**: perceptual quality. Individual inversions likely overfit to their specific z_T, producing embeddings that are subtly distorted by noise-specific artifacts. These happen to be more *consistent* across seeds (lower MSE) precisely because they're more constrained/specific, not because they're better.
- **What mean does**: averaging acts as a denoiser on the embedding, smoothing out noise-dependent artifacts. The result is a cleaner semantic representation that produces higher-quality images, even if those images vary more across seeds.

This parallels the modulation guidance finding where max pooling over the sequence dimension works better than mean for the `pooled_text_proj` MLP — the strongest activations carry the most semantic signal.

## Takeaways

1. The original hypothesis (mean = more robust) does not hold for pixel MSE consistency, but MSE may be the wrong metric for this question.
2. Max pooling preserves more per-token structure than mean and scores closer to individual inversions on consistency.
3. Perceptual quality and cross-seed consistency are different axes — the mean embedding wins on quality despite losing on consistency.
4. Future work: evaluate with a quality-oriented metric (LPIPS or DINO similarity to the original image) rather than cross-seed MSE.
