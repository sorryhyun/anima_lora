# HydraLoRA: Multi-Style Routing via Text Embeddings

Design notes for MoE-style multi-head LoRA with automatic style routing, targeting multi-artist training in a single LoRA.

## Motivation

When training a single LoRA on multiple artists/art-styles, a standard LoRA blends all styles into one shared low-rank subspace. This causes style bleed — distinct artists lose their visual identity. HydraLoRA addresses this by routing each sample to specialized expert heads based on learned style affinity.

## Architecture

```
gate = softmax(W_router @ max_pool(crossattn_emb))   # [num_experts]
lx = down_shared(x)                                   # shared feature extraction
y = org(x) + Σᵢ gateᵢ · up_i(lx) · scale             # expert-weighted output
```

- **Shared `lora_down`**: captures common artistic knowledge across all styles.
- **Per-expert `lora_up_i`**: each head specializes in a style cluster.
- **Router**: lightweight `Linear(1024, num_experts)` on max-pooled `crossattn_emb`.

The router runs once per sample (before the denoising loop) and broadcasts constant gate weights to all DiT layers across all timesteps.

## Routing Signal Analysis

We evaluated multiple strategies for extracting a style-discriminative signal from the cached text encoder embeddings (`crossattn_emb`, shape `(512, 1024)`, stored in `post_image_dataset/*_anima_te.npz`).

### Pooling strategy comparison

Dataset: 1416 images from 37 artists (>=10 samples each). Metric: NMI of k-means clusters against ground-truth artist labels.

| Strategy | KMeans NMI | Linear Classifier (5-fold CV) |
|---|---|---|
| **max pool** | **0.926** | 0.516 |
| **std pool** | 0.790 | 0.092 |
| **mean+max+std** | 0.911 | 0.472 |
| mean pool | 0.551 | 0.535 |
| EOS token | 0.089 | — |

**Max pool is dramatically superior for clustering** (NMI 0.926 vs 0.551 for mean). The gap between k-means NMI and linear classifier accuracy indicates highly non-linear cluster geometry — k-means finds tight per-artist clusters that a linear boundary cannot separate, suggesting the signal lives in specific activation peaks rather than a linearly separable subspace.

### What max pool captures

Max pool over the sequence selects the peak activation per embedding dimension. These peaks are dominated by positions 16–50 — the visual attribute tags that follow the `@artist` tag in booru-style captions:

| Positions | Content (booru tag order) | Max pool NMI |
|---|---|---|
| 0–4 | Resolution, rating (`absurdres, highres, explicit`) | — |
| 5–15 | Character names, franchise, `@artist` tag | — |
| 16–50 | Visual attributes (`bare_shoulders, black_panties, ...`) | — |
| Full | All | 0.900 |
| Skip 0–19 | Removes `@artist` token entirely | **0.907** |
| Only [20:] | Visual attributes only | **0.874** |
| Skip 0–49 | | 0.398 |
| Only [50:] | Late detail tags only | 0.395 |

Key finding: **max pool does not rely on the `@artist` token**. Skipping positions 0–19 (which contain the artist tag at median position 7) actually *improves* NMI slightly. The signal comes from positions 20–50 — the artist's characteristic visual attribute patterns. Each artist has a consistent "recipe" of attributes (compositions, clothing, visual features) that creates a unique fingerprint in the peak activations.

### Why EOS and mean pool fail

- **EOS (last real token)**: clusters by tokenization artifacts — the literal last character of the caption. NMI against all semantic attributes is near-zero. The t-SNE clusters correspond to `last_token_id` (NMI=0.454) — purely noise.
- **Mean pool**: averages across all tokens, drowning the artist signal in shared content tags (`1girl, blush, breasts, ...` appear across all artists). Captures content/franchise more than style. Confused artist pairs share similar tag overlap (Jaccard 0.27–0.45) with well-separated pairs (0.18–0.35).

### Per-position discriminability

Artist signal concentrates in early-to-mid positions and decays toward padding:

```
Positions [  0- 32]   NMI=0.367
Positions [ 32- 64]   NMI=0.277
Positions [ 64- 96]   NMI=0.157
...
Positions [288-320]   NMI=0.026
Positions [320+]      NMI≈0.000
```

### Embedding space structure

The `crossattn_emb` is the post-projection embedding (mean norm ~1.6) that DiT cross-attends to. It differs from raw `prompt_embeds` (mean norm ~40) — the projection transform concentrates style information:

| Signal | crossattn_mean | crossattn_max | prompt_mean | prompt_eos |
|---|---|---|---|---|
| Artist NMI | 0.55 | **0.93** | 0.40 | 0.17 |
| Content bias | Strong | Moderate | Moderate | None |
| Artifact bias | None | None | None | Dominant |

### Auxiliary features: character count tags

We tested whether adding character count information (`1girl`, `1boy`, `2girls`, etc.) improves cluster separation, particularly for the large catch-all cluster that forms at lower k values.

| Strategy | k=8 NMI | k=8 max% | k=12 NMI | k=12 max% | k=16 NMI | k=16 max% |
|---|---|---|---|---|---|---|
| max only | 0.617 | 45% | 0.703 | 40% | **0.830** | **18%** |
| max + char_region (pos 3–8) | 0.539 | 40% | **0.710** | **27%** | 0.795 | 25% |
| max + char_onehot | 0.129 | 50% | 0.376 | 37% | 0.504 | 29% |
| max + char_region + onehot | 0.160 | 50% | 0.283 | 44% | 0.438 | 37% |

Character count tags do not meaningfully help. The catch-all cluster persists because the artists within it share **rendering style similarity** regardless of whether they draw solo or multi-character compositions. One-hot char features actively harm NMI by forcing hard splits along a style-irrelevant axis. The char_region embedding (positions 3–8) mixes character *names* with count tags, blurring the signal.

### Expert count selection

| k | NMI | Largest expert | Largest % |
|---|---|---|---|
| 8 | 0.619 | 561 | 40% |
| 12 | 0.753 | 444 | 31% |
| 16 | 0.821 | 217 | 15% |

**k=12 is recommended as the starting point.** At k=12, most distinctive artists get dedicated experts while similar-style artists share (e.g., @greatodoggo + @deyui, @belko + @onono imoko). The remaining catch-all cluster (31%) will be further subdivided by the learned router's load-balancing loss during training — the router can discover finer-grained style boundaries that static k-means misses.

k=12 expert composition (static k-means, for reference — learned routing will differ):

| Expert | Size | Top artists | Character |
|---|---|---|---|
| E0 | 444 | @pepper0, @sincos, @bee, @iumu | Catch-all — similar attribute recipes |
| E1 | 97 | @mikozin (solo) | Blue Archive specialist |
| E2 | 62 | @kase daiki (solo) | |
| E3 | 64 | @fizz (solo) | Bright, soft rendering |
| E4 | 62 | @coro fae (solo) | Detailed illustration |
| E5 | 92 | @greatodoggo + @deyui | Similar tone/rendering |
| E6 | 145 | @asou + @ie + @wagashi | Similar composition patterns |
| E7 | 88 | @tottotonero (solo) | |
| E8 | 126 | @hews (solo) | Polished rendering |
| E9 | 65 | @belko + @onono imoko | Paired by style |
| E10 | 130 | @nora higuma, @wantan meo, @koh | Multi-artist cluster |
| E11 | 41 | @abmayo (solo) | |

### Note on embedding quality

The strong clustering from max pool does **not** indicate that the text encoder or its projection is brittle or poorly trained. The opposite: it reflects that the encoder faithfully encodes distinct information at each token position, and the crossattn projection preserves that structure into the space DiT attends to. Max pool simply reveals per-dimension peak activations that happen to fingerprint artist-specific attribute patterns. The encoder is doing exactly what it should — encoding the full caption with positional fidelity. It is the *pooling strategy* that determines which aspect of that rich representation we extract: mean pool emphasizes shared content, max pool exposes distinctive peaks, and EOS captures nothing useful.

## Router Design

Based on the analysis, the recommended router:

```python
# In LoRANetwork, computed once per sample before denoising
pooled = crossattn_emb.max(dim=0).values          # (1024,)
gate = softmax(self.router(pooled))                 # (num_experts,)

# In each HydraLoRAModule
lx = self.lora_down(x)                             # shared
expert_outs = [up(lx) for up in self.lora_ups]     # per-expert
out = sum(g * e for g, e in zip(gate, expert_outs))
return org_out + out * scale * multiplier
```

**Parameters**: `1024 × num_experts` for the router (e.g., 4096 params for 4 experts). Negligible vs LoRA parameters.

**Training**: the router trains end-to-end with LoRA. A mild load-balancing auxiliary loss prevents expert collapse:

```
L_balance = α · num_experts · Σᵢ (frac_i · gate_mean_i)
```

where `frac_i` is the fraction of samples routed primarily to expert `i`, and `gate_mean_i` is the mean gate value for expert `i` across the batch. Recommended `α ≈ 0.01`.

**Inference**: the router runs automatically from the text conditioning. With fewer experts than artists, similar-style artists share experts naturally. For explicit control, override gate weights manually.

**Saving**: two export modes:
1. **Standard LoRA** (ComfyUI drop-in): bake down `lora_up = Σᵢ wᵢ · up_i` with specified weights. One export per style or blend.
2. **Multi-head format**: full expert weights + router for custom inference nodes with live style blending.

## Composition with Other Variants

- **T-LoRA**: timestep masking applies to the shared `lora_down` latent. Experts specialize by style, T-LoRA controls rank utilization by noise level. Orthogonal axes.
- **DoRA**: each expert head could have its own magnitude vector, or share one. Shared is simpler; per-expert gives more expressiveness.
- **OrthoLoRA**: experts could use orthogonal parameterization independently. Increases init cost proportional to num_experts.

## Open Questions

- **Catch-all cluster**: at k=12, ~31% of samples land in a single expert. The learned router with load-balancing loss should subdivide this during training — validate that it does.
- **Router granularity**: single shared router (all layers get same gate) vs per-block routers. Shared is simpler and gives globally coherent style assignment.
- **Image-level features**: max pool on text embeddings captures artist attribute *recipes* well, but visual style (line weight, shading, color palette) is only indirectly represented. CLIP/DINOv2 image embeddings could capture visual style more directly. Worth comparing if text-based routing proves insufficient for grouping artists with similar subjects but different rendering.
- **Dataset balance**: experts receiving unbalanced gradient signal may undertrain minority heads. The GRAFT curation loop can help maintain per-style balance.
