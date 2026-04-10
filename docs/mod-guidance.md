# Modulation Guidance

Training-free quality steering via text-conditioned AdaLN modulation, based on [Starodubcev et al., "Rethinking Global Text Conditioning in Diffusion Transformers" (ICLR 2026)](https://arxiv.org/abs/2602.09268).

## How it works

Anima's AdaLN modulation path is originally text-blind — the shift/scale/gate coefficients that control every sublayer are functions of timestep only. Text conditioning enters exclusively through cross-attention.

Modulation guidance adds a second text-conditioning channel by:

1. **Injecting a pooled text embedding into the modulation path** — a small learned MLP projects `max_pool(crossattn_emb)` into the timestep embedding space, making AdaLN coefficients text-aware.
2. **Applying guidance in modulation space at inference** — steering the AdaLN coefficients toward quality-positive and away from quality-negative directions, orthogonal to classifier-free guidance in noise space.

| Component | Text-dependent? | Notes |
|-----------|:-:|-------|
| Cross-attention KV | Yes | Qwen3 → LLMAdapter → 28 blocks |
| AdaLN shift/scale/gate | **Yes** (after training) | `t_embedder` + `pooled_text_proj` |
| CFG | Yes | Noise-space guidance (cond − uncond) |
| **Modulation guidance** | **Yes** | AdaLN-space guidance (pos − neg) |

## Architecture

### Pooled text projection

A 2-layer MLP injected into the model's forward path:

```python
# In Anima.__init__:
self.pooled_text_proj = nn.Sequential(
    nn.Linear(1024, model_channels),   # crossattn_emb dim → 2048
    nn.SiLU(),
    nn.Linear(model_channels, model_channels),
)
# Output layer is zero-initialized (no-op before distillation training)

# In Anima.forward, after t_embedding_norm:
pooled = crossattn_emb.max(dim=1).values   # (B, 1024)
t_embedding_B_T_D = t_embedding_B_T_D + self.pooled_text_proj(pooled).unsqueeze(1)
```

**Pooled text source** — max-pool from `crossattn_emb` (post-LLMAdapter), not raw Qwen3 outputs. The HydraLoRA routing analysis ([hydra-lora.md](hydra-lora.md)) empirically evaluated pooling strategies on this encoder with 1416 images across 37 artists:

| Strategy | Source | KMeans NMI |
|----------|--------|:---:|
| **Max pool** | **crossattn_emb** | **0.926** |
| Mean pool | crossattn_emb | 0.551 |
| Mean pool | prompt_embeds | 0.400 |
| EOS token | prompt_embeds | 0.170 |
| EOS token | crossattn_emb | 0.089 |

Max pool captures per-dimension peak activations from positions 16–50 (visual attribute tags), which fingerprint prompt-specific features. HydraLoRA already computes `crossattn_emb.max(dim=1).values` for expert routing — one pooling, two consumers.

### Injection point: after `t_embedding_norm`

The projection output is added to `t_embedding_B_T_D` **after** `t_embedding_norm`. Benchmarked against two alternatives:

| Injection point | MSE @ α=2.0 | MSE @ α=8.0 | Growth α=4→8 |
|-----------------|-------------|-------------|---------------|
| before_norm | 4.77e-4 | 1.08e-2 | 6.1x |
| **after_norm** | **4.76e-3** | **1.89e-1** | **7.2x** |
| adaln_lora | 4.29e-3 | 4.06e-2 | 2.6x |

`after_norm` has ~10x more sensitivity than `before_norm` (the norm re-centers perturbations) and ~4.7x the saturation headroom of `adaln_lora`.

### Inference guidance

At inference time, modulation guidance steers AdaLN coefficients using quality-axis prompts:

```
emb = t_embedding + proj(pool(main)) + w * (proj(pool(p₊)) − proj(pool(p₋)))
```

The guidance delta `w * (proj(pool(p₊)) − proj(pool(p₋)))` is computed once and reused across all denoising steps. It is stored on `anima._mod_guidance_delta` and added to `t_embedding_B_T_D` in every forward pass.

Default guidance prompts use booru-style quality/score tags (in-distribution for Anima's text encoder):

| Aspect | p₊ | p₋ |
|--------|-----|-----|
| Quality+resolution | *"absurdres, masterpiece, score_9"* | *"worst quality, low quality, score_1"* |

Resolution tags (`absurdres`) are included because the quality and resolution directions correlate per-content (cosine 0.50), which works in our favor — quality guidance naturally pulls toward resolution too.

## Usage

### Inference

```bash
python inference.py \
    --pooled_text_proj path/to/pooled_text_proj.safetensors \
    --mod_w 3.0 \
    --mod_pos_prompt "absurdres, masterpiece, score_9" \
    --mod_neg_prompt "worst quality, low quality, score_1" \
    # ... other args
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pooled_text_proj` | — | Path to trained projection weights (.safetensors) |
| `--mod_w` | 3.0 | Guidance strength. Higher = stronger quality steering |
| `--mod_pos_prompt` | `"absurdres, masterpiece, score_9"` | Positive quality prompt |
| `--mod_neg_prompt` | `"worst quality, low quality, score_1"` | Negative quality prompt |

### Distillation training

The projection MLP must be trained via distillation before modulation guidance can be used. This trains only `pooled_text_proj` (~8M params) with the rest of the model frozen.

**How distillation works:**
1. Teacher forward: full model with `pooled_text_proj` disabled (original behavior)
2. Student forward: `crossattn_emb` zeroed out, but real pooled text injected via `pooled_text_proj` — forces the model to perceive text only through modulation
3. Loss: MSE between student and teacher noise predictions

```bash
python scripts/distill_modulation.py \
    --data_dir image_dataset \
    --dit_path models/anima_v2-F16.safetensors \
    --iterations 4000 \
    --lr 1e-4 \
    --batch_size 2 \
    --blocks_to_swap 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | — | Directory with cached latents/text embeddings |
| `--dit_path` | — | Base model path |
| `--iterations` | 100 | Training iterations |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 2 | Batch size |
| `--blocks_to_swap` | 0 | CPU-offload N blocks for VRAM savings |
| `--torch_compile` | false | Enable torch.compile on block forwards |

Output: `pooled_text_proj.safetensors` in the data directory.

## Compatibility

| Feature | Interaction |
|---------|-------------|
| **T-LoRA** | Orthogonal — T-LoRA masks LoRA rank by timestep; modulation guidance steers AdaLN coefficients. Different parameter spaces. |
| **CFG** | Complementary — CFG in noise space, modulation guidance in AdaLN space. They stack. |
| **P-GRAFT** | Compatible — modulation guidance runs independently of LoRA presence. |
| **Spectrum** | Compatible — Spectrum skips blocks but still runs `t_embedder` + `final_layer`. Guidance delta applies to `emb_B_T_D` before blocks, carried through on cached steps. |
| **HydraLoRA** | Shared pooling — both consume `crossattn_emb.max(dim=1).values`. Same pooled vector, orthogonal purposes. |
| **LoRA training** | No conflict — LoRA explicitly excludes `pooled_text_proj` via pattern matching in `lora_anima.py`. |

## ComfyUI

Two node implementations exist:

- `custom_nodes/comfyui-mod-guidance/` — standalone node, takes pre-computed pooled embeddings
- `Anima-Mod-Guidance-ComfyUI-Node/` — full node with adapter auto-download, per-block scaling via `start_layer`/`end_layer` controls, and CLIP pooled output integration

## Design rationale

### Quality-axis separation in embedding space

`max_pool(crossattn_emb)` separates quality-positive from quality-negative prompts. Tested with 8 diverse content prompts:

| Metric | Value |
|--------|-------|
| Avg pos↔neg cosine distance (content-varied) | 0.038 |
| Avg same-quality cosine distance (different content) | 0.031 |
| Separation ratio | 1.22x |

### Quality direction consistency across content

The quality direction `max_pool(p₊) - max_pool(p₋)` is consistent across 8 diverse content types:

| Metric | Value |
|--------|-------|
| Average pairwise cosine similarity | 0.814 |
| Minimum pairwise cosine similarity | 0.770 |

All 28 pairwise similarities exceed 0.77 — a single global guidance direction generalizes across content.

### Modulation sensitivity

Perturbing `emb_B_T_D` with the quality direction produces smooth, monotonic changes in noise predictions. The high-noise regime (t=0.9) is most sensitive, consistent with the model relying more on modulation at early denoising steps.
