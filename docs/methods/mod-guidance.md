# Modulation Guidance

Quality steering via text-conditioned AdaLN modulation, based on [Starodubcev et al., "Rethinking Global Text Conditioning in Diffusion Transformers" (ICLR 2026)](https://arxiv.org/abs/2602.09268). Requires a short distillation step to train the `pooled_text_proj` MLP; thereafter inference-time steering is training-free.

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

At inference time, modulation guidance steers AdaLN coefficients using quality-axis prompts. The base projection is applied uniformly (training-consistent), but the steering delta is scheduled **per DiT block**:

```
base_emb     = t_embedding + proj(pool(main))                    # uniform, training-consistent
delta_unit   = proj(pool(p₊)) − proj(pool(p₋))                   # unit steering direction
emb_at_block[ℓ] = base_emb + w(ℓ) · delta_unit                   # scheduled per block
emb_at_final    = base_emb + w_final · delta_unit                # final_layer override
```

`delta_unit` and `w(ℓ)` are computed once at setup and reused across all denoising steps. They live on `anima._mod_guidance_delta` (unit direction, no `w` baked in), `anima._mod_guidance_schedule` (list of `w(ℓ)` of length `num_blocks`), and `anima._mod_guidance_final_w` (scalar).

**Why schedule instead of uniform?** Applying one global `w` to every block plus `final_layer` caused a drift failure mode on some LoRAs ("channel" collapsing to uniform pink at `w=3` while "sweetonedollar" stayed clean at the same `w`). Per-block functional-gap analysis on Anima's 28-block DiT showed early blocks 0–7 set coarse layout / tonal DC and are sensitive to DC blowout, while block 27 is a compensation/fix-up layer. Matches the Starodubcev et al. App. B–C analysis on FLUX (Strategy 4 — per-block `w`, not per-timestep).

### Profiles

Two named profiles cover the useful operating points — pick the safe one if a LoRA shows anatomy drift (missing fingers, extra digits) at the default.

| Profile | `start_layer` | `end_layer` | `final_w` | Use when |
|---|:-:|:-:|:-:|---|
| **`step_i8_skip27`** (default) | 8 | 27 | 0.0 | Best overall quality. Protects blocks 0–7 + 27. May occasionally show minor anatomy drift on drift-prone LoRAs. |
| **`step_i14`** (safe) | 14 | — | 0.0 | Reliably stays inside the trained manifold. Slightly less expressive — use when `step_i8_skip27` shows drift on the prompt at hand. |

Default guidance prompts use booru-style quality/score tags (in-distribution for Anima's text encoder):

| Aspect | p₊ | p₋ |
|--------|-----|-----|
| Quality+resolution | *"absurdres, masterpiece, score_9"* | *"worst quality, low quality, score_1"* |

Resolution tags (`absurdres`) are included because the quality and resolution directions correlate per-content (cosine 0.50), which works in our favor — quality guidance naturally pulls toward resolution too.

## Usage

### Inference

Default CLI invocation (reproduces the `step_i8_skip27` profile — recommended starting point):

```bash
python inference.py \
    --pooled_text_proj path/to/pooled_text_proj.safetensors \
    --mod_w 3.0 \
    --mod_pos_prompt "absurdres, masterpiece, score_9" \
    --mod_neg_prompt "worst quality, low quality, score_1" \
    # ... other args
```

To switch to the safe `step_i14` profile, add:

```bash
    --mod_start_layer 14 --mod_end_layer -1
```

To recover pre-0413 uniform behavior (not recommended — prone to pink-collapse on drift-prone LoRAs):

```bash
    --mod_start_layer 0 --mod_end_layer -1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pooled_text_proj` | — | Path to trained projection weights (.safetensors) |
| `--mod_w` | 3.0 | Peak guidance strength (`w` applied inside `[start_layer, end_layer)`) |
| `--mod_pos_prompt` | `"absurdres, masterpiece, score_9"` | Positive quality prompt |
| `--mod_neg_prompt` | `"worst quality, low quality, score_1"` | Negative quality prompt |
| `--mod_start_layer` | `8` | Inclusive first block that receives the steering delta. `0` = uniform (pre-0413). `8` = protect tonal-DC blocks 0–7. `14` = safe option. |
| `--mod_end_layer` | `27` | Exclusive last block + 1. `-1` = all remaining blocks. `27` skips Anima's final compensation block. |
| `--mod_taper` | `0` | Number of late slots inside `[start, end)` to scale by `--mod_taper_scale`. `0` disables taper. |
| `--mod_taper_scale` | `0.25` | Multiplier applied to tapered slots. |
| `--mod_final_w` | `0.0` | `w` passed to `final_layer`. `0.0` = don't disturb the output head. |

### Distillation training

The projection MLP must be trained via distillation before modulation guidance can be used. This trains only `pooled_text_proj` (~8M params) with the rest of the model frozen.

**How distillation works:**
1. Teacher forward: full model with `pooled_text_proj` disabled (original behavior)
2. Student forward: `crossattn_emb` zeroed out, but real pooled text injected via `pooled_text_proj` — forces the model to perceive text only through modulation
3. Loss: MSE between student and teacher noise predictions

```bash
make distill-mod
```

Or invoke directly:

```bash
python scripts/distill_modulation.py \
    --data_dir post_image_dataset \
    --dit_path models/diffusion_models/anima-preview3-base.safetensors \
    --output_path output/pooled_text_proj.safetensors \
    --iterations 1500 \
    --lr 1e-5 \
    --warmup 0.05 \
    --blocks_to_swap 0 \
    --attn_mode flash \
    --no_grad_ckpt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | — | Directory with cached latents/text embeddings |
| `--dit_path` | — | Base model path |
| `--iterations` | 100 | Training iterations |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 2 | Batch size |
| `--grad_ckpt` / `--no_grad_ckpt` | on | Gradient checkpointing w/ CPU offload. Disable if you have VRAM headroom — faster iteration. |
| `--blocks_to_swap` | 0 | CPU-offload N blocks. Only needed on very tight VRAM. |
| `--torch_compile` | false | Enable torch.compile on block forwards |

**VRAM notes.** The teacher forward runs under `torch.no_grad()` so it holds almost nothing; the student forward is what dominates peak VRAM (~12 GB on the default config). With `--no_grad_ckpt` you'll see VRAM swing between the weights-only baseline and that student peak — this is normal. Leave `--grad_ckpt` on (the default) only if the peak doesn't fit; if it does, `--no_grad_ckpt --blocks_to_swap 0` is faster.

Output: `output/pooled_text_proj.safetensors`. Use it at inference via `--pooled_text_proj <path>`.

## Compatibility

| Feature | Interaction |
|---------|-------------|
| **T-LoRA** | Orthogonal — T-LoRA masks LoRA rank by timestep; modulation guidance steers AdaLN coefficients. Different parameter spaces. |
| **CFG** | Complementary — CFG in noise space, modulation guidance in AdaLN space. They stack. |
| **P-GRAFT** | Compatible — modulation guidance runs independently of LoRA presence. |
| **Spectrum** | Compatible — Spectrum skips blocks but still runs `t_embedder` + `final_layer`. Guidance delta applies to `emb_B_T_D` before blocks, carried through on cached steps. |
| **HydraLoRA** | Both have consumed pooled crossattn in the past, but HydraLoRA's current router reads the post-`lora_down` rank-R signal — so no shared pool any more. Orthogonal paths. |
| **LoRA training** | No conflict — LoRA explicitly excludes `pooled_text_proj` via the exclude pattern in `networks/lora_anima/factory.py`. |

## ComfyUI

Mod guidance ships inside the [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) repo as a drop-in KSampler replacement. The adapter (`pooled_text_proj_0413.safetensors`, ~12MB) is auto-downloaded on first use from the anima_lora GitHub release page.

| Node | Inputs | When to use |
|---|---|---|
| **KSampler (Spectrum + Mod Guidance)** | `quality_tags`, `mod_w_profile` dropdown | Everyday use. Pick a profile preset; no manual knobs. |
| **KSampler (Spectrum + Mod Guidance Advanced)** | full `start_layer` / `end_layer` / `taper` / `final_w` sliders + Spectrum knobs | Workflow tuning or per-LoRA experimentation. |

The `mod_w_profile` dropdown exposes the same two profiles documented above plus `uniform_w3` for reproducing pre-0413 behavior. Default is `step_i8_skip27`; switch to `step_i14` when a LoRA shows anatomy drift.

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
