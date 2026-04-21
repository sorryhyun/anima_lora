# Prefix Tuning

Prefix tuning learns a handful of continuous vectors that are prepended to the cached text embeddings before they enter the DiT. No model weights are modified — the DiT runs in inference mode, and gradients flow only to the prefix vectors through the cross-attention path.

**12 GB peak VRAM, ~1 step/s** on a single consumer GPU — faster than any LoRA configuration.

## Why it's fast

LoRA attaches trainable parameters to every DiT block. Prefix tuning has **zero trainable parameters inside the DiT** — only 16 vectors (16,384 params) sitting outside it:

```
                            LoRA training
                    ┌─────────────────────────┐
                    │  DiT Block 0            │
                    │  ┌─────┐  ┌──────────┐  │
  text embeddings ──┼─►│cross│  │self-attn │  │
   (512 × 1024)    │  │attn │  │ + LoRA   │  │
                    │  │+LoRA│  └──────────┘  │
                    │  └─────┘                │
                    │           ┌──────────┐  │
                    │           │  FFN     │  │
                    │           │ + LoRA   │  │
                    │           └──────────┘  │
                    ├─────────────────────────┤
                    │  DiT Block 1 … 27      │
                    │  (same LoRA in every    │
                    │   block)                │
                    └────────────┬────────────┘
                                 │
                              loss ──► grads to ALL LoRA params
                                       (~300K+ params, optimizer
                                        state for each)


                          Prefix tuning
                    ┌─────────────────────────┐
  ┌──────────┐      │  DiT Block 0            │
  │ 16 prefix│      │  ┌─────┐  ┌──────────┐  │
  │ vectors  ├──┐   │  │cross│  │self-attn │  │
  │(learnable)  │   │  │attn │  │(frozen)  │  │
  └──────────┘  │   │  └──┬──┘  └──────────┘  │
                ▼   │     │                   │
  text embeds ─►cat─┼─────┘    ┌──────────┐  │
  (cached)      │   │          │  FFN     │  │
                │   │          │(frozen)  │  │
                │   │          └──────────┘  │
                │   ├─────────────────────────┤
                │   │  DiT Block 1 … 27      │
                │   │  (all frozen, no LoRA)  │
                │   └────────────┬────────────┘
                │                │
                │             loss ──► grads to 16 prefix vectors only
                │                      (16,384 params total)
                └── prepended once, used by every block's cross-attention
```

This eliminates three major costs:

| Cost | LoRA | Prefix |
|------|------|--------|
| Trainable params | ~300K+ across all blocks | 16,384 (16 vectors) |
| Optimizer state (AdamW8bit) | ~600 KB (2 states per param) | ~32 KB |
| LoRA forward overhead | Extra matmul per block per layer | Zero |
| Block swapping needed | Yes (20 blocks to CPU by default) | No (`blocks_to_swap = 0`) |
| Gradient checkpointing | Typically needed | Not needed |

The last two rows are the real speed wins. Block swapping trades PCIe bandwidth for VRAM — eliminating it removes the IO bottleneck entirely. No gradient checkpointing means no recomputation overhead during backward.

## How it works

### Injection

The prefix vectors are prepended to the cached LLM adapter output (T5-compatible embedding space, dim=1024). To keep the total sequence length at 512 (preserving the static token count for `torch.compile`), trailing zero-padding positions are trimmed:

```
Before:  [tok₁ tok₂ … tokₙ  0  0  0  … 0  0  0  0]   ← 512 tokens
                              ▲ zero-padding (attention sinks)

After:   [p₁ p₂ … p₁₆  tok₁ tok₂ … tokₙ  0  0  … 0]  ← still 512 tokens
          ▲ learned prefix        trailing padding trimmed by 16 ▲
```

The pretrained DiT treats zero-padded positions as attention sinks (they contribute `exp(0) = 1` to the softmax denominator). Displacing 16 of ~430 padding positions has negligible impact on this mechanism.

### Training loop

1. Cache text encoder outputs and latents to disk (as usual).
2. Load DiT — all weights frozen, no LoRA attached.
3. Each step:
   - Load cached embeddings and latents from disk.
   - Prepend prefix vectors to `crossattn_emb`.
   - Forward through frozen DiT → compute diffusion loss.
   - Backward → gradients flow through cross-attention back to prefix vectors only.
   - Update 16,384 parameters with AdamW8bit.

### What it learns

The prefix vectors occupy positions that every DiT block's cross-attention attends to. They effectively learn a "style prompt" in continuous embedding space — discovering quality signals, aesthetic biases, or stylistic directions that text tokens alone can't express.

Unlike LoRA which modifies internal model representations, prefix tuning works entirely in the input embedding space. This makes it:

- **Composable** — stack multiple prefix weights at inference by concatenating their vectors.
- **Interpretable** — the learned vectors live in the same space as text embeddings.
- **Non-destructive** — the base model is completely untouched.

## Quick start

### Training

Prefix is a variant of the postfix family. Two ways to run it:

```bash
# Clean per-variant path (recommended for most users):
make lora-gui GUI_PRESETS=prefix

# Toggle-block path: edit configs/methods/postfix.toml to activate the prefix
# block, then run:
make postfix
```

### Inference

```bash
python inference.py \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --prefix_weight output/anima_prefix.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024
```

Or `make test-prefix` to run against the most recent `output/anima_prefix*.safetensors`.

## Config reference

`configs/gui-methods/prefix.toml` (and the prefix toggle block in `configs/methods/postfix.toml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 16 | Number of prefix vectors |
| `learning_rate` | 1e-3 | Higher than LoRA (1e-3 vs 2e-5) — fewer params need stronger signal |
| `max_train_epochs` | 1 | Converges fast with so few parameters |
| `loss_type` | pseudo_huber | Smooth L1-like, robust to outliers |
| `pseudo_huber_c` | 0.3 | Controls L1/MSE interpolation |
| `caption_shuffle_variants` | 2 | Caption diversity per image |
| `blocks_to_swap` | 0 | No CPU offloading needed |
| `cache_llm_adapter_outputs` | true | Required — prefix operates on cached adapter outputs |
| `trim_crossattn_kv` | false | Deprecated; the trim path was FA4-only and FA4 has been removed. See `docs/optimizations/fa4.md`. |

## Prefix vs postfix vs LoRA

| | Prefix | Postfix | LoRA |
|---|---|---|---|
| **Injection point** | Before text tokens | After text tokens | Inside every DiT block |
| **Params** (default) | 16K | 16K | ~300K+ |
| **Peak VRAM** | ~12 GB | ~12 GB | 6–15 GB (config dependent) |
| **Speed** | ~1 step/s | ~1 step/s | 0.3–1.3 step/s |
| **Learning capacity** | Embedding-space only | Embedding-space only | Full model adaptation |
| **Output size** | ~32 KB | ~32 KB | ~1.2 MB |
| **Mergeable** | No | No | Yes |
| **Use case** | Style/quality transfer | Style/quality transfer | Subject/concept learning |

### When to use prefix over postfix

Both achieve similar results. The key difference is positional:

- **Prefix** vectors appear at the start of the sequence — they get attended to first and set the "context" for all subsequent tokens. Better for global style/quality signals.
- **Postfix** vectors appear right after the last real text token — they extend the caption in continuous space. Better for additive detail (e.g., "and also make it look like X").

In practice, prefix tends to converge slightly faster for style transfer tasks.

## Combining with LoRA

Prefix tuning and LoRA are complementary. A common workflow:

1. Train prefix for global style/quality (1 epoch, ~2 minutes).
2. Train LoRA for subject fidelity (64 epochs).
3. Use both at inference:

```bash
python inference.py \
    --lora_weight output/anima_lora.safetensors \
    --prefix_weight output/anima_prefix.safetensors \
    --prompt "your prompt" \
    ...
```
