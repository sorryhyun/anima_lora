# anima_lora

LoRA training and inference engine for the Anima diffusion model (DiT-based, flow-matching). Supports standard LoRA, DoRA, OrthoLoRA, and T-LoRA with timestep-dependent rank masking.

## Highlights

**15.2 GB peak VRAM · 1.3 s/step** on a single consumer GPU — achieved by co-designing the data pipeline, attention, and compiler stack:

| Optimization | What it does |
|---|---|
| **Constant-token bucketing** | All bucket resolutions are chosen so that `(H/16)×(W/16) ≈ 4096` patches. Every batch element is then zero-padded to exactly 4096 tokens, giving `torch.compile` a single static shape to trace — no recompilation across aspect ratios. |
| **Max-padded text encoder** | Text encoder outputs are padded to `max_length` (512) and zero-filled. The pretrained DiT treats these zero keys as learned **attention sinks** in cross-attention softmax, so removing padding produces black images. Keeping it preserves model behavior *and* gives the compiler another fixed dimension. |
| **Cross-attention KV trim** | Typical captions use 30–80 tokens out of 512 — ~85% is zero-padding. KV is trimmed to a bucketed length (64/128/256/512) and an LSE-based sigmoid correction restores the exact attention-sink contribution. **~4× less cross-attention compute** with no quality loss, compile-safe (only 4 possible shapes). |
| **Flash Attention 4** | Uses `flash_attn.cute` (Hopper-optimized FA4 kernels) for both fixed-length and variable-length attention, with automatic fallback to FA2/SDPA. Official FA4 lacks consumer Blackwell (SM120) support — we use a [fork](https://github.com/sorryhyun/flash-attention-sm120-fix) of [sisgrad's SM120 branch](https://github.com/sisgrad/flash-attention/tree/dz/sm120_tma_optimized) with minor bug fixes. |
| **Per-block `torch.compile`** | Each DiT block is compiled independently with the Inductor backend. Combined with static token counts this eliminates Dynamo guard recompilation entirely. |
| **Disk-cached latents & text embeddings** | VAE latents, text encoder outputs, and LLM adapter outputs are pre-computed and cached to disk — the VAE and text encoder never occupy training VRAM. |
| **Unsloth gradient checkpointing** | Activations are offloaded to CPU with non-blocking transfers during the forward pass and streamed back for the backward pass, trading PCIe bandwidth for VRAM. |

## Benchmarks

Tested on RTX 5060 Ti 16GB. LoRA rank=32, lr=5e-5, batch_size=2, epochs=2 (182 steps), seed=42.
Validation loss measured with fixed seed at timestep sigma = {0.05, 0.1, 0.2, 0.35}.
gradient_checkpointing=true, unsloth_offload_checkpointing=true, latent and text embeddings cached to disk.

| Configuration | Peak VRAM | Total Time | 2nd Epoch | Train Loss | Val Loss |
|---|---|---|---|---|---|
| FA2 (plain) | 7.0 GB | 14:51 | 7:26 | 0.092 | 0.212 |
| FA2 + compile (eager fallback) | 7.7 GB | 15:10 | 7:26 | 0.089 | 0.211 |
| FA2 + compile (static tokens) | 6.2 GB | 11:07 | 5:01 | 0.086 | 0.193 |
| FA2 + compile - grad ckpt | 15.2 GB | **7:07** | **3:30** | 0.088 | 0.206 |
| FA4 + compile (static tokens) | 6.3 GB | 11:05 | 5:15 | 0.092 | 0.187 |
| + fp32 accumulation | 6.4 GB | 10:57 | 5:15 | 0.089 | 0.196 |
| + DoRA + fp32 accumulation | 6.4 GB | 12:04 | 5:25 | 0.092 | 0.204 |
| + T-LoRA + fp32 accumulation | 6.9 GB | 12:57 | 5:44 | 0.093 | 0.210 |

Last 3 rows use FA4 + compile (static tokens) as baseline.

## Setup

```bash
# 1. Install dependencies (Python 3.13)
uv sync

# 2. Authenticate with Hugging Face (needed for model downloads)
huggingface-cli login

# 3. Download model weights (DiT, text encoder, VAE)
make download-models

# 4. Place training images in image_dataset/ with .txt caption sidecars

# 5. Preprocess images (VAE-compatible resizing & validation)
make preprocess
```

Optional: install `flash-attn` for flash attention support.

### Model weights

Downloaded automatically by `make download-models` from [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) into `models/`:

| File | Path |
|------|------|
| Anima DiT | `models/diffusion_models/anima-preview2.safetensors` |
| Qwen3 0.6B text encoder | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

## Training

All training is config-driven via TOML files. Run with HF Accelerate:

```bash
accelerate launch --mixed_precision bf16 train.py --config_file configs/training_config.toml
```

Override any config value from the CLI:

```bash
accelerate launch --mixed_precision bf16 train.py --config_file configs/training_config.toml \
    --network_dim 32 --max_train_epochs 64 --learning_rate 2e-5
```

### Provided configs

| Config | Description |
|--------|-------------|
| `configs/training_config.toml` | Standard LoRA (rank 32, 64 epochs) |
| `configs/training_config_dora.toml` | DoRA (rank 16) |
| `configs/training_config_doratimestep.toml` | DoRA + T-LoRA timestep masking |
| `configs/dataset_config.toml` | Dataset layout with dynamic bucketing |

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 16–32 | LoRA rank |
| `network_alpha` | = dim | LoRA scaling alpha |
| `learning_rate` | 2e-5 | Base learning rate |
| `optimizer_type` | AdamW8bit | AdamW8bit, Lion, DAdapt, Prodigy |
| `max_train_epochs` | 4–64 | Training epochs |
| `mixed_precision` | bf16 | bf16, fp16, or no |
| `attn_mode` | flash | flash, torch, xformers |
| `gradient_checkpointing` | true | Memory-efficient backprop |
| `cache_latents_to_disk` | true | Cache VAE latents to disk |
| `cache_text_encoder_outputs` | true | Cache text encoder outputs |

## Inference

```bash
python inference.py \
    --dit ../models/diffusion_models/anima-preview2.safetensors \
    --text_encoder ../models/text_encoders/qwen_3_06b_base.safetensors \
    --vae ../models/vae/qwen_image_vae.safetensors \
    --lora_weight ../output/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 50 \
    --guidance_scale 3.5 \
    --save_path ../output/images
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/training.md](docs/training.md) | LoRA variants (DoRA, OrthoLoRA, T-LoRA), KV trim, caption shuffle, masked loss, dataset config |
| [docs/inference.md](docs/inference.md) | Inference flags, P-GRAFT inference, prompt file format, LoRA format conversion |
