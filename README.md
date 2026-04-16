# anima_lora

[한국어](README.ko.md)

LoRA training and inference engine for the Anima diffusion model (DiT-based, flow-matching). Supports standard LoRA, OrthoLoRA, and T-LoRA with timestep-dependent rank masking.

## Highlights

**15.2 GB peak VRAM · 1.3 s/step** on a single consumer GPU — achieved by co-designing the data pipeline, attention, and compiler stack:

| Optimization | What it does |
|---|---|
| **Constant-token bucketing** | All bucket resolutions are chosen so that `(H/16)×(W/16) ≈ 4096` patches. Every batch element is then zero-padded to exactly 4096 tokens, giving `torch.compile` a single static shape to trace — no recompilation across aspect ratios. |
| **Max-padded text encoder** | Text encoder outputs are padded to `max_length` (512) and zero-filled. The pretrained DiT treats these zero keys as learned **attention sinks** in cross-attention softmax, so removing padding produces black images. Keeping it preserves model behavior *and* gives the compiler another fixed dimension. |
| **Flash Attention 2** | Uses `flash_attn` 2.x for fixed-length and variable-length attention, with automatic fallback to SDPA. FA4 was evaluated and removed — see [docs/optimizations/fa4.md](docs/optimizations/fa4.md). |
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
| FA2 + compile - grad ckpt (fast, rank 32) | 15.6 GB | 6:20 | 2:59 | 0.09 | 0.212 |

## Setup

```bash
# 1. Install dependencies (Python 3.13)
uv sync

# 2. Authenticate with Hugging Face (needed for model downloads)
hf auth login

# 3. Download model weights (DiT, text encoder, VAE)
make download-models

# 4. Place training images in image_dataset/ with .txt caption sidecars

# 5. launch gui if needed
make gui

# Or, start via cli from preprocessing images (VAE-compatible resizing & validation)
make preprocess
make lora
```

Optional: install `flash-attn` for flash attention support.

### Model weights

Downloaded automatically by `make download-models` from [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) into `models/`:

| File | Path |
|------|------|
| Anima DiT | `models/diffusion_models/anima-preview3-base.safetensors` |
| Qwen3 0.6B text encoder | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

## Training

Training is config-driven via a three-layer chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. Method settings win over preset settings on overlap. Run with HF Accelerate:

```bash
accelerate launch --mixed_precision bf16 train.py --method lora --preset default
```

Override any config value from the CLI:

```bash
accelerate launch --mixed_precision bf16 train.py --method tlora --preset low_vram \
    --network_dim 32 --max_train_epochs 64 --learning_rate 2e-5
```

### Methods (`configs/methods/`)

| Method | Description |
|--------|-------------|
| `lora` | Standard LoRA (rank 16) |
| `tlora` | T-LoRA with OrthoLoRA + timestep masking |
| `hydralora` | HydraLoRA multi-head routing |
| `postfix` / `postfix_exp` / `postfix_func` | Postfix tuning variants |
| `prefix` | Prefix tuning |
| `graft` | Defaults used by the GRAFT rejection-sampling loop |

### Presets (`configs/presets.toml`)

| Preset | Description |
|--------|-------------|
| `default` | Linux daily driver / Windows 16GB (`blocks_to_swap=8`) |
| `fast_16gb` | No block swap + `layer_start=2` for ~16GB cards |
| `low_vram` | Gradient checkpointing + unsloth offload (also serves as Windows 8GB) |
| `graft` | GRAFT-specific swap budget |

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 16–32 | LoRA rank |
| `network_alpha` | = dim | LoRA scaling alpha |
| `learning_rate` | 2e-5 | Base learning rate |
| `optimizer_type` | AdamW8bit | AdamW8bit, Lion, DAdapt, Prodigy |
| `max_train_epochs` | 4–64 | Training epochs |
| `mixed_precision` | bf16 | bf16, fp16, or no |
| `attn_mode` | flash | flash (FA2), torch, xformers, flex, sageattn |
| `gradient_checkpointing` | true | Memory-efficient backprop |
| `cache_latents_to_disk` | true | Cache VAE latents to disk |
| `cache_text_encoder_outputs` | true | Cache text encoder outputs |

## Inference

```bash
python inference.py \
    --dit ../models/diffusion_models/anima-preview3-base.safetensors \
    --text_encoder ../models/text_encoders/qwen_3_06b_base.safetensors \
    --vae ../models/vae/qwen_image_vae.safetensors \
    --lora_weight ../output/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 50 \
    --guidance_scale 3.5 \
    --save_path ../output/images
```

## Embedding Inversion

Optimize a text embedding to match a target image by backpropagating through the frozen DiT. Reveals how the model interprets an image in embedding space.

```bash
make invert                    # batch inversion on preprocessed dataset
make invert INVERT_SWAP=12     # use block swap for low-VRAM GPUs
```

See [docs/methods/invert.md](docs/methods/invert.md) for details on initialization, VRAM modes, and block gradient analysis.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/guidelines/training.md](docs/guidelines/training.md) | LoRA variants (OrthoLoRA, T-LoRA), caption shuffle, masked loss, dataset config |
| [docs/optimizations/fa4.md](docs/optimizations/fa4.md) | Why FA4 / flash-attention-sm120 and cross-attention KV trim were removed |
| [docs/methods/prefix-tuning.md](docs/methods/prefix-tuning.md) | Prefix tuning — 12 GB VRAM, ~1 step/s, how it works, config reference |
| [docs/guidelines/inference.md](docs/guidelines/inference.md) | Inference flags, P-GRAFT inference, prompt file format, LoRA format conversion |
| [docs/methods/invert.md](docs/methods/invert.md) | Embedding inversion — optimization flags, VRAM modes, block gradient logging |
