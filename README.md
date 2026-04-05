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

## LoRA Variants

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Separates magnitude and direction in weight updates for improved learning efficiency at lower ranks. ([arXiv 2402.09353](https://arxiv.org/abs/2402.09353))

```toml
# network_args
use_dora = true
```

During inference/merge, the magnitude vector is exported as `dora_scale` for ComfyUI compatibility.

### OrthoLoRA

SVD-based orthogonal weight parameterization with zero-init guarantee and orthogonality regularization.

```toml
# network_args
use_ortho = true
sig_type = "last"           # "principal", "last", or "middle"
ortho_reg_weight = 0.01     # orthogonality penalty weight
```

Note: OrthoLoRA is not compatible with DoRA. Linear layers only (no Conv2d).

### T-LoRA (Timestep-Dependent Rank Masking)

Dynamically adjusts effective LoRA rank based on the denoising timestep. Early (high-noise) steps use full rank; later steps use reduced rank. Compatible with both LoRA and DoRA.

```toml
# network_args
use_timestep_mask = true
min_rank = 1                # minimum rank to preserve
alpha_rank_scale = 1.0      # power-law schedule exponent
```

The rank schedule follows:

```
r = ((max_t - t) / max_t) ^ alpha_rank_scale * (max_rank - min_rank) + min_rank
```

## FP32 Accumulation

Computes the LoRA forward pass (down → up projection) in fp32 for improved numerical precision, then casts back to bf16. Negligible overhead; recommended for training stability.

```toml
lora_fp32_accumulation = true
```

## Cross-Attention KV Trim

Eliminates wasted compute on zero-padding tokens in cross-attention. The pretrained model pads text encoder outputs to 512 tokens, but typical captions only use 30–80 — the rest are zeros that act as attention sinks (contributing `exp(0) = 1` to the softmax denominator and zero to the numerator).

**How it works:** KV is trimmed to a bucketed length (`KV_BUCKETS = [64, 128, 256, 512]`) before projection, and FA4's returned LSE (log-sum-exp) is used to apply a post-hoc sigmoid correction that exactly restores the attention-sink contribution:

```
out_corrected = out_trimmed * sigmoid(lse - log(N_pad))
```

This is mathematically identical to full-padding attention — not an approximation. The bucketed trim lengths (only 4 possible shapes) keep `torch.compile` stable with no recompilation after warmup.

Requires Flash Attention 4 (`attn_mode = "flash4"`). Other backends fall back to full 512-length KV automatically.

## Caption Shuffle Variants

Generates multiple shuffled caption permutations per image per epoch, cached as separate text encoder outputs. Increases caption diversity without disk overhead.

```toml
caption_shuffle_variants = 8    # number of variants per image
shuffle_caption = true
cache_text_encoder_outputs = true
```

The smart shuffle algorithm preserves `@artist` tags and section delimiters (`On the ...`, `In the ...`) while shuffling tags within each section. During training, one variant is randomly selected per batch item.

## Masked Loss (SAM / MIT)

Exclude regions (e.g., text bubbles) from the training loss using spatial masks.

### Generating masks

**SAM3** (Segment Anything Model):

```bash
python scripts/generate_masks.py \
    --config configs/sam_mask.yaml \
    --image-dir ../image_dataset \
    --mask-dir ../image_dataset/masks \
    --device cuda
```

**MIT** (Manga-Image-Translator / ComicTextDetector):

```bash
python scripts/generate_masks_mit.py \
    --image-dir ../image_dataset \
    --mask-dir ../image_dataset/masks \
    --device cuda \
    --detect-size 1024 \
    --text-threshold 0.5 \
    --dilate 5
```

Both produce grayscale PNGs: 255 = train, 0 = exclude.

### Using masks in training

```toml
# training config
masked_loss = true

# dataset config — point to mask directory
[[datasets.subsets]]
image_dir = '../image_dataset'
mask_dir = '../image_dataset/masks'
```

Masks are interpolated to match the latent spatial dimensions and applied element-wise to the loss.

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

### Key inference flags

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_weight` | — | LoRA weight path(s), space-separated for multiple |
| `--lora_multiplier` | 1.0 | LoRA strength multiplier(s) |
| `--infer_steps` | 50 | Denoising steps |
| `--guidance_scale` | 3.5 | CFG scale |
| `--flow_shift` | 5.0 | Flow-matching schedule shift |
| `--sampler` | euler | euler (deterministic ODE) or er_sde (stochastic) |
| `--from_file` | — | Batch prompts from text file |
| `--interactive` | false | Interactive prompt mode |
| `--fp8` | false | FP8 quantization for DiT |
| `--compile` | false | torch.compile speedup |

### P-GRAFT inference

Loads LoRA as dynamic hooks instead of a static merge, allowing mid-denoising cutoff:

```bash
python inference.py ... \
    --pgraft \
    --lora_cutoff_step 37    # LoRA active for steps 0–36, disabled 37+
```

### Prompt file format

```
a girl standing in a field --w 1024 --h 1024 --s 50 --g 3.5
another prompt --seed 42 --flow_shift 4.0
```

## LoRA Format Conversion

Convert between anima and ComfyUI key formats:

```bash
python scripts/convert_lora_to_comfy.py input.safetensors output.safetensors          # anima → ComfyUI
python scripts/convert_lora_to_comfy.py --reverse input.safetensors output.safetensors  # ComfyUI → anima
```

## Dataset Configuration

```toml
[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 3              # preserve first N tokens from shuffling

[[datasets]]
resolution = 1024
batch_size = 4
enable_bucket = true         # dynamic aspect-ratio bucketing
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64
validation_split = 0.05
validation_seed = 42

  [[datasets.subsets]]
  image_dir = '../image_dataset'
  num_repeats = 1
```

Each image needs a corresponding `.txt` caption sidecar file in the same directory.

## Project Structure

```
anima_lora/
├── train.py                    # AnimaTrainer — main training loop
├── inference.py                # Standalone image generation
├── configs/                    # TOML training/dataset configs
├── networks/
│   ├── lora_anima.py           # Network creation, module targeting, T-LoRA logic
│   ├── lora_modules.py         # LoRA, DoRA, OrthoLoRA module implementations
│   └── postfix_anima.py        # Continuous postfix tuning for LLM adapter
├── library/
│   ├── anima_models.py         # Anima DiT architecture
│   ├── anima_utils.py          # Model loading/saving
│   ├── anima_train_utils.py    # Caption shuffle, loss weighting, EMA, validation
│   ├── strategy_anima.py       # Tokenization/encoding strategy (Qwen3 + T5)
│   ├── train_util.py           # Re-exporting facade for training utilities
│   ├── custom_train_functions.py  # Masked loss application
│   ├── inference_utils.py      # Flow-matching samplers (Euler, ER-SDE)
│   ├── attention.py            # Attention backends (flash, xformers, torch)
│   ├── config_util.py          # TOML parsing (Voluptuous)
│   ├── qwen_image_autoencoder_kl.py  # QwenImageVAE (WanVAE)
│   ├── datasets/               # Dataset classes, bucketing, image utils
│   └── training/               # Optimizer/scheduler/checkpoint utilities
└── scripts/
    ├── generate_masks.py       # SAM3 text bubble masking
    ├── generate_masks_mit.py   # MIT/ComicTextDetector masking
    ├── merge_masks.py          # Combine multiple masks
    └── convert_lora_to_comfy.py  # LoRA format conversion
```
