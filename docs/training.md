# Training Reference

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

Removed — the trim path only ran under `attn_mode = "flash4"`, which we evaluated and removed. See [fa4.md](fa4.md) for the postmortem. Training now always runs full 512-length cross-attention KV; the zero-padded positions act as attention sinks and cost negligible compute on FA2.

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
