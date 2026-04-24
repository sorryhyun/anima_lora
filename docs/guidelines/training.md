# Training Reference

## LoRA Variants

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Separates magnitude and direction in weight updates for improved learning efficiency at lower ranks ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353)).

```toml
# network_args
use_dora = true
```

During inference/merge, the magnitude vector is exported as `dora_scale` for ComfyUI compatibility. Not compatible with `use_ortho` or `use_hydra`.

### OrthoLoRA (Cayley + PSOFT-inspired)

Cayley-parameterized orthogonal rotation of frozen SVD bases with a zero-init guarantee. Orthogonality is structural — no regularization hyperparameter.

```toml
# network_args
use_ortho = true
```

Linear layers only (no Conv2d). See [`../methods/psoft-integrated-ortholora.md`](../methods/psoft-integrated-ortholora.md) for the full design.

### T-LoRA (Timestep-Dependent Rank Masking)

Dynamically adjusts effective LoRA rank based on the denoising timestep. Early (high-noise) steps use full rank; later steps use reduced rank. Composes with LoRA, DoRA, OrthoLoRA, HydraLoRA, and ReFT. See [`../methods/timestep_mask.md`](../methods/timestep_mask.md).

```toml
# network_args
use_timestep_mask = true
min_rank = 1                # minimum rank to preserve
alpha_rank_scale = 1.0      # power-law schedule exponent
```

The rank schedule follows:

```
r(t) = floor((1 - t)^alpha_rank_scale * (network_dim - min_rank)) + min_rank
```

### HydraLoRA and ReFT

See the dedicated docs for multi-head expert routing ([`../methods/hydra-lora.md`](../methods/hydra-lora.md)) and block-level residual-stream intervention ([`../methods/reft.md`](../methods/reft.md)). The default `configs/methods/lora.toml` stacks LoRA + OrthoLoRA + T-LoRA + ReFT; flip the individual toggles to test subsets.

## FP32 Accumulation

Unconditional. LoRA/Hydra/ReFT bottleneck matmuls run in fp32 regardless of autocast; stored parameters stay bf16. The previous `lora_fp32_accumulation` flag is deprecated and ignored.

## Cross-Attention KV Trim

Removed — the trim path only ran under `attn_mode = "flash4"`, which we evaluated and removed. See [`../optimizations/fa4.md`](../optimizations/fa4.md) for the postmortem. Training now always runs full 512-length cross-attention KV; the zero-padded positions act as attention sinks and cost negligible compute on FA2.

## Caption Shuffle Variants

Generates multiple shuffled caption permutations per image per epoch, cached as separate text encoder outputs. Increases caption diversity without disk overhead.

```toml
caption_shuffle_variants = 4    # number of variants per image
shuffle_caption = true
cache_text_encoder_outputs = true
```

The smart shuffle algorithm preserves `@artist` tags and section delimiters (`On the ...`, `In the ...`) while shuffling tags within each section. During training, one variant is randomly selected per batch item.

## Masked Loss (SAM / MIT)

Exclude regions (e.g. text bubbles) from the training loss using spatial masks.

### Generating masks

Easiest path — one make target:

```bash
make mask             # SAM3 + MIT; calls both then merges into masks/
make mask-sam         # SAM3 only
make mask-mit         # MIT / ComicTextDetector only
make mask-clean       # remove all generated masks
```

These operate on `post_image_dataset/` (the resized output of `make preprocess-resize`).

### Using masks in training

```toml
# training config
masked_loss = true

# dataset config — point to the merged mask directory
[[datasets.subsets]]
image_dir = 'post_image_dataset'
mask_dir = 'masks'
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
  image_dir = 'post_image_dataset'
  num_repeats = 1
```

Each image needs a corresponding `.txt` caption sidecar file in the same directory.
