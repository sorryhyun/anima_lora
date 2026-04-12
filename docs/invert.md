# Embedding Inversion

Finds the optimal text embedding (crossattn_emb) for a target image by optimizing in the post-T5, pre-DiT embedding space. The frozen DiT acts as a fixed decoder — only the embedding is updated via gradient descent on the flow-matching loss.

This reveals "how the DiT interprets the image" in embedding space, producing a `.safetensors` file that can be used as a conditioning input for inference or as an analysis tool for understanding model behavior.

## Quick start

```bash
# Preprocess images first (caches latents + text encoder outputs)
make preprocess

# Run inversion on 10 random images
make invert
```

Or with a single image:

```bash
python scripts/invert_embedding.py \
    --image path/to/image.png \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --output_dir inversions
```

## How it works

1. **Load target** — either encode a raw image via VAE, or load cached latents from `post_image_dataset/`
2. **Initialize embedding** — from cached text encoder output, a text prompt, a saved embedding, or zeros
3. **Optimize** — for each step, sample random noise levels (sigmas), run the frozen DiT forward, compute MSE between predicted and target noise, and backpropagate through the DiT to update only the embedding
4. **Save** — the best embedding (lowest loss) is saved as a `.safetensors` file with metadata

### Optimization details

Each step samples `timesteps_per_step × grad_accum` random timesteps. The embedding is maintained in float32 for optimizer precision and cast to bfloat16 for the DiT forward pass. Gradient norm is clipped to 1.0.

## Two input modes

### Single image (`--image`)

Encodes the image via VAE on the fly. Requires `--vae`. The VAE is loaded, used, and freed before the DiT loads.

```bash
python scripts/invert_embedding.py --image photo.png \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --init_prompt "a photo of a cat" --text_encoder models/text_encoders/qwen_3_06b_base.safetensors
```

### Batch from preprocessed directory (`--image_dir`)

Uses cached latents (`.npz`) and optionally cached text encoder outputs (`_anima_te.safetensors`) from `make preprocess`. No VAE needed at runtime. Skips images that already have an output file.

```bash
python scripts/invert_embedding.py --image_dir post_image_dataset \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --num_images 10 --shuffle
```

## VRAM modes

| `--blocks_to_swap` | Behavior | VRAM |
|---|---|---|
| `0` (default) | Full model on GPU + `torch.compile` | Lowest (~12.5 GB) |
| `N > 0` | Swap N blocks to CPU, no compile | Higher (more swap = less VRAM but slower) |
| `-1` | Gradient checkpointing + `torch.compile` | Medium |

`torch.compile` provides significant memory savings through operator fusion, often outweighing block swap for small swap counts. Block swap is useful when the model doesn't fit on GPU at all.

Override in the Makefile: `make invert INVERT_SWAP=12` or `make invert INVERT_SWAP=-1`.

## Embedding initialization

Priority order (first match wins):

| Flag | Source |
|------|--------|
| `--init_embedding path.safetensors` | Load from a previously saved embedding |
| `--init_prompt "text"` + `--text_encoder` | Encode prompt through text encoder + LLM adapter |
| (default) `--init_from_cache` | Use cached `crossattn_emb` from `_anima_te.safetensors` (batch mode) |
| `--init_zeros` | Start from zero embedding |

## Verification

Add `--verify` to generate an image from the inverted embedding after optimization (requires `--vae`):

```bash
python scripts/invert_embedding.py --image photo.png \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --verify --verify_steps 50 --verify_seed 42
```

## Block gradient logging

`--log_block_grads` logs per-block gradient norms during optimization, saved as `_block_grads.json` alongside the loss CSV. This shows which DiT blocks are most sensitive to the embedding — useful for understanding which blocks drive cross-attention behavior.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dit` | (required) | DiT checkpoint path |
| `--vae` | — | VAE path (required for `--image` mode and `--verify`) |
| `--text_encoder` | — | Text encoder path (for `--init_prompt`) |
| `--attn_mode` | flash | Attention backend |
| `--image` | — | Single target image path |
| `--image_dir` | — | Preprocessed dataset directory |
| `--num_images` | all | Process N images from `--image_dir` |
| `--shuffle` | off | Shuffle image order |
| `--steps` | 100 | Optimization steps per image |
| `--lr` | 0.01 | Learning rate |
| `--lr_schedule` | cosine | `cosine` or `constant` |
| `--timesteps_per_step` | 1 | Timesteps sampled per step (batched) |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--sigma_sampling` | uniform | `uniform` or `sigmoid` |
| `--blocks_to_swap` | 0 | Blocks to swap to CPU (0 = compile, -1 = grad ckpt) |
| `--verify` | off | Generate verification image after optimization |
| `--log_block_grads` | off | Log per-block gradient norms |
| `--output_dir` | inversions | Output directory |

## Output structure

```
inversions/
├── results/
│   ├── image1_inverted.safetensors   # optimized embedding
│   ├── image1_verify.png             # verification image (if --verify)
│   └── ...
└── logs/
    ├── image1.csv                    # per-step loss, lr, grad_norm
    ├── image1_block_grads.json       # per-block gradient norms (if --log_block_grads)
    └── ...
```
