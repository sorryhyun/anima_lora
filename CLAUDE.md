# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Includes a GRAFT human-in-the-loop fine-tuning system that iteratively trains LoRA, generates candidates, and retrains on user-curated survivors.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.13)
hf auth login      # Authenticate for model downloads
make download-models       # Download DiT, text encoder, VAE from HuggingFace
# Training images go in image_dataset/ with .txt caption sidecars
make preprocess            # VAE-compatible resizing & validation
```

## Commands

Both `make` (Unix) and `python tasks.py` (cross-platform) are supported. The examples below show both forms.

```bash
# Training (run from anima_lora/)
make lora                  # Standard LoRA (configs/training_config_plain.toml)
python tasks.py lora       # Same, works on Windows too
make lora-low-vram         # Low-VRAM LoRA (configs/training_config_low_vram.toml)
make dora                  # DoRA (configs/training_config_dora.toml + use_dora=true)
make tlora                 # T-LoRA: OrthoLoRA + timestep masking (configs/training_config.toml)
make tdora                 # DoRA + timestep masking (configs/training_config_doratimestep.toml)
make hydralora             # HydraLoRA: MoE multi-head routing (configs/training_config_hydralora.toml)

# Inference (test with most recent output)
make test
make test-spectrum         # Spectrum-accelerated inference (~3.75x speedup)

# GUI (PySide6 — config editing, GRAFT curation, dataset browsing)
make gui
python tasks.py gui        # Windows

# GRAFT loop (human-in-the-loop iterative training)
make step                  # Train -> generate candidates -> await curation
# Delete bad images from graft/candidates/iter_NNN/, then:
make step                  # Ingest survivors -> retrain -> new candidates

# Masking (for masked loss training)
make mask                  # Generate SAM3 + MIT masks, then merge
make mask-sam              # SAM3 only
make mask-mit              # MIT/ComicTextDetector only
make mask-clean            # Remove all generated masks

# Deploy & batch
make sync                  # Copy output/*.safetensors to ComfyUI loras dir
make comfy-batch           # Run ComfyUI batch workflow

# Linting
ruff check . --fix && ruff format .
```

All training invocations use `accelerate launch --mixed_precision bf16`. Override any config value from CLI: `--network_dim 32 --max_train_epochs 64`.

On Windows, use `python tasks.py <command>` instead of `make <command>`. Extra args are forwarded: `python tasks.py lora --network_dim 32`.

## Key entry points

| File | Purpose |
|------|---------|
| `train.py` | `AnimaTrainer` class — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `library/spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting) |
| `graft_step.py` | GRAFT orchestrator: holdout -> train -> generate -> await review |
| `gui.py` | PySide6 GUI: config editing with presets, GRAFT curation, dataset browser, training monitor |
| `gui_i18n.py` | i18n layer for GUI (Korean/English) |
| `tasks.py` | Cross-platform task runner (Windows-compatible Makefile alternative) |

## Config flow

Training is config-driven. TOML configs specify model paths, hyperparams, and dataset layout:
- `configs/base.toml` — base/shared config values
- `configs/training_config_plain.toml` — standard LoRA config (used by `make lora`)
- `configs/training_config.toml` — T-LoRA config (used by `make tlora`)
- `configs/training_config_dora.toml` — DoRA config
- `configs/training_config_doratimestep.toml` — DoRA + timestep masking
- `configs/training_config_hydralora.toml` — HydraLoRA multi-head routing (used by `make hydralora`)
- `configs/training_config_low_vram.toml` — low-VRAM LoRA config
- `configs/training_config_win8gb.toml` / `win16gb.toml` — Windows VRAM presets (GUI presets)
- `configs/training_config_fa4_8gb.toml` / `fa4_16gb.toml` — Flash Attention 4 VRAM presets (GUI presets)
- `configs/dataset_config.toml` — dataset buckets, subsets, caption settings
- `graft/graft_config.toml` — GRAFT-specific params (epochs_per_step, candidates_per_prompt, pgraft settings)

All paths in configs are relative to `anima_lora/` (e.g., `models/...`, `output/`).

## Architecture

- **Modular `library/`**: `train_util.py` is a re-exporting facade; actual code lives in `library/datasets/` (dataset classes, buckets, image utils) and `library/training/` (optimizer, scheduler, checkpoint logic)
- **Strategy pattern** for model-specific tokenization/encoding (`library/strategy_anima.py`, `strategy_base.py`)
- **Network modules** are pluggable via `network_module` config key:
  - `networks/lora_anima.py` — LoRA network creation, module targeting, timestep masking orchestration
  - `networks/lora_modules.py` — LoRA, DoRA, OrthoLoRA module implementations
  - `networks/postfix_anima.py` — Continuous postfix tuning: learns N vectors appended to adapter cross-attention (modes: hidden, embedding, cfg, dual)
- **Attention dispatch** (`library/attention.py`): Unified `attention()` routing to torch SDPA, xformers, flash-attn v2/v3, flash-attn v4, sageattn, or flex attention. Layout varies by backend (BHLD vs BLHD).

### LoRA variants

All in `networks/lora_modules.py`:
- **LoRA** — Classic low-rank: `y = x + (x @ down @ up) * scale * multiplier`
- **DoRA** — Weight-decomposed: separate magnitude (`dora_scale`) and direction learning
- **OrthoLoRA** — SVD-based orthogonal parameterization with orthogonality regularization (linear layers only, incompatible with DoRA)
- **T-LoRA** — Timestep-dependent rank masking: effective rank varies with denoising step via power-law schedule. Compatible with both LoRA and DoRA.
- **HydraLoRA** — MoE-style multi-head routing: shared `lora_down` + per-expert `lora_up_i` heads. Router on max-pooled `crossattn_emb` selects expert contributions per sample. Requires `cache_llm_adapter_outputs=true`. Compatible with T-LoRA. See `docs/hydra-lora.md`.

### Training flow (train.py)

1. Load text encoder -> cache text encoder outputs to disk -> unload text encoder
2. Load VAE -> cache latents to disk -> unload VAE
3. Load DiT lazily (after caching frees VRAM)
4. Create LoRA/Postfix network, apply to target modules via monkey-patching
5. Training loop: noise sampling -> DiT forward -> loss -> backward -> manual all_reduce -> optimizer step
6. Optional validation: multi-timestep loss + sample generation

## Critical invariants

### Text encoder padding

The pretrained model expects max-padded text encoder outputs — zero-padded positions act as attention sinks in cross-attention softmax. Trimming to actual text length produces black images. Both training and inference must pad to `max_length` and must NOT mask out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` files after any tokenizer/padding changes.

### Constant-token bucketing

All bucket resolutions ensure `(H/16)*(W/16) ~ 4096` patches. Batch elements are zero-padded to exactly 4096 tokens, giving `torch.compile` a single static shape — no recompilation across aspect ratios.

### Flash4 LSE correction

When cross-attention KV is trimmed (zero-padding removed for efficiency), the softmax denominator must be corrected. `library/attention.py` applies a sigmoid-based LSE correction using `crossattn_full_len` to account for removed zero-key contributions.

### DDP gradient sync

Built-in DDP grad sync is disabled for LoRA-only training efficiency. Instead, `all_reduce_network()` is called manually after backward to sync only LoRA gradients.

### Lazy model loading

DiT is loaded AFTER text encoder/VAE caching and unloading to avoid OOM. The sequence is: text encoder -> cache -> free -> VAE -> cache -> free -> load DiT.

## Spectrum inference acceleration

Training-free speedup via Chebyshev polynomial feature forecasting (Han et al., CVPR 2026). `--spectrum` flag on `inference.py` enables it. On cached steps, all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` run. A `register_forward_pre_hook` on `final_layer` captures block outputs without monkey-patching the model. The adaptive window schedule (controlled by `--spectrum_window_size` and `--spectrum_flex_window`) concentrates actual forwards on early high-noise steps and increasingly predicts later refinement steps. See `Spectrum/` for the reference repo and `library/spectrum.py` for the Anima integration.

## GRAFT / P-GRAFT

The GRAFT loop (`graft_step.py`) implements rejection-sampling-based fine-tuning:
1. Holds out a subset of training images, trains LoRA on the rest + accumulated survivors
2. Generates candidates using the trained LoRA (with P-GRAFT: LoRA disabled for last 25% of denoising)
3. User curates by deleting bad candidates; survivors join the training set next iteration

See `docs/graft-guideline.md` for detailed curation guidance.

## Scripts

Utility scripts in `scripts/`:
- `post_images.py` — VAE-compatible image resizing & latent/embedding caching (used by `make preprocess`)
- `generate_masks.py` — SAM3-based text bubble mask generation
- `generate_masks_mit.py` — MIT/ComicTextDetector mask generation (manga-specific)
- `merge_masks.py` — Combine SAM3 + MIT masks into final mask set
- `convert_lora_to_comfy.py` — Convert LoRA key names between anima and ComfyUI formats
- `comfy_batch.py` — Run ComfyUI batch workflow from `workflows/` directory

## Custom nodes

`custom_nodes/comfyui-spectrum/` — ComfyUI drop-in KSampler replacement for Spectrum inference acceleration. Published to ComfyUI registry via `.github/workflows/publish_action.yml`.

`custom_nodes/comfyui-hydralora/` — HydraLoRA loader nodes for ComfyUI. Two nodes: **HydraLoRA Loader (Manual)** with per-expert weight sliders, and **HydraLoRA Loader (Auto Router)** that computes expert weights from text conditioning via the learned router. Loads `*_hydra.safetensors` files.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).
