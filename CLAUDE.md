# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Includes a GRAFT human-in-the-loop fine-tuning system that iteratively trains LoRA, generates candidates, and retrains on user-curated survivors.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.11+)
# Model weights go in models/ (not tracked):
#   models/diffusion_models/anima-preview2.safetensors
#   models/text_encoders/qwen_3_06b_base.safetensors
#   models/vae/qwen_image_vae.safetensors
# Training images go in image_dataset/ with .txt caption sidecars
```

## Commands

```bash
# Training (run from anima_lora/)
make lora                  # Standard LoRA
make dora                  # DoRA (use_dora=true)
make tlora                 # T-LoRA: OrthoLoRA + timestep masking
make tdora                 # DoRA + timestep masking

# Inference (test with most recent output)
make test

# GRAFT loop (human-in-the-loop iterative training)
make step                  # Train → generate candidates → await curation
# Delete bad images from graft/candidates/iter_NNN/, then:
make step                  # Ingest survivors → retrain → new candidates

# Deploy
make sync                  # Copy output/*.safetensors to ComfyUI loras dir

# Linting
ruff check . --fix && ruff format .
```

## Key entry points

| File | Purpose |
|------|---------|
| `train.py` | `AnimaTrainer` class — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `graft_step.py` | GRAFT orchestrator: holdout → train → generate → await review |

## Config flow

Training is config-driven. TOML configs specify model paths, hyperparams, and dataset layout:
- `configs/training_config.toml` — default training config
- `configs/dataset_config.toml` — dataset buckets, subsets, caption settings
- `graft/graft_config.toml` — GRAFT-specific params (epochs_per_step, candidates_per_prompt, pgraft settings)

All paths in configs are relative to `anima_lora/` (e.g., `models/...`, `output/`).

## Architecture

- **Modular `library/`**: `train_util.py` is a re-exporting facade; actual code lives in `library/datasets/` (dataset classes, buckets, image utils) and `library/training/` (optimizer, scheduler, checkpoint logic)
- **Strategy pattern** for model-specific tokenization/encoding (`library/strategy_anima.py`, `strategy_base.py`)
- **Network modules** are pluggable via `network_module` config key (`networks/lora_anima.py`, `lora_modules.py`, `postfix_anima.py`)
- LoRA variants: standard LoRA, DoRA, OrthoLoRA — all in `networks/lora_modules.py`
- T-LoRA adds timestep-dependent masking with `use_timestep_mask=true` and `min_rank` network args
- Memory optimization: gradient checkpointing, latent/text-encoder caching to disk, VAE chunking
- All training uses Accelerate with bf16 mixed precision and flash attention (`attn_mode = "flash"`)

## GRAFT / P-GRAFT

The GRAFT loop (`graft_step.py`) implements rejection-sampling-based fine-tuning:
1. Holds out a subset of training images, trains LoRA on the rest + accumulated survivors
2. Generates candidates using the trained LoRA (with P-GRAFT: LoRA disabled for last 25% of denoising)
3. User curates by deleting bad candidates; survivors join the training set next iteration

See `graft-guideline.md` for detailed curation guidance.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).
