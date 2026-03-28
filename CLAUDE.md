# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Core LoRA training and inference engine for the Anima diffusion model. Parent directory handles orchestration (Makefile, GRAFT loop); this directory contains the ML pipeline.

## Commands

```bash
# Training (usually via parent Makefile, but can run directly)
accelerate launch --mixed_precision bf16 train.py --config_file configs/training_config.toml

# Override network args on the command line
accelerate launch --mixed_precision bf16 train.py --config_file configs/training_config.toml \
    --network_args use_ortho=true use_timestep_mask=true min_rank=1

# Inference
python inference.py --dit ../models/diffusion_models/anima-preview2.safetensors \
    --text_encoder ../models/text_encoders/qwen_3_06b_base.safetensors \
    --vae ../models/vae/qwen_image_vae.safetensors \
    --lora_weight ../output/anima_lora.safetensors \
    --prompt "your prompt" --image_size 1024 1024

# LoRA format conversion
python scripts/convert_lora_to_comfy.py <src> <dst>            # anima → ComfyUI
python scripts/convert_lora_to_comfy.py --reverse <src> <dst>  # ComfyUI → anima

# Linting
ruff check . --fix && ruff format .
```

## Architecture

- **train.py** — `AnimaTrainer` class: dataset setup, training loop, validation, checkpoint saving. Uses HF Accelerate.
- **inference.py** — Standalone generation. Supports `--from_file` for batch prompts, `--pgraft` + `--lora_cutoff_step` for P-GRAFT inference.
- **networks/** — Pluggable LoRA implementations selected via `network_module` config key:
  - `lora_anima.py` — Network creation, target module selection, T-LoRA timestep masking logic
  - `lora_modules.py` — Module-level implementations: LoRA, DoRA (`use_dora`), OrthoLoRA (`use_ortho`)
  - `postfix_anima.py` — Continuous postfix tuning network
- **library/** — Core utilities:
  - `train_util.py` — Re-exporting facade: arg parsing, metadata, hashing, accelerator setup, loss functions. Imports from `datasets/` and `training/` sub-packages.
  - `datasets/` — Dataset classes extracted from train_util:
    - `buckets.py` — `BucketManager`, `make_bucket_resolutions`, bucket resolution logic
    - `subsets.py` — `ImageInfo`, `BaseSubset`, `DreamBoothSubset`, `AugHelper`
    - `image_utils.py` — Image loading, caching, transforms, `glob_images`
    - `base.py` — `BaseDataset`, `DreamBoothDataset`, `DatasetGroup`, `MinimalDataset`, `collator_class`, `LossRecorder`
  - `training/` — Training utilities extracted from train_util:
    - `optimizers.py` — `get_optimizer`, optimizer factory for AdamW/Lion/DAdapt/Prodigy/etc.
    - `schedulers.py` — `get_scheduler_fix`, LR scheduler factory
    - `checkpoints.py` — Checkpoint naming, save/remove logic for epochs and steps
  - `anima_models.py` — Anima DiT architecture with Unsloth-style gradient checkpointing
  - `anima_utils.py` — Model loading/saving (DiT, Qwen3 text encoder, VAE)
  - `anima_train_utils.py` — Caption shuffle, loss weighting, validation sampling
  - `strategy_anima.py` / `strategy_base.py` — Strategy pattern for tokenization, text encoding, latent caching
  - `config_util.py` — TOML config parsing and validation (Voluptuous schemas)
  - `qwen_image_autoencoder_kl.py` — QwenImageVAE (WanVAE)
  - `inference_utils.py` — Flow-matching samplers (Euler, ER-SDE, etc.)
  - `attention.py` — Attention implementations (flash, xformers, torch SDPA)
- **configs/** — TOML configs and tokenizer configs (Qwen3, T5)

## Key Conventions

- Config-driven: all training params come from TOML files, with CLI overrides
- Safetensors format for all weights
- All paths in configs are relative to `anima_lora/` (model weights at `../models/...`)
- Memory optimization: `cache_latents_to_disk`, `cache_text_encoder_outputs`, `gradient_checkpointing`, `vae_chunk_size`
- `attn_mode = "flash"` for flash attention; falls back to torch SDPA
- `caption_shuffle_variants` controls augmented caption permutations per image per epoch
