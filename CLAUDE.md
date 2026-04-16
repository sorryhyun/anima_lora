# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Includes a GRAFT human-in-the-loop fine-tuning system that iteratively trains LoRA, generates candidates, and retrains on user-curated survivors.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.13)
hf auth login              # Authenticate for model downloads
make download-models       # Download DiT, text encoder, VAE from HuggingFace
# Training images go in image_dataset/ with .txt caption sidecars
make preprocess            # VAE-compatible resizing & validation
```

## Commands

Both `make` (Unix) and `python tasks.py` (cross-platform) are supported. The examples below show both forms.

```bash
# Training (run from anima_lora/)
# Each training invocation selects a method + hardware preset. Method settings win
# over preset settings on overlap (e.g. postfix forces blocks_to_swap=0).
make lora                   # Standard LoRA (methods/lora.toml + presets.toml[default])
python tasks.py lora        # Same, works on Windows too
make lora PRESET=low_vram   # Override preset: methods/lora.toml + presets.toml[low_vram]
make lora-fast              # Shortcut: methods/lora.toml + presets.toml[fast_16gb]
make lora-low-vram          # Shortcut: methods/lora.toml + presets.toml[low_vram]
make tlora                 # T-LoRA: OrthoLoRA + timestep masking (methods/tlora.toml)
make hydralora             # HydraLoRA: MoE multi-head routing (methods/hydralora.toml)
make postfix               # Postfix tuning (methods/postfix.toml)
make postfix-exp           # Postfix tuning, exp variant (methods/postfix_exp.toml)
make postfix-func          # Postfix tuning, func variant (methods/postfix_func.toml)
make prefix                # Prefix tuning (methods/prefix.toml)

# Modulation guidance distillation
make distill-mod           # Train pooled_text_proj MLP (text → AdaLN modulation)

# Embedding inversion
make invert                # Optimize text embedding for target images
make test-invert           # Verify inversion quality

# Inference (test with most recent output)
make test
make test-mod              # Test with modulation guidance (pooled_text_proj)
make test-prefix           # Test with prefix tuning
make test-postfix          # Test with postfix tuning
make test-postfix-exp      # Test with postfix tuning (exp variant)
make test-postfix-func     # Test with postfix tuning (func variant)
make test-spectrum         # Spectrum-accelerated inference (~3.75x speedup)

# GUI (PySide6 — config editing, GRAFT curation, dataset browsing)
make gui
python tasks.py gui        # Windows

# GRAFT loop (human-in-the-loop iterative training)
make graft-step            # Train -> generate candidates -> await curation
# Delete bad images from graft/candidates/iter_NNN/, then:
make graft-step            # Ingest survivors -> retrain -> new candidates
python tasks.py step       # Same, works on Windows

# Masking (for masked loss training)
make mask                  # Generate SAM3 + MIT masks, then merge
make mask-sam              # SAM3 only
make mask-mit              # MIT/ComicTextDetector only
make mask-clean            # Remove all generated masks

# Batch
make comfy-batch           # Run ComfyUI batch workflow

# Linting
ruff check . --fix && ruff format .
```

All training invocations use `accelerate launch --mixed_precision bf16` with `train.py --method <name> --preset <name>`. Override any config value from CLI: `--network_dim 32 --max_train_epochs 64`. Override preset with `PRESET=low_vram make lora` or `python tasks.py lora` plus `PRESET` env.

On Windows, use `python tasks.py <command>` instead of `make <command>`. Extra args are forwarded: `python tasks.py lora --network_dim 32`.

## Key entry points

| File | Purpose |
|------|---------|
| `train.py` | `AnimaTrainer` class — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `networks/spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting) |
| `scripts/graft_step.py` | GRAFT orchestrator: holdout -> train -> generate -> await review |
| `gui/` | PySide6 GUI package: config editing with presets, GRAFT curation, dataset browser, training monitor |
| `tasks.py` | Cross-platform task runner (Windows-compatible Makefile alternative) |

## Config flow

Training is config-driven via a three-layer chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. Method settings win over preset settings on overlap, so a method can force its own hardware requirements (e.g. postfix forces `blocks_to_swap=0`).

Layout:
- `configs/base.toml` — shared infrastructure (model paths, optimizer, compile flags, etc.)
- `configs/presets.toml` — all hardware profiles in one file as TOML sections: `[default]` (Linux daily driver + Windows 16GB, `blocks_to_swap=8`), `[fast_16gb]`, `[low_vram]` (also serves as Windows 8GB), `[graft]`. Holds `blocks_to_swap`, `gradient_checkpointing`, `unsloth_offload_checkpointing`, etc.
- `configs/methods/` — one file per algorithm. Holds rank, method flags (`use_hydra`, …), and the method's opinionated learning rate / epochs / output_name. Files: `lora`, `tlora`, `hydralora`, `postfix`, `postfix_exp`, `postfix_func`, `prefix`, `graft`.
- `configs/dataset_config.toml` — dataset buckets, subsets, caption settings
- `graft/graft_config.toml` — GRAFT-specific params (epochs_per_step, candidates_per_prompt, pgraft settings)

`library.train_util.load_method_preset(method, preset)` is the reusable merge helper (used by `train.py` and `scripts/graft_step.py`). All paths in configs are relative to `anima_lora/` (e.g., `models/...`, `output/`).

## Architecture

- **Modular `library/`**: `train_util.py` is a re-exporting facade; actual code lives in `library/datasets/` (dataset classes, buckets, image utils) and `library/training/` (optimizer, scheduler, checkpoint logic)
- **Strategy pattern** for model-specific tokenization/encoding (`library/strategy_anima.py`, `strategy_base.py`)
- **Network modules** are pluggable via `network_module` config key:
  - `networks/lora_anima.py` — LoRA network creation, module targeting, timestep masking orchestration
  - `networks/lora_modules.py` — LoRA, OrthoLoRA module implementations
  - `networks/postfix_anima.py` — Continuous postfix tuning: learns N vectors appended to adapter cross-attention (modes: hidden, embedding, cfg, dual)
- **Attention dispatch** (`networks/attention.py`): Unified `attention()` routing to torch SDPA, xformers, flash-attn v2/v3, sageattn, or flex attention. Layout varies by backend (BHLD vs BLHD). FA4 (flash-attention-sm120) was evaluated and is currently disabled — see `docs/optimizations/fa4.md`.

### LoRA variants

All in `networks/lora_modules.py`:
- **LoRA** — Classic low-rank: `y = x + (x @ down @ up) * scale * multiplier`
- **OrthoLoRA** — SVD-based orthogonal parameterization with orthogonality regularization (linear layers only)
- **T-LoRA** — Timestep-dependent rank masking: effective rank varies with denoising step via power-law schedule.
- **HydraLoRA** — MoE-style multi-head routing: shared `lora_down` + per-expert `lora_up_i` heads. Router on max-pooled `crossattn_emb` selects expert contributions per sample. Requires `cache_llm_adapter_outputs=true`. Compatible with T-LoRA. See `docs/methods/hydra-lora.md`.

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

When cross-attention KV is trimmed (zero-padding removed for efficiency), the softmax denominator must be corrected. `networks/attention.py` applies a sigmoid-based LSE correction using `crossattn_full_len` to account for removed zero-key contributions.

### DDP gradient sync

Built-in DDP grad sync is disabled for LoRA-only training efficiency. Instead, `all_reduce_network()` is called manually after backward to sync only LoRA gradients.

### Lazy model loading

DiT is loaded AFTER text encoder/VAE caching and unloading to avoid OOM. The sequence is: text encoder -> cache -> free -> VAE -> cache -> free -> load DiT.

## Spectrum inference acceleration

Training-free speedup via Chebyshev polynomial feature forecasting (Han et al., CVPR 2026). `--spectrum` flag on `inference.py` enables it. On cached steps, all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` run. A `register_forward_pre_hook` on `final_layer` captures block outputs without monkey-patching the model. The adaptive window schedule (controlled by `--spectrum_window_size` and `--spectrum_flex_window`) concentrates actual forwards on early high-noise steps and increasingly predicts later refinement steps. See `networks/spectrum.py` for the Anima integration and `docs/methods/spectrum.md` for usage notes.

## GRAFT / P-GRAFT

The GRAFT loop (`scripts/graft_step.py`) implements rejection-sampling-based fine-tuning:
1. Holds out a subset of training images, trains LoRA on the rest + accumulated survivors
2. Generates candidates using the trained LoRA (with P-GRAFT: LoRA disabled for last 25% of denoising)
3. User curates by deleting bad candidates; survivors join the training set next iteration

See `docs/guidelines/graft-guideline.md` for detailed curation guidance.

## Modulation guidance

Text-conditioned AdaLN modulation via a learned `pooled_text_proj` MLP (Starodubcev et al., ICLR 2026). Distilled with `make distill-mod`: teacher uses real cross-attention, student uses zeroed cross-attention but receives pooled text through modulation. At inference, steers AdaLN coefficients toward quality-positive directions. See `docs/methods/mod-guidance.md`.

## Embedding inversion

Optimizes text embeddings (post-T5, pre-DiT space) to minimize flow-matching loss for a target image through the frozen DiT. Reveals how the model interprets images in embedding space. `make invert` runs batch inversion from `post_image_dataset/`, `make test-invert` verifies results. See `docs/methods/invert.md`.

## Preprocessing

Data preparation scripts in `preprocess/`:
- `resize_images.py` — VAE-compatible image resizing (used by `make preprocess-resize`)
- `cache_latents.py` — Cache VAE latents to disk (used by `make preprocess-vae`)
- `cache_text_embeddings.py` — Cache text encoder outputs to disk (used by `make preprocess-te`)
- `generate_masks.py` — SAM3-based text bubble mask generation
- `generate_masks_mit.py` — MIT/ComicTextDetector mask generation (manga-specific)
- `merge_masks.py` — Combine SAM3 + MIT masks into final mask set

## Scripts

Utility scripts in `scripts/`:
- `distill_modulation.py` — Train pooled_text_proj MLP for modulation guidance (used by `make distill-mod`)
- `invert_embedding.py` — Optimize text embedding for target images (used by `make invert`)
- `interpret_inversion.py` — Verify/visualize embedding inversion results (used by `make test-invert`)
- `convert_lora_to_comfy.py` — Convert LoRA key names between anima and ComfyUI formats
- `comfy_batch.py` — Run ComfyUI batch workflow from `workflows/` directory

## Custom nodes

Spectrum KSampler and mod guidance ComfyUI nodes live in a separate repo: https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler

`custom_nodes/comfyui-hydralora/` — HydraLoRA loader node for ComfyUI: **HydraLoRA Loader (Manual)** with per-expert weight sliders. Loads `*_moe.safetensors` multi-head files (sibling of the baked-down `anima_hydra.safetensors`). The previous Auto Router node was removed when HydraLoRA moved to per-module layer-local routing — routing now reads live layer input, which can't be replicated by weight-add patching.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).
