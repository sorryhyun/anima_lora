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
# Only four method files exist: lora.toml, postfix.toml, apex.toml, graft.toml.
# Variants are toggle blocks inside them — uncomment the target block to switch:
#   lora.toml    — classic LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT (default
#                  stacks LoRA + OrthoLoRA + T-LoRA + ReFT together)
#   postfix.toml — postfix / postfix_exp / postfix_func / postfix_sigma / prefix
make lora                   # LoRA family (methods/lora.toml + presets.toml[default])
python tasks.py lora        # Same, works on Windows too
make lora PRESET=low_vram   # Override preset: methods/lora.toml + presets.toml[low_vram]
make lora-fast              # Shortcut: methods/lora.toml + presets.toml[fast_16gb]
make lora-low-vram          # Shortcut: methods/lora.toml + presets.toml[low_vram]
make lora-half              # Shortcut: methods/lora.toml + presets.toml[half] (sample_ratio=0.5)
make postfix                # Postfix/prefix family (methods/postfix.toml)
make apex                   # APEX self-adversarial 1-NFE distillation (methods/apex.toml)

# GUI-friendly per-variant path (configs/gui-methods/<variant>.toml — clean,
# self-contained, no toggle blocks). Intended for basic users who don't want
# to hand-edit methods/lora.toml's comment-toggle system.
make lora-gui GUI_PRESETS=tlora                     # gui-methods/tlora.toml + preset default
make lora-gui GUI_PRESETS=hydralora PRESET=low_vram # override preset as usual
python tasks.py lora-gui hydralora_sigma            # Windows; variant can also be 1st positional arg

# Modulation guidance distillation
make distill-mod           # Train pooled_text_proj MLP (text → AdaLN modulation)

# Embedding inversion
make invert                # Optimize text embedding for target images
make test-invert           # Verify inversion quality

# Inference (test with most recent output)
make test
make test-mod              # Test with modulation guidance (pooled_text_proj)
make test-apex             # APEX 4-NFE euler inference
make test-hydra            # HydraLoRA router-live (anima_hydra*_moe.safetensors)
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

# Merge LoRA into DiT (standalone ComfyUI-compatible checkpoint)
make merge ADAPTER_DIR=output                    # bake latest bakeable LoRA in dir
make merge ADAPTER_DIR=output MULTIPLIER=0.8     # scale strength
python scripts/merge_to_dit.py --adapter path/to/lora.safetensors --allow-partial
# Supports: LoRA / OrthoLoRA / DoRA / T-LoRA. Refuses ReFT / Hydra moe / postfix
# / prefix by default (they can't be folded into Linear weights); --allow-partial
# drops them and bakes only the LoRA portion.

# Batch
make comfy-batch           # Run ComfyUI batch workflow

# Debugging + tests
make print-config METHOD=lora PRESET=default   # Dump merged config chain (base→preset→method→CLI)
make test-unit                                  # pytest on tests/ (smoke, config, loss/network registries)

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

Deep-dives in `docs/methods/`: `apex.md`, `hydra-lora.md`, `invert.md`, `mod-guidance.md`, `postfix-sigma.md`, `prefix-tuning.md`, `psoft-integrated-ortholora.md`, `reft.md`, `spectrum.md`, `timestep_mask.md`.

## Config flow

Training is config-driven via a three-layer chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. Method settings win over preset settings on overlap, so a method can force its own hardware requirements (e.g. postfix forces `blocks_to_swap=0`).

Layout:
- `configs/base.toml` — shared infrastructure (model paths, optimizer, compile flags, etc.) AND the default dataset blueprint (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`). The dataset sections are consumed by `BlueprintGenerator` and skipped by the flat method+preset merge chain (see `_DATASET_CONFIG_SECTIONS` in `library/train_util.py`). Override with `--dataset_config <path>` when you need a different blueprint (e.g. GRAFT uses `graft/dataset_config.toml`).
- `configs/presets.toml` — all hardware profiles in one file as TOML sections: `[default]`, `[fast_16gb]`, `[low_vram]` (also serves as Windows 8GB), `[graft]`, `[half]` (experiment preset — sets `sample_ratio=0.5` for every subset via the global `--sample_ratio` override). Holds `blocks_to_swap`, `gradient_checkpointing`, `unsloth_offload_checkpointing`, etc.
- `configs/methods/` — one file per algorithm family. Holds rank, method flags (`use_hydra`, `add_reft`, …), and the method's opinionated learning rate / epochs / output_name. Four files only:
  - `lora.toml` — LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT. Variants are toggle blocks; default stacks classic LoRA + OrthoLoRA + T-LoRA + ReFT.
  - `postfix.toml` — postfix / postfix_exp / postfix_func / postfix_sigma / prefix. Toggle blocks.
  - `apex.toml` — APEX self-adversarial distillation (arXiv:2604.12322). Warm-starts from a prior LoRA via `network_weights` + `dim_from_weights`.
  - `graft.toml` — GRAFT training runs invoked by `scripts/graft_step.py`.
- `configs/gui-methods/` — GUI-friendly parallel tree. One self-contained TOML per **variant** instead of per family (`lora`, `ortholora`, `tlora`, `reft`, `tlora_ortho_reft`, `hydralora`, `hydralora_sigma`, `postfix`, `postfix_exp`, `postfix_func`, `postfix_sigma`, `prefix`, plus copies of `apex` and `graft`). No toggle blocks — what you see is what runs. Selected via `train.py --methods_subdir gui-methods` (wrapped by `make lora-gui GUI_PRESETS=<variant>` / `python tasks.py lora-gui <variant>`). Intended for basic users and as the eventual source of truth for the GUI's variant picker.
- `graft/graft_config.toml` — GRAFT-specific params (epochs_per_step, candidates_per_prompt, pgraft settings)

`library.train_util.load_method_preset(method, preset, methods_subdir="methods")` is the reusable merge helper (used by `train.py` and `scripts/graft_step.py`). Pass `methods_subdir="gui-methods"` to resolve against the clean per-variant tree instead of the toggle-block method files. All paths in configs are relative to `anima_lora/` (e.g., `models/...`, `output/`).

## Architecture

- **Modular `library/`**: `train_util.py` is a re-exporting facade; actual code lives in domain subpackages:
  - `library/anima/` — anima-specific code: `models.py` (DiT class), `training.py` (training helpers, CLI args), `weights.py` (model/tokenizer loading + save), `strategy.py` (tokenization/encoding strategies), `configs/` (bundled Qwen3/T5 tokenizer configs).
  - `library/datasets/` — dataset classes, buckets, image utils.
  - `library/training/` — optimizer, scheduler, checkpoint, loss/sampler/metric registries (absorbs former `custom_train_functions`).
  - `library/inference/` — generation, sampling, output.
  - `library/models/` — ancillary model defs: `qwen_vae.py` (VAE), `sai_spec.py` (metadata spec).
  - `library/config/` — `schema.py` (validation), `loader.py` (TOML merge chain).
  - `library/io/` — `cache.py` (disk cache helpers), `safetensors.py`.
  - `library/runtime/` — `device.py`, `offloading.py`, `noise.py` (flow-matching sampling).
  - `library/log.py` — logging setup + `fire_in_thread`.
- **Strategy pattern** for model-specific tokenization/encoding (`library/anima/strategy.py`, `library/strategy_base.py`)
- **Network modules** are pluggable via `network_module` config key:
  - `networks/lora_anima.py` — LoRA network creation, module targeting, timestep masking orchestration
  - `networks/lora_modules.py` — LoRA, OrthoLoRA module implementations
  - `networks/postfix_anima.py` — Continuous postfix tuning: learns N vectors appended to adapter cross-attention (modes: hidden, embedding, cfg, dual)
- **Attention dispatch** (`networks/attention.py`): Unified `attention()` routing to torch SDPA, xformers, flash-attn v2/v3, sageattn, or flex attention. Layout varies by backend (BHLD vs BLHD). FA4 (flash-attention-sm120) was evaluated and is currently disabled — see `docs/optimizations/fa4.md`.

### LoRA variants

All in `networks/lora_modules.py`. Stack freely via toggle flags in `configs/methods/lora.toml` — the default is LoRA + OrthoLoRA + T-LoRA + ReFT together.
- **LoRA** — Classic low-rank: `y = x + (x @ down @ up) * scale * multiplier`
- **OrthoLoRA** — SVD-based orthogonal parameterization with orthogonality regularization (linear layers only). Saved as plain LoRA via thin SVD on ΔW at save time. See `docs/methods/psoft-integrated-ortholora.md`.
- **T-LoRA** — Timestep-dependent rank masking: effective rank varies with denoising step via power-law schedule. See `docs/methods/timestep_mask.md`.
- **HydraLoRA** — MoE-style multi-head routing: shared `lora_down` + per-expert `lora_up_i` heads, layer-local router on the adapted Linear's input. Requires `cache_llm_adapter_outputs=true`. Produces a `*_moe.safetensors` sibling for router-live inference. See `docs/methods/hydra-lora.md`.
- **ReFT** — Block-level residual-stream intervention (LoReFT, Wu et al. NeurIPS 2024). One `ReFTModule` per selected DiT block wraps the block's `forward` and adds `R^T·(ΔW·h + b)·scale` to the output; orthogonality regularized on `R`. Additive side-channel, composes with any LoRA variant, lives in the same `.safetensors`. Vanilla ComfyUI can't load ReFT (weight-patcher silently drops `reft_*` keys) — use the `AnimaAdapterLoader` custom node. See `docs/methods/reft.md`.

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

## APEX (1-NFE distillation)

Self-adversarial condition-shift distillation — turns the pretrained velocity-field DiT into a 1–4 NFE generator without a discriminator or external teacher (`configs/methods/apex.toml`, `docs/methods/apex.md`). The "adversarial" signal comes from querying the same network under a learned shifted text condition (`ConditionShift`, `c_fake = A·c + b`). Training does **3 DiT forwards per step** (real + fake@real_xt stop-grad + fake@fake_xt), so `blocks_to_swap = 0` is method-forced — block swapping would crash on the second forward with a `FakeTensor` device mismatch. Warm-start from a prior LoRA checkpoint is effectively mandatory (`network_weights` + `dim_from_weights=true`); cold-start catastrophically regressed in Phase 0 testing. Inference is `make test-apex` (4 euler steps, `guidance_scale=1.0`).

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

`custom_nodes/comfyui-hydralora/` — **Anima Adapter Loader** node for ComfyUI (unified LoRA / Hydra / ReFT + prefix/postfix). One node with two independently-toggled sections; auto-detects by key sniff and applies each component with its own strength. Code is split across `adapter.py` (LoRA/Hydra/ReFT), `postfix.py` (prefix/postfix/cond), and `nodes.py` (the loader); `__init__.py` only re-exports the node mappings.
- Plain LoRA / HydraLoRA → `ModelPatcher.add_patches` (Hydra experts baked down with uniform weighting since the trained layer-local router can't run under weight-patching).
- ReFT → per-block `forward_hook` installed via `ModelPatcher.add_object_patch` on `diffusion_model.blocks.<idx>._forward_hooks`, replaying `h + R^T·(ΔW·h + b)·scale·strength`. Hooking (not `forward` override) is load-bearing — overriding `forward` strands block weights on CPU under ComfyUI's cast-weights path.
- Prefix / postfix / cond → `ModelPatcher.add_object_patch` on `diffusion_model.forward`, splicing learned vectors into the T5-compatible crossattn embedding *after* the LLM adapter + pad-to-512 step. Positive-batch rows only via `cond_or_uncond` from `transformer_options` (CFG-safe).

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).
