# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Supports several adapter families (LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT / postfix-prefix / APEX / IP-Adapter / EasyControl) selectable via method config + hardware preset.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.13)
hf auth login              # Authenticate for model downloads
make download-models       # Download DiT, text encoder, VAE from HuggingFace
# Training images go in image_dataset/ with .txt caption sidecars
make preprocess            # Resize → post_image_dataset/resized/, cache → post_image_dataset/lora/
```

## Commands

Both `make` (Unix) and `python tasks.py` (cross-platform) are supported. The examples below show both forms.

```bash
# Training (run from anima_lora/)
# Each training invocation selects a method + hardware preset. Method settings win
# over preset settings on overlap (e.g. postfix forces blocks_to_swap=0).
# Method files in configs/methods/: lora.toml, postfix.toml, apex.toml,
# ip_adapter.toml, easycontrol.toml. Variants are toggle blocks
# inside them — uncomment the target block to switch:
#   lora.toml         — classic LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT
#                       (default stacks LoRA + OrthoLoRA + T-LoRA + ReFT)
#   postfix.toml      — postfix / postfix_exp / postfix_func / postfix_sigma / prefix
#   ip_adapter.toml   — decoupled image cross-attention (PE-Core resampler)
#   easycontrol.toml  — extended self-attn image conditioning (per-block cond LoRA)
make lora                   # LoRA family (methods/lora.toml + presets.toml[default])
python tasks.py lora        # Same, works on Windows too
make lora PRESET=low_vram   # Override preset: methods/lora.toml + presets.toml[low_vram]
make lora-fast              # Shortcut: methods/lora.toml + presets.toml[fast_16gb]
make lora-low-vram          # Shortcut: methods/lora.toml + presets.toml[low_vram]
make lora-half              # Shortcut: methods/lora.toml + presets.toml[half] (sample_ratio=0.5)
make postfix                # Postfix/prefix family (methods/postfix.toml)
make apex                   # APEX self-adversarial 1-NFE distillation (methods/apex.toml)
make ip-adapter             # IP-Adapter image cross-attention (methods/ip_adapter.toml)
                            # Source: ip-adapter-dataset/   Cache: post_image_dataset/ip-adapter/
make ip-adapter-preprocess  # Resize + VAE + text + PE caches into post_image_dataset/ip-adapter/
make ip-adapter-cache       # PE-Core features only → post_image_dataset/ip-adapter/{stem}_anima_pe.safetensors
make easycontrol            # EasyControl image conditioning (methods/easycontrol.toml)
                            # Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/
make easycontrol-preprocess # Resize + VAE + text caches into post_image_dataset/easycontrol/

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
make invert-ref            # Learn K prefix-slot vectors for a reference image
make test-invert           # Verify inversion quality
make bench-inversion       # Benchmark inversion stability (bench/inversion/)

# Inference (test with most recent output)
make test
make test-mod              # Test with modulation guidance (pooled_text_proj)
make test-apex             # APEX 4-NFE euler inference
make test-hydra            # HydraLoRA router-live (anima_hydra*_moe.safetensors)
make test-prefix           # Test with prefix tuning
make test-postfix          # Test with postfix tuning
make test-postfix-exp      # Test with postfix tuning (exp variant)
make test-postfix-func     # Test with postfix tuning (func variant)
make test-ip REF_IMAGE=... # IP-Adapter inference (image-conditioned)
make test-easycontrol REF_IMAGE=...  # EasyControl inference (image-conditioned)
make test-ref              # Inference with a learned prefix-slot weight (--prefix_weight)
make test-merge            # Inference with a merged/baked DiT (no adapter loaded)
make test-spectrum         # Spectrum-accelerated inference (~3.75x speedup)

# GUI (PySide6 — config editing, IP-Adapter / EasyControl preprocess+train, dataset browsing)
make gui
python tasks.py gui        # Windows
make gui-shortcut          # Create "Anima LoRA GUI.lnk" on the Windows desktop (no console window)

# Masking (for masked loss training)
# Outputs under masks/{sam,mit,merged}/. Subsets auto-pick masks/merged/ when
# it exists, falling back to masks/sam/ then masks/mit/.
make mask                  # Generate SAM3 + MIT masks under masks/{sam,mit}/, then merge → masks/merged/
make mask-sam              # SAM3 only → masks/sam/
make mask-mit              # MIT/ComicTextDetector only → masks/mit/
make mask-clean            # Remove masks/

# Merge LoRA into DiT (standalone ComfyUI-compatible checkpoint)
make merge ADAPTER_DIR=output/ckpt                    # bake latest bakeable LoRA in dir
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8     # scale strength
python scripts/merge_to_dit.py --adapter path/to/lora.safetensors --allow-partial
# Supports: LoRA / OrthoLoRA / DoRA / T-LoRA. Refuses ReFT / Hydra moe / postfix
# / prefix by default (they can't be folded into Linear weights); --allow-partial
# drops them and bakes only the LoRA portion.

# Batch
make comfy-batch           # Run ComfyUI batch workflow

# Debugging + tests
make print-config METHOD=lora PRESET=default   # Dump merged config chain (base→preset→method→CLI)
make test-unit                                  # pytest on tests/ (smoke, config, loss/network registries)
make export-logs RUN=...                        # Export TensorBoard run to JSON (scripts/export_logs_json.py)

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
| `gui/` | PySide6 GUI package: config editing with presets, IP-Adapter / EasyControl preprocess+train tabs, dataset browser, training monitor |
| `tasks.py` | Cross-platform task runner (Windows-compatible Makefile alternative) |

Deep-dives in `docs/methods/`: `apex.md`, `easycontrol.md`, `hydra-lora.md`, `invert.md`, `ip-adapter.md`, `mod-guidance.md`, `postfix-sigma.md`, `prefix-tuning.md`, `psoft-integrated-ortholora.md`, `reft.md`, `spectrum.md`, `timestep_mask.md`.

## Config flow

Training is config-driven via a three-layer chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. Method settings win over preset settings on overlap, so a method can force its own hardware requirements (e.g. postfix forces `blocks_to_swap=0`).

Layout:
- `configs/base.toml` — shared infrastructure (model paths, optimizer, compile flags, etc.) AND the default LoRA dataset blueprint (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`). LoRA reads resized images from `post_image_dataset/resized/` with caches redirected to `post_image_dataset/lora/` via `cache_dir`. Captions live in `image_dataset/` (master) — TE caching reads `.txt` from there, training reads only the cached prompt embeddings. The dataset sections are consumed by `BlueprintGenerator` and skipped by the flat method+preset merge chain (see `_DATASET_CONFIG_SECTIONS` in `library/train_util.py`). Override with `--dataset_config <path>` when you need a different blueprint.
- `configs/presets.toml` — all hardware profiles in one file as TOML sections: `[default]`, `[fast_16gb]`, `[low_vram]` (also serves as Windows 8GB), `[half]` (experiment preset — sets `sample_ratio=0.5` for every subset via the global `--sample_ratio` override). Holds `blocks_to_swap`, `gradient_checkpointing`, `unsloth_offload_checkpointing`, etc.
- `configs/methods/` — one file per algorithm family. Holds rank, method flags (`use_hydra`, `add_reft`, …), and the method's opinionated learning rate / epochs / output_name. Five files:
  - `lora.toml` — LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT. Variants are toggle blocks; default stacks classic LoRA + OrthoLoRA + T-LoRA + ReFT.
  - `postfix.toml` — postfix / postfix_exp / postfix_func / postfix_sigma / prefix. Toggle blocks.
  - `apex.toml` — APEX self-adversarial distillation (arXiv:2604.12322). Warm-starts from a prior LoRA via `network_weights` + `dim_from_weights`.
  - `ip_adapter.toml` — IP-Adapter image cross-attention (DiT frozen; trains resampler + per-block `to_k_ip`/`to_v_ip`). Source: `ip-adapter-dataset/`. Caches: `post_image_dataset/ip-adapter/` (subset-level `cache_dir`). Defaults to PRE-CACHED PE features (`make ip-adapter-cache`).
  - `easycontrol.toml` — EasyControl image conditioning (DiT frozen; trains per-block cond LoRA on self-attn + FFN + scalar `b_cond` gate). Source: `easycontrol-dataset/`. Caches: `post_image_dataset/easycontrol/`. Reuses cached VAE latents — no new sidecar.
- `configs/gui-methods/` — GUI-friendly parallel tree. One self-contained TOML per **variant** instead of per family (`lora`, `lora-8gb`, `lora_longer`, `ortholora`, `tlora`, `reft`, `tlora_ortho_reft`, `hydralora`, `hydralora_sigma`, `postfix`, `postfix_exp`, `postfix_func`, `postfix_sigma`, `prefix`, `ip_adapter`, `easycontrol`, plus a copy of `apex`). No toggle blocks — what you see is what runs. Selected via `train.py --methods_subdir gui-methods` (wrapped by `make lora-gui GUI_PRESETS=<variant>` / `python tasks.py lora-gui <variant>`). Intended for basic users and as the eventual source of truth for the GUI's variant picker.

Subsets accept an optional `cache_dir` key — when set, all VAE / text-encoder / PE caches are written to (and read from) that directory using stem-mirrored filenames, instead of sitting next to the source image. IP-Adapter and EasyControl method configs use this to keep `ip-adapter-dataset/` and `easycontrol-dataset/` purely user-facing source dirs while caches live under `post_image_dataset/`.

`library.train_util.load_method_preset(method, preset, methods_subdir="methods")` is the reusable merge helper. Pass `methods_subdir="gui-methods"` to resolve against the clean per-variant tree instead of the toggle-block method files. All paths in configs are relative to `anima_lora/` (e.g., `models/...`, `output/ckpt/`). Runtime outputs are split by kind: trained checkpoints (+ `.snapshot.toml` + `_moe` siblings) in `output/ckpt/`, inference images in `output/tests/`, embedding-inversion results in `output/inversions/`, img2emb artifacts in `output/img2embs/`.

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
  - `networks/lora_anima/` — LoRA network creation, module targeting, timestep masking orchestration (split into `network.py`, `factory.py`, `loading.py`, `config.py`).
  - `networks/lora_modules/` — Per-variant module implementations: `lora.py`, `ortho.py`, `hydra.py`, `reft.py`, plus `base.py` and `custom_autograd.py`.
  - `networks/lora_save.py` / `lora_utils.py` — Save-time SVD distillation (OrthoLoRA → plain LoRA) and shared helpers.
  - `networks/methods/postfix.py` — Continuous postfix tuning: learns N vectors appended to adapter cross-attention (modes: hidden, embedding, cfg, dual).
  - `networks/methods/ip_adapter.py` — IP-Adapter: PE-Core-L14-336 vision encoder + Perceiver resampler + per-block `to_k_ip`/`to_v_ip`.
  - `networks/methods/easycontrol.py` — EasyControl: per-block cond LoRA on self-attn (q/k/v/o) + FFN + scalar `b_cond` logit-bias gate; two-stream block forward at training, KV-cache prefill at inference.
  - `networks/methods/apex.py` — APEX `ConditionShift` module (`c_fake = A·c + b`).
- **Attention dispatch** (`networks/attention_dispatch.py`): Unified `dispatch_attention()` routing to torch SDPA, xformers, flash-attn v2/v3, sageattn, or flex attention. Layout varies by backend (BHLD vs BLHD). FA4 (flash-attention-sm120) was evaluated and is currently disabled — see `docs/optimizations/fa4.md`.

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
5. Training loop: noise sampling -> DiT forward -> loss -> backward -> optimizer step
6. Optional validation: multi-timestep loss + sample generation

## Critical invariants

### Text encoder padding

The pretrained model expects max-padded text encoder outputs — zero-padded positions act as attention sinks in cross-attention softmax. Trimming to actual text length produces black images. Both training and inference must pad to `max_length` and must NOT mask out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` files after any tokenizer/padding changes.

### Constant-token bucketing

All bucket resolutions ensure `(H/16)*(W/16) ~ 4096` patches. Batch elements are zero-padded to exactly 4096 tokens, giving `torch.compile` a single static shape — no recompilation across aspect ratios.

### Flash4 LSE correction

When cross-attention KV is trimmed (zero-padding removed for efficiency), the softmax denominator must be corrected. `networks/attention_dispatch.py` applies a sigmoid-based LSE correction using `crossattn_full_len` to account for removed zero-key contributions.

### Lazy model loading

DiT is loaded AFTER text encoder/VAE caching and unloading to avoid OOM. The sequence is: text encoder -> cache -> free -> VAE -> cache -> free -> load DiT.

## Spectrum inference acceleration

Training-free speedup via Chebyshev polynomial feature forecasting (Han et al., CVPR 2026). `--spectrum` flag on `inference.py` enables it. On cached steps, all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` run. A `register_forward_pre_hook` on `final_layer` captures block outputs without monkey-patching the model. The adaptive window schedule (controlled by `--spectrum_window_size` and `--spectrum_flex_window`) concentrates actual forwards on early high-noise steps and increasingly predicts later refinement steps. See `networks/spectrum.py` for the Anima integration and `docs/methods/spectrum.md` for usage notes.

## P-GRAFT inference

P-GRAFT (`--pgraft` flag on `inference.py`) is a mid-denoise LoRA cutoff: it loads the LoRA as dynamic forward hooks rather than static merge so it can be disabled at a given step (typically the last ~25%) to let the base model handle late-step refinement. Independent of the GRAFT training loop, which has been deprecated and moved to `archive/graft/`.

## APEX (1-NFE distillation)

Self-adversarial condition-shift distillation — turns the pretrained velocity-field DiT into a 1–4 NFE generator without a discriminator or external teacher (`configs/methods/apex.toml`, `docs/methods/apex.md`). The "adversarial" signal comes from querying the same network under a learned shifted text condition (`ConditionShift`, `c_fake = A·c + b`). Training does **3 DiT forwards per step** (real + fake@real_xt stop-grad + fake@fake_xt), so `blocks_to_swap = 0` is method-forced — block swapping would crash on the second forward with a `FakeTensor` device mismatch. Warm-start from a prior LoRA checkpoint is effectively mandatory (`network_weights` + `dim_from_weights=true`); cold-start catastrophically regressed in Phase 0 testing. Inference is `make test-apex` (4 euler steps, `guidance_scale=1.0`).

## Modulation guidance

Text-conditioned AdaLN modulation via a learned `pooled_text_proj` MLP (Starodubcev et al., ICLR 2026). Distilled with `make distill-mod`: teacher uses real cross-attention, student uses zeroed cross-attention but receives pooled text through modulation. At inference, steers AdaLN coefficients toward quality-positive directions. See `docs/methods/mod-guidance.md`.

## IP-Adapter

Decoupled image cross-attention (Ye et al. 2023). DiT is frozen; trains only the Perceiver resampler and per-block parallel `to_k_ip`/`to_v_ip` projections (~150M params at default `K=16`, 28 blocks). Reference image → frozen vision tower (PE-Core-L14-336 by default) → resampler → K compact IP tokens → per-block KV → patched cross-attention adds `scale * SDPA(text_q, ip_k, ip_v)` to the existing text cross-attention. Source images live in `ip-adapter-dataset/`; preprocessed caches (latents, text-emb, PE features) go to `post_image_dataset/ip-adapter/` via the subset-level `cache_dir` knob. Defaults to PRE-CACHED PE features (`{stem}_anima_pe.safetensors` from `make ip-adapter-cache`) so training never loads the vision encoder. CFG dropout (`image_drop_p`) zeros image conditioning so inference can do image-CFG independently of text-CFG. See `docs/methods/ip-adapter.md`.

## EasyControl

Extended self-attention image conditioning. DiT is frozen; trains per-block cond LoRA on self-attn (q/k/v/o) + FFN (layer1/layer2) plus a per-block scalar logit-bias `b_cond` (init `-10`) that gates cond-position softmax mass. Reference is VAE-encoded and patch-embedded by the DiT's frozen `x_embedder` into condition tokens that flow through every block alongside the target stream; target self-attention attends to a key set extended with the cond stream's keys/values. Training uses a **two-stream block forward** (target + cond in one scope, no deferred-backward dance); inference prefills a per-block `(K_c, V_c)` cache once at setup and reuses it across every denoising step and every CFG branch (cond is deterministic — `cond_temb = t_embedder(0)`). Source images live in `easycontrol-dataset/`; caches go to `post_image_dataset/easycontrol/` via subset `cache_dir`. Reuses cached VAE latents — no new sidecar. See `docs/methods/easycontrol.md`.

## Embedding inversion

Optimizes text embeddings (post-T5, pre-DiT space) to minimize flow-matching loss for a target image through the frozen DiT. Reveals how the model interprets images in embedding space. `make invert` runs batch inversion from `post_image_dataset/`, `make test-invert` verifies results. See `docs/methods/invert.md`.

## Preprocessing

Data preparation scripts in `preprocess/`:
- `resize_images.py` — VAE-compatible image resizing (used by `make preprocess-resize`). Reads `image_dataset/`, writes resized PNGs to `post_image_dataset/resized/`. Drops images below `--min_pixels` (default 0.5MP). `--no_copy_captions` skips the `.txt` copy so captions stay only in `image_dataset/`.
- `cache_latents.py` — Cache VAE latents (used by `make preprocess-vae`). Reads `post_image_dataset/resized/`, writes `{stem}_{WxH}_anima.npz` into `post_image_dataset/lora/` via `--cache_dir`.
- `cache_text_embeddings.py` — Cache text encoder outputs (used by `make preprocess-te`). Reads `image_dataset/` (where `.txt` lives) and writes `{stem}_anima_te.safetensors` into `post_image_dataset/lora/` via `--cache_dir`. Mirrors `resize_images.py`'s `--min_pixels` filter so caches don't accumulate for images that would be dropped at resize.
- `cache_pe_encoder.py` — Cache PE-Core vision encoder features (`{stem}_anima_pe.safetensors`); used by `make ip-adapter-cache` for IP-Adapter training. Reads `ip-adapter-dataset/`, writes into `post_image_dataset/ip-adapter/`.
- `generate_masks.py` — SAM3-based text bubble mask generation
- `generate_masks_mit.py` — MIT/ComicTextDetector mask generation (manga-specific)
- `merge_masks.py` — Combine SAM3 + MIT masks into final mask set

## Scripts

Utility scripts in `scripts/`:
- `distill_modulation.py` — Train pooled_text_proj MLP for modulation guidance (used by `make distill-mod`)
- `convert_lora_to_comfy.py` — Convert LoRA key names between anima and ComfyUI formats
- `comfy_batch.py` — Run ComfyUI batch workflow from `workflows/` directory
- `merge_to_dit.py` — Bake a LoRA adapter into the base DiT (used by `make merge`)
- `bench_methods.py` — Benchmark inference across method configurations (writes to `bench/`)
- `export_logs_json.py` — Export TensorBoard run scalars to JSON/JSONL (used by `make export-logs`)

Archived utilities (legacy, no longer wired up): `archive/img2emb/` (resampler training + inference) and `archive/inversion/` (embedding/reference inversion). The shared resampler/encoder/bucket modules they used have been extracted into `library/vision/` for live IP-Adapter use.

## Custom nodes

Spectrum KSampler and mod guidance ComfyUI nodes live in a separate repo: https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler

`custom_nodes/comfyui-hydralora/` — **Anima Adapter Loader** node for ComfyUI (unified LoRA / Hydra / ReFT + prefix/postfix). Auto-detects components by key sniff and applies each via its correct path (`ModelPatcher.add_patches` for plain LoRA; per-Linear / per-block `forward_hook`s installed through `ModelPatcher.add_object_patch` for Hydra and ReFT; `diffusion_model.forward` object patch for prefix / postfix / cond). See `custom_nodes/comfyui-hydralora/README.md` for installation, hook mechanics, and changelog.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).
