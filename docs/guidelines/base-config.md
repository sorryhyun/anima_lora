# `base.toml` Reference

`configs/base.toml` is the bottom layer of the config merge chain:

```
configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args
```

It holds **shared infrastructure** that rarely changes between experiments —
model paths, the dataset blueprint, optimizer/schedule defaults, the noise
schedule, caching, compile, and the memory knobs. Presets override hardware
profile values on top; method TOMLs override method-specific values; CLI args
win last. **Method settings beat preset settings on overlap**, so a frozen-DiT
method can force its own requirements.

You usually don't edit `base.toml` directly — override the one value you need
from the method TOML or the CLI (`--network_dim 32`, `PRESET=low_vram`). This
doc explains what each key *is* so you know which lever to reach for. Dump the
fully merged result for any combo with:

```bash
make print-config METHOD=lora PRESET=default
```

> **Two sub-files split off from base.toml** — both documented at the bottom:
> - `configs/preprocess.toml` — preprocess-only knobs (`source_image_dir`, …).
> - The `[general]` / `[[datasets]]` dataset blueprint — *in* base.toml but
>   **skipped by the flat merge** (it's consumed separately by the dataset
>   loader, not argparse).

---

## Model paths

| Key | Default | What it controls |
|---|---|---|
| `pretrained_model_name_or_path` | `models/diffusion_models/anima-base-v1.0.safetensors` | The DiT weights. Override with `$ANIMA_DIT`. |
| `qwen3` | `models/text_encoders/qwen_3_06b_base.safetensors` | Qwen3 text encoder. Override with `$ANIMA_TEXT_ENCODER`. |
| `vae` | `models/vae/qwen_image_vae.safetensors` | The Qwen-Image VAE. Override with `$ANIMA_VAE`. |

All paths are **repo-relative** and resolve under `anima_home()` (the repo
root, or `$ANIMA_HOME`), *not* the current working directory — so training and
the programmatic API work from any directory.

## Output & logging

| Key | Default | What it controls |
|---|---|---|
| `output_dir` | `output/ckpt` | Where checkpoints land (`<output_name>.safetensors` + `.snapshot.toml` sidecar). Method TOMLs usually set their own `output_name`. |
| `output_name` | `anima_lora` | Base filename for the adapter. Per-variant method files differentiate this (`anima_tlora_ortho`, `anima_hydra`, …). |
| `logging_dir` | `output/logs` | TensorBoard run directory (`make export-logs RUN=…`). |
| `log_with` | `tensorboard` | Logging backend. |
| `log_every_n_steps` | `2` | How often metrics are flushed. Each log is a sync point, so very small values add overhead. |

## Network selection

| Key | Default | What it controls |
|---|---|---|
| `network_module` | `networks.lora_anima` | Python import path of the adapter family. The LoRA family routes its variants through the three-axis surface (`use_moe_style` / `route_per_layer` / `router_source`) in the **method** TOML, not here. |
| `network_train_unet_only` | `true` | Train only the DiT-side adapter (the only supported mode — the text encoder is cached and frozen). Leave it on; `false` would expect a text-encoder LoRA the cached workflow can't provide. |

## Dataset paths & selection

These are top-level scalars (not inside `[[datasets]]`) so any preset/method
can override them, and the dataset blueprint interpolates them via
`{resized_image_dir}` / `{lora_cache_dir}`.

| Key | Default | What it controls |
|---|---|---|
| `resized_image_dir` | `post_image_dataset/resized` | Where `make preprocess` writes bucket-resized PNGs; what training reads images from. |
| `lora_cache_dir` | `post_image_dataset/lora` | Flat, stem-keyed cache dir for VAE/TE/PE sidecars. |
| `path_pattern` | `"*"` | `fnmatch` glob applied to each image's path **relative to its subset's `image_dir`**. `*` (or unset) = everything. OR-combine with `\|`: `char_a/*\|char_b/*`, or `*portrait*` for a substring. Applies to both training and validation enumeration. |
| `target_res` | `[1024]` | Multi-scale constant-token tiers (allowed edges `512 768 896 1024 1280 1536`). Each image is assigned to the tier that **resizes it the least**. **You MUST pass the same `--target_res …` at training time as at preprocess** — it builds the bucket table and sizes the compile cache. Omit a tier and its caches get snapped into a 1024 bucket and silently never loaded. See `library/datasets/buckets.py` and the bucketing invariant in `CLAUDE.md`. |

## Optimizer & schedule

| Key | Default | What it controls |
|---|---|---|
| `optimizer_type` | `AdamW` | Optimizer class. `DAdapt*` / `Prodigy` get auto-detected for `d*lr` logging. |
| `lr_scheduler` | `cosine` | LR schedule. |
| `lr_warmup_steps` | `0.05` | **Fraction OR absolute** — a `float < 1` is read as a *fraction* of total training steps (here: 5%); an `int ≥ 1` is taken literally. |

`learning_rate`, `network_dim`, `network_alpha`, `max_train_epochs`, etc. are
**not** in base.toml — they're set per-method (see `configs/methods/lora.toml`
and §8.3 of the [guidebook](guidebook.md)).

## Noise schedule (flow-matching)

| Key | Default | What it controls |
|---|---|---|
| `timestep_sampling` | `sigmoid` | How training σ is drawn each step: `uniform`, `sigmoid`, `shift`, `flux_shift`, … Each branch weights the noise-level distribution differently. |
| `sigmoid_bias` | `0.0` | Logit-space mean shift for the sigmoid family: `sigmoid(scale·randn + bias)`. `> 0` skews toward the high-noise (structure) regime; `0.0` is unbiased. |
| `discrete_flow_shift` | `1.0` | σ-space shift for `shift` / `flux_shift` sampling: `σ' = σ·s / (1 + (s−1)·σ)`. `1.0` = no shift. This is the **training-time** analogue of the inference `--flow_shift` — they are separate knobs. |

> Channel-scaling calibration is σ-grid-insensitive at `sigmoid_bias = 0`;
> changing the sampling weighting won't move it (see `CLAUDE.md` memory note).

## Caching

The pipeline caches VAE latents, text-encoder outputs, and PE features to disk
during `make preprocess`, then training reads only the caches.

| Key | Default | What it controls |
|---|---|---|
| `use_vae_cache` | `true` | Read VAE latents from disk instead of encoding live each step. |
| `use_text_cache` | `true` | Read cached text-encoder outputs (expands to `cache_text_encoder_outputs{,_to_disk}`). Requires `network_train_unet_only = true`. |
| `skip_cache_check` | `true` | **Trust** that on-disk caches are valid without opening every file to verify shape/keys. Fast path — assumes preprocessing was clean. Set `false` to re-validate integrity on load (slower) if you suspect stale caches. |
| `vae_chunk_size` | `64` | Spatial chunk size for VAE encode/decode (must be even). Lowers peak VRAM at a small speed cost. |
| `vae_disable_cache` | `true` | Disables the VAE's *internal activation cache* during encode/decode. `true` here is intentional — it's faster and uses less VRAM than the official default, at no quality cost for this pipeline. |

## Loss & validation

| Key | Default | What it controls |
|---|---|---|
| `masked_loss` | `true` | Zero the loss outside mask regions (e.g. exclude speech bubbles). Needs masks under `post_image_dataset/masks/` (run `make mask`); **missing masks are simply ignored**, so leaving this on is harmless. Turn off in a method TOML to force unmasked even when masks exist. |
| `use_cmmd` | `false` | Validation signal selector. **Validation is OFF in base.toml** (`validation_split_num = 0`). When you turn validation on (see blueprint below), `use_cmmd = true` uses paired CMMD² (PE-Core MMD), which tracks sample quality better than the legacy per-σ FM-MSE fallback on Anima. |

## Compile, attention, precision

| Key | Default | What it controls |
|---|---|---|
| `torch_compile` | `true` | Enable `torch.compile` via `compile_blocks()` — the blessed path (bit-exact, lowers memory). It flips on native-shape bucketing and keys the dynamo graph on token-count families derived from `target_res`. **Enable this first on OOM**, before gradient checkpointing. |
| `attn_mode` | `flash` | Attention backend for training: `flash` (FA2), `torch` (SDPA), `sageattn`, `flex`. Falls back to `torch` (SDPA) if unavailable. |
| `save_precision` | `bf16` | Dtype for saved adapter weights. Stored params stay bf16 even though LoRA/Hydra bottleneck matmuls always accumulate in fp32. |

## Memory & throughput knobs

| Key | Default | What it controls |
|---|---|---|
| `gradient_checkpointing` | `false` | Recompute activations to save VRAM. Reach for `torch_compile` / `blocks_to_swap` first. |
| `unsloth_offload_checkpointing` | `false` | Unsloth offload variant of grad checkpointing — **auto-enables `gradient_checkpointing`**; incompatible with `blocks_to_swap`. ⚠️ Its reentrant default can silently zero gradients on detached-input passes; the codebase forces `use_reentrant=False` where it matters. |
| `channel_scaling_alpha` | `0.5` | SmoothQuant-style per-channel input pre-scaling, baked into the adapter **at training-init time** (not an inference knob). `0.0` disables; `0.5` = sqrt balance; `1.0` = fully flatten channel dominance. Calibration is vendored at `networks/calibration/channel_stats.safetensors` (cond-stream sibling for EasyControl). Only affects variants with a *trainable* down-projection — exactly inert on frozen-basis ortho variants (`use_ortho` / OrthoHydra). See `docs/optimizations/channel_scaling.md`. |
| `dataloader_pin_memory` | `true` | Pin DataLoader tensors in host RAM for faster GPU transfer. |
| `persistent_data_loader_workers` | `true` | Keep DataLoader workers alive across epochs. |

`blocks_to_swap`, gradient/CPU offload-checkpointing, and `mixed_precision`
live in the **preset** (`configs/presets.toml`), not here — that's the
hardware-profile layer (`[default]` / `[fast_16gb]` / `[low_vram]` / `[half]`).

---

## The dataset blueprint (`[general]` / `[[datasets]]`)

This block lives *inside* base.toml but is **not** part of the flat
method+preset merge — `_DATASET_CONFIG_SECTIONS` skips it so it never pollutes
argparse. It's consumed separately by `BlueprintGenerator`, with the top-level
path keys above interpolated in at load time.

```toml
[general]
# (empty) — kohya-legacy keep_tokens / caption_extension were removed (inert in
# the cached workflow). Add a real dataset-wide default here only if one applies.

[[datasets]]
batch_size           = 1
validation_split_num = 0      # validation OFF by default
validation_seed      = 42

  [[datasets.subsets]]
  image_dir   = '{resized_image_dir}'   # interpolated from the top-level key
  cache_dir   = '{lora_cache_dir}'
  num_repeats = 1
  recursive   = true
```

| Key | Default | What it controls |
|---|---|---|
| `batch_size` | `1` | Per-subset training batch size. |
| `validation_split_num` | `0` | **Count-based** held-out validation size. `> 0` wins over the fractional `validation_split`. `0` = validation off. Auto-disabled if the pool is too small to leave a usable train set. |
| `validation_split` | *(commented)* | **Fractional** alternative (e.g. `0.025`). Ignored when `validation_split_num > 0`. |
| `validation_seed` | `42` | Seed for the deterministic shuffle before the split. |
| `image_dir` | `{resized_image_dir}` | Subset image source. The `{…}` template resolves against the top-level path keys. |
| `cache_dir` | `{lora_cache_dir}` | Redirects **every** VAE/TE/PE sidecar to a flat, stem-keyed location. EasyControl uses this to keep its source dir user-facing while caches land under `post_image_dataset/`. |
| `num_repeats` | `1` | How many times the subset is cycled per epoch. Usually leave at 1. |
| `recursive` | `true` | Walk subfolders under `image_dir`. Caches/resized output stay flat, so **image stems must be unique across the whole tree** (the trainer enforces this). |

**Overriding the blueprint:** drop a `[general]` / `[[datasets]]` block into a
method TOML to shallow-override **top-level scalars** (e.g. `batch_size`) via
`_apply_dataset_overrides`. Subset-level keys (per-subset `num_repeats`, extra
subsets) must go through `--dataset_config <path>`.

The `[half]` preset sets `sample_ratio = 0.5` via the global `--sample_ratio`
override — it shrinks **train only**; validation count stays exact.

---

## `configs/preprocess.toml` (preprocess-only)

Split out of base.toml because **base.toml is overwritten on `make update`** —
preprocess.toml is user-owned and preserved. Layered `preprocess.toml →
base.toml → preset → method` (preprocess.toml read first, so a legacy copy of
any key still in base.toml keeps winning — backward compatible).

| Key | What it controls |
|---|---|
| `source_image_dir` | The raw input dir `make preprocess` resizes from (`image_dataset/`). |
| `drop_lowres_images` | Skip images below the resolution floor instead of upscaling. |
| `min_pixels` | The low-res floor used by `drop_lowres_images`. |
| `target_res` | Which `EDGE_TOKEN_BANDS` tiers preprocess is allowed to use. |
| `mask_dir` | Where `make mask` writes merged masks, and where training looks them up. |

Two of these are **dual-use** — `load_method_preset` seeds them from
preprocess.toml at lowest priority (preset / method / CLI still override):

- `target_res` is inert at train time (training is self-describing from the
  cached latents); it is seeded only so it lands in the run's
  `.snapshot.toml`.
- `mask_dir` is load-bearing. It reaches every dataset subset that doesn't name
  a `mask_dir` of its own, via the BlueprintGenerator's argparse fallback, and
  `--mask_dir` overrides it. It is **gated on the directory existing** — the
  key names where `make mask` *would* write, so a checkout that never masked
  keeps falling back to the legacy `masks/{merged,sam,mit}` auto-resolution
  instead of silently enabling the masked-loss path over an empty tree. Same
  key drives `make mask` / `make mask-clean`, `make preprocess-reconcile`, the
  GUI mask counter and image overlay, and the turbo distill loop.

The rest of the **shared** path contract (`resized_image_dir`,
`lora_cache_dir`, model paths) stays in base.toml because the dataset blueprint
interpolates the dirs.

---

## See also

- [`training.md`](training.md) — method/variant selection and the three-axis LoRA surface.
- [`inference.md`](inference.md) — generation flags and workflows.
- [`guidebook.md`](guidebook.md) — end-to-end setup → preprocess → train → infer walkthrough.
- `CLAUDE.md` → **Config flow** and **Critical invariants** for the authoritative merge-order and bucketing rules.
