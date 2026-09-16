# Inference Guide

Start at §1; the flag tables in §4 are for looking up a specific knob.

> Model paths, `--attn_mode`, `--vae_chunk_size`, and `--compile` mean the same
> as in training — see [`base-config.md`](base-config.md). The adapter family is
> read from the **checkpoint metadata**, so there is no flag for it.

---

## 1. Just test what I trained

Every `make test-*` target auto-picks the latest adapter in `output/ckpt/` and
runs it with the preset values below (`INFERENCE_BASE` in
`scripts/tasks/_common.py`).

```bash
make test                  # latest LoRA / OrthoLoRA / T-LoRA
make test-hydra            # latest HydraLoRA / FeRA *_moe.safetensors (router-live)
make test-merge            # a baked/merged DiT under MODEL_DIR= (no adapter)
```

`SPECTRUM=1`, `MOD=1`, and `NOLORA=1` compose into every `test-*` target:

```bash
make test SPECTRUM=1       # + Spectrum acceleration
make test MOD=1            # + distilled pooled_text_proj (modulation guidance)
make test NOLORA=1         # bare DiT (skips --lora_weight); MOD=1 → mod-only sample
make test SPECTRUM=1 MOD=1 # stack them
```

What `make test` runs:

```
--image_size 1024 1024  --infer_steps 28  --flow_shift 3.0
--guidance_scale 4.0    --sampler euler   --attn_mode flash
--vae_chunk_size 64     --vae_disable_cache  --seed 42
```

> ⚠️ Bare `inference.py` defaults differ (`--infer_steps 50`,
> `--guidance_scale 3.5`, `--attn_mode torch`). When you write a command by
> hand, start from the `make test` values above.

Correction / conditioning test targets (each composes with `SPECTRUM`/`MOD`):

| Target | Adds |
|---|---|
| `make test-smc-cfg` | SMC-CFG velocity-space correction |
| `make test-easycontrol REF_IMAGE=…` | EasyControl image conditioning |
| `make exp-test-directedit PROMPT='…'` | DirectEdit on a random source image |
| `make exp-test-directedit-dry` | DirectEdit reconstruction sanity check |

---

## 2. Generate by hand

Call `inference.py` directly for full control:

```bash
python inference.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --lora_weight output/ckpt/anima_lora.safetensors \
    --prompt "your prompt" \
    --negative_prompt "worst quality, low quality, blurry" \
    --image_size 1024 1024 \
    --infer_steps 28 \
    --flow_shift 3.0 \
    --guidance_scale 4.0 \
    --sampler euler \
    --attn_mode flash \
    --save_path output/tests
```

Stack multiple adapters by space-separating `--lora_weight` (one
`--lora_multiplier` per weight, or a single scalar for all):

```bash
--lora_weight a.safetensors b.safetensors --lora_multiplier 0.8 0.6
```

Programmatic generation (`import anima_lora`) builds a typed
`GenerationRequest` instead — see `examples/01_generate.py`.

---

## 3. Common goals → which flags

### Batch a list of prompts
```bash
python inference.py … --from_file prompts.txt
```
One prompt per line; per-line flag overrides are appended inline:
```
a girl standing in a field --w 1024 --h 1024 --s 50 --g 5.0
another prompt --seed 42 --flow_shift 4.0
```
(`--w/--h` size, `--s` steps, `--g`/`--l` guidance, `--d`/`--seed` seed, `--fs` flow_shift.)

### Iterate interactively
`--interactive` opens a REPL prompt loop (keeps the model resident between prompts).

### Pick a sampler
| `--sampler` | Use when |
|---|---|
| `euler` | Default deterministic ODE. |
| `er_sde` | Stochastic (Extended Reverse-Time SDE); required for `--cns`. |
| `lcm` | x0 re-noise — distilled few-step models only (see Turbo below). |

### Few-step (Turbo / distilled) checkpoints
Turbo output is a normal LoRA but expects the DP-DMD rollout it was trained at
(`student_steps` in `configs/methods/turbo.toml`, currently 4):
```bash
python inference.py … --lora_weight turbo.safetensors --infer_steps 4 --guidance_scale 1.0
```

### Go faster (training-free)
| Goal | Flag | Notes |
|---|---|---|
| Skip blocks on cached steps | `--spectrum` | Chebyshev feature forecasting. See [`../inference/spectrum.md`](../inference/spectrum.md). |
| Compile the DiT | `--compile` / `--compile_blocks` | `torch.compile` speedup; first run pays the trace cost. |
| Lower-VRAM text encoder | `--text_encoder_cpu` | Keeps the TE on CPU. |

### Improve quality (training-free corrections)
| Goal | Flag | Notes |
|---|---|---|
| Sliding-mode CFG | `--smc_cfg` | α-adaptive velocity-space correction (λ=5, α=0.2). [`../inference/smc_cfg.md`](../inference/smc_cfg.md) |
| SDE noise recoloring | `--cns` | `--sampler er_sde` only (no-op on euler/lcm). [`../inference/cns.md`](../inference/cns.md) |
| Text-conditioned AdaLN steer | `--pooled_text_proj` + `--mod_w` | Modulation guidance (global tone, not content). [`../inference/mod-guidance.md`](../inference/mod-guidance.md) |
| Weak-tag / relation adherence | `--xattn_boost 2` | Cross-attn gain in the σ ≥ 0.85 plan-writing window, cond pass only. Amplifies *all* caption tags incl. framing. [`../inference/xattn_boost.md`](../inference/xattn_boost.md) |

### Condition on a reference image
| Goal | Flags |
|---|---|
| EasyControl (extended self-attn) | `--easycontrol_weight … --easycontrol_image … [--easycontrol_scale 1.0] [--easycontrol_image_match_size]` |

### High-resolution output that won't fit the VAE
```bash
--tiled_diffusion --tile_size 1024 --tile_overlap 64
```
(`--tile_size` and `--tile_overlap` must be even; overlap < size.)

### Debug: cut the LoRA off mid-trajectory (P-GRAFT)
```bash
--pgraft --lora_cutoff_step 37   # LoRA active steps 0–36, disabled 37+
```

---

## 4. Flag reference

### Core
| Flag | Default | Description |
|---|---|---|
| `--lora_weight` | — | Adapter path(s); space-separated to stack |
| `--lora_multiplier` | 1.0 | Scalar (one per weight, or one for all) |
| `--infer_steps` | 50 | Denoising steps (28 via `make test`) |
| `--guidance_scale` | 3.5 | Text CFG (4.0 via `make test`) |
| `--flow_shift` | 3.0 | Flow-matching schedule shift |
| `--sampler` | `euler` | `euler` / `er_sde` / `lcm` |
| `--attn_mode` | `torch` | `torch` / `flash` / `flex` / `sageattn` (`sdpa`→`torch`) |
| `--image_size` | 1024 1024 | H W |
| `--seed` | random | Eval seed |
| `--negative_prompt` | "" | — |
| `--from_file` | — | Batch prompts from a file |
| `--interactive` | off | REPL loop |
| `--compile` / `--compile_blocks` | off | `torch.compile` |
| `--text_encoder_cpu` | off | TE on CPU (low-VRAM) |
| `--vae_chunk_size` | — | VAE decode tile size |
| `--vae_disable_cache` | off | Skip the per-tile VAE cache |
| `--no_metadata` | off | Don't embed training metadata in the PNG |
| `--save_path` | — | Output directory (required) |

> `--fp8` and `--prefix_weight` were removed.

### Modulation guidance
| Flag | Description |
|---|---|
| `--pooled_text_proj` | Path to the distilled MLP |
| `--mod_w` | Guidance strength (positive boosts) |
| `--mod_pos_prompt` / `--mod_neg_prompt` | Text targets for the AdaLN delta |
| `--mod_start_layer` / `--mod_end_layer` | Layer band |
| `--mod_taper` / `--mod_taper_scale` / `--mod_final_w` | Schedule shaping |

### Spectrum
| Flag | Description |
|---|---|
| `--spectrum` | Enable |
| `--spectrum_warmup` | Steps before caching starts |
| `--spectrum_window_size` / `--spectrum_flex_window` | Adaptive window schedule |
| `--spectrum_w` / `--spectrum_m` / `--spectrum_lam` | Forecasting hyperparameters |
| `--spectrum_stop_caching_step` | Last cached step |
| `--spectrum_calibration` | Bias adjustment |

### SMC-CFG / CNS
| Flag | Description |
|---|---|
| `--smc_cfg` | Enable SMC-CFG |
| `--smc_cfg_lambda` / `--smc_cfg_alpha` | λ / α (defaults 5 / 0.2) |
| `--cns` | Enable CNS (`er_sde` only) |
| `--cns_strength` | CNS recoloring strength |

### Cross-attn boost
| Flag | Description |
|---|---|
| `--xattn_boost` | Cross-attn residual gain λ, cond forward only (1.0 = off; 2.0 = shipped setting) |
| `--xattn_boost_band` | σ cutoff (boost at σ ≥ band; default 0.85) |

---

## 5. LoRA in ComfyUI

Plain LoRA / OrthoLoRA / T-LoRA files load in ComfyUI's stock `LoraLoader`;
HydraLoRA / FeRA / postfix checkpoints need the Anima Adapter Loader node. Node
links and the merge-to-checkpoint route: [guidebook §10](guidebook.md#10-deploying-to-comfyui).
For ComfyUI-vs-CLI behaviour differences see
[`difference_between_comfy.md`](difference_between_comfy.md).
