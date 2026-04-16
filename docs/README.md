# Documentation

## Methods

Training and inference algorithms.

| Doc | Description |
|-----|-------------|
| [methods/hydra-lora.md](methods/hydra-lora.md) | HydraLoRA — MoE multi-head routing with per-layer experts |
| [methods/mod-guidance.md](methods/mod-guidance.md) | Modulation guidance — text-conditioned AdaLN steering via distilled MLP |
| [methods/prefix-tuning.md](methods/prefix-tuning.md) | Prefix tuning — 12 GB VRAM, ~1 step/s, continuous prefix vectors |
| [methods/spectrum.md](methods/spectrum.md) | Spectrum — training-free inference acceleration via Chebyshev forecasting |
| [methods/invert.md](methods/invert.md) | Embedding inversion — optimize text embeddings through the frozen DiT |
| [methods/apex.md](methods/apex.md) | APEX — self-adversarial one-step distillation |
| [methods/psoft-integrated-ortholora.md](methods/psoft-integrated-ortholora.md) | OrthoLoRA (exp) — Cayley parameterization + SVD-informed init |

## Optimizations

Compiler, kernel, and hardware setup.

| Doc | Description |
|-----|-------------|
| [optimizations/for_compile.md](optimizations/for_compile.md) | Changes from sd-scripts for torch.compile / dynamo |
| [optimizations/fa4.md](optimizations/fa4.md) | Flash Attention 4 — why it was evaluated and removed |
| [optimizations/cuda132.md](optimizations/cuda132.md) | CUDA 13.2 installation (driver, FA2, bitsandbytes) |

## Guidelines

User-facing guides and references.

| Doc | Description |
|-----|-------------|
| [guidelines/training.md](guidelines/training.md) | Training reference — LoRA variants, caption shuffle, masked loss, dataset config |
| [guidelines/inference.md](guidelines/inference.md) | Inference reference — flags, P-GRAFT, prompt files, LoRA format conversion |
| [guidelines/graft-guideline.md](guidelines/graft-guideline.md) | GRAFT curation guideline |
| [guidelines/difference_between_comfy.md](guidelines/difference_between_comfy.md) | anima_lora vs ComfyUI implementation differences |
| [guidelines/가이드북.md](guidelines/가이드북.md) | 한국어 종합 가이드 (Korean comprehensive guide) |
