# anima_lora

> **New: [MOSS-Image LoRA](docs/moss/README.md).** Train LoRAs for OpenMOSS's 5B [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) on a 16 GB card at 1.1 s/step.

🛠️ Install: [English](docs/installation/README.md) · [한국어](docs/installation/설치.md) · [日本語](docs/installation/インストール.md) · [中文](docs/installation/安装.md)

📖 Guidebook: [English](docs/guidelines/guidebook.md) · [한국어](docs/guidelines/가이드북.md) · [日本語](docs/guidelines/ガイドブック.md) · [中文](docs/guidelines/指南书.md)

<p align="center">
  <img src="docs/gui.png" alt="Anima LoRA GUI — training-config editor with method/variant picker, inline method help, and live training monitor" width="900">
</p>

LoRA / T-LoRA training and inference engine for the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model (DiT, flow matching).

## Install

Following command installs [uv](https://astral.sh/uv), picks NVIDIA CUDA or AMD ROCm on Windows, runs `uv sync` and opens the GUI.

```bash
# Linux / macOS
curl -LsSf https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh | sh
```
```powershell
# Windows (PowerShell)
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

Sign in to Hugging Face and download models from the GUI. Requirements, checksum-verified and pinned installs, and the manual clone path: [installation guide](docs/installation/README.md).

## Features

Diagrams for every method: [`docs/anima/structure_images/`](docs/anima/structure_images/), prose walkthroughs in [`docs/anima/structure/`](docs/anima/structure/).

### Fast training

**MOSS-Image, fast and accurate: 13 GB peak VRAM · 1.1 s/step** on an RTX 5070 Ti (rank 32, 1024²), with lower prediction error than bf16. Details: [training_step.md](docs/moss/optimization/training_step.md), [accuracy](docs/moss/optimization/int8.md#accuracy).

**Anima (bf16): 13.4 GB peak VRAM · 1.1 s/step** on an RTX 5060 Ti (rank 32, 1 MP). Details: [for_compile.md](docs/anima/optimizations/for_compile.md).

**Anima's fp32 stream.** The residual stream, modulation, q / k norms, RoPE, embedder and head run in fp32. Details: [fp32_stream.md](docs/anima/optimizations/fp32_stream.md).

### On by default

| Feature | Doc |
|---|---|
| LoRA with SVD-Down init (`down_init = "weight_svd"`) | [svd-down-lora.md](docs/anima/methods/svd-down-lora.md) |
| T-LoRA timestep rank masking | [timestep_mask.md](docs/anima/methods/timestep_mask.md) |
| Channel scaling (`channel_scaling_alpha = 0.5`, Adam-specific) | [channel_scaling.md](docs/anima/optimizations/channel_scaling.md) |
| Free-fit native-aspect bucketing (the only resize mode) | [for_compile.md](docs/anima/optimizations/for_compile.md) |
| AdaLN low-rank delta (`train_adaln`, provisional) | [adaln.md](docs/anima/methods/adaln.md) |
| CJK vocab pack (Japanese text in captions and images) | [cjk_vocab_pack.md](docs/anima/methods/cjk_vocab_pack.md) |

LoRA and T-LoRA bake losslessly into a standalone DiT: `make merge [ADAPTER_DIR=… MULTIPLIER=0.8]`.

### Inference stacks

Training-free; each composes with any checkpoint. Overview: [docs/anima/inference/](docs/anima/inference/README.md).

| Method | Doc |
|---|---|
| Spectrum — feature forecasting, ≈1.75× ([ComfyUI node](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)) | [spectrum.md](docs/inference/spectrum.md) |
| SMC-CFG — sliding-mode CFG correction | [smc_cfg.md](docs/anima/inference/smc_cfg.md) |
| Modulation guidance | [mod-guidance.md](docs/anima/inference/mod-guidance.md) |
| SPD — progressive multi-resolution sampling | [spd.md](docs/anima/inference/spd.md) |
| Embedding inversion | [invert.md](docs/anima/inference/invert.md) |

### Training stacks

| Method | Doc |
|---|---|
| HydraLoRA — multi-head LoRA with per-sample routing ([ComfyUI node](https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter)) | [hydra-lora.md](docs/anima/methods/hydra-lora.md) |
| Turbo — DP-DMD few-step distillation, output is a normal LoRA ([4-step student](https://huggingface.co/sorryhyun/anima-turbo-4step)) | [turbo.md](docs/anima/methods/turbo.md) |
| Soft Tokens (experimental) | [soft_tokens.md](docs/anima/experimental/soft_tokens.md) |
| DirectEdit + Anima Tagger (experimental) | [directedit_editing_v3.md](docs/anima/experimental/directedit_editing_v3.md) |
| EasyControl (experimental) | [easycontrol.md](docs/anima/experimental/easycontrol.md) |

> **Want to contribute?** EasyControl adapters (canny / depth / pose / …, one PR per control type) — see [CONTRIBUTING.md → Priority areas](CONTRIBUTING.md#priority-areas).

## Usage

```bash
make gui                  # config editor + dataset browser + training monitor
make preprocess           # resize + cache latents / text embeddings
make lora                 # or PRESET=low_vram make lora
make test                 # sample with the latest LoRA
```

Config chain: `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`. Flags: [training.md](docs/anima/guidelines/training.md), [inference.md](docs/anima/guidelines/inference.md). All docs: [docs/](docs/README.md).

## Try it in the browser

**[anima-turbo-webgpu.vercel.app](https://anima-turbo-webgpu.vercel.app)** runs [Anima Turbo 4-step](https://huggingface.co/sorryhyun/anima-turbo-4step) (v4, int8 ONNX) on your own GPU through WebGPU, with no install.

Source: [`project/webgpu_int8/demo/`](project/webgpu_int8/demo/index.html).

## License

Toolkit code: [MIT](LICENSE).

Portions are **derived from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)** (Apache License 2.0) and remain under it — full text in [LICENSE-APACHE](LICENSE-APACHE), attribution and a statement of modifications in [NOTICE](NOTICE). Thanks to kohya-ss and the sd-scripts contributors.

Anima / CircleStone **base model weights** ship under the **CircleStone Labs Non-Commercial License v1.0** and are not relicensed by this repo. Any LoRA, fine-tune, or merged checkpoint trained from them is a Derivative and inherits the non-commercial terms. See [NOTICE](NOTICE).
