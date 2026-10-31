# anima_lora

> **New: [MOSS-Image LoRA](docs/moss/README.md).** Train LoRAs for OpenMOSS's 5B [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) on a 16 GB card at 1.1 s/step, with Spectrum sampling (2.6×) — `make gui-moss`. A separate line from Anima.

🛠️ Install: [English](docs/installation/README.md) · [한국어](docs/installation/설치.md) · [日本語](docs/installation/インストール.md) · [中文](docs/installation/安装.md)

📖 Guidebook: [English](docs/guidelines/guidebook.md) · [한국어](docs/guidelines/가이드북.md) · [日本語](docs/guidelines/ガイドブック.md) · [中文](docs/guidelines/指南书.md)

<p align="center">
  <img src="docs/gui.png" alt="Anima LoRA GUI — training-config editor with method/variant picker, inline method help, and live training monitor" width="900">
</p>

LoRA / T-LoRA training and inference engine for the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model (DiT, flow matching).

## Install

One line: installs [uv](https://astral.sh/uv), picks NVIDIA CUDA or AMD ROCm on Windows, fetches the latest release, runs `uv sync` and opens the GUI (no git needed).

```bash
# Linux / macOS
curl -LsSf https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh | sh
```
```powershell
# Windows (PowerShell)
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

Sign in to Hugging Face and download models from the GUI. Requirements, checksum-verified and pinned installs, and the manual clone path: [Advanced installation](#advanced-installation).

## Features

Diagrams for every method: [`docs/anima/structure_images/`](docs/anima/structure_images/), prose walkthroughs in [`docs/anima/structure/`](docs/anima/structure/).

### Fast training

**MOSS-Image, fast and accurate: 13 GB peak VRAM · 1.1 s/step** on an RTX 5070 Ti (rank 32, 1024²), against 1.6 s/step on bf16 at the same peak, with lower prediction error than bf16 at every σ tested (1.62% vs 1.93% at σ = 0.5). A LoRA freezes the base, so its linears run as GPTQ-calibrated int8 ConvRot GEMMs in both directions, over an fp32 residual stream. Details: [training_step.md](docs/moss/optimization/training_step.md), [accuracy](docs/moss/optimization/int8.md#accuracy).

**Anima (bf16): 13.4 GB peak VRAM · 1.1 s/step** on an RTX 5060 Ti (rank 32, 1 MP). Per-block `torch.compile` with a dynamic sequence axis (one graph per token-count tier), fused QKV/KV, an activation memory budget instead of gradient checkpointing, and a compile-clean hot path. Details: [for_compile.md](docs/anima/optimizations/for_compile.md).

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
| Spectrum — feature forecasting, ≈1.75× ([ComfyUI node](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)) | [spectrum.md](docs/anima/inference/spectrum.md) |
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

## Advanced installation

**Requirements:** NVIDIA Ampere or newer with driver ≥595 (Python 3.13 + PyTorch 2.14 + CUDA 13.2). Windows ROCm targets RDNA 4 (`gfx1200` / `gfx1201`; certified on RX 9060 XT / 9070 XT) with PyTorch 2.13 + ROCm 10.0.

The installer writes to `./anima_lora/` (`ANIMA_DIR` overrides) and, on Windows, adds an "Anima LoRA GUI" desktop shortcut. Force the backend with `$env:ANIMA_BACKEND='cuda'` or `'rocm'`. Pin a release with `ANIMA_VERSION=<tag>`. Update in place with `make update`.

<details>
<summary><b>Safer install</b> — verify the script before running</summary>

Every release ships a `checksums.txt` (SHA-256 of the installers and source archives).

```bash
# Linux / macOS
curl -fLO https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh
curl -fLO https://github.com/sorryhyun/anima_lora/releases/latest/download/checksums.txt
grep install.sh checksums.txt | sha256sum -c -    # must print "install.sh: OK"
less install.sh
sh install.sh
```
```powershell
# Windows (PowerShell)
iwr https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 -OutFile install.ps1
iwr https://github.com/sorryhyun/anima_lora/releases/latest/download/checksums.txt -OutFile checksums.txt
(Get-FileHash install.ps1 -Algorithm SHA256).Hash.ToLower()   # compare against checksums.txt
notepad install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
```
</details>

### Manual (from a clone)

```bash
uv sync                                                    # CUDA (default, every platform)
uv sync --no-group cuda-windows --group rocm-windows       # Windows AMD only
uv sync --no-default-groups --group torch-213              # older torch (or torch-212)

hf auth login             # or sign in from the GUI
make download-models      # DiT, TE, VAE, PE-Spatial, CJK vocab pack, tagger, tag DB → models/
make download-list        # installed vs missing; SAM3 and OCR are opt-in
```

- ROCm users repeat the same `--group` flags on every later manual `uv sync`. To keep an opt-in torch across updates, run `ANIMA_TORCH=2.13 make update` once.
- The manual CUDA path does not install the CUDA 13.2 toolkit that `torch.compile` / Triton needs: see [installation guide, Appendix B](docs/installation/README.md#appendix-b-manual-cuda-install).
- Install with `uv sync` against the committed `uv.lock`, not `pip install`: pip ignores uv's index strategy and the prebuilt flash-attn wheels.

## License

Toolkit code: [MIT](LICENSE).

Portions are **derived from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)** (Apache License 2.0) and remain under it — full text in [LICENSE-APACHE](LICENSE-APACHE), attribution and a statement of modifications in [NOTICE](NOTICE). Thanks to kohya-ss and the sd-scripts contributors.

Anima / CircleStone **base model weights** ship under the **CircleStone Labs Non-Commercial License v1.0** and are not relicensed by this repo. Any LoRA, fine-tune, or merged checkpoint trained from them is a Derivative and inherits the non-commercial terms. See [NOTICE](NOTICE).
