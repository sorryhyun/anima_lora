# CUDA 13.2 setup

## How to install

### Linux — default

`uv sync` already resolves to torch 2.12 nightly + CUDA 13.2 on Linux (see `[tool.uv.sources]` in `pyproject.toml`). Prerequisites:

1. Install CUDA 13.2 toolkit. `nvidia-driver-595` **open** variant required.
2. `uv sync` — pulls the [prebuilt FA2 wheel](https://github.com/sorryhyun/flash-attention-sm120-fix/releases/download/fa2cuda132/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl) (Python 3.13, CUDA 13.2, torch 2.12).

### Windows — opt-in

Default Windows is torch 2.11 stable + CUDA 13.0. To switch to CUDA 13.2 + torch 2.12 nightly:

1. **Install CUDA 13.2 toolkit** at the standard path (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`).
2. **Toggle `pyproject.toml` comments.** In the `dependencies` list:
   - Comment out the two lines under `# Windows: stable (default).` (torch and torchvision).
   - Uncomment the two lines under `# Windows: cuda132 opt-in.` (torch and torchvision).
   - Comment out the line under `# Windows: stable (default) — built against torch 2.11 + CUDA 13.0.` (flash-attn).
   - Uncomment the line under `# Windows: cuda132 opt-in — trimmed FA2 ...` (flash-attn).
3. **Re-sync**: `uv sync`. Pulls torch 2.12 nightly from the cu132 index and the prebuilt trimmed FA2 wheel from the GitHub release.

No build tools needed for this path — the FA2 wheel is prebuilt at the URL referenced in `pyproject.toml`.

### Building the trimmed FA2 wheel (maintainer / fork)

If you're targeting a different GPU / Python version / your own fork and need to rebuild the wheel:

1. **CUDA 13.2 toolkit** at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`.
2. **MSVC 2019 Build Tools** + Windows 10 SDK + Python 3.13. Open the *x64 Native Tools Command Prompt for VS 2019* and `set DISTUTILS_USE_SDK=1` before building if MSVC isn't on `PATH` by default.
3. Build:
   ```powershell
   uv pip install --no-build-isolation -e flash-attention
   python flash-attention\repack_wheel.py   # produces .whl at project root
   ```
   Sources are trimmed to bf16 + `head_dim=128` + non-causal kernels only (see `flash-attention/setup.py` and `csrc/flash_attn/src/static_switch.h` for the `FLASHATTENTION_DIFFUSION_ONLY` / `FLASHATTENTION_DISABLE_CAUSAL` macros). Cuts the build from 92 → 3 `.cu` files. Misuse (calling FA2 with fp16, head_dim ≠ 128, or causal=True) fails loudly via `TORCH_CHECK`.
4. Upload the resulting `flash_attn-2.8.4-cp313-cp313-win_amd64.whl` to your release tag (or use a local `file://` URL in `pyproject.toml`).

## How much faster?

Roughly 10%. Benchmark results on a single RTX 5060 Ti 16 GB, fixed seed, bsz=1, 2 epochs / 446 steps, fully compiled (4096-token static pad), r=16, lr=5e-5:

- `make lora`: 1.15 s/step
- `PRESET=fast_16gb make lora`: 0.97 s/step
- `make lora-gui GUI_PRESETS=tlora`: 1.30 s/step

Environment: CUDA 13.2, torch nightly (2.12), FA2 without gradient checkpointing.

### FA2, no gradient checkpointing

| Variant | Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| baseline | 14.1 GB | 9:07 | 4:26 | 1.19 s | 0.087 | 0.225 |
| + RoPE cache, fused qkv | 14.1 GB | 8:54 | 4:23 | 1.18 s | 0.086 | 0.223 |

### FA2, no checkpoint, no block swap (skipped 4 LoRA layers)

| Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|
| 14.1 GB | 7:48 | 3:52 | 1.05 s | 0.084 | 0.241 |

### FA4, no gradient checkpointing

| Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|
| 15.0 GB | 9:41 | 4:36 | 1.24 s | 0.088 | 0.216 |

### FA4, no checkpoint, no block swap (skipped 4 LoRA layers)

| Peak VRAM | Total | 2nd Epoch | sec/step | Train Loss | Val Loss |
|---|---|---|---|---|---|
| 14.9 GB | 9:07 | 4:01 | 1.12 s | 0.087 | 0.209 |

FA4 was subsequently dropped from the default training path; see [fa4.md](fa4.md) for why.
