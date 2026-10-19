# Windows ROCm Guide

The AMD Radeon / ROCm path for Anima LoRA on Windows. The workflow is the same
as on NVIDIA: run the one-line installer, then use the GUI. You do not need to
choose PyTorch indexes or `uv` extras.

## Beginner install

Open PowerShell in the folder where you want Anima LoRA to be installed and run:

```powershell
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

The installer will:

1. detect the Windows GPU vendor;
2. select the CUDA or ROCm dependency set automatically;
3. install the locked Python / PyTorch environment;
4. save the selected backend for later updates;
5. verify the ROCm runtime when AMD is selected;
6. create the Anima LoRA GUI desktop shortcut; and
7. launch the GUI.

After this, use the GUI for model download, preprocessing, training, and
updates.

### Updating

Use the Update button in the GUI.

The installer stores the selected backend in `.anima_backend` and the updater
reuses it, so a ROCm install stays on ROCm.

## Current verified hardware scope

The locked ROCm environment packages and verifies RDNA 4 only:

| Architecture | Tested hardware | Status |
|---|---|---|
| `gfx1200` | Radeon RX 9060 XT | Verified |
| `gfx1201` | Radeon RX 9070 XT | Verified |

Both were tested on the locked ROCm 10.1 Windows environment (tensor
allocation, `torch.compile`, BF16 SDPA, native FlashAttention, and backward).
FlashAttention outputs and gradients are also compared against independent
CPU/FP64 references, including unequal tail lengths and grouped-query attention.

> The installer detects the GPU vendor, not the AMD architecture. It may select
> ROCm for any Radeon, but GPUs other than `gfx1200` / `gfx1201` (including
> RDNA 3) are unverified and are not targets of the bundled FlashAttention wheel.

## Locked software stack

The supported Windows ROCm environment is pinned to:

- Python 3.13
- PyTorch `2.14.0+rocm10.1.0`
- torchvision `0.29.0a0+rocm10.1.0`
- ROCm 10.1 runtime, development SDK, and device packages for `gfx1200` / `gfx1201`
- `triton-windows` 3.8 for the Windows `torch.compile` runtime
- A separately rebuilt Python 3.13 native FlashAttention 2.8.4 wheel for this stack

These torch/torchvision pins follow [AMD's ROCm 10.1 installation instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?rocm-ver=10.1.0).
The `a0` in torchvision is the version AMD published on its **stable** index,
not a request to use the nightly index. Pinning it explicitly avoids installing
the incompatible torchvision 0.28 build alongside torch 2.14 (see
[AMD's package compatibility tracker](https://github.com/ROCm/TheRock/issues/8802)).
The CUDA stacks remain unchanged: PyTorch 2.14 + CUDA 13.2 by default, 2.13 / 2.12 opt-in.

The ROCm SDK is installed inside the Python environment; a separate machine-wide
ROCm 10.1 installation is not needed. A compatible AMD graphics driver is still
required. Existing portable ComfyUI environments do not need to be changed.

## Attention behavior on ROCm

On supported Windows RDNA 4 GPUs, `attn_mode = "flash"` uses the bundled native
ROCm FlashAttention wheel. It contains `gfx1200` / `gfx1201` kernels for head
dimensions 32, 64, 128, and 256, including the non-tile-aligned backward tail fix.
`torch_compile = true` stays enabled. Set `attn_mode = "torch"` for explicit SDPA.

The runtime checks the wheel's build marker, exact torch version, GPU architecture,
and both fixed-length / varlen entry points before selecting FlashAttention. A
missing, unloadable, or mismatched wheel retains the previous SDPA fallback.
CUDA backend selection and its FlashAttention wheels are unchanged. The ROCm wheel
is a custom build, not an official AMD FlashAttention release; other GPU
architectures and arbitrary nightly torch versions are not supported by it.
The [custom build release](https://github.com/qweqweewqe7-create/flash-attention/releases/tag/windows-rocm101-torch214-gfx12-20261016)
includes the source archive, rebuild instructions, checksums, and validation reports.

## Manual clone / advanced setup

If you install from a git clone on Windows, CUDA is the default backend (the
`cuda-windows` dependency group is on by default); ROCm swaps it out explicitly.

ROCm:

```powershell
uv sync --no-group cuda-windows --group rocm-windows
```

CUDA:

```powershell
uv sync
```

The two groups are mutually exclusive. Reuse the same ROCm flags for every
later manual sync — a flagless `uv sync` reverts to CUDA.

`uv run` also syncs the environment by default. After installing ROCm, use
`uv run --no-sync python tasks.py gui` to launch with the installed packages,
or pass `--no-group cuda-windows --group rocm-windows` to `uv run` as well.

To force ROCm when using the one-line installer:

```powershell
$env:ANIMA_BACKEND = 'rocm'
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

`ANIMA_BACKEND` accepts `auto`, `cuda`, or `rocm`. On a mixed NVIDIA + AMD
system, `auto` picks CUDA; set `rocm` to override.

## Runtime smoke test

For a manual ROCm install, run the installer's post-install check:

```powershell
uv run --no-group cuda-windows --group rocm-windows python tests/rocm_smoke_test.py
```

It exercises allocation, compilation, SDPA, native FlashAttention outputs and
gradients, Anima's masked/varlen dispatcher, and compiled backward, not just
`import torch`. Each GPU is tested in a separate child process. It is not a
substitute for a complete model training / image-quality evaluation.

## Troubleshooting

### `Found no NVIDIA driver` on an AMD GPU

The environment contains CUDA PyTorch instead of ROCm PyTorch. Older Windows
installers could replace the verified ROCm build during shortcut creation or
GUI launch by running `uv run` without the ROCm group flags.

Close the GUI and stop active jobs, then repair the existing installation from
PowerShell in the install directory:

```powershell
uv sync --no-group cuda-windows --group rocm-windows
uv run --no-sync python tests/rocm_smoke_test.py
uv run --no-sync python tasks.py gui
```

ROCm still uses PyTorch's `torch.cuda` API and the `cuda` device name. Those
names alone do not indicate that the wrong PyTorch build is installed.

### The installer selected ROCm for an older or unverified AMD GPU

Only `gfx1200` / `gfx1201` are verified. To try another architecture (e.g.
RDNA 3), install the matching ROCm device package and validate SDPA separately
before training. The bundled native FlashAttention kernels and the full smoke
test intentionally target RDNA 4 only; SDPA fallback is not evidence of native
FlashAttention support on another architecture.

### `No compatible ROCm FlashAttention build: using PyTorch SDPA`

The requested FlashAttention wheel did not match the current torch / GPU stack,
or could not be imported. The runtime keeps SDPA instead of loading incompatible
kernels. Re-sync with the ROCm group if native FlashAttention was expected, and
run the smoke test. Do not install a CUDA or differently built ROCm wheel over it.

### The ROCm smoke test fails

Don't train until it passes. Confirm the environment was installed with the
`rocm-windows` group and that the GPU is `gfx1200` / `gfx1201`. For a clean
reinstall, re-run the one-line installer into a new empty directory.
