# Windows ROCm Guide

This document covers the AMD Radeon / ROCm path for Anima LoRA on Windows.
The normal beginner workflow is intentionally the same as the CUDA workflow:
use the one-line installer, then use the GUI. You do not need to choose PyTorch
indexes or `uv` extras manually.

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
6. create the **Anima LoRA GUI** desktop shortcut; and
7. launch the GUI.

For normal use, there is no separate AMD installation procedure after this.
Use the GUI for model download, preprocessing, training, and updates.

### Updating

Use the **Update** button in the GUI.

The installer stores the selected Windows backend in `.anima_backend`, and the
updater reuses it. An existing ROCm installation therefore stays on the ROCm
dependency path instead of being silently replaced by CUDA packages.

## Current verified hardware scope

The locked ROCm environment in this project currently packages and verifies
RDNA 4 only:

| Architecture | Tested hardware | Status |
|---|---|---|
| `gfx1200` | Radeon RX 9060 XT | Verified |
| `gfx1201` | Radeon RX 9070 XT | Verified |

Both devices were tested with the exact ROCm 10.0 Windows environment using
tensor allocation, `torch.compile`, bf16 PyTorch SDPA, backward, finite-value
checks, and device synchronization.

AMD's ROCm 10.0 compatibility matrix includes RDNA 3 targets (`gfx1100`,
`gfx1101`, and `gfx1102`) on Windows. The same packaging approach is expected
to extend to those architectures, but they are not part of the declared support
set here until they receive the same hardware smoke test.

> The installer currently detects the GPU vendor, not the exact AMD GPU
> architecture. The committed ROCm dependency set contains only `gfx1200` and
> `gfx1201`, so other AMD GPUs should be treated as unverified even if the
> installer selects ROCm.

## Locked software stack

The supported Windows ROCm environment for this integration is pinned to:

- Python 3.13
- PyTorch `2.13.0+rocm10.0.0`
- torchvision `0.28.0+rocm10.0.0`
- ROCm 10.0 device packages for `gfx1200` and `gfx1201`
- `triton-windows` for the Windows `torch.compile` runtime

The runtime/backend code is not fundamentally tied to one PyTorch minor version.
The Windows ROCm path intentionally uses PyTorch 2.13 because ROCm 10.0 validates
that combination on Windows; the CUDA path remains on its independently tested
PyTorch 2.12 + CUDA 13.2 stack.

For development-only compatibility checking, the same core Anima / ROCm path
has also been exercised locally with:

- PyTorch 2.14 alpha + ROCm 7.15 alpha
- PyTorch 2.15 alpha + ROCm 10.1 alpha

Those development stacks are **not** the locked or advertised supported
configuration. They only indicate that the backend integration itself is not
intrinsically coupled to one PyTorch minor version.

## Attention behavior on ROCm

The CUDA build keeps the existing Flash Attention path.

On ROCm, an explicit `attn_mode = "flash"` request is normalized to PyTorch
SDPA instead. The ROCm environment therefore does not install an AMD Flash
Attention dependency, while `torch_compile = true` remains enabled.

This separation is intentional: ROCm-specific dependency and attention choices
stay behind the backend selection so CUDA users keep the existing CUDA toolkit,
Flash Attention, and dependency workflow.

## Manual clone / advanced setup

The one-line installer above is recommended. If you intentionally install from
a git clone on Windows: CUDA is the **default** backend (the `cuda-windows`
dependency group is default-on, so a plain `uv sync` installs it — GH #92);
ROCm swaps that group out explicitly.

ROCm:

```powershell
uv sync --no-group cuda-windows --group rocm-windows
```

CUDA:

```powershell
uv sync
```

The two Windows backend groups are declared mutually exclusive so `uv` cannot
resolve a mixed CUDA/ROCm Torch environment. Reuse the same ROCm flags for
every later manual sync — a flagless `uv sync` reverts to CUDA.

To force ROCm when using the one-line installer:

```powershell
$env:ANIMA_BACKEND = 'rocm'
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

`ANIMA_BACKEND` accepts `auto`, `cuda`, or `rocm`. Automatic detection prefers
CUDA on a mixed NVIDIA + AMD system in order to preserve the historical CUDA
default; use the override above if ROCm is intentional.

## Runtime smoke test

A manual ROCm installation can run the same post-install check used by the
installer:

```powershell
uv run --no-group cuda-windows --group rocm-windows python tests/rocm_smoke_test.py
```

A successful run verifies the key training path rather than only checking that
`import torch` works.

## Troubleshooting

### The installer selected ROCm for an older or unverified AMD GPU

The current locked device set is `gfx1200` / `gfx1201`. Other Radeon
architectures are not yet declared verified by this project. If you want to add
an RDNA 3 target, use the appropriate ROCm device package and run
`tests/rocm_smoke_test.py` on real hardware before treating it as supported.

### `ROCm detected: using PyTorch SDPA instead of CUDA Flash Attention`

This message is expected when the configuration explicitly requests `flash` on
a HIP build. The runtime switches that request to PyTorch SDPA. `attn_mode=None`
or an explicit `torch` request does not count as a ROCm fallback.

### The ROCm smoke test fails

Do not continue with training until the smoke test passes. Confirm that the
installed environment is the `rocm-windows` extra and that the GPU is within the
currently packaged architecture set. Re-running the one-line installer in a new
empty target directory is the simplest clean-install check.

## Maintenance summary

For the normal Windows workflow, backend selection remains centralized in the
installer and updater:

- NVIDIA users keep CUDA 13.2 + CUDA PyTorch + Flash Attention.
- AMD users receive the ROCm PyTorch package set + PyTorch SDPA.
- `.anima_backend` preserves the selected path across updates.
- Beginner documentation can continue to point to the same one-line installer
  and GUI Update button for both vendors.
