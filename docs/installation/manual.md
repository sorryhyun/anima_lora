# Manual Installation

Setting up from a `git clone` instead of the one-liner in the [installation guide §2](README.md#2-installation), and installing the CUDA toolkit by hand when the one-liner's CUDA step fails.

---

## 1. Install from a clone (uv · git)

```powershell
irm https://astral.sh/uv/install.ps1 | iex   # 1) Install uv (check `uv --version` in a new shell)
```

```bash
git clone https://github.com/sorryhyun/anima_lora.git   # 2) Clone
cd anima_lora
winget install ezwinports.make                          # 3) (Optional) make
make sync                                               # 4) Install dependencies (torch stack picked from the GPU)
hf auth login                                           # 5) Or sign in from the GUI
make download-models                                    # 6) Anima set: DiT, TE, VAE, PE-Spatial, CJK vocab pack, tagger, tag DB → models/
make download-list                                      #    installed vs missing; SAM3 and OCR are opt-in
```

`make sync` syncs the torch stack saved in `.anima_torch`; with no file yet it picks one from the hardware ([installation guide §5](README.md#5-rtx-50-series-patched-flashattention-optional)) and saves it. `ANIMA_TORCH=2.13 make sync` picks one by hand. A plain `uv sync` installs the default stack; other stacks by hand:

```bash
uv sync                                                    # CUDA (default, every platform)
uv sync --no-group cuda-windows --group rocm-windows       # Windows AMD only
uv sync --no-default-groups --group torch-213              # older torch (or torch-212)
```

ROCm users repeat the same `--group` flags on every later manual `uv sync`. The manual path does not install the CUDA 13.2 toolkit that `torch.compile` / Triton needs: see [§2](#2-manual-cuda-install).

`uv sync` installs everything into `.venv/` inside `anima_lora/`, against the committed `uv.lock`. Don't `pip install`: pip ignores uv's index strategy and the prebuilt flash-attn wheels.

Running `python tasks.py ...` in a fresh terminal uses the *system* Python and fails with `ModuleNotFoundError`. Fix it either way:

- Prefix with `uv run` (no activation needed, works anywhere): `uv run python tasks.py lora`
- Activate the venv (once per terminal window):

  ```powershell
  .venv\Scripts\activate        # Windows
  ```
  ```bash
  source .venv/bin/activate     # Linux / macOS / WSL
  ```

  A `(anima_lora)` or `(.venv)` prefix in the prompt means it worked; `deactivate` to exit. In VSCode, pick the `.venv` interpreter once (Command Palette → Python: Select Interpreter) and the integrated terminal activates it automatically.

---

## 2. Manual CUDA Install

The one-liner in [installation guide §2](README.md#2-installation) handles CUDA 13.2 automatically and skips the step if it's already installed. This section is only for when that fails.

- CUDA 13.x requires NVIDIA driver 595 or newer. If you already have a newer driver, keep it (deselect the bundled driver in the Linux `.run` installer menu).
- To manage CUDA yourself, set `$env:ANIMA_SKIP_CUDA='1'` (PowerShell) / `ANIMA_SKIP_CUDA=1` (shell) to skip the automatic CUDA step.

Manual install: get 13.2 from the NVIDIA archive at <https://developer.nvidia.com/cuda-13-2-0-download-archive>

1. Windows: choose Operating System: Windows → Architecture: x86_64 → Version: 11/10 → Installer Type: exe (local), run it, and pick "Express (Recommended)".
2. Verify with `nvidia-smi` and `nvcc --version` (expect `release 13.2`).
3. If `nvcc` isn't found, add these to the system `Path`, reboot, and check again:

   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\libnvvp
   ```
