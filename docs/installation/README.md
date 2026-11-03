# Installation

Setup shared by every line in this repo (Anima, MOSS-Image): requirements, install, Hugging Face sign-in, updating. After installing, download your line's models — Anima: [guidebook §3](../guidelines/guidebook.md#3-hugging-face-sign-in-and-model-download); MOSS-Image: [MOSS guidebook §2](../guidelines/moss-guidebook.md#2-setup).

---

## 1. System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 or better (20xx series and below unsupported) | 16 GB VRAM or more |
| System RAM | 16 GB | 32 GB or more |
| Disk | 60 GB free | 200 GB or more (caches + accumulated outputs) |
| OS | Windows 11 / Ubuntu 22.04+ | Ubuntu 24.04 |
| Python | 3.13 required | - |
| NVIDIA driver | 595 or newer (CUDA 13.x requirement) | - |

MOSS-Image is measured on a 16 GB card; its memory plan is in [MOSS guidebook §4.2](../guidelines/moss-guidebook.md#42-memory-plan-16-gb).

---

## 2. Installation

Paste the line for your OS into PowerShell (Windows) or a terminal (Linux / macOS). It installs `uv`, the CUDA 13.2 toolkit, Python 3.13, and all dependencies automatically, then opens the GUI for you when it's done.

```powershell
# Windows (PowerShell)
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```
```bash
# Linux / macOS
curl -LsSf https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh | sh
```

- Installs into `.\anima_lora\` (change with `$env:ANIMA_DIR`, pin a version with `$env:ANIMA_VERSION='v2.0.0'`).
- If it asks for a reboot (common after the CUDA step), reboot and re-run the same one-liner — it picks up where it left off.
- To reopen it later, use the "Anima LoRA GUI" desktop shortcut.

> Prefer to `git clone` yourself, or the automatic install failed? See [Appendix A](#appendix-a-manual-install-uv--git) / [Appendix B](#appendix-b-manual-cuda-install).

> AMD Radeon on Windows: the same one-liner selects ROCm. Details: [rocm.md](rocm.md).

---

## 3. Hugging Face Sign-in

1. Create a read token at <https://huggingface.co/settings/tokens>.
2. Paste it into the Hugging Face sign-in field in the GUI.

The token is stored in the standard Hugging Face cache, so the GUI and the CLI share the same credentials.

---

## 4. Updating

Run this inside the install folder to pull the latest GitHub release and sync dependencies (no git required).

```bash
python tasks.py update              # Apply
python tasks.py update -- --dry-run # Preview which files change
```

Updating never touches `image_dataset/`, `post_image_dataset/`, `output/`, or `models/`, and asks before overwriting config files you have modified.

---

## 5. RTX 50-series: Patched FlashAttention (Optional)

On an RTX 50-series card, our FlashAttention build ([sorryhyun/flash-attention-fp16](https://github.com/sorryhyun/flash-attention-fp16)) makes a MOSS-Image or Anima training step about 5% faster. Generation gains little. Wheels exist for Windows and Linux x86_64. Run this in the install folder:

```bash
uv sync --no-default-groups --group fa2-sm120
```

- A plain `uv sync` or `python tasks.py update` puts the stock wheel back. To keep the build across updates, run the update once with `ANIMA_TORCH=2.14-sm120` set (PowerShell: `$env:ANIMA_TORCH='2.14-sm120'; python tasks.py update`).
- It holds sm_120 kernels only: on any other card, stay on the stock wheel.
- ComfyUI running on the same venv gets the build too.

Details: [fa2_sm120.md](../moss/optimization/fa2_sm120.md).

---

## Appendix A. Manual Install (uv · git)

Only needed if you want to set things up yourself instead of the one-liner in [§2](#2-installation).

```powershell
irm https://astral.sh/uv/install.ps1 | iex   # 1) Install uv (check `uv --version` in a new shell)
```

```bash
git clone https://github.com/sorryhyun/anima_lora.git   # 2) Clone
cd anima_lora
winget install ezwinports.make                          # 3) (Optional) make
uv sync                                                 # 4) Install dependencies
```

`uv sync` installs everything into `.venv/` inside `anima_lora/`; you don't need `pip install`.

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

## Appendix B. Manual CUDA Install

The one-liner in [§2](#2-installation) handles CUDA 13.2 automatically and skips the step if it's already installed. This appendix is only for when that fails.

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
