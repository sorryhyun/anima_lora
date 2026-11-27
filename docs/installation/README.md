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

The default stack is Python 3.13 + PyTorch 2.14 + CUDA 13.2 on NVIDIA Ampere or newer. Windows ROCm targets RDNA 4 (`gfx1200` / `gfx1201`; certified on RX 9060 XT / 9070 XT) with PyTorch 2.13 + ROCm 10.0.

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
- On Windows the installer picks CUDA or ROCm from the card; force one with `$env:ANIMA_BACKEND='cuda'` or `'rocm'`.

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

> Prefer to `git clone` yourself, or the automatic install failed? See [manual.md](manual.md).

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

- A plain `uv sync` puts the stock wheel back. `python tasks.py sync` (`make sync`) and `python tasks.py update` sync the stack saved in `.anima_torch`. On Linux x86_64 with no saved stack, they detect an RTX 50-series card and pick this build by themselves. On Windows, run the sync once with `ANIMA_TORCH=2.14-sm120` set (PowerShell: `$env:ANIMA_TORCH='2.14-sm120'; python tasks.py sync`); later syncs and updates keep it.
- It holds sm_120 kernels only: on any other card, stay on the stock wheel.
- ComfyUI running on the same venv gets the build too.
- Linux x86_64 only: the `fa2-sm120-fp32div` stack (`uv sync --no-default-groups --group fa2-sm120-fp32div`, or `ANIMA_TORCH=2.14-sm120-fp32div make sync`) adds `comfy-kitchen` with the INT8 activation fix ([comfy-kitchen#289](https://github.com/Comfy-Org/comfy-kitchen/issues/289)). The int8 DiT linears then skip their fp32 input copy at the same output, and the int8 backward and the MOSS int8 text encoder divide in fp32. The trainer detects the kernel at runtime; the stock one keeps working. It needs libstdc++ from GCC 13+ (Ubuntu 24.04+); `make sync` picks this stack by itself on an RTX 50-series Linux box that meets that floor.

Details: [fa2_sm120.md](../moss/optimization/fa2_sm120.md).
