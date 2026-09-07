# Anima LoRA Guidebook

A complete guide to the **Anima LoRA** training/inference pipeline: install → dataset → preprocessing → training → inference → ComfyUI deployment. It is written for Windows beginners and **assumes GUI use**; every terminal command has been collected into [Appendix A](#appendix-a-cli-reference). For WSL, Linux, and training optimization (this project's main focus), see the other docs.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Hugging Face Sign-in and Model Download](#3-hugging-face-sign-in-and-model-download)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Preprocessing](#5-preprocessing)
6. [Using the GUI](#6-using-the-gui)
7. [Training](#7-training)
8. [Adapter Variant Selection Guide](#8-adapter-variant-selection-guide)
9. [Inference](#9-inference)
10. [Deploying to ComfyUI](#10-deploying-to-comfyui)
11. [Updating](#11-updating)

- [Appendix A. CLI Reference](#appendix-a-cli-reference)
- [Appendix B. Manual Install (uv · git)](#appendix-b-manual-install-uv--git)
- [Appendix C. Manual CUDA Install](#appendix-c-manual-cuda-install)
- [Appendix D. Other Settings](#appendix-d-other-settings)

---

## 1. System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| GPU | **RTX 3060 or better** (20xx series and below unsupported) | 16 GB VRAM or more |
| System RAM | 16 GB | 32 GB or more |
| Disk | 60 GB free | 200 GB or more (caches + accumulated outputs) |
| OS | Windows 11 / Ubuntu 22.04+ | Ubuntu 24.04 |
| Python | **3.13 required** | - |
| NVIDIA driver | **595 or newer** (CUDA 13.x requirement) | - |

---

## 2. Installation

Paste this one line into PowerShell. It installs `uv`, the CUDA 13.2 toolkit, Python 3.13, and all dependencies automatically, then **opens the GUI for you** when it's done.

```powershell
irm https://raw.githubusercontent.com/sorryhyun/anima_lora/main/install.ps1 | iex
```

- Installs into `.\anima_lora\` (change with `$env:ANIMA_DIR`, pin a version with `$env:ANIMA_VERSION='v1.4.0'`).
- **If it asks for a reboot** (common after the CUDA step), reboot and re-run the same one-liner — it picks up where it left off.
- To reopen it later, use the **"Anima LoRA GUI"** desktop shortcut.

Once installed, continue in the GUI with **§3 → §4 → §5**.

> Prefer to `git clone` yourself, or the automatic install failed? See [Appendix B](#appendix-b-manual-install-uv--git) / [Appendix C](#appendix-c-manual-cuda-install).

---

## 3. Hugging Face Sign-in and Model Download

### 3.1 Sign in

1. Create a **read** token at <https://huggingface.co/settings/tokens>.
2. Paste it into the **Hugging Face sign-in** field in the GUI.

The token is stored in the standard Hugging Face cache, so the GUI and the CLI share the same credentials.

### 3.2 Download models

The **Models** dialog in the GUI downloads all three with one button.

| File | Path |
|---|---|
| Anima DiT (the diffusion model itself) | `models/diffusion_models/anima-base-v1.0.safetensors` |
| Qwen3 0.6B text encoder | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

The **SAM3** and **MIT** checkpoints are fetched as well. Both are used only by the optional masked-loss feature ([§7.4](#74-masked-loss-excluding-text-bubbles)).

> **SAM3 is a gated model.** Go to <https://huggingface.co/facebook/sam3>, click **Request access**, and wait for approval (minutes to days); until then the SAM3 download fails with a 403. The three core models are *not* gated and the download continues regardless, so you can start training while waiting for SAM3 approval.
>
> If downloads keep breaking, fetch them individually with the targets in [Appendix A](#appendix-a-cli-reference).

---

## 4. Dataset Preparation

The structure is *image + same-name `.txt` caption sidecar*. Put originals in `image_dataset/` (subfolders are free).

```
image_dataset/
├─ 00001.png
├─ 00001.txt
├─ 00002.jpg
├─ 00002.txt
└─ subfolder/
   ├─ 00010.webp
   └─ 00010.txt
```

### 4.1 Caption Writing Tips

Following Anima's official guidelines, tag order is always `[meta] [character] [series] [artist] [general]`.

```
absurdres, safe, 1girl, chitanda eru, hyouka, @channel (caststation), full body, serafuku, She is saying hi.
```

- Based on personal experimentation, **quality tags** such as `absurdres`, `highres`, and `masterpiece` are best omitted or kept to a minimum. (Once the officially released mod guidance is available, you can skip them entirely.)
- **Don't want to caption by hand?** In the GUI's **Dataset** tab, select an image and click **Autotag** — the built-in **Anima Tagger** fills the caption in the correct order ([§6.3](#63-dataset-tab-autotag--grouping)). Treat the result as a starting point: review the tags, especially character/series/artist names, before training.

---

## 5. Preprocessing

To optimize training speed and VRAM, **resize → VAE latent caching → text embedding caching** are done in advance. The GUI's **Preprocess** tab runs all three with one button.

| Step | What it does | Output |
|---|---|---|
| Resize | Resizes to the pixel alignment the VAE requires, assigns each image a fixed token bucket, and excludes images that are too small (default: below 0.5 MP) | `post_image_dataset/resized/` |
| VAE latent caching | Runs the VAE once and saves the result — the VAE is never loaded onto the GPU during training | `post_image_dataset/lora/{stem}_{WxH}_anima.npz` |
| Text embedding caching | Pre-computes Qwen3 0.6B + LLM adapter outputs (including comma-shuffled caption variants) | `post_image_dataset/lora/{stem}_anima_te.safetensors` |

PE vision feature caching (`{stem}_anima_pe.safetensors`) is an optional step needed only for CMMD validation.

> **⚠️ Caches are reused and never automatically deleted.**
> Re-running preprocessing *never* overwrites or deletes existing `.npz` / `_te.safetensors` / `_pe.safetensors` files — only **missing entries** are processed. That makes re-runs very fast and safe to interrupt.
>
> - **Added images** → just run it again.
> - **Edited captions** or changed tokenizer/resize options → a plain re-run will *not* pick the change up. Delete the cache directory (`post_image_dataset/lora/`) manually, then re-run.

---

## 6. Using the GUI

Edit configs, browse the dataset, preprocess, start/monitor training, and merge LoRA — all in one window.

- **Training Config**: pick a variant from the dropdown (recommended: `tlora`) and your card in the **Hardware** dropdown (Default 16GB+ / Low VRAM 8GB), edit the training keys, then start training. Data scope (`sample_ratio`, `artists_shard`) is a plain form field in the Basic section.
- **Preprocess**: resize + VAE + text embedding caching in one shot.
- **Dataset**: preview and edit images/captions, Autotag, and Grouping ([§6.3](#63-dataset-tab-autotag--grouping)).
- **Merge**: bake a trained LoRA into the base DiT to produce a standalone ComfyUI checkpoint (base LoRA / OrthoLoRA / T-LoRA only).

The GUI reads `configs/gui-methods/<variant>.toml` (one clean file per variant) and calls `train.py` internally, so any GUI setup is reproducible from the CLI ([Appendix A](#appendix-a-cli-reference)).

### 6.1 Form Editing and Save Behavior

Training/preprocess subprocesses re-read the variant TOML from disk, so edits you don't save never reach training. The GUI handles this two ways:

- **Change detection**: editing any field (or the `+ Extra args` box) turns the `Save` button orange and marks it `Save *` — the signal that *the screen and the disk differ*.
- **Auto-save**: if you forget and press `Train` / `Preprocess` anyway, the current form values are written to the variant file first, then the subprocess starts. **What you see is what runs.** (`Test` infers from the last checkpoint, so it is not auto-saved.)

> To discard edits, switch to another variant and back — the file is reloaded from disk.

### 6.2 Stopping Training and Closing the GUI

Training does not run *inside* the GUI window — pressing `Train` hands the job to a background **training daemon** that runs `train.py` as a detached process.

- **`Stop` cancels the current job only.** The daemon stays up and moves on to the next queued job.
- **Closing the GUI does not stop training.** Reopen it and it automatically re-attaches to the running job (`Re-attached to running job …`), with the progress bar and logs resuming live. Useful for long overnight runs.
- Exceptions: **`Test` and `Preprocess` run as in-window subprocesses**, so closing the GUI cancels them.
- To shut training down completely and release the GPU, use the CLI's `make daemon-terminate` ([Appendix A](#appendix-a-cli-reference)).

### 6.3 Dataset Tab: Autotag & Grouping

**Autotag — caption generation.** Select an image and click **Autotag**: the **Anima Tagger** predicts tags in the correct `[meta] [character] [series] [artist] [general]` order and fills the caption box. The first click downloads the tagger model, so it takes a moment; afterwards the model stays warm in the background and subsequent images are near-instant. The tagger releases its GPU memory automatically before any other GPU job starts.

**Grouping — clustering similar images.** Click **Group** to cluster near-duplicates and images showing the same scene/character; they collapse into **green group headers** in the image list. Use it to drop duplicates or rebalance how much weight each concept gets.

- It compares *visual content*, not filenames or captions, and computes groups **per top-level folder** (artist/character buckets), so images from different folders never merge.
- Cached features are reused, so re-clicking **Group** after adding images is cheap.

> Both work on the **original** `image_dataset/` images and do not touch preprocessing caches — run them *before* preprocessing to clean up captions.

---

## 7. Training

The config merge order is `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`, and **method settings win over preset settings**.

The best starting point is **OrthoLoRA + T-LoRA (the `tlora` variant)** — the most balanced combination of stability, detail, and style preservation, and directly usable for ordinary character/style LoRAs. In the GUI, just pick `tlora` in the variant dropdown plus your Hardware preset.

### 7.1 Commonly Adjusted Settings (LoRA defaults)

| Parameter | Default | Description |
|---|---|---|
| `network_dim` | `32` | LoRA rank. Higher = more expressive, more parameters |
| `network_alpha` | `32` | LoRA scale (usually equal to `network_dim`) |
| `learning_rate` | `2e-5` | Learning rate. Hydra can go lower |
| `max_train_epochs` | `4` | Smaller dataset → more epochs |
| `save_every_n_epochs` | `2` (gui-methods) / `4` (methods) | Cumulative adapter-weight save interval |
| `checkpointing_epochs` | `2` (gui-methods) / `4` (methods) | Resume-state save interval (single file, overwritten) |
| `caption_dropout_rate` | `0.1` | Replaces some captions with an empty string (helps CFG) |
| `use_shuffled_caption_variants` | `true` | Use comma-shuffled caption variants |

Variant toggles (`use_ortho`, `use_timestep_mask`, `use_moe_style`, `router_source`, …) are already set in each variant file. The recommended `tlora` is `use_ortho = true` + `use_timestep_mask = true`.

### 7.2 Auto-Resume (checkpointing_epochs)

If training is interrupted, it **automatically resumes from the last saved point** — covering power loss, OOM, and accidentally closing the window. It is on by default: just press `Train` again, and `auto-resuming from checkpoint at step N` in the log confirms it worked.

This is a different job from `save_every_n_epochs`:

| Key | What it saves | Cumulative? | Purpose |
|---|---|---|---|
| `save_every_n_epochs` | Adapter weights (`anima_lora-000004.safetensors`, …) | **Cumulative** (cap with `save_last_n_epochs`) | Run inference on intermediate results, compare overfitting points |
| `checkpointing_epochs` | Full resume state (optimizer / scheduler / RNG / weights) | **Overwrites a single file** (disk doesn't grow) | Continue after an interruption |

- When training finishes normally the resume files are deleted automatically, leaving only the final output.
- **If you changed the dataset or core settings (rank, LR, epoch count, …)**, resuming from the old state is meaningless or harmful. Delete `output/ckpt/<output_name>-checkpoint-state/` manually and start fresh.

### 7.3 Outputs

- Trained weights: `output/ckpt/<output_name>.safetensors` (named per variant — `anima`, `anima_tlora_ortho`, `anima_hydra`, `anima_postfix`, …)
- Intermediate checkpoints: `output/ckpt/` (with a `.snapshot.toml` sidecar, plus a `_moe` companion file for Hydra)
- Validation samples: `output/ckpt/sample/` · inference images: `output/tests/`

### 7.4 Masked Loss (Excluding Text Bubbles)

For manga/comic-style data, excluding *speech bubbles and text regions* from the loss produces noticeably cleaner results. Masks are generated with SAM3 + MIT (`make mask`, [Appendix A](#appendix-a-cli-reference)) and the resulting PNGs are black-and-white: **white (255) = trained on**, **black (0) = excluded**.

Subsets use `post_image_dataset/masks/` automatically when present, otherwise falling back to the legacy `masks/{merged,sam,mit}/` layout in order. Missing masks are simply ignored, so this step is optional.

---

## 8. Adapter Variant Selection Guide

> **🌟 Recommended**: if this is your first run or an ordinary character/style LoRA, start with **`tlora` (OrthoLoRA + T-LoRA)**.

| Variant | GUI variant name | When to use |
|---|---|---|
| **OrthoLoRA + T-LoRA** ⭐ | `tlora` | **Recommended.** SVD-based orthogonal rotation (OrthoLoRA) stacked with per-timestep rank masking (T-LoRA). Produces `anima_tlora_ortho.safetensors` |
| **Plain LoRA** | `lora` | Simplest baseline, for comparison runs |
| **HydraLoRA** | `hydralora` | MoE multi-head routing, many concepts in one adapter |
| **ChimeraHydra** *(experimental)* | `chimera_hydra` | Content/frequency dual-pool MoE — research use |

For 8–12 GB VRAM, don't switch variants — pick **Low VRAM in the Hardware dropdown** instead (`PRESET=low_vram` from the CLI). The current list of variants is whatever is in `configs/gui-methods/`.

> **Compatibility notes**
> - Adapter variants such as HydraLoRA require `cache_llm_adapter_outputs = true` (on by default).
> - `tlora` can be baked into the base DiT with `make merge` to produce a standalone ComfyUI checkpoint.

For per-variant options see [`docs/guidelines/training.md`](training.md) and the individual documents under `docs/methods/`.

---

## 9. Inference

To sample from the adapter you just trained, press **Test** in the GUI — it automatically picks the most recent checkpoint in `output/ckpt/` and writes results to `output/tests/`. For batch generation and fine-grained flag control the CLI is easier ([Appendix A](#appendix-a-cli-reference)).

Commonly used options:

| Option | Description |
|---|---|
| `--lora_weight` | Path to the trained adapter (multiple allowed) |
| `--lora_multiplier` | Adapter strength (0.0–1.5) |
| `--image_size H W` | Output resolution (e.g. `1024 1024`, `1024 1536`) |
| `--infer_steps` | Denoising steps (typically 20–50) |
| `--guidance_scale` | CFG strength (3.0–5.0 recommended) |
| `--sampler` | `er_sde`, `euler`, `dpm++`, … |
| `--seed` | Seed for reproducibility |
| `--spectrum` | Enable Spectrum acceleration |
| `--pgraft` | P-GRAFT (LoRA cutoff late in denoising) — the base model handles late detail |

The full option list is in [`docs/guidelines/inference.md`](inference.md).

---

## 10. Deploying to ComfyUI

ComfyUI core supports the Anima base DiT natively (load it with `UNETLoader` / `CLIPLoader`). Deployment differs by adapter type.

### 10.1 Classic LoRA / OrthoLoRA / T-LoRA

Copy the `.safetensors` from `output/ckpt/` into `ComfyUI/models/loras/` and use ComfyUI's stock LoraLoader node.

For a cleaner standalone checkpoint, bake it into the base DiT from the GUI's **Merge** tab (or `make merge`, [Appendix A](#appendix-a-cli-reference)). The resulting `*_merged.safetensors` loads as a standalone model in `UNETLoader`.

### 10.2 HydraLoRA / Postfix

These carry routing and token insertion rather than a plain weight delta, so the stock LoraLoader cannot load them — they need dedicated nodes:

- **Anima Adapter Loader** — <https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter> (unified LoRA / Hydra / postfix handling)
- **Spectrum KSampler / Mod Guidance / DCW nodes** — <https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler>

---

## 11. Updating

Run this inside the install folder to pull the latest GitHub release and sync dependencies (no git required).

```bash
python tasks.py update              # Apply
python tasks.py update -- --dry-run # Preview which files change
```

Updating never touches `image_dataset/`, `post_image_dataset/`, `output/`, or `models/`, and asks before overwriting config files you have modified.

---

## Appendix A. CLI Reference

Everything the GUI does is also available from the CLI. `make <target>` and `python tasks.py <target>` are equivalent, and since the one-line install does not install `make`, **use `python tasks.py` if `make` is missing** (or `winget install ezwinports.make`).

> **Getting `ModuleNotFoundError`?** That's the venv. Either prefix commands with `uv run` (`uv run python tasks.py lora`) or activate the venv once per terminal — see [Appendix B](#appendix-b-manual-install-uv--git).

**Model download**

```bash
hf auth login                # Same token cache as the GUI sign-in
make download-models         # DiT + text encoder + VAE + SAM3 + MIT
make download-anima          # Retry pieces individually if a download breaks
make download-sam3           # Run separately once SAM3 access is approved
make download-mit
```

**Preprocessing**

```bash
make preprocess              # All three steps
make preprocess-resize       # 1) image_dataset/ → post_image_dataset/resized/
make preprocess-vae          # 2) VAE latent caching
make preprocess-te           # 3) Text embedding caching
make preprocess-pe           # (Optional) PE vision features — CMMD validation only
make mask                    # Generate masks for masked loss (SAM3 + MIT)
make mask-clean              # Delete post_image_dataset/masks/
make autotag --image <path>  # Print the predicted caption for one image
make curate-group            # Group similar images → post_image_dataset/groups/groups.json
```

**Training**

```bash
make lora-gui GUI_PRESETS=tlora                  # Recommended combo (same per-variant files as the GUI)
PRESET=low_vram make lora-gui GUI_PRESETS=tlora  # 8–12 GB VRAM
make lora-gui GUI_PRESETS=lora                   # Plain LoRA
make lora-gui GUI_PRESETS=hydralora              # MoE multi-head
make exp-chimera                                 # ChimeraHydra (experimental)

make lora                                        # Toggle-block style (configs/methods/lora.toml)
PRESET=half make lora                            # Half the dataset, for quick experiments

make lora -- --network_dim 32 --max_train_epochs 24   # Override any key
make daemon-terminate                            # Kill the running job + stop the daemon (frees the GPU)
```

**Inference**

```bash
make test                        # Plain LoRA / OrthoLoRA / T-LoRA
make test SPECTRUM=1             # Spectrum acceleration
make test MOD=1                  # Modulation guidance (composes with SPECTRUM=1)
make test NOLORA=1               # Base DiT only
make test-hydra                  # HydraLoRA (router-live)
make test-merge                  # Inference with a baked standalone DiT
```

Manual inference:

```bash
python inference.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --lora_weight output/ckpt/anima_lora.safetensors \
    --lora_multiplier 1.0 \
    --prompt "masterpiece, best quality, an anime girl in a sunlit forest" \
    --negative_prompt "worst quality, low quality, blurry" \
    --image_size 1024 1024 \
    --infer_steps 30 \
    --guidance_scale 4.0 \
    --sampler er_sde \
    --flow_shift 1.0 \
    --seed 42 \
    --save_path output/tests
```

**Merge · GUI · update**

```bash
make merge ADAPTER_DIR=output/ckpt                 # Bake the latest weights into the base DiT
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8  # Adjust strength
make gui                                           # Launch the GUI
make gui-shortcut                                  # Create a Windows desktop shortcut
make update                                        # = python tasks.py update
```

`make help` lists every target.

---

## Appendix B. Manual Install (uv · git)

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

`uv sync` creates **an isolated Python environment in `.venv/`** inside `anima_lora/` and installs everything there. Your system Python is untouched, and you never need to run `pip install` yourself.

**The most common beginner trap**: opening a fresh terminal and running `python tasks.py ...` uses the *system* Python and fails with `ModuleNotFoundError`. Fix it either way:

- **Prefix with `uv run`** (no activation needed, works anywhere): `uv run python tasks.py lora`
- **Activate the venv** (once per terminal window):

  ```powershell
  .venv\Scripts\activate        # Windows
  ```
  ```bash
  source .venv/bin/activate     # Linux / macOS / WSL
  ```

  A `(anima_lora)` or `(.venv)` prefix in the prompt means it worked; `deactivate` to exit. In VSCode, pick the `.venv` interpreter once (Command Palette → *Python: Select Interpreter*) and the integrated terminal activates it automatically.

---

## Appendix C. Manual CUDA Install

The one-liner in [§2](#2-installation) handles CUDA 13.2 automatically and skips the step if it's already installed. This appendix is **only for when that fails**.

- CUDA 13.x requires NVIDIA driver **595 or newer**. If you already have a newer driver, keep it (deselect the bundled driver in the Linux `.run` installer menu).
- To manage CUDA yourself, set `$env:ANIMA_SKIP_CUDA='1'` (PowerShell) / `ANIMA_SKIP_CUDA=1` (shell) to skip the automatic CUDA step.

Manual install: get 13.2 from the NVIDIA archive at <https://developer.nvidia.com/cuda-13-2-0-download-archive>

1. Windows: choose **Operating System: Windows → Architecture: x86_64 → Version: 11/10 → Installer Type: exe (local)**, run it, and pick "Express (Recommended)".
2. Verify with `nvidia-smi` and `nvcc --version` (expect `release 13.2`).
3. If `nvcc` isn't found, add these to the system `Path`, reboot, and check again:

   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\libnvvp
   ```

---

## Appendix D. Other Settings

### `num_repeats` (summary: **leave it alone**)

A kohya-ss style option in `configs/base.toml`'s `[[datasets.subsets]]` specifying how many times each image is used per epoch. It shows up in a lot of other trainer guides, but —

- **In this guide's workflow, leave it at `1`.** With all images in one folder, raising it only *lengthens each epoch* — the effect is identical to raising `max_train_epochs`. Every preset and method config in this project is tuned assuming `num_repeats = 1`.
- **It only makes sense as a balancing tool** when a run has multiple subsets (folders) with very different image counts (e.g. Character A with 1000 images + B with 50: set only the B subset to `num_repeats = 20`).
- It's a dataset setting, so it is not exposed in the GUI or method files. If you really need it, edit `configs/base.toml` (or the TOML given via `--dataset_config`) directly.

---

## Further Reading

- [`docs/guidelines/base-config.md`](base-config.md) — per-key reference for `base.toml` (model paths, noise schedule, caching, compile, memory knobs, dataset blueprint)
- [`docs/guidelines/training.md`](training.md) — adapter variants, caption shuffling, masked loss, dataset config details
- [`docs/guidelines/inference.md`](inference.md) — inference workflow, flags, Spectrum, prompt file format
- [`docs/guidelines/difference_between_comfy.md`](difference_between_comfy.md) — anima_lora ↔ ComfyUI core implementation differences
- [`docs/methods/timestep_mask.md`](../methods/timestep_mask.md) — T-LoRA timestep mask
- [`docs/methods/svd-down-lora.md`](../methods/svd-down-lora.md) — SVD-Down LoRA (the OrthoLoRA family)
- [`docs/inference/spectrum.md`](../inference/spectrum.md) — how Spectrum acceleration works and its options
- [`docs/inference/mod-guidance.md`](../inference/mod-guidance.md) — modulation guidance
- [`docs/methods/hydra-lora.md`](../methods/hydra-lora.md) — HydraLoRA multi-head routing

Questions and bug reports are welcome on GitHub Issues. Happy training!
