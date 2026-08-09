# Anima LoRA Guidebook

This document is a comprehensive English guide for using the **Anima LoRA** training/inference pipeline from start to finish. It covers everything from CUDA driver installation to dataset preparation, training, inference, and ComfyUI deployment. This guide is written for Windows beginners; for WSL, Linux, and training optimization topics, please refer to other documents.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [CUDA 13.2 (handled by the installer)](#2-cuda-132-handled-by-the-installer)
3. [Python Environment and Repository Setup](#3-python-environment-and-repository-setup)
4. [Model Download (Hugging Face sign-in is in the GUI)](#4-model-download-hugging-face-sign-in-is-in-the-gui)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Preprocessing: Resize · Latents · Text Embedding Cache](#6-preprocessing-resize--latents--text-embedding-cache)
7. [Using the GUI](#7-using-the-gui)
8. [Running Training](#8-running-training)
9. [LoRA / Adapter Variant Selection Guide](#9-lora--adapter-variant-selection-guide)
10. [Inference](#10-inference)
11. [Deploying to ComfyUI](#11-deploying-to-comfyui)
12. [Updating](#12-updating)

---

## 1. System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| GPU | At least **RTX 3060 or higher — 2xxx series and below are not supported** | VRAM 16 GB or more |
| System RAM | 16 GB | 32 GB or more |
| Disk | 60 GB free | 200 GB or more (for cache + accumulated outputs) |
| OS | Windows 11 / Ubuntu 22.04+ | Ubuntu 24.04 (stable FA2/CUDA 13 builds) |
| Python | **Must be 3.13** | - |

---

## 2. CUDA 13.2 (handled by the installer)

**You normally don't need to do anything here.** The one-line installer in **§3** installs the **CUDA 13.2 toolkit automatically** when it isn't already present: it downloads the official NVIDIA installer, launches it (choose **"Express (Recommended)"**), waits for you to finish, then verifies that `nvcc --version` reports `release 13.2` before continuing. If 13.2 is already installed, it's detected and skipped.

> **Driver note**: CUDA 13.x requires NVIDIA driver **595 or higher** — this is the one piece you must supply yourself. The CUDA installer can install a matching driver, but if you already have a newer one, keep it (on Linux, deselect the bundled driver in the `.run` installer's menu). Update via GeForce Experience or the NVIDIA Download Center if needed.
>
> **If a reboot is requested** (common on Windows): reboot, then re-run the same one-liner — CUDA is detected and the install continues from there. Set `$env:ANIMA_SKIP_CUDA='1'` (PowerShell) / `ANIMA_SKIP_CUDA=1` (shell) to skip the CUDA step entirely if you manage CUDA yourself.

<details>
<summary>Manual install (only if the automatic step fails)</summary>

Download 13.2 from the NVIDIA archive: <https://developer.nvidia.com/cuda-13-2-0-download-archive>

1. On Windows, select **Operating System: Windows → Architecture: x86_64 → Version: 11/10 → Installer Type: exe (local)**, run it, and choose "Express (Recommended)".
2. Verify in PowerShell: `nvidia-smi` and `nvcc --version`.
3. If `nvcc` is not recognized, add to your system `Path`:

   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\libnvvp
   ```

   then reboot and verify `nvcc --version` again.

</details>

## 3. Python Environment and Repository Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management with Python 3.13.

### 3.0 One-line install (easiest, no git required) ✅

Recommended for beginners. Paste this single line into PowerShell — it installs `uv` if missing, **installs the CUDA 13.2 toolkit if missing** (see **§2**), downloads the latest release, runs `uv sync` (Python 3.13 + torch are resolved automatically), creates an **"Anima LoRA GUI" desktop shortcut**, and **opens the GUI for you**:

```powershell
irm https://raw.githubusercontent.com/sorryhyun/anima_lora/main/install.ps1 | iex
```

Installs into `.\anima_lora\` (override with `$env:ANIMA_DIR`; pin a version with `$env:ANIMA_VERSION='v1.4.0'`). **When it finishes, the GUI opens by itself** — from there:

1. **Sign in to Hugging Face** right in the GUI — the old `hf auth login` terminal step is built in now (see **§4**).
2. Open the **Models** dialog and download the DiT + text encoder + VAE.
3. That's it — preprocess, train, and merge all happen in the same window.

Re-launch later from the **"Anima LoRA GUI"** desktop shortcut (or `python tasks.py gui`).

> This path assumes GUI-centric use and does **not** install `make`. Most tasks are GUI buttons, so that's fine — but if you want to run `make ...` from the CLI, either run `winget install ezwinports.make` (§3.3) or use `python tasks.py` instead of `make`.
>
> Update later from inside the folder with `python tasks.py update` (release-tarball merge, no git needed).

Sections **§3.1–§3.3** below are the manual install for those who prefer `git clone` or want to understand each step.

### 3.1 Install `uv`

  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```

After installation, open a new shell and confirm `uv --version` prints output.

### 3.2 Clone the Repository

```bash
git clone https://github.com/sorryhyun/anima_lora.git
cd anima_lora
```

> This guide uses `anima_lora/` as the base for all paths. Run all commands from inside this directory.

### 3.3 Install Dependencies

```bash
winget install ezwinports.make
uv sync
```

`uv sync` reads `pyproject.toml`/`uv.lock`, then **creates a virtual environment (a self-contained Python install) in a `.venv/` folder inside `anima_lora/`** and installs every dependency into it. It does *not* touch your system Python, so nothing here pollutes other projects. This is normal — you do not need to `pip install` anything yourself.

> **This is the step most newcomers get stuck on.** After `uv sync` finishes, the packages live inside `.venv/`, not in your global Python. If you just open a terminal and run `python tasks.py ...` or `make lora`, you'll likely hit `ModuleNotFoundError` because that shell is still using the *system* Python, which doesn't have the dependencies. You have to point the commands at the `.venv/` interpreter. There are two ways to do that:

**Option A — `uv run` (no activation, works everywhere).** Prefix any command with `uv run` and it transparently uses `.venv/`:

```bash
uv run make lora              # or: uv run python tasks.py lora
uv run hf auth login
uv run make download-models
```

This is the most foolproof option — it never depends on your shell state, so it can't "forget" the environment.

**Option B — activate the venv once per terminal.** Activating puts `.venv/`'s `python` first on your `PATH` for that shell session, so plain `make ...` / `python tasks.py ...` work without the `uv run` prefix:

```powershell
.venv\Scripts\activate        # Windows (PowerShell / cmd)
```
```bash
source .venv/bin/activate     # Linux / macOS / WSL
```

You'll know it worked when your prompt shows a `(anima_lora)` (or `(.venv)`) prefix. **You must re-activate in every new terminal window** — activation does not persist. To leave it, run `deactivate`. VSCode users can select the `.venv` interpreter once (Command Palette → *Python: Select Interpreter*) and its integrated terminal will auto-activate.

> Throughout the rest of this guide, commands are written as plain `make ...` / `python tasks.py ...`, which assume you've **either** activated the venv (Option B) **or** are prefixing with `uv run` (Option A). If a command fails with `ModuleNotFoundError` or `command not found: make`, this is almost always the cause — activate the venv or add `uv run`.

---

> ## 🖥️ Prefer not to use the command line? Use the GUI.
>
> With dependencies installed, you can do almost everything from here on in the GUI instead of typing commands — **model download, preprocessing, training, dataset/caption browsing, and merging are all buttons in one window.** Most newcomers will be happiest this way:
>
> ```bash
> make gui          # (or: uv run make gui)
> ```
>
> You'll **sign in to Hugging Face right in the GUI** (see **§4** — no terminal step) before the download button can fetch models, and lay out your dataset per **§5**. After that, the GUI covers the rest. Sections §6–§11 document the equivalent CLI commands — read them if you want to understand or script what the GUI does, but you don't have to run them by hand. Full GUI walkthrough: **[§7 Using the GUI](#7-using-the-gui)**.

---

## 4. Model Download (Hugging Face sign-in is in the GUI)

Hugging Face sign-in is **built into the GUI** now — you no longer need to run `hf auth login` in a terminal.

### 4.1 Sign in (in the GUI)

1. Create a token with **read** permissions at <https://huggingface.co/settings/tokens>.
2. Launch the GUI (it opens automatically right after install) and paste the token into the **Hugging Face sign-in** field. It's saved to the standard Hugging Face cache, so the GUI and the CLI (`make download-models`) share the same login.

> **CLI-only?** You can still run `hf auth login` in a terminal instead — it's the same token cache.

### 4.2 Download Models

In the GUI, the **Models** dialog downloads everything below with one button. The CLI equivalent is:

```bash
make download-models
```

This downloads the following three items and organizes them under `models/`:

| File | Path |
|---|---|
| Anima DiT (diffusion model) | `models/diffusion_models/anima-base-v1.0.safetensors` |
| Qwen3 0.6B text encoder | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

The same command also pulls **SAM3** and **MIT**, which are only used by the optional masked-loss step (see §8.2). Masking itself is opt-in, and even within it either segmenter can be toggled off — feel free to ignore these checkpoints if you're not training with masked loss.

> **SAM3 is a gated model.** Its weights live in a gated Hugging Face repository, so before the download can succeed you must visit <https://huggingface.co/facebook/sam3>, click **Request access**, and **wait for approval** (granted by the repo owner — this can take anywhere from minutes to a few days). Until access is granted, the SAM3 download will fail with a 403/gated error. The three core models above (DiT, text encoder, VAE) are *not* gated, and `make download-models` continues past a blocked SAM3 — so a pending SAM3 request won't stop you from training without masked loss. Once approved, run `make download-sam3` to fetch it.

> **If the download is interrupted**: you can re-download individual components with targets like `make download-anima`, `make download-sam3`, or `make download-mit`.

---

## 5. Dataset Preparation

Anima LoRA uses an *image + same-name `.txt` caption sidecar* structure. Example layout of `image_dataset/`:

```
image_dataset/
├─ 00001.png
├─ 00001.txt
├─ 00002.jpg
├─ 00002.txt
├─ subfolder/
│  ├─ 00010.webp
│  └─ 00010.txt
└─ ...
```

### 5.1 Caption Writing Tips

- Following Anima's official guidelines, tag order is always [meta] [character] [series] [artist] [general]. For example:

```
absurdres, safe, 1girl, chitanda eru, hyouka, @channel (caststation), full body, serafuku, She is saying hi.
```

- Based on personal experimentation, quality tags such as absurdres, highres, and masterpiece are best omitted or kept to a minimum. Alternatively, once the officially released mod guidance is available, you can skip quality tags entirely.
- Place original images in `image_dataset/` (filenames are free — keep them in this location).
- **Don't want to caption by hand?** Use the built-in **Anima Tagger** to auto-generate captions in the correct Anima tag order. In the GUI's **Dataset** tab, select an image and click **Autotag** to fill its caption (see [§7.4](#74-dataset-tab-autotag--grouping)); from the CLI, `make autotag --image <path>` prints the predicted caption for one image (the tagger model is downloaded automatically on first use). Treat the result as a starting point — review and touch up the tags before training.

### 5.2 What is `num_repeats` and when should I touch it? (Summary: **leave it alone**)

Inside `configs/base.toml`'s `[[datasets.subsets]]` you'll see `num_repeats = 1`. This specifies **how many times each image is used per epoch** — a kohya-ss style option that appears frequently in other LoRA trainer guides.

- **In this guide's standard workflow, leave it at `1`.** When training with all images in a single `image_dataset/` folder, increasing `num_repeats` only *lengthens each epoch* — the effect is identical to increasing `max_train_epochs`. Adjusting training volume through epoch count is more intuitive, and all presets and method configs in this project are tuned assuming `num_repeats = 1`.
- **When does increasing it make sense?** Only as a *balancing tool* when a single run has *multiple subsets (folders)* with very different image counts — to boost the exposure frequency of smaller folders (e.g., Character A with 1000 images + Character B with 50 images: set only the B subset to `num_repeats = 20`). It does not apply to single-folder training.
- **Where do I change it?** `num_repeats` is a dataset setting, not a method setting, so it is not exposed in `configs/methods/`, `configs/gui-methods/`, or the GUI training tab. If you truly need to change it, edit `[[datasets.subsets]]` in `configs/base.toml` directly (or in a separate TOML specified via `--dataset_config <path>`). *If you simply want more training on the same images*, increase `max_train_epochs` rather than `num_repeats`.

---

## 6. Preprocessing: Resize · Latents · Text Embedding Cache

To optimize training speed and VRAM usage, **resize → VAE latent caching → text embedding caching** are done in advance.

```bash
make preprocess              # Run all three steps (for LoRA / standard training)
# Or step by step
make preprocess-resize       # 1) image_dataset/ → post_image_dataset/resized/
make preprocess-vae          # 2) VAE latent caching → post_image_dataset/lora/
make preprocess-te           # 3) Text encoder output caching → post_image_dataset/lora/
make preprocess-pe           # (Optional) PE-Core vision encoder feature caching — CMMD validation only
```

> **⚠️ Caches are reused — they are never automatically deleted.**
> `make preprocess` (and the GUI's *Preprocess* button) **reuses existing caches as-is**. The `.npz` / `_te.safetensors` / `_pe.safetensors` files inside `post_image_dataset/lora/` are *never overwritten or deleted* — only missing entries are processed. This makes re-running very fast and safe to interrupt.
>
> In other words, running `make preprocess` again with existing caches won't lose any data. Conversely, if you **change captions, the tokenizer, or resize options and need to regenerate from scratch**, you must manually delete the cache directory (`post_image_dataset/lora/` or `post_image_dataset/easycontrol/`) and re-run.

### 6.1 What the Resize Step Does

- Resizes images to the pixel alignment required by the VAE
- Automatically sorts images into fixed token buckets, two token-count families of *(H/16) × (W/16) = 4032 or 4200 patches* (each bucket fills its count exactly)
- Automatically excludes images that are too small (default: below 0.5 MP) and reports them
- Saves results as PNGs in `post_image_dataset/resized/`

### 6.2 Latent Caching

- Runs the VAE once on all resized images and saves the results to disk
- The VAE is not loaded onto the GPU during training, saving significant VRAM
- Cache location: `post_image_dataset/lora/{stem}_{WxH}_anima.npz`
- Script: `scripts/preprocess/cache_latents.py`

### 6.3 Text Embedding Caching

- Pre-computes Qwen3 0.6B + LLM adapter outputs
- If `use_shuffled_caption_variants = true`, also caches comma-shuffled caption variants (randomly selected during training)
- Cache location: `post_image_dataset/lora/{stem}_anima_te.safetensors`
- Captions are always read from the original `.txt` files in `image_dataset/` (not copied to the resized folder)
- Script: `scripts/preprocess/cache_text_embeddings.py`

### 6.4 PE Vision Feature Caching (Optional)

- Only needed for CMMD validation
- Pre-computes PE-Core-L14-336 vision encoder outputs so the vision encoder doesn't need to be loaded during training
- Cache location: `post_image_dataset/lora/{stem}_anima_pe.safetensors`

> **When do I need to regenerate caches?**
> - New images were added → just run `make preprocess` again (existing caches are kept, only new items are processed).
> - **Captions were modified** or **tokenizer/padding options were changed** → follow the ⚠️ instructions above and manually delete the cache directory (`post_image_dataset/lora/`) before re-running. A simple re-run *reuses existing caches*, so changes won't take effect.

---

## 7. Using the GUI

The PySide6-based GUI lets you edit configs, browse datasets, run preprocessing, start/monitor training, and merge LoRA — all in one window.

```bash
make gui
make gui-shortcut   # (Optional) Create a no-console-window .lnk shortcut on the Windows desktop
```

Main GUI tabs:

- **Training Config**: Select a LoRA family variant from the dropdown (recommended: `tlora` — Ortho + T-LoRA / others include `lora`, `hydralora`, etc.), pick your card in the **Hardware** dropdown (Default 16GB+ / Low VRAM 8GB — the `presets.toml` hardware presets), edit the training keys, then start training. Data scope (`sample_ratio`, `artists_shard`) is a plain form field in the Basic section.
- **Preprocess**: Run resize + VAE + text embedding caching in one shot.
- **Dataset**: Preview images/captions and edit captions directly. Also **auto-generates captions** with the Anima Tagger (the *Autotag* button) and **groups visually similar images** so duplicates and same-scene shots are easy to spot — see [§7.4](#74-dataset-tab-autotag--grouping).
- **Merge**: Bake a trained LoRA into the base DiT to produce a standalone ComfyUI checkpoint (supports base LoRA / OrthoLoRA / T-LoRA only).

GUI training internally calls `train.py`, so the same parameters can be reproduced identically on the CLI. The GUI reads `configs/gui-methods/<variant>.toml` (one clean file per variant, no toggle blocks), so the variant list in the GUI matches `make lora-gui GUI_PRESETS=<variant>` on the CLI. Check the current variant list with `ls configs/gui-methods/`.

### 7.1 Form Editing and Save Behavior

Training/preprocessing subprocesses re-read the variant TOML file from disk, so edits made in the form but not saved will not be reflected in training. The GUI handles this in two ways:

- **Change detection**: When any field (or the `+ Extra args` text box) is edited, the `Save` button turns orange and shows `Save *` — meaning *the disk file and the screen differ*. This clears when you press `Save` or re-select the variant to reload from disk.
- **Auto-save**: If you forget to save and click `Train` / `Preprocess` anyway, the current form values are automatically written to the variant file before the subprocess starts. What you see on screen is what gets trained. (`Test` runs inference on the last trained checkpoint and is not auto-saved.)

> To try a change without committing it, edit the form and then switch to a different variant and back — the form reloads from disk and your edits are discarded.

### 7.2 Auto-Resume (checkpointing_epochs)

If training is interrupted, clicking `Train` again **automatically resumes from the last saved checkpoint**. This is one of the most useful features for handling power cuts, OOM errors, and accidentally closed windows — it's enabled by default.

How to use it in the GUI:

- The **Training** group in the Training Config tab has a `checkpointing_epochs` field (default `2` in gui-methods variants, `4` in `methods/lora.toml`). The state is saved every N epochs, overwriting a single file, so disk usage doesn't grow.
- After an interruption, click `Train` again with the same variant — `auto-resuming from checkpoint at step N` in the log means it resumed successfully. No manual flags needed.
- When training finishes normally, the resume files are automatically deleted. The final output is `output/ckpt/<output_name>.safetensors`.
- **After changing the dataset or core settings (rank/LR/epoch count etc.)** and wanting a fresh start, manually delete `output/ckpt/<output_name>-checkpoint-state/` before clicking `Train`, otherwise training continues on top of the old state.

Detailed behavior is covered in [§8.6 Auto-Resume](#86-auto-resume-checkpointing_epochs), including the difference from `save_every_n_epochs`.

### 7.3 Stopping Training and Closing the GUI

Training does not run *inside* the GUI window — when you click `Train`, the job is handed to a small background **training daemon** that runs `train.py` as a detached process. This has two practical consequences:

- **The `Stop` button aborts the current training job.** The daemon keeps running and advances to the next queued job (if any), so stopping one run never tears down the queue. This is the same as `make daemon-kill` on the CLI. (`Stop` also cancels an in-progress `Test` or `Preprocess`, which *do* run inside the GUI.)
- **Closing the GUI does NOT stop training.** Because the job runs in the detached daemon, training keeps going after you close the window — handy for long runs you want to leave overnight. When you reopen the GUI (`make gui`), it automatically reconnects to the still-running job and you'll see `Re-attached to running job …` in the log, with the progress bar and output picking up live again. (This also surfaces jobs started from the CLI or the ComfyUI trainer node.)

> To fully shut training down — kill the active job *and* stop the daemon to free the GPU — use `make daemon-terminate` on the CLI. `Stop` alone leaves the daemon up.
>
> `Test` and `Preprocess` are the exception: they run as in-window subprocesses, so closing the GUI cancels them.

### 7.4 Dataset Tab: Autotag & Grouping

The **Dataset** tab is for getting your `image_dataset/` into shape before preprocessing — previewing images, writing captions, and tidying up the collection. Two helpers make this much faster.

**Autotag — auto-generate captions.** Select an image and click **Autotag** to run the **Anima Tagger**, which predicts tags in the correct Anima order (`[meta] [character] [series] [artist] [general]`) and fills them into the caption box. The first click downloads the tagger model automatically and may take a moment; after that the model stays loaded in the background so subsequent images tag almost instantly. A small status line shows whether the tagger is loading or ready.

- The tagger frees its GPU memory automatically before you start any other GPU work (grouping, preprocessing, or training), so you never have to unload it by hand.
- Autotag is a **starting point, not a final answer** — review the tags and fix mistakes (especially character/series/artist names) before training. See the caption tips in [§5.1](#51-caption-writing-tips).
- CLI equivalent for a single image: `make autotag --image <path>`.

**Grouping — cluster similar images.** Click **Group** to scan the dataset and cluster images that look near-identical or show the same scene/character. When it finishes, the image list folds these into collapsible **green group headers**, so duplicates, alternate versions, and near-twins sit together — making it easy to thin out redundancy or balance how much of each concept you keep.

- Grouping runs as a background job with its own progress bar; you can keep working while it runs.
- It compares images by *visual content* (not filenames or captions), and groups are computed **per top-level folder** (artist/character bucket) so different folders never merge together.
- Re-running is cheap — it reuses cached image features — so feel free to click **Group** again after adding images.
- CLI equivalent: `make curate-group` (writes `post_image_dataset/groups/groups.json`, which the GUI reads). Tighten or loosen the clustering with `ARGS="--match-frac-min 0.4 --cell-match-min 0.9"` (higher = stricter, fewer images per group).

> Both features operate on your **original** `image_dataset/` images and don't touch the preprocessing caches — run them before `make preprocess` to clean up captions and trim duplicates first.

---

## 8. Running Training

All training runs through TOML config files and HuggingFace Accelerate. The config merge order is `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`, with method settings winning over preset settings on overlap.

### 8.1 Quick Start

**The recommended starting point is OrthoLoRA + T-LoRA (the `tlora` variant).** This is the most balanced combination for stability, detail, and style preservation, and can be used as-is for typical character/style LoRA training.

```bash
# Recommended: Ortho + T-LoRA (gui-methods/tlora.toml)
make lora-gui GUI_PRESETS=tlora                  # Standard environment
PRESET=low_vram make lora-gui GUI_PRESETS=tlora  # VRAM 8~12 GB (GUI: Hardware dropdown)

# Other variants (configs/gui-methods/<variant>.toml — clean single files)
make lora-gui GUI_PRESETS=lora                   # Plain baseline LoRA
make lora-gui GUI_PRESETS=hydralora              # MoE multi-head routing

# Toggle-block style (select variant inside configs/methods/lora.toml directly)
make lora                          # presets.toml[default]
PRESET=low_vram make lora          # presets.toml[low_vram] — VRAM 8~12 GB
PRESET=half make lora              # Use half the dataset for quick experiments
```

> **Overriding keys from the CLI**: pass extra args like `make lora -- --network_dim 32 --max_train_epochs 24` (`tasks.py` works the same way).

### 8.2 Masked Loss (Excluding Text Bubbles)

On manga/comic-style data, excluding *speech bubbles and text regions* from the training loss produces much cleaner results.

```bash
make mask          # SAM3 + MIT (runs via temp dir) → post_image_dataset/masks/
make mask-clean    # Delete post_image_dataset/masks/
```

Output PNGs are black-and-white: **white (255) = train**, **black (0) = exclude**. Dataset subsets use `post_image_dataset/masks/` automatically if present, falling back in order to the legacy `masks/{merged,sam,mit}/` directories (so old-layout users still work). Missing mask files are simply ignored, so not generating them is perfectly fine.

### 8.3 Commonly Adjusted Settings (LoRA defaults)

| Parameter | Default | Description |
|---|---|---|
| `network_dim` | `32` | LoRA rank. Higher = more expressive, more parameters |
| `network_alpha` | `32` | LoRA scale (usually the same as `network_dim`) |
| `learning_rate` | `2e-5` | Learning rate. Can be lowered for Hydra |
| `max_train_epochs` | `4` | Increase as dataset size decreases |
| `save_every_n_epochs` | `2` (gui-methods) / `4` (methods) | Accumulative adapter weight save interval |
| `checkpointing_epochs` | `2` (gui-methods) / `4` (methods) | Resume-state save interval (single file, overwritten) |
| `caption_dropout_rate` | `0.1` | Replaces some captions with empty strings (helps CFG) |
| `use_shuffled_caption_variants` | `true` | Use comma-shuffled caption variants |

Variant toggles (`use_ortho`, `use_timestep_mask`, `use_moe_style`, `router_source`, etc.) are activated by uncommenting the relevant block in `configs/methods/lora.toml`, or by using the per-variant single-file from `configs/gui-methods/<variant>.toml`. **The recommended `tlora` variant has `use_ortho = true` + `use_timestep_mask = true` pre-enabled, giving you the OrthoLoRA + T-LoRA combination out of the box.**

### 8.4 What Happens During Training

1. Load text encoder → generate/verify cache → unload
2. Load VAE → generate/verify cache → unload
3. *Lazily* load DiT to avoid VRAM conflicts with the caching stages
4. Patch adapter network into the DiT's attention / FFN modules (targets differ per variant)
5. Sample noise → DiT forward pass → flow-matching loss → backward → optimizer step
6. (Optional) Compute validation loss and generate sample images using `validation_split`

### 8.5 Outputs

- Trained weights: `output/ckpt/<output_name>.safetensors` (auto-differentiated per variant as `anima`, `anima_tlora_ortho`, `anima_hydra`, `anima_postfix`, etc.)
- Checkpoints: saved to `output/ckpt/` every `save_every_n_epochs` (`.snapshot.toml` sidecar + `_moe` companion for Hydra)
- Validation samples: `output/ckpt/sample/`
- Inference output images: `output/tests/`

### 8.6 Auto-Resume (checkpointing_epochs)

If training is interrupted, **it can automatically resume from the last saved point**. This is extremely useful for power cuts, OOM errors, accidentally closing the window, or pausing to shut down for the night. It is already enabled in the default method files; no additional configuration needed.

```toml
checkpointing_epochs = 2     # Save resume state every 2 epochs (overwrite)
```

Difference from `save_every_n_epochs`:

| Key | Saves | Accumulates? | Purpose |
|---|---|---|---|
| `save_every_n_epochs` | Adapter weights (e.g., `anima_lora-000004.safetensors`) | **Yes** (or limited with `save_last_n_epochs`) | Compare intermediate results or find the overfitting point |
| `checkpointing_epochs` | Full resume state (optimizer / scheduler / RNG / adapter weights) | **No — overwrites a single file** | Automatic resume after interruption |

How it works:

- **Auto-save**: Every `checkpointing_epochs` epochs, `output/ckpt/<output_name>-checkpoint-state/` (state directory) and `<output_name>-checkpoint.safetensors` (weights) are saved together, overwriting the previous version. Disk usage stays flat.
- **Auto-resume**: Running the same command again (`make lora` etc.) automatically continues training if a checkpoint exists and `max_train_steps` hasn't been reached. No `--resume` flag needed. `auto-resuming from checkpoint at step N` in the log confirms it worked.
- **Auto-cleanup**: When training finishes normally, both files are automatically deleted — the resume state is temporary, and the final artifact is `output/ckpt/<output_name>.safetensors` (see 8.5).
- **Manual resume**: To restore to a specific earlier state, you can pass `--resume <state_dir>` explicitly.

> **When to disable**: for short experimental runs or when disk space is tight, comment out `checkpointing_epochs`. For serious training on larger datasets, it's almost always worth keeping enabled.
>
> **Warning**: if you change the dataset, captions, or core settings (rank/LR/epoch count etc.), resuming from an old checkpoint is meaningless or harmful. Delete `output/ckpt/<output_name>-checkpoint-state/` manually and start fresh.

---

## 9. LoRA / Adapter Variant Selection Guide

> **🌟 Recommended**: if you're new or making a typical character/style LoRA, start with **`tlora` (OrthoLoRA + T-LoRA)**. It offers the best balance of detail, style preservation, and training stability.

| Variant | How to Run | When to Use |
|---|---|---|
| **OrthoLoRA + T-LoRA** ⭐ | `make lora-gui GUI_PRESETS=tlora` | **Recommended.** SVD-based orthogonal rotation (OrthoLoRA) + timestep rank masking (T-LoRA) stacked. Produces `anima_tlora_ortho.safetensors` |
| **OrthoLoRA + T-LoRA (8 GB)** | `PRESET=low_vram make lora-gui GUI_PRESETS=tlora` (GUI: Hardware → Low VRAM) | Same recommended combination on VRAM 8~12 GB |
| **Plain LoRA** | `make lora-gui GUI_PRESETS=lora` or `make lora` | Simplest baseline, for comparison experiments |
| **Plain LoRA (8 GB)** | `PRESET=low_vram make lora-gui GUI_PRESETS=lora` or `PRESET=low_vram make lora` | VRAM 8~12 GB |
| **HydraLoRA** | `make lora-gui GUI_PRESETS=hydralora` (8 GB: `PRESET=low_vram`) | MoE multi-head routing; fit multiple concepts in one adapter |
| **ChimeraHydra** *(experimental)* | `make exp-chimera` or `make lora-gui GUI_PRESETS=chimera_hydra` | Content/frequency dual-pool MoE — for research |

For detailed options per variant, see [`docs/guidelines/training.md`](training.md) and the individual docs under `docs/methods/`.

> **Compatibility notes**
> - HydraLoRA and similar adapter variants require `cache_llm_adapter_outputs = true` (enabled by default) to work correctly.
> - The OrthoLoRA + T-LoRA portions of `tlora` can be baked into the base DiT with `make merge` to produce a standalone ComfyUI checkpoint.

---

## 10. Inference

### 10.1 Quickest Test

To immediately generate samples with the adapter you just trained, use the matching `make test-*` command. All of them automatically pick the most recently saved adapter from `output/ckpt/`.

```bash
make test                        # Plain LoRA / OrthoLoRA / T-LoRA
make test SPECTRUM=1             # Spectrum-accelerated inference
make test MOD=1                  # Modulation guidance (pooled_text_proj) — composable with SPECTRUM=1
make test NOLORA=1               # Base DiT only (omits --lora_weight); combine with MOD=1 for mod-only path
make test-hydra                  # HydraLoRA (router-live, anima_hydra*_moe.safetensors)
make test-merge                  # Inference with baked standalone DiT (*_merged.safetensors)
# Experimental inference
```

### 10.2 Manual Inference

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

Commonly used flags:

| Flag | Description |
|---|---|
| `--lora_weight` | Path to the trained adapter. Multiple adapters can be provided |
| `--lora_multiplier` | Adapter strength (0.0–1.5) |
| `--image_size H W` | Output resolution (e.g., `1024 1024`, `1024 1536`) |
| `--infer_steps` | Number of denoising steps (typically 20–50) |
| `--guidance_scale` | CFG strength (3.0–5.0 recommended) |
| `--sampler` | `er_sde`, `euler`, `dpm++`, etc. |
| `--seed` | Seed for reproducibility |
| `--spectrum` | Enable Spectrum acceleration |
| `--pgraft` | P-GRAFT (LoRA cutoff in late denoising — base model handles late detail) |

For the full list of inference options and P-GRAFT usage, see [`docs/guidelines/inference.md`](inference.md).

---

## 11. Deploying to ComfyUI

ComfyUI core natively supports the Anima base DiT (`UNETLoader` / `CLIPLoader` work out of the box). Deployment differs by adapter type.

### 11.1 Classic LoRA / OrthoLoRA / T-LoRA

Copy the `.safetensors` file from `output/ckpt/` directly to `ComfyUI/models/loras/` — the standard ComfyUI LoraLoader node will work immediately. For a cleaner standalone checkpoint:

```bash
make merge ADAPTER_DIR=output/ckpt                 # Bake the latest weights into the base DiT
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8  # Adjust strength
```

The baked `*_merged.safetensors` can be loaded as a standalone model with ComfyUI's `UNETLoader`.

### 11.2 HydraLoRA / Postfix

These variants cannot be loaded with ComfyUI's default LoraLoader (they involve routing and token insertion, not simple weight deltas) and require dedicated nodes:

- **Anima Adapter Loader** (`https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter`) — unified handling for LoRA / Hydra / postfix. See the `README.md` in that folder for usage details.
- **Spectrum KSampler / Mod Guidance / DCW nodes** — separate repository at <https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler>

---

## 12. Updating

```bash
make update              # Fetch the latest release from GitHub and apply it + auto-run uv sync
make update -- --dry-run # Preview which files would change
```

`update` does not touch `image_dataset/`, `post_image_dataset/`, `output/`, or `models/`, and will prompt you on config file conflicts.

---

## Further Reading

- [`docs/guidelines/base-config.md`](base-config.md) — `base.toml` key-by-key reference (model paths, noise schedule, caching, compile, memory knobs, the dataset blueprint)
- [`docs/guidelines/training.md`](training.md) — Adapter variants, caption shuffling, masked loss, dataset config details
- [`docs/guidelines/inference.md`](inference.md) — Inference workflows, flags, Spectrum, prompt file format
- [`docs/guidelines/difference_between_comfy.md`](difference_between_comfy.md) — Implementation differences between anima_lora and ComfyUI core
- [`docs/methods/timestep_mask.md`](../methods/timestep_mask.md) — T-LoRA timestep mask
- [`_archive/methods/psoft-integrated-ortholora.md`](../../_archive/methods/psoft-integrated-ortholora.md) — OrthoLoRA details (the orthogonal rotation part of the recommended `tlora` variant; archived — superseded by [SVD-Down LoRA](../methods/svd-down-lora.md) in the showcased variant lineup)
- [`docs/inference/spectrum.md`](../inference/spectrum.md) — Spectrum acceleration: how it works and options
- [`docs/inference/mod-guidance.md`](../inference/mod-guidance.md) — Modulation guidance
- [`docs/methods/hydra-lora.md`](../methods/hydra-lora.md) — HydraLoRA multi-head routing

Questions and bug reports are welcome on GitHub Issues. Happy training!
