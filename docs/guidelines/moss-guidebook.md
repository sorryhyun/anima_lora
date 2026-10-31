# MOSS-Image LoRA Guidebook

A guide to training and testing LoRAs for MOSS-Image in the GUI. Open it with `make gui` and set the top-bar **Model** dropdown to **MOSS-Image** (`make gui-moss` opens the same panel in its own window). The panel is in English, or in Chinese when the GUI language is Chinese.

The MOSS line is **alpha**: one style set so far (30 images of one artist), with many A/Bs on it, and flags and defaults may still move. The numbers below come from those A/Bs and from benchmarks on a 16 GB RTX 5070 Ti.

---

## Table of Contents

1. [What MOSS-Image Is](#1-what-moss-image-is)
2. [Setup](#2-setup)
3. [Preprocess, Train, Test](#3-preprocess-train-test)
4. [Recommended Settings](#4-recommended-settings)
5. [Testing](#5-testing)
6. [Troubleshooting](#6-troubleshooting)

- [Appendix. CLI](#appendix-cli)

---

## 1. What MOSS-Image Is

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) (OpenMOSS, Apache-2.0) is a separate model from Anima, trained by this repo as its own line.

| | Anima | MOSS-Image |
|---|---|---|
| Diffusion model | Anima DiT | 5B single-stream DiT (42 blocks) |
| Text encoder | Qwen3 0.6B | Qwen3.5-9B |
| VAE | QwenImage VAE | FLUX.2 VAE |
| Caches | `post_image_dataset/lora/` | `post_image_dataset/moss/` |
| LoRAs | `output/ckpt/` | `output/ckpt_moss/` |
| Settings | variant TOML + Hardware preset | the panel's fields |

- Caches, settings and LoRAs do not cross: a MOSS LoRA does not load on Anima, and an Anima LoRA does not load on MOSS. Anima's variants, presets and `make lora` do not apply here.
- MOSS ships two DiTs. **Base** is the pretrained model; **SFT** is a fine-tune of it with a glossy, gacha-like look. Train on Base. A LoRA trained on Base also works on SFT: it removes most of SFT's gloss and lands close to Base + LoRA.

---

## 2. Setup

> Install the repo first with the shared [installation guide](../installation/README.md).

| Pack | Contents | Size |
|---|---|---|
| `moss-base` | DiT, text encoder, VAE, tokenizer | ~27 GB |
| `moss-sft` (optional) | the SFT DiT alone; reuses Base's text encoder and VAE | ~10 GB |

Download them in the GUI's Models dialog, or:

```bash
make download-model ARGS=moss-base
make download-model ARGS=moss-sft
```

Weights land in `models/moss_image/MOSS-Image-Base/` and `models/moss_image/MOSS-Image-SFT/`. To keep them elsewhere, set `ANIMA_MOSS_MODEL_DIR` in `.env`, or `model_dir` under **Advanced** in the panel (the folder that holds both `MOSS-Image-*` folders). The panel shows `Model not found: …` until the folder is in place.

The default caption format (`layout_json`, §3.1) also needs the danbooru tag CSV, which `make download-models` already fetches (`make download-model ARGS=danbooru_tags` on its own).

---

## 3. Preprocess, Train, Test

The panel has two tabs, **Preprocess** and **Train**; the **Test** box sits at the bottom of the Train tab. Less-used fields are folded under **Advanced**. Each button submits a job to the training daemon: switching the Model dropdown back to Anima or closing the GUI leaves it running, reopening re-attaches to it (`Re-attached to running job …`), and **Stop** cancels it.

### 3.1 Dataset and captions

Images with same-name `.txt` captions, subfolders included. The default source is `post_image_dataset/resized`, the Anima resized tree with its revised captions, so a dataset already preprocessed for Anima works as is. Point `src` at one subfolder to train on one set.

- **Anima-style tag captions work** for training and prompting. MOSS itself was trained on natural language; long or character-heavy captions are untested.
- **`layout_json` (default)**: the Anima caption is mapped onto MOSS's layout-JSON prompt. Rating/meta, characters, series and artists get their own keys, the other tags become the caption, each position clause becomes a subject with its position, and each OCR text line becomes a text item. On the 30-image set this kept the style as strong as plain tags and leaked fewer of the set's attributes onto other characters.
- **Train and prompt in the same format**: `caption_format` (Preprocess) and `prompt_format` (Test) must match — both `layout_json`, or both `text` for plain captions. Prompts are typed in Anima tag format either way; `layout_json` converts them.
- **The trigger is the artist tag as it appears in the captions**, e.g. `@channel (caststation)`. Put it in the prompt the same way.
- A natural-language prompt passes through `layout_json` as one caption string and behaves like the plain-text prompt.
- In ComfyUI, the **MOSS Caption Helper** node with `task = layout` ([MOSS Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit) pack) does the same conversion in front of the text encoder.

### 3.2 Preprocess

Caches the Qwen3.5-9B text embedding and the FLUX.2 VAE latent of every image, so neither model loads during training.

| Field | Default | What it does |
|---|---|---|
| `src` | `post_image_dataset/resized` | Image + caption folder |
| `out` | `post_image_dataset/moss` | Cache folder: two files per image, flat |
| `resolution` | `1024` | Pixel-area cap: at most ~1024² pixels at the image's own aspect, both sides multiples of 16, never upscaled |
| `caption_format` | `layout_json` | `layout_json` or `text` (§3.1) |

Press **Run preprocessing**. The status line counts images, captions and cached files. Re-running is incremental: new images are added, captions edited since their cache are re-encoded (the status line says how many), and a `caption_format` change re-encodes the text cache. Images that share a file name across subfolders get the folder name as a cache prefix.

### 3.3 Train

| Field | Default | What it does |
|---|---|---|
| `cache` | `post_image_dataset/moss` | The Preprocess output |
| `output` | `output/ckpt_moss/moss_lora.safetensors` | LoRA path; a `.json` run report (per-step loss) lands beside it |
| `model` | `base` | Which DiT to train on |
| `epochs` | `8` | Passes over the cache (one image per step) |
| `save_every_epochs` | `0` | Also save every N epochs, as `<name>/<name>_e10.safetensors` etc. beside the final file (the last epoch is the final file itself) |

Rank, alpha, lr and the adapter type are in §4. **Train** reads `cache`; **Preprocess, then train** runs Preprocess first and points `cache` at its `out`.

### 3.4 Test

Renders the prompt twice with the same seed: once with the LoRA at `lora_multiplier`, once without it. The two images appear side by side in the panel and differ by the LoRA only.

- `lora` empty = the Train tab's `output`. Set it to test another file, e.g. an `_e10` checkpoint.
- `model` = `sft` tests the Base LoRA on SFT.
- Output: `output/tests/<date-time>/lora/` and `…/base/`, each with a `manifest.json`. Every Test gets a new folder, so earlier runs stay.

---

## 4. Recommended Settings

The comparisons ran on 30 images of one artist (`@channel (caststation)`), Base, 20 epochs = 600 steps (about 18–21 minutes).

| Field | Default | Notes |
|---|---|---|
| `algo` | `lora` | `lokr` / `loha`: §4.1 |
| `rank` | `32` | 37.2 M parameters at rank 16 (the earlier default) |
| `alpha` | `128` for `lora` (scale 4 at rank 32); = rank for `lokr` / `loha` | scale = alpha / rank |
| `lr` | `5e-5` | 5 % linear warmup |
| `optimizer` | `muon` | `adamw` is the earlier default; `lokr` / `loha` always use AdamW |
| `down_init` | `kaiming` | `kaiming` or `weight_svd` |
| `epochs` | `8` | the measured run used 20 |
| `plain_caption_rate` | `0.2` | `layout_json` caches only: share of steps trained on the plain Anima caption instead of its layout JSON |

- **lr, channel scaling, DC isolation**: one input channel is far louder than the rest at the FFN input. Under AdamW two fixes kept the LoRA from spending its updates on a global shift through it (channel scaling and `dc_isolate`). Under the Muon default with a kaiming `down` init neither changed the trained LoRA, so both were removed. The earlier defaults and the comparisons behind each change: `docs/moss/channel_scaling.md`.
- **optimizer**: Muon (the optimizer MOSS was pretrained with) at the same lr moves the LoRA by the same amount as AdamW, spread over more directions, and looked better on the same seed. It also keeps the LoRA off that global shift on its own. `docs/moss/findings/muon_lora.md`.
- **down_init**: `kaiming` is the random init; `weight_svd` starts the LoRA's down projection on the base weight's top singular vectors. Those vectors read the input's constant offset strongly, and with Muon a `weight_svd` LoRA shifts every token's output about three times as far as a `kaiming` one. The images are about a seed change apart. `docs/moss/findings/muon_channel_init.md`.
- **epochs**: at 10 of 20 epochs the results were looser and more varied; at 20 they were cleaner and more uniform, with a mild drift toward frontal, centred upper bodies. Set `save_every_epochs` to compare both from one run.
- **sigma_\*** (Advanced): the defaults are MOSS's final pretraining noise schedule. Anima's schedule (`sigma_mu 0`, `sigma_s 1`, `sigma_cutoff 1.0`, `sigma_uniform 0`) gave the weakest transfer of every arm.

### 4.1 LoKr and LoHa

`algo` = `lokr` (Kronecker product) or `loha` (Hadamard product of two rank-r pairs), from LyCORIS. Files keep LyCORIS's tensor names and load through the same `lora` field. `down_init` applies to `lora` only.

| Setup | Fields | Speed (16 GB, 1024²) | Result on the artist set at lr 1e-4 |
|---|---|---|---|
| LoKr f8 | `rank 100000`, `factor 8` | 2.26 s/step, 3.6 GB free | Close to the LoRA, slightly less character in the faces |
| LoHa r16 | `rank 16` | 2.49 s/step, 1.7 GB free | Overshot: grey-teal palette shift, the set's costume on other characters |

- These were measured on the bf16 plan (§4.2), which LoHa requires: the default int8 base refuses it. LoKr runs on int8, but its peak there is unmeasured.
- On the bf16 plan both need more memory than a LoRA: set `activation_memory_budget 0.1` (Advanced) and keep `blocks_to_swap` at 24 or more. Swap 12 with budget 0.15 ran out of memory at 1024².
- LoKr with `rank` at or above half the larger factor's side (the `rank 100000` above) uses a full matrix for that factor and ignores alpha (scale 1).
- LoHa: lr 5e-5 is the next setting to try; it has not been measured.

### 4.2 Memory plan (16 GB)

Defaults: `grad_checkpointing` off, `compile` on, and under Advanced `int8 stream` with the GPTQ file (the DiT's block weights run as int8; the file downloads on first use), `stream fp32` (the rest of the block runs in fp32), `blocks_to_swap 0` (every block stays on the card), `activation_memory_budget 0.1` (how much of the forward pass is kept for the backward; the rest is recomputed), `ema_card` on and `expandable_segments` on (keeps the CUDA memory pool from fragmenting when sequence lengths vary).

This runs 1.07 s/step at a 10.78 GB peak on 1024² samples and also pins 4.4 GiB of system memory. On this plan 0.1–0.3 run at the same speed and each 0.1 adds about 1 GB of peak. At 0.05 the peak drops to 9.98 GB but attention is recomputed and the step slows to 1.16 s. No LoRA has been trained to completion on it yet ([details](../moss/optimization/int8.md)).

**bf16 plan.** The int8 base refuses LoHa and has run no i2i step, so the i2i task configs stay on bf16. `--config configs/moss/lora_bf16.toml`, or under Advanced `int8 off`, `stream bf16`, `ema_card` off, `blocks_to_swap 36` (DiT blocks kept on the CPU and streamed in), `activation_memory_budget 0.5`.

| Plan | Fields | s/step | Free VRAM |
|---|---|---|---|
| Earlier bf16 default | swap 12, budget 0.1 (rank 32) | 1.85 | ~1.7 GiB |
| Older bf16 default | swap 12, budget 0.15 (rank 16) | 1.77 | ~0.7 GiB |
| More room | swap 24, budget 0.2 | 1.78 | 2.26 GiB |
| Checkpointing | `grad_checkpointing` on, swap 0 | 2.09 | 1.60 GiB |

The bf16 plan (swap 36, budget 0.5, `swap_h2d_only`, `expandable_segments`, rank 32) runs 1.66 s/step on the caststation set with 0.85 GB free on its largest sample (4.2k image + 0.4k text tokens). Raising the budget stops helping past ~0.45 (0.3: 2.05 s/step), and swap 28–36 run at the same speed, so a lower swap only gives up free VRAM. i2i samples also carry a reference image, so the i2i task configs use swap 38, budget 0.175: at swap 28, budget 0.2 runs out of memory, and swap 38 makes room for the higher budget, 3.79 s/step against 4.17 for swap 28 / budget 0.15 at ~8.3k joint tokens (sanitize). `swap_h2d_only` keeps a page-locked copy of every swapped block in RAM (up to 8.8 GB) so a swap only copies to the card, never back. The results and VRAM are the same as without it; it is as fast up to swap 38 and 8% faster at swap 40. It is on by default and only acts when blocks are swapped; `--no-swap_h2d_only` turns it off.

Table: synthetic 1024² image + 300 text tokens (~4.4k tokens in all), LoRA rank 16, RTX 5070 Ti. The earlier default (rank 32, budget 0.1, swap 12) measured 1.85 s/step with ~1.7 GB free and a peak reserved of ~13.1 GB at ~4.5k joint tokens (image 4096 + text 455); rank 16 at budget 0.1 measured 1.82 s/step with ~2.0 GB free. Real samples ran up to ~4.9k tokens, and longer captions use more memory, which `swap 24` leaves room for at the same speed.

- Between swap 8 and 36 the speed barely moves (1.75–1.84 s/step), and budgets 0.1–0.3 barely change it either; swap 40 is slower (2.34 s/step) because the PCIe transfers stop hiding behind compute. Below that point, the swap count trades VRAM, not speed.
- The best pair depends on the PCIe link and the GPU. `scripts/moss/tune_swap.py` (CLI, [Appendix](#appendix-cli), about 4 minutes) times each swap count on the cache's largest sample on the bf16 plan and prints the fields to train with.
- `blocks_to_swap -1` sizes the swap from free VRAM; it is calibrated for LoRA and under-reserves for LoKr/LoHa.
- `compile` off needs `grad_checkpointing` on. Compiled training ran 2.0 against 3.1 s/step, for ~30 s of compile at the start.

---

## 5. Testing

The sampler defaults come from MOSS's reference pipeline; Anima's sampler habits do not carry over.

| Field | Default | What it does |
|---|---|---|
| `cfg` | `3.0` | CFG scale; `1.0` = no uncond pass |
| `uncond_skip` | `22` | The uncond pass is the model with block 22 skipped (still two passes per step); `''` = plain CFG |
| `shift` | `5.0` (Advanced) | Scheduler shift |
| `steps` | `50` | |
| `sampler` | `er_sde` | `er_sde` or `euler` |
| `width` / `height` | `1024` | Multiples of 16 |
| `negative_prompt` | empty | The pipeline default |
| `dave` | empty | `auto` = DAVE: more variety across seeds for one prompt (framing, pose, layout), with slightly looser hands; `docs/moss/dave.md` |
| `dave_strength` | `0.3` | How much DAVE removes; `dave_tau` (Advanced, `0.10`) = the share of early steps it runs on |

- **er_sde vs euler**: `euler` is the reference pipeline's sampler. `er_sde` costs the same and gives sharper line work and denser detail; with the LoRA, hands came out cleaner. It also drifts from some prompt details (sunset light, hair colour, "sitting on a bench") and gave up a little of the set's look. The two give different images at the same seed, so compare LoRA against base within one sampler.
- **Spectrum** (checkbox, on by default): most steps forecast the model's output instead of running all 42 blocks. At 50 steps with the default `spectrum_flex_window 0.5` (Advanced) it is 2.6× faster for +0.85 GiB, and hard to tell apart from the full 50 steps at grid scale. `0.25` is 2.2× and more conservative; `1.0` is 3.1× and loses fine detail.
- **compile** (Advanced, on): about 1.37× on top, with or without Spectrum, for a few seconds of compile per job. Composition is the same as uncompiled; fine detail (hair strands, patterns) differs, so compare compiled images with compiled images.

| Base, 1024², 50 steps | Time per image |
|---|---|
| Plain, uncompiled | 72 s |
| Plain, compiled | 52 s |
| Spectrum, compiled | 20 s |

- **ComfyUI**: a MOSS LoRA loads through ComfyUI's LoRA loader on a model from the `moss-image` nodes (files saved before 2026-10-17 do not). **KSampler (Spectrum for MOSS)** (Spectrum node pack) carries these sampler defaults and Spectrum, and `ComfyUI-Anima-DAVE` 0.2.0 or later runs DAVE on MOSS.

**Comparing**: the Test button already renders LoRA and no-LoRA at the same seed. To compare two LoRAs or epochs, keep the prompt, seed, sampler and compile setting the same and change only `lora`.

---

## 6. Troubleshooting

| Symptom | What to do |
|---|---|
| Out of memory in training | Advanced: `blocks_to_swap 20` (2.1 GB off the peak on the int8 plan, same speed), `activation_memory_budget 0.05` (1 GB off, 8% slower), or `grad_checkpointing` on (9.61 GB peak on the int8 plan, 16% slower). Then the bf16 plan (§4.2) with `blocks_to_swap 24`, or run `tune_swap.py`; on it LoKr/LoHa need 24 / 0.1 |
| `int8 base: loha builds W + ΔW` | The int8 base does not run LoHa: use the bf16 plan (§4.2) |
| Style too weak | More epochs (the measured run used 20). Check the prompt carries the artist tag spelled as in the captions, and that `prompt_format` matches `caption_format` |
| Style too strong, the set's signature or watermark shows up, every image has the same composition | Test an earlier `_e<N>` checkpoint or lower `lora_multiplier`. A higher effective lr is what started copying the signature; try `lr 2e-5` |
| The set's costume or colours leak onto other characters | Use `caption_format layout_json` (the default; it leaked less than plain tags). Switch LoHa back to LoRA or LoKr |
| Switched `caption_format` | Run Preprocess again: the text cache is re-encoded, latents are reused. The status line counts do not show the switch. Set `prompt_format` to match, then retrain |
| `layout_json needs the danbooru tag CSV` | `make download-model ARGS=danbooru_tags` |
| `No LoRA at …` on Test | Train first, or set `lora` in the Test box |

---

## Appendix. CLI

The GUI submits these jobs; they can also be run through the daemon directly. `--help` on each script lists every flag. `train.py --config configs/moss/lora.toml` reads flags from a TOML file (the shipped defaults; copy it to keep a recipe); flags given on the command line still win.

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'post_image_dataset/moss/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora_bf16.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors"
make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
```

LoKr / LoHa:

```bash
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo lokr --rank 100000 --factor 8 --config configs/moss/lora_bf16.toml --blocks_to_swap 24 --activation_memory_budget 0.1"
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo loha --rank 16 --lr 5e-5 --config configs/moss/lora_bf16.toml --blocks_to_swap 24 --activation_memory_budget 0.1"
```

More measurements: [`docs/moss/README.md`](../moss/README.md).
