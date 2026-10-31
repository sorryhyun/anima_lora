# MOSS-Image LoRA

**Not Anima.** [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
(OpenMOSS, Apache-2.0) is a 5B single-stream DiT on a frozen Qwen3.5-9B text encoder and
the FLUX.2 VAE. This repo trains LoRAs for it and samples it with Spectrum on a 16 GB
card. It is a separate line (`library/moss/`): Anima configs, presets, `make lora`,
caches and LoRAs do not apply to it, and a MOSS LoRA does not load on Anima.

The measurements below come from one style set (caststation, 30 images of one artist).
Step-by-step GUI walkthrough, field tables and troubleshooting:
`docs/guidelines/moss-guidebook.md`.

## Quickstart

```bash
make download-model ARGS=moss-base   # DiT + text encoder + VAE + tokenizer, ~27 GB
make download-model ARGS=moss-sft    # optional: the SFT DiT alone, ~10 GB (reuses Base's TE/VAE)
make gui-moss                        # Preprocess → Train → Test, each a daemon job
```

Weights land in `models/moss_image/MOSS-Image-{Base,SFT}/` (elsewhere:
`ANIMA_MOSS_MODEL_DIR` in `.env`, or `model_dir` in the GUI). The GUI caches text
embeddings and latents for a folder of images with `.txt` captions (default
`post_image_dataset/resized`, the Anima resized tree; point `src` at one subfolder to train
on one set), trains a LoRA on the cache, and renders a prompt with and without the LoRA at
the same seed. It shares its code with `make gui-qwen` (`gui/lines/`).

The same jobs from the CLI, through the daemon:

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'post_image_dataset/moss/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors"
```

`--help` on each script lists every flag (one definition: `library/moss/requests.py`).
`--config <file.toml>` sets any flag from a flat TOML of field names
(`configs/moss/cache.toml` and `configs/moss/lora.toml` hold the shipped defaults; copy one
for a recipe). Order: field default → `--config` → CLI flags. Unknown keys, wrong types and
values outside a flag's choices are an error.

## Text encoder precision

`--text_encoder` (cache and generate; GUI: Advanced) picks the Qwen3.5-9B weights. The
tokenizer always comes from Base. A quantized file downloads on first use, or via
`make download-model ARGS=moss-te-int8` (`moss-te-fp8`).

| Value | File | Size | Notes |
|---|---|---|---|
| `int8` (default) | [`sorryhyun/MOSS-Image-Qwen3.5-9B-int8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-int8) | 9.2 GB | W8A8; all 32 layers on a 16 GB card, peak 6.5 GiB |
| `fp8` | [`sorryhyun/MOSS-Image-Qwen3.5-9B-fp8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-fp8) | 9.2 GB | float8 weights, bf16 matmul; ~1.7× slower to encode |
| `bf16` | Base's shards | 16.8 GB | the reference; peak 13.0 GiB |

Both quantized files are GPTQ-calibrated on MOSS prompts and pages. Their embeddings differ
from bf16 by ~4–5 % (relative L2) on captions and ~10 % on long JSON prompts and on pages
through the vision tower; fp8 images on the same seed keep composition and text and change
in small details (`_archive/bench/moss_te_fp8`). **Cache and generate with the same setting**: the TE
cache records which encoder wrote it and switching re-encodes it, so a cache made before
int8 became the default re-encodes on the next `cache` run unless it passes
`--text_encoder bf16` (or `text_encoder = "bf16"` in a copy of `cache.toml`).

## Training

- **Defaults**: `--rank 32`, alpha 128 for `lora` (scale 4; `lokr` / `loha` use
  alpha = rank), `--optimizer muon`, `--lr 5e-5`, `--down_init kaiming`,
  no channel scaling, `--ema_window 0.1` (an EMA over 10 % of the updates, t2i and i2i;
  the saved file is the EMA, the raw weights land in
  `<name>/<name>_raw`, `--ema_extra` adds more spans, `--ema_card` keeps them on the card; `docs/moss/findings/ema_lr1e4.md`). Why kaiming and no channel scaling under Muon:
  `docs/moss/findings/muon_channel_init.md`, `muon_no_channel_scaling.md`. Channel scaling
  and `dc_isolate`, the AdamW-era fixes for the loud FFN input channel, were removed once
  Muon made them no-ops; mechanism and the lr / alpha history:
  `docs/moss/channel_scaling.md`.
- **Time**: 30 images × 20 epochs (600 steps) took about 18–21 min on a 5070 Ti at the
  bf16 plan's then 1.9 s/step (1.66 now; the int8 plan runs 1.07). The
  LoRA moves line work and palette toward the set; character identity holds, including
  characters the set does not contain.
- **Memory**: the default plan is the int8 base (`--int8 stream`, GPTQ forward copies,
  `--stream fp32`), `--blocks_to_swap 0`, `--activation_memory_budget 0.1`, `--ema_card`,
  no checkpointing: 1.07 s/step at a 10.78 GB peak on 1024² samples, plus 4.4 GiB of pinned
  host memory. The file downloads on first use, or via `make download-model
  ARGS=moss-dit-int8`. One LoRA has completed on this plan (848 steps), with no A/B
  against bf16 yet: `docs/moss/optimization/int8.md`.
- **bf16 plan** (`--config configs/moss/lora_bf16.toml`): needed for `--algo loha` and the
  i2i configs, since the int8 base refuses LoHa and has run no i2i step. `--blocks_to_swap 36`, `--activation_memory_budget 0.5`, `--swap_h2d_only`: 1.66
  s/step at 1024² with 0.85 GB free on the largest caststation sample (4.2k image + 0.4k
  text tokens). The step time stops improving past budget ~0.45 (0.3: 2.05 s/step), and
  swap 28–36 run at the same speed, so the swap count trades VRAM, not step time. The i2i
  configs run swap 38 / budget 0.175: reference tokens make their samples larger, and at
  swap 28 budget 0.2 runs out of memory. At ~8.3k joint tokens (sanitize) 38 / 0.175 runs
  3.79 s/step against 4.17 for 28 / 0.15. `--swap_h2d_only` keeps a page-locked CPU copy
  of each swapped block (up to 8.8 GB of RAM) and drops the device-to-host half of every
  swap: same numerics and VRAM, tied with the classic swap up to 38, 8% faster at 40
  (4.69 vs 5.09 s/step on gen 4 x16); `--no-swap_h2d_only` turns it off. `--grad_checkpointing --blocks_to_swap 0` fits
  without swap, slower. The best swap
  count depends on the PCIe link and GPU; `tune_swap.py` times each count on the cache's
  largest sample on this plan (about 4 min) and prints the flags to train with. Full plan
  table: guidebook §4.2.

  ```bash
  make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
  ```

- **`--down_init`**: `kaiming` (default) or `weight_svd`, which starts `down` on the base
  weight's top-r right singular vectors. Under AdamW `weight_svd` kept the artist's style
  better by eye; under Muon it was dropped because its basis feeds 4–6× more input DC into
  the adapter (output DC shift 1.24 vs 0.43 with kaiming, images about a seed change
  apart). A DC-free variant is proposed in `docs/moss/proposal/lora_dcfree_svd_init.md`.
  `--svd_slice k` (needs `--down_init weight_svd`) seeds from singular vectors
  `[k·r, (k+1)·r)` and `--svd_slice_count N` interleaves `{k, k+N, …}`, as in Anima's
  `svd_slice` (`docs/anima/methods/svd-down-lora.md`), so LoRAs on different slices start in
  orthogonal input subspaces. MOSS's targets are all ≥ 3072 wide, so rank 32 fits 96
  slices.
- **LyCORIS** (`--algo lokr` / `--algo loha`, from `lycoris-lora`): LoKr is the
  Kronecker product (`--factor` caps the small factor; a `--rank` at or above half the
  larger factor's side makes `w2` full and the scale 1, hence `--rank 100000 --factor 8`),
  LoHa the Hadamard product of two rank-r pairs. Files keep LyCORIS tensor names and load
  through the same `--lora`. LoHa needs the bf16 plan, and on it both need
  `--activation_memory_budget 0.1` with `--blocks_to_swap 24` or more at 1024² (swap 12 /
  budget 0.15 ran out of memory); `tune_swap.py` measures a LoRA only. LoKr runs on the int8
  base, but its peak there is unmeasured.
  `--down_init` applies to `lora` only. At lr 1e-4, LoKr f8 came close to the LoRA and LoHa
  r16 overshot (palette shift, costume leakage); LoHa at lr 5e-5 is unmeasured.
- **`--mod_rank`** (opt-in, 0 = off; `lora` only): also adapts `shared_block_modulation`,
  the one timestep → shift / scale / gate Linear all 42 blocks share, at its own rank
  (`--mod_alpha`, default `alpha · √(mod_rank / rank)`, Anima's adaln √r law). The
  block adapters' init and the training noise stay those of the same run without it.
  Stamped as `rank_pattern` in the file. On caststation (240 updates) and ama_mitsuki
  (848) at r16 the loss matched the run without it; the modulation moved < 1.2 % (shift
  only), and on ama_mitsuki its shift flips sign between low and high σ. Images differ
  mostly on prompts far from the set. On i2i (uncensor) it kept the area outside the edit
  visibly better, text most of all, with the edit a tie: `docs/moss/findings/mod_rank_i2i.md`.
  ComfyUI's LoRA loader patches it like the block adapters
  (`lora_unet_shared_block_modulation`; checked against the `moss-image` node's HF loader).
- **Checkpoints**: `--save_every_epochs N` also saves `<name>/<name>_eN.safetensors`
  beside the final file (the last epoch is the final file). A `.json` run report
  (per-step loss) lands beside the LoRA.
- **`--fit_report`** (off): trains the largest sample (longest caption) at step 1 and
  reports its peak against free VRAM, with a `--blocks_to_swap` suggestion.
- **Caption dropout** (`--caption_dropout`, default 0): keep 0. On caststation 0.05 and 0.1
  weakened the style under CFG, because the uncond prediction picks up the style and CFG
  cancels part of it (`_archive/moss_lora/caption_dropout.md`).
- **A LoRA trained on Base also works on SFT**: it removes most of SFT's glossy look and
  lands close to Base + LoRA.

## Captions

Anima-style tag captions train and prompt fine (MOSS was trained on natural language; long
or character-heavy captions are untested). By default (`caption_format` on the cache,
`prompt_format` on generate, both `layout_json`) the Anima caption is mapped onto MOSS's
layout-JSON prompt (`anime_tools.captions.layout_json`):

- rating / meta, characters, series and artists (with their `@`) get their own `summary`
  keys; the rating collapses to `sfw` (safe, sensitive) or `nsfw` (nsfw, explicit);
- pixel-describing meta tags (`cropped`, `scan`, …) and the remaining tags become
  `summary.caption`;
- each position clause becomes a `character` item with its relative position (never a box),
  each OCR line a `text` item. Only a `{stem}.layout.txt` from anime_tools' Export
  (`caption_format layout_json` with `combine_ocr`) gives text items an OCR `bbox` (MOSS's
  `abs_centralized` / `yxyx`) and the top-level `bbox_coord_space` / `bbox_format` keys; the
  cache's own conversion of a `.txt` has no boxes.

On caststation this kept the style strength of plain tags and leaked fewer of the set's
attributes onto other characters. A `layout_json` cache also stores the plain caption, used
on `plain_caption_rate` (default 0.2) of the steps. **Train and prompt in the same form**
(`text` on both for plain captions); switching re-encodes the TE cache. In ComfyUI the
**MOSS Caption Helper** node, `task = layout` ([MOSS Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit),
needs `anime_tools`) does the same conversion.

## Sampling

- **Sampler defaults** come from the reference pipeline: CFG 3, the uncond branch skips
  block 22, shift 5. Don't carry Anima habits over.
- **`--sampler er_sde`** (default; `euler` is the pipeline's ODE): Anima's ER-SDE-Solver-3
  on the same σ grid, same speed, composes with Spectrum. Sharper line work and denser
  detail than Euler, but drifts from some prompt details (sunset light, hair colour); a
  different sample at the same seed.
- **`--spectrum`** (on; `--no-spectrum` runs every step in full): 2.6× at 50 steps with
  the default `flex_window 0.5`, hard to tell apart from the full 50 steps at grid scale.
- **`--compile`** (on; `--no-compile` for eager): about 1.37× on top, with or without
  Spectrum (Spectrum + compile 20 s vs plain eager 72 s at 1024²), for a few seconds of
  compile per job. Fine detail drifts from the eager image; composition does not.
- **DAVE** (`--dave auto`, off): more same-prompt variety across seeds; `docs/moss/dave.md`.
- **`--stream fp32`** (`bf16` default): the residual stream, modulation, norms, embedders
  and head run in fp32 around the bf16 block GEMMs. Prediction error against an fp32
  forward drops 2.4–3.5×, for +1.5% time and +0.6 GB; renders change in detail (a few
  /255, occasionally the composition on SFT) with neither precision consistently better.
  A LoRA loads under either, whatever it trained with (`train.py --stream fp32` exists;
  it needs ~0.25 GB more at the worst-case sample). The ComfyUI MOSS Toolkit loader has
  the same `stream` knob. `docs/moss/findings/bf16_residual_stream.md`.
- **CNS** has no `generate.py` flag. The six MOSS γ files
  (`networks/calibration/moss_cns_gamma*.npz`, `docs/cns.md`) are used by the
  bench and by the ComfyUI Spectrum node's `er_sde_cns`, which picks one by variant and cfg.

## ComfyUI

LoRA files use the kohya `lora_unet_` key layout and load through ComfyUI's LoRA loader on a
model from the `moss-image` loader nodes (files saved before 2026-10-17 used another layout
and do not). The Spectrum node pack's **KSampler (Spectrum for MOSS)** carries the sampler
defaults and Spectrum; `ComfyUI-Anima-DAVE` 0.2.0 and later runs DAVE on MOSS. The fp8 text
encoder loads in **MOSS-Image Load Qwen3.5** as-is (from `models/text_encoders/`); the INT8
file is in ComfyUI's `int8_tensorwise` + `convrot` format for the same loader.

## i2i (image to image)

An i2i LoRA learns a reference → target edit; the reference rides every forward as clean
latent tokens on RoPE's `r` axis. Recipes for `colorize` and `sanitize` live in
`configs/moss/i2i/<task>/`. Tasks, flags, commands, the sanitize run and its ComfyUI
workflow: `docs/moss/i2i.md`.

## Research flags

Advanced, not recommended: `--repa_weight` / `--repa_layer` / `--repa_pe_root` (relational
REPA, parked, `_archive/moss_lora/repa.md`); `--ref_timestep` / `--ref_isolate` /
`--caption_variant_rate` (set by the i2i recipes, `docs/moss/i2i.md`; probes in `project/finished/moss_lora/done/i2i.md`).

## More

- Channel scaling and `dc_isolate` (removed): `docs/moss/channel_scaling.md`. DAVE: `docs/moss/dave.md`. The i2i tone term: `docs/moss/tone_loss.md`.
- Findings (attention sink, vision-tower reference, σ staircase, Spectrum schedule):
  `docs/moss/findings/`.
- Massive channels (1356, 42) as a semantic carrier, from arXiv 2605.13974 (reading note,
  unmeasured): `docs/moss/proposal/massive_activation_semantics.md`.
- i2i design + probes: `project/finished/moss_lora/done/i2i.md`. The Anima-negatives sweep is archived (`_archive/proposals/moss_anima_negatives_revisit.md`).
- Run logs: `project/finished/moss_lora/` (`README.md`, sub-line logs, technical-report notes);
  settled logs (sampler, channel scaling, DAVE, seed soup) in `project/finished/moss_lora/done/`;
  parked lines (caption dropout, REPA) in `_archive/moss_lora/`. LoRA / sampler A/B grids:
  `bench/moss_ab/`; colorize and tone evals: `bench/moss_i2i_eval/`.
- Architecture walkthrough (TE pass, latent boundary, blocks, loss, sampler):
  `docs/moss/structure.md`. Code map: `library/moss/CLAUDE.md`.
