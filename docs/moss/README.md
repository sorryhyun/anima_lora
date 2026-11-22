# MOSS-Image LoRA

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) (OpenMOSS, Apache-2.0)
is a 5B single-stream DiT on a frozen Qwen3.5-9B text encoder and the FLUX.2 VAE. This repo
trains LoRAs for it and samples it with Spectrum on a 16 GB card. It is a separate line
(`library/moss/`): Anima configs, presets, `make lora`, caches and LoRAs do not apply to it,
and a MOSS LoRA does not load on Anima.

The measurements below come from one style set (caststation, 30 images of one artist).
Step-by-step GUI walkthrough, field tables and troubleshooting:
`docs/guidelines/moss-guidebook.md`.

## Quickstart

```bash
make download-model ARGS=moss-base   # DiT + text encoder + VAE + tokenizer, ~27 GB
make download-model ARGS=moss-sft    # optional: the SFT DiT alone, ~10 GB (reuses Base's TE/VAE)
make gui                             # Model → MOSS-Image: Preprocess → Train → Test, each a daemon job
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
cache records which encoder wrote it and switching re-encodes it. A cache made before int8
became the default re-encodes on the next `cache` run unless that run passes
`--text_encoder bf16` (or `text_encoder = "bf16"` in a copy of `cache.toml`).

## Training

- **Defaults**: `--rank 32`, alpha 128 for `lora` (scale 4; `lokr` / `loha` use
  alpha = rank), `--optimizer muon`, `--lr 5e-5`, `--down_init kaiming`,
  no channel scaling, `--ema_window 0.1` (an EMA over 10 % of the updates, t2i and i2i;
  the saved file is the EMA, the raw weights land in `<name>/<name>_raw`, `--ema_extra` adds
  more spans, `--ema_card` keeps them on the card). Why
  these defaults: `docs/moss/findings/t2i_lora_recipe.md`. Removed options and the lr / alpha history:
  `project/finished/moss_lora/done/channel_scaling.md`.
- **Time**: 30 images × 20 epochs (600 steps) took about 18–21 min on a 5070 Ti at the
  bf16 plan's then 1.9 s/step (1.66 now; the int8 plan runs 1.05). The
  LoRA moves line work and palette toward the set; character identity holds, including
  characters the set does not contain.
- **Memory**: the default plan is the int8 base (`--int8 stream`, GPTQ forward copies,
  `--stream fp32`), `--blocks_to_swap 20`, `--activation_memory_budget 0.4`, `--ema_card`,
  no checkpointing (`configs/moss/lora.toml`). Swap 0 at budget 0.3 measured 1.05 s/step at a
  12.88 GB peak on 1024² samples, plus 4.4 GiB of pinned host memory. Below 16 GB run budget 0.1 (1.07 s/step, 10.78 GB). The file downloads on
  first use, or via `make download-model ARGS=moss-dit-int8`. Two LoRAs have completed on this
  plan: ama mitsuki, 848 steps (`docs/moss/optimization/int8.md`), and swap_text i2i, 3,000
  steps (`docs/moss/methods/i2i.md`). What each part of the plan buys:
  `docs/moss/optimization/training_step.md`.
- **bf16 plan** (`--config configs/moss/lora_bf16.toml`): needed for `--algo loha`, which the
  int8 base refuses. The i2i configs train on the int8 base except `sanitize`, whose bf16
  `train.toml` (swap 38 / budget 0.175) is the v3 recipe.
  `--blocks_to_swap 36`, `--activation_memory_budget 0.5`, `--swap_h2d_only`: 1.66 s/step at
  1024². Budget, swap and i2i measurements and
  the plan table: guidebook §4.2. `--swap_h2d_only` is 8% faster than the classic swap at 40
  (4.69 vs 5.09 s/step on gen 4 x16). `tune_swap.py` times each swap count on the cache's
  largest sample on this plan (about 4 min) and prints the flags to train with.

  ```bash
  make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
  ```

- **`--down_init`**: `kaiming` (default), `weight_svd` or `weight_svd_nodc`. `weight_svd`
  starts `down` on the base weight's top-r right singular vectors, a basis that under Muon
  feeds 4–6× more input DC into the adapter; `weight_svd_nodc` projects the calibrated
  input DC direction out of that basis (`docs/moss/findings/t2i_lora_recipe.md` §3).
  `--svd_slice k` (needs `--down_init weight_svd`) seeds from singular vectors
  `[k·r, (k+1)·r)` and `--svd_slice_count N` interleaves `{k, k+N, …}`, as in Anima's
  `svd_slice` (`docs/anima/methods/svd-down-lora.md`), so LoRAs on different slices start in
  orthogonal input subspaces. MOSS's targets are all ≥ 3072 wide, so rank 32 fits 96
  slices.
- **LyCORIS** (`--algo lokr` / `--algo loha`, from `lycoris-lora`): `--factor` caps LoKr's
  small factor; a `--rank` at or above half the larger factor's side makes `w2` full and the
  scale 1, hence `--rank 100000 --factor 8`. LoHa needs the bf16 plan. `--down_init` applies
  to `lora` only, and `tune_swap.py` measures a LoRA only. Memory settings and the lr 1e-4
  comparison: guidebook §4.1.
- **`--mod_rank`** (opt-in, 0 = off; `lora` only): also adapts `shared_block_modulation`,
  the one timestep → shift / scale / gate Linear all 42 blocks share, at its own rank
  (`--mod_alpha`, default `alpha · √(mod_rank / rank)`, Anima's adaln √r law). The block
  adapters' init and the training noise stay those of the same run without it. Stamped as
  `rank_pattern` in the file. In t2i the loss matched the run without it. On i2i (uncensor)
  it kept the area outside the edit visibly better, text most of all, with the edit a tie:
  `docs/moss/findings/mod_rank_i2i.md`. ComfyUI's LoRA loader patches it like the block
  adapters (`lora_unet_shared_block_modulation`).
- **Checkpoints**: `--save_every_epochs N` also saves `<name>/<name>_eN.safetensors`
  beside the final file (the last epoch is the final file). A `.json` run report
  (per-step loss) lands beside the LoRA.
- **Samples** (`--sample_prompts`, `--sample_every_epochs N`): the prompts (one per line,
  or a `.txt` / `.json` file) are rendered every N epochs and after the last one, into
  `<name>/samples/e<N>_<k>.png` (`--sample_at_first` adds `e0`, before the first step).
  They are written in the cache's caption form (`--sample_prompt_format auto`) and encoded
  once at the start, on the cache's text encoder and prompt tail. A render uses the
  weights the epoch save holds (the EMA once started): ER-SDE, CFG 3 with the block-22
  uncond skip, `--sample_steps 28` at `--sample_width` × `--sample_height` (1024²). Under `compile`
  the marked token range takes in the sample lengths, so the no-grad render graph compiles
  once. The VAE decodes on the CPU in the
  background (~15 s per 1024² image) while training goes on. Block swap is fine; not on an
  i2i cache, with `trigger_registers` or with `delta_rows`.
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

- meta, characters, series and artists (with their `@`) get their own `summary` keys;
- the rating tag (safe / general / sensitive / nsfw / questionable / explicit) is dropped
  before the TE, in every caption format, from the plain caption too and from `sfw` / `nsfw`
  in a layout-JSON `summary.meta` (training and generate alike). The reference prompts carry
  no rating, and on caststation (2026-11-14) a LoRA trained without it rendered with or
  without `sfw` in the prompt alike, as did one trained with it;
- pixel-describing meta tags (`cropped`, `scan`, …) and the remaining tags become
  `summary.caption`;
- each position clause becomes a `character` item with its relative position (never a box),
  each OCR line a `text` item with its OCR `bbox` (MOSS's `abs_centralized` / `yxyx`, on
  the cache canvas) and the top-level `bbox_coord_space` / `bbox_format` keys, when
  anime_tools' OCR tree (`post_image_dataset/ocr`) has the image's sidecar (`text_boxes`, on
  by default). A `{stem}.layout.txt` from anime_tools' Export is read as published instead.

On caststation this kept the style strength of plain tags and leaked fewer of the set's
attributes onto other characters. A `layout_json` cache also stores the plain caption, used
on `plain_caption_rate` (default 0.2) of the steps. **Train and prompt in the same form**
(`text` on both for plain captions); switching re-encodes the TE cache.

For a LoRA meant to be prompted with tags, `caption_format tags` (and `prompt_format tags` on
generate) keeps the tag caption and moves its flat tags into the USERGUIDE order: quality,
score, meta, year, character, series, artist, then the count tag and the general tags;
position and text clauses stay as written, except that each OCR'd line of a text clause gets
its box after the quote (`text_boxes`, on by default): `Japanese text reads as "こいつ…"
[202,126,280,154]`, the same `abs_centralized` `[y0, x0, y1, x1]` a layout-JSON text item
carries, on the cache canvas. The boxes come from anime_tools' OCR tree
(`post_image_dataset/ocr`, mirroring the resized tree, so the image must sit there or be a
symlink into it); a line with no sidecar entry stays bare. To place text at generation, write
the box the same way. Leave quality tags (`best quality, highres`) out
of the training captions: on caststation (2026-11-14) adding them to every caption weakened
the style (`proposal/tag_captions.md`, "Tag order"). In ComfyUI the
**MOSS Caption Helper** node, `task = layout` ([MOSS Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit),
needs `anime_tools`) does the same conversion.

**MOSS-format captions** (`scripts/moss/captions.py`, `CaptionsRequest`): the Anima caption
keeps a full shared tag bag beside its position clauses; the MOSS format keeps only the head
(rating, count, names, series, artist) and the tags no subject owns there, and puts every tag
a subject's crop carries into that subject's clause, repeated across clauses when several carry
it. It also cuts a tag a more specific one already says (`long hair` beside `very long hair`,
`bow` / `neckerchief` beside a bowtie). On caststation (2026-11-15) a LoRA trained on them beat
its control trained on the Anima captions, same recipe and `caption_format tags`. The job
(SAM3 + the Anima Tagger, a few minutes for 30 images) writes a tree of its own with the
images symlinked, `post_image_dataset/moss_captions/<set>` by default, and leaves the Anima
captions as they are; point the cache's `--src` at it:

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/captions.py --src 'post_image_dataset/resized/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/moss_captions/<set>' --out 'post_image_dataset/moss/<set>' --caption_format tags"
```

The crop tagger sees each subject's mask grown by `mask_dilate` (16 px, never into another
subject's mask), so a held object (a phone, a straw) lands in its holder's clause; with the bare
mask (`0`) it stayed in the bag. Needs anime_tools with `--mask_dilate`.

The job also writes `{stem}.variants.txt` from each finished caption (`variants` 4: v0 = the
caption, v1–v3 = head kept, the rest shuffled with tag dropout 0.1, each clause shuffled inside
and dropped whole at 0.1, text clauses verbatim). The Anima variants describe the full bag, so
none are copied. To train on them, cache with `--caption_variants` and set `caption_variant_rate`.

**Prompt tail** (`prompt_tail`, default `sure`): the cache and generate append
`\n</think>\n\nSure,` after the chat template, closing the empty think block and starting
an answer. On caststation a LoRA trained and rendered with it beat its control by eye, and
kept most of the gain rendered without it (`findings/chat_template.md`). The TE cache and
the LoRA stamp it; generate's `auto` follows the LoRA (unstamped: `off`) and renders `sure`
without one. The i2i configs pin `off`. ComfyUI does not add it.

## Sampling

Trade-offs and the timing table: guidebook §5.

- **Sampler defaults** come from the reference pipeline: CFG 3, the uncond branch skips
  block 22, shift 5.
- **`--sampler er_sde`** (default; `euler` is the pipeline's ODE): Anima's ER-SDE-Solver-3
  on the same σ grid, same speed, composes with Spectrum; a different sample at the same
  seed.
- **`--spectrum`** (on; `--no-spectrum` runs every step in full): 2.6× at 50 steps with
  the default `flex_window 0.5`.
- **`--compile`** (on; `--no-compile` for eager): about 1.37× on top, with or without
  Spectrum, for a few seconds of compile per job. Fine detail drifts from the eager image;
  composition does not.
- **DAVE** (`--dave auto`, off): more same-prompt variety across seeds; `docs/inference/dave.md`.
- **`--int8 on`** (default; `off` runs the bf16 DiT): the DiT's block linears as int8 GEMMs
  from the GPTQ file, 11–15 s against 22 s per 1024² render (ER-SDE 50 + Spectrum) and 7.8
  against 10.7 GiB on the fp32 stream. Refuses LoHa and `--blocks_to_swap`.
  `docs/moss/optimization/int8.md`.
- **`--stream fp32`** (default in generation and training): fp32 around the
  bf16 block GEMMs. Prediction error against an fp32 forward drops 2.4–3.5×, for +1.5% time
  and +0.6 GB. A LoRA loads under either, whatever it trained with.
  `docs/moss/methods/fp32_stream.md`.
- **CNS** has no `generate.py` flag. The six MOSS γ files
  (`networks/calibration/moss_cns_gamma*.npz`, `docs/inference/cns.md`) are used by the
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
latent tokens on RoPE's `r` axis. Recipes for `colorize`, `sanitize`, `swap_text` and `uncensor` live in
`configs/moss/i2i/<task>/`. Tasks, flags, commands, the sanitize run and its ComfyUI
workflow: `docs/moss/methods/i2i.md`.

## Research flags

Advanced, not recommended: `--repa_weight` / `--repa_layer` / `--repa_pe_root` (relational
REPA, parked, `_archive/moss_lora/repa.md`); `--ref_timestep` / `--ref_isolate` /
`--caption_variant_rate` (set by the i2i recipes, `docs/moss/methods/i2i.md`; probes in `project/finished/moss_lora/done/i2i.md`).

## More

- Architecture walkthrough (TE pass, latent boundary, blocks, loss, sampler):
  `docs/moss/structure.md`. Code map: `library/moss/CLAUDE.md`.
- The i2i tone term: `docs/moss/methods/tone_loss.md`.
- Position clauses on RoPE h / w (`clause_hw`, on in `lora.toml`): `docs/moss/methods/clause_hw.md`.
- Self-distilled LoRA (experimental, `configs/moss/selfdistill.toml`, the GUI's Recipe
  dropdown): `docs/moss/experimental/self_distill.md`.
- Speed: `docs/moss/optimization/`.
- Measurements and verdicts: `docs/moss/findings/README.md`; closed numerics / kernel
  checks already applied: `docs/moss/reports/README.md`. Unbuilt or closed designs:
  `docs/moss/proposal/`.
- Run logs: `project/finished/moss_lora/README.md`; settled logs in
  `project/finished/moss_lora/done/`; parked lines (caption dropout, REPA) in
  `_archive/moss_lora/`. LoRA / sampler A/B grids: `bench/moss_ab/`; colorize and tone evals:
  `bench/moss_i2i_eval/`.
