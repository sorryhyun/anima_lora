# MOSS i2i (image to image)

An i2i LoRA learns a reference → target edit on MOSS-Image. The reference image is
VAE-encoded and goes into every forward (both CFG branches) as clean latent tokens after the
image and text rows, on RoPE's reserved `r` axis (`r = 1`; `canvas`: `r = 0`).

| Flag | Meaning | Stamped in the LoRA |
|---|---|---|
| `ref_scale` | reference side = target side / `ref_scale` (1 = full size, 2 = ¼ area) | checked against the cache |
| `ref_aspect` | `target`: the reference is cropped to the target's aspect; `native`: it keeps its own, at the target's area / `ref_scale`² (cache and generate; the ComfyUI Reference node crops) | no (cache recipe) |
| `ref_timestep` | the ref rows are modulated at this timestep in every block (0 = clean) instead of the step's | yes |
| `ref_isolate` | ref queries attend only to ref keys; image and text still read the ref | yes |
| `ref_pos` | `aligned` (default): over the image, each ref token on the cell it covers; `offset`: one image width to the right; `canvas`: beside the image at `r = 0` | yes |
| `ref_adapters` | `off` (default): one LoRA bank on image, text and ref rows; `kv`: the ref rows run Base, no backward, and the image and text rows read them through k / v export adapters (`ref_rank`) | yes |
| `ref_gate` | `cut`: on `gate_rate` of the steps the image queries in a random hole lose the ref keys in the hole (training only) | yes |
| `ref_vision` | the page also enters the text encoder through Qwen3.5's vision tower, ahead of the caption | no (cache recipe) |

`generate.py` and the ComfyUI Reference node read `ref_timestep` / `ref_isolate` from the
LoRA. `ref_scale`, `ref_vision` and the prompt format must match the cache the LoRA trained on.
With `ref_isolate` and a `ref_timestep`, the ref rows depend only on the page, so they run
through the blocks once per page and their K/V are cached (`RefCache`; ComfyUI `ref_cache`).

## Tasks

Recipes: `configs/moss/i2i/<task>/{cache,train,generate}.toml`, one per task, the latest;
superseded ones are in `_archive/configs/moss_i2i/`. The task → cache-request map is
`library.moss.requests.I2I_CACHE`. Staging and data: `i2i_tasks/README.md`. Run histories:
`project/lora_blocks/reports/`, `project/finished/moss_lora/`.

| Task | Reference | Text encoder input | Shipped |
|---|---|---|---|
| `colorize` | B&W page, full size, isolated | control captions, no OCR text, no vision tower. Manga page: `comic`, optionally a box-only `text` item per lettered line and `character` items (box + hair / eye / clothing colours). Illustration: its colour + series tags | v5 (`colorize/moss_colorize_v5` + `colorize/moss-i2i-colorize-v5.json`): [colorize_v5_control.md](../proposal/colorize_v5_control.md) |
| `sanitize` | texted image, full size, isolated | one fixed caption for every pair: `no sfx, no dialogue, no watermark` | v3 (`sanitize/moss_i2i_sanitize_v3`), with the tone term: [tone_loss.md](tone_loss.md) |
| `swap_text` | JA manga page, full size, isolated | layout JSON, one text item per line to letter (translated text + bbox), Hangul split into syllable tokens | v6 recipe (`train.toml`, from the bw10 copy block) |
| `character_ref` | a solo image of the character by the same artist, own aspect (`ref_aspect native`), full or half size by pair (`ref_scale_alt 2`), isolated | the target's caption without character, series, artist and meta tags, and without the tags the reference's caption shares | none: no run reads the reference yet |

## Running

```bash
make i2i-stage TASK=<task>                      # pair tree under post_image_dataset/i2i/<task>/
make i2i-preprocess TASK=<task> MODEL=moss      # cache (colorize: scripts/moss/colorize_cache.py; the others: cache.py)
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/i2i/<task>/train.toml"
make daemon-run ARGS="scripts/moss/generate.py --config configs/moss/i2i/<task>/generate.toml \
    --lora output/ckpt_moss_i2i/<lora>.safetensors --ref_image <page or folder>"
```

`--ref_image` takes one image, a folder or a quoted glob (one job per image and prompt). The output keeps
the reference's aspect ratio, with the area capped at `width × height`. For an eval set in one
job, `--ref_prompts` gives each reference the `.txt` beside it (`<page>.src.webp` →
`<page>.txt`) and `--seeds 0,1,2` renders every page once per seed. The GUI's **i2i Preprocess**
(stage, preprocess) and **i2i Train** (train, test) tabs (`make gui`, Model set to MOSS-Image)
run the same four steps. Staging is shared with EasyControl.

## sanitize

Removes text, SFX, speech-bubble text and watermarks and repaints what was underneath.

- **Data.** In-artist near-twins (`i2i_tasks/tools/near_twins/`, `[staging]` of
  `i2i_tasks/configs/sanitize.toml`): a texted image (`{id}_tags`, the condition) and its clean
  twin (`{id}_no_tags`, the target); `add_identity_pairs` adds clean → clean pairs.
- **Cache.** `caption_format = "fixed"` encodes `fixed_caption` for every pair; the reference
  carries the edit. Copies of `fixed_caption` live in `generate.toml` (as `prompt`, with
  `prompt_format = "text"`) and in the ComfyUI MOSS Toolkit (`FIXED_CAPTIONS["sanitize"]`);
  change all three together.
- **Train.** `ref_timestep = 0`, `ref_isolate = true`; the tone term late in training
  (`tone_weight 0.003`, `tone_start 0.5`) keeps the page from brightening and cooling.
- **Generate.** Use the fixed caption; it is what the LoRA saw.

## swap_text

Re-letters a Japanese manga page in a translated edition's text, inside the balloons the
prompt names.

- **Data.** Hitocrawl edition twins (`i2i_tasks/swap_text/prep.py`, `i2i_tasks/configs/swap_text.toml`):
  each target is the JA page with the twin's clean text groups pasted in, the caption one
  layout-JSON text item per pasted line in `abs_centralized` `yxyx` pixels. Stages, pasting
  rules and gates: the docstrings of `i2i_tasks/swap_text/prep.py` and its stage modules.
- **Cache.** A caption that already is layout JSON passes through as written, so no
  `embeds_plain` is stored and `plain_caption_rate` has no effect. `token_split = "syllable"`
  splits Hangul into syllable tokens ([ko_token_split.md](../findings/ko_token_split.md)); the
  LoRA stamps it and `generate` follows the stamp.
- **Train.** `ref_timestep = 0`, `ref_isolate = true`, init from the copy block
  (`init_lora`, `mod_rank 16`).
- **Generate.** The page's own caption (or a hand-written one) as `--prompt`, `--width` /
  `--height` at the page size so the bboxes land where they were measured.

## ComfyUI

[ComfyUI-MOSS-Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit) 0.7.0 or later,
`example_workflows/moss-i2i-<task>.json`:

```
MOSS-Image Loader ─ MODEL ─ MOSS i2i Reference (LoRA, task auto) ─────────────── MODEL ─┐
                  ─ CLIP ─┬ CLIP Text Encode (Caption Helper, task sanitize) ── positive ┤
                          └ CLIP Text Encode ("") ──────────────────────────── negative ┤ MOSS Spectrum KSampler
                  ─ VAE ──── MOSS i2i Reference ────────────────────────────── LATENT ──┘ (30 steps, CFG 4, skip 22)
Load Image → Reference.image
```

The Caption Helper needs no image and registers only when `anime-tools` is installed in
ComfyUI's environment. For swap_text the layout-JSON prompt is the positive (typed into
`CLIP Text Encode`, or through Caption Helper `task layout`, which passes it through). LoRA +
workflow per task: `sorryhyun/MOSS-I2I-LORAs` on the Hub.

Ref forward: `library/moss/model.py` (`ref=`, `_ref_only`, `ref_rows`, `RefCache`); the
ComfyUI copy is `reference.py` in the toolkit.
