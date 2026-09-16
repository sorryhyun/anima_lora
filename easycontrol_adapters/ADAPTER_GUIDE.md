# Building your own EasyControl adapter

This guide adds a **new EasyControl control task** to Anima, following the
worked example **colorize** (`easycontrol_adapters/colorization/`). A local
adapter lives under `easycontrol_adapters/<your_task>/`.

The network, forward pass, `b_cond` gate and inference cache are shared by every
control task (`networks/methods/easycontrol.py`). A new task changes **how the
condition image is built** and the wiring around it; it does not touch
`networks/`.

---

## 0. What a control task is

EasyControl runs a **reference image** through the VAE into *cond tokens* that
flow alongside the image being generated (architecture:
`docs/experimental/easycontrol.md`).

Plain EasyControl uses the same image as reference and target, so it learns to
copy. A control task pairs each target with a **different** reference, so the
model learns `reference → target`.

| | plain EasyControl | a control task (e.g. colorize) |
|---|---|---|
| target | image X | the color image X |
| reference | image X (identical) | a changed version of X (B&W manga of X) |
| learns | copy | manga → color |
| text | full caption | (optional) shorter caption |

When the reference kind has no paired ground truth (real B&W manga has no color
version), start from images you have (the targets) and **derive** each reference
from its target. The derived reference must look like what you will feed at
inference. For `depth → image` or `pose → image` the derivation step would be a
depth estimator or pose detector.

The job is: **write `target_image → reference_image`, cache its output, describe
the dataset in a descriptor, and register the task name.**

---

## 1. The four things you touch

| # | Thing | colorize version | what it's for |
|---|-------|------------------|---------------|
| 1 | `easycontrol_adapters/<task>/` | `colorization/` (`mangafy*.py`, `wb.py`, `color_caption.py`, `prep.py`) | builds and caches the reference (and optionally a task-specific text cache) |
| 2 | `configs/easycontrol/<task>.toml` | `configs/easycontrol/colorize.toml` | the descriptor: `name` slug, `[staging]` / `[preprocess]` / `[training]` tables, `[[datasets]]` blueprint, optional `[variant]` |
| 3 | `_EASY_ADAPTERS` in `scripts/tasks/training.py` | `"colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess}` | makes `EASYADAPTER=<task>` work for `make easycontrol-staging` / `easycontrol-preprocess` / `easycontrol` |
| 4 | `_ADAPTERS` in `cmd_test_easycontrol` (`scripts/tasks/inference.py`) | the `"colorize"` row | checkpoint prefix, output folder, reference fallback folder, empty-prompt default for `make test-easycontrol` |

---

## 2. Thing 1 — the adapter project (`easycontrol_adapters/<task>/`)

It does two jobs: **make the reference image** and **cache it**, plus optionally
a task-specific text cache.

### 2a. The function that builds the reference

Take an RGB `uint8 (H,W,3)` image plus a seed; return an RGB `uint8 (H,W,3)`
reference of the **same size** (see §3). The same seed must give the same output,
so re-runs and parallel workers agree.

In colorize this is `mangafy.py::mangafy_array` (GPU twin
`mangafy_gpu.py::mangafy_array_gpu`):

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

Patterns worth copying:

- **Seed each image from its stem with `zlib.crc32(stem)`**, not Python's
  `hash()`, which is salted per process and makes parallel workers disagree.
  Derive any per-image variety (colorize jitters screen angle/period) from that
  seed.
- **Import heavy dependencies lazily**, only in the engine that needs them.
- **Prefer an engine that needs no downloads** — colorize's `cv2`/`gpu` engines
  let you prep and train on a fresh checkout.
- **Write files atomically.** `_save_png_atomic` writes a temp file and
  `os.replace`s it. An interrupted direct save leaves a truncated PNG that the
  "skip if it exists" check trusts forever.

If the reference already exists on disk (real depth maps, real sketches), skip
the build step and cache those directly.

### 2b. (Optional) a task-specific text cache

colorize trims captions to color tags (`color_caption.py`): the reference already
fixes shape and layout, so text only needs to carry what B&W cannot — hue — and
every remaining word is something the model cannot get from the reference.

Ask what your reference fixes and what is left for text. A `pose → image`
reference fixes pose but not clothing or setting, so you would likely keep the
full caption. Many adapters keep captions as-is and omit `text_cache_dir` (§4).

If you do build a variant text cache, two knobs are easy to confuse:

- **`caption_dropout_rate`** — the fraction of steps that drop the caption
  entirely, which trains the no-prompt behaviour. A high value makes prompts
  weak.
- **`use_shuffled_caption_variants`** — the full-vs-partial balance. The cache
  holds v0 (full set) plus shuffled, tag-dropped v1+; the loader draws v0 20% /
  v1+ 80% of captioned steps (`use_shuffled_caption_variants_only` drops v0).

### 2c. `prep.py` — the cache builder

Stages, each **idempotent** (skips work already done):

1. **Build** — walk every image under `--src` (`post_image_dataset/resized`), run
   the builder, write the reference PNG into `--staging` mirroring the source
   layout.
2. **Encode** — VAE-encode the staged references into `--cond_cache_dir` with
   `library.preprocess.cache_latents` at each image's **native size**. Same
   `{stem}_{WxH}_anima.npz` format as the normal cache.
3. **(Optional) Text** — re-encode captions into `--text_cache_dir` with
   `library.preprocess.cache_text_embeddings` and a `caption_transform=` (plus
   `caption_shuffle_variants` / `caption_tag_dropout_rate`).

colorize adds a fourth stage (white-balanced target latents into
`--target_cache_dir`); add stages like that only if your targets need them.

Use the library helpers — `library.preprocess.{cache_latents,
cache_text_embeddings, tqdm_progress}` and `library.preprocess._dataset.walk_images`
— rather than a hand-written encode loop.

Two correctness traps:

- **Stems must match.** Cache file names must line up with the target stems the
  loader enumerates from `image_dir`; unmatched names are silently never paired.
  colorize reads captions from a tree laid out like `resized/` (`--caption_src`).
- **The uncond sidecar.** If you build a text cache and use caption dropout,
  re-stage the shared `T5("")` sidecar as colorize does
  (`library.preprocess.uncond.stage_uncond_sidecar_with_models`).

---

## 3. Cond token count

The cond stream runs at the cond latent's native token count; there is no
padding knob (`docs/experimental/easycontrol.md`, "Cond token count"; bucket
bands: the `bucketing` skill). Encoding a same-size reference at native size
(§2a, §2c) puts the cond latent on the same bucket as its target. Cond and
target shapes may also differ — the loader falls back to the cond filed at its
own shape, and `cond_diff_loss` skips on a mismatch — as the cross-image pair
tasks (`tools/subject_pairs.py`, `tools/phash_edit_pairs.py`) rely on. For a
smaller reference, set `cond_res_scale` or shrink the image before encoding so
the latent still lands on a real bucket.

---

## 4. Thing 2 — the descriptor (`configs/easycontrol/<task>.toml`)

One file holds everything task-specific (see `configs/easycontrol/colorize.toml`):

- top-level **`name`** — the slug. Trees live under
  `post_image_dataset/easycontrol/<name>/`, and `output_name` defaults to
  `anima_easycontrol_<name>`.
- **`[staging]`** / **`[preprocess]`** — flat tables your stage/preprocess
  functions turn into `prep.py` flags (`_toml_table_to_argv`: `--key value`,
  lists spread, `true` → bare `--flag`, `false` → omitted).
- **`[training]`** — folded into `train.py` as CLI overrides on the base
  `easycontrol` method.
- **`[general]` / `[[datasets]]`** — the dataset blueprint. `make easycontrol`
  writes it to a generated `dataset_config.toml` under the slug dir (with
  `{name}` interpolated) because `train.py`'s dataset validator rejects the other
  top-level keys.
- **`[variant]`** (optional) — `family = "easycontrol"`, `label`, `description`,
  `order`; descriptors with this block appear in the GUI's EasyControl tab.

colorize's blueprint subset:

```toml
  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'                         # the color targets
  cache_dir = 'post_image_dataset/lora'                            # shared TE/PE cache, reused
  cond_cache_dir = 'post_image_dataset/easycontrol/{name}/cond'    # reference latents (prep.py)
  text_cache_dir = 'post_image_dataset/easycontrol/{name}/text'    # color-only text cache (prep.py)
  latent_cache_dir = 'post_image_dataset/easycontrol/{name}/target' # WB target latents (prep.py)
  recursive = true
  flip_aug = false
  num_repeats = 1
```

- **`cond_cache_dir`** — makes this a control task. The loader matches each
  target to a reference latent by stem and keeps only targets that have one.
- **`text_cache_dir`** — redirects only the text cache. Omit it to use the
  shared text cache and skip the text stage.
- **`latent_cache_dir`** — redirects only the target latents. Omit it to use
  `cache_dir`. A latent missing here is re-encoded from the original,
  uncorrected image (with a warning), so populate it before training.
- **`flip_aug = false`** — required. The cond cache has no flipped latent, and
  the loader raises if flip is on.

`[training]` knobs to think about:

- **`network_args`** — `b_cond_init` sets how much the reference contributes at
  step 0 (`-10` ≈ plain DiT at step 0; see "Step-0 baseline equivalence" in
  `docs/experimental/easycontrol.md`; colorize loosens it so the reference kicks
  in sooner; learnable either way), `cond_scale`, `apply_ffn_lora` (`0` drops
  the FFN LoRA, about half the trainable params), `cond_res_scale`.
- **`easycontrol_cond_noise_max`** — noise added to the reference in training
  (σ ~ `U(0, max)`, `cond + σ·ε`). `0` treats the reference as an exact
  blueprint; higher values make it a rough hint and push detail onto text.
- **`easycontrol_drop_p`** — how often the whole reference is dropped, for
  image-CFG. Tasks that always have a reference set `0`.
- **`output_name`** — the inference selector finds your latest checkpoint by
  this prefix (§5).
- **`blocks_to_swap = 0`** — what the shipped descriptors use.

---

## 5. Things 3 and 4 — register the task

**`scripts/tasks/training.py`:** write two functions with the signature
`(adapter, cfg, base, extra)` and add them to `_EASY_ADAPTERS`:

```python
_EASY_ADAPTERS = {
    ...
    "colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess},
    "<task>": {"stage": _task_stage, "preprocess": _task_preprocess},
}
```

`_colorize_stage` / `_colorize_preprocess` are the model: each runs
`easycontrol_adapters/colorization/prep.py` with stage-skip flags, the
slug-derived paths (`_colorize_prep_paths(base)`), the table knobs, and `extra`
last so user `ARGS` win. `_easyadapter()` rejects names missing from this dict.
`make easycontrol EASYADAPTER=<task>` needs no further edit.

**`scripts/tasks/inference.py`** (`cmd_test_easycontrol`): add a row to
`_ADAPTERS`:

```python
"<task>": {
    "weight": "anima_easycontrol_<task>",   # prefix of output_name
    "out": "<task>",                         # output/tests/<out>/
    "ref_dir": ROOT / "post_image_dataset" / "resized",  # random ref if none given
    "empty_prompt": False,                   # True → default to --prompt ""
},
```

---

## 6. Run it

```bash
# First time only: shared target latents + text cache in post_image_dataset/lora.
make preprocess

# 1. Build the reference tree. QA a few first, then eyeball the PNGs under
#    post_image_dataset/easycontrol/<name>/staging/.
make easycontrol-staging EASYADAPTER=<task> ARGS="--limit 8"
make easycontrol-staging EASYADAPTER=<task>

# 2. Encode the caches. Idempotent.
make easycontrol-preprocess EASYADAPTER=<task>

# 3. Train (DiT frozen, adapter only).
make easycontrol EASYADAPTER=<task>

# 4. Inference — give it a real, in-distribution reference image.
REF_IMAGE=path/to/condition.png make test-easycontrol EASYADAPTER=<task>
#    Steer with text:  ... ARGS='--prompt "..."'
```

### Inference notes

- The reference is VAE-encoded as-is at inference; there is no build step. Feed
  what your builder was imitating (colorize: a real screentoned page, not a
  grayscale photo).
- `make test-easycontrol` always passes `--easycontrol_image_match_size`, which
  picks the token bucket matching the reference's aspect ratio.
- `--easycontrol_scale` (`EC_SCALE=`) sets how closely output follows the
  reference; `1.0` is the trained default.
- colorize's tuned `--guidance_scale` / scale / step ranges are in
  `colorization/README.md` ("Inference settings").

---

## 7. Checklist

- [ ] `easycontrol_adapters/<task>/` with a deterministic, atomic-writing builder
      (`(img, seed) → reference`, same size) and an idempotent `prep.py`.
- [ ] References encoded at **native size** (§3).
- [ ] `configs/easycontrol/<task>.toml`: `name`, `[staging]` / `[preprocess]` /
      `[training]`, blueprint with `cond_cache_dir` (+ optional
      `text_cache_dir` / `latent_cache_dir`) and `flip_aug = false`.
- [ ] `_EASY_ADAPTERS` entry (training.py) and `_ADAPTERS` row (inference.py).
- [ ] Eyeballed a `--limit 8` staging batch before the full run.

---

## 8. Where to read more

- **`easycontrol_adapters/colorization/README.md`** — the colorize reference
  implementation: caption policy, screentone bands, tag slices, inference
  settings.
- **`docs/experimental/easycontrol.md`** — the network: two-stream forward,
  `b_cond` step-0 bench, `cond_res_scale`, inference cache, memory, limits.
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` and the patched
  `Block.forward`.
- **`networks/CLAUDE.md`** — the per-module map and dispatch rules.
