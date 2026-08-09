# easycontrol_inpaint — masked-region regeneration as an EasyControl variant

A second EasyControl control-task variant, alongside `colorize`. Where colorize
conditions on a synthetic manga/lineart latent and learns hue, **inpaint**
conditions on the original image with a **gray hole** punched in it and learns to
regenerate that hole consistent with the surrounding pixels and the caption.

It rides the same frozen-DiT `networks.methods.easycontrol` network — **no
network code changes**. A variant here is just *(cond transform, caption policy,
a few hypers)*, and inpaint is the cleanest possible instance of that recipe:

- **cond ≠ target**, like colorize → a parallel `cond_cache_dir`.
- **captions reused verbatim** (unlike colorize, which filters to color-only):
  the hole content *is* the ambiguous variable, and the full caption already
  describes it. So there is **no text stage and no `text_cache_dir` override** —
  the shared LoRA TE cache is reused as-is. This makes inpaint strictly simpler
  than colorize.

```bash
make easycontrol-preprocess EASYADAPTER=inpaint   # mask + VAE-encode cond
make easycontrol EASYADAPTER=inpaint              # train → output/ckpt/anima_inpaint
REF_IMAGE=masked.png EASYADAPTER=inpaint make test-easycontrol --prompt "..."
```

## Concept

| | target | cond |
|---|---|---|
| **what** | original full image | same image, region replaced by flat mid-gray |
| **latents** | `post_image_dataset/lora/` (reused, `make preprocess`) | `post_image_dataset/easycontrol/inpaint/cond/` (new) |
| **text** | shared LoRA TE cache (reused) | — |

The model sees most of the image intact + a flat-gray hole + the caption, and
learns to fill the hole. Empty prompt → context-only fill; a caption → steered
fill. This matches colorize's training contract exactly — the loader pairs the
cond latent to the target by stem via the subset `cond_cache_dir` knob
(`networks/methods/easycontrol.py::set_cond` reads `batch["cond_latents"]`).

### Hole fill: mid-gray

The masked region is filled with **flat mid-gray** (the SD-inpaint convention;
`0` in the VAE's `[-1, 1]` input space) *before* VAE-encoding. It encodes to a
flat latent region the per-block `b_cond` gate + cond LoRA learn to read as
"regenerate here." Chosen over noise-fill (noisier cond, less standard) and
black (collides with legitimately-black content).

### Masks: free-form + rectangles, one seeded mask per image (v1)

The cond cache is keyed `{stem}_{WxH}_anima.npz` — **one cond per image** — so v1
bakes **one mask per image**, seeded per-stem (reuse colorize's
`_stable_seed = crc32(stem)`). The generator is a LaMa-style mix: a few thick
random brush strokes (random-walk polylines with random width) + 1–2 random
rectangles, covering ~10–40% of the frame. Holes therefore vary in
position/shape/size *across* the dataset even though each image's hole is fixed,
so the model learns general inpainting rather than memorizing one hole shape.

No external deps — pure numpy/cv2, fanned out across a `ProcessPoolExecutor` like
colorize's cv2 engine.

## Deferred upgrades (explicitly not in v1)

Both are real improvements but need work beyond the colorize skeleton, so they're
called out rather than silently scoped out:

1. **N mask variants per image.** Caching K masked variants per stem and drawing
   one per epoch would kill the fixed-per-image-hole limitation. The loader
   currently supports *text* variants (`caption_shuffle_variants`) but **not cond
   variants** — adding a cond-variant suffix + loader draw is the prerequisite.
2. **Hole-focused FM loss.** Feeding the hole mask as the flow-matching loss mask
   (so gradient concentrates on the filled region instead of the given context,
   which is trivially reconstructable from the cond) should sharpen learning.
   Needs the mask cached and wired through to the existing `masked_loss` path —
   note `masked_loss` today consumes the SAM3/MIT *subject* masks in
   `post_image_dataset/masks/`, not an inpaint hole mask, so this is a distinct
   mask source. v1 ships `masked_loss = false` (whole-frame loss, standard
   EasyControl).

A third, lighter future option: blend in **SAM3 subject masks** from
`post_image_dataset/masks/` for a fraction of images (object-removal /
regeneration coverage) behind an `--object_mask_p` flag. v1 ships the synthetic
generator only.

## Files

### New

**`easycontrol_adapters/inpainting/prep.py`** — sibling of
`easycontrol_adapters/colorization/prep.py`, simpler (two stages, no text):
- `stage_mask(src, staging, …)` — walk `post_image_dataset/resized/`; per image,
  generate a seeded mask, paint the hole mid-gray, write to `…/inpaint/staging/`
  mirroring the source subpath. Idempotent + atomic write (lift colorize's
  `_save_png_atomic`). Reuse the cv2 `ProcessPoolExecutor` fan-out + per-stem
  seed from colorize.
- `stage_encode(…)` — **reused verbatim** from colorize: VAE-encode `staging/`
  into `cond_cache_dir` via `library.preprocess.cache_latents`, at each image's
  native size so the cond latent shape matches its target exactly.
- **No `stage_text`.**
- Defaults: `--src post_image_dataset/resized`,
  `--staging post_image_dataset/easycontrol/inpaint/staging`,
  `--cond_cache_dir post_image_dataset/easycontrol/inpaint/cond`.

**`easycontrol_adapters/inpainting/mask_image.py`** — `mask_array(img_rgb, seed)
-> img_rgb` (free-form brush + rect generator, mid-gray fill). Kept separate so
the generator is unit-testable and the optional SAM-object path can slot in
later.

> **Shipped differently.** The two-file split below (a `configs/datasets/`
> blueprint plus a `configs/methods/` config) was superseded by the
> self-contained per-method layout before this landed: both halves live in
> **`configs/easycontrol/inpaint.toml`**, a single descriptor carrying the
> `[staging]` / `[preprocess]` / `[training]` tables *and* the inline
> `[general]` / `[[datasets]]` blueprint, driven by
> `make easycontrol EASYADAPTER=inpaint`. The tables below are the design
> record — read the shipped file for what actually runs.

**The dataset blueprint** (now the `[general]`/`[[datasets]]` tail of
`configs/easycontrol/inpaint.toml`) — mirror colorize's blueprint, minus the
text override:
```toml
[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1
validation_split = 0.005
validation_seed = 42

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'
  cache_dir = 'post_image_dataset/lora'                       # target latents + TE reused
  cond_cache_dir = 'post_image_dataset/easycontrol/inpaint/cond'
  recursive = true
  flip_aug = false        # latents/cond can't be flipped post-hoc
  num_repeats = 1
```

**The method config** (now the head of the same file) — same network as
colorize, tuned hypers:
```toml
# shipped: no dataset_config cross-reference — the blueprint is inline
network_module = "networks.methods.easycontrol"

network_dim = 32
network_alpha = 32
network_args = ["b_cond_init=-6.0", "cond_scale=1.0", "apply_ffn_lora=1"]

output_name = "anima_inpaint"
sigmoid_bias = 0
logging_dir = "output/logs"
log_with = "tensorboard"

learning_rate = 2e-5
max_train_epochs = 4
save_every_n_epochs = 4
checkpointing_epochs = 4

use_easycontrol = true
easycontrol_drop_p = 0.0          # the masked image is essential — never drop cond
easycontrol_cond_noise_max = 0.0  # don't blur the given context
masked_loss = false               # whole-frame loss in v1 (see deferred upgrade #2)
caption_dropout_rate = 0.15       # mostly text-guided, some context-only fills

blocks_to_swap = 0
gradient_checkpointing = true
unsloth_offload_checkpointing = true
```
Hyper rationale vs colorize: `b_cond_init=-6.0` and `easycontrol_drop_p=0.0`
because the cond is essential (same as colorize). `network_dim=32` (default
EasyControl rank — most of the image is given; bump to 48 if fills lack detail).
`max_train_epochs=4` (broader task than colorize's 3). No
`use_shuffled_caption_variants` — full captions reused from the shared LoRA TE
cache.

### Touched

**`scripts/tasks/training.py`**
- `_EASYADAPTERS = {"colorize", "inpaint"}`
- `cmd_easycontrol_preprocess`: add
  `if adapter == "inpaint": run([PY, "easycontrol_adapters/inpainting/prep.py", *extra]); return`
- `cmd_easycontrol` needs **no change** — it already routes `_easyadapter()` to
  `configs/methods/<name>.toml`, so `inpaint` resolves for free.

**`scripts/tasks/inference.py::cmd_test_easycontrol`** — generalize the current
two-way `is_colorize` branch into a small per-adapter table
(`weight_name`/`out_sub`/`ref_fallback_dir`/default-prompt). For `inpaint`:
`weight_name="anima_inpaint"`, `out_sub="inpaint"`, ref fallback =
`post_image_dataset/resized`. The ref image is the **already-masked PNG**. Nice
follow-up (not required for v1): a `--mask` convenience that gray-fills a supplied
binary mask before encoding so users needn't pre-mask in an editor.

### Optional

**A gui-methods variant** (`inpaint.toml` in `configs/gui-methods/`) — add a
`[variant] family = "easycontrol"` block (mirror colorize's GUI config; rank
16/alpha 16, `b_cond_init=-4.0` per the GUI colorize convention) so it appears
in the GUI method dropdown. **Still not done** — the gui-methods tree has no
`inpaint` entry, so the variant is CLI-only.

**Docs** — add an `inpaint` section to `docs/experimental/easycontrol.md` and the
`EASYADAPTER=inpaint` rows to the root `CLAUDE.md` EasyControl one-liners once
landed.

## Why this is low-risk

- Zero changes to the EasyControl network, training loop, or save/load path — the
  cond-latent pairing (`cond_cache_dir` → `batch["cond_latents"]` →
  `set_cond`) and the `easycontrol_drop_p` / `easycontrol_cond_noise_max` knobs
  already exist and are exercised by colorize.
- `prep.py` is ~⅔ a copy of colorize's prep with the mangafy stage swapped for a
  mask stage and the text stage deleted.
- Targets/captions are reused from the existing LoRA cache, so preprocess only
  adds the cond latents.
