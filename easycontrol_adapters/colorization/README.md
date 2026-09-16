# Colorization (EasyControl)

Manga / illustration **colorization** as an EasyControl control task: feed a
black-and-white screentoned page (line art + halftone tone), get a colorized
image. It uses the shipped EasyControl network
(`networks/methods/easycontrol.py`); only the condition source differs from the
default task (cond ≠ target, wired via the `cond_cache_dir` subset knob).

## Training pairs

Real B&W manga has no color ground truth, so pairs are built from color images:

- **target** = the color illustrations in `post_image_dataset/resized`,
  white-balanced (`wb.py`) and VAE-encoded into a colorize-specific target cache
  (`latent_cache_dir`). TE/PE stay on the shared `post_image_dataset/lora/` cache.
  Targets below `target_drop_sat` mean saturation (effectively monochrome) are
  paired out by deleting their cond latents.
- **condition** = a synthetic *mangafied* version of the same image: XDoG
  lineart + algorithmic screentone, where the toned value range is split into a
  few luminance bands and each band gets its own pattern (dot / line / cross),
  cached to `cond_cache_dir`. With a text mask (`--mask_dir`, black = text), the
  source's own grayscale is pasted back over text/speech-bubble regions so
  lettering stays pixel-exact in the condition.

A real screentoned page is the inference input, so the synthetic condition has
to imitate one; plain grayscale is out of distribution.

## Caption policy — color tags plus a protected prefix

The condition already carries composition, pose and layout; what B&W cannot
carry is **hue** (and which series the page is from). The text cache therefore
keeps only color tags — hair/eye/skin color, `<color> <garment>`, background
color — preceded by the copyright/series tag (`--text_keep_copyright`, on) and
comic/panel-format tags (`--text_keep_comic`, on). See `color_caption.py`;
copyright tags are recognized against `groups.copyright` in
`post_image_dataset/captions/caption_index.json`, minus `original`.

The TE cache lives in its own `text_cache_dir`
(`post_image_dataset/easycontrol/colorize/text`), built by `prep.py`'s text stage
as a multi-variant cache: v0 = the full set, v1+ = shuffled with each tag
dropped at `text_tag_dropout_rate`. **Comic tags are protected from that
dropout; the copyright tag is not.** Values for both knobs are in the
descriptor's `[preprocess]` table.

Three knobs shape the per-step caption:

- **`caption_dropout_rate`** — the fraction of steps that drop the caption
  entirely (→ uncond `T5("")`), training the empty-prompt default. A high rate
  over-trains the unconditional path into weak steering.
- **`use_shuffled_caption_variants = true`** — on captioned steps the loader
  draws 20% v0 / 80% v1+ (`library/anima/strategy.py`), so partial prompts
  ("pink hair" alone) work.
- **`use_shuffled_caption_variants_only = true`** (colorize sets it) — v0 is
  never drawn; captioned steps are uniform over the partial v1+ variants.

At inference an **empty prompt** auto-colorizes with modal colors (B&W carries
no hair/eye/costume color, so it guesses); a **color prompt**
(`pink hair, blue eyes, white dress`) steers, and partial specs work.

## Files

| File | Role |
|------|------|
| `mangafy.py` | color RGB → B&W manga (XDoG lineart + banded dot/line/cross screentone), per-stem jitter |
| `mangafy_gpu.py` | torch/CUDA twin of `mangafy.py`; same seeded structure (`--engine gpu`, the default) |
| `wb.py` | target white-balance + mean-saturation statistic |
| `color_caption.py` | reduce a full Anima caption to the protected prefix + color tags |
| `prep.py` | mangafy → encode cond latents → WB target latents → color-only text |
| `pool_from_tags.py` | carve a tag slice out of a raw crawl pool into a symlink tree (see *Appending a tag slice*) |

Config: `configs/easycontrol/colorize.toml` — a self-contained descriptor
(top-level `name` + `[staging]` / `[preprocess]` / `[training]` tables + a
`[general]`/`[[datasets]]` blueprint). Colorize trains the base `easycontrol`
method with `[training]` folded in as CLI overrides; the staging/preprocess
tables drive `prep.py`.

## Run

```bash
# 1a. Staging — synthesize the B&W condition tree (mangafy only). Idempotent.
make easycontrol-staging EASYADAPTER=colorize
#    QA a few first:  make easycontrol-staging EASYADAPTER=colorize ARGS="--limit 8"
#    Inspect PNGs under post_image_dataset/easycontrol/colorize/staging/
# 1b. Preprocess — cond latents, WB target latents, color-only text. Idempotent.
make easycontrol-preprocess EASYADAPTER=colorize
#    (re-stage inline:  ARGS="--no-skip_mangafy")

# 2. Train (frozen DiT, adapter only).
make easycontrol EASYADAPTER=colorize

# 3. Inference — a real B&W manga page as the control image (empty prompt).
REF_IMAGE=post_image_dataset/resized/takaman_\(gaffe\)/7645571.png \
    make test-easycontrol EASYADAPTER=colorize
#    Color steer:  ... ARGS='--prompt "pink hair, blue eyes, white dress"'
```

`EASYADAPTER=colorize` in `test-easycontrol` loads the latest `anima_colorize*`
checkpoint, saves to `output/tests/colorize/`, and defaults to an empty prompt.

### Appending a tag slice from the crawl pool

The curated corpus (`post_image_dataset/resized`) is thin in some slices —
`korean text` is 22 pages there vs 349 in the crawler's `retrieved/` pool. To
add such a slice without touching the shared corpus, give it its own resized
tree + caches as a second dataset subset. Worked example, the Korean-text slice:

```bash
KO=post_image_dataset/easycontrol/colorize/korean

# 1. Select — symlink the tagged pages (image + .txt) out of the crawl pool,
#    minus the stems the curated master already has.
python easycontrol_adapters/colorization/pool_from_tags.py \
    --src ~/gelcrawl/retrieved --dst $KO/src \
    --include-tags "korean text" --skip-existing-in image_dataset

# 2. Resize — same free-fit tiers as the corpus ([training].target_res).
#    --recursive is required: the pool mirrors the crawl's artist subdirs.
#    Captions are mirrored into $KO/resized, so it doubles as --caption_src.
python -m anime_tools.stages.cli.resize_images --src $KO/src --dst $KO/resized \
    --target_res 1024 896 --recursive

# 3. Masks (recommended for a text slice — keeps glyphs pixel-exact in the
#    condition instead of screentoned into mush).
python -m anime_tools.masking.cli.generate_masks --config configs/sam_mask.yaml \
    --image-dir $KO/resized --mask-dir /tmp/ko-sam --prompts "speech bubble,text" \
    --checkpoint models/sam3/sam3.pt --batch-size 4 --recursive
python -m anime_tools.masking.cli.merge_masks /tmp/ko-sam --output-dir $KO/masks

# 4. Stage + preprocess into the slice's own trees. `ARGS` is appended last, so
#    these path flags override the slug-derived defaults.
SLICE="--src $KO/resized --caption_src $KO/resized --mask_dir $KO/masks \
       --staging $KO/staging --cond_cache_dir $KO/cond \
       --target_cache_dir $KO/target --text_cache_dir $KO/text"
make easycontrol-staging    EASYADAPTER=colorize ARGS="$SLICE"
make easycontrol-preprocess EASYADAPTER=colorize ARGS="$SLICE"
```

Then add the slice as another `[[datasets.subsets]]` in
`configs/easycontrol/colorize.toml` (already wired for `korean/`), pointing
`image_dir`/`cond_cache_dir`/`text_cache_dir`/`latent_cache_dir` at the trees
above. Only targets with a cached cond latent train, so
`[staging].exclude_data_includes` + `target_drop_sat` still pair out the
slice's monochrome pages.

Before raising `num_repeats` on such a slice: a crawl slice is usually
**artist-skewed** (the Korean-text one is ~80% two artists), so repeats buy tag
presence at the cost of style skew.

### Inference settings

- `--easycontrol_image_match_size` — always on for this target; picks the token
  bucket matching the page aspect ratio so tall pages don't squash.
- `--easycontrol_scale` (`EC_SCALE=`) — **1.0** trained default; 1.1–1.2 if
  color bleeds past lines, 0.7–0.8 for looser coloring.
- `--guidance_scale` — **empty prompt → 1.0–1.5** (higher oversaturates);
  **color prompt → 3.0–4.5** (needed for the prompt to take effect).
- `--infer_steps` 20–28, `--sampler euler` — more steps buy little.
- Feed a **real screentoned B&W page**; it is VAE-encoded as-is (no XDoG at
  inference). A flat grayscale photo or clean lineart is out of distribution.

## Phase B (deferred)

- Learned lineart (Anime2Sketch / sketchKeras) + **ScreenVAE** screentone synthesis
  for on-manifold tones.
- **Synthetic speech bubbles** on the cond, masked from the loss, so the model
  leaves text untouched on pages without a text mask.
- Gradient/special screentones; difficulty curriculum (single clean figure →
  multi-figure dense pages).
