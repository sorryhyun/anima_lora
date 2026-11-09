# i2i tasks

One folder per image-to-image task: how its condition (or pair) tree is built from the
dataset. Staging is model-agnostic; each model line encodes and trains from the staged
tree on its own.

```
i2i_tasks/<task>/  ──make i2i-stage TASK=<task>──▶  post_image_dataset/i2i/<name>/staging/  (+ pairs.jsonl)
                                                        │
             ┌──────────────────────────────────────────┴───────────────────┐
             ▼                                                              ▼
  EasyControl (Anima)                                          MOSS i2i (VAE reference on RoPE r)
  make i2i-preprocess TASK=<task>   (= easycontrol-preprocess)  make i2i-preprocess TASK=<task> MODEL=moss
  make easycontrol EASYADAPTER=<task>                           → configs/moss/i2i/<task>/cache.toml
                                                                → target latent + TE + {stem}.moss_ref
                                                                scripts/moss/train.py on that cache
```

`pairs.jsonl` is the model-agnostic pair list (`{stem, target, cond, caption}` per
line, paths relative to it); near_twins, watermark, `tools/textless_pairs.py`,
`tools/mosaic_pairs.py` and swap_text's caption stage write it, and
`library/moss/caching/sources.py::manifest_pairs` reads it. Colorize's MOSS cache reads the staging
tree directly (`scripts/moss/colorize_cache.py`, `library/moss/colorize.py`: vision TE +
OCR captions). The MOSS task list is `library.moss.requests.I2I_CACHE`; the GUI's two i2i tabs
(`gui/moss/i2i.py`) run stage → cache → train → test for each.

| Task (`TASK=`) | Folder | Staging builds |
|---|---|---|
| `colorize` | `colorization/` | mangafied B&W page per colour image (`mangafy`, `wb`); colour-only captions (`anime_tools.captions.color_tags`) |
| `inpaint` | `inpainting/` | masked condition per image |
| `region` | `region/` | SAM3 masks → paint-to-character condition + positioned captions |
| `near_twins`, `sanitize` | `tools/near_twins/` (`stage = "near_twins"`) | in-artist near-twin pair tree |
| `subject`, `subject_edit` | `tools/subject_pairs.py`, `tools/subject_edit_pairs.py` | subject pairs |
| `phash_edit` | `tools/phash_edit_pairs.py` | perceptual-hash aligned edit pairs |
| `swap_text` | `swap_text/` | hitocrawl JA page → translated-edition lettering, registered + composited; target-side OCR captions, half the pairs swap only some regions (MOSS-only) |
| `watermark` | `tools/watermark_pairs.py` | faint tiled URL / handle watermarks (lossless) over the sanitize tree's targets, added to that tree as `wm_*` pairs (`name = "sanitize"`; MOSS-only) |

A task is a file, `configs/<task>.toml`:

```toml
name = "sanitize"        # post_image_dataset/i2i/<name>/ (default: the file stem)
stage = "near_twins"     # staging tool (default: the task id)

[staging]                # that tool's knobs; ARGS on the command line override them
tag_any = ["speech bubble", "english text", "japanese text"]
```

`stage` names a row of `_I2I_STAGES` in `scripts/tasks/training.py` (a `stage` body and an
EasyControl `preprocess` body, `None` for MOSS-only kinds): `near_twins`, `colorize`,
`inpaint`, `region`, `subject`, `subject_edit`, `phash_edit`, `swap_text`, `watermark`. A new task on
an existing tool is one file, e.g. a local, untracked `configs/uncensor.toml` with
`stage = "near_twins"` and censor tags runs as `make i2i-stage TASK=uncensor`.

The EasyControl side of a task (`[preprocess]` / `[training]` knobs, the `[[datasets]]`
blueprint, the GUI `[variant]` block) is `configs/easycontrol/<task>.toml`; the miners
(`near_twins`, `subject*`, `phash_edit`) rewrite its blueprint tail on every stage, and
`near_twins` scaffolds it on a first run. MOSS-only tasks (`swap_text`) have none.
`make easycontrol-staging EASYADAPTER=<task>` is an alias of `i2i-stage`.
`post_image_dataset/easycontrol` is a symlink to `post_image_dataset/i2i` so pre-rename
paths in configs and caches still resolve.

Adding a task: `ADAPTER_GUIDE.md`.
