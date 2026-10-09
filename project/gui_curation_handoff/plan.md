# GUI curation handoff → anime_tools panel

The trainer GUI stops curating. Curation (captions, tagging, masks, grouping,
exclusion) moves to the `anime_tools` web panel (`anime-tools-gui`). The trainer
GUI keeps cache building and training.

## Phases

1. **Cache-only Preprocess tab** — done, `11fdd794`. Autotag / position / SAM
   sections are gone, and the GUI env pins `CAPTION_AUTOTAG` /
   `CAPTION_POSITION_CLAUSES` to `0`.
2. **"Open anime_tools" button** — done, `ca6af718` (see *As built*).
3. **Retire the Dataset tab** — done in the same commit, so there was never a
   window with two caption writers. The four modules, their ~110 strings per
   language and the five Dataset-only Settings rows are gone. The resize
   preview is a dialog off the Preprocess tab.
4. **(Optional) Embed** — a `QWebEngineView` tab behind `LazyTabHolder`, only if
   switching windows turns out to be a real annoyance. It reuses phase 2's
   launcher.

## Data flow

Images never leave `image_dataset/`. The panel curates in its own workspace and
exports only the decisions; the trainer resizes from the originals itself.

```
image_dataset/                         originals, read-only for the panel
  ├─ panel: workspace/                 resized proxy, captions, masks, ledger
  │    └─ Export --sidecars_only  ──→  post_image_dataset/resized/{rel}.txt (+ .variants.txt)
  │                               ──→  post_image_dataset/masks/…  (original size)
  │                               ──→  image_dataset/{rel}.txt     (revised master)
  │    workspace/_excluded/excluded.json  ── read by the trainer as a skip list
  └─ trainer resize (image_dataset → post_image_dataset/resized/{rel}.png, skips excluded)
       └─ VAE / TE / PE caches
```

- `--sidecars_only` (anime_tools **v0.7.6**) publishes no `image` rows and no
  `_excluded/` mirror. Each mask is fitted to its original, uncapped. The
  request refuses `--resize_cap` / `--webp` with it.
- The caption row falls back revised → workspace master → `image_dataset/`
  master, so every image in the workspace gets a `resized/{rel}.txt`. That
  closes the position-clause trap: TE reads what the panel wrote.
- Exclusion already works. `curation_actions.py` unions `workspace/_excluded`
  (`PACKAGE_WORKSPACE_EXCLUDED_DIR`) into `ResizeRequest.skip`.
- `workspace/resized` stays as the panel's working copy. Masking, grouping and
  OCR need one geometry, so it is not removable without reworking the stages.

## As built (phases 2–3)

- **Pin**: `v0.7.8` (`pyproject.toml`, `uv.lock`).
- **Launcher**: `gui.core.anime_tools_panel` (Qt-free), called by the GUI's
  **anime_tools** tab (`gui/tabs/anime_tools_tab.py`, a `QWebEngineView`). The
  button first shipped in `ca6af718` was replaced by the tab, and the server now
  runs with `--exit-with-window` instead of `--open`.
  - `seed_settings` writes `<home>/.anime_tools_gui.json` (gitignored,
    `ff6e5e54`): `values.export.sidecars_only = true`, drops the saved
    `resize_cap` / `webp`, which sidecars-only refuses. It sets `dataset.src`
    only when `source_image_dir` is not `image_dataset`, and blanks a saved
    `src` otherwise. `dst` / `masks` stay the package defaults. A saved `dst` /
    `masks` inside `post_image_dataset/` gets a warning dialog, not a rewrite.
  - `find_running` probes `/api/info` on the listening ports in 8790–8839 and
    matches `home`, so a second click opens the running panel. It lists the
    listening ports with psutil, because a connect to a closed loopback port on
    Windows waits out its timeout (a full scan took 15 s).
  - `launch` spawns `python -m anime_tools.gui --home <home> --open` detached
    (`anima_daemon.proc.spawn_detached`, `pythonw.exe`), log at
    `output/anime_tools_gui.log`. Not a daemon job; `--open` implies
    `--exit-with-window`.
- **Stale-Export hint**: `export_is_stale` compares the newest
  `workspace/{master,resized}/**/*.txt` and `workspace/masks/**/*.png` against
  `workspace/captions/export/report.json`. A report with `apply: false` counts
  as no Export. The Preprocess status line shows it.
- **Resize preview**: `gui.tabs.preprocess.resize_preview.ResizePreviewDialog`,
  a non-modal dialog. It lists each source image under `source_image_dir` +
  `preprocess_path_pattern` with source size, bucket, tier and the share of
  the image kept, and draws the crop. It skips images in either exclusion
  ledger, reads the tab's live resize widgets, and refreshes on their change
  and on a variant switch.
- **Removed**: the Dataset tab (`image_tab`, `_caption_editor`, `_autotag`,
  `_image_overlays`); its Settings rows (autotag confidence, insert/validate
  artist tags, two grouping thresholds) and their `_paths` defaults;
  `discovery._image_dirs`; the `scripts/tasks/` readers of the GUI's
  `autotag` / `masks_sam` forms.
- **Docs**: root / `gui/` `CLAUDE.md`, `CONTRIBUTING.md` (e), and §6.3 of the
  four guidebooks (now "Curation: the anime_tools panel").
- **Tests**: `test_gui_anime_tools_panel.py` (new) plus a resize-preview case
  in `test_gui_preprocess_tab.py`. Only an offscreen smoke test and a probe
  against a headless panel ran; the button has not been clicked against a
  real window yet.

## Resolved

- **Export default**: the settings file carries per-stage form values
  (`values.<stage>`, read by the panel's `stages.ts`), so the seed works.
  No docs-only fallback was needed.
- **Two writers during phase 2**: moot, phases 2 and 3 shipped together.
- **Task-side `scripts/tasks/` leftovers**: pruned (`preprocess.py` autotag
  form, `masking.py` GUI rule cards).

## Open questions

- **First real click**: check the button end to end — window opens, a second
  click reuses it, Export → `make preprocess` picks the captions up.
- **Mask crop**: masks publish at the original's size. `_load_mask`
  (`library/datasets/cache.py`) interpolates to latent size but ignores the
  resize crop (anchor / margins), so a cropped image's mask is off by the crop.
  Either the trainer resize crops masks alongside images, or `_load_mask` reads
  the `anima_resize_*` keys off the resized PNG and crops first.
- **Excluded after resize**: an image excluded after the trainer already
  resized it keeps its `resized/*.png` and caches. Check whether
  `make preprocess-reconcile` drops skipped images, or add that.
- **Coverage**: Export walks `workspace/resized`, so an image added to
  `image_dataset/` after the panel last resized gets no caption from Export.
  The panel runs resize as a preflight. Check that the Export button triggers
  it too, or warn on a count mismatch.
- **Migration**: existing curation lives in `post_image_dataset/resized`.
  `anime_tools.workspace.migrate --apply` *moves* it into `workspace/`, which
  empties the trainer tree until the next resize + Export. Document the
  sequence: migrate → Export (sidecars only) → resize.
- **Pin lag**: the panel runs at the pinned tag, so any further panel change
  this needs means cutting a tag first.
- **`stage_form.py` leftovers**: its `autotag` / `masks_sam` entries
  (`TRAINER_FIELDS`, `SHOWN_BOUND`, `FIELD_ORDER`, `_PP_SEEDS`, the card
  branches of `load_stage_values` / `merge_stages_into_meta`) and the
  `test_gui_stage_form` cases that render those schemas. They keep an old
  variant's `[variant.stages.masks_sam]` rows round-tripping on Save; drop
  them with those rows.
- **`curation_decisions.json`**: nothing writes it any more (the Dataset
  tab's skip / move marks), but `_curation_skips` still honours an existing
  file. Drop the reader once no checkout has one.
- **TE staleness relies on mtimes**: the text cache re-encodes a caption
  newer than its cache. If Export ever preserves source mtimes, edits would
  slip past it.
- **Pre-existing test failures** (not from this work):
  `test_curation_boundary` flags `scripts/tasks/masking.py` importing
  `library.config.sam_masks`, and `test_doc_refs` flags stale paths in
  unrelated docs.
