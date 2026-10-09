# gui/CLAUDE.md

PySide6 (Qt6) desktop GUI. Root `CLAUDE.md` owns the training/config/daemon contracts this
GUI drives. Most of the surface is `tabs/config_tab.py` and the `tabs/preprocess/` package.
The GUI does not curate — captions, tagging, masks, grouping and exclusion are the
`anime_tools` web panel's, embedded in the anime_tools tab. For recipe-style changes
(new training field / variant / language, job submission, a new job-submitting tab,
action-button colors) **load the `gui-changes` skill**.

## What it is

Edits TOML configs and submits jobs to the daemon; no training/torch logic. Layout:

| Package | Holds |
|---|---|
| `core/` | **Qt-free** (no PySide6 import, headless-unit-testable): `paths.py` (ROOT, `gui_settings.json`), `config_io.py`, `submit.py`, `validation.py`, `discovery.py`, `anime_tools_panel.py`, `debug_report.py` |
| `jobs/` | `daemon.py` (client), `mixin.py` (`DaemonJobMixin`), `progress.py`, `process.py` (`StreamingProcess`, tree kill) |
| `dialogs/` | `confirm.py` (pre-launch confirmations + cache/checkpoint probes), `guidebook.py`, `settings.py`, `system.py` (Models + Update) |
| `tabs/` | one module per tab, the `preprocess/` package, `tensorboard.py` (overlay panel) |
| `widgets/` | reusable widgets (incl. `gpu_status.py`) |

`app.py` (MainWindow) and `theme.py` stay at the root. `gui/__init__.py` re-exports the
common names (`from gui import ROOT, variant_path, …`) **lazily**, so importing a `core`
module never pulls in Qt; `tabs/preprocess/knobs.py` is Qt-free too. `library/` imports are torch-free leaves only (e.g. `library.config.dataset_keys`,
`library.config.io`, `library.datasets.path_filter`, `library.preprocess.resize_preview`,
`library.datasets.curation_actions`, `library.downloads`); a torch/cv2-importing module
slows startup by seconds. Verify with `python -X importtime -c "import gui.app"` — torch
must not appear.

## Launch

- `make gui` → `tasks.py gui` → `scripts/tasks/gui.py::cmd_gui` → `python -m gui` →
  `gui/__main__.py` → `gui/__init__.py::main` → `gui/app.py::main`.
- `app.py::main`: `load_language()` → build + show `MainWindow` →
  `ensure_daemon_quietly()` (deferred via `QTimer.singleShot(0, ...)` so a cold daemon
  boot doesn't block the window) → Qt loop.
- `make gui-qwen` → `python -m gui.qwen21` — a separate en/cn window for the
  Qwen-Image-2.1 line (not Anima; see `library/qwen21/CLAUDE.md`). Reuses `theme`,
  `jobs.daemon`, `DaemonJobMixin`, `widgets` only; its strings live in `gui/qwen21/strings.py`,
  not `gui/i18n/`.
- `make lora-gui GUI_PRESETS=<variant>` trains from `gui-methods/` configs; it does not
  launch the GUI.

## Architecture

- **`app.py::MainWindow`** — top bar (Guidebook / Models / Update / Queue + TensorBoard
  overlay toggles / ⚙ Settings → `dialogs/settings.py::SettingsDialog`: language, theme,
  MCP registration; a language change offers an in-place rebuild via `_reload_ui`) over
  one tab set in a `QStackedWidget` shared with the Queue and TensorBoard overlays. Tabs:
  Config (MethodsTab over the LoRA family + Turbo), anime_tools, Preprocess, Merge,
  Experimental (MethodsTab over research methods + soup), EasyControl. Theme applied by
  `_dark()` → `theme.apply_theme`.
- **Lazy construction.** Tabs inherit `LazyTabMixin` (first directory scan deferred to
  first show). Every tab but Config/Preprocess also sits behind a
  `widgets.LazyTabHolder`, so the widget tree is built on first open; launch builds only
  Config, `PreprocessingTab` (the Train auto-chain needs it) and the TensorBoard panel.
  Consequences: a lazy tab's `_try_reattach` runs on first open, not at launch; code
  reaching into a lazy tab from outside goes through the holder's `.inner` (None until
  built).
- **`tabs/methods_tab.py::MethodsTab`** — plain `QWidget`: a Method dropdown over an inner
  `ConfigTab` (flat `train.py --method` methods) + the distill editors (`TurboTrainTab`,
  soup) in a `QStackedWidget`. `EasyControlTab` extends `ConfigTab`; `_DistillConfigTab`
  (distill editors' base) is standalone and lazy. Config-style tabs compose
  `DirtyTrackingMixin` (`widgets/mixins.py`) + `DaemonJobMixin` (`jobs/mixin.py`), mixed
  in **before** `LazyTabMixin`/`QWidget`.
- **`core/config_io.py`** — config discovery + merge + lint, pure TOML/pathlib.
  `merged_gui_variant_preset(variant, preset)` returns `(dict, origin_map)` (key →
  base/preset/method). Variants are auto-discovered from `configs/gui-methods/*.toml`
  `[variant]` blocks (`family`/`order`); customs live in `gui-methods/custom/`. The
  **Hardware dropdown** (ConfigTab top bar) is the `preset` axis: options from
  `list_hardware_presets()` (presets.toml sections with `[<name>.gui] group="hardware"`),
  persisted machine-wide as `hardware_preset` in `gui_settings.json`, threaded through
  every merge/save/submit by `ConfigTab._current_preset()`. Variant files must not pin
  hardware keys (method beats preset).
- **`tabs/preprocess/`** — `tab.py` =
  `PreprocessingTab`, the **cache builder** (resize → VAE → caption mirror → TE, plus
  PE). Curation (autotag, position clauses, SAM masks) is the `anime_tools` panel's:
  every env the tab builds, the Train auto-chain's included, pins `CAPTION_AUTOTAG` /
  `CAPTION_POSITION_CLAUSES` to `0` (`knobs.CURATION_GATES_OFF`), so a
  `preprocess.toml` that turns one on never starts it from the GUI. The status line flags a
  workspace caption/mask newer than the last applied Export. **Resize preview**
  (`resize_preview.py`, a non-modal dialog) lists each source image's bucket / tier /
  kept share under the tab's live resize widgets and draws the crop. Method/variant bar,
  Save + Run split buttons, status row,
  explanation panel, log, daemon observer, and the ConfigTab contract (`set_variant` /
  `preprocess_env` / `preprocess_overrides` / `preprocess_config_snapshot` /
  `persist_preprocess_inputs`). The form is a stack of **`KnobSection`**s (`_section.py`:
  a `QGroupBox` whose `add_knob(key, widget, label)` wires change→dirty, the `enabled_by`
  gate and `values()`/`set_values()`, dispatched on *widget type*).
  - Two of three sections are **`stage_form.py::StageFormSection`**s rendering an
    `anime_tools.gui.stages.schema` as a `KnobSection` keyed by the stage's dests
    (`STAGE_IDS` = `resize`, `correct`). Bound and trainer-owned dests (`TRAINER_FIELDS`)
    are hidden; trainer-native rows sit beside them via `add_trainer_knob` →
    `knob_widgets`. Values persist under `[variant.stages.<stage_id>]`, elided against
    `preprocess.toml`-seeded defaults (`seeded_defaults` / `persistable_values` /
    `merge_stages_into_meta`). An older GUI's `autotag` / `masks_sam` tables are left
    as they are on Save.
  - Sections: `image_prep.py` (`ImagePrepSection` over `resize`, with the tier / crop-anchor
    / crop-margin domain widgets mapped by dest), `text_caching.py` (trainer-native),
    `captions.py` (`CaptionEditingSection` over `correct` — the mirror TE encodes from).
  - `tab.values()` = trainer knobs, `tab.stage_values()` = `{stage_id: form}`. At submit
    the forms ride as `PREPROCESS_STAGES_JSON` in `preprocess_env()`, and
    `request_from_form` (`scripts/tasks/_common.py`) builds each request through the
    package's `build_argv` with the trainer's roots; the tab validates the same way at
    Save/Run (`_validate_stages`). Field labels/help come from
    `explanations/guides/<lang>/_stage_fields.json` (English from the schema as fallback).
  - Legacy `tab.<widget>` names (`source_dir_edit`, `shuffle_spin`, …) resolve via
    `_WIDGET_ALIASES` in `__getattr__` for one release — tests and the resize preview
    still use them; new code uses `values()` / `stage_values()` / `tab.<section>.widgets[dest]`.
    Tests monkeypatching `_load_preprocess_toml` / `read_gui_settings` must patch
    `gui.tabs.preprocess.tab`.
- **`tabs/anime_tools_tab.py::AnimeToolsTab`** (lazy) — the `anime_tools` panel in a
  `QWebEngineView`, over `gui/core/anime_tools_panel.py` (Qt-free). First open seeds the
  panel's `<home>/.anime_tools_gui.json` (Export form → `sidecars_only`; `dataset.src`
  only when the Preprocess tab's `source_image_dir` isn't `image_dataset`), reuses a panel
  already serving this home (ports 8790+, `/api/info`), else spawns `python -m
  anime_tools.gui --home … --exit-with-window` detached (not a daemon job) and reads its
  URL off `output/anime_tools_gui.log`. The server stops a few seconds after its last page
  is gone, so closing the GUI reaps it. A failed start retries on the next show. `_blank`
  links go to the system browser. `QtWebEngine*` is imported only when the tab is built;
  `app.main` sets `AA_ShareOpenGLContexts` before `QApplication`, and `MainWindow` holds a
  hidden 0×0 `QRhiWidget` so the window is RHI-composited from creation. Without it, the
  first web view makes Qt recreate the top-level window, which flashes. The view itself stays
  off-stack until its first load lands, over a `window`-colored page background, so
  Chromium's white pre-paint never shows.
- **`tabs/preprocess/knobs.py`** — the **trainer-native knob table** (`KNOBS:
  tuple[Knob]`: kind / default / `default_from` (const · `preprocess.toml` ·
  `gui_settings.json`) / env name / `persist` elision rule / snapshot flag). Only what is
  *not* a stage-request field: dataset roots + scope and the TE-cache variant knobs. The
  former curation gates are `RETIRED_KEYS`: dropped on save, but kept in
  `PREPROCESS_ONLY_KEYS` so an old variant's copy never reaches `train.py`. Pure
  functions:
  `resolved_defaults` → `load_values` (`set_variant`), `to_env` (`preprocess_env`),
  `to_overrides` (`preprocess_overrides`), `merge_into_meta` (`[variant]` pop-or-set
  elision). `PREPROCESS_ONLY_KEYS` (rows + `stages`) is ConfigTab's training-snapshot strip
  list. A knob a stage already has is not a row; a trainer-native knob = one `Knob` row +
  one `add_knob(...)` / `add_trainer_knob(...)` call, no per-knob code in the tab.
  Contract pinned byte-for-byte by `tests/test_gui_preprocess_characterization.py`
  (fixture `tests/fixtures/gui_preprocess_knobs.json`, regenerate only deliberately via
  `--write`); unit tests in `tests/test_gui_preprocess_knobs.py` and
  `tests/test_gui_stage_form.py`.
- **`jobs/daemon.py`** — client wrapper over `anima_daemon.client`. `submit_training()` /
  `submit_command()` POST to the localhost daemon; jobs are **observed** by `QTimer`
  polling of on-disk job.json / progress.jsonl / stdout.log (no thread, no SSE).
  `active_job_id()` re-attaches to a job from a previous session / the ComfyUI node / CLI.
- **`jobs/mixin.py::DaemonJobMixin`** — `_submit_job(submit_fn, *, on_fail)` (submit →
  error-check → job-id, used by every launch site) and the one 400 ms stdout observer
  every daemon-watching tab uses (`_init_job_observer` / `_watch_job` / `_poll_job` /
  `_end_job_watch` / `_stop_job`). Hosts customise it through hooks only:
  `_emit_log_line` (log sink), `_route_progress_line` (which lines feed the bar) and
  `_on_job_tick` (per-poll extras). ConfigTab uses the last two for progress.jsonl and
  the live sample gallery; its `_on_job_finished` adds the preprocess→train chain and
  queue-successor follow. ConfigTab's direct child (Test, EasyControl preprocess) feeds the
  same `_route_line`.
- **`core/submit.py`** — the submit plan, pure dict-in/dict-out: `path_scope` layering
  (`scoped_paths`), `training_snapshot` (strips `PREPROCESS_ONLY_KEYS` + merge
  bookkeeping, resolves the dataset blueprint), `preprocess_snapshot`, `preprocess_env`,
  `chain_train_spec`, `cache_dir`, `repa_requirements`. ConfigTab / EasyControl /
  PreprocessingTab pass in the widget state they own; unit tests in
  `tests/test_gui_submit.py`, snapshot contract in `tests/test_gui_snapshot_preprocess_keys.py`.
- **`widgets/`** — package re-exporting its modules (`from gui.widgets import <name>`):
  `fields.py` (`_widget(value, key)` TOML value → Qt widget, `_read(widget)` back, label /
  tooltip helpers), `mixins.py` (`LazyTabMixin`, `LazyTabHolder`, `DirtyTrackingMixin`),
  `buttons.py` (`action_button` / `apply_variant` / `SplitButtonStyle`), `target_res.py`,
  `sample_prompts.py`, `image_view.py`, `_qt_utils.py` (leaf helpers like `_no_wheel`).
  Imports are one-way — `fields.py`/`mixins.py` import the domain widgets, never the
  reverse — and nothing here imports `gui.jobs.daemon`.
- **`i18n/`** — `en/ko/ja/cn.py`, each `STRINGS: dict[str,str]` (~420–470 keys).
  `t(key, **kwargs)` falls back to English, then to the key itself. New language: see the
  `gui-changes` skill.
- **`explanations/`** — lazy-loaded help under `guides/<lang>/`: `_fields.json` (field
  tooltips), `_preprocess_fields.json` (trainer-native preprocess knobs),
  `_stage_fields.json` (stage-form overlay keyed `<stage_id>.<dest>` → `{label, help,
  choices?}`, read by `stage_form.label_for` / `help_for`), `<method>.html`, all with English
  fallback.

## Gotchas

- **Save is comment-destructive.** `config_io._save` round-trips via `toml.dumps()`. Don't
  route hand-commented files (e.g. `base.toml`) through a GUI save — edit
  presets/variants instead.
- **Tab ownership is partitioned.** `_SKIP` keys (`target_res`) are hidden from ConfigTab
  because PreprocessingTab owns them (persisted to `preprocess.toml`, not the training
  config); the retired `drop_lowres_images` / `min_pixels` stay in `_SKIP` so a stale key
  in a user's TOML never draws a widget. `_VIRTUAL_KEYS` (`use_valid`,
  `validation_split_num`) are written into per-dataset `[[datasets]]` overrides, not flat
  keys. `_BASIC` (`core/config_io.py`) controls the "Advanced" fold. A knob in the wrong tab
  drifts silently.
- **i18n key parity is manual.** Nothing enforces shared keys across the four language
  files; a missing key silently shows English. Add every string to all four (and the
  matching `_fields.json` / `.html` for help text); the `translator` agent propagates
  English → ko/ja/cn.
- **The daemon outlives the GUI.** Closing the window does not stop training.
- **Process kill must walk the tree.** A directly-spawned `QProcess`'s real work runs in a
  grandchild, so `QProcess.kill()` leaks it. Spawn a Python child with
  `jobs/process.py::StreamingProcess` (kill-safe session, `PYTHONUNBUFFERED`, decoded
  `chunk` / `line` / `finished` signals; `.kill()` walks the tree). Daemon jobs stop via
  `daemon.stop_job()`.
- **`gui_settings.json`** holds UI state (language, 6 h update-check cache, preprocess
  knobs, hardware preset) — outside `configs/` so it survives a config reset.
- **No app-wide Python event filter.** `app.installEventFilter` routes every Qt event
  through Python (~82k construction-time events, ~0.7 s of launch), and on Linux it
  segfaults the anime_tools tab: wrapping QtWebEngine's internal
  `RenderWidgetHostViewQtDelegateItem` from inside a filter recurses until the stack
  overflows. `MainWindow` filters each top-level `QWindow` instead (`_filter_window`, on
  show and on `focusWindowChanged`): right-click arrives there as `ContextMenu`, and
  tooltips are rewrapped on `MouseMove` over the widget under the cursor. Launch budget:
  `tests/test_gui_launch_speed.py`.
