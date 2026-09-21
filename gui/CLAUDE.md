# gui/CLAUDE.md

PySide6 (Qt6) desktop GUI. Root `CLAUDE.md` owns the training/config/daemon contracts this
GUI drives. Most of the surface is `tabs/config_tab.py` and the `tabs/preprocess/` package;
`tabs/image_tab.py`'s autotag worker is `tabs/_autotag.py`. For recipe-style changes
(new training field / variant / language, job submission, a new job-submitting tab,
action-button colors) **load the `gui-changes` skill**.

## What it is

Edits TOML configs and submits jobs to the daemon; no training/torch logic. `config_io.py`, `_paths.py` and `tabs/preprocess/knobs.py`
are **Qt-free** (no PySide6 import) so they stay headless-unit-testable. `library/` imports are torch-free leaves only (e.g. `library.config.dataset_keys`,
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
- `make lora-gui GUI_PRESETS=<variant>` trains from `gui-methods/` configs; it does not
  launch the GUI.

## Architecture

- **`app.py::MainWindow`** — top bar (Guidebook / Models / Update / Queue + TensorBoard
  overlay toggles / ⚙ Settings → `settings_dialog.py::SettingsDialog`: language, theme,
  MCP registration; a language change offers an in-place rebuild via `_reload_ui`) over
  one tab set in a `QStackedWidget` shared with the Queue and TensorBoard overlays. Tabs:
  Config (MethodsTab over the LoRA family + Turbo), Preprocess, Dataset, Merge,
  Experimental (MethodsTab over research methods + soup), EasyControl. Theme applied by
  `_dark()` → `theme.apply_theme`.
- **Lazy construction.** Tabs inherit `LazyTabMixin` (first directory scan deferred to
  first show). Every tab after Config/Preprocess also sits behind a
  `widgets.LazyTabHolder`, so the widget tree is built on first open; launch builds only
  Config, `PreprocessingTab` (the Train auto-chain needs it) and the TensorBoard panel.
  Consequences: a lazy tab's `_try_reattach` runs on first open, not at launch; code
  reaching into a lazy tab from outside goes through the holder's `.inner` (None until
  built — see `MainWindow._reload_image_tab_kb`).
- **`tabs/methods_tab.py::MethodsTab`** — plain `QWidget`: a Method dropdown over an inner
  `ConfigTab` (flat `train.py --method` methods) + the distill editors (`TurboTrainTab`,
  soup) in a `QStackedWidget`. `EasyControlTab` extends `ConfigTab`; `_DistillConfigTab`
  (distill editors' base) is standalone and lazy. Config-style tabs compose
  `DirtyTrackingMixin` (`widgets/mixins.py`) + `DaemonJobMixin` (`_job_mixin.py`), mixed
  in **before** `LazyTabMixin`/`QWidget`.
- **`config_io.py`** — config discovery + merge + lint, pure TOML/pathlib.
  `merged_gui_variant_preset(variant, preset)` returns `(dict, origin_map)` (key →
  base/preset/method). Variants are auto-discovered from `configs/gui-methods/*.toml`
  `[variant]` blocks (`family`/`order`); customs live in `gui-methods/custom/`. The
  **Hardware dropdown** (ConfigTab top bar) is the `preset` axis: options from
  `list_hardware_presets()` (presets.toml sections with `[<name>.gui] group="hardware"`),
  persisted machine-wide as `hardware_preset` in `gui_settings.json`, threaded through
  every merge/save/submit by `ConfigTab._current_preset()`. Variant files must not pin
  hardware keys (method beats preset).
- **`tabs/preprocess/`** (`preprocess_tab.py` is a re-export shim). `tab.py` =
  `PreprocessingTab`: method/variant bar, Save + Run split buttons, status row,
  explanation panel, log, daemon observer, and the ConfigTab contract (`set_variant` /
  `preprocess_env` / `preprocess_overrides` / `preprocess_config_snapshot` /
  `persist_preprocess_inputs`). The form is a stack of **`KnobSection`**s (`_section.py`:
  a `QGroupBox` whose `add_knob(key, widget, label)` wires change→dirty, the `enabled_by`
  gate and `values()`/`set_values()`, dispatched on *widget type*).
  - Four of five sections are **`stage_form.py::StageFormSection`**s rendering an
    `anime_tools.gui.stages.schema` as a `KnobSection` keyed by the stage's dests. Bound
    and trainer-owned dests (`TRAINER_FIELDS`) are hidden; trainer-native rows sit beside
    them via `add_trainer_knob` → `knob_widgets` (chain gate via `gate=`). Values persist
    under `[variant.stages.<stage_id>]`, elided against `preprocess.toml`-seeded defaults
    (`seeded_defaults` / `persistable_values` / `merge_stages_into_meta`).
  - Sections: `image_prep.py` (`ImagePrepSection` over `resize`, with the tier / crop-anchor
    / crop-margin domain widgets mapped by dest), `captions.py` (`AutotagSection` over
    `autotag` behind `caption_autotag`; `CaptionEditingSection` over `correct` + the
    `caption_position_clauses` gate), `masking.py` (`SamMaskSection`: `run_sam_mask` + one
    `_RuleCard` = `StageFormSection(masks_sam)` per rule, each with its own
    `path_pattern`; `[[variant.stages.masks_sam]]`), `text_caching.py` (trainer-native).
  - `tab.values()` = trainer knobs, `tab.stage_values()` = `{stage_id: form}`. At submit
    the forms ride as `PREPROCESS_STAGES_JSON` in `preprocess_env()`, and
    `request_from_form` (`scripts/tasks/_common.py`) builds each request through the
    package's `build_argv` with the trainer's roots; the tab validates the same way at
    Save/Run (`_validate_stages`). Field labels/help come from
    `explanations/guides/<lang>/_stage_fields.json` (English from the schema as fallback).
  - Legacy `tab.<widget>` names (`source_dir_edit`, `caption_autotag_chk`, …) resolve via
    `_WIDGET_ALIASES` in `__getattr__` for one release — tests and `image_tab` still use
    them; new code uses `values()` / `stage_values()` / `tab.<section>.widgets[dest]`.
    Tests monkeypatching `_load_preprocess_toml` / `read_gui_settings` / `_load_sam_yaml`
    must patch `gui.tabs.preprocess.tab`.
- **`tabs/preprocess/knobs.py`** — the **trainer-native knob table** (`KNOBS:
  tuple[Knob]`: kind / default / `default_from` (const · `preprocess.toml` ·
  `gui_settings.json`) / env name / `persist` elision rule / snapshot flag). Only what is
  *not* a stage-request field: dataset roots + scope, the TE-cache variant knobs, and the
  three chain gates (`caption_autotag` / `caption_position_clauses` / `run_sam_mask`). Pure
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
- **`daemon.py`** — client wrapper over `anima_daemon.client`. `submit_training()` /
  `submit_command()` POST to the localhost daemon; jobs are **observed** by `QTimer`
  polling of on-disk job.json / progress.jsonl / stdout.log (no thread, no SSE).
  `active_job_id()` re-attaches to a job from a previous session / the ComfyUI node / CLI.
- **`_job_mixin.py::DaemonJobMixin`** — `_submit_job(submit_fn, *, on_fail)` (submit →
  error-check → job-id, used by every launch site) and the 400 ms stdout observer
  (`_init_job_observer` / `_watch_job` / `_drain_job_stdout` / `_poll_job` / `_stop_job`,
  log sink `_emit_log_line`) used by distill + preprocess. ConfigTab/EasyControl keep
  their own observer (progress.jsonl + live sample preview + preprocess→train chain) and
  borrow only `_submit_job`.
- **`widgets/`** — package re-exporting its modules (`from gui.widgets import <name>`):
  `fields.py` (`_widget(value, key)` TOML value → Qt widget, `_read(widget)` back, label /
  tooltip helpers), `mixins.py` (`LazyTabMixin`, `LazyTabHolder`, `DirtyTrackingMixin`),
  `buttons.py` (`action_button` / `apply_variant` / `SplitButtonStyle`), `target_res.py`,
  `sample_prompts.py`, `image_view.py`, `_qt_utils.py` (leaf helpers like `_no_wheel`).
  Imports are one-way — `fields.py`/`mixins.py` import the domain widgets, never the
  reverse — and nothing here imports `gui.daemon`.
- **`i18n/`** — `en/ko/ja/cn.py`, each `STRINGS: dict[str,str]` (~540–590 keys).
  `t(key, **kwargs)` falls back to English, then to the key itself. New language: see the
  `gui-changes` skill.
- **`explanations/`** — lazy-loaded help under `guides/<lang>/`: `_fields.json` (field
  tooltips), `_preprocess_fields.json` (trainer-native preprocess knobs),
  `_stage_fields.json` (stage-form overlay keyed `<stage_id>.<dest>` → `{label, help,
  choices?}`, read by `stage_form.label_for` / `help_for`), `<method>.html`, all with English
  fallback.
- Support modules: `progress.py` (JSONL/tqdm parse), `process.py` (`kill_process_tree`),
  `tensorboard.py`, `validation.py`, `dialogs.py` (pre-launch confirmations +
  `GuidebookDialog`), `settings_dialog.py`, `discovery.py`, `system_dialog.py` (update +
  model manager), `theme.py`.

## Gotchas

- **Save is comment-destructive.** `config_io._save` round-trips via `toml.dumps()`. Don't
  route hand-commented files (e.g. `base.toml`) through a GUI save — edit
  presets/variants instead.
- **Tab ownership is partitioned.** `_SKIP` keys (`target_res`) are hidden from ConfigTab
  because PreprocessingTab owns them (persisted to `preprocess.toml`, not the training
  config); the retired `drop_lowres_images` / `min_pixels` stay in `_SKIP` so a stale key
  in a user's TOML never draws a widget. `_VIRTUAL_KEYS` (`use_valid`,
  `validation_split_num`) are written into per-dataset `[[datasets]]` overrides, not flat
  keys. `_BASIC` (`config_io.py`) controls the "Advanced" fold. A knob in the wrong tab
  drifts silently.
- **i18n key parity is manual.** Nothing enforces shared keys across the four language
  files; a missing key silently shows English. Add every string to all four (and the
  matching `_fields.json` / `.html` for help text); the `translator` agent propagates
  English → ko/ja/cn.
- **The daemon outlives the GUI.** Closing the window does not stop training.
- **Process kill must walk the tree.** A directly-spawned `QProcess`'s real work runs in a
  grandchild, so `QProcess.kill()` leaks it — use `process.py::kill_process_tree`. Daemon
  jobs stop via `daemon.stop_job()`.
- **`gui_settings.json`** holds UI state (language, 6 h update-check cache, preprocess
  knobs, hardware preset) — outside `configs/` so it survives a config reset.
- **Install app-wide event filters last.** `app.installEventFilter(self)` routes every Qt
  event through Python, including ~82k construction-time events (~0.7 s of launch), so
  `MainWindow` installs its filter after `setCentralWidget`. Launch budget:
  `tests/test_gui_launch_speed.py`.
