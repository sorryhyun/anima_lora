# gui/CLAUDE.md

Guidance for the PySide6 (Qt6) desktop GUI. Scoped to `gui/` — read the root `CLAUDE.md`
for the training/config/daemon contracts this GUI drives. ~19.9k lines across 53 Python
files; `tabs/config_tab.py` (1665) and the `tabs/preprocess/` package (`tab.py` 1019 +
section modules ~930 + the Qt-free knob table `knobs.py` 515) hold most of the surface
(module-layout cleanup complete — history in `_archive/proposals/gui_refactor_plan.md`;
`tabs/image_tab.py`'s autotag worker lives in `tabs/_autotag.py`).

## What it is

A thin **front-end over the existing pipeline** — it edits TOML configs and submits jobs
to the daemon; it does **not** contain training/torch logic. `config_io.py`, `_paths.py`
and `tabs/preprocess/knobs.py` are deliberately **Qt-free** (no PySide6 import) so they
stay headless-unit-testable; keep them that way. The only `library/` imports are
torch-free leaves (`library.config.dataset_keys`, `library.config.io`,
`library.datasets.path_filter`) — don't pull torch/cv2-importing modules into the GUI or
you slow startup by seconds (verify with `python -X importtime -c "import gui.app"`;
torch must not appear).

## Launch

- `make gui` → `tasks.py gui` → `scripts/tasks/gui.py::cmd_gui` → `python -m gui`.
- `python -m gui` → `gui/__main__.py` → `gui/__init__.py::main` → `gui/app.py::main`.
- `app.py::main` (gui/app.py:377): `load_language()` → `ensure_daemon_quietly()` → build
  `MainWindow` → Qt loop.
- The legacy CLI `make lora-gui GUI_PRESETS=<variant>` is a *training* entry (runs
  `gui-methods/` configs directly); it is not this GUI. The GUI submits to the daemon
  instead.

## Architecture

- **`app.py::MainWindow`** — top bar (Guidebook / Models / Update / Queue + TensorBoard
  overlay toggles / ⚙ Settings at top right — language + the checkout-specific MCP
  registration (`claude mcp add` line + generic `mcpServers` JSON) via `SettingsDialog`,
  now in `settings_dialog.py`; a language change offers an immediate in-place window
  rebuild (`_reload_ui`) instead of requiring an app restart) + one tab set (Config =
  MethodsTab picker over the LoRA family + the promoted Turbo distiller, Preprocess,
  Dataset, Merge, Experimental = MethodsTab picker over the research methods + soup
  pipeline, EasyControl) in a `QStackedWidget` with the global TensorBoard and Queue
  overlay views. Dark `QPalette` via `_dark()`. `GuidebookDialog` + `_guidebook_path`
  live in `dialogs.py`; `_prefer_cleartype_font_engine` lives in `theme.py` next to
  `_load_bundled_fonts`.
- **Tabs** (`gui/tabs/`) inherit `LazyTabMixin` (first directory scan deferred to first
  view). On top of that, every tab after Config/Preprocess (Dataset, Merge,
  Experimental, EasyControl) sits behind a `widgets.LazyTabHolder` in `app.py` — the
  whole widget tree is only *constructed* on first open, so launch builds just the
  Config `MethodsTab` (eager inner `ConfigTab`), `PreprocessingTab` (the Train
  auto-chain needs it), and the TensorBoard panel; the daemon + Train button are live
  immediately. Consequence: a lazy tab's `ConfigTab._try_reattach` runs on first open,
  not at launch (the Queue overlay is the global job view either way), and code reaching
  into a lazy tab from outside must go through the holder's `.inner` (None until built —
  see `MainWindow._reload_image_tab_kb`). `MethodsTab` is a plain `QWidget` wrapper: a
  Method dropdown over an inner `ConfigTab` (flat `train.py --method` methods) + the
  distill editors (`TurboTrainTab` / soup) in a `QStackedWidget`; it backs both the
  **Config** tab (LoRA family + Turbo) and the **Experimental** tab (research methods +
  soup). `EasyControlTab` extends `ConfigTab` (inherits its dirty/observer machinery);
  `_DistillConfigTab` (the distill editors' base) is standalone and **lazy** — its TOML
  scan + `_try_reattach` run from `_lazy_init` on first show. The config-style tabs
  compose two shared mixins instead of re-implementing: `DirtyTrackingMixin`
  (`widgets/mixins.py`) + `DaemonJobMixin` (`_job_mixin.py`) — mix them in **before**
  `LazyTabMixin`/`QWidget`. A tab keeps only what genuinely differs (busy-UI,
  `_on_job_finished` teardown, log sink); don't copy the dirty/submit/poll boilerplate
  back into a tab.
- **`config_io.py`** — config discovery + merge + lint, all pure TOML/pathlib.
  `merged_gui_variant_preset(variant, preset)` returns `(dict, origin_map)` (origin_map
  = which key came from base/preset/variant). Variants are **auto-discovered** from
  `configs/gui-methods/*.toml` `[variant]` blocks (`family`/`order`) — adding a variant
  is one new file, no map to update; custom ones live in `gui-methods/custom/`. The
  **Hardware dropdown** (ConfigTab top bar) feeds the `preset` axis: options come from
  `list_hardware_presets()` (presets.toml sections tagged `[<name>.gui]
  group="hardware"`), the choice persists machine-wide in `gui_settings.json`
  (`hardware_preset`), and `ConfigTab._current_preset()` threads it through every
  merge/save/submit — variant files must not pin the hardware keys (method beats
  preset).
- **`tabs/preprocess/`** — the Preprocessing tab as a package
  (`gui/tabs/preprocess_tab.py` is a re-export shim). `tab.py` = `PreprocessingTab`:
  method/variant bar, Save + Run split buttons, status row, explanation panel, log,
  daemon observer, and the ConfigTab contract (`set_variant` / `preprocess_env` /
  `preprocess_overrides` / `preprocess_config_snapshot` / `persist_preprocess_inputs`).
  The form is a stack of **`KnobSection`** panels (`_section.py`: a `QGroupBox` whose
  `add_knob(key, widget, label)` registers the editor for a knob-table key and wires
  change→dirty, the `enabled_by` gate and the generic `values()`/`set_values()` —
  dispatched on *widget type*, so a `float` knob may be a spin or a free-text edit):
  `image_prep.py` (`ImagePrepSection` + `_ResizeCropAnchorWidget`/`_CropMarginsWidget`),
  `text_caching.py`, `captions.py` (`AutotagSection`, `CaptionEditingSection`),
  `masking.py` (`_RuleCard`, `SamMaskSection` with `collect_rules()`, `MitMaskSection`).
  `tab.values()` merges the sections. Legacy `tab.<widget>` names (`source_dir_edit`,
  `caption_autotag_chk`, …) resolve through `_WIDGET_ALIASES` in `__getattr__` for one
  release — tests and `image_tab` still use them; new code goes through `values()` or
  `tab.<section>.widgets[key]`. Tests that monkeypatch `_load_preprocess_toml` /
  `read_gui_settings` / `_load_sam_yaml` must patch `gui.tabs.preprocess.tab`.
- **`tabs/preprocess/knobs.py`** — the Preprocessing tab's **knob table** (`KNOBS:
  tuple[Knob]`, one row per key: kind / default / `default_from` policy (const ·
  `preprocess.toml` · `gui_settings.json` · `sam_mask.yaml`) / env name / `persist`
  elision rule / snapshot flag) plus the pure functions the tab calls —
  `resolved_defaults` → `load_values` (what `set_variant` shows), `to_env`
  (`preprocess_env`), `to_overrides` (`preprocess_overrides`), `merge_into_meta` (the
  `[variant]` pop-or-set elision, incl. the caption-master "compare against the
  TOML-resolved default" rule). `PREPROCESS_ONLY_KEYS` is derived from it and is
  ConfigTab's training-snapshot strip list. **Adding a preprocess knob = one `Knob` row +
  one `add_knob(...)` call in its section** (a domain widget needs
  `value()`/`set_value()`/`changed`) — never re-open a per-knob ladder in the tab.
  Contract pinned byte-for-byte by `tests/test_gui_preprocess_characterization.py`
  (fixture `tests/fixtures/gui_preprocess_knobs.json`, regenerate only deliberately via
  `--write`) and unit-tested Qt-free in `tests/test_gui_preprocess_knobs.py`. Phases 2
  and 4 (`caption_drop_groups`, collapsible/pipeline-ordered sections) in
  `docs/proposal/gui_preprocess_tab_refactor.md`.
- **`daemon.py`** — GUI-side client wrapper over `anima_daemon.client`.
  `submit_training()` / `submit_command()` POST to the localhost daemon; the GUI only
  **observes** jobs by polling files on disk (job.json / progress.jsonl / stdout.log)
  via a `QTimer`. No background thread, no SSE. `active_job_id()` re-attaches to a job
  from a previous session / the ComfyUI node / CLI on restart.
- **`widgets/`** (a package since the Phase 2 split) — re-exports everything from
  sibling modules so `from gui.widgets import <name>` is unchanged: `fields.py` (the
  field factory — `_widget(value, key)` maps a TOML value → Qt widget by type,
  `_read(widget)` maps back — plus
  `ClickableLabel`/`make_field_label`/`hint_label`/tooltip-wrap helpers), `mixins.py`
  (`LazyTabMixin`, `DirtyTrackingMixin` — Save-button dirty state via
  `_connect_dirty_signal`/`_mark_dirty`/`_clear_dirty`/`_update_save_button`, mixed into
  every config-style tab), `buttons.py`
  (`action_button`/`apply_variant`/`SplitButtonStyle`), `target_res.py` (the multi-scale
  tier row), `sample_prompts.py` (the sample-prompt editor), `image_view.py`
  (`ScaledImageLabel`/`ImageViewerDialog`), `_qt_utils.py` (tiny leaf helpers, e.g.
  `_no_wheel`, shared by modules that must not import each other). Import direction is
  one-way — `fields.py`/`mixins.py` import the domain widgets (`target_res.py`,
  `sample_prompts.py`), never the reverse. Stays Qt-only, no `gui.daemon` import (keeps
  it cycle-free).
- **`_job_mixin.py`** — `DaemonJobMixin`: the daemon plumbing every job-submitting tab
  shares. `_submit_job(submit_fn, *, on_fail)` wraps submit → error-check →
  job-id-extract (used by all launch sites); the stdout *observer* (`_init_job_observer`
  / `_watch_job` / `_drain_job_stdout` / `_poll_job` / `_stop_job`, log sink =
  overridable `_emit_log_line`) drives the 400 ms tail-and-poll loop for the standalone
  observers (distill, preprocess). ConfigTab/EasyControl keep their **richer** observer
  (progress.jsonl + live sample preview + preprocess→train auto-chain) and only borrow
  `_submit_job`.
- **`i18n/`** — one module per language (`en/ko/ja/cn.py`), each exporting `STRINGS:
  dict[str,str]` (~370 keys). `t(key, **kwargs)` (gui/i18n/__init__.py:72) looks up
  current lang then **falls back to English, then to the key itself** — so a key missing
  from `ko.py` silently shows English, not an error. Register new languages in
  `TRANSLATIONS`.
- **`explanations/`** — lazy-loaded help: `guides/<lang>/_fields.json` (per-field
  tooltips) + `_preprocess_fields.json` + `<method>.html` (per-method overviews). Same
  English fallback.
- Support modules: `progress.py` (JSONL/tqdm parse), `process.py` (`kill_process_tree`
  via psutil), `tensorboard.py`, `validation.py`, `dialogs.py` (pre-launch confirmation
  dialogs + `GuidebookDialog`), `settings_dialog.py` (`SettingsDialog` + MCP
  registration helpers), `discovery.py`, `system_dialog.py` (update + model manager).

## Gotchas

- **Save is comment-destructive.** `config_io._save` round-trips via `toml.dumps()`,
  stripping comments. Don't route hand-commented files (e.g. `base.toml`) through a GUI
  save if the comments matter — edit presets/variants instead.
- **Tab ownership is partitioned — edit the right one.** `_SKIP` keys (`target_res`,
  `drop_lowres_images`, `min_pixels`) are hidden from ConfigTab because
  **PreprocessingTab** owns them (they persist to `preprocess.toml`, not the training
  config). `_VIRTUAL_KEYS` (`use_valid`, `validation_split_num`) are not flat TOML keys
  — ConfigTab writes them into per-dataset `[[datasets]]` overrides. `_BASIC`
  (config_io.py:269) controls the collapsible "Advanced" fold. Putting a knob in the
  wrong tab causes silent drift.
- **i18n key parity is manual.** The 4 language files are independent; nothing enforces
  that they share keys. A missing key just falls back to English. When you add a string,
  add it to all four (and the matching `_fields.json` / `.html` if it's help text) — see
  the `translator` agent for propagating English → ko/ja/cn.
- **Daemon outlives the GUI.** Closing the window does not stop training; jobs live in
  the daemon. The GUI is a pure observer of on-disk job files.
- **Process kill must walk the tree.** Training is a grandchild of any directly-spawned
  `QProcess`, so `QProcess.kill()` alone leaks it — use `process.py::kill_process_tree`
  (psutil). Daemon jobs are stopped via `daemon.stop_job()`.
- **`gui_settings.json`** holds UI state (language, 6h update-check cache, preprocess
  knobs) — separate from `configs/` so it survives a config reset.
- **Install app-wide event filters LAST.** `app.installEventFilter(self)` routes *every*
  Qt event through Python, including the ChildAdded/Polish/LayoutRequest storm of
  building the widget tree. `MainWindow` installed its context-menu filter at the top of
  `__init__` and paid ~82k Python round-trips (~0.7s of a 2.05s launch) for events it
  drops — it now installs after `setCentralWidget`, since ContextMenu/ToolTip can't fire
  before the window is shown. Same rule for any new filter: if it only handles
  interaction events, install it once the tree exists. Guarded by
  `tests/test_gui_launch_speed.py` (which also records the remaining launch costs).

## Common changes

For the recipe-style changes — new training field, new variant, new language, changing
job submission, composing a new job-submitting tab, and the action-button/theming rules
(`theme.ACTION_COLORS` / `widgets.action_button` / `apply_variant`; never inline
`setStyleSheet` colors) — **load the `gui-changes` skill** before making the change.
