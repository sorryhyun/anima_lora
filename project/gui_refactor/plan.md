# gui/ refactor — plan

Behaviour-preserving cleanup of the PySide6 GUI. Architecture and invariants live in
`gui/CLAUDE.md`; this file tracks only what is done and what is left.

## Done (2026-10-09)

- **Nesting.** `gui/core/` (Qt-free: paths, config_io, submit, validation, discovery,
  anime_tools_panel, debug_report), `gui/jobs/` (daemon, mixin, progress, process),
  `gui/dialogs/` (confirm, guidebook, settings, system); `tensorboard` → `tabs/`,
  `gpu_status` → `widgets/`. The `gui/__init__.py` facade re-exports lazily, so importing a
  `gui.core` module never loads PySide6. Dropped the `tabs/preprocess_tab.py` shim.
- **Submit plan → `gui/core/submit.py`.** Snapshot / env / chain-spec / `path_scope` logic
  moved out of ConfigTab and PreprocessingTab (which no longer imports ConfigTab).
  Verified byte-identical output for every built-in variant × hardware preset.
  Tests: `tests/test_gui_submit.py`.
- **One job observer.** ConfigTab's private observer folded into `DaemonJobMixin` via the
  `_route_progress_line` / `_emit_log_line` / `_on_job_tick` hooks; every tab's
  `_on_job_finished` starts with `_end_job_watch()`. Dead `_jsonl_timer` removed.
- ConfigTab picker lock/unlock → `_set_pickers_enabled`.

- **`StreamingProcess`** (`gui/jobs/process.py`): one QObject owns the kill-safe
  `QProcess`, per-stream incremental UTF-8 decode and line splitting, and emits
  `chunk` / `line` / `finished`. Hosts: ConfigTab Test + EasyControl preprocess (lines →
  the mixin's `_route_line`), MergeTab (stdout lines for `ANALYZE_RESULT`, stderr chunks),
  `dialogs/system.py::_StreamingDialog` (chunks). EasyControl's launch env now keeps
  `PYTHONUNBUFFERED=1` (its old `QProcessEnvironment` dropped it). Left alone:
  `widgets/gpu_status.py` (a one-shot `nvidia-smi` probe read at exit, not a stream) and
  `bench/ip_adapter/impl/gui_adapter_tab.py` (bench copy, merged channels).
  Tests: `tests/test_gui_streaming_process.py`.
- **`ExplainPanel`** (`gui/widgets/explain_panel.py`): the help / gallery pane —
  `show_guide` / `show_field_help` / `show_gallery` / `mode`, the `magnify:` zoom anchor and
  the gallery-signature skip — plus `newest_images`. ConfigTab keeps only which guide /
  directory to show; the distill editors and PreprocessingTab use the same panel, so their
  field help now renders inline markdown and uses the `text_dim` token instead of `#888`.
- **Form ↔ variant file → `gui/core/variant_form.py`** (Qt-free): `FIELD_ORDER` /
  `field_sort_key`, `group_fields` (basic / advanced split), `parse_extra_args` (Windows
  backslash retry, `ExtraArgsError`), and `variant_from_form` — the whole Save writeback
  (preset values never baked in, `path_scope` → `[variant]`, validation / folder-repeat
  `[[datasets]]` override, extras last) over a `read(key, baseline)` callback. ConfigTab's
  `_reload` builds through `_origin_styles` / `_field_group_box` / `_basic_section` /
  `_advanced_section`; `_clear_form` is shared with EasyControl's descriptor form. Save
  output verified identical over every variant × hardware preset (plain, edited + extra
  args, malformed extra args: 330 cases). Tests: `tests/test_gui_variant_form.py`.
- **EasyControlTab stays a subclass**, over documented override points (ConfigTab's class
  docstring): both Train paths go through `ConfigTab._submit_training` (busy UI → submit →
  attach), so the descriptor train no longer open-codes the submit; Preprocess locks with
  the pickers via a `_set_pickers_enabled` override (drops `_ec_set_busy` and the
  `_restore_idle_ui` override); `_try_reattach` shares `_reattach`. Composition was
  rejected: the subclass reuses the whole form / job UI, and the overrides are now few
  and named.

`config_tab.py`: 1664 → 1248 lines; `easycontrol_tab.py`: 522 → 465.

## Next

Rough order. Each step should keep the submit-plan equivalence check passing: dump
`_queue_config_snapshot` / `_preprocess_config_snapshot` / `_preprocess_env` /
`_chain_train_spec` / `_resolve_cache_dir` over every variant × preset before and after,
then compare.

1. **Small items**
   - The config-warning banner hardcodes `#5c1a1a` / `#ffd9d9` / `#a33`; switch to
     theme tokens. `dialogs/guidebook.py` and other files also hardcode hex colors
     (`grep -rln "#[0-9a-fA-F]\{6\}" gui`).
   - i18n key-parity test across `gui/i18n/{en,ko,ja,cn}.py`. `gui/CLAUDE.md` currently
     says parity is manual.
   - Remove `_WIDGET_ALIASES` (`tabs/preprocess/tab.py`). It was meant to last one
     release; port its remaining users (tests, `resize_preview.py`) first.

## Out of scope

- `tabs/preprocess/` — already refactored (knob table + characterization fixture).
- `gui/qwen21/` — a separate line; it only consumes `jobs/` and `widgets/`.
