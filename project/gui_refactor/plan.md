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

`config_tab.py`: 1664 → 1488 lines.

## Next

Rough order. Each step should keep the submit-plan equivalence check passing: dump
`_queue_config_snapshot` / `_preprocess_config_snapshot` / `_preprocess_env` /
`_chain_train_spec` / `_resolve_cache_dir` over every variant × preset before and after,
then compare.

1. **`StreamingProcess` helper** (`gui/jobs/`). Five copies of the QProcess pattern
   (`setup_kill_safe` + readyRead + line buffering + finished): ConfigTab Test,
   EasyControl `_ec_launch`, MergeTab (+ `ANALYZE_RESULT` marker), `dialogs/system.py`
   `_StreamingDialog`, `widgets/gpu_status.py`. Make it one QObject that emits lines and
   a finished signal; the hosts keep only their own line handling.
2. **Explanation / gallery panel out of ConfigTab.** `_show_explain*`,
   `_set_explain_html`, `_on_explain_anchor`, `_render_image_gallery`,
   `_newest_images`, `_show_test_output`, `_show_sample_output` (~150 lines) become a
   `widgets/` panel that ConfigTab owns. Watch the `_explain_mode` / `_gallery_sig`
   state that the job tick reads.
3. **Split ConfigTab's form building.** `_reload` (~120 lines, contains the nested
   `_build_subgroup_box`) and `_save_preset` (~110 lines: path_scope meta,
   validation/folder-repeat writeback, extra-args TOML parse). Move the pure
   "form values → variant dict" part into `gui/core/` next to `config_io` so it can be
   tested headless.
4. **EasyControlTab.** It subclasses ConfigTab and overrides 10+ methods (`_ec_*`, its
   own QProcess path, `_attach_to_job` / `_restore_idle_ui` / `_try_reattach`). Once
   1–3 shrink ConfigTab, decide between composition and a smaller documented set of
   override points.
5. **Small items**
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
