---
name: gui-changes
description: Recipes for common changes to the PySide6 GUI — adding a training field, variant, or language; changing job submission; composing a new job-submitting tab; and the action-button/theming rules. Load before making one of these changes under gui/.
---

# GUI common-change recipes

Architecture these plug into: `gui/CLAUDE.md`.

- **New training field**: add the TOML key → it surfaces in ConfigTab via `_widget()`
  auto-mapping (check `_SKIP`/`_BASIC` placement) → a tooltip in `_fields.json` (next
  item).
- **New UI string or help text**: the key goes into all four `gui/i18n/` tables (parity
  test: `gui/CLAUDE.md` § Gotchas); help text into each of en/ko/ja/cn under
  `guides/<lang>/` (`_fields.json` / `<method>.html`).
- **New variant**: drop `configs/gui-methods/<name>.toml` with a `[variant] family="…"`
  block; nothing else registers it.
- **New language**: `gui/i18n/<code>.py` with `STRINGS`, register in `TRANSLATIONS`, add
  `guides/<code>/` files.
- **Change job submission**: `ConfigTab._start_training` and `PreprocessingTab._submit`
  (the helper its `_run_*` handlers call). Both go through `jobs.daemon.submit_training` /
  `submit_command`, wrapped by `DaemonJobMixin._submit_job` (pass the submit call as a
  lambda + an `on_fail` rollback). What rides in the job (config snapshot, env, chain
  spec, `path_scope`) is built by `gui/core/submit.py`; change it there.
- **New job-submitting / config-editing tab**: compose `DaemonJobMixin` +
  `DirtyTrackingMixin` (mixin order: `gui/CLAUDE.md` § `tabs/methods_tab.py`). Call
  `_init_job_observer()` in `__init__`, set `self._dirty`, provide
  `_on_job_finished(state)` (start it with `job_id = self._end_job_watch()`), and override
  a host hook (`gui/CLAUDE.md` § `jobs/mixin.py`) only where its default does not fit. The
  Save button's dirty look is the mixin's default `"warning"` variant; override
  `_save_btn_dirty_variant` only to change it.
- **A colored action button**: colors live in one table: `theme.ACTION_COLORS`
  (saturated `primary`/`secondary`/`info`/`success`/`danger`/`warning`/`busy`) and
  `theme.NAV_COLORS` (muted top-bar toggles); never inline `setStyleSheet` colors. Build
  with `widgets.action_button(text, variant=…, tooltip=…, on_click=…)`; flip state
  (idle↔busy) with `widgets.apply_variant(btn, variant_or_None)` (sets a dynamic property
  + repolishes; the global stylesheet `[variant="…"]` rules do the painting, incl.
  hover/pressed/disabled). **Split buttons** (`SplitButtonStyle`) must use a *per-widget*
  stylesheet instead, `setStyleSheet(theme.action_button_qss(variant))`: a global
  stylesheet bypasses the proxy style and miscentres the menu label. Spacing magic
  numbers have tokens too: `theme.Pad` / `theme.Gap`.
