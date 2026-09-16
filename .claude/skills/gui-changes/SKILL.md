---
name: gui-changes
description: Recipes for common changes to the PySide6 GUI — adding a training field, variant, or language; changing job submission; composing a new job-submitting tab; and the action-button/theming rules. Load before making one of these changes under gui/.
---

# GUI common-change recipes

Architecture these plug into: `gui/CLAUDE.md`.

- **New training field**: add the TOML key → it surfaces in ConfigTab via `_widget()`
  auto-mapping (check `_SKIP`/`_BASIC` placement) → add a tooltip to
  `guides/en/_fields.json` and replicate to ko/ja/cn.
- **New variant**: drop `configs/gui-methods/<name>.toml` with a `[variant] family="…"`
  block — auto-discovered.
- **New language**: `gui/i18n/<code>.py` with `STRINGS`, register in `TRANSLATIONS`, add
  `guides/<code>/` files.
- **Change job submission**: `ConfigTab._start_training` and `PreprocessingTab._submit`
  (the helper its `_run_*` handlers call) — both go through `daemon.submit_training` /
  `submit_command`, wrapped by `DaemonJobMixin._submit_job` (pass the submit call as a
  lambda + an `on_fail` rollback).
- **New job-submitting / config-editing tab**: compose `DaemonJobMixin` +
  `DirtyTrackingMixin` rather than hand-rolling. Call `_init_job_observer()` in
  `__init__`, set `self._dirty`, provide `_on_job_finished(state)`, and override
  `_emit_log_line` only if your log sink isn't `appendPlainText`. The Save button's dirty
  look is the mixin's default `"warning"` variant — override `_save_btn_dirty_variant`
  only to change it.
- **A colored action button**: colors live in one table — `theme.ACTION_COLORS`
  (saturated `primary`/`secondary`/`info`/`success`/`danger`/`warning`/`busy`) and
  `theme.NAV_COLORS` (muted top-bar toggles); never inline `setStyleSheet` colors. Build
  with `widgets.action_button(text, variant=…, tooltip=…, on_click=…)`; flip state
  (idle↔busy) with `widgets.apply_variant(btn, variant_or_None)` (sets a dynamic property
  + repolishes — the global stylesheet `[variant="…"]` rules do the painting, incl.
  hover/pressed/disabled). **Split buttons** (`SplitButtonStyle`) must use a *per-widget*
  stylesheet instead — `setStyleSheet(theme.action_button_qss(variant))` — because a
  global stylesheet bypasses the proxy style and miscentres the menu label. Spacing magic
  numbers have tokens too: `theme.Pad` / `theme.Gap`.
