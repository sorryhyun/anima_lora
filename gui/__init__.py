"""Anima LoRA — PySide6 GUI package.

Subpackages: ``core`` (Qt-free paths / config / validation / discovery / submit
plan), ``jobs`` (daemon client, observer mixin, progress, process kill),
``dialogs``, ``widgets``, ``tabs``, ``i18n``, ``explanations``, ``qwen21``.

The package root re-exports the commonly used names (``from gui import <name>``)
lazily, so importing a ``gui.core`` module never pulls in PySide6.
"""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, tuple[str, ...]] = {
    "gui.core.paths": (
        "CONFIGS_DIR",
        "CUSTOM_DIR",
        "CUSTOM_VARIANTS_DIR",
        "DEFAULT_THEME_COLOR",
        "GUI_METHODS_DIR",
        "GUI_SETTINGS_FILE",
        "IMAGE_EXTS",
        "METHODS_DIR",
        "PRESETS_FILE",
        "ROOT",
        "get_setting",
        "set_setting",
    ),
    "gui.core.config_io": (
        "_BASIC",
        "_GROUPS",
        "_K2G",
        "_SKIP",
        "_VIRTUAL_KEYS",
        "_builtin_variants_by_family",
        "_dataset_lint_sources",
        "_load",
        "_load_all_presets",
        "_load_base",
        "_read_variant_metadata",
        "_save",
        "custom_preset_path",
        "custom_variant_path",
        "dataset_cache_root",
        "default_lora_cache_dir",
        "default_mask_dir",
        "default_resized_dir",
        "is_basic_field",
        "is_custom_preset",
        "is_custom_variant",
        "lint_variant_configs",
        "list_gui_variants",
        "list_hardware_presets",
        "list_methods",
        "list_presets",
        "merged_gui_variant_preset",
        "merged_method_preset",
        "remove_unknown_dataset_keys",
        "variant_metadata",
        "variant_path",
    ),
    "gui.dialogs.confirm": (
        "confirm_existing_caches",
        "confirm_resumable_checkpoint",
        "confirm_train_using_cache",
        "count_preprocess_caches",
        "find_resumable_checkpoint",
    ),
    "gui.core.discovery": (
        "_adapter_dirs",
        "_imgs",
        "_safetensors_in",
    ),
    "gui.core.validation": (
        "_base_folder_repeats",
        "apply_folder_repeats_choice",
        "apply_validation_choice",
    ),
    "gui.widgets": (
        "ClickableLabel",
        "DirtyTrackingMixin",
        "LazyTabMixin",
        "ScaledImageLabel",
        "_SamplePromptsWidget",
        "_no_wheel",
        "_read",
        "_TargetResWidget",
        "_widget",
        "make_field_label",
    ),
}

_ORIGIN = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = [*_ORIGIN, "main"]


def __getattr__(name: str):
    mod = _ORIGIN.get(name)
    if mod is None:
        raise AttributeError(f"module 'gui' has no attribute {name!r}")
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value
    return value


def main():
    from gui.app import main as _main

    _main()
