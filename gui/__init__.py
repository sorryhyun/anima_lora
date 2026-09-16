"""Anima LoRA — PySide6 GUI package.

The package root is a thin facade over :mod:`gui._paths` and these submodules:

* :mod:`gui.config_io`   — variant/preset discovery, load/save, merge, lint (Qt-free)
* :mod:`gui.validation`  — validation-split encoding (Qt-free)
* :mod:`gui.dialogs`     — resume/cache confirmation popups + on-disk probes
* :mod:`gui.discovery`   — image/adapter/dataset directory walks (Qt-free)
* :mod:`gui.widgets`     — LazyTabMixin, the config-form field factory, ScaledImageLabel

Their public names are re-exported here (``from gui import <name>``).
"""

from __future__ import annotations

from gui._paths import (
    CONFIGS_DIR,
    CUSTOM_DIR,
    CUSTOM_VARIANTS_DIR,
    DEFAULT_AUTOTAG_CONFIDENCE,
    DEFAULT_GROUP_CELL_MATCH_MIN,
    DEFAULT_GROUP_MATCH_FRAC_MIN,
    DEFAULT_THEME_COLOR,
    GUI_METHODS_DIR,
    GUI_SETTINGS_FILE,
    IMAGE_EXTS,
    METHODS_DIR,
    PRESETS_FILE,
    ROOT,
    get_setting,
    set_setting,
)
from gui.config_io import (
    _BASIC,
    _GROUPS,
    _K2G,
    _SKIP,
    _VIRTUAL_KEYS,
    _builtin_variants_by_family,
    _dataset_lint_sources,
    _load,
    _load_all_presets,
    _load_base,
    _read_variant_metadata,
    _save,
    custom_preset_path,
    custom_variant_path,
    dataset_cache_root,
    default_lora_cache_dir,
    default_mask_dir,
    default_resized_dir,
    is_basic_field,
    is_custom_preset,
    is_custom_variant,
    lint_variant_configs,
    list_gui_variants,
    list_hardware_presets,
    list_methods,
    list_presets,
    merged_gui_variant_preset,
    merged_method_preset,
    remove_unknown_dataset_keys,
    variant_metadata,
    variant_path,
)
from gui.dialogs import (
    confirm_existing_caches,
    confirm_resumable_checkpoint,
    confirm_train_using_cache,
    count_preprocess_caches,
    find_resumable_checkpoint,
)
from gui.discovery import (
    _adapter_dirs,
    _image_dirs,
    _imgs,
    _safetensors_in,
)
from gui.validation import (
    _base_folder_repeats,
    apply_folder_repeats_choice,
    apply_validation_choice,
)
from gui.widgets import (
    ClickableLabel,
    DirtyTrackingMixin,
    LazyTabMixin,
    ScaledImageLabel,
    _SamplePromptsWidget,
    _no_wheel,
    _read,
    _TargetResWidget,
    _widget,
    make_field_label,
)

__all__ = [
    "ROOT",
    "CONFIGS_DIR",
    "IMAGE_EXTS",
    "METHODS_DIR",
    "GUI_METHODS_DIR",
    "PRESETS_FILE",
    "CUSTOM_DIR",
    "CUSTOM_VARIANTS_DIR",
    "GUI_SETTINGS_FILE",
    "DEFAULT_AUTOTAG_CONFIDENCE",
    "DEFAULT_GROUP_CELL_MATCH_MIN",
    "DEFAULT_GROUP_MATCH_FRAC_MIN",
    "DEFAULT_THEME_COLOR",
    "get_setting",
    "set_setting",
    "ClickableLabel",
    "DirtyTrackingMixin",
    "LazyTabMixin",
    "ScaledImageLabel",
    "_SamplePromptsWidget",
    "_TargetResWidget",
    "_no_wheel",
    "_read",
    "_widget",
    "make_field_label",
    "_load",
    "_load_base",
    "_save",
    "default_lora_cache_dir",
    "default_resized_dir",
    "default_mask_dir",
    "dataset_cache_root",
    "_load_all_presets",
    "_builtin_variants_by_family",
    "_read_variant_metadata",
    "_dataset_lint_sources",
    "_GROUPS",
    "_K2G",
    "_SKIP",
    "_BASIC",
    "_VIRTUAL_KEYS",
    "is_basic_field",
    "list_methods",
    "list_gui_variants",
    "list_hardware_presets",
    "list_presets",
    "is_custom_variant",
    "is_custom_preset",
    "custom_variant_path",
    "custom_preset_path",
    "variant_path",
    "variant_metadata",
    "lint_variant_configs",
    "remove_unknown_dataset_keys",
    "merged_method_preset",
    "merged_gui_variant_preset",
    "apply_validation_choice",
    "apply_folder_repeats_choice",
    "_base_folder_repeats",
    "confirm_resumable_checkpoint",
    "confirm_existing_caches",
    "confirm_train_using_cache",
    "count_preprocess_caches",
    "find_resumable_checkpoint",
    "_imgs",
    "_safetensors_in",
    "_adapter_dirs",
    "_image_dirs",
    "main",
]


def main():
    from gui.app import main as _main

    _main()
