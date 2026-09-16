"""Compatibility shim re-exporting ``gui.tabs.preprocess`` (``tab.py`` + one
module per section); import ``PreprocessingTab`` from there. Tests that monkeypatch the default-source loaders must patch
``gui.tabs.preprocess.tab``, not this shim."""

from __future__ import annotations

from gui.tabs.preprocess.image_prep import _ResizeCropAnchorWidget  # noqa: F401
from gui.tabs.preprocess.masking import _RuleCard  # noqa: F401
from gui.tabs.preprocess.tab import (  # noqa: F401
    LORA_CACHE_DIR,
    MASK_DIR,
    PREPROCESS_METHODS,
    PREPROCESS_TOML,
    RESIZED_DIR,
    SAM_YAML,
    PreprocessingTab,
    _load_preprocess_toml,
    _load_sam_yaml,
)

__all__ = ["PreprocessingTab"]
