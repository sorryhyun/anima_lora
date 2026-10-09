"""Shared path constants for the GUI package.

Imported by every ``gui`` submodule; deliberately dependency-free (no Qt,
no other ``gui`` imports) to avoid import cycles.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Persistent UI state (language, update-check cache, preprocess knobs, prefs).
# Separate from configs/ so it survives a config reset.
GUI_SETTINGS_FILE = ROOT / "gui" / "gui_settings.json"

DEFAULT_THEME_COLOR = "#3c78c8"  # backward compat; live accent comes from gui/theme.py
DEFAULT_THEME = "dark"  # one of "dark" / "light" / "sepia" (gui/theme.py THEMES)


def read_gui_settings() -> dict:
    """Whole gui_settings.json as a dict (``{}`` if absent/unparseable)."""
    if not GUI_SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(GUI_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def get_setting(key: str, default=None):
    """Read one preference from gui_settings.json, or ``default`` if missing."""
    return read_gui_settings().get(key, default)


def set_setting(key: str, value) -> None:
    """Persist one preference into gui_settings.json (merge, don't clobber)."""
    settings = read_gui_settings()
    settings[key] = value
    try:
        GUI_SETTINGS_FILE.write_text(json.dumps(settings), encoding="utf-8")
    except OSError:
        pass


METHODS_DIR = CONFIGS_DIR / "methods"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
CUSTOM_DIR = CONFIGS_DIR / "custom"
# User-created variants get their own subdir so they don't pollute the built-in family list.
CUSTOM_VARIANTS_DIR = GUI_METHODS_DIR / "custom"


_METHOD_ORDER = (
    "lora",
    "tlora",
    "hydralora",
    "soft_tokens",
    "easycontrol",
)
