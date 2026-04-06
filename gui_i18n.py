"""Internationalization for the Anima LoRA GUI."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Window / tabs
        "window_title": "Anima LoRA",
        "tab_config": "Config",
        "tab_graft": "GRAFT",
        "tab_images": "Images",

        # ConfigTab
        "preset": "Preset:",
        "save": "Save",
        "train": "Train",
        "stop": "Stop",
        "log_placeholder": "Training output will appear here...",
        "from_base": "From base.toml",
        "dataset_config": "Dataset Config",
        "save_dataset_config": "Save Dataset Config",
        "saved": "Saved",
        "saved_file": "Saved {name}",
        "dataset_saved": "Dataset config saved.",
        "invalid_toml": "Invalid TOML",
        "error": "Error",
        "accelerate_not_found": "accelerate not found on PATH",
        "preprocess": "Preprocess",
        "preprocess_required": "Please run Preprocess before training.",
        "finished": "--- Finished (exit code {code}) ---",
        "locked_by_preset": "Locked by preset (performance settings are fixed for this VRAM profile)",

        # GraftTab
        "iterations": "Iterations",
        "refresh": "Refresh",
        "select_all": "Select All",
        "invert": "Invert",
        "deselect": "Deselect",
        "n_images": "{n} images",
        "n_images_selected": "{n} images, {s} selected",
        "delete_selected": "Delete Selected",
        "delete": "Delete",
        "delete_confirm": "Delete {n} image(s)?",
        "graft_config": "GRAFT Config",
        "save_graft_config": "Save GRAFT Config",
        "graft_saved": "GRAFT config saved.",

        # ImageViewerTab
        "directory": "Directory:",
        "caption": "Caption:",
        "no_caption": "(no caption)",

        # Language
        "language": "Language:",
    },
    "ko": {
        # Window / tabs
        "window_title": "Anima LoRA",
        "tab_config": "\uc124\uc815",
        "tab_graft": "GRAFT",
        "tab_images": "\uc774\ubbf8\uc9c0",

        # ConfigTab
        "preset": "\ud504\ub9ac\uc14b:",
        "save": "\uc800\uc7a5",
        "train": "\ud559\uc2b5",
        "stop": "\uc815\uc9c0",
        "log_placeholder": "\ud559\uc2b5 \ucd9c\ub825\uc774 \uc5ec\uae30\uc5d0 \ud45c\uc2dc\ub429\ub2c8\ub2e4...",
        "from_base": "base.toml\uc5d0\uc11c \uc0c1\uc18d",
        "dataset_config": "\ub370\uc774\ud130\uc14b \uc124\uc815",
        "save_dataset_config": "\ub370\uc774\ud130\uc14b \uc124\uc815 \uc800\uc7a5",
        "saved": "\uc800\uc7a5 \uc644\ub8cc",
        "saved_file": "{name} \uc800\uc7a5\ub428",
        "dataset_saved": "\ub370\uc774\ud130\uc14b \uc124\uc815\uc774 \uc800\uc7a5\ub418\uc5c8\uc2b5\ub2c8\ub2e4.",
        "invalid_toml": "\uc798\ubabb\ub41c TOML",
        "error": "\uc624\ub958",
        "accelerate_not_found": "PATH\uc5d0\uc11c accelerate\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4",
        "preprocess": "\uc804\ucc98\ub9ac",
        "preprocess_required": "\ud559\uc2b5 \uc804\uc5d0 \uc804\ucc98\ub9ac\ub97c \uba3c\uc800 \uc2e4\ud589\ud574\uc8fc\uc138\uc694.",
        "finished": "--- \uc644\ub8cc (\uc885\ub8cc \ucf54\ub4dc {code}) ---",
        "locked_by_preset": "\ud504\ub9ac\uc14b\uc5d0 \uc758\ud574 \uc7a0\uae40 (\uc774 VRAM \ud504\ub85c\ud544\uc758 \uc131\ub2a5 \uc124\uc815\uc740 \uace0\uc815\ub418\uc5b4 \uc788\uc2b5\ub2c8\ub2e4)",

        # GraftTab
        "iterations": "\ubc18\ubcf5",
        "refresh": "\uc0c8\ub85c\uace0\uce68",
        "select_all": "\ubaa8\ub450 \uc120\ud0dd",
        "invert": "\uc120\ud0dd \ubc18\uc804",
        "deselect": "\uc120\ud0dd \ud574\uc81c",
        "n_images": "\uc774\ubbf8\uc9c0 {n}\uac1c",
        "n_images_selected": "\uc774\ubbf8\uc9c0 {n}\uac1c, {s}\uac1c \uc120\ud0dd\ub428",
        "delete_selected": "\uc120\ud0dd \uc0ad\uc81c",
        "delete": "\uc0ad\uc81c",
        "delete_confirm": "\uc774\ubbf8\uc9c0 {n}\uac1c\ub97c \uc0ad\uc81c\ud558\uc2dc\uaca0\uc2b5\ub2c8\uae4c?",
        "graft_config": "GRAFT \uc124\uc815",
        "save_graft_config": "GRAFT \uc124\uc815 \uc800\uc7a5",
        "graft_saved": "GRAFT \uc124\uc815\uc774 \uc800\uc7a5\ub418\uc5c8\uc2b5\ub2c8\ub2e4.",

        # ImageViewerTab
        "directory": "\ub514\ub809\ud1a0\ub9ac:",
        "caption": "\uce21\uc158:",
        "no_caption": "(\uce21\uc158 \uc5c6\uc74c)",

        # Language
        "language": "\uc5b8\uc5b4:",
    },
}

_current_lang = "en"
_SETTINGS_FILE = None


def _settings_path():
    global _SETTINGS_FILE
    if _SETTINGS_FILE is None:
        from pathlib import Path
        _SETTINGS_FILE = Path(__file__).resolve().parent / "gui_settings.json"
    return _SETTINGS_FILE


def load_language() -> str:
    """Load saved language preference."""
    global _current_lang
    import json
    p = _settings_path()
    if p.exists():
        try:
            _current_lang = json.loads(p.read_text(encoding="utf-8")).get("language", "en")
        except (json.JSONDecodeError, OSError):
            _current_lang = "en"
    return _current_lang


def save_language(lang: str):
    """Persist language preference."""
    global _current_lang
    import json
    _current_lang = lang
    p = _settings_path()
    settings = {}
    if p.exists():
        try:
            settings = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    settings["language"] = lang
    p.write_text(json.dumps(settings), encoding="utf-8")


def set_language(lang: str):
    global _current_lang
    _current_lang = lang


def t(key: str, **kwargs) -> str:
    """Translate a key using the current language."""
    s = TRANSLATIONS.get(_current_lang, TRANSLATIONS["en"]).get(key)
    if s is None:
        s = TRANSLATIONS["en"].get(key, key)
    if kwargs:
        s = s.format(**kwargs)
    return s


def current_language() -> str:
    return _current_lang


def available_languages() -> list[str]:
    return list(TRANSLATIONS.keys())
