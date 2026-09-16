"""Internationalization for the Anima LoRA GUI.

Per-language string tables live in sibling modules (``en.py``, ``ko.py``,
``ja.py``, ``cn.py``). Add a new language by dropping in ``<code>.py`` exporting
``STRINGS: dict[str, str]`` and registering it in ``TRANSLATIONS`` below.
Missing keys fall back to English via ``t()``.
"""

from __future__ import annotations

from gui._paths import get_setting, set_setting
from gui.i18n import cn, en, ja, ko

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": en.STRINGS,
    "ko": ko.STRINGS,
    "cn": cn.STRINGS,
    "ja": ja.STRINGS,
}

_current_lang = "en"


def load_language() -> str:
    """Load saved language preference."""
    global _current_lang
    _current_lang = get_setting("language", "en")
    return _current_lang


def save_language(lang: str):
    """Persist language preference."""
    global _current_lang
    _current_lang = lang
    set_setting("language", lang)


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
