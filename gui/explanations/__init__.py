"""Bilingual help text for config fields and LoRA variant descriptions.

Per-field tooltips live in ``guides/<lang>/_fields.json`` and
``guides/<lang>/_preprocess_fields.json`` — one JSON file per language,
loaded lazily on first access. Missing keys fall back to English.
``guides/<lang>/_stage_fields.json`` is the overlay for the Preprocessing
tab's ``anime_tools`` stage forms: keyed ``<stage_id>.<dest>`` →
``{"label", "help", "choices"?}``, read by ``tabs/preprocess/stage_form``
(``label_for`` / ``help_for``) with the schema's English as the fallback.
The key scheme is the package's, so the file is the seed of a package-side
translation table (proposal §4).

Method/variant guide HTML blocks live under ``guides/<lang>/<name>.html``
and are also loaded lazily. Shared snippets (``_apply_note``,
``_not_mergeable``) follow the same convention with an underscore prefix.
"""

from __future__ import annotations

import functools
import html
import json
import re
from pathlib import Path

from gui.i18n import current_language

_GUIDES_DIR = Path(__file__).parent / "guides"

# Inline markdown used in the field-help strings: `code` spans and **bold**.
# The method guides are authored as real HTML, so the help panel renders rich
# text; the per-field JSON strings instead lean on these markers (e.g.
# "run `make mask`"). Convert them to the same bare <code>/<b> tags the guides
# use so both surfaces render identically.
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")


def field_help_html(text: str) -> str:
    """HTML-escape *text*, then render its inline markdown (`code`, **bold**).

    Returns a fragment safe to drop straight into the help panel — escaping
    happens before marker conversion, so literal ``<``/``&`` in the prose stay
    literal while the markdown markers become tags."""
    escaped = html.escape(text)
    escaped = _MD_CODE.sub(r"<code>\1</code>", escaped)
    escaped = _MD_BOLD.sub(r"<b>\1</b>", escaped)
    return escaped


@functools.lru_cache(maxsize=None)
def _read_guide(name: str, lang: str) -> str:
    path = _GUIDES_DIR / lang / f"{name}.html"
    if not path.exists():
        path = _GUIDES_DIR / "en" / f"{name}.html"
    return path.read_text(encoding="utf-8")


def _guide(name: str) -> str:
    return _read_guide(name, current_language())


@functools.lru_cache(maxsize=None)
def _read_fields(lang: str) -> dict[str, str]:
    path = _GUIDES_DIR / lang / "_fields.json"
    if not path.exists():
        path = _GUIDES_DIR / "en" / "_fields.json"
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _read_preprocess_fields(lang: str) -> dict[str, str]:
    path = _GUIDES_DIR / lang / "_preprocess_fields.json"
    if not path.exists():
        path = _GUIDES_DIR / "en" / "_preprocess_fields.json"
    return json.loads(path.read_text(encoding="utf-8"))


def field_help(key: str) -> str | None:
    """Return the help string for *key* in the current language, or None."""
    lang = current_language()
    value = _read_fields(lang).get(key)
    if value is not None:
        return value
    if lang != "en":
        return _read_fields("en").get(key)
    return None


def preprocess_field_help(key: str) -> str | None:
    """Per-field help for the Preprocessing tab. Falls back to field_help."""
    lang = current_language()
    value = _read_preprocess_fields(lang).get(key)
    if value is not None:
        return value
    if lang != "en":
        en_value = _read_preprocess_fields("en").get(key)
        if en_value is not None:
            return en_value
    return field_help(key)


@functools.lru_cache(maxsize=None)
def _read_stage_fields(lang: str) -> dict[str, dict]:
    path = _GUIDES_DIR / lang / "_stage_fields.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stage_field(key: str) -> dict | None:
    """The overlay entry for a stage form field (``"<stage_id>.<dest>"``):
    the current language's, with any key it lacks filled from English.
    ``None`` when neither names it (the schema's English is used then)."""
    lang = current_language()
    own = _read_stage_fields(lang).get(key)
    base = _read_stage_fields("en").get(key) if lang != "en" else None
    if own is None and base is None:
        return None
    return {**(base or {}), **(own or {})}


def preprocess_guide() -> str:
    return _guide("preprocess")


# Methods that can't be baked into a plain DiT via scripts/toolkits/merge_to_dit.py (router is
# layer-local / hook-only / not a weight delta) — render the "not mergeable" callout.
_NOT_MERGEABLE = frozenset({"hydralora", "fera", "chimera", "soft_tokens"})
_KNOWN_METHODS = frozenset(
    {
        "lora",
        "tlora",
        "hydralora",
        "fera",
        "chimera",
        "soft_tokens",
        "turbo",
        "soup",
        "easycontrol",
        "colorize",
    }
)


def method_guide(method: str) -> str | None:
    """Right-panel default HTML for *method*, or None if no guide is registered."""
    if method not in _KNOWN_METHODS:
        return None
    parts = [_guide("_apply_note")]
    if method in _NOT_MERGEABLE:
        parts.append(_guide("_not_mergeable"))
    parts.append(_guide(method))
    return "".join(parts)


def method_overview(method: str) -> str | None:
    """Translated method guide body *without* the Apply / not-mergeable chrome.

    For surfaces that have no Apply button (the distill/methods tab for
    ``turbo``) and just want the localized overview. Returns None if no guide
    is registered, so callers can fall back to their own text.
    """
    if method not in _KNOWN_METHODS:
        return None
    return _guide(method)
