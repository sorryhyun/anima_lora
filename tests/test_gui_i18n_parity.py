"""Every GUI language table carries every English key, with the same
``str.format`` fields, and nothing English doesn't have."""

from __future__ import annotations

import importlib
import string

import pytest

_LANGS = ("ko", "ja", "cn")
_EN = importlib.import_module("gui.i18n.en").STRINGS


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


@pytest.mark.parametrize("lang", _LANGS)
def test_same_keys_as_english(lang):
    table = importlib.import_module(f"gui.i18n.{lang}").STRINGS
    assert [k for k in _EN if k not in table] == []
    assert [k for k in table if k not in _EN] == []


@pytest.mark.parametrize("lang", _LANGS)
def test_same_format_fields_as_english(lang):
    table = importlib.import_module(f"gui.i18n.{lang}").STRINGS
    assert [k for k in _EN if k in table and _fields(table[k]) != _fields(_EN[k])] == []
