"""``make gui-moss``: torch-free at launch, every language complete, the model
combo switches the whole plan."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.core
def test_gui_is_torch_free():
    code = (
        "import sys, gui.moss.app, library.moss.scan; sys.exit('torch' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "gui.moss.app imports torch"


@pytest.mark.parametrize("lang", ["ko", "ja", "cn"])
def test_every_basic_field_is_translated(lang):
    """The GUI's basic (non-advanced) fields all carry a translated label + help."""
    import dataclasses

    from gui.moss.app import SPEC
    from library.moss.requests import I2I_CACHE

    fields = SPEC.fields[lang]
    fields_test = SPEC.fields_test[lang]
    classes = {SPEC.cache_cls, SPEC.train_cls, SPEC.test_cls, *I2I_CACHE.values()}
    missing = [
        f"{cls.__name__}.{f.name}"
        for cls in classes
        for f in dataclasses.fields(cls)
        if not f.metadata.get("advanced")
        and f.name not in fields
        and not (cls is SPEC.test_cls and f.name in fields_test)
    ]
    assert not missing


def test_ui_keys_match_english():
    from gui.lines import strings as line_strings
    from gui.moss.strings import UI

    for table in (UI, line_strings.UI):
        for lang, strings in table.items():
            assert set(strings) == set(table["en"]), lang


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(monkeypatch, qapp):
    from gui.lines import window
    from gui.moss.app import SPEC

    monkeypatch.setattr(window, "get_setting", lambda key, default=None: default)
    monkeypatch.setattr(window, "_spec", SPEC)
    monkeypatch.setattr(window, "_lang", "en")
    monkeypatch.setattr(window.LinePanel, "_reattach", lambda self: None)
    p = window.LinePanel(SPEC)
    yield p
    p.cleanup_subprocess()


def _pick(combo, text: str) -> None:
    combo.setCurrentText(text)
    combo.activated.emit(combo.currentIndex())


@pytest.mark.core
def test_model_items_change_the_whole_plan(panel):
    import tomllib

    from library.moss.requests import TrainRequest

    form = panel.train_form
    combo = form.widgets["model"]
    assert combo.currentText() == "base (int8)"
    assert form.changed_values() == {}

    bf16 = tomllib.loads((ROOT / "configs/moss/lora_bf16.toml").read_text("utf-8"))
    _pick(combo, "sft (bf16)")
    req = form.request()
    assert req.model == "sft"
    assert {k: getattr(req, k) for k in bf16} == bf16

    _pick(combo, "base (int8)")
    assert form.request() == TrainRequest()


def test_model_item_follows_the_int8_field_and_saved_values(panel):
    form = panel.train_form
    combo = form.widgets["model"]
    form.set_values({"model": "sft", "int8": "off", "activation_memory_budget": 0.1})
    assert combo.currentText() == "sft (bf16)"
    form.set_values({"int8": "fwd"})
    assert combo.currentText() == "sft (int8)"
    # Following the field, or a variant change on one plan, writes nothing back.
    _pick(combo, "base (int8)")
    req = form.request()
    assert (req.model, req.int8, req.activation_memory_budget) == ("base", "fwd", 0.1)


@pytest.mark.core
def test_i2i_forms_return_to_the_tasks_bf16_plan(qapp):
    from gui.lines import window
    from gui.moss.app import SPEC
    from gui.moss.i2i import _Task

    window._spec, window._lang = SPEC, "en"
    task = _Task("sanitize", {}, lambda *_: None, lambda *_: None)
    train = task.forms["train"]
    combo = train.widgets["model"]
    assert combo.currentText() == "base (bf16)"
    assert task.state()["train"] == {}

    _pick(combo, "base (int8)")
    assert train.request().int8 == "stream"
    assert task.state()["train"]["blocks_to_swap"] == 20
    assert train.request().activation_memory_budget == 0.1

    _pick(combo, "base (bf16)")
    assert task.state()["train"] == {}
