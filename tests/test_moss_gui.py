"""``make gui-moss``: torch-free at launch, and the GUI's scan agrees with the cache."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from library.moss.cache import LATENT_SUFFIX, TE_SUFFIX, caption_sha
from library.moss.scan import cache_counts, scan

ROOT = Path(__file__).resolve().parents[1]


def test_gui_is_torch_free():
    code = (
        "import sys, gui.moss.app, library.moss.scan; sys.exit('torch' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "gui.moss.app imports torch"


def _te(path: Path, caption: str) -> None:
    save_file(
        {"embeds": torch.zeros(3, 8, dtype=torch.bfloat16)},
        path,
        metadata={"caption_sha256": caption_sha(caption), "max_text_tokens": "4096"},
    )


def test_scan_counts_prefixed_stems_and_stale_captions(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    for sub in ("x", "y"):
        (src / sub).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(src / sub / "img.png")
        (src / sub / "img.txt").write_text(f"caption {sub}\n", encoding="utf-8")
    Image.new("RGB", (32, 32)).save(src / "x" / "nocap.png")
    out.mkdir()
    _te(out / f"x__img{TE_SUFFIX}", "caption x")  # current
    _te(out / f"y__img{TE_SUFFIX}", "old caption")  # caption edited since
    save_file({"latent": torch.zeros(1)}, out / f"x__img_32x32{LATENT_SUFFIX}")

    s = scan(src, out)
    assert (s.images, s.pairs, s.duplicates) == (3, 2, 2)
    assert (s.text_cached, s.stale_text, s.latents_cached) == (1, 1, 1)
    c = cache_counts(out)
    assert (c.text, c.latents, c.samples) == (2, 1, 1)


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


def test_train_summary_names_cache_subfolders(panel, tmp_path):
    set_a = tmp_path / "set_a"
    set_a.mkdir()
    (tmp_path / "empty").mkdir()
    save_file({"embeds": torch.zeros(1)}, set_a / f"img{TE_SUFFIX}")
    save_file({"latent": torch.zeros(1)}, set_a / f"img_32x32{LATENT_SUFFIX}")

    panel.train_form.set_values({"cache": str(tmp_path)})
    panel._rescan()
    assert "set_a (1)" in panel.train_summary.text()
    assert "empty" not in panel.train_summary.text()

    panel.train_form.set_values({"cache": str(set_a)})
    panel._rescan()
    assert panel.train_summary.text() == "1 cached samples ready."

    panel.train_form.set_values({"cache": str(tmp_path / "empty")})
    panel._rescan()
    assert "run preprocessing first" in panel.train_summary.text()


def test_test_form_items_set_the_int8_render(panel):
    from library.moss.requests import GenerateRequest

    form = panel.test_form
    combo = form.widgets["model"]
    assert combo.currentText() == "base (bf16)"

    _pick(combo, "base (int8)")
    req = form.request()
    assert (req.model, req.int8, req.stream, req.blocks_to_swap) == (
        "base",
        "on",
        "fp32",
        0,
    )
    assert "--int8" in req.to_argv()

    _pick(combo, "base (bf16)")
    assert form.request() == GenerateRequest()


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
