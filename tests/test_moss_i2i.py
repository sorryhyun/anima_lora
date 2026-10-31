"""MOSS i2i: the task recipes load against their requests, ``generate`` expands a
reference into sized jobs, and the GUI's i2i tab chains its jobs through the panel."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from library.moss.requests import (
    I2I_CACHE,
    GenerateRequest,
    TrainRequest,
    i2i_config,
)


@pytest.mark.parametrize("task", sorted(I2I_CACHE))
def test_task_recipes_load(task):
    cache = I2I_CACHE[task].load_config(i2i_config(task, "cache"))
    train = TrainRequest.load_config(i2i_config(task, "train"))
    gen = GenerateRequest.load_config(i2i_config(task, "generate"))
    out = cache.get("out", I2I_CACHE[task]().out)
    assert train["cache"] == out
    # The test reference matches what the cache encoded.
    ref_scale = cache.get("ref_scale", I2I_CACHE[task]().ref_scale)
    assert gen.get("ref_scale", GenerateRequest().ref_scale) == ref_scale
    # int8 refuses the swap these run.
    assert train["int8"] == "off" and train["blocks_to_swap"] > 0


def test_ref_jobs_follow_each_reference(tmp_path):
    from library.moss.generate import _fingerprint, _jobs

    Image.new("RGB", (1200, 1700)).save(tmp_path / "page one.png")
    Image.new("RGB", (800, 800)).save(tmp_path / "b.webp")
    (tmp_path / "notes.txt").write_text("x")
    req = GenerateRequest(prompt="", ref_image=str(tmp_path), ref_vision=True)
    jobs = _jobs(req)
    assert [(j["key"], j["width"], j["height"]) for j in jobs] == [
        ("000_b", 800, 800),  # never upscaled
        ("000_page_one", 864, 1216),
    ]
    assert all(j["prompt"] == "" and j["ref_vision"] for j in jobs)
    # A reference edit invalidates the saved latent.
    assert _fingerprint(req, jobs[0])["ref_mtime"] == jobs[0]["ref_mtime"]

    # Any reference may stand alone (sanitize's fixed caption), vision or not;
    # without one an empty prompt is an error.
    alone = _jobs(GenerateRequest(prompt="", ref_image=str(tmp_path)))
    assert [j["prompt"] for j in alone] == ["", ""]
    assert not any(j["ref_vision"] for j in alone)
    with pytest.raises(SystemExit, match="no prompts"):
        _jobs(GenerateRequest(prompt=""))


def test_plain_jobs_unchanged():
    """No reference: the job dicts (and so saved-latent fingerprints) carry no
    ref keys."""
    from library.moss.generate import _jobs

    (job,) = _jobs(GenerateRequest(prompt="a", prompt_format="text"))
    assert not {k for k in job if k.startswith("ref")}


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_i2i_chain_submits_through_panel(qapp, monkeypatch):
    import gui.lines.window as window
    import gui.moss.i2i as i2i
    from gui.moss.app import SPEC

    store = {}
    for mod in (window, i2i):
        monkeypatch.setattr(mod, "get_setting", lambda k, d=None: store.get(k, d))
        monkeypatch.setattr(mod, "set_setting", store.__setitem__)
    monkeypatch.setattr(window.LinePanel, "_reattach", lambda self: None)
    panel = window.build_panel(SPEC, "en")
    sent = []
    monkeypatch.setattr(
        panel, "_submit", lambda req, label, env=None: sent.append((label, env)) or True
    )
    monkeypatch.setattr(panel, "_model_ok", lambda req: True)
    prep, train = panel._pages
    assert panel.tabs.tabText(panel._line_pages - 1) == train.title
    task = train.i2i.current.task
    train.i2i.run_chain()
    assert sent == [(i2i.STAGE, {"TASK": task})]
    assert [j.label for j in panel._pending] == [i2i.CACHE, i2i.TRAIN]
    assert panel._pending[1].req.cache == panel._pending[0].req.out
    # Only edits against the task's seed are saved.
    assert store[i2i.SETTINGS_KEY][task]["train"] == {}
    panel.cleanup_subprocess()
