"""Main-window model mode: the Anima / MOSS dropdown, its persistence, and the
embedded MOSS panel (``gui.lines.window.build_panel``)."""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    import gui.core.paths as paths

    path = tmp_path / "gui_settings.json"
    monkeypatch.setattr(paths, "GUI_SETTINGS_FILE", path)
    return path


@pytest.mark.core
def test_mode_switch_is_lazy_persisted_and_restored(qapp, settings_file, monkeypatch):
    import gui.app as app_mod

    monkeypatch.setattr(app_mod, "check_for_update_async", lambda *a: None)
    win = app_mod.MainWindow()
    assert win._mode == "anima" and win._moss_panel is None
    assert not settings_file.exists()  # restoring at launch writes nothing

    assert win._mode_bar.property("mode") == "anima"
    win.mode_combo.setCurrentIndex(1)
    assert win._mode == "moss"
    assert win.tab_stack.currentWidget() is win._moss_panel
    assert win._mode_bar.property("mode") == "moss"
    # Preprocess / Train / the two i2i tabs and the anime_tools view share one tab
    # row, as in Anima.
    assert win._moss_panel.tabs.count() == 5
    win._moss_panel.tabs.setCurrentIndex(4)
    assert win._moss_panel._bottom.isHidden()  # no line log under anime_tools
    win._moss_panel.tabs.setCurrentIndex(0)
    # The menu stays whole in MOSS mode.
    assert not win.tensorboard_btn.isHidden() and not win.guide_btn.isHidden()
    assert json.loads(settings_file.read_text())["model_mode"] == "moss"

    # The queue overlay stays reachable in MOSS mode and returns to the MOSS page.
    win.queue_btn.setChecked(True)
    assert win.tab_stack.currentWidget() is win._queue_tab
    win.queue_btn.setChecked(False)
    assert win.tab_stack.currentWidget() is win._moss_panel
    win.close()

    again = app_mod.MainWindow()
    assert again._mode == "moss" and again.mode_combo.currentData() == "moss"
    again.mode_combo.setCurrentIndex(0)
    assert again.tab_stack.currentWidget() is again.tabs
    again.close()


def test_line_panel_claims_cli_submitted_sidecar_job(qapp, settings_file, monkeypatch):
    from gui.lines import window
    from gui.moss.app import SPEC

    panel = window.build_panel(SPEC, "en")
    jobs = {
        "gui": ("moss-test", []),
        "cli": ("train", ["scripts/moss/train.py", "--epochs", "20"]),
        "other": ("train", ["scripts/qwen21/train.py"]),
        "anima": ("lora", []),
    }
    monkeypatch.setattr(window.gui_daemon, "read_job_label", lambda j: jobs[j][0])
    monkeypatch.setattr(window.gui_daemon, "read_job_argv", lambda j: jobs[j][1])
    assert panel._job_line_label("gui") == "moss-test"
    assert panel._job_line_label("cli") == "moss-train"
    assert panel._job_line_label("other") is None
    assert panel._job_line_label("anima") is None
    panel.cleanup_subprocess()
