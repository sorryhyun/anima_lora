"""``gui.core.submit`` (the Qt-free submit plan) and the shared job-observer hooks."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from gui.core import submit


def test_submit_plan_imports_without_qt():
    code = (
        "import sys, gui.core.submit\n"
        "assert not any(m.startswith('PySide6') for m in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("data_group1", "data_group1"),
        (" a\\b/ ", "a/b"),
        ("x/*", "x"),
        ("bad/../x", None),
        ("p|q", None),
        ("a*b", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_path_scope(raw, expected):
    assert submit.normalize_path_scope(raw) == expected


def test_scoped_paths_appends_scope_once():
    merged = {
        "path_scope": "g1",
        "output_dir": "out/ckpt",
        "lora_cache_dir": "post_image_dataset/lora/g1",
        "variant": {"family": "lora"},
    }
    out = submit.scoped_paths(merged)
    assert out["output_dir"] == "out/ckpt/g1"
    assert out["lora_cache_dir"] == "post_image_dataset/lora/g1"
    assert out["source_image_dir"] == "image_dataset/g1"
    assert "path_scope" not in out and "variant" not in out
    assert merged["output_dir"] == "out/ckpt"  # input untouched


def test_scoped_paths_unscoped_is_identity():
    merged = {"output_dir": "out"}
    assert submit.scoped_paths(merged) is merged


def test_preprocess_snapshot_strips_meta_and_none():
    merged = {
        "method": "lora",
        "preset": "default",
        "path_scope": "g1",
        "preprocess_path_pattern": "*.png",
        "keep": 1,
        "drop": None,
    }
    snap = submit.preprocess_snapshot(merged, {"caption_shuffle_variants": 2})
    assert snap["keep"] == 1 and snap["caption_shuffle_variants"] == 2
    assert snap["source_image_dir"] == "image_dataset/g1"
    for key in ("method", "preset", "path_scope", "preprocess_path_pattern", "drop"):
        assert key not in snap


def test_chain_spec_and_env():
    assert submit.chain_train_spec("lora", "low_vram") == {
        "method": "lora",
        "preset": "low_vram",
        "methods_subdir": "gui-methods",
    }
    env = submit.preprocess_env("lora", "default", {"PRESET": "x", "A": "1"})
    assert env == {
        "METHOD": "lora",
        "METHODS_SUBDIR": "gui-methods",
        "PRESET": "x",
        "A": "1",
    }


@pytest.mark.parametrize(
    "use_repa, expected", [(True, True), ("yes", True), (False, False), (None, False)]
)
def test_repa_requirements(use_repa, expected):
    require_pe, encoder = submit.repa_requirements({"use_repa": use_repa})
    assert require_pe is expected and encoder == "pe_spatial"


def test_observer_routes_lines_and_flushes_tail(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication

    from gui.jobs.mixin import DaemonJobMixin

    QApplication.instance() or QApplication([])

    class _Tracker:
        def __init__(self):
            self.fed: list[str] = []

        def feed(self, line):
            if line.startswith("50%|"):
                self.fed.append(line)
                return True
            return False

    class _Host(DaemonJobMixin, QObject):
        def __init__(self):
            super().__init__()
            self._progress_tracker = _Tracker()
            self.lines: list[str] = []
            self._init_job_observer()

        def _emit_log_line(self, line):
            self.lines.append(line)

    host = _Host()
    tail = host._consume_lines("a\n50%|##   \rb\r\npart")
    assert tail == "part"
    assert host.lines == ["a", "b"]
    assert host._progress_tracker.fed == ["50%|##   "]

    log = tmp_path / "stdout.log"
    log.write_text("x\ny-tail", encoding="utf-8")
    host._job_id = "job"
    host._stdout_tailer.watch(log)
    assert host._end_job_watch() == "job"
    assert host.lines[-2:] == ["x", "y-tail"]
    assert host._job_id is None and host._stdout_buf == ""
