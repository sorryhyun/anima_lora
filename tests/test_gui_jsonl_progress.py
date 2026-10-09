"""Phase-0 GUI cutover: JsonlProgressReader tails progress.jsonl onto a bar.

Runs the Qt widget headless via the offscreen platform plugin. Skips cleanly
if PySide6 isn't installed in the test environment.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QProgressBar  # noqa: E402

from gui.jobs.progress import JsonlProgressReader  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_inactive_until_first_event(qapp, tmp_path):
    bar = QProgressBar()
    reader = JsonlProgressReader(bar)
    reader.watch(str(tmp_path / "missing.jsonl"))
    reader.poll()  # file absent
    assert reader.active is False


def test_drives_bar_from_step_events(qapp, tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    bar = QProgressBar()
    reader = JsonlProgressReader(bar)
    reader.watch(path)

    _write(
        path,
        [
            {"ev": "run_start", "ts": 0.0, "total_steps": 200, "total_epochs": 4},
            {"ev": "step", "ts": 1.0, "global_step": 50, "epoch": 1, "loss": 0.4},
        ],
    )
    reader.poll()
    assert reader.active is True
    assert bar.maximum() == 200
    assert bar.value() == 50


def test_incremental_tail(qapp, tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    bar = QProgressBar()
    reader = JsonlProgressReader(bar)
    reader.watch(path)

    _write(
        path, [{"ev": "run_start", "ts": 0.0, "total_steps": 100, "total_epochs": 1}]
    )
    reader.poll()
    assert bar.value() == 0

    # append a step without rewriting the head — reader resumes at saved pos
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ev": "step", "global_step": 30, "epoch": 0}) + "\n")
    reader.poll()
    assert bar.value() == 30


def test_reset_clears_state(qapp, tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    bar = QProgressBar()
    reader = JsonlProgressReader(bar)
    reader.watch(path)
    _write(path, [{"ev": "run_start", "ts": 0.0, "total_steps": 10, "total_epochs": 1}])
    reader.poll()
    assert reader.active is True
    reader.reset()
    assert reader.active is False
