"""Trainer side of the daemon release-pause (library/runtime/pause.py):
request polling, ack, and the ``paused`` run_end mapping in run_scope."""

import json

import pytest

from library.runtime import pause as pause_mod
from library.runtime.pause import PauseWatcher, TrainingPaused
from library.runtime.progress import ProgressSink, run_scope


def test_watcher_polls_request_and_acks(tmp_path):
    w = PauseWatcher(str(tmp_path), min_interval=0.0)
    assert w.poll() is False
    (tmp_path / pause_mod.PAUSE_REQUEST_NAME).write_text('{"release_model": true}')
    assert w.poll() is True
    w.acknowledge("/some/state", global_step=12, epoch=3)
    assert not (tmp_path / pause_mod.PAUSE_REQUEST_NAME).exists()
    ack = json.loads((tmp_path / pause_mod.PAUSE_ACK_NAME).read_text())
    assert ack["state_dir"] == "/some/state"
    assert ack["global_step"] == 12 and ack["epoch"] == 3


def test_run_scope_maps_training_paused(tmp_path):
    path = tmp_path / "progress.jsonl"
    sink = ProgressSink(str(path), run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=10, total_epochs=1, pid=1)
    with pytest.raises(TrainingPaused):
        with run_scope(sink, final_step=lambda: 7):
            raise TrainingPaused("/ckpt-state", 7, 2)
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    end = [e for e in events if e["ev"] == "run_end"][-1]
    assert end["status"] == "paused"
    assert end["final_step"] == 7
    assert end["state_dir"] == "/ckpt-state"
