"""Phase 0 structured-progress sink: schema + tail-while-write smoke tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from library.training.progress import (
    ProgressSink,
    _find_cmmd,
    _flatten_logs,
    read_status,
)


def _load_run_status():
    """``scripts/`` is not an installed package — load the CLI by path."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_status.py"
    spec = importlib.util.spec_from_file_location("run_status", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_status"] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load_run_status()


def _read_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_resolve_path_default_derives_to_sibling_logs_dir(tmp_path):
    ckpt = tmp_path / "ckpt"
    args = argparse.Namespace(
        progress_jsonl=None, output_dir=str(ckpt), output_name="my_run"
    )
    resolved = ProgressSink.resolve_path(args)
    # sibling logs/ dir, not the checkpoint dir
    assert resolved == os.path.join(str(tmp_path), "logs", "my_run.progress.jsonl")


def test_resolve_path_disable_tokens():
    base = dict(output_dir="/tmp/x", output_name="r")
    for tok in ("", "  ", "none", "OFF", "None"):
        args = argparse.Namespace(progress_jsonl=tok, **base)
        assert ProgressSink.resolve_path(args) is None


def test_resolve_path_explicit_override(tmp_path):
    explicit = str(tmp_path / "custom.jsonl")
    args = argparse.Namespace(
        progress_jsonl=explicit, output_dir="/ignored", output_name="r"
    )
    assert ProgressSink.resolve_path(args) == explicit


def test_full_lifecycle_schema(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="run", method="lora", preset="default", t0=0.0)
    sink.run_start(total_steps=100, total_epochs=4, pid=4242)
    sink.log({"loss": 0.5, "lr": 1e-4}, global_step=10, epoch=1)
    sink.log({"loss/val_average": 0.03, "loss/val_cmmd": 0.03}, global_step=10, epoch=1)
    sink.ckpt(global_step=10, path="output/ckpt/run-step10.safetensors")
    sink.run_end(status="ok", final_step=100)

    evs = _read_events(path)
    kinds = [e["ev"] for e in evs]
    assert kinds == ["run_start", "step", "val", "ckpt", "run_end"]

    start = evs[0]
    assert start["run"] == "run" and start["method"] == "lora"
    assert start["total_steps"] == 100 and start["pid"] == 4242

    step = evs[1]
    assert step["global_step"] == 10 and step["epoch"] == 1 and step["loss"] == 0.5

    val = evs[2]
    assert val["cmmd"] == 0.03 and val["global_step"] == 10

    assert evs[3]["path"].endswith("run-step10.safetensors")
    assert evs[4]["status"] == "ok" and evs[4]["final_step"] == 100
    # every line carries an event tag + timestamp
    assert all("ev" in e and "ts" in e for e in evs)


def test_stopped_run_end(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=10, total_epochs=1, pid=1)
    sink.run_end(status="stopped", final_step=3)
    evs = _read_events(path)
    assert evs[-1]["ev"] == "run_end" and evs[-1]["status"] == "stopped"


def test_log_before_run_start_is_noop(tmp_path):
    # Sink not yet opened (no run_start) → log/ckpt must not create the file.
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None)
    sink.log({"loss": 1.0}, global_step=1, epoch=0)
    sink.ckpt(global_step=1, path="x")
    assert not os.path.exists(path)


def test_tail_while_write(tmp_path):
    # A reader can open + read the file while the sink keeps appending
    # (the concurrency contract the daemon relies on).
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=2, total_epochs=1, pid=1)
    sink.log({"loss": 0.1}, global_step=1, epoch=0)
    # read mid-run, before run_end
    mid = _read_events(path)
    assert [e["ev"] for e in mid] == ["run_start", "step"]
    sink.run_end(status="ok", final_step=2)
    assert [e["ev"] for e in _read_events(path)][-1] == "run_end"


def test_log_mirror_writes_log_events(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=1, total_epochs=1, pid=1)
    sink.attach_log_mirror()
    logging.getLogger("anima.test").warning("boom %d", 7)
    logging.getLogger("anima.test").info("quiet")  # below WARNING → not mirrored
    sink.run_end(status="ok", final_step=1)
    # close() detached the handler: later records must not write (or raise)
    logging.getLogger("anima.test").warning("after-close")

    evs = _read_events(path)
    logs = [e for e in evs if e["ev"] == "log"]
    assert len(logs) == 1
    assert logs[0]["level"] == "WARNING"
    assert logs[0]["logger"] == "anima.test"
    assert logs[0]["msg"] == "boom 7"
    assert not any(e.get("msg") == "after-close" for e in evs)


def test_log_mirror_cap(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=1, total_epochs=1, pid=1)
    sink.attach_log_mirror(max_events=2)
    for i in range(5):
        logging.getLogger("anima.test").warning("w%d", i)
    sink.run_end(status="ok", final_step=1)

    logs = [e for e in _read_events(path) if e["ev"] == "log"]
    # 2 mirrored records + the cap notice, nothing past the cap
    assert len(logs) == 3
    assert [e["msg"] for e in logs[:2]] == ["w0", "w1"]
    assert "cap" in logs[2]["msg"]


def test_log_mirror_before_run_start_is_noop(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None)
    sink.attach_log_mirror()  # stream not open yet → must not attach
    logging.getLogger("anima.test").warning("early")
    assert not os.path.exists(path)
    sink.close()


def test_flatten_logs_drops_nonscalar():
    flat = _flatten_logs({"loss": 0.5, "ok": True, "name": "x", "arr": [1, 2, 3]})
    assert flat == {"loss": 0.5, "ok": True, "name": "x"}


def test_find_cmmd():
    assert _find_cmmd({"loss/val_cmmd": 0.042, "loss": 1.0}) == 0.042
    assert _find_cmmd({"loss": 1.0}) is None


# --- read_status (issue #4: the "where is this run at" digest, so callers don't
# export the TB events file and reimplement the parse) ---


def _write_stream(tmp_path, events, *, torn: str = "") -> str:
    path = tmp_path / "run.progress.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events) + torn)
    return str(path)


def _run_start(**over) -> dict:
    ev = dict(
        ev="run_start",
        ts=0,
        run="r",
        method="turbo",
        preset=None,
        total_steps=1000,
        total_epochs=1,
        pid=os.getpid(),
    )
    ev.update(over)
    return ev


def test_read_status_live_run_reports_step_rate_eta_and_ckpt(tmp_path):
    events = [_run_start()]
    # 10s per 100 steps → 10 it/s; 600 steps left → 60s ETA.
    events += [
        {"ev": "step", "ts": i * 10.0, "global_step": i * 100, "epoch": 0, "loss": 0.5}
        for i in range(1, 5)
    ]
    events.append({"ev": "ckpt", "ts": 41.0, "global_step": 400, "path": "/ck.sft"})
    events.append(
        {"ev": "log", "ts": 42.0, "level": "WARNING", "logger": "x", "msg": "!"}
    )
    # a torn trailing line (live stream, mid-write) must not break the read
    st = read_status(_write_stream(tmp_path, events, torn='{"ev": "step", "ts'))

    assert st["status"] == "running"  # our own pid → alive
    assert (st["global_step"], st["total_steps"], st["pct"]) == (400, 1000, 40.0)
    assert st["rate"] == pytest.approx(10.0)
    assert st["eta"] == pytest.approx(60.0)
    assert st["metrics"] == {"loss": 0.5}
    assert st["ckpt"]["path"] == "/ck.sft"
    assert st["warnings"] == 1


def test_read_status_terminal_states_come_from_run_end(tmp_path):
    for status, final in (("ok", 1000), ("stopped", 300), ("error", 12)):
        events = [
            _run_start(pid=1),  # alive pid must NOT override a real run_end
            {"ev": "step", "ts": 1.0, "global_step": final, "epoch": 0},
            {
                "ev": "run_end",
                "ts": 2.0,
                "status": status,
                "final_step": final,
                "error": "RuntimeError: boom" if status == "error" else None,
            },
        ]
        st = read_status(_write_stream(tmp_path, events))
        assert st["status"] == status
        assert st["global_step"] == final
        assert st["eta"] is None  # never an ETA for a run that has stopped
    assert st["error"] == "RuntimeError: boom"


def test_read_status_flags_a_vanished_pid_as_dead(tmp_path):
    # No run_end and the writer is gone: killed / OOMed, not "still running".
    events = [
        _run_start(pid=_unused_pid()),
        {"ev": "step", "ts": 1.0, "global_step": 5},
    ]
    assert read_status(_write_stream(tmp_path, events))["status"] == "dead"


def test_read_status_val_event_and_empty_stream(tmp_path):
    events = [_run_start(), {"ev": "val", "ts": 1.0, "global_step": 100, "cmmd": 0.3}]
    st = read_status(_write_stream(tmp_path, events))
    assert st["val"]["cmmd"] == 0.3
    assert st["global_step"] is None and st["rate"] is None  # no step events yet

    empty = tmp_path / "empty.progress.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        read_status(str(empty))


def _unused_pid() -> int:
    """A pid that is not running (walk down from an implausible one)."""
    for pid in range(4_194_303, 4_194_000, -1):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except Exception:
            continue
    raise RuntimeError("no free pid found")


# --- run_status.py stream discovery (issues.md §1: the default launch path is
# the daemon, whose per-job stream lives outside output/logs) ---


def _daemon_stream(jobs_dir, job_id: str, run: str) -> None:
    """Write what a daemon train job produces: <jobs>/<id>/progress.jsonl.

    The daemon overrides --progress_jsonl with a per-job path (manager._build_cmd),
    so the filename is bare — the run name exists only inside `run_start`.
    """
    d = jobs_dir / job_id
    d.mkdir(parents=True)
    (d / "progress.jsonl").write_text(
        json.dumps(_run_start(run=run))
        + "\n"
        + json.dumps({"ev": "step", "ts": 1.0, "global_step": 5, "epoch": 0})
        + "\n"
    )


def test_run_status_finds_daemon_job_streams(tmp_path):
    """`make run-status RUN=<name>` used to exit 1 on a healthy daemon run: it
    scanned only output/logs, where a daemon run leaves the snapshot + TB events
    but no progress.jsonl."""
    logs, jobs = tmp_path / "logs", tmp_path / "jobs"
    logs.mkdir()
    (logs / "inline_run.progress.jsonl").write_text(
        json.dumps(_run_start(run="x")) + "\n"
    )
    _daemon_stream(jobs, "20260901-180249-1334d2", "cjk_unmask_a")

    found = rs._find_streams(logs, jobs)
    assert len(found) == 2  # both launch paths, one list

    # a daemon run resolves by its run name (matched inside the stream)…
    hit = rs._resolve("cjk_unmask_a", logs, jobs)
    assert hit.parent.name == "20260901-180249-1334d2"
    assert rs._stream_run_name(hit) == "cjk_unmask_a"
    # …and by the job id, which is all a daemon-status listing gives you
    assert rs._resolve("20260901-180249-1334d2", logs, jobs) == hit
    # the flat logs/ name still resolves from its filename, no file read needed
    assert rs._resolve("inline_run", logs, jobs).name == "inline_run.progress.jsonl"

    # a genuine miss still fails loudly, naming both places it looked
    with pytest.raises(SystemExit) as ei:
        rs._resolve("nope", logs, jobs)
    assert "logs" in str(ei.value) and "jobs" in str(ei.value)


def test_run_status_skips_command_jobs_and_survives_empty_dirs(tmp_path):
    """A command job's dir exists but never materializes a progress.jsonl."""
    logs, jobs = tmp_path / "logs", tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "20260901-1200-abcdef").mkdir()  # command job: no stream
    assert rs._find_streams(logs, jobs) == []  # missing logs dir is not an error
    assert rs._find_streams(logs, None) == []
