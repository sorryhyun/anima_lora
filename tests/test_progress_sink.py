"""Phase 0 structured-progress sink: schema + tail-while-write smoke tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from library.runtime.progress import (
    ProgressSink,
    command_job_path,
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


def test_command_job_path_prefers_the_daemon_job_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)
    assert command_job_path("logs/run.progress.jsonl") == "logs/run.progress.jsonl"
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    assert command_job_path("logs/run.progress.jsonl") == str(
        tmp_path / "progress.jsonl"
    )


@pytest.mark.core
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


@pytest.mark.core
def test_read_status_flags_a_vanished_pid_as_dead(tmp_path):
    # No run_end and the writer is gone: killed / OOMed, not "still running".
    events = [
        _run_start(pid=_unused_pid()),
        {"ev": "step", "ts": 1.0, "global_step": 5},
    ]
    assert read_status(_write_stream(tmp_path, events))["status"] == "dead"


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
    # …and by the job id, which is all a `daemon-jobs` listing gives you
    assert rs._resolve("20260901-180249-1334d2", logs, jobs) == hit
    # the flat logs/ name still resolves from its filename, no file read needed
    assert rs._resolve("inline_run", logs, jobs).name == "inline_run.progress.jsonl"

    # a genuine miss still fails loudly, naming both places it looked
    with pytest.raises(SystemExit) as ei:
        rs._resolve("nope", logs, jobs)
    assert "logs" in str(ei.value) and "jobs" in str(ei.value)


def test_run_status_falls_back_to_the_job_record(tmp_path, capsys):
    """A bench job writes no progress stream; its run name still answers, with
    the daemon-jobs line (state, rc, job dir) instead of a bare miss."""
    logs, jobs = tmp_path / "logs", tmp_path / "jobs"
    logs.mkdir()

    def job(job_id, *argv):
        (jobs / job_id).mkdir(parents=True)
        rec = {"id": job_id, "kind": "command", "argv": list(argv), "state": "done"}
        (jobs / job_id / "job.json").write_text(json.dumps(rec | {"returncode": 0}))

    job("20261121-094910-3d8f94", "x2_head.py", "--label", "x2b4k_1121_0950")
    job("20261121-095445-c6dba4", "x2_head.py", "--init", "x2b4k_1121_0950")

    argv = ["--logs-dir", str(logs), "--jobs-dir", str(jobs)]
    assert rs.main(["x2b4k_1121_0950", *argv]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("no progress stream for 'x2b4k_1121_0950'")
    assert len(out) == 2  # the job that only passes --init <name> is not that run
    assert "20261121-094910-3d8f94" in out[1] and "rc=0" in out[1]
    assert out[1].endswith(str(jobs / "20261121-094910-3d8f94"))

    # a job id resolves the same way
    assert rs.main(["20261121-095445-c6dba4", *argv]) == 0
    assert "20261121-095445-c6dba4" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rs.main(["nope", *argv])


def test_run_status_names_a_running_job_before_its_stream(tmp_path, monkeypatch):
    """A train job writes run_start only after its model loads; until then a bare
    run-status showed the previous (dead) run instead of the job on the GPU."""
    jobs = tmp_path / "jobs"
    _daemon_stream(jobs, "20261023-222131-3bd611", "old")
    new = jobs / "20261023-222244-7072b5"
    new.mkdir()
    rec = {"id": new.name, "method": "moss-i2i-train", "state": "running", "pid": 1}
    (new / "job.json").write_text(json.dumps(rec))
    monkeypatch.setattr(rs, "_pid_alive", lambda pid: True)
    assert rs._streamless_running_job(jobs)["id"] == new.name

    (new / "progress.jsonl").write_text(json.dumps(_run_start(run="new")) + "\n")
    assert rs._streamless_running_job(jobs) is None  # its stream speaks for it

    (new / "progress.jsonl").unlink()
    monkeypatch.setattr(rs, "_pid_alive", lambda pid: False)
    assert rs._streamless_running_job(jobs) is None  # a stale "running" record


# --- the console side and the trailing rate (one log system for every trainer) ---


def test_console_lines_follow_the_run(tmp_path, capsys):
    sink = ProgressSink(None, run="r", method="x1", preset=None, unit="rollout")
    sink.run_start(
        total_steps=100, total_epochs=1, pid=1, config={"pegan_crop": 512, "init": None}
    )
    sink.step({"loss": 0.5}, global_step=1)
    sink.step({"loss": 0.4}, global_step=2)  # inside the console cadence: no line
    sink.step({"loss": 0.3}, global_step=100)  # the last step always prints
    sink.val({"psnr": 30.0}, global_step=100, text="test 30.00 dB")
    sink.ckpt(global_step=100, path="out/head.safetensors")
    sink.run_end(status="ok", final_step=100)
    out = capsys.readouterr().out.splitlines()

    assert out[0] == "settings (2):"
    assert out[1] == "  pegan_crop=512"  # None values left out
    assert out[2].startswith("start r [x1]: 100 rollouts")
    steps = [line for line in out if line.startswith("rollout ")]
    assert len(steps) == 2
    assert steps[0].startswith("rollout 1/100 (1.0%)") and "loss 0.5" in steps[0]
    assert "first rollout at" in steps[0]  # no rate before two samples
    assert steps[1].startswith("rollout 100/100") and "eta" in steps[1]
    assert "eval @ rollout 100: test 30.00 dB" in out
    assert "saved out/head.safetensors @ rollout 100" in out
    assert out[-1].startswith("end ok: rollout 100/100")


def test_console_keys_pick_the_shown_metrics(capsys):
    sink = ProgressSink(
        None, run="r", method=None, preset=None, console_keys=("loss/average", "lr/")
    )
    sink.run_start(total_steps=10, total_epochs=1, pid=1)
    sink.step({"loss/average": 0.2, "lr/unet": 1e-4, "grad/norm": 3.0}, global_step=1)
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("step ")][0]
    assert "loss/average 0.2" in line and "lr/unet 0.0001" in line
    assert "grad/norm" not in line
    sink.close()


def test_logs_dict_keys_named_like_event_fields_do_not_clash(tmp_path):
    # dispatch_logs adds global_step / epoch to a wandb logs dict before the sink
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, console=False)
    sink.run_start(total_steps=10, total_epochs=1, pid=1, config={"lr": 1e-4})
    sink.log({"loss": 0.1, "global_step": 99, "epoch": 7}, global_step=5, epoch=1)
    sink.log({"loss/val_cmmd": 0.03, "val_step": 3}, global_step=5, epoch=1, val_step=2)
    sink.run_end(status="ok", final_step=10)
    evs = _read_events(path)
    assert evs[0]["config"] == {"lr": 1e-4}
    assert (evs[1]["global_step"], evs[1]["epoch"]) == (5, 1)
    assert evs[2]["ev"] == "val" and evs[2]["val_step"] == 2 and evs[2]["cmmd"] == 0.03


def test_trailing_rate_leaves_the_compile_out():
    from library.runtime.progress import trailing_rate

    # step 1 lands after 1000 s of load + compile, then 1 step/s
    samples = [(0.0, 0), (1000.0, 1), (1010.0, 11), (1020.0, 21)]
    assert trailing_rate(samples) == pytest.approx(1.0)
    assert trailing_rate(samples[:1]) is None


def test_read_status_keeps_the_eval_history_and_settings(tmp_path):
    events = [_run_start(config={"pegan_crop": 1024})]
    events += [
        {"ev": "val", "ts": 10.0 * i, "global_step": 100 * i, "epoch": 0, "psnr": i}
        for i in (1, 2, 3)
    ]
    st = read_status(_write_stream(tmp_path, events))
    assert [v["psnr"] for v in st["vals"]] == [1, 2, 3]
    assert st["val"]["psnr"] == 3
    assert st["config"] == {"pegan_crop": 1024}
    assert "eval @ 300: psnr=3" in rs.format_status(st)
