"""Phase 1 training daemon: job persistence, liveness, and an
end-to-end serial-queue run over the real HTTP surface with fake training
subprocesses.

The fake "trainer" is a tiny ``python -c`` script that writes a well-formed
Phase-0 ``progress.jsonl`` and exits — exercising the spawn → tail → finalize
path without launching torch/accelerate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil
import pytest

from anima_daemon import config, gpu, gpu_guard, jobs, proc, record, watchdog

# Bound at import time so tests that monkeypatch the client module's attribute
# can still build a real (dead) client without recursing into their own patch.
from anima_daemon.client import DaemonClient as _RealDaemonClient
from anima_daemon.manager import JobManager
from anima_daemon.mcp import MCPServer
from anima_daemon.server import serve

# This suite drives the real daemon job queue over live subprocesses and timers:
# ~55s serial, but ~4x faster under xdist since the waits overlap. `make test-unit`
# runs it in the parallel phase (`-m slow -n auto`). See pyproject `[tool.pytest]`.
pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------


def test_split_daemon_run_args_label_after_script_belongs_to_child():
    # The trap this fixes: bench scripts take --label themselves — a --label
    # after the script path must reach the child, not name the daemon job.
    from scripts.tasks.daemon import split_daemon_run_args

    label, stall, head, tail = split_daemon_run_args(
        ["bench/foo/run_bench.py", "--label", "smoke", "--stall-timeout", "600"]
    )
    assert label is None and stall is None
    assert head == [
        "bench/foo/run_bench.py",
        "--label",
        "smoke",
        "--stall-timeout",
        "600",
    ]
    assert tail == []


def test_split_daemon_run_args_own_flags_before_script():
    from scripts.tasks.daemon import split_daemon_run_args

    label, stall, head, tail = split_daemon_run_args(
        ["--label", "myjob", "--stall-timeout", "600", "run.py", "--limit", "5"]
    )
    assert label == "myjob" and stall == "600"
    assert head == ["run.py", "--limit", "5"]
    assert tail == []


def _seed_job(state: str, age_days: float, job_id: str) -> jobs.Job:
    """A persisted job in ``state``, finished ``age_days`` ago, with a stdout.log."""
    ts = time.time() - age_days * 86400.0
    job = jobs.Job(id=job_id, method="lora", preset="default", state=state)
    job.submitted_at = ts
    job.ended_at = ts if state in record.TERMINAL_STATES else None
    job.persist()
    (job.dir / "stdout.log").write_text("x" * 1024, encoding="utf-8")
    return job


def test_prune_jobs_age_state_and_floor(tmp_path, monkeypatch):
    """Old + terminal prunes; young, active, and floor-protected survive."""
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    _seed_job(record.STATE_DONE, 60, "old-done")
    _seed_job(record.STATE_ERROR, 45, "old-error")
    _seed_job(record.STATE_STOPPED, 40, "old-stopped")
    _seed_job(record.STATE_DONE, 1, "young-done")
    # Never a candidate however old: it holds the queue slot / is mid-flight.
    _seed_job(record.STATE_RUNNING, 90, "old-running")
    _seed_job(record.STATE_QUEUED, 90, "old-queued")

    summary = jobs.prune_jobs(max_age_days=30, keep_recent=0, dry_run=True)
    assert set(summary["pruned"]) == {"old-done", "old-error", "old-stopped"}
    assert summary["freed_bytes"] > 0
    assert set(jobs.load_all()) == {  # dry run touched nothing
        "old-done",
        "old-error",
        "old-stopped",
        "young-done",
        "old-running",
        "old-queued",
    }

    # keep_recent protects the N newest *terminal* jobs regardless of age, so the
    # two newest terminal records (young-done, old-stopped) are off the table.
    summary = jobs.prune_jobs(max_age_days=30, keep_recent=2, dry_run=False)
    assert set(summary["pruned"]) == {"old-done", "old-error"}
    assert summary["kept"] == 2
    assert set(jobs.load_all()) == {
        "old-stopped",
        "young-done",
        "old-running",
        "old-queued",
    }


def test_prune_jobs_disabled_and_unreadable(tmp_path, monkeypatch):
    """0 days disables; an unparseable record is left alone, never guessed at."""
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    _seed_job(record.STATE_DONE, 99, "ancient")
    assert jobs.prune_jobs(max_age_days=0, keep_recent=0)["pruned"] == []
    assert set(jobs.load_all()) == {"ancient"}

    (config.JOBS_DIR / "corrupt").mkdir()
    (config.JOBS_DIR / "corrupt" / "job.json").write_text("{not json", encoding="utf-8")
    summary = jobs.prune_jobs(max_age_days=30, keep_recent=0)
    assert summary["pruned"] == ["ancient"]
    assert (config.JOBS_DIR / "corrupt").is_dir()


# --------------------------------------------------------------------------
# end-to-end over the HTTP surface
# --------------------------------------------------------------------------

_FAKE_TRAINER = r"""
import json, sys, time
path, dur = sys.argv[1], float(sys.argv[2])
with open(path, "w", buffering=1) as f:
    f.write(json.dumps({"ev": "run_start", "ts": 0.0}) + "\n")
    f.write(json.dumps({"ev": "step", "ts": 0.1, "global_step": 1, "loss": 0.5}) + "\n")
    time.sleep(dur)
    f.write(json.dumps({"ev": "ckpt", "ts": dur, "global_step": 1, "path": "/tmp/fake.safetensors"}) + "\n")
    f.write(json.dumps({"ev": "run_end", "ts": dur, "status": "ok", "final_step": 1}) + "\n")
"""


def _fake_build_cmd(self, job):
    dur = float(job.overrides.get("duration", 1.0))
    cmd = [sys.executable, "-c", _FAKE_TRAINER, job.progress_path, str(dur)]
    return cmd, os.environ.copy()


def _wait_until(pred, timeout=20.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """An in-process daemon (manager + HTTP server) with fake training cmds."""
    from anima_daemon import client

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PIDFILE", tmp_path / "daemon.json")
    monkeypatch.setattr(config, "DAEMON_LOG", tmp_path / "daemon.log")
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_build_cmd)
    # Fake trainers don't touch the GPU; stub the guard so the test doesn't
    # block on whatever real workload happens to hold VRAM on the host.
    monkeypatch.setattr(gpu, "gpu_pids", lambda: set())

    mgr = JobManager()
    mgr.start()
    srv = serve(mgr, port=0, fingerprint=config.source_fingerprint())
    t = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    t.start()
    port = srv.server_address[1]
    cl = client.DaemonClient(port)
    assert _wait_until(lambda: cl.health() is not None, timeout=5)
    try:
        yield cl, mgr
    finally:
        srv.request_shutdown(True)
        srv.server_close()


@pytest.mark.core
def test_serial_queue(daemon):
    cl, _ = daemon
    j1 = cl.submit(method="lora", overrides={"duration": 1.0})["job_id"]
    j2 = cl.submit(method="lora", overrides={"duration": 1.0})["job_id"]

    assert _wait_until(lambda: cl.get(j1)["state"] == "done", timeout=15)
    assert _wait_until(lambda: cl.get(j2)["state"] == "done", timeout=15)

    g1, g2 = cl.get(j1), cl.get(j2)
    # serial: the second job can't start before the first ends
    assert g2["started_at"] >= g1["ended_at"] - 0.5
    # ckpt path picked up from the progress stream
    assert g1["ckpt_path"] == "/tmp/fake.safetensors"
    assert g1["latest"]["ev"] == "run_end"


def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PIDFILE", tmp_path / "daemon.json")
    monkeypatch.setattr(config, "DAEMON_LOG", tmp_path / "daemon.log")
    monkeypatch.setattr(gpu, "gpu_pids", lambda: set())


@pytest.mark.core
def test_worker_survives_build_exception(tmp_path, monkeypatch):
    """A job whose _build_cmd raises must fail ERROR without killing the worker;
    the next queued job still runs. Regression for the silent-worker-death hang
    that left every later job stuck `queued` forever with no error (the stall
    watchdog only guards *running* jobs, so a never-launched job spins forever)."""
    _isolate_state(tmp_path, monkeypatch)

    def build_or_boom(self, job):
        if job.method == "boom":
            raise RuntimeError("kaboom while building the command")
        return _fake_build_cmd(self, job)

    monkeypatch.setattr(JobManager, "_build_cmd", build_or_boom)
    mgr = JobManager()
    mgr.start()
    try:
        bad = mgr.submit(
            method="boom", preset="default", methods_subdir=None, start=True
        )
        good = mgr.submit(
            method="lora",
            preset="default",
            methods_subdir=None,
            overrides={"duration": 0.2},
            start=True,
        )
        assert _wait_until(lambda: mgr.get(bad.id).state == "error", timeout=10)
        assert _wait_until(lambda: mgr.get(good.id).state == "done", timeout=15)
        assert "launch failed" in (mgr.get(bad.id).error or "")
        assert mgr.worker_alive() is True
    finally:
        mgr.shutdown(kill_jobs=False)


def test_worker_survives_prelaunch_exception(tmp_path, monkeypatch):
    """An exception *before* launch (here in the GPU guard, outside
    _launch_and_monitor's own try) is caught by the worker-loop crash guard:
    the job fails ERROR and the worker keeps draining the queue."""
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_build_cmd)

    def boom_guard(self, job, **kw):
        if job.method == "boom":
            raise RuntimeError("guard blew up")

    monkeypatch.setattr(JobManager, "_gpu_guard", boom_guard)
    mgr = JobManager()
    mgr.start()
    try:
        bad = mgr.submit(
            method="boom", preset="default", methods_subdir=None, start=True
        )
        good = mgr.submit(
            method="lora",
            preset="default",
            methods_subdir=None,
            overrides={"duration": 0.2},
            start=True,
        )
        assert _wait_until(lambda: mgr.get(bad.id).state == "error", timeout=10)
        assert _wait_until(lambda: mgr.get(good.id).state == "done", timeout=15)
        assert "unexpected error" in (mgr.get(bad.id).error or "")
        assert mgr.worker_alive() is True
    finally:
        mgr.shutdown(kill_jobs=False)


def test_cli_queue_submits_instead_of_launching(daemon, monkeypatch):
    """`train(..., extra=["--queue"])` enqueues on the daemon and returns,
    rather than calling accelerate_launch inline."""
    from scripts.tasks import _common

    cl, _ = daemon
    # Point the CLI's daemon client at the in-process test daemon (train() does
    # a local `from anima_daemon import client` then calls ensure_daemon).
    import anima_daemon.client as daemon_client

    monkeypatch.setattr(daemon_client, "ensure_daemon", lambda **kw: cl)
    launched = []
    monkeypatch.setattr(_common, "accelerate_launch", lambda *a: launched.append(a))

    _common.train("tlora", ["--queue"], methods_subdir="gui-methods")

    assert launched == []  # inline path skipped
    jobs_list = cl.list_jobs()
    assert len(jobs_list) == 1
    job = jobs_list[0]
    assert job["method"] == "tlora"
    assert job["methods_subdir"] == "gui-methods"
    assert "--queue" not in job["extra"]


def test_cli_queue_folds_artist_into_extra(daemon, monkeypatch):
    """ARTIST env is folded into the queued job's extra (the daemon's own
    build_method_args doesn't read env vars)."""
    from scripts.tasks import _common

    cl, _ = daemon
    import anima_daemon.client as daemon_client

    monkeypatch.setattr(daemon_client, "ensure_daemon", lambda **kw: cl)
    monkeypatch.setattr(_common, "accelerate_launch", lambda *a: None)
    monkeypatch.setenv("ARTIST", "alice")

    _common.train("lora", ["--queue"])

    job = cl.list_jobs()[-1]
    assert "--artist_filter" in job["extra"]
    assert "alice" in job["extra"]


@pytest.mark.core
def test_stop_running_job(daemon):
    cl, mgr = daemon
    jid = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=10)
    pid = cl.get(jid)["pid"]
    assert pid and psutil.pid_exists(pid)

    cl.stop(jid)
    assert _wait_until(lambda: cl.get(jid)["state"] == "stopped", timeout=10)
    # tree torn down → the training pid is gone
    assert _wait_until(lambda: not psutil.pid_exists(pid), timeout=5)


def test_stop_queued_job_finalizes_immediately(daemon):
    """Cancelling a job that's still queued behind a running one finalizes it
    *now* (not lazily when the worker eventually dequeues it), so a UI watching
    the job list sees it leave the queue right away."""
    cl, _ = daemon
    # j1 holds the worker for a while; j2 stays queued behind it.
    j1 = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    j2 = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(j1)["state"] == "running", timeout=10)
    assert cl.get(j2)["state"] == "queued"

    cl.stop(j2)
    # Finalized immediately while j1 is still running — no need to wait for the
    # worker to reach j2.
    assert _wait_until(lambda: cl.get(j2)["state"] == "stopped", timeout=3)
    assert cl.get(j1)["state"] == "running"  # the running job is untouched

    # The stale FIFO entry is harmless: when the worker eventually dequeues j2's
    # id it skips it (state != queued), never relaunching it.
    cl.stop(j1)
    assert _wait_until(lambda: cl.get(j1)["state"] == "stopped", timeout=10)
    time.sleep(0.5)
    assert cl.get(j2)["state"] == "stopped"


def test_queue_start_true_flushes_held_backlog(daemon):
    """``start=True`` (the main Train/Run button) resumes a paused queue, so a
    job held earlier via ``start=False`` runs too."""
    cl, _ = daemon
    held = cl.submit(method="lora", overrides={"duration": 1.0}, start=False)["job_id"]
    assert cl.health()["paused"] is True

    run_now = cl.submit(method="lora", overrides={"duration": 1.0}, start=True)[
        "job_id"
    ]
    assert cl.health()["paused"] is False
    # Both drain in FIFO order once the gate opens.
    assert _wait_until(lambda: cl.get(held)["state"] == "done", timeout=15)
    assert _wait_until(lambda: cl.get(run_now)["state"] == "done", timeout=15)
    assert cl.get(run_now)["started_at"] >= cl.get(held)["ended_at"] - 0.5


def test_add_to_queue_while_running_auto_advances(daemon):
    """Regression: ``start=False`` ("add to queue") must NOT pause a queue that's
    already playing. Adding a job while another was running used to clear the
    global run gate, so the new job stalled ``queued`` the moment the running one
    finished — the GUI's "infinite loading" report. It should auto-advance on its
    own (cassette-tape behaviour), with no Start Queue press."""
    cl, _ = daemon
    running = cl.submit(method="lora", overrides={"duration": 3.0}, start=True)[
        "job_id"
    ]
    assert _wait_until(lambda: cl.get(running)["state"] == "running", timeout=10)

    # Add-to-queue while a job is running: the gate must stay open.
    queued = cl.submit(method="lora", overrides={"duration": 1.0}, start=False)[
        "job_id"
    ]
    assert cl.health()["paused"] is False

    # Both drain without anyone pressing Start Queue, in FIFO order.
    assert _wait_until(lambda: cl.get(running)["state"] == "done", timeout=20)
    assert _wait_until(lambda: cl.get(queued)["state"] == "done", timeout=20)
    assert cl.get(queued)["started_at"] >= cl.get(running)["ended_at"] - 0.5


def test_pause_does_not_interrupt_running_job(daemon):
    """Pausing the queue holds the *next* launch but never stops a job already
    running."""
    cl, _ = daemon
    running = cl.submit(method="lora", overrides={"duration": 60.0}, start=True)[
        "job_id"
    ]
    queued = cl.submit(method="lora", overrides={"duration": 1.0})["job_id"]
    assert _wait_until(lambda: cl.get(running)["state"] == "running", timeout=10)

    cl.pause_queue()
    assert cl.health()["paused"] is True
    assert cl.get(running)["state"] == "running"  # untouched

    cl.stop(running)
    assert _wait_until(lambda: cl.get(running)["state"] == "stopped", timeout=10)
    # The queued one stays held while paused — it must not advance.
    time.sleep(0.7)
    assert cl.get(queued)["state"] == "queued"
    cl.start_queue()
    assert _wait_until(lambda: cl.get(queued)["state"] == "done", timeout=15)


# --------------------------------------------------------------------------
# Phase 2a — pause/resume a running job (tree-freeze)
# --------------------------------------------------------------------------


def test_pause_resume_running_job(daemon):
    """`pause` SIGSTOPs the job's tree (state → paused, OS process stopped, the
    slot still owned); `resume` SIGCONTs it and it runs to completion."""
    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"duration": 5.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=15)
    pid = cl.get(jid)["pid"]

    res = cl.pause_job(jid)
    assert res["state"] == "paused"
    assert not res.get("error")
    # The process tree is genuinely frozen (POSIX exposes STATUS_STOPPED).
    if not sys.platform.startswith("win"):
        assert _wait_until(
            lambda: psutil.Process(pid).status() == psutil.STATUS_STOPPED, timeout=5
        )
    assert cl.get(jid)["state"] == "paused"
    # A paused job still owns the slot → it's the active job, and its staleness
    # clock is frozen (None), not a wedged running-clock the watchdog would flag.
    assert cl.health()["active_job"] == jid
    assert cl.get(jid)["stale_for"] is None

    res = cl.resume_job(jid)
    assert res["state"] == "running"
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=25)


def test_stop_while_paused_thaws_and_kills(daemon):
    """Stopping a frozen job must thaw the tree first so the SIGTERM lands — it
    dies promptly instead of waiting out the kill grace on a stopped process."""
    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=15)
    pid = cl.get(jid)["pid"]

    cl.pause_job(jid)
    assert _wait_until(lambda: cl.get(jid)["state"] == "paused", timeout=5)

    cl.stop(jid)
    assert _wait_until(lambda: cl.get(jid)["state"] == "stopped", timeout=10)
    assert _wait_until(lambda: not psutil.pid_exists(pid), timeout=8)


@pytest.mark.core
def test_reconcile_orphan_requeue_adopt(tmp_path, monkeypatch):
    """Boot sweep: dead `running` → orphaned error; `queued` → re-enqueued;
    live `running` → adopted for monitoring."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")

    # a `running` job whose process died while the daemon was down
    dead = jobs.Job(
        id="dead",
        method="lora",
        preset="default",
        state=record.STATE_RUNNING,
        pid=2_147_483_000,
        create_time=1.0,
    )
    dead.progress_path = str(dead.dir / "progress.jsonl")
    dead.persist()

    # a `queued` job that never started
    pend = jobs.Job(id="pend", method="lora", preset="default", state=record.STATE_QUEUED)
    pend.persist()

    # a `running` job that's actually alive (use this test process as the pid)
    me = os.getpid()
    live = jobs.Job(
        id="live",
        method="lora",
        preset="default",
        state=record.STATE_RUNNING,
        pid=me,
        create_time=proc.create_time(me),
    )
    live.persist()

    mgr = JobManager()
    mgr._reconcile()  # sweep without starting the worker

    assert mgr.get("dead").state == record.STATE_ERROR
    assert mgr.get("dead").status_detail == "orphaned"
    assert mgr._queue.get_nowait()[-1] == "pend"  # re-enqueued (prio, seq, id)
    assert "live" in mgr._adopt  # re-attached for monitoring


def test_command_job_loads_with_train_default():
    """A legacy job.json (written before `kind` existed) loads as a train job."""
    job = jobs.Job.from_dict({"id": "old", "method": "lora", "preset": "default"})
    assert job.kind == "train"
    assert job.argv == [] and job.extra_env == {}
    # Phase 0 fields default cleanly on a legacy record.
    assert job.captured_env == {} and job.returncode is None


# --------------------------------------------------------------------------
# Phase 0a: source fingerprint + stale-daemon detection
# --------------------------------------------------------------------------


def test_source_fingerprint_stable_and_content_sensitive(tmp_path, monkeypatch):
    """The fingerprint is stable across calls and changes when any *.py byte
    changes; an unreadable file is skipped, not fatal."""
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    (src / "b.py").write_text("y = 2\n")
    monkeypatch.setattr(config, "_SRC_DIR", src)

    fp1 = config.source_fingerprint()
    assert fp1 == config.source_fingerprint()  # stable
    (src / "b.py").write_text("y = 3\n")  # one byte changes
    assert config.source_fingerprint() != fp1
    # a non-.py sibling doesn't participate
    (src / "notes.txt").write_text("ignored")
    fp3 = config.source_fingerprint()
    (src / "notes.txt").write_text("still ignored, differently")
    assert config.source_fingerprint() == fp3


@pytest.mark.core
def test_daemon_is_stale_matrix(monkeypatch):
    from anima_daemon import client as daemon_client

    monkeypatch.setattr(config, "source_fingerprint", lambda: "CURRENT")
    assert daemon_client.daemon_is_stale(None) is False  # nothing running
    assert daemon_client.daemon_is_stale({"fingerprint": "CURRENT"}) is False
    assert daemon_client.daemon_is_stale({"fingerprint": "OLD"}) is True
    # a daemon predating the field is stale (gets replaced by a current one)
    assert daemon_client.daemon_is_stale({"ok": True}) is True


def test_health_reports_boot_fingerprint(daemon):
    """/health echoes the fingerprint the daemon booted with; on an unchanged
    tree that equals the current on-disk hash → not stale."""
    from anima_daemon import client as daemon_client

    cl, _ = daemon
    h = cl.health()
    assert h["fingerprint"] == config.source_fingerprint()
    assert daemon_client.daemon_is_stale(h) is False


# --------------------------------------------------------------------------
# Phase 0b: submit-time env capture
# --------------------------------------------------------------------------


def test_capture_env_whitelist():
    src = {
        "ANIMA_DIT": "/models/dit.safetensors",
        "CUDA_VISIBLE_DEVICES": "1",
        "HF_TOKEN": "hf_x",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_LOGS": "recompiles",
        "NCCL_DEBUG": "INFO",
        "ANIMA_DAEMON_PORT": "8765",  # daemon-config → excluded
        "PATH": "/usr/bin",  # never captured
        "VIRTUAL_ENV": "/x/.venv",  # never captured
        "HOME": "/home/x",  # not whitelisted
    }
    got = config.capture_env(src)
    assert got == {
        "ANIMA_DIT": "/models/dit.safetensors",
        "CUDA_VISIBLE_DEVICES": "1",
        "HF_TOKEN": "hf_x",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_LOGS": "recompiles",
        "NCCL_DEBUG": "INFO",
    }


def test_build_cmd_layers_captured_env_under_extra_env():
    """daemon-env ← captured_env ← extra_env: the submitter's capture overrides
    the daemon's inherited value, and a command's extra_env overrides both."""
    job = jobs.Job(
        id="c1",
        method="preprocess",
        preset="",
        kind="command",
        argv=["tasks.py", "preprocess"],
        captured_env={"CUDA_VISIBLE_DEVICES": "2", "HF_TOKEN": "from_shell"},
        extra_env={"HF_TOKEN": "from_extra"},
    )
    mgr = JobManager.__new__(JobManager)
    _, env = mgr._build_cmd(job)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"  # captured beats daemon boot env
    assert env["HF_TOKEN"] == "from_extra"  # extra_env wins over captured


def test_submit_stores_captured_env(daemon):
    cl, _ = daemon
    jid = cl.submit(
        method="lora",
        overrides={"duration": 0.2},
        captured_env={"ANIMA_DIT": "/x.safetensors"},
    )["job_id"]
    assert cl.get(jid)["captured_env"] == {"ANIMA_DIT": "/x.safetensors"}


# --------------------------------------------------------------------------
# Phase 0c: returncode mirroring + attach exit code
# --------------------------------------------------------------------------


@pytest.mark.core
def test_returncode_mirrored_on_nonzero_exit(real_cmd_daemon):
    """A command job's OS exit code lands in job.returncode, and _exit_code_for
    reads it back (what run_gpu exits with)."""
    from scripts.tasks import _common

    cl, _ = real_cmd_daemon
    jid = cl.submit_command(label="boom", argv=["-c", "import sys; sys.exit(3)"])[
        "job_id"
    ]
    assert _wait_until(lambda: cl.get(jid)["state"] == "error", timeout=15)
    assert cl.get(jid)["returncode"] == 3
    assert _common._exit_code_for(cl, jid) == 3


@pytest.mark.core
def test_attach_streams_and_returns_exit_code(real_cmd_daemon, capsys):
    """_attach_and_wait streams the job's stdout to the terminal and returns its
    exit code — the attach-by-default core (what run_gpu exits with)."""
    from scripts.tasks import _common

    cl, _ = real_cmd_daemon
    jid = cl.submit_command(
        label="chatty",
        argv=[
            "-c",
            "print('line-one'); print('line-two'); import sys; sys.exit(2)",
        ],
    )["job_id"]
    rc = _common._attach_and_wait(cl, jid)
    assert rc == 2
    out = capsys.readouterr().out
    assert "line-one" in out and "line-two" in out


@pytest.fixture
def real_cmd_daemon(tmp_path, monkeypatch):
    """Daemon with the *real* `_build_cmd` (no fake-trainer patch) so command
    jobs actually exec their argv. GPU guard stubbed so the queue never blocks
    on the host's VRAM."""
    from anima_daemon import client

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PIDFILE", tmp_path / "daemon.json")
    monkeypatch.setattr(config, "DAEMON_LOG", tmp_path / "daemon.log")
    monkeypatch.setattr(gpu, "gpu_pids", lambda: set())

    mgr = JobManager()
    mgr.start()
    srv = serve(mgr, port=0, fingerprint=config.source_fingerprint())
    t = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    t.start()
    cl = client.DaemonClient(srv.server_address[1])
    assert _wait_until(lambda: cl.health() is not None, timeout=5)
    try:
        yield cl, mgr
    finally:
        srv.request_shutdown(True)
        srv.server_close()


@pytest.mark.core
def test_command_job_end_to_end(real_cmd_daemon):
    """submit_command → detached exec → exit-code finalize (no progress.jsonl),
    with extra_env applied and stdout captured."""
    cl, _ = real_cmd_daemon
    resp = cl.submit_command(
        label="preprocess",
        argv=[
            "-c",
            "import os;print('shuf=' + os.environ['CAPTION_SHUFFLE_VARIANTS'])",
        ],
        extra_env={"CAPTION_SHUFFLE_VARIANTS": "7"},
    )
    jid = resp["job_id"]
    assert resp["state"] == "queued"
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)
    job = cl.get(jid)
    assert job["kind"] == "command"
    assert job["argv"][0] == "-c"
    log = (config.job_dir(jid) / "stdout.log").read_text()
    assert "shuf=7" in log


@pytest.mark.core
def test_command_job_nonzero_exit_outranks_its_run_end_ok(real_cmd_daemon):
    """A command job that streams a training loop (moss train) then fails later
    in the same process tree (a train → eval chain) ends in error, not done."""
    cl, _ = real_cmd_daemon
    script = (
        "import json, os, sys\n"
        "p = os.path.join(os.environ['ANIMA_DAEMON_JOB_DIR'], 'progress.jsonl')\n"
        "with open(p, 'w') as f:\n"
        "    f.write(json.dumps({'ev': 'run_end', 'ts': 1.0, 'status': 'ok',"
        " 'final_step': 4}) + '\\n')\n"
        "sys.exit(3)\n"
    )
    jid = cl.submit_command(label="chain", argv=["-c", script])["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "error", timeout=15)
    job = cl.get(jid)
    assert job["returncode"] == 3
    assert job["latest"]["ev"] == "run_end"  # the stream is still read


def test_command_job_result_lift_end_to_end(real_cmd_daemon):
    """End-to-end Phase 1a: a command job reads the daemon-exported
    ANIMA_DAEMON_JOB_DIR, writes an envelope + `result_path.json` pointer (as
    bench/_common + write_gen_manifest do), and the monitor lifts `result_path`
    + `result_summary` onto the record on the terminal transition."""
    cl, _ = real_cmd_daemon
    script = (
        "import os, json;"
        "d = os.environ['ANIMA_DAEMON_JOB_DIR'];"
        "env = os.path.join(d, 'gen_manifest.json');"
        "open(env, 'w').write(json.dumps("
        "{'label': 'foo-lora', 'metrics': {'n_images': 3}}));"
        "open(os.path.join(d, 'result_path.json'), 'w').write("
        "json.dumps({'path': env}))"
    )
    jid = cl.submit_command(label="gen", argv=["-c", script])["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)
    job = cl.get(jid)
    assert job["result_path"] == str(config.job_dir(jid) / "gen_manifest.json")
    assert job["result_summary"] == {"label": "foo-lora", "metrics": {"n_images": 3}}


@pytest.mark.core
def test_log_stream_closes_on_terminal_job(real_cmd_daemon):
    """The SSE log stream must END, not park the socket.

    An SSE body has no Content-Length and no chunked framing, so the client's
    only EOF signal is the connection closing — under HTTP/1.1 keep-alive the
    handler returned but ThreadingHTTPServer held the socket, so `curl -N`,
    DaemonClient.stream() and `make daemon-attach` all blocked forever on an
    already-finished job. Regression guard: consuming the whole stream of a
    terminal job must terminate on its own (the timeout below is the failure
    mode, not the assertion).
    """
    cl, _ = real_cmd_daemon
    jid = cl.submit_command(label="quick", argv=["-c", "print('hi')"])["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)

    payloads: list[str] = []
    done = threading.Event()

    def drain():
        try:
            payloads.extend(cl.stream_logs(jid))
        finally:
            done.set()

    threading.Thread(target=drain, daemon=True).start()
    assert done.wait(timeout=20), "log stream never closed on a finished job"
    assert any("hi" in p for p in payloads)
    assert any('"eof"' in p for p in payloads)


def test_log_stream_outlives_a_pause(daemon):
    """A frozen job isn't over: the stream stays open across the freeze and only
    ends with the job, so an attached `make daemon-run` doesn't exit 1 mid-run."""
    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"duration": 3.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=15)

    payloads: list[str] = []
    done = threading.Event()

    def drain():
        try:
            payloads.extend(cl.stream_logs(jid))
        finally:
            done.set()

    threading.Thread(target=drain, daemon=True).start()
    assert cl.pause_job(jid)["state"] == "paused"
    assert not done.wait(timeout=1.5), "log stream closed on a frozen job"

    cl.resume_job(jid)
    assert done.wait(timeout=20), "log stream never closed after the job ended"
    assert json.loads(payloads[-1]) == {"ev": "eof", "state": "done"}


def _run_past_cancelled_job(mgr):
    """Submit a job the test cancels mid-launch, then a sentinel behind it. The
    queue is serial, so the sentinel finishing proves the worker moved on."""
    cancelled = mgr.submit(
        method="cancel",
        preset="default",
        methods_subdir=None,
        overrides={"duration": 30.0},
        start=True,
    )
    sentinel = mgr.submit(
        method="lora",
        preset="default",
        methods_subdir=None,
        overrides={"duration": 0.2},
        start=True,
    )
    assert _wait_until(lambda: mgr.get(sentinel.id).state == "done", timeout=15)
    return mgr.get(cancelled.id)


def test_stop_during_gpu_guard_never_launches(tmp_path, monkeypatch):
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_build_cmd)

    def stop_in_guard(self, job):
        if job.method == "cancel":
            self.stop(job.id)

    monkeypatch.setattr(JobManager, "_gpu_guard", stop_in_guard)
    mgr = JobManager()
    mgr.start()
    try:
        job = _run_past_cancelled_job(mgr)
        assert job.state == "stopped"
        assert job.pid is None and job.started_at is None
    finally:
        mgr.shutdown(kill_jobs=False)


def test_stop_during_spawn_kills_the_child(tmp_path, monkeypatch):
    _isolate_state(tmp_path, monkeypatch)

    def stop_while_building(self, job):
        if job.method == "cancel":
            self.stop(job.id)
        return _fake_build_cmd(self, job)

    spawned = []
    real_spawn = proc.spawn_detached

    def spy_spawn(*args, **kwargs):
        popen = real_spawn(*args, **kwargs)
        spawned.append(popen)
        return popen

    monkeypatch.setattr(JobManager, "_build_cmd", stop_while_building)
    monkeypatch.setattr(proc, "spawn_detached", spy_spawn)
    mgr = JobManager()
    mgr.start()
    try:
        job = _run_past_cancelled_job(mgr)
        assert job.state == "stopped"
        assert job.pid is None and job.started_at is None
        assert spawned[0].poll() is not None
    finally:
        mgr.shutdown(kill_jobs=False)


def test_tools_manifest_operations_are_routed(daemon):
    """Every manifest entry reaches a handler. A bogus job id answers `no such
    job` and an empty submit a 400; only an unrouted path says `not found`."""
    import urllib.error

    from anima_daemon.tools import TOOLS

    cl, _ = daemon
    for tool in TOOLS:
        if tool["name"] == "shutdown":
            continue
        path = tool["path"].replace("{id}", "no-such-job")
        body = {} if tool["method"] == "POST" else None
        try:
            cl._request(tool["method"], path, body)
        except urllib.error.HTTPError as e:
            assert json.loads(e.read()).get("error") != "not found", tool["name"]


def test_mcp_bridge_does_not_load_the_job_manager():
    code = (
        "import sys, anima_daemon.mcp; "
        "sys.exit('anima_daemon.manager' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_record_readers_skip_a_non_dict_record(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    _seed_job(record.STATE_DONE, 99, "ancient")
    bad = config.JOBS_DIR / "list"
    bad.mkdir()
    (bad / record.RECORD_NAME).write_text("[1]", encoding="utf-8")

    assert record.read_record(bad) is None
    assert [r["id"] for r in record.iter_records()] == ["ancient"]
    assert set(jobs.load_all()) == {"ancient"}
    assert jobs.prune_jobs(max_age_days=30, keep_recent=0)["pruned"] == ["ancient"]
    assert bad.is_dir()


def test_client_wait_timeout_is_not_an_outcome(real_cmd_daemon):
    """A wait that times out raises rather than returning a running record —
    "still running" must never read as a verdict."""
    cl, _ = real_cmd_daemon
    jid = cl.submit_command(label="slow", argv=["-c", "import time;time.sleep(30)"])[
        "job_id"
    ]
    try:
        with pytest.raises(TimeoutError):
            cl.wait(jid, timeout=1.0)
    finally:
        cl.stop(jid)


def test_wait_falls_back_to_disk_when_daemon_unreachable(real_cmd_daemon):
    """job_record (and thus wait) reads the persisted job.json when HTTP fails,
    so a wait survives the eager stale-code daemon restart mid-poll."""
    cl, _ = real_cmd_daemon
    jid = cl.submit_command(label="quick", argv=["-c", "print('x')"])["job_id"]
    cl.wait(jid, timeout=30)

    dead = _RealDaemonClient(port=1)  # nothing listens there
    rec = dead.job_record(jid)
    assert rec is not None and rec["id"] == jid and rec["state"] == "done"
    assert dead.wait(jid, timeout=5)["state"] == "done"


def test_submit_command_stall_timeout_is_recorded_and_honored(real_cmd_daemon):
    """A per-job stall budget rides the submission and wins over the per-kind
    default — the supported way to run a legitimately quiet loop (§4)."""
    cl, mgr = real_cmd_daemon
    jid = cl.submit_command(label="quiet", argv=["-c", "print('x')"], stall_timeout=0)[
        "job_id"
    ]
    assert cl.get(jid)["stall_timeout"] == 0
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)

    # Watchdog: budget 0 → unwatched even with output frozen far past the default.
    job = mgr.get(jid)
    job.started_at = time.time() - 10_000
    os.utime(job.stdout_path, (time.time() - 10_000,) * 2)
    assert watchdog.StallWatchdog().check(job) is None
    job.stall_timeout = 5
    assert watchdog.StallWatchdog().check(job) is not None  # explicit budget → watched

    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as ei:
        cl._request(
            "POST",
            "/jobs",
            {
                "kind": "command",
                "label": "x",
                "argv": ["-c", "0"],
                "stall_timeout": "?",
            },
        )
    assert ei.value.code == 400


@pytest.mark.core
def test_stall_watchdog_spares_a_quiet_but_computing_tree(tmp_path, monkeypatch):
    """Frozen output alone isn't a wedge: an embed/eval loop between prints is
    quiet *and* burning CPU. The watchdog only fires once the tree goes idle too
    — with a hard ceiling so a busy-spinning deadlock still can't park the queue."""
    monkeypatch.setattr(config, "CMD_STALL_TIMEOUT", 120, raising=False)
    stdout = tmp_path / "stdout.log"
    stdout.write_text("caching...\n")
    old = time.time() - 200  # 200s of silence, past the 120s budget
    os.utime(stdout, (old, old))
    job = jobs.Job(
        id="cpu-probe",
        method="quiet",
        preset="",
        kind="command",
        started_at=old,
        stdout_path=str(stdout),
        pid=4242,
    )
    # Drive both clocks by hand: check() reads time.time() more than once
    # per call, so the fixture has to be a value, not a sequence.
    clock = [old + 200]
    cpu = [100.0]
    monkeypatch.setattr(watchdog.time, "time", lambda: clock[0])
    monkeypatch.setattr(watchdog.proc, "tree_cpu_seconds", lambda pid: cpu[0])

    # 5 CPU-seconds burned over a 10s gap → computing → spared.
    dog = watchdog.StallWatchdog()
    assert dog.check(job) is None  # first sample → optimistic
    clock[0] += 10
    cpu[0] += 5.0
    assert dog.check(job) is None  # measured busy → spared

    # Same silence, but the tree burned nothing → the watchdog fires as before.
    dog = watchdog.StallWatchdog()
    clock[0] = old + 200
    assert dog.check(job) is None  # first look, no rate yet
    clock[0] += 10  # no CPU advance
    reason = dog.check(job)
    assert reason is not None and "no output for" in reason

    # Busy but silent past the hard ceiling (8× budget) → killed regardless, and
    # the error says why so the fix (raise/disable stall_timeout) is obvious.
    dog = watchdog.StallWatchdog()
    clock[0] = old + 120 * 9
    reason = dog.check(job)
    assert reason is not None and "still burning CPU" in reason


def test_serve_defers_to_a_live_sibling_daemon(daemon):
    """If an anima daemon already answers on the port, ``serve_with_fallback``
    re-raises so the second process stands down (no duplicate daemon)."""
    from anima_daemon.server import serve_with_fallback

    cl, mgr = daemon  # a real in-process daemon is already serving here
    port = cl.port
    with pytest.raises(OSError):
        serve_with_fallback(JobManager.__new__(JobManager), port=port)


# --------------------------------------------------------------------------
# MCP stdio bridge (anima_daemon/mcp.py)
# --------------------------------------------------------------------------


def _mcp_for(cl):
    """A bridge wired to an in-process daemon client (no pidfile discovery)."""
    return MCPServer(client_factory=lambda: cl, ensure=lambda: cl)


def _call_tool(srv, name, arguments=None, msg_id=1):
    resp = srv.handle(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    result = resp["result"]
    payload = json.loads(result["content"][0]["text"])
    return result, payload


def _dead_client():
    """A client pointed at a port nothing listens on (health → None fast)."""
    return _RealDaemonClient(port=1)


def test_mcp_initialize_and_tools_list():
    srv = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    resp = srv.handle(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        }
    )
    res = resp["result"]
    assert res["protocolVersion"] == "2025-06-18"
    assert "tools" in res["capabilities"]
    # notifications get no response
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    tools = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in tools["result"]["tools"]}
    assert {
        "submit_training",
        "submit_command",
        "list_jobs",
        "get_job",
        "stop_job",
        "tail_log",
        "pause_queue",
        "start_queue",
        "health",
        "shutdown",
    } <= names
    assert "tail_logs" not in names  # SSE endpoint replaced, not registered
    for t in tools["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"


def test_mcp_submit_command_and_tail_log(real_cmd_daemon):
    cl, _ = real_cmd_daemon
    srv = _mcp_for(cl)

    # the bridge injects kind="command" so the daemon doesn't treat it as train
    result, payload = _call_tool(
        srv,
        "submit_command",
        {"label": "echo", "argv": ["-c", "print('hello-mcp')"]},
    )
    assert result["isError"] is False
    jid = payload["job_id"]

    def done():
        _, job = _call_tool(srv, "get_job", {"id": jid})
        return job["state"] == "done"

    assert _wait_until(done, timeout=15)

    result, payload = _call_tool(srv, "tail_log", {"id": jid, "lines": 5})
    assert result["isError"] is False
    assert payload["state"] == "done"
    assert any("hello-mcp" in line for line in payload["lines"])

    # tail_log survives the daemon going away (reads job.json + stdout.log)
    down = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    result, payload = _call_tool(down, "tail_log", {"id": jid})
    assert result["isError"] is False
    assert payload["state"] == "done"
    assert any("hello-mcp" in line for line in payload["lines"])


# --------------------------------------------------------------------------
# structured progress queries (get_progress)
# --------------------------------------------------------------------------


def test_read_events_filters(tmp_path):
    from anima_daemon import tail

    p = tmp_path / "progress.jsonl"
    stream = [{"ev": "run_start", "ts": 0.0}]
    for i in range(1, 11):
        stream.append({"ev": "step", "ts": float(i), "global_step": i, "loss": 1.0 / i})
    stream += [
        {"ev": "log", "ts": 10.5, "level": "WARNING", "logger": "x", "msg": "boom"},
        {"ev": "ckpt", "ts": 11.0, "global_step": 10, "path": "/tmp/x.safetensors"},
        {"ev": "run_end", "ts": 12.0, "status": "ok", "final_step": 10},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for ev in stream:
            f.write(json.dumps(ev) + "\n")

    assert len(tail.read_events(str(p))) == len(stream)

    # ev-kind filter
    steps = tail.read_events(str(p), events=["step"])
    assert [e["global_step"] for e in steps] == list(range(1, 11))

    # since_step — step-less events inherit the preceding step
    late = tail.read_events(str(p), since_step=8)
    assert [e["ev"] for e in late] == ["step", "step", "step", "log", "ckpt", "run_end"]

    # every_nth thins step events but always keeps the latest one
    thinned = tail.read_events(str(p), events=["step"], every_nth=4)
    assert [e["global_step"] for e in thinned] == [1, 5, 9, 10]

    # last_n trailing cap
    assert [e["ev"] for e in tail.read_events(str(p), last_n=2)] == ["ckpt", "run_end"]

    # a half-written tail line is skipped, not fatal
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"ev": "step", "global_st')
    assert len(tail.read_events(str(p))) == len(stream)

    # missing / unset path → empty
    assert tail.read_events(None) == []
    assert tail.read_events(str(tmp_path / "nope.jsonl")) == []


def test_mcp_get_progress(daemon):
    cl, _ = daemon
    srv = _mcp_for(cl)
    _, payload = _call_tool(
        srv, "submit_training", {"method": "lora", "overrides": {"duration": 0.2}}
    )
    jid = payload["job_id"]

    def done():
        _, job = _call_tool(srv, "get_job", {"id": jid})
        return job["state"] == "done"

    assert _wait_until(done, timeout=15)

    # registered in the catalog (rides in from tools.TOOLS)
    tools = srv.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
    assert "get_progress" in {t["name"] for t in tools["result"]["tools"]}

    result, payload = _call_tool(srv, "get_progress", {"id": jid})
    assert result["isError"] is False
    assert [e["ev"] for e in payload["events"]] == [
        "run_start",
        "step",
        "ckpt",
        "run_end",
    ]

    # filters ride through (comma-string form, as in the manifest schema)
    _, payload = _call_tool(srv, "get_progress", {"id": jid, "events": "step"})
    assert [e["ev"] for e in payload["events"]] == ["step"]

    # …and it survives the daemon going away (reads progress.jsonl from disk)
    down = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    result, payload = _call_tool(down, "get_progress", {"id": jid})
    assert result["isError"] is False
    assert payload["events"][-1]["ev"] == "run_end"

    result, payload = _call_tool(down, "get_progress", {"id": "nope"})
    assert payload.get("error") == "no such job"


# --- relocation invariants: stdlib-only boundary + compat shim ---------------


@pytest.mark.core
def test_anima_daemon_is_stdlib_only():
    """The daemon package must not import ``library`` / ``networks`` / ``torch``.

    This is the load-bearing invariant behind the disposable-daemon design: a
    package that stays import-light boots (and eagerly restarts) in ~1s. The
    relocation out of ``scripts/`` turns the docstring promise into a testable
    boundary — AST-scan every module for a forbidden top-level import.
    """
    import ast
    from pathlib import Path

    pkg = Path(config.__file__).resolve().parent
    forbidden = {"library", "networks", "torch"}
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level>0 is a relative import (never a top-level lib); skip.
                names = [node.module] if (node.module and node.level == 0) else []
            else:
                continue
            for name in names:
                if name and name.split(".")[0] in forbidden:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, f"daemon must stay stdlib-only, found: {offenders}"


# --------------------------------------------------------------------------
# Phase 1a: result-envelope lift (bench/test-* jobs become citizens)
# --------------------------------------------------------------------------


def test_build_cmd_exports_job_identity_env(tmp_path, monkeypatch):
    """Every spawned job learns its id + dir via env, so a bench script inside
    it can drop a result_path.json pointer (train and command kinds alike)."""
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    mgr = JobManager.__new__(JobManager)
    for kind, argv in (("command", ["tasks.py", "x"]), ("train", [])):
        job = jobs.Job(
            id=f"j-{kind}", method="lora", preset="default", kind=kind, argv=argv
        )
        _, env = mgr._build_cmd(job)
        assert env["ANIMA_DAEMON_JOB_ID"] == f"j-{kind}"
        assert env["ANIMA_DAEMON_JOB_DIR"] == str(config.job_dir(f"j-{kind}"))


def test_write_result_drops_daemon_pointer(tmp_path, monkeypatch):
    """Under a daemon spawn (ANIMA_DAEMON_JOB_DIR set), write_result drops a
    pointer to its envelope; inline (unset) it's a no-op."""
    from bench import _common

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = argparse.Namespace(x=1)

    # Inline: no daemon env → no pointer.
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)
    _common.write_result(run_dir, script=__file__, args=args, metrics={"a": 1})
    assert not (job_dir / "result_path.json").exists()

    # Daemon spawn: pointer written to the envelope we just wrote.
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(job_dir))
    out = _common.write_result(run_dir, script=__file__, args=args, metrics={"a": 1})
    pointer = json.loads((job_dir / "result_path.json").read_text())
    assert pointer["path"] == str(out.resolve())


@pytest.mark.core
def test_gpu_guard_skips_recycled_pid(tmp_path, monkeypatch):
    """The pre-launch reaper acts on (pid, create_time), never a bare pid.

    Issue #83: a 3-day-old job record's pid had been recycled onto `dwm.exe`,
    which appeared in the (WDDM-polluted) holder list; matching on the number
    alone made the guard try to kill the desktop compositor. A stale record
    whose create_time no longer matches must be left strictly alone.
    """
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")

    me = os.getpid()  # a live pid that really is holding the "GPU"
    monkeypatch.setattr(gpu, "gpu_pids", lambda: {me})
    monkeypatch.setattr(gpu, "gpu_mem", lambda: (0, 10_000))  # free → one pass

    killed: list[int] = []
    monkeypatch.setattr(proc, "kill_tree", lambda pid, **kw: killed.append(pid))

    monkeypatch.setattr(gpu_guard, "evict_resident_inference", lambda: None)
    launching = jobs.Job(id="new", method="lora", preset="default")

    def guard(known):
        return gpu_guard.wait_for_free_gpu(
            launching, jobs=lambda: known, on_wait=print, retries=1, delay=0
        )

    # Stale record: right pid, wrong create_time → the pid was recycled.
    stale = jobs.Job(
        id="stale",
        method="lora",
        preset="default",
        state=record.STATE_DONE,
        pid=me,
        create_time=1.0,
    )
    guard([stale])
    assert killed == []  # dwm.exe lives

    # Genuinely ours: pid *and* create_time match → reap it.
    leaked = jobs.Job(
        id="leaked",
        method="lora",
        preset="default",
        state=record.STATE_DONE,
        pid=me,
        create_time=proc.create_time(me),
    )
    guard([leaked])
    assert killed == [me]


def test_kill_tree_survives_access_denied(monkeypatch):
    """`kill_tree` must not raise when a family member can't be waited on.

    psutil.wait_procs lets AccessDenied escape from its inner Process.wait(),
    which crashed the daemon worker thread in issue #83 (`worker crashed
    handling job ...`) instead of merely failing to kill an unkillable target.
    One unwaitable member must also not abort the reap for the rest.
    """
    me = os.getpid()  # the denied "parent" (stands in for dwm.exe)
    reaped: list[int] = []

    # A real second process to play the ordinary family member, so the patched
    # constructor stays a genuine psutil.Process subclass (psutil's own __eq__
    # does isinstance() against it) and every pid here really exists.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    class Fake(psutil.Process):
        """Real psutil.Process, with terminate/wait/kill scripted per pid."""

        def children(self, recursive=False):
            return [Fake(child.pid)]

        def terminate(self):
            if self.pid == me:
                raise psutil.AccessDenied(pid=self.pid)

        def wait(self, timeout=None):
            if self.pid == me:  # unwaitable — WinError 5 in the field
                raise psutil.AccessDenied(pid=self.pid)
            raise psutil.TimeoutExpired(timeout or 0, pid=self.pid)

        def kill(self):
            reaped.append(self.pid)

    try:
        monkeypatch.setattr(proc.psutil, "Process", Fake)
        proc.kill_tree(me, grace_seconds=0.01)  # must not raise
    finally:
        monkeypatch.undo()
        child.kill()
        child.wait()
    # The denied member is skipped, but the reachable one still gets escalated.
    assert reaped == [child.pid]


# --- issues.md fixes: bounded wait -------------------------------------------


def test_wait_timeout_reports_where_the_job_stands(tmp_path, capsys):
    """`--timeout` must hand back the in-flight status, not an empty buffer (§5)."""
    from anima_daemon import cli as daemon_cli

    stream = tmp_path / "progress.jsonl"
    stream.write_text(
        '{"ev":"run_start","run":"r","total_steps":100}\n'
        '{"ev":"step","global_step":7,"loss":0.5}\n',
        encoding="utf-8",
    )

    class _Cl:
        def wait(self, job_id, timeout=None):
            raise TimeoutError(f"timed out after {timeout}s")

        def job_record(self, job_id):
            return {
                "id": job_id,
                "state": "running",
                "progress_path": str(stream),
                "stdout_path": str(tmp_path / "stdout.log"),
            }

    rc = daemon_cli._wait_and_report(_Cl(), "job-1", timeout=1.0)
    captured = capsys.readouterr()
    assert rc == 124  # timeout(1)'s convention, unchanged
    assert "timed out" in captured.err
    snap = json.loads(captured.out)
    assert snap["timed_out"] is True and snap["state"] == "running"
    # the last progress event is the point: healthy-but-slow vs wedged
    assert snap["latest"]["global_step"] == 7


# --------------------------------------------------------------------------
# Release-pause: pause(release_model=True) → trainer saves + exits, GPU free,
# queue advances; resume relaunches with --resume.
# --------------------------------------------------------------------------

# A fake trainer that honors the cooperative protocol: polls the job dir for
# `pause.request`, writes a state dir, acks, and emits `run_end paused`. On a
# relaunch with --resume it finishes immediately with `run_end ok`.
_FAKE_RELEASABLE_TRAINER = r"""
import json, os, sys, time
path, dur, job_dir, resumed = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
with open(path, "w", buffering=1) as f:
    f.write(json.dumps({"ev": "run_start", "ts": 0.0}) + "\n")
    if resumed != "-":
        f.write(json.dumps({"ev": "step", "ts": 0.1, "global_step": 8, "loss": 0.4, "resumed_from": resumed}) + "\n")
        f.write(json.dumps({"ev": "run_end", "ts": 0.2, "status": "ok", "final_step": 8}) + "\n")
        sys.exit(0)
    t0 = time.time()
    step = 0
    while time.time() - t0 < dur:
        step += 1
        f.write(json.dumps({"ev": "step", "ts": time.time() - t0, "global_step": step, "loss": 0.5}) + "\n")
        req = os.path.join(job_dir, "pause.request")
        if os.path.exists(req):
            state_dir = os.path.join(job_dir, "fake-checkpoint-state")
            os.makedirs(state_dir, exist_ok=True)
            os.remove(req)
            with open(os.path.join(job_dir, "pause.ack.json"), "w") as a:
                json.dump({"state_dir": state_dir, "global_step": step}, a)
            f.write(json.dumps({"ev": "run_end", "ts": time.time() - t0, "status": "paused", "final_step": step, "state_dir": state_dir}) + "\n")
            sys.exit(0)
        time.sleep(0.1)
    f.write(json.dumps({"ev": "run_end", "ts": dur, "status": "ok", "final_step": step}) + "\n")
"""


def _fake_releasable_build_cmd(self, job):
    dur = float(job.overrides.get("duration", 1.0))
    extra = list(job.extra or [])
    resumed = extra[extra.index("--resume") + 1] if "--resume" in extra else "-"
    cmd = [
        sys.executable,
        "-c",
        _FAKE_RELEASABLE_TRAINER,
        job.progress_path,
        str(dur),
        str(job.dir),
        resumed,
    ]
    return cmd, os.environ.copy()


@pytest.fixture
def release_daemon(daemon, monkeypatch):
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_releasable_build_cmd)
    return daemon


@pytest.mark.core
def test_release_pause_frees_slot_and_resume_relaunches(release_daemon):
    """pause(release_model=True): the trainer saves + exits, the job parks as
    paused/released with no pid, the *next queued job runs* (the slot is
    free), and resume re-enqueues it at the front with --resume <state_dir>."""
    cl, mgr = release_daemon
    jid = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    nxt = cl.submit(method="lora", overrides={"duration": 0.3})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=15)
    pid = cl.get(jid)["pid"]

    res = cl.pause_job(jid, release_model=True)
    assert not res.get("error"), res
    assert res["release_requested"] is True
    assert res["state"] == "running"  # still running until the trainer exits

    assert _wait_until(
        lambda: cl.get(jid)["state"] == "paused" and cl.get(jid)["released"],
        timeout=15,
    )
    j = cl.get(jid)
    assert j["pid"] is None
    assert j["resume_state_dir"] and os.path.isdir(j["resume_state_dir"])
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() in (
        psutil.STATUS_ZOMBIE,
        psutil.STATUS_DEAD,
    )
    # The request file was consumed, the ack written.
    assert not (Path(j["resume_state_dir"]).parent / "pause.request").exists()
    assert (Path(j["resume_state_dir"]).parent / "pause.ack.json").exists()
    # Released ⇒ not the active job; the queue advanced past it.
    assert cl.health()["active_job"] is None or cl.health()["active_job"] == nxt
    assert _wait_until(lambda: cl.get(nxt)["state"] == "done", timeout=15)

    # Plain (freeze) pause on a released job is a no-op, not a crash.
    assert cl.pause_job(jid)["state"] == "paused"

    # Resume enqueues at priority 0: with the slot busy and another job already
    # waiting, the resumed one launches first once the slot frees.
    blocker = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(blocker)["state"] == "running", timeout=15)
    later = cl.submit(method="lora", overrides={"duration": 0.3})["job_id"]
    res = cl.resume_job(jid)
    assert not res.get("error"), res
    assert res["state"] == "queued" and res["relaunch"] == 1
    cl.stop(blocker)
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=20)
    assert _wait_until(lambda: cl.get(later)["state"] == "done", timeout=20)
    fin, lat = cl.get(jid), cl.get(later)
    assert fin["started_at"] <= lat["started_at"]
    assert "--resume" in fin["extra"] and "--skip_until_initial_step" in fin["extra"]
    assert fin["extra"][fin["extra"].index("--resume") + 1] == fin["resume_state_dir"]
    assert fin["resume_count"] == 1


def test_release_pause_of_frozen_job_thaws_first(release_daemon):
    """A SIGSTOP-frozen job can't act on the request: release thaws it, then
    the trainer saves and exits as usual."""
    cl, _ = release_daemon
    jid = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=15)
    assert cl.pause_job(jid)["state"] == "paused"
    res = cl.pause_job(jid, release_model=True)
    assert not res.get("error"), res
    assert _wait_until(
        lambda: cl.get(jid)["state"] == "paused" and cl.get(jid)["released"],
        timeout=15,
    )


def test_stop_released_job_is_terminal(release_daemon):
    cl, _ = release_daemon
    jid = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=15)
    cl.pause_job(jid, release_model=True)
    assert _wait_until(lambda: cl.get(jid)["released"], timeout=15)
    cl.stop(jid)
    assert cl.get(jid)["state"] == "stopped"
    assert "release" in (cl.get(jid)["status_detail"] or "")


@pytest.mark.parametrize("kill_jobs", [True, False])
def test_shutdown_queue_on_disk(tmp_path, monkeypatch, kill_jobs):
    """Terminate (``kill_jobs``) finalizes queued jobs ``stopped`` on disk, so the
    next boot's reconcile does not re-enqueue them; a graceful restart keeps them
    ``queued`` for the next daemon."""
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_build_cmd)
    mgr = JobManager()
    mgr.start()
    running = mgr.submit(
        method="lora",
        preset="default",
        methods_subdir=None,
        overrides={"duration": 30},
        start=True,
    )
    queued = [
        mgr.submit(method="lora", preset="default", methods_subdir=None, start=True)
        for _ in range(2)
    ]
    assert _wait_until(lambda: mgr.get(running.id).state == "running", timeout=10)
    mgr.shutdown(kill_jobs=kill_jobs)
    on_disk = jobs.load_all()
    want = "stopped" if kill_jobs else "queued"
    assert all(on_disk[j.id].state == want for j in queued)
    if not kill_jobs:
        mgr.stop(running.id)
