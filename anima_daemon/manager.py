"""The job manager: FIFO serial queue + worker thread + state table.

One worker thread drains a ``queue.Queue`` of job ids, spawns each detached
(so a console ctrl-C can't reach it), and monitors by polling
``(pid, create_time)`` liveness rather than awaiting a subprocess transport
(avoids Windows ProactorEventLoop bugs). On boot it reconciles ``jobs/`` to
re-attach a still-alive orphan or finalize a dead one as ``error`` (detail
``orphaned``). Exactly one job runs at a time. See ``anima_daemon/README.md`` for the job lifecycle.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from typing import Optional

import toml

from . import config, gpu, proc, tail
from .jobs import (
    ACTIVE_STATES,
    STATE_DONE,
    STATE_ERROR,
    STATE_PAUSED,
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_STOPPED,
    TERMINAL_STATES,
    Job,
    load_all,
    new_job_id,
    prune_jobs,
)

logger = logging.getLogger("anima.daemon")

_POLL_INTERVAL = 1.0  # seconds between liveness checks
_SENTINEL = "__stop__"
# Release-pause request file in the job dir (library/training/pause.py reads it).
PAUSE_REQUEST_NAME = "pause.request"

# Stall watchdog CPU-activity cross-check (see _stall_reason /
# _tree_cpu_advancing): past the output-freeze budget, only kill a job whose
# process tree has also gone CPU-idle (a quiet-but-computing embed/eval loop
# survives) — up to `budget × _STALL_CPU_GRACE`, after which it's killed
# regardless so a busy-spinning deadlock can't park the queue forever.
# job_id → (wall, cpu_seconds, verdict).
_CPU_SAMPLES: dict[str, tuple[float, float, bool]] = {}
_CPU_SAMPLE_MIN_GAP = 5.0  # seconds between samples (a short gap measures noise)
_CPU_BUSY_FRAC = 0.05  # ≥5% of one core, averaged over the gap → "computing"
_STALL_CPU_GRACE = 8.0  # hard ceiling multiplier on the configured budget

# Signal -> user-actionable hint for a process that died without a run_end
# event. POSIX Popen.poll() reports a signal death as a negative number;
# accelerate launch relays it as 128+N instead.
_SIGNAL_HINTS = {
    9: "killed (SIGKILL) — almost always out of memory. Lower batch size, "
    "raise blocks_to_swap, or try PRESET=low_vram.",
    6: "aborted (SIGABRT) — usually a CUDA assert / illegal memory access. "
    "See the last traceback above.",
    11: "segfault (SIGSEGV) — a native crash. See the last traceback above.",
    15: "terminated (SIGTERM).",
}


def _classify_exit(rc) -> str:
    """Human-readable diagnosis for a nonzero/unknown process exit code."""
    sig = None
    if rc is not None and rc < 0:
        sig = -rc
    elif rc is not None and rc > 128:
        sig = rc - 128
    if sig in _SIGNAL_HINTS:
        return f"process exited (code={rc}): {_SIGNAL_HINTS[sig]}"
    return (
        f"process exited (code={rc}) — crashed before finishing. "
        "See the last traceback above."
    )


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        # (priority, seq, job_id): normal submissions FIFO at 1; a resumed
        # release-paused job re-enters at 0 so it runs as soon as the slot
        # frees; the shutdown sentinel at -1 wins over everything.
        self._queue: "queue.PriorityQueue[tuple[int, int, str]]" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._popens: dict[str, object] = {}  # job_id -> Popen (spawned only)
        self._adopt: list[str] = []  # running orphans to monitor before the queue
        self._subscribers: set["queue.Queue[dict]"] = set()
        self._stopping = False
        self._kill_on_shutdown = False
        # Queue run gate: set -> worker launches queued jobs; cleared -> paused
        # (dequeued jobs held `queued`, a running job left alone). Default set.
        self._run_gate = threading.Event()
        self._run_gate.set()
        # Bumped every loop iteration/monitor poll; exposed via /health so a
        # wedged-or-dead worker is distinguishable from a healthy long run.
        self._worker_heartbeat = time.time()
        self._worker = threading.Thread(
            target=self._run, name="anima-job-worker", daemon=True
        )

    def start(self) -> None:
        config.ensure_state_dirs()
        self._reconcile()
        self._worker.start()

    def shutdown(self, *, kill_jobs: bool) -> None:
        """Stop accepting work and unblock the worker. With ``kill_jobs`` the
        active job tree is torn down and the GPU freed before the daemon exits.
        """
        with self._lock:
            self._stopping = True
            self._kill_on_shutdown = kill_jobs
            current = self._current_running_locked()
        if kill_jobs and current is not None:
            current.stop_requested = True
            self._kill_job_tree(current)
        self._run_gate.set()  # release a worker parked on a paused queue
        self._enqueue(_SENTINEL, priority=-1)  # wake the worker so it can exit

    def submit(
        self,
        *,
        method: str,
        preset: str,
        methods_subdir: Optional[str],
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
        overrides: Optional[dict] = None,
        extra: Optional[list[str]] = None,
        from_chain: bool = False,
        start: Optional[bool] = None,
        captured_env: Optional[dict] = None,
    ) -> Job:
        job = Job(
            id=new_job_id(),
            method=method,
            preset=preset,
            methods_subdir=methods_subdir,
            overrides=dict(overrides or {}),
            extra=list(extra or []),
            from_chain=from_chain,
            captured_env=dict(captured_env or {}),
        )
        self._attach_config_file(
            job, config_snapshot=config_snapshot, config_file=config_file
        )
        return self._register_and_queue(job, start=start)

    def submit_command(
        self,
        *,
        label: str,
        argv: list[str],
        extra_env: Optional[dict] = None,
        chain_train: Optional[dict] = None,
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
        start: Optional[bool] = None,
        captured_env: Optional[dict] = None,
        stall_timeout: Optional[float] = None,
    ) -> Job:
        """Enqueue a plain ``python <argv>`` task on the same serial queue as
        training. ``chain_train`` auto-enqueues that training spec on success
        (see ``_finalize``). ``stall_timeout`` overrides the command-job stall
        budget for this job."""
        job = Job(
            id=new_job_id(),
            method=label,
            preset="",
            kind="command",
            argv=list(argv or []),
            extra_env=dict(extra_env or {}),
            captured_env=dict(captured_env or {}),
            chain_train=dict(chain_train) if chain_train else None,
            stall_timeout=stall_timeout,
        )
        self._attach_config_file(
            job, config_snapshot=config_snapshot, config_file=config_file
        )
        if job.config_file:
            job.extra_env["CONFIG_FILE"] = job.config_file
            if job.chain_train is not None:
                job.chain_train.setdefault("config_file", job.config_file)
        return self._register_and_queue(job, start=start)

    def _attach_config_file(
        self,
        job: Job,
        *,
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
    ) -> None:
        """Write/copy an immutable config snapshot into this job directory."""
        if not config_snapshot and not config_file:
            return
        dst = config.job_dir(job.id) / "config.snapshot.toml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if config_snapshot:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            tmp.write_text(toml.dumps(config_snapshot), encoding="utf-8")
            tmp.replace(dst)
        else:
            src = os.path.abspath(str(config_file))
            if os.path.abspath(str(dst)) != src:
                shutil.copyfile(src, dst)
        job.config_file = str(dst)

    def _register_and_queue(self, job: Job, *, start: Optional[bool] = None) -> Job:
        # `start` sets the run gate atomically with enqueue. False pauses the
        # gate only when the queue is otherwise idle — with a job already
        # running/queued, pausing would stop this one auto-advancing behind it.
        # True resumes (flushing any held backlog); None leaves the gate as-is.
        if start is False:
            with self._lock:
                queue_idle = self._queue_is_idle_locked()
            if queue_idle:
                self.pause()
        d = config.job_dir(job.id)
        job.progress_path = str(d / "progress.jsonl")
        job.stdout_path = str(d / "stdout.log")
        with self._lock:
            self._jobs[job.id] = job
            job.persist()
        self._enqueue(job.id)
        if start is True:
            self.resume()
        self._broadcast({"ev": "submitted", "job_id": job.id, "state": job.state})
        return job

    def pause(self) -> None:
        """Hold the queue: queued jobs stay ``queued`` until :meth:`resume`. A
        job already running is left alone — only the next launch waits."""
        if self._run_gate.is_set():
            self._run_gate.clear()
            self._broadcast({"ev": "queue_state", "paused": True})

    def resume(self) -> None:
        """Release a paused queue so the worker launches queued jobs in order."""
        if not self._run_gate.is_set():
            self._run_gate.set()
            self._broadcast({"ev": "queue_state", "paused": False})

    def is_paused(self) -> bool:
        return not self._run_gate.is_set()

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def stale_for(self, job: Job) -> Optional[float]:
        """Seconds since the job's last progress event, for a running job."""
        if job.state != STATE_RUNNING:
            return None
        ev = tail.last_event(job.progress_path)
        if not ev:
            return None
        # progress ts is relative to run start; compare wall clock instead.
        try:
            mtime = os.path.getmtime(job.progress_path)
        except OSError:
            return None
        return round(time.time() - mtime, 1)

    def stop(self, job_id: Optional[str] = None) -> Optional[Job]:
        """Abort a job. ``None`` → the running job. Queued → cancelled in place;
        running → tree killed, GPU freed. The daemon stays up and advances to
        the next queued job."""
        with self._lock:
            job = self._jobs.get(job_id) if job_id else self._current_running_locked()
            if job is None or job.state in TERMINAL_STATES:
                return job
            job.stop_requested = True
            state = job.state
            if state == STATE_PAUSED and job.released:
                self._finalize(job, STATE_STOPPED, detail="cancelled while released")
                return job
            if state == STATE_QUEUED:
                # Finalize now (reentrant RLock): the worker may be blocked on a
                # running job and won't reach this id for a while. When it does,
                # it skips ids whose state isn't QUEUED.
                self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
                return job
            job.persist()
        if state in ACTIVE_STATES:
            # Running or paused (frozen): either way tree-kill it. _kill_job_tree
            # thaws a paused tree first so the SIGTERM is actually delivered.
            self._kill_job_tree(job)
        return job

    def pause_job(self, job_id: str, *, release_model: bool = False) -> Optional[dict]:
        """Freeze a running job's process tree (SIGSTOP), method-agnostically.
        CUDA context/VRAM stay put. The queue does NOT advance past a paused
        job — it still owns its slot. Refuses anything not ``running`` and a
        multi-GPU ``accelerate launch`` run (a frozen NCCL rank trips the
        collective heartbeat). Returns ``{job_id, state, error?}``, or
        ``None`` when no such job (server maps that to 404).

        ``release_model=True`` is the cooperative variant for train.py jobs:
        drop a ``pause.request`` in the job dir; the trainer saves a resumable
        state at its next optimizer step and exits with ``run_end paused``,
        the job parks as ``paused`` with no process (``released``), the GPU is
        free and the queue advances. ``resume_job`` relaunches it with
        ``--resume``. The job stays ``running`` until the trainer has actually
        exited; ``release_requested`` marks the in-between."""
        if release_model:
            return self._request_release(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state == STATE_PAUSED:
                return {"job_id": job.id, "state": job.state}  # idempotent
            if job.release_requested:
                return {
                    "job_id": job.id,
                    "state": job.state,
                    "error": "a release-pause is already in flight (trainer is saving state)",
                }
            if job.state != STATE_RUNNING:
                return {
                    "job_id": job.id,
                    "state": job.state,
                    "error": f"can only pause a running job (current state: {job.state})",
                }
            if job.accelerate_launched:
                return {
                    "job_id": job.id,
                    "state": job.state,
                    "error": "refusing to pause a multi-GPU accelerate-launch run "
                    "(a frozen NCCL rank trips the collective heartbeat timeout)",
                }
            # Flip to paused BEFORE suspending, so the monitor loop already skips
            # the stall watchdog by the time the tree stops writing output. If the
            # suspend then races a natural exit, _finalize_from_exit (paused isn't
            # terminal) still finalizes correctly.
            job.state = STATE_PAUSED
            job.paused_at = time.time()
            job.persist()
            pid = job.pid
        if pid is not None:
            proc.suspend_tree(pid)
        self._broadcast({"ev": "paused", "job_id": job_id})
        return {"job_id": job_id, "state": STATE_PAUSED}

    def resume_job(self, job_id: str) -> Optional[dict]:
        """Thaw a paused job's process tree (SIGCONT) back to ``running``.
        Returns ``{job_id, state, error?}`` (error if not paused), or
        ``None`` for no such job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state != STATE_PAUSED:
                return {
                    "job_id": job.id,
                    "state": job.state,
                    "error": f"job is not paused (current state: {job.state})",
                }
            if job.released:
                return self._requeue_released_locked(job)
            pid = job.pid
        if pid is not None:
            proc.resume_tree(pid)
        with self._lock:
            # Guard against a concurrent stop/exit having moved it on already.
            if job.state == STATE_PAUSED:
                job.state = STATE_RUNNING
                job.paused_at = None
                job.persist()
        self._broadcast({"ev": "resumed", "job_id": job_id})
        return {"job_id": job_id, "state": job.state}

    def _request_release(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state == STATE_PAUSED and job.released:
                return {"job_id": job.id, "state": job.state, "released": True}
            if job.release_requested:
                return {"job_id": job.id, "state": job.state, "release_requested": True}
            if job.kind != "train":
                return {
                    "job_id": job.id,
                    "state": job.state,
                    "error": "release_model needs a train.py job (cooperative "
                    "checkpoint); a command job only supports the plain freeze",
                }
            if job.state not in ACTIVE_STATES:
                return {
                    "job_id": job.id,
                    "state": job.state,
                    "error": f"can only pause a running job (current state: {job.state})",
                }
            was_frozen = job.state == STATE_PAUSED
            try:
                job.dir.mkdir(parents=True, exist_ok=True)
                with open(job.dir / PAUSE_REQUEST_NAME, "w", encoding="utf-8") as f:
                    json.dump({"release_model": True, "ts": time.time()}, f)
            except OSError as exc:
                return {
                    "job_id": job.id,
                    "state": job.state,
                    "error": f"could not write pause request: {exc}",
                }
            job.release_requested = True
            job.status_detail = "release-pause requested: saving resumable state"
            if was_frozen:
                # A frozen tree can't act on the request — thaw it first.
                job.state = STATE_RUNNING
                job.paused_at = None
            job.persist()
            pid = job.pid
        if was_frozen and pid is not None:
            proc.resume_tree(pid)
        self._broadcast({"ev": "release_requested", "job_id": job_id})
        return {"job_id": job_id, "state": STATE_RUNNING, "release_requested": True}

    def _park_released(self, job: Job, ev: dict) -> None:
        """The trainer exited on a release-pause: keep the job as ``paused``
        with no process. Not terminal — ``resume_job`` relaunches it."""
        _CPU_SAMPLES.pop(job.id, None)
        with self._lock:
            job.state = STATE_PAUSED
            job.released = True
            job.release_requested = False
            job.paused_at = time.time()
            job.pid = None
            job.create_time = None
            job.resume_state_dir = ev.get("state_dir") or job.resume_state_dir
            job.ckpt_path = tail.last_ckpt_path(job.progress_path) or job.ckpt_path
            job.status_detail = (
                f"released at step {ev.get('final_step')}; GPU free — "
                "resume relaunches from the saved state"
            )
            job.persist()
        self._broadcast({"ev": "paused", "job_id": job.id, "released": True})

    def _requeue_released_locked(self, job: Job) -> dict:
        """Relaunch a release-paused train job from its saved state (called
        under the lock). ``--resume <state_dir> --skip_until_initial_step``
        replaces any earlier resume flags in ``extra``."""
        if not job.resume_state_dir or not os.path.isdir(job.resume_state_dir):
            return {
                "job_id": job.id,
                "state": job.state,
                "error": f"resume state dir missing: {job.resume_state_dir!r}",
            }
        extra = list(job.extra or [])
        for flag in ("--resume",):
            while flag in extra:
                i = extra.index(flag)
                del extra[i : i + 2]
        while "--skip_until_initial_step" in extra:
            extra.remove("--skip_until_initial_step")
        extra += ["--resume", job.resume_state_dir, "--skip_until_initial_step"]
        job.extra = extra
        job.state = STATE_QUEUED
        job.released = False
        job.paused_at = None
        job.started_at = None
        job.ended_at = None
        job.returncode = None
        job.resume_count += 1
        job.status_detail = (
            f"resuming from {job.resume_state_dir} (relaunch #{job.resume_count})"
        )
        job.persist()
        self._enqueue(job.id, priority=0)
        self._broadcast({"ev": "resumed", "job_id": job.id, "relaunch": True})
        return {"job_id": job.id, "state": STATE_QUEUED, "relaunch": job.resume_count}

    def _run(self) -> None:
        # Drain re-attached orphans before touching the queue so the serial
        # GPU invariant holds across a daemon restart. Crash-guarded like the
        # main loop: a monitor that raises must not strand the queue behind it.
        for job_id in self._adopt:
            self._worker_heartbeat = time.time()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._monitor(job, popen=None)
            except Exception:  # noqa: BLE001
                logger.exception("monitor crashed for adopted job %s", job_id)
                self._fail_safely(job_id, "daemon monitor crashed; see daemon.log")
        while True:
            _prio, _seq, job_id = self._queue.get()
            self._worker_heartbeat = time.time()
            if job_id == _SENTINEL:
                break
            with self._lock:
                if self._stopping:
                    break
            # A dead worker leaves every later job stuck `queued` forever, so a
            # crash fails only this job and the loop keeps draining.
            try:
                self._process_one(job_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "worker crashed handling job %s; queue continues", job_id
                )
                self._fail_safely(
                    job_id, "daemon worker hit an unexpected error; see daemon.log"
                )

    def _process_one(self, job_id: str) -> None:
        """Launch + monitor a single dequeued job. Uses ``return`` (not the
        loop's ``continue``) so it can run under the crash guard in ``_run``."""
        job = self._jobs.get(job_id)
        if job is None or job.state != STATE_QUEUED:
            return
        if job.stop_requested:
            self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
            return
        # Hold while the queue is paused; re-validate after waking (cancelled
        # while held, or daemon shutting down).
        if not self._await_run_gate(job):
            return
        with self._lock:
            if job.state != STATE_QUEUED or job.stop_requested:
                return
        # Auto-chained train steps skip the guard (see Job.from_chain).
        if not job.from_chain:
            self._gpu_guard(job)
        self._launch_and_monitor(job)

    def _fail_safely(self, job_id: str, error: str) -> None:
        """Finalize a job ERROR without propagating, even if finalize raises."""
        job = self.get(job_id)
        if job is None or job.state in TERMINAL_STATES:
            return
        try:
            self._finalize(job, STATE_ERROR, error=error)
        except Exception:  # noqa: BLE001
            logger.exception("failed to finalize crashed job %s", job_id)

    def worker_idle_for(self) -> float:
        """Seconds since the worker last advanced. Large + a job stuck ``queued``
        ⇒ the worker is wedged or dead. Exposed via /health."""
        return round(time.time() - self._worker_heartbeat, 1)

    def worker_alive(self) -> bool:
        return self._worker.is_alive()

    def _await_run_gate(self, job: Job) -> bool:
        """Block while the queue is paused. Returns True when cleared to launch,
        False if the worker should skip this job (daemon stopping, or the job was
        cancelled while held). Polls so a stop/shutdown is noticed promptly even
        though the gate itself stays closed."""
        if self._run_gate.is_set():
            return True
        self._broadcast({"ev": "queue_held", "job_id": job.id})
        while not self._run_gate.wait(timeout=1.0):
            with self._lock:
                if self._stopping:
                    return False
                cur = self._jobs.get(job.id)
                if cur is None or cur.stop_requested or cur.state in TERMINAL_STATES:
                    return False
        return not self._stopping

    def _launch_and_monitor(self, job: Job) -> None:
        # A stale pause.request (daemon restarted mid-release, or a run that
        # ended before acting on it) must not pause the relaunch on step 1.
        try:
            (job.dir / PAUSE_REQUEST_NAME).unlink()
        except OSError:
            pass
        job.release_requested = False
        d = config.job_dir(job.id)
        try:
            # _build_cmd runs the config merge + task-runner import for train
            # jobs; inside the try so a bad config / import fails only this job.
            cmd, env = self._build_cmd(job)
            popen = proc.spawn_detached(
                cmd,
                cwd=config.ROOT,
                stdout_path=d / "stdout.log",
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            self._finalize(job, STATE_ERROR, error=f"launch failed: {exc}")
            return
        with self._lock:
            job.state = STATE_RUNNING
            job.started_at = time.time()
            job.pid = popen.pid
            job.create_time = proc.create_time(popen.pid)
            # pause_job refuses accelerate-launched (multi-GPU) runs.
            job.accelerate_launched = "accelerate.commands.accelerate_cli" in cmd
            job.persist()
            self._popens[job.id] = popen
        self._broadcast({"ev": "started", "job_id": job.id, "pid": job.pid})
        self._monitor(job, popen=popen)

    def _monitor(self, job: Job, *, popen) -> None:
        """Block until the job process exits, then finalize. Works for both a
        process we spawned (``popen`` reaps the child) and an adopted orphan
        (``popen is None`` → psutil liveness)."""
        config.point_current_job(job.id)
        while self._proc_running(job, popen):
            self._worker_heartbeat = time.time()
            if self._kill_on_shutdown:
                self._kill_job_tree(job)
                break
            if job.state == STATE_PAUSED:
                # A frozen tree writes nothing: skip the stall watchdog, keep
                # polling liveness.
                time.sleep(_POLL_INTERVAL)
                continue
            stalled = self._stall_reason(job)
            if stalled is not None:
                logger.warning("job %s killed by stall watchdog: %s", job.id, stalled)
                self._kill_job_tree(job)
                # Finalize now so _finalize_from_exit no-ops and the stall
                # diagnostic isn't overwritten by the SIGKILL classification.
                self._finalize(job, STATE_ERROR, error=stalled)
                break
            time.sleep(_POLL_INTERVAL)
        # Reap our own child to avoid a zombie.
        if popen is not None:
            try:
                popen.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self._popens.pop(job.id, None)
        self._finalize_from_exit(job, popen)

    @staticmethod
    def _proc_running(job: Job, popen) -> bool:
        if popen is not None:
            return popen.poll() is None
        return proc.is_alive(job.pid, job.create_time)

    @staticmethod
    def _stall_reason(job: Job) -> Optional[str]:
        """If the job has produced no output (stdout.log or progress.jsonl
        mtime) for longer than its stall budget, return an actionable error
        naming where it wedged; otherwise ``None``. Per-kind budgets are in
        config.py; a tree still burning CPU is spared up to
        ``_STALL_CPU_GRACE`` × the budget."""
        timeout = (
            job.stall_timeout
            if getattr(job, "stall_timeout", None) is not None
            else (
                config.CMD_STALL_TIMEOUT
                if job.kind == "command"
                else config.JOB_STALL_TIMEOUT
            )
        )
        if not timeout or timeout <= 0 or job.started_at is None:
            return None
        last = job.started_at
        for path in (job.stdout_path, job.progress_path):
            if not path:
                continue
            try:
                last = max(last, os.path.getmtime(path))
            except OSError:
                continue
        idle = time.time() - last
        if idle < timeout:
            _CPU_SAMPLES.pop(job.id, None)  # output flowing again → forget samples
            return None
        busy = JobManager._tree_cpu_advancing(job)
        if busy and idle < timeout * _STALL_CPU_GRACE:
            return None
        where = JobManager._last_output_line(job)
        detail = f" last output: {where!r}" if where else " (no output captured)"
        if busy:
            detail += (
                f" (process tree still burning CPU, but silent past "
                f"{_STALL_CPU_GRACE:g}× the budget — raise or disable it with "
                "stall_timeout on submit if this run is healthy)"
            )
        return (
            f"stalled: no output for {int(idle)}s (limit {int(timeout)}s); daemon "
            f"killed the job so the queue can advance.{detail}"
        )

    @staticmethod
    def _tree_cpu_advancing(job: Job) -> bool:
        """True iff the job's process tree has burned CPU since the last
        sample. Needs two samples ``_CPU_SAMPLE_MIN_GAP`` apart for a
        meaningful rate; the first look (or one inside the gap) reuses the
        previous verdict, starting optimistic. Unreadable CPU times (pid gone)
        return False, falling back to output-mtime-only behavior."""
        cpu = proc.tree_cpu_seconds(getattr(job, "pid", None))
        if cpu is None:
            return False
        now = time.time()
        prev = _CPU_SAMPLES.get(job.id)
        if prev is None:
            _CPU_SAMPLES[job.id] = (now, cpu, True)
            return True
        gap = now - prev[0]
        if gap < _CPU_SAMPLE_MIN_GAP:
            return prev[2]
        verdict = (cpu - prev[1]) / gap >= _CPU_BUSY_FRAC
        _CPU_SAMPLES[job.id] = (now, cpu, verdict)
        return verdict

    @staticmethod
    def _last_output_line(job: Job, *, max_bytes: int = 8192) -> Optional[str]:
        """Best-effort last non-empty stdout line (carriage-return aware, so a
        tqdm bar's latest redraw wins), for the stall error."""
        path = job.stdout_path
        if not path:
            return None
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                blob = f.read()
        except OSError:
            return None
        parts = [
            p.strip() for p in re.split(r"[\r\n]", blob.decode("utf-8", "replace"))
        ]
        parts = [p for p in parts if p]
        return parts[-1] if parts else None

    def _finalize_from_exit(self, job: Job, popen) -> None:
        if job.state in TERMINAL_STATES:
            return
        ev = tail.last_event(job.progress_path)
        rc = popen.poll() if popen is not None else None
        # Mirror the OS exit code before finalizing so the persist captures it
        # (CLI waiters exit with it). None for an adopted orphan.
        job.returncode = rc
        if job.stop_requested:
            self._finalize(job, STATE_STOPPED)
            return
        if ev and ev.get("ev") == "run_end":
            status = ev.get("status")
            if status == "paused":
                self._park_released(job, ev)
                return
            mapped = {
                "ok": STATE_DONE,
                "stopped": STATE_STOPPED,
                "error": STATE_ERROR,
            }.get(status, STATE_ERROR)
            self._finalize(job, mapped, error=ev.get("error"))
            return
        if rc == 0:
            self._finalize(job, STATE_DONE)
        else:
            # No run_end + nonzero exit: classify the code — signal deaths
            # (SIGKILL/OOM, CUDA SIGABRT, segfault) leave no traceback.
            self._finalize(job, STATE_ERROR, error=_classify_exit(rc))

    def _finalize(
        self,
        job: Job,
        state: str,
        *,
        error: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        _CPU_SAMPLES.pop(job.id, None)  # drop the watchdog's CPU-rate samples
        with self._lock:
            job.state = state
            job.ended_at = time.time()
            job.release_requested = False
            if error:
                job.error = error
            if detail:
                job.status_detail = detail
            job.ckpt_path = tail.last_ckpt_path(job.progress_path)
            self._lift_result(job)
            # Auto-chain: a done command job with a chain_train spec enqueues its
            # follow-on train job. chained_job_id persists in the same write that
            # flips the state to `done`.
            if (
                state == STATE_DONE
                and job.kind == "command"
                and job.chain_train
                and not job.chained_job_id
            ):
                ct = job.chain_train
                follow = self.submit(
                    method=ct.get("method"),
                    preset=ct.get("preset") or "default",
                    methods_subdir=ct.get("methods_subdir"),
                    config_snapshot=ct.get("config_snapshot") or None,
                    config_file=ct.get("config_file") or None,
                    overrides=ct.get("overrides") or {},
                    extra=ct.get("extra") or [],
                    from_chain=True,
                    # Inherit the originating command's captured env.
                    captured_env=job.captured_env,
                )
                job.chained_job_id = follow.id
                logger.info(
                    "auto-chain: job %s done → enqueued training %s",
                    job.id,
                    follow.id,
                )
            job.persist()
        self._broadcast({"ev": "ended", "job_id": job.id, "state": state})

    def _lift_result(self, job: Job) -> None:
        """Follow a bench envelope pointer into the job record (README "Result
        envelopes"). Best-effort: a missing or corrupt pointer leaves the fields
        None; ``label``/``metrics`` are lifted opaquely, never validated."""
        import json
        from pathlib import Path

        pointer = job.dir / "result_path.json"
        try:
            path = json.loads(pointer.read_text(encoding="utf-8")).get("path")
        except (OSError, ValueError, AttributeError):
            return
        if not path:
            return
        job.result_path = path
        try:
            envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(envelope, dict):
            job.result_summary = {
                "label": envelope.get("label"),
                "metrics": envelope.get("metrics"),
            }

    def _gpu_guard(
        self,
        job: Job,
        *,
        retries: int = config.GPU_GUARD_RETRIES,
        delay: float = config.GPU_GUARD_DELAY,
        busy_frac: float = config.GPU_GUARD_BUSY_FRAC,
    ) -> None:
        """Before launching, make sure the GPU is actually free.

        Busy/free is decided from **total VRAM in use**, not the process list
        (on Windows WDDM desktop apps show up as compute processes). Process
        enumeration is used only to reap VRAM leaked by our *own* dead jobs,
        matched by **(pid, create_time)** — a bare pid match could kill a
        stranger that reused the pid (issue #83). If memory can't be probed,
        assume free. Tunable via ANIMA_DAEMON_GPU_{BUSY_FRAC,RETRIES,DELAY}.
        """
        # Ask a resident inference server to free VRAM before we launch — it
        # stays alive and reloads on its next request. Best-effort.
        self._evict_resident_inference()

        for attempt in range(retries):
            # Reap only holders matching a known job on (pid, create_time).
            holders = gpu.gpu_pids() or set()
            with self._lock:
                known = {
                    j.pid: j
                    for j in self._jobs.values()
                    if j.pid in holders and proc.is_alive(j.pid, j.create_time)
                }
            reaped = False
            for pid, owner in known.items():
                if owner.id == job.id:
                    continue
                logger.warning(
                    "gpu_guard: reaping leaked VRAM from job %s (pid %s)", owner.id, pid
                )
                proc.kill_tree(pid)
                reaped = True
            if reaped:
                time.sleep(0.5)  # let the killed procs release VRAM

            mem = gpu.gpu_mem()
            if mem is None:  # can't tell → don't deadlock the queue
                return
            used, total = mem
            if total <= 0 or used / total < busy_frac:
                return  # GPU effectively free → go
            logger.warning(
                "gpu_guard: GPU busy — %d/%d MiB used (attempt %d/%d)",
                used,
                total,
                attempt + 1,
                retries,
            )
            self._broadcast(
                {
                    "ev": "gpu_wait",
                    "job_id": job.id,
                    "used_mib": used,
                    "total_mib": total,
                }
            )
            time.sleep(delay)
        # Give up waiting and launch anyway; never kill what we didn't start.
        job.status_detail = "launched despite busy GPU"

    def _kill_job_tree(self, job: Job) -> None:
        if job.pid is None:
            return
        # A SIGSTOP'd tree ignores SIGTERM until resumed — thaw first so the
        # graceful terminate lands instead of waiting out the kill grace.
        if job.state == STATE_PAUSED:
            proc.resume_tree(job.pid)
        proc.kill_tree(job.pid)

    def _evict_resident_inference(self) -> None:
        """Ask a resident inference server (if any) to free VRAM before launch.

        Discovery mirrors scripts/inference_server.py's pidfiles (inline, no
        import). Every failure is swallowed; the server's idle-TTL frees the
        card eventually anyway.
        """
        import json
        import urllib.request
        from pathlib import Path

        candidates = []
        override = os.environ.get("ANIMA_INFERENCE_PIDFILE")
        if override:
            candidates.append(Path(override))
        candidates += [
            config.ROOT / "output" / "inference" / "server.json",
            Path.home() / ".anima" / "inference.json",
        ]
        for pf in candidates:
            try:
                port = json.loads(pf.read_text()).get("port")
            except (OSError, ValueError):
                continue
            if not port:
                continue
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"http://127.0.0.1:{port}/unload", method="POST"
                    ),
                    timeout=5,
                ).read()
                logger.info("gpu_guard: inference server (port %s) unloaded", port)
                time.sleep(1.0)  # let VRAM release before we measure
            except Exception:  # noqa: BLE001 — best-effort
                pass
            return

    def _build_cmd(self, job: Job) -> tuple[list[str], dict]:
        from .client import venv_python

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Force UTF-8 stdio in the job tree so a non-ASCII char (em-dash, etc.)
        # never crashes a child on a non-UTF-8 console locale (e.g. Korean
        # Windows cp949 → UnicodeEncodeError). Inherited by grandchildren.
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # tqdm redraws at 0.1s cadence drown stdout.log's real lines; the GUI
        # tracker parses only the latest line and training has progress.jsonl.
        env.setdefault("TQDM_MININTERVAL", "10")

        # daemon-env ← captured_env ← extra_env: update, not setdefault, so the
        # caller's value wins; a command job's extra_env is applied after.
        env.update(job.captured_env or {})

        # Result envelope lift: bench/_common.write_result reads these to drop
        # a `result_path.json` pointer.
        env["ANIMA_DAEMON_JOB_ID"] = job.id
        env["ANIMA_DAEMON_JOB_DIR"] = str(job.dir)

        # Windowless interpreter (see client.venv_python). No --progress_jsonl:
        # command jobs finalize on exit code.
        if job.kind == "command":
            env.update(job.extra_env or {})
            return [venv_python(windowless=True), *job.argv], env

        # Lazy: the task runner's imports load only when a train job launches.
        from scripts.tasks._common import build_launch_cmd, build_method_args

        overrides = dict(job.overrides or {})
        extra = list(job.extra or [])
        # Dict overrides → --key value (unless already in extra). train.py bools
        # are `store_true`: a False override is dropped, so a caller can't force
        # a flag the config chain turns on back off here.
        for key, val in overrides.items():
            flag = f"--{key}"
            if flag in extra:
                continue
            if isinstance(val, bool):
                if val:
                    extra.append(flag)
            elif key == "target_res" and isinstance(val, (list, tuple)):
                extra += [flag, *[str(v) for v in val]]
            else:
                extra += [flag, str(val)]
        # Point the progress stream at the job dir.
        if "--progress_jsonl" not in extra:
            extra += ["--progress_jsonl", job.progress_path or ""]
        if job.config_file:
            args = ["--config_file", job.config_file, *extra]
        else:
            args = build_method_args(
                job.method,
                preset=job.preset,
                methods_subdir=job.methods_subdir,
                extra=extra,
            )
        # Windowless interpreter (see client.venv_python).
        cmd = build_launch_cmd(*args, python_exe=venv_python(windowless=True))
        return cmd, env

    def _reconcile(self) -> None:
        # Prune BEFORE load_all: a pruned job in the in-memory table would be
        # recreated by a later persist(). Best-effort; never blocks boot.
        try:
            pruned = prune_jobs()
            if pruned["pruned"]:
                logger.info(
                    "pruned %d terminal job dirs older than %gd (%.1f MB freed, "
                    "%d kept)",
                    len(pruned["pruned"]),
                    pruned["max_age_days"],
                    pruned["freed_bytes"] / 1e6,
                    pruned["kept"],
                )
        except Exception as exc:  # noqa: BLE001 — never block boot on retention
            logger.warning("job-dir prune failed (continuing): %s", exc)

        self._jobs = load_all()
        for job in self._jobs.values():
            if job.state == STATE_PAUSED and job.released:
                continue  # parked on disk, no process to re-attach
            if job.state in ACTIVE_STATES:
                if proc.is_alive(job.pid, job.create_time):
                    # A paused tree stays SIGSTOP'd across a daemon restart;
                    # re-adopt it as-is.
                    logger.info(
                        "reconcile: re-attaching live %s job %s", job.state, job.id
                    )
                    self._adopt.append(job.id)
                else:
                    logger.info("reconcile: job %s died while we were down", job.id)
                    job.stop_requested = False
                    self._finalize(
                        job,
                        STATE_ERROR,
                        error="daemon was down when the process exited",
                        detail="orphaned",
                    )
            elif job.state == STATE_QUEUED:
                self._enqueue(job.id)

    def _enqueue(self, job_id: str, *, priority: int = 1) -> None:
        self._queue.put((priority, next(self._seq), job_id))

    @staticmethod
    def _occupies_slot(job: Job) -> bool:
        """Running, or frozen in place (VRAM held). A release-paused job has
        no process and owns nothing."""
        return job.state in ACTIVE_STATES and not (
            job.state == STATE_PAUSED and job.released
        )

    def _current_running_locked(self) -> Optional[Job]:
        # The job occupying the worker/GPU slot — running or frozen. A paused
        # job still owns its VRAM, so stop/shutdown/health must all see it.
        for job in self._jobs.values():
            if self._occupies_slot(job):
                return job
        return None

    def _queue_is_idle_locked(self) -> bool:
        """True when no job is running or waiting to run. The just-submitted
        job is not yet in ``_jobs`` when ``_register_and_queue`` calls this."""
        return not any(
            job.state == STATE_QUEUED or self._occupies_slot(job)
            for job in self._jobs.values()
        )

    def active_job(self) -> Optional[Job]:
        """The running or paused job, if any (lock-safe public accessor)."""
        with self._lock:
            return self._current_running_locked()

    def subscribe(self) -> "queue.Queue[dict]":
        q: "queue.Queue[dict]" = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "queue.Queue[dict]") -> None:
        with self._lock:
            self._subscribers.discard(q)

    def _broadcast(self, event: dict) -> None:
        event.setdefault("ts", time.time())
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow consumer; drop rather than block the worker
