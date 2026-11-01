"""The job manager: FIFO serial queue + worker thread + state table.

One worker thread drains a ``queue.Queue`` of job ids, spawns each detached
(so a console ctrl-C can't reach it), and monitors by polling
``(pid, create_time)`` liveness rather than awaiting a subprocess transport
(avoids Windows ProactorEventLoop bugs). On boot it reconciles ``jobs/`` to
re-attach a still-alive orphan or finalize a dead one as ``error`` (detail
``orphaned``). Exactly one job runs at a time. Job lifecycle:
``anima_daemon/README.md``.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import shutil
import threading
import time
from typing import Optional

import toml

from . import config, gpu_guard, launch, proc, tail
from .jobs import Job, load_all, new_job_id, prune_jobs
from .record import (
    ACTIVE_STATES,
    STATE_DONE,
    STATE_ERROR,
    STATE_PAUSED,
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_STOPPED,
    TERMINAL_STATES,
)
from .watchdog import StallWatchdog

logger = logging.getLogger("anima.daemon")

_POLL_INTERVAL = 1.0  # seconds between liveness checks
_SENTINEL = "__stop__"
# Release-pause request file in the job dir (library/runtime/pause.py reads it).
PAUSE_REQUEST_NAME = "pause.request"

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
        self._watchdog = StallWatchdog()
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
        # `start`: False pauses the run gate only when the queue is otherwise
        # idle (with a job already running/queued, pausing would stop this one
        # auto-advancing behind it). True resumes, flushing any held backlog;
        # None leaves the gate as-is.
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
            self._kill_job_tree(job)
        return job

    def pause_job(self, job_id: str, *, release_model: bool = False) -> Optional[dict]:
        """Freeze a running job's process tree (SIGSTOP). Refuses anything not
        ``running`` and a multi-GPU ``accelerate launch`` run. Returns
        ``{job_id, state, error?}``, or ``None`` when no such job (server maps
        that to 404).

        ``release_model=True`` drops a ``pause.request`` for the trainer instead
        (README ``POST /jobs/{id}/pause``). The job stays ``running`` until the
        trainer has actually exited; ``release_requested`` marks the in-between.
        """
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
        self._watchdog.forget(job.id)
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
        """Launch + monitor a single dequeued job."""
        job = self._jobs.get(job_id)
        if job is None or job.state != STATE_QUEUED:
            return
        if job.stop_requested:
            self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
            return
        # Hold while the queue is paused; re-validate after waking (cancelled
        # while held, or daemon shutting down).
        if not self._await_run_gate(job) or not self._launchable(job):
            return
        # Auto-chained train steps skip the guard (see Job.from_chain).
        if not job.from_chain:
            self._gpu_guard(job)
            if not self._launchable(job):  # cancelled while the guard waited
                return
        self._launch_and_monitor(job)

    def _launchable(self, job: Job) -> bool:
        with self._lock:
            return job.state == STATE_QUEUED and not job.stop_requested

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
        """Seconds since the worker last advanced (``_worker_heartbeat``)."""
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
            # stop() finalizes a queued job on the spot; it may have run while
            # this one was spawning, and that verdict stands.
            cancelled = job.state != STATE_QUEUED
            if not cancelled:
                job.state = STATE_RUNNING
                job.started_at = time.time()
                job.pid = popen.pid
                job.create_time = proc.create_time(popen.pid)
                job.accelerate_launched = "accelerate.commands.accelerate_cli" in cmd
                job.persist()
                self._popens[job.id] = popen
        if cancelled:
            proc.kill_tree(popen.pid)
            self._reap(popen)
            return
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
            stalled = self._watchdog.check(job)
            if stalled is not None:
                logger.warning("job %s killed by stall watchdog: %s", job.id, stalled)
                self._kill_job_tree(job)
                # Finalize now so _finalize_from_exit no-ops and the stall
                # diagnostic isn't overwritten by the SIGKILL classification.
                self._finalize(job, STATE_ERROR, error=stalled)
                break
            time.sleep(_POLL_INTERVAL)
        if popen is not None:
            self._reap(popen)
        self._popens.pop(job.id, None)
        self._finalize_from_exit(job, popen)

    @staticmethod
    def _reap(popen) -> None:
        """Wait on our own child so it doesn't linger as a zombie."""
        try:
            popen.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _proc_running(job: Job, popen) -> bool:
        if popen is not None:
            return popen.poll() is None
        return proc.is_alive(job.pid, job.create_time)

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
        # A command job can run a training loop and then more work (train →
        # eval chains): its nonzero exit outranks that loop's run_end "ok".
        ok_but_failed = job.kind == "command" and rc not in (0, None)
        if ev and ev.get("ev") == "run_end" and not ok_but_failed:
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
        self._watchdog.forget(job.id)
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
            # Auto-chain: chained_job_id persists in the same write that flips
            # the state to `done`.
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

    def _gpu_guard(self, job: Job) -> None:
        free = gpu_guard.wait_for_free_gpu(
            job, jobs=self.list_jobs, on_wait=self._broadcast
        )
        if not free:
            with self._lock:
                if job.state == STATE_QUEUED:
                    job.status_detail = "launched despite busy GPU"

    def _kill_job_tree(self, job: Job) -> None:
        if job.pid is None:
            return
        # A SIGSTOP'd tree ignores SIGTERM until resumed — thaw first so the
        # graceful terminate lands instead of waiting out the kill grace.
        if job.state == STATE_PAUSED:
            proc.resume_tree(job.pid)
        proc.kill_tree(job.pid)

    def _build_cmd(self, job: Job) -> tuple[list[str], dict]:
        return launch.build_cmd(job)

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
        # Frozen jobs count: stop, shutdown and health must all see them.
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
