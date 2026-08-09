"""The job manager: FIFO serial queue + worker thread + state table.

One worker thread drains a ``queue.Queue`` of job ids. Per job it builds the
same ``accelerate launch … train.py`` command the CLI builds, spawns it
detached (so a console ctrl-C can't reach it), points ``--progress_jsonl`` at
the job dir, then monitors by polling ``(pid, create_time)`` liveness — never
by awaiting a subprocess transport (sidesteps Windows ProactorEventLoop
subprocess bugs). On boot it reconciles ``jobs/`` so it can re-attach a
still-alive orphan or mark a dead one ``orphaned``.

Serial by design (single local GPU): exactly one job runs at a time.
"""

from __future__ import annotations

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
)

logger = logging.getLogger("anima.daemon")

_POLL_INTERVAL = 1.0  # seconds between liveness checks
_SENTINEL = "__stop__"

# Stall watchdog, CPU-activity arm (see _stall_reason / _tree_cpu_advancing).
# Once a job's output has frozen past its budget we sample its process tree's
# CPU time: a *quiet but computing* job (an embed/eval loop between prints) keeps
# burning cycles, while a wedged one (stalled socket, deadlock) burns none. Only
# a frozen-and-idle job is killed — until `budget × _STALL_CPU_GRACE` of silence,
# after which we kill regardless, so a busy-spinning deadlock can't park the
# queue forever. job_id → (wall, cpu_seconds, verdict).
_CPU_SAMPLES: dict[str, tuple[float, float, bool]] = {}
_CPU_SAMPLE_MIN_GAP = 5.0  # seconds between samples (a short gap measures noise)
_CPU_BUSY_FRAC = 0.05  # ≥5% of one core, averaged over the gap → "computing"
_STALL_CPU_GRACE = 8.0  # hard ceiling multiplier on the configured budget

# Signal → user-actionable hint, for a process that died without writing a
# run_end event. POSIX ``Popen.poll()`` reports a signal death as a negative
# number; a shell/launcher layer (``accelerate launch``) relays it as 128+N.
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
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._popens: dict[str, object] = {}  # job_id -> Popen (spawned only)
        self._adopt: list[str] = []  # running orphans to monitor before the queue
        self._subscribers: set["queue.Queue[dict]"] = set()
        self._stopping = False
        self._kill_on_shutdown = False
        # Queue run gate: set → worker launches queued jobs as the GPU frees;
        # cleared → queue paused (dequeued jobs held `queued` until `resume()`,
        # a running job left alone). Default set so non-opt-in callers run now.
        self._run_gate = threading.Event()
        self._run_gate.set()
        # Worker liveness: bumped every loop iteration and every monitor poll.
        # Exposed via /health so a wedged-or-dead worker is observable (the GUI
        # spinner otherwise looks identical to a healthy long-running job).
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
        self._queue.put(_SENTINEL)  # wake the worker so it can exit

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
        """Enqueue a plain ``python <argv>`` task (preprocess / mask).

        Goes through the same serial queue as training so a cache-build and a
        training run can't fight over the single local GPU. ``label`` is the
        display name; ``argv`` is passed straight to the venv interpreter (e.g.
        ``["tasks.py", "preprocess"]``); ``extra_env`` carries the GUI's knobs
        (``CAPTION_SHUFFLE_VARIANTS``, ``RUN_SAM_MASK``, …).

        ``chain_train`` (``{method, preset, methods_subdir}``) makes this an
        auto-chain step: on successful completion the daemon enqueues that
        training job itself (see ``_finalize``), so the chain runs to the end
        even if the GUI that started it has since closed.

        ``stall_timeout`` overrides the 120 s command-job stall budget for this
        job (0 disables it) — the supported way to run a legitimately quiet
        embed/eval loop without a stdout heartbeat (see ``_stall_reason``)."""
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
        # ``start`` controls the run gate atomically with enqueue, so there's no
        # window where a "hold this one" job could slip past the worker:
        #   False → "add to queue". Hold this job for a later Start Queue ONLY
        #           when the queue is otherwise idle (so a user can stage several
        #           jobs before pressing Start). If a job is already running or
        #           queued, the queue is "playing" — leave the gate alone so this
        #           job auto-advances behind the current ones (cassette-tape
        #           behaviour). Pausing a playing queue here was the bug that
        #           stalled auto-advance the moment the running job finished.
        #   True  → enqueue, then resume (run now — flushes any held backlog);
        #   None  → leave the gate as-is (legacy: runs if not currently paused).
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
        self._queue.put(job.id)
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
            if state == STATE_QUEUED:
                # Finalize the queued job *now* (reentrant RLock) so its cancel
                # is visible immediately: the worker is blocked monitoring a
                # running job and won't reach this id, so the old lazy path left
                # a stopped-but-"queued" entry the UI couldn't clear. The worker
                # skips dequeued ids whose state isn't QUEUED → stale FIFO entry
                # is harmless.
                self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
                return job
            job.persist()
        if state in ACTIVE_STATES:
            # Running or paused (frozen): either way tree-kill it. _kill_job_tree
            # thaws a paused tree first so the SIGTERM is actually delivered.
            self._kill_job_tree(job)
        return job

    def pause_job(self, job_id: str) -> Optional[dict]:
        """Freeze a running job's process tree (SIGSTOP), method-agnostically.

        The CUDA context and VRAM stay put; only SM scheduling stops, so resume
        is instant (no reload/recompile, mid-step optimizer state intact). The
        queue does NOT advance — the paused job still owns its slot; ``pause`` is
        "hold my run", not "yield it". Returns ``{job_id, state, error?}``, or
        ``None`` when no such job (the server maps that to 404). Refuses anything
        not ``running`` and refuses a multi-GPU ``accelerate launch`` run (a
        frozen NCCL rank trips the collective heartbeat).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state == STATE_PAUSED:
                return {"job_id": job.id, "state": job.state}  # idempotent
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
        """Thaw a paused job's process tree (SIGCONT) → back to ``running``.

        Returns ``{job_id, state, error?}``, or ``None`` for no such job. A
        no-op error for a job that isn't paused. Wall-clock throughput/ETA
        inside the run blips across the pause window; accepted, not compensated.
        """
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
            job_id = self._queue.get()
            self._worker_heartbeat = time.time()
            if job_id == _SENTINEL:
                break
            with self._lock:
                if self._stopping:
                    break
            # One bad job must NEVER kill the worker thread: a dead worker leaves
            # every later job stuck `queued` forever with no error and no
            # watchdog (the stall watchdog only guards *running* jobs). Catch
            # everything, fail the offending job loudly, and keep draining.
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
        # Hold here while the queue is paused (the GUI's "Start Queue" button
        # resumes it). Re-validate after waking: the job may have been
        # cancelled while held, or the daemon may be shutting down.
        if not self._await_run_gate(job):
            return
        with self._lock:
            if job.state != STATE_QUEUED or job.stop_requested:
                return
        # Auto-chained train steps skip the guard: the daemon just ran the
        # preceding preprocess on this same serial queue, so the only VRAM
        # in flight is that step's still-releasing allocation, which the
        # guard would needlessly wait on. Standalone jobs still guard.
        if not job.from_chain:
            self._gpu_guard(job)
        self._launch_and_monitor(job)

    def _fail_safely(self, job_id: str, error: str) -> None:
        """Finalize a job ERROR without ever propagating — the last line of
        defense so the worker survives even a finalize that itself raises."""
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
        d = config.job_dir(job.id)
        try:
            # _build_cmd runs the full config merge + lazy task-runner import for
            # train jobs; keep it INSIDE the guard so a bad config / import error
            # fails just this job instead of crashing the worker.
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
            # Record whether this went out under the accelerate launcher, so
            # pause can refuse a multi-GPU run without re-deriving it (Phase 2a).
            job.accelerate_launched = "accelerate.commands.accelerate_cli" in cmd
            job.persist()
            self._popens[job.id] = popen
        self._broadcast({"ev": "started", "job_id": job.id, "pid": job.pid})
        self._monitor(job, popen=popen)

    def _monitor(self, job: Job, *, popen) -> None:
        """Block until the job process exits, then finalize. Works for both a
        process we spawned (``popen`` reaps the child) and an adopted orphan
        (``popen is None`` → psutil liveness)."""
        while self._proc_running(job, popen):
            self._worker_heartbeat = time.time()
            if self._kill_on_shutdown:
                self._kill_job_tree(job)
                break
            if job.state == STATE_PAUSED:
                # Frozen tree writes nothing — the stall watchdog would fire and
                # kill it. Keep polling liveness (a `stop` while paused, or the
                # process dying, still exits the loop) but skip the watchdog.
                time.sleep(_POLL_INTERVAL)
                continue
            stalled = self._stall_reason(job)
            if stalled is not None:
                logger.warning("job %s killed by stall watchdog: %s", job.id, stalled)
                self._kill_job_tree(job)
                # Finalize now so the post-loop _finalize_from_exit (which would
                # otherwise classify the SIGKILL exit) sees a terminal state and
                # no-ops, preserving the actionable stall diagnostic.
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
        """If the running job has produced no output for longer than the
        configured stall timeout, return an actionable error naming where it
        wedged; otherwise ``None``.

        Liveness is the most recent mtime of stdout.log *or* progress.jsonl, so
        both a preprocess job (tqdm-to-stdout, no progress.jsonl) and a training
        job (progress.jsonl) are covered, and any phase that still flushes the
        occasional line — including a slow download's tqdm bar — counts as
        alive. A truly wedged process (stalled socket with no bytes, a
        symlink-cycle walk, a deadlock) writes nothing, so its files freeze and
        the watchdog fires. ``TQDM_MININTERVAL`` (10s) keeps even a busy bar
        well under either budget.

        The budget is per *kind*: a command (preprocess / mask) job is tight
        (it never legitimately goes quiet for more than a model-load), while a
        train job is unwatched by default (budget 0 → skipped here) because its
        silent first-step torch.compile trace would false-positive; it can be
        opted in via ANIMA_DAEMON_JOB_STALL_TIMEOUT. A submitter that knows its
        loop goes quiet sets ``stall_timeout`` on the job (0 → unwatched), which
        wins over both defaults — the supported alternative to hand-rolling a
        stdout heartbeat in every long-quiet bench script.

        Frozen output alone is not proof of a wedge: an embed/eval loop is quiet
        *and computing*. So past the budget we also sample the process tree's CPU
        time and only fire once the tree has gone idle too — up to
        ``_STALL_CPU_GRACE ×`` the budget, after which we fire regardless so a
        busy-spinning deadlock can't hold the queue forever.
        """
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
        """True iff the job's process tree has burned CPU since the last sample.

        Two samples ``_CPU_SAMPLE_MIN_GAP`` apart are needed for a meaningful
        rate, so the first look (and any call inside the gap) reuses the previous
        verdict — starting optimistic, since a job we've never sampled is more
        likely mid-compute than wedged. Unreadable CPU times (pid gone, no
        permission) return False so the watchdog falls back to its historical
        output-mtime-only behaviour rather than becoming un-fireable.
        """
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
        tqdm bar's latest redraw is returned rather than an empty fragment) —
        this is the "where did it wedge" hint folded into the stall error."""
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
        # Mirror the OS exit code into the record (Phase 0c) before finalizing,
        # so the persist inside _finalize captures it and run_gpu can exit with
        # it. None for an adopted orphan (psutil liveness gives no code).
        job.returncode = rc
        if job.stop_requested:
            self._finalize(job, STATE_STOPPED)
            return
        if ev and ev.get("ev") == "run_end":
            status = ev.get("status")
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
            # No run_end + nonzero exit: the trainer died before its terminal
            # event. Classify the code — signal deaths (SIGKILL/OOM, CUDA
            # SIGABRT, segfault) leave no traceback, so it's the only signal.
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
            if error:
                job.error = error
            if detail:
                job.status_detail = detail
            job.ckpt_path = tail.last_ckpt_path(job.progress_path)
            # Phase 1a — follow the bench envelope pointer (if the job wrote one)
            # and record its abs path + a {label, metrics} digest on the record.
            self._lift_result(job)
            # Auto-chain: a done command job with a chain_train spec enqueues its
            # follow-on train job here (survives the GUI closing). chained_job_id
            # persists in the same write that flips us to `done` → atomic for a
            # client observing this job.
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
                    # Inherit the originating command's captured env so the
                    # chained train step runs with the same shell settings
                    # (Phase 0b) — the submitter is long gone by now.
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
        """Follow a bench envelope pointer (Phase 1a) into the job record.

        A GPU job whose script called ``bench/_common.write_result`` under a
        daemon spawn dropped ``<job_dir>/result_path.json`` → ``{"path": <abs
        result.json>}``. Read it, then read the envelope for its ``label`` /
        ``metrics`` digest. Best-effort and schema-blind: the envelope contract is
        bench-owned (we lift ``label``/``metrics`` opaquely, never validate them),
        and a missing/corrupt pointer is the common no-envelope case — leave the
        fields None. Absolute paths only, so a ledger can index them as-is.
        """
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

        Busy/free is decided from **total VRAM in use**, not the process list:
        on Windows WDDM every desktop app (dwm, explorer, browser, …) shows up
        as a "compute" process, so gating on process presence stalled the queue
        on a dozen innocent renderers every launch. A real training run holds
        GBs; an idle desktop holds <1 GB — so `used/total < busy_frac` reliably
        means "go". The threshold is deliberately loose (default 0.85): the only
        thing the guard *must* catch is VRAM leaked by our own dead jobs, and
        that is reaped by pid below regardless of the fraction; the fraction only
        guesses whether some *other* process owns the card, so a partially-loaded
        ComfyUI / browser shouldn't trip it. Process enumeration is kept only to
        reap VRAM leaked by our *own* dead jobs, matched by **(pid, create_time)**
        — a bare pid match is not enough: PIDs get recycled (aggressively so on
        Windows), and a stale job record whose number now belongs to a stranger
        would otherwise make us kill that stranger. Issue #83: a 3-day-old job's
        pid had been reused by ``dwm.exe``, which the guard then tried to reap.
        If we can't probe memory at all we assume free rather than
        deadlock the queue. Tunable via ANIMA_DAEMON_GPU_{BUSY_FRAC,RETRIES,DELAY}.
        """
        # A resident inference server (scripts/inference_server.py) holds a warm
        # DiT on the card. Politely ask it to free VRAM before we launch — it
        # stays alive and reloads on its next request. Best-effort; if none is
        # running this is a couple of cheap stat() calls.
        self._evict_resident_inference()

        for attempt in range(retries):
            # Reap leftovers from our own (now-terminal/dead) jobs. Safe even
            # when gpu_pids() is polluted: a holder acts only if it matches a
            # known job on (pid, create_time) — i.e. it really is the process we
            # spawned, still alive. A recycled pid fails the create_time check
            # and is left alone (issue #83).
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
        # Give up waiting — proceed (the OS will OOM us if there genuinely
        # isn't room; we won't kill what we didn't start).
        job.status_detail = "launched despite busy GPU"

    def _kill_job_tree(self, job: Job) -> None:
        if job.pid is None:
            return
        # A SIGSTOP'd tree won't act on the SIGTERM kill_tree sends until it's
        # resumed (only SIGKILL reaches a stopped process) — thaw first so the
        # graceful terminate lands and we don't wait out the full kill grace.
        if job.state == STATE_PAUSED:
            proc.resume_tree(job.pid)
        proc.kill_tree(job.pid)

    def _evict_resident_inference(self) -> None:
        """Ask a resident inference server (if any) to free VRAM before launch.

        Discovery mirrors scripts/inference_server.py's pidfiles (in-repo +
        per-user mirror + $ANIMA_INFERENCE_PIDFILE). Done inline (no import) so
        the daemon stays decoupled from the inference server; every failure is
        swallowed — coexistence is a courtesy, and the server's own idle-TTL
        eventually frees the card anyway.
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
        # tqdm redraws ride "\r"; at 0.1s cadence they drown stdout.log's real
        # lines (warnings/tracebacks). 10s is plenty — the GUI tracker parses
        # only the latest line, and training has its own progress.jsonl.
        env.setdefault("TQDM_MININTERVAL", "10")

        # Layer the submitter's captured env (Phase 0b) over the daemon's boot
        # env, so a job runs with the shell's CUDA_VISIBLE_DEVICES / ANIMA_DIT /
        # HF_TOKEN rather than week-old daemon values. Applied via update (not
        # setdefault): the caller's explicit value must win over an inherited
        # one. A command job's extra_env is applied AFTER this, so it stays the
        # highest-priority layer (daemon-env ← captured_env ← extra_env).
        env.update(job.captured_env or {})

        # Phase 1a — result envelope lift. Advertise the job's identity + dir so a
        # bench script running inside it can drop a `result_path.json` pointer
        # (bench/_common.write_result reads these). Absent → the script runs as a
        # plain inline invocation and writes no pointer (zero coupling). Set for
        # every kind so any GPU job that happens to emit an envelope is lifted.
        env["ANIMA_DAEMON_JOB_ID"] = job.id
        env["ANIMA_DAEMON_JOB_DIR"] = str(job.dir)

        # Command jobs (preprocess / mask): a plain task invocation under
        # pythonw.exe (windowless). A uv-venv python.exe re-execs the real
        # interpreter and CREATE_NO_WINDOW doesn't survive that, so it pops a
        # console whose close kills the job with STATUS_CONTROL_C_EXIT
        # (0xC000013A); pythonw never allocates one (stdout still lands via
        # spawn_detached's file redirect). No --progress_jsonl — these emit tqdm
        # to stdout and finalize on exit code (no run_end event).
        if job.kind == "command":
            env.update(job.extra_env or {})
            return [venv_python(windowless=True), *job.argv], env

        # Imported lazily so loading the daemon package never drags in the task
        # runner's transitive imports until a job actually launches.
        from scripts.tasks._common import build_launch_cmd, build_method_args

        overrides = dict(job.overrides or {})
        extra = list(job.extra or [])
        # Dict overrides → --key value (unless already in extra). NOTE: train.py
        # bools are `store_true`, so a True override emits `--flag` but a False
        # one can only be expressed by omitting it (train.py then keeps the
        # chain's value) — a caller can't force a preset-on flag back off here.
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
        # Point the structured progress stream at the job dir so we always know
        # where it is, regardless of the method's output_name default.
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
        # Windowless interpreter for the same reason as command jobs above, so
        # nothing in the train tree (incl. accelerate-launched workers) pops a
        # closable console that would CTRL_CLOSE the run.
        cmd = build_launch_cmd(*args, python_exe=venv_python(windowless=True))
        return cmd, env

    def _reconcile(self) -> None:
        self._jobs = load_all()
        for job in self._jobs.values():
            if job.state in ACTIVE_STATES:
                if proc.is_alive(job.pid, job.create_time):
                    # A paused job's tree is still SIGSTOP'd on disk (the freeze
                    # outlives the daemon) — re-adopt it in whatever state it was
                    # left; the monitor loop skips the stall watchdog while it's
                    # `paused` and a later resume thaws it.
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
                self._queue.put(job.id)

    def _current_running_locked(self) -> Optional[Job]:
        # The job occupying the worker/GPU slot — running or frozen. A paused job
        # still owns its VRAM and blocks the queue, so `stop`/`shutdown`/health's
        # active_job must all see it.
        for job in self._jobs.values():
            if job.state in ACTIVE_STATES:
                return job
        return None

    def _queue_is_idle_locked(self) -> bool:
        """True when no job is running or waiting to run — the worker is parked
        on an empty queue. Used to decide whether an ``add to queue`` submission
        should hold the gate (idle → stage it) or leave the gate alone (a job is
        already playing → let the new one auto-advance behind it). The
        just-submitted job is not in ``_jobs`` yet when this is consulted."""
        return not any(
            job.state == STATE_QUEUED or job.state in ACTIVE_STATES
            for job in self._jobs.values()
        )

    def active_job(self) -> Optional[Job]:
        """The currently-running job, if any (lock-safe public accessor)."""
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
