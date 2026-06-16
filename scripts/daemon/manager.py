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
    STATE_DONE,
    STATE_ERROR,
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
        # Queue run gate: set → the worker launches queued jobs as the GPU frees
        # (the historical always-on behavior); cleared → the queue is *paused*,
        # so dequeued jobs are held (still `queued`) until `resume()`. A running
        # job is never interrupted by a pause — only the *next* launch waits.
        # Default set so callers that don't opt into hold-then-start (CLI,
        # ComfyUI node) keep running immediately.
        self._run_gate = threading.Event()
        self._run_gate.set()
        self._worker = threading.Thread(
            target=self._run, name="anima-job-worker", daemon=True
        )

    # ----- lifecycle -----

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

    # ----- submission / query -----

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
    ) -> Job:
        job = Job(
            id=new_job_id(),
            method=method,
            preset=preset,
            methods_subdir=methods_subdir,
            overrides=dict(overrides or {}),
            extra=list(extra or []),
            from_chain=from_chain,
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
        even if the GUI that started it has since closed."""
        job = Job(
            id=new_job_id(),
            method=label,
            preset="",
            kind="command",
            argv=list(argv or []),
            extra_env=dict(extra_env or {}),
            chain_train=dict(chain_train) if chain_train else None,
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
        #   False → pause *before* the job is visible to the worker (hold it);
        #   True  → enqueue, then resume (run now — flushes any held backlog);
        #   None  → leave the gate as-is (legacy: runs if not currently paused).
        if start is False:
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

    # ----- queue run gate (pause / start) -----

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

    # ----- stop (job-scoped) -----

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
                # Finalize the queued job *now* (under the lock — _finalize takes
                # the RLock reentrantly) so its cancellation is visible to clients
                # immediately, instead of whenever the worker happens to dequeue
                # it. While another job is running the worker is blocked monitoring
                # it and would never reach this id, so the old lazy path left a
                # stopped-but-still-"queued" entry the UI couldn't clear. The
                # worker skips any dequeued id whose state isn't QUEUED, so the
                # stale FIFO entry is harmless.
                self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
                return job
            job.persist()
        if state == STATE_RUNNING:
            self._kill_job_tree(job)
        return job

    # ----- worker -----

    def _run(self) -> None:
        # Drain re-attached orphans before touching the queue so the serial
        # GPU invariant holds across a daemon restart.
        for job_id in self._adopt:
            job = self.get(job_id)
            if job is not None:
                self._monitor(job, popen=None)
        while True:
            job_id = self._queue.get()
            if job_id == _SENTINEL:
                break
            with self._lock:
                if self._stopping:
                    break
                job = self._jobs.get(job_id)
            if job is None or job.state != STATE_QUEUED:
                continue
            if job.stop_requested:
                self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
                continue
            # Hold here while the queue is paused (the GUI's "Start Queue" button
            # resumes it). Re-validate after waking: the job may have been
            # cancelled while held, or the daemon may be shutting down.
            if not self._await_run_gate(job):
                continue
            with self._lock:
                if job.state != STATE_QUEUED or job.stop_requested:
                    continue
            # Auto-chained train steps skip the guard: the daemon just ran the
            # preceding preprocess on this same serial queue, so the only VRAM
            # in flight is that step's still-releasing allocation, which the
            # guard would needlessly wait on. Standalone jobs still guard.
            if not job.from_chain:
                self._gpu_guard(job)
            self._launch_and_monitor(job)

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
        cmd, env = self._build_cmd(job)
        d = config.job_dir(job.id)
        try:
            popen = proc.spawn_detached(
                cmd,
                cwd=config.ROOT,
                stdout_path=d / "stdout.log",
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            self._finalize(job, STATE_ERROR, error=f"spawn failed: {exc}")
            return
        with self._lock:
            job.state = STATE_RUNNING
            job.started_at = time.time()
            job.pid = popen.pid
            job.create_time = proc.create_time(popen.pid)
            job.persist()
            self._popens[job.id] = popen
        self._broadcast({"ev": "started", "job_id": job.id, "pid": job.pid})
        self._monitor(job, popen=popen)

    def _monitor(self, job: Job, *, popen) -> None:
        """Block until the job process exits, then finalize. Works for both a
        process we spawned (``popen`` reaps the child) and an adopted orphan
        (``popen is None`` → psutil liveness)."""
        while self._proc_running(job, popen):
            if self._kill_on_shutdown:
                self._kill_job_tree(job)
                break
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
        opted in via ANIMA_DAEMON_JOB_STALL_TIMEOUT.
        """
        timeout = (
            config.CMD_STALL_TIMEOUT
            if job.kind == "command"
            else config.JOB_STALL_TIMEOUT
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
            return None
        where = JobManager._last_output_line(job)
        detail = f" last output: {where!r}" if where else " (no output captured)"
        return (
            f"stalled: no output for {int(idle)}s (limit {int(timeout)}s); daemon "
            f"killed the job so the queue can advance.{detail}"
        )

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
            # No run_end and a nonzero/unknown exit — the trainer died before it
            # could write its terminal event. Classify the code into something
            # actionable: signal deaths (OOM-killer SIGKILL, CUDA SIGABRT,
            # segfault) leave NO Python traceback in stdout.log, so the exit
            # code is the only signal the user gets.
            self._finalize(job, STATE_ERROR, error=_classify_exit(rc))

    def _finalize(
        self,
        job: Job,
        state: str,
        *,
        error: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        with self._lock:
            job.state = state
            job.ended_at = time.time()
            if error:
                job.error = error
            if detail:
                job.status_detail = detail
            job.ckpt_path = tail.last_ckpt_path(job.progress_path)
            # Daemon-managed auto-chain: a successfully finished command job that
            # carries a chain_train spec enqueues its follow-on training job
            # right here, so the chain survives the GUI closing. Recorded on this
            # job (chained_job_id) and persisted in the same write that flips us
            # to `done`, so a client observing this job sees both atomically.
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
                )
                job.chained_job_id = follow.id
                logger.info(
                    "auto-chain: job %s done → enqueued training %s",
                    job.id,
                    follow.id,
                )
            job.persist()
        self._broadcast({"ev": "ended", "job_id": job.id, "state": state})

    # ----- gpu guard -----

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
        reap VRAM leaked by our *own* dead jobs, matched by pid (a stranger's pid
        never matches a job, so the polluted holder list is harmless on that
        path). If we can't probe memory at all we assume free rather than
        deadlock the queue. Tunable via ANIMA_DAEMON_GPU_{BUSY_FRAC,RETRIES,DELAY}.
        """
        for attempt in range(retries):
            # Reap leftovers from our own (now-terminal/dead) jobs. Safe even
            # when gpu_pids() is polluted: only pids that match a known job act.
            holders = gpu.gpu_pids() or set()
            with self._lock:
                known = {j.pid: j for j in self._jobs.values() if j.pid in holders}
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
        if job.pid is not None:
            proc.kill_tree(job.pid)

    # ----- command building -----

    def _build_cmd(self, job: Job) -> tuple[list[str], dict]:
        from .client import venv_python

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # tqdm redraws ride "\r"; at the default 0.1s cadence a cached-dataset
        # scan writes thousands of bar updates into stdout.log, drowning the
        # lines a reader actually tails for (warnings, tracebacks). One redraw
        # per 10s is plenty: the GUI's TqdmProgressTracker only parses the
        # latest line, and training progress has its own progress.jsonl stream.
        env.setdefault("TQDM_MININTERVAL", "10")

        # Command jobs (preprocess / mask) are a plain task invocation. Launch
        # under pythonw.exe (windowless): a uv-venv python.exe is a trampoline
        # that re-execs the real interpreter, and CREATE_NO_WINDOW doesn't
        # survive that re-exec — so a python.exe child pops a console window
        # that, when closed (or torn down with the GUI), kills the job with
        # STATUS_CONTROL_C_EXIT (0xC000013A). pythonw.exe never allocates a
        # console; the tqdm progress the GUI tails from stdout.log still lands
        # because spawn_detached redirects the child's stdout/stderr to that
        # file (a real handle, not an inherited console). No --progress_jsonl
        # injection — these emit tqdm to stdout and the monitor finalizes them
        # on exit code (no run_end event).
        if job.kind == "command":
            env.update(job.extra_env or {})
            return [venv_python(windowless=True), *job.argv], env

        # Imported lazily so loading the daemon package never drags in the task
        # runner's transitive imports until a job actually launches.
        from scripts.tasks._common import build_launch_cmd, build_method_args

        overrides = dict(job.overrides or {})
        extra = list(job.extra or [])
        # Translate dict overrides into --key value pairs unless already present.
        # NOTE: most train.py bool flags are `store_true`, so a True override
        # emits `--flag` but a False one can only be expressed by *omitting* it —
        # train.py then keeps whatever the base→preset→method chain set. That's
        # fine for the ComfyUI node's overrides (the only preset that turns these
        # bools on is low_vram, which it never pairs with a False override), but
        # a caller can't force a preset-on flag back off through this path.
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
        # Windowless interpreter for the same reason as command jobs above: the
        # train.py worker (and, under ANIMA_ACCELERATE_LAUNCH, the accelerate
        # launcher parent + the workers it re-spawns via sys.executable) all run
        # as pythonw.exe, so nothing pops a closable console that would
        # CTRL_CLOSE the run.
        cmd = build_launch_cmd(*args, python_exe=venv_python(windowless=True))
        return cmd, env

    # ----- reconciliation (boot) -----

    def _reconcile(self) -> None:
        self._jobs = load_all()
        for job in self._jobs.values():
            if job.state == STATE_RUNNING:
                if proc.is_alive(job.pid, job.create_time):
                    logger.info("reconcile: re-attaching live job %s", job.id)
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

    # ----- helpers -----

    def _current_running_locked(self) -> Optional[Job]:
        for job in self._jobs.values():
            if job.state == STATE_RUNNING:
                return job
        return None

    def active_job(self) -> Optional[Job]:
        """The currently-running job, if any (lock-safe public accessor)."""
        with self._lock:
            return self._current_running_locked()

    # ----- pub/sub for SSE -----

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
