"""GUI-side bridge to the local training daemon (``anima_daemon``).

The Train button submits a job to the daemon (so training survives the GUI
closing) and the tab then *observes* that job by polling the per-job files the
daemon writes to local disk — ``job.json`` for state, ``progress.jsonl`` for the
bar, ``stdout.log`` for the log. Everything is poll-driven off the tab's
``QTimer``, with no background thread or SSE consumer.

The daemon client is pure ``urllib`` and ``config`` is pure ``pathlib``, so no
``library.*`` / torch import reaches the GUI.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from anima_daemon import client as _client
from anima_daemon import config as _cfg
from anima_daemon.record import (
    STATE_DONE,
    STATE_ERROR,
    STATE_STOPPED,
    TERMINAL_STATES,
    read_record,
)


class DaemonUnavailable(RuntimeError):
    """No root-matching daemon is currently running.

    Raised by the passive (non-spawning) query path so a monitor can show an
    "unavailable" state instead of booting a daemon as a side effect.
    """


def _current_health() -> Optional[dict]:
    health = _client.DaemonClient().health()
    if health and _client.daemon_matches_root(health, _cfg.ROOT):
        return health
    return None


def ensure_daemon(*, timeout: float = 60.0):
    """Return a daemon client for this checkout, never a sibling checkout."""
    return _client.ensure_daemon(timeout=timeout, expected_root=_cfg.ROOT)


def is_running() -> bool:
    return _current_health() is not None


def ensure_daemon_quietly(*, timeout: float = 20.0) -> bool:
    """Bring the daemon up at GUI launch; idempotent.

    A startup failure is swallowed and reported via the return value: the GUI
    must always open, and the Train button's own ``ensure_daemon`` surfaces a
    real error if it's still down.
    """
    try:
        ensure_daemon(timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 — launch must never fail on a daemon hiccup
        return False


def submit_training(
    *,
    method: str,
    preset: str,
    methods_subdir: Optional[str],
    config_snapshot: Optional[dict] = None,
    config_file: Optional[str] = None,
    overrides: Optional[dict] = None,
    extra: Optional[list[str]] = None,
    start: Optional[bool] = None,
) -> dict:
    """Auto-start the daemon if needed and enqueue a training job.

    Equivalent to ``tasks.py lora-gui <variant>``: ``method`` is the gui-methods
    variant stem and ``methods_subdir`` is ``"gui-methods"``. ``start`` controls the queue gate: ``True`` (the main
    Train button) runs it now, ``False`` (the queue dropdown) holds it until
    "Start Queue". Returns the daemon's ``{job_id, state}`` response.
    """
    cl = ensure_daemon()
    return cl.submit(
        method=method,
        preset=preset,
        methods_subdir=methods_subdir,
        config_snapshot=config_snapshot or None,
        config_file=config_file,
        overrides=overrides or {},
        extra=extra or [],
        start=start,
    )


def submit_command(
    *,
    label: str,
    argv: list[str],
    extra_env: Optional[dict] = None,
    chain_train: Optional[dict] = None,
    config_snapshot: Optional[dict] = None,
    config_file: Optional[str] = None,
    start: Optional[bool] = None,
) -> dict:
    """Auto-start the daemon if needed and enqueue a plain task job.

    Equivalent to ``python tasks.py <target>`` (e.g. preprocess / mask), but
    runs it through the daemon's serial queue so it survives the GUI closing and
    can't fight a training run for the GPU.

    ``chain_train`` (``{method, preset, methods_subdir}``) makes the daemon
    enqueue that training job once this one succeeds, even if the GUI closes
    mid-way. ``start`` controls the queue gate: ``True`` (a main Run action)
    runs it now, ``False`` (the "add to queue" dropdown) holds it until "Start
    Queue". Returns the daemon's ``{job_id, state}`` response.
    """
    cl = ensure_daemon()
    return cl.submit_command(
        label=label,
        argv=list(argv),
        extra_env=extra_env or {},
        chain_train=chain_train or None,
        config_snapshot=config_snapshot or None,
        config_file=config_file,
        start=start,
    )


def stop_job(job_id: str) -> dict:
    """Abort a running/queued job (daemon stays up, advances the queue)."""
    return _client.DaemonClient().stop(job_id)


def start_queue() -> None:
    """Resume a paused queue (the Queue tab's "Start Queue" button)."""
    _client.DaemonClient().start_queue()


def pause_queue() -> None:
    """Hold the queue so newly-added jobs wait for "Start Queue"."""
    _client.DaemonClient().pause_queue()


def list_jobs() -> list:
    """Return all daemon jobs for this checkout, starting the daemon if needed."""
    return ensure_daemon(timeout=20.0).list_jobs()


def list_jobs_passive() -> list:
    """List jobs only if a root-matching daemon is already up.

    Unlike :func:`list_jobs`, this never spawns a daemon or waits for one to
    boot, so a monitor polling on a timer (the Queue tab) neither starts a daemon
    nor freezes the UI thread. Raises :class:`DaemonUnavailable` when no matching
    daemon answers ``/health``.
    """
    if _current_health() is None:
        raise DaemonUnavailable("no training daemon is running for this checkout")
    return _client.DaemonClient().list_jobs()


def queue_snapshot_passive() -> tuple[list, bool]:
    """``(jobs, paused)`` for the Queue tab — passive, never spawns a daemon.

    Raises :class:`DaemonUnavailable` when no matching daemon answers ``/health``.
    """
    health = _current_health()
    if health is None:
        raise DaemonUnavailable("no training daemon is running for this checkout")
    jobs = _client.DaemonClient().list_jobs()
    return jobs, bool(health.get("paused"))


def active_job_id() -> Optional[str]:
    """The daemon's currently-running job id, or ``None`` (daemon down/idle).

    Used on tab construction to re-attach the UI to a job that's still running
    from a previous GUI session (or that the ComfyUI node / CLI submitted).
    """
    health = _current_health()
    return health.get("active_job") if health else None


def progress_path(job_id: str) -> str:
    return str(_cfg.job_dir(job_id) / "progress.jsonl")


def stdout_path(job_id: str) -> str:
    return str(_cfg.job_dir(job_id) / "stdout.log")


def read_job_state(job_id: str) -> Optional[str]:
    """Read the persisted job state straight from ``job.json``, which the daemon
    writes atomically on every transition. ``None`` if the file isn't there yet
    (job dir not created) or is unreadable.
    """
    data = _read_job_record(job_id)
    return data.get("state") if data else None


def read_job_kind(job_id: str) -> str:
    """Job kind ('train' | 'command') from ``job.json``.

    ConfigTab and PreprocessingTab each re-attach only to jobs of their own
    kind. Defaults to 'train' for a missing/legacy record so pre-existing
    checkpoints still re-attach to the training tab.
    """
    data = _read_job_record(job_id)
    return (data.get("kind") if data else None) or "train"


def read_job_label(job_id: str) -> Optional[str]:
    """Display label of a command job (its ``method`` field doubles as the
    label — see ``anima_daemon/jobs.Job``). Lets a tab re-claim *its own*
    command job (e.g. ``turbo``) on GUI reopen without grabbing another tab's.
    ``None`` for a missing/unreadable record."""
    data = _read_job_record(job_id)
    return data.get("method") if data else None


def read_job_argv(job_id: str) -> list[str]:
    """A command job's child argv (``[]`` for a train job or a missing record).
    Lets a line panel claim a sidecar job the CLI submitted under another label."""
    data = _read_job_record(job_id)
    return list(data.get("argv") or []) if data else []


def _read_job_record(job_id: str) -> Optional[dict]:
    return read_record(_cfg.job_dir(job_id))


def read_job_started_at(job_id: str) -> Optional[float]:
    """Wall-clock epoch when the daemon began running ``job_id`` (falling back to
    its submit time, then ``None`` for a missing/legacy record).

    Floor for the training-sample gallery: sample PNGs accumulate under
    ``<output_dir>/sample`` and are never deleted, so only those written at or
    after this time belong to the current run.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    started = data.get("started_at")
    if started is None:
        started = data.get("submitted_at")
    return started if isinstance(started, (int, float)) else None


def read_job_chain_variant(job_id: str) -> Optional[str]:
    """The training variant a command job is the auto-chain preprocess for
    (``chain_train.method``), or ``None`` for a job that isn't one.

    ConfigTab re-claims its own preprocess by this marker on GUI reopen, to keep
    the bar live and Train blocked; a standalone preprocess/mask belongs to
    PreprocessingTab.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    return (data.get("chain_train") or {}).get("method")


def read_job_chained_id(job_id: str) -> Optional[str]:
    """The follow-on training job id the daemon spawned for ``job_id``, or
    ``None`` until the chain fires (``manager._finalize``).

    ConfigTab hops to this job instead of launching training itself, which would
    double-submit.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    return data.get("chained_job_id")


def read_job_error(job_id: str) -> Optional[str]:
    """The daemon's terminal-state diagnosis from ``job.json`` (None if clean),
    for the finish banner.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    return data.get("error") or data.get("status_detail")


def is_terminal(state: Optional[str]) -> bool:
    return state in TERMINAL_STATES


def is_success(state: Optional[str]) -> bool:
    """True only on a clean ``done`` (not stopped / errored)."""
    return state == STATE_DONE


# Exception markers scanned (tail-first) when a job ends in error, re-surfaced next to the
# finish banner. High-value (actionable) patterns beat a generic ``SomethingError:`` so an
# accelerate-launch CalledProcessError wrapper doesn't mask the child's real OOM/missing-file.
_HIGH_VALUE_RE = re.compile(
    r"(torch\.cuda\.OutOfMemoryError|CUDA out of memory|CUDA error:?|"
    r"FileNotFoundError|ModuleNotFoundError|ImportError|"
    r"KeyError|AssertionError|NotImplementedError)[^\n]*"
)
_GENERIC_ERR_RE = re.compile(r"^\s*[\w.]*(?:Error|Exception):[^\n]*", re.MULTILINE)


def _tail_text(path: str, *, max_bytes: int = 65536) -> str:
    """Last ``max_bytes`` of a (possibly huge) log, decoded leniently."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def extract_error_summary(job_id: str) -> Optional[str]:
    """Best-guess cause line from the tail of ``stdout.log`` (None if unfound).

    Two-pass over the log tail: prefer the last *actionable* exception line
    (OOM, missing file/module, CUDA error); else fall back to the last generic
    ``…Error:``/``…Exception:`` line. Returns a single trimmed line.
    """
    tail_txt = _tail_text(stdout_path(job_id))
    if not tail_txt:
        return None
    high = _HIGH_VALUE_RE.findall(tail_txt)
    if high:
        # findall on a single-group regex yields the matched prefixes; re-find
        # the full last match for the complete message line.
        last = list(_HIGH_VALUE_RE.finditer(tail_txt))[-1].group(0)
        return last.strip()
    generic = _GENERIC_ERR_RE.findall(tail_txt)
    if generic:
        return generic[-1].strip()
    return None


def format_finish_banner(job_id: str, state: Optional[str]) -> str:
    """The GUI finish banner: state line + the daemon's error reason + a
    best-guess cause line scraped from stdout.log. Multi-line, no surrounding
    newlines. Shared by ConfigTab and PreprocessingTab.
    """
    from gui.i18n import t

    st = state or "ended"
    err = read_job_error(job_id) if state in (STATE_ERROR, STATE_STOPPED) else None
    if err:
        lines = [t("daemon_job_failed", job_id=job_id, state=st, error=err)]
    else:
        lines = [t("daemon_job_finished", job_id=job_id, state=st)]
    if state == STATE_ERROR:
        summary = extract_error_summary(job_id)
        if summary and (not err or summary not in err):
            lines.append(t("daemon_error_cause", summary=summary))
    return "\n".join(lines)


class FileTailer:
    """Tail a growing text file by byte offset, poll-driven.

    A fresh :meth:`watch` (pos 0) replays the whole file, which re-attach relies
    on to repopulate the log widget after a GUI restart.
    """

    def __init__(self) -> None:
        self._path: Optional[str] = None
        self._pos = 0

    def watch(self, path: Optional[str]) -> None:
        self._path = path
        self._pos = 0

    def reset(self) -> None:
        self.watch(None)

    def read_new(self) -> str:
        """Return text appended since the last read (``""`` if none/absent)."""
        if not self._path or not os.path.exists(self._path):
            return ""
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
        except OSError:
            return ""
        return chunk
