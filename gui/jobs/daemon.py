"""GUI-side bridge to the local training daemon (``anima_daemon``).

Jobs are submitted to the daemon (so they survive the GUI closing) and observed
by polling the per-job files it writes: ``job.json`` for state,
``progress.jsonl`` for the bar, ``stdout.log`` for the log. The client is pure
``urllib``/``pathlib``, so no torch import reaches the GUI.
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
    """No root-matching daemon is running (raised by the passive queries)."""


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

    A failure only returns False: the GUI must always open, and a later
    submit's ``ensure_daemon`` surfaces the real error.
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

    Equivalent to ``tasks.py lora-gui <variant>``. ``start=True`` runs it now,
    ``False`` holds it until "Start Queue". Returns ``{job_id, state}``.
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

    Like ``python tasks.py <target>``, but on the daemon's serial queue so it
    never competes with a training run for the GPU. ``chain_train``
    (``{method, preset, methods_subdir}``) has the daemon enqueue that training
    job once this one succeeds, GUI open or not. ``start`` as in
    :func:`submit_training`.
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

    Never spawns or waits for a daemon, so a timer-driven monitor cannot start
    one or freeze the UI thread. Raises :class:`DaemonUnavailable` otherwise.
    """
    if _current_health() is None:
        raise DaemonUnavailable("no training daemon is running for this checkout")
    return _client.DaemonClient().list_jobs()


def queue_snapshot_passive() -> tuple[list, bool]:
    """``(jobs, paused)`` for the Queue tab; passive like :func:`list_jobs_passive`."""
    health = _current_health()
    if health is None:
        raise DaemonUnavailable("no training daemon is running for this checkout")
    jobs = _client.DaemonClient().list_jobs()
    return jobs, bool(health.get("paused"))


def active_job_id() -> Optional[str]:
    """The daemon's running job id, or ``None``. Tabs re-attach to it (from a
    previous GUI session, the ComfyUI node or the CLI)."""
    health = _current_health()
    return health.get("active_job") if health else None


def progress_path(job_id: str) -> str:
    return str(_cfg.job_dir(job_id) / "progress.jsonl")


def stdout_path(job_id: str) -> str:
    return str(_cfg.job_dir(job_id) / "stdout.log")


def read_job_state(job_id: str) -> Optional[str]:
    """Job state from ``job.json`` (written atomically per transition), or
    ``None`` if absent/unreadable."""
    data = _read_job_record(job_id)
    return data.get("state") if data else None


def read_job_kind(job_id: str) -> str:
    """Job kind ('train' | 'command') from ``job.json``.

    ConfigTab and PreprocessingTab re-attach only to their own kind. A
    missing/legacy record reads as 'train'.
    """
    data = _read_job_record(job_id)
    return (data.get("kind") if data else None) or "train"


def read_job_label(job_id: str) -> Optional[str]:
    """A command job's label (its ``method`` field), so a tab re-claims only
    its own command job on reopen. ``None`` for a missing record."""
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
    """Epoch when ``job_id`` started (else its submit time, else ``None``).

    The sample gallery's floor: PNGs under ``<output_dir>/sample`` are never
    deleted, so only those written since belong to this run.
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

    ConfigTab re-claims its own preprocess by this on reopen; a standalone
    preprocess belongs to PreprocessingTab.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    return (data.get("chain_train") or {}).get("method")


def read_job_chained_id(job_id: str) -> Optional[str]:
    """The follow-on training job id the daemon spawned for ``job_id``, or
    ``None`` until the chain fires (``manager._finalize``).

    ConfigTab follows this job; launching training itself would double-submit.
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


# Actionable patterns beat a generic ``…Error:`` so an accelerate-launch CalledProcessError
# wrapper does not mask the child's real OOM / missing file.
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
    """Best-guess cause line from the tail of ``stdout.log``: the last
    actionable exception line, else the last generic one, else None."""
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
    """Finish banner: state line, the daemon's error reason, and a cause line
    scraped from stdout.log."""
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

    A fresh :meth:`watch` replays the whole file; re-attach relies on it.
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
