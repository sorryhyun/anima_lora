"""GUI-side bridge to the local training daemon (``scripts/daemon``).

Phase 2 turns the GUI into a daemon *client*: the Train button submits a job to
the daemon (so training survives the GUI closing) and the tab then *observes*
that job by polling the per-job files the daemon already writes to local disk —
``job.json`` for state, ``progress.jsonl`` for the bar, ``stdout.log`` for the
log. Everything is poll-driven off the tab's existing ``QTimer``; there is
deliberately **no background thread / SSE consumer**, because the daemon is
localhost-only (a non-goal forbids remote) so the files are right there to read.

This keeps the heavy ``library.*`` / torch imports out of the GUI: the daemon
client is pure ``urllib`` and ``config`` is pure ``pathlib``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from scripts.daemon import client as _client
from scripts.daemon import config as _cfg
from scripts.daemon.jobs import STATE_ERROR, STATE_STOPPED, TERMINAL_STATES


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
    """Bring the daemon up at GUI launch — idempotent, never blocks startup.

    ``ensure_daemon`` is a no-op when ``/health`` already answers, so this is
    safe to call unconditionally (a daemon left running by the CLI / a previous
    GUI session is reused, not duplicated). Any startup failure is swallowed and
    reported via the return value: the GUI must always open, and the Train
    button's own ``ensure_daemon`` will surface a real error if it's still down.
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
) -> dict:
    """Auto-start the daemon if needed and enqueue a training job.

    Mirrors what ``tasks.py lora-gui <variant>`` would have launched inline:
    ``method`` is the gui-methods variant stem and ``methods_subdir`` is
    ``"gui-methods"``. Returns the daemon's ``{job_id, state}`` response.
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
    )


def submit_command(
    *,
    label: str,
    argv: list[str],
    extra_env: Optional[dict] = None,
    chain_train: Optional[dict] = None,
    config_snapshot: Optional[dict] = None,
    config_file: Optional[str] = None,
) -> dict:
    """Auto-start the daemon if needed and enqueue a plain task job.

    Mirrors what ``python tasks.py <target>`` would have launched inline (e.g.
    preprocess / mask), but runs it through the daemon's serial queue so it
    survives the GUI closing and can't fight a training run for the GPU.

    ``chain_train`` (``{method, preset, methods_subdir}``) makes the daemon
    enqueue that training job itself once this one finishes successfully — the
    "preprocess → train" auto-chain then completes even if the GUI closes
    mid-way. Returns the daemon's ``{job_id, state}`` response.
    """
    cl = ensure_daemon()
    return cl.submit_command(
        label=label,
        argv=list(argv),
        extra_env=extra_env or {},
        chain_train=chain_train or None,
        config_snapshot=config_snapshot or None,
        config_file=config_file,
    )


def stop_job(job_id: str) -> dict:
    """Abort a running/queued job (daemon stays up, advances the queue)."""
    return _client.DaemonClient().stop(job_id)


def list_jobs() -> list:
    """Return all daemon jobs for this checkout, starting the daemon if needed."""
    return ensure_daemon(timeout=20.0).list_jobs()


def list_jobs_passive() -> list:
    """List jobs only if a root-matching daemon is already up.

    Unlike :func:`list_jobs`, this never spawns a daemon and never blocks waiting
    for one to boot — it is for passive monitors (the Queue tab) that poll on a
    timer and must not start a daemon merely by being viewed, nor freeze the UI
    thread on a slow/failed daemon launch. Raises :class:`DaemonUnavailable` when
    no matching daemon answers ``/health``.
    """
    if _current_health() is None:
        raise DaemonUnavailable("no training daemon is running for this checkout")
    return _client.DaemonClient().list_jobs()


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
    """Read the persisted job state straight from ``job.json``.

    Cheaper and hang-proof vs. an HTTP round-trip every poll tick: the daemon
    writes ``job.json`` atomically on every state transition, so a local read is
    always a complete, current record. Returns ``None`` if the file isn't there
    yet (job dir not created) or is mid-rewrite.
    """
    try:
        data = json.loads(
            (_cfg.job_dir(job_id) / "job.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return data.get("state")


def read_job_kind(job_id: str) -> str:
    """Job kind ('train' | 'command') from ``job.json``.

    Lets the two daemon-observing tabs (ConfigTab, PreprocessingTab) share the
    daemon's single active job without each other's job: each only re-attaches
    to jobs of its own kind. Defaults to 'train' for a missing/legacy record so
    pre-existing checkpoints still re-attach to the training tab.
    """
    try:
        data = json.loads(
            (_cfg.job_dir(job_id) / "job.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return "train"
    return data.get("kind") or "train"


def read_job_label(job_id: str) -> Optional[str]:
    """Display label of a command job (its ``method`` field doubles as the
    label — see ``scripts/daemon/jobs.Job``). Lets a tab re-claim *its own*
    command job (e.g. ``exp-spd``) on GUI reopen without grabbing another tab's.
    ``None`` for a missing/unreadable record."""
    data = _read_job_record(job_id)
    return data.get("method") if data else None


def _read_job_record(job_id: str) -> Optional[dict]:
    try:
        return json.loads(
            (_cfg.job_dir(job_id) / "job.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def read_job_chain_variant(job_id: str) -> Optional[str]:
    """The training variant a command job is the auto-chain preprocess for.

    The ConfigTab's Train button submits its auto-chain preprocess as a
    ``command`` job carrying a ``chain_train`` spec (the daemon uses it to
    enqueue the follow-on training; persisted in ``job.json``). The variant is
    ``chain_train.method``. This marker is how the ConfigTab re-claims *its own*
    preprocess on GUI reopen — distinguishing it from a standalone
    preprocess/mask the PreprocessingTab owns — to keep the bar live and Train
    blocked. Returns ``None`` for a job that isn't an auto-chain step.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    return (data.get("chain_train") or {}).get("method")


def read_job_chained_id(job_id: str) -> Optional[str]:
    """The follow-on training job id the daemon spawned for ``job_id``, if any.

    Set by the daemon when an auto-chain preprocess finishes successfully (see
    ``manager._finalize``). Lets the ConfigTab hop straight from observing the
    preprocess to observing the training the daemon just enqueued — instead of
    launching training itself (which would double-submit). ``None`` until/unless
    the chain fired.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    return data.get("chained_job_id")


def read_job_error(job_id: str) -> Optional[str]:
    """The daemon's terminal-state diagnosis from ``job.json`` (None if clean).

    The manager always records *why* a job ended in ``error``/``stopped`` —
    e.g. ``"process exited (code=137): killed (SIGKILL) — almost always out of
    memory…"`` for a silent OOM-killer death that left no traceback. The GUI
    reads it here so the finish banner can show the reason, not just the state.
    """
    data = _read_job_record(job_id)
    if not data:
        return None
    return data.get("error") or data.get("status_detail")


def is_terminal(state: Optional[str]) -> bool:
    return state in TERMINAL_STATES


# Exception markers scanned (tail-first) when a job ends in error. The real
# traceback is streamed live but scrolls far above the finish banner; on
# failure we re-surface its salient line right next to the banner so the cause
# sits where the user is already looking. High-value (actionable) patterns are
# preferred over a generic ``SomethingError:`` so an accelerate-launch
# CalledProcessError wrapper doesn't mask the child's real OOM/missing-file.
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
    newlines. Shared by ConfigTab and PreprocessingTab so they stay in sync.

    On a clean ``done`` it's just the plain banner; on ``error`` it carries the
    daemon's diagnosis (``job.error``) and, when distinct, the salient
    traceback line so the user doesn't have to scroll up for the cause.
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
    """Tail a growing text file by byte offset — thread-free, poll-driven.

    The daemon captures the training subprocess's stdout+stderr to
    ``stdout.log``; this reads whatever's been appended since the last call. A
    fresh :meth:`watch` (pos 0) replays the whole file, which is what re-attach
    relies on to repopulate the log widget after a GUI restart.
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
