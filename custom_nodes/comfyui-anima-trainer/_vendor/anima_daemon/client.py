"""HTTP client for the daemon — used by the CLI commands and the ComfyUI node.

``urllib``-based, with no ``library.*`` / torch imports, so it loads inside
ComfyUI. ``ensure_daemon`` auto-starts a console-detached daemon and waits for
``/health``.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

from . import config, proc

logger = logging.getLogger("anima.daemon")

# Mirrors jobs.TERMINAL_STATES, restated so this module stays importable on its
# own (the ComfyUI node vendors it).
TERMINAL_STATES = frozenset({"done", "error", "stopped"})


def venv_python(*, windowless: bool = False) -> str:
    """Resolve the anima_lora venv interpreter (not the caller's, e.g. ComfyUI's).
    Probes the usual venv layouts under the repo root and its parent, then
    falls back to ``sys.executable``.

    ``windowless=True`` (Windows only) prefers ``pythonw.exe``, which never
    allocates a console. The uv venv ``python.exe`` is a trampoline that
    re-launches the real interpreter, so ``CREATE_NO_WINDOW`` doesn't reliably
    suppress its console, and closing that console kills the process
    (CTRL_CLOSE_EVENT / ``STATUS_CONTROL_C_EXIT``). Every daemon and job spawn
    uses this.
    """
    if sys.platform == "win32":
        exe = "pythonw.exe" if windowless else "python.exe"
        for base in (config.ROOT, config.ROOT.parent):
            cand = base / ".venv" / "Scripts" / exe
            if cand.exists():
                return str(cand)
    else:
        for base in (config.ROOT, config.ROOT.parent):
            cand = base / ".venv" / "bin" / "python"
            if cand.exists():
                return str(cand)
    return sys.executable


def _resolve_port() -> int:
    info = proc.read_pidfile(config.discover_pidfile())
    if info and info.get("port"):
        return int(info["port"])
    return config.DEFAULT_PORT


def _norm_root(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def daemon_matches_root(health: Optional[dict], expected_root: str | Path) -> bool:
    """True iff a daemon health response belongs to ``expected_root``. Falls
    back to the local in-repo pidfile for legacy daemons lacking ``root`` in
    ``/health``; a rootless daemon found only via the per-user global pidfile
    is treated as not matching."""
    if not health:
        return False
    expected = _norm_root(expected_root)
    root = health.get("root")
    if root:
        return _norm_root(root) == expected

    pidfile = config.discover_pidfile()
    info = proc.read_pidfile(pidfile) or {}
    root = info.get("root")
    if root:
        return _norm_root(root) == expected

    try:
        return pidfile.resolve() == config.PIDFILE.resolve()
    except OSError:
        return False


def _root_mismatch_message(health: dict, expected_root: str | Path) -> str:
    actual = health.get("root") or "unknown checkout"
    return (
        "training daemon belongs to a different anima_lora checkout "
        f"({actual}); expected {Path(expected_root).resolve()}"
    )


def daemon_is_stale(health: Optional[dict]) -> bool:
    """True iff a live daemon's boot fingerprint differs from a fresh hash of
    the on-disk source (a daemon without a fingerprint counts as stale). Used by
    ``ensure_daemon`` (restart) and ``python -m anima_daemon status``
    (``stale_code``).
    """
    if not health:
        return False
    running = health.get("fingerprint")
    if not running:
        return True
    return running != config.source_fingerprint()


def _await_down(client: "DaemonClient", timeout: float) -> None:
    """Poll until ``client`` stops answering ``/health`` or ``timeout`` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline and client.health() is not None:
        time.sleep(0.2)


def _has_live_jobs(client: "DaemonClient") -> bool:
    try:
        return any(
            (job.get("state") or "") in {"queued", "running"}
            for job in client.list_jobs()
        )
    except Exception:
        return True


class DaemonClient:
    def __init__(self, port: Optional[int] = None) -> None:
        self.port = port or _resolve_port()

    @property
    def base(self) -> str:
        return f"http://{config.HOST}:{self.port}"

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        timeout: float = 30.0,
    ):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def health(self, *, timeout: float = 3.0) -> Optional[dict]:
        # Fast-fail when nothing is listening: on Windows a connect to a closed
        # port isn't refused for ~2s (SYN retransmit), so a bare urlopen would
        # stall every "is the daemon up?" probe. Bound it at 0.25s.
        try:
            with socket.create_connection((config.HOST, self.port), timeout=0.25):
                pass
        except OSError:
            return None
        try:
            return self._request("GET", "/health", timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def submit(
        self,
        *,
        method: str,
        preset: str = "default",
        methods_subdir: Optional[str] = None,
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
        overrides: Optional[dict] = None,
        extra: Optional[list[str]] = None,
        start: Optional[bool] = None,
        captured_env: Optional[dict] = None,
    ) -> dict:
        # Snapshot the caller's whitelisted env; pass captured_env={} to opt out.
        if captured_env is None:
            captured_env = config.capture_env()
        return self._request(
            "POST",
            "/jobs",
            {
                "method": method,
                "preset": preset,
                "methods_subdir": methods_subdir,
                "config_snapshot": config_snapshot or None,
                "config_file": config_file,
                "overrides": overrides or {},
                "extra": extra or [],
                "start": start,
                "captured_env": captured_env,
            },
        )

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
    ) -> dict:
        """Enqueue a plain ``python <argv>`` job. ``stall_timeout`` overrides
        the command-job stall budget (0 disables it; see README)."""
        if captured_env is None:
            captured_env = config.capture_env()
        return self._request(
            "POST",
            "/jobs",
            {
                "kind": "command",
                "label": label,
                "argv": list(argv),
                "extra_env": extra_env or {},
                "chain_train": chain_train or None,
                "config_snapshot": config_snapshot or None,
                "config_file": config_file,
                "start": start,
                "captured_env": captured_env,
                "stall_timeout": stall_timeout,
            },
        )

    def start_queue(self) -> Optional[dict]:
        """Resume a paused queue — the worker launches queued jobs in order."""
        return self._request("POST", "/queue/start")

    def pause_queue(self) -> Optional[dict]:
        """Hold the queue — queued jobs wait until ``start_queue``."""
        return self._request("POST", "/queue/pause")

    def list_jobs(self) -> list:
        return self._request("GET", "/jobs") or []

    def get(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def job_record(self, job_id: str) -> Optional[dict]:
        """One job record, tolerant of a down/restarting daemon.

        Tries ``GET /jobs/{id}`` and falls back to the on-disk
        ``jobs/<id>/job.json`` the daemon persists on every state change — so a
        reader survives the eager stale-code restart mid-poll. ``None`` only
        when neither source knows the id.
        """
        try:
            rec = self._request("GET", f"/jobs/{job_id}")
        except (urllib.error.URLError, OSError, ValueError):
            rec = None
        if isinstance(rec, dict) and rec.get("id"):
            return rec
        try:
            disk = json.loads(
                (config.job_dir(job_id) / "job.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        return disk if isinstance(disk, dict) else None

    def wait(
        self,
        job_id: str,
        *,
        poll: float = 5.0,
        timeout: Optional[float] = None,
    ) -> dict:
        """Block until ``job_id`` is terminal; return its final record.

        Survives a daemon restart mid-wait (see :meth:`job_record`). Poll
        interval ramps 0.25s -> ``poll``.

        Raises ``LookupError`` if no such job, and ``TimeoutError`` if
        ``timeout`` elapses first.
        """
        deadline = None if timeout is None else time.time() + timeout
        interval = 0.25
        while True:
            rec = self.job_record(job_id)
            if rec is None:
                raise LookupError(f"no such job: {job_id}")
            if (rec.get("state") or "") in TERMINAL_STATES:
                return rec
            now = time.time()
            if deadline is not None and now >= deadline:
                raise TimeoutError(
                    f"job {job_id} still {rec.get('state')} after {timeout:.0f}s"
                )
            nap = interval
            if deadline is not None:
                nap = min(nap, max(0.05, deadline - now))
            time.sleep(nap)
            interval = min(interval * 1.6, max(0.25, poll))

    def stop(self, job_id: Optional[str] = None) -> dict:
        # No job_id → resolve the active job from /health.
        if job_id is None:
            health = self.health() or {}
            job_id = health.get("active_job")
            if not job_id:
                return {"error": "no active job"}
        return self._request("POST", f"/jobs/{job_id}/stop")

    def pause_job(self, job_id: Optional[str] = None) -> dict:
        """Freeze a running job's tree (SIGSTOP). ``None`` → the active job.
        Returns ``{job_id, state, error?}``; ``error`` on a refusal (wrong state
        / multi-GPU accelerate run)."""
        if job_id is None:
            health = self.health() or {}
            job_id = health.get("active_job")
            if not job_id:
                return {"error": "no active job"}
        return self._request("POST", f"/jobs/{job_id}/pause")

    def resume_job(self, job_id: Optional[str] = None) -> dict:
        """Thaw a paused job's tree (SIGCONT) back to running. ``None`` → the
        active (paused) job. Returns ``{job_id, state, error?}``."""
        if job_id is None:
            health = self.health() or {}
            job_id = health.get("active_job")
            if not job_id:
                return {"error": "no active job"}
        return self._request("POST", f"/jobs/{job_id}/resume")

    def shutdown(self, *, kill_jobs: bool = True) -> Optional[dict]:
        try:
            return self._request("POST", "/shutdown", {"kill_jobs": kill_jobs})
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def stream(self, path: str) -> Iterator[str]:
        """Yield ``data:`` payloads from an SSE endpoint until the socket drops."""
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=None) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("data: "):
                    yield line[len("data: ") :]

    def stream_events(self) -> Iterator[str]:
        return self.stream("/events")

    def stream_logs(self, job_id: str) -> Iterator[str]:
        return self.stream(f"/jobs/{job_id}/logs")


Client = DaemonClient


def ensure_daemon(
    *,
    timeout: float = 60.0,
    port: Optional[int] = None,
    expected_root: Optional[str | Path] = None,
) -> DaemonClient:
    """Return a client to a live daemon, starting one if needed.

    Idempotent: if ``/health`` answers we return a client; otherwise spawn
    ``python -m anima_daemon`` detached (stdout -> ``daemon.log``) and poll
    ``/health`` until it answers or ``timeout`` elapses.

    The daemon may bind a different port than requested if the preferred one
    is taken by a stranger (``server.serve_with_fallback``), so the port is
    re-resolved from the pidfile each tick. Poll cadence ramps 0.1s -> 0.5s.
    """
    requested = port or _resolve_port()
    client = DaemonClient(requested)
    health = client.health()
    if health is not None:
        if expected_root is None or daemon_matches_root(health, expected_root):
            # Our checkout: reuse it unless it's running stale code; restart is
            # lossless (reconcile re-adopts the running job).
            if not daemon_is_stale(health):
                return client
            logger.info("daemon is running stale code; restarting")
            client.shutdown(kill_jobs=False)
            _await_down(client, min(timeout, 5.0))
        else:
            # A different checkout's daemon holds the port. Refuse to evict it if
            # it has live work; otherwise reclaim the port for ours.
            if health.get("active_job") or _has_live_jobs(client):
                raise RuntimeError(
                    f"{_root_mismatch_message(health, expected_root)}; "
                    "it still has queued or running jobs"
                )
            client.shutdown(kill_jobs=False)
            _await_down(client, min(timeout, 5.0))

    config.ensure_state_dirs()
    proc.spawn_detached(
        # pythonw.exe on Windows (see venv_python); logs go to daemon.log.
        [venv_python(windowless=True), "-m", "anima_daemon", str(requested)],
        cwd=config.ROOT,
        stdout_path=config.DAEMON_LOG,
    )
    deadline = time.time() + timeout
    interval = 0.1
    while time.time() < deadline:
        resolved = _resolve_port()  # follow a fallback-to-ephemeral daemon
        if resolved != client.port:
            client = DaemonClient(resolved)
        health = client.health()
        if health is not None and expected_root is None:
            return client
        if health is not None and expected_root is not None:
            if daemon_matches_root(health, expected_root):
                return client
            if health.get("active_job") or _has_live_jobs(client):
                raise RuntimeError(
                    f"{_root_mismatch_message(health, expected_root)}; "
                    "it still has queued or running jobs"
                )
        time.sleep(interval)
        interval = min(interval * 1.5, 0.5)  # ramp 0.1 → 0.5s
    raise RuntimeError(
        f"daemon did not come up within {timeout:.0f}s; see {config.DAEMON_LOG}"
    )


def is_running() -> bool:
    return DaemonClient().health() is not None
