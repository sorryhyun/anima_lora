"""Stdlib HTTP surface for the daemon — zero new deps, localhost only.

A hand-written ``(method, path)`` dispatch on a ``BaseHTTPRequestHandler``;
request bodies are unvalidated ``json.loads``'d dicts. Served by
``ThreadingHTTPServer``, so a parked SSE stream holds one thread.

Full endpoint reference: ``anima_daemon/README.md`` (also served live at
``GET /``); machine-readable manifest at ``GET /tools`` (``TOOLS`` below).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, tail
from .manager import JobManager

logger = logging.getLogger("anima.daemon")

_JOB_RE = re.compile(r"^/jobs/(?P<id>[^/]+)$")
_JOB_STOP_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/stop$")
_JOB_PAUSE_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/pause$")
_JOB_RESUME_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/resume$")
_JOB_LOGS_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/logs$")
_JOB_PROGRESS_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/progress$")

_README = Path(__file__).resolve().parent / "README.md"

# Machine-readable manifest served at GET /tools and registered by mcp.py — one
# entry per operation with a JSON-Schema `input_schema`. Kept in sync with the
# handlers by hand.
TOOLS = [
    {
        "name": "submit_training",
        "description": (
            "Enqueue a train.py run built from method + preset + overrides. "
            "Runs when it reaches the front of the serial queue. Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["method"],
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Method/adapter config name (e.g. 'lora', 'chimera', 'easycontrol').",
                },
                "preset": {
                    "type": "string",
                    "default": "default",
                    "description": "Hardware preset: default | low_vram | half | quarter | tenth | graft | debug (configs/presets.toml).",
                },
                "methods_subdir": {
                    "type": "string",
                    "description": "Config subdir, e.g. 'gui-methods' for the clean per-variant tree. Omit for 'methods'.",
                },
                "overrides": {
                    "type": "object",
                    "description": '{key: value} → --key value CLI overrides, e.g. {"network_dim": 32, "max_train_epochs": 64}.',
                },
                "extra": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra CLI args appended verbatim.",
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "A fully-merged config dict to pin instead of re-resolving the base→preset→method→overrides chain at launch.",
                },
                "config_file": {
                    "type": "string",
                    "description": "Path to a config snapshot file (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now (resume queue); false → add to queue (held for Start Queue only if the queue is idle; auto-advances behind a job that's already running); omit → leave the gate as-is.",
                },
            },
        },
    },
    {
        "name": "submit_command",
        "description": (
            "Enqueue a plain `python <argv>` task (preprocess, mask, a distill loop). "
            "Optionally carries a chain_train spec the daemon auto-enqueues on success "
            "(how 'preprocess → train' survives the caller closing). Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["label", "argv"],
            "properties": {
                "kind": {
                    "type": "string",
                    "const": "command",
                    "description": "Must be 'command'.",
                },
                "label": {
                    "type": "string",
                    "description": "Display label (doubles as the job's 'method' field).",
                },
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argv run under the anima venv, e.g. ['tasks.py', 'preprocess-config', …].",
                },
                "extra_env": {
                    "type": "object",
                    "description": "Extra environment variables for the subprocess.",
                },
                "chain_train": {
                    "type": "object",
                    "description": "Training spec {method, preset, methods_subdir, overrides} auto-enqueued when this command finishes successfully.",
                },
                "stall_timeout": {
                    "type": "number",
                    "description": (
                        "Per-job stall-watchdog budget in seconds, overriding the "
                        "120s command-job default; 0 disables it. Raise (or disable) "
                        "for a legitimately quiet embed/eval loop."
                    ),
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "Config dict to pin (forwarded to the chained train job).",
                },
                "config_file": {
                    "type": "string",
                    "description": "Config snapshot file path (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now; false → add to queue (held only if the queue is idle; auto-advances behind a running job); omit → leave gate as-is.",
                },
            },
        },
    },
    {
        "name": "list_jobs",
        "description": "List all jobs (full records, submission order). Each has state ∈ queued|running|paused|done|error|stopped.",
        "method": "GET",
        "path": "/jobs",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_job",
        "description": (
            "Fetch one job record plus live fields: 'latest' (last progress.jsonl event, "
            "train jobs only) and 'stale_for' (seconds since last progress tick). Poll this for completion."
        ),
        "method": "GET",
        "path": "/jobs/{id}",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."}
            },
        },
    },
    {
        "name": "get_progress",
        "description": (
            "Query a train job's structured progress.jsonl — events are "
            "run_start | step (loss/lr/metrics) | val (cmmd) | ckpt | log "
            "(mirrored WARNING+ records) | run_end. Filters keep the payload "
            "small on long runs; returns {job_id, state, count, events}. "
            "This is the surface to debug/analyze a run from (loss curve, "
            "warnings, checkpoints) — stdout is only for raw crash output."
        ),
        "method": "GET",
        "path": "/jobs/{id}/progress",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."},
                "events": {
                    "type": "string",
                    "description": "Comma-separated ev kinds to keep, e.g. 'step,val' or 'log,run_end'. Omit for all.",
                },
                "since_step": {
                    "type": "integer",
                    "description": "Keep events at/after this global_step (step-less events inherit the preceding step).",
                },
                "every_nth": {
                    "type": "integer",
                    "description": "Thin step events to every n-th (latest step always kept).",
                },
                "last_n": {
                    "type": "integer",
                    "default": 200,
                    "description": "Trailing cap on returned events.",
                },
            },
        },
    },
    {
        "name": "stop_job",
        "description": "Abort a running or queued job (tree-kills the process). Returns {job_id, state}.",
        "method": "POST",
        "path": "/jobs/{id}/stop",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to stop."}},
        },
    },
    {
        "name": "pause_job",
        "description": (
            "Freeze a running job's process tree (SIGSTOP) — VRAM/CUDA context "
            "stay put, SM utilisation drops to zero, resume is instant. The queue "
            "does NOT advance past it (it still owns its slot). Refuses anything "
            "not running, and refuses a multi-GPU accelerate-launch run. Returns "
            "{job_id, state, error?}."
        ),
        "method": "POST",
        "path": "/jobs/{id}/pause",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to pause."}},
        },
    },
    {
        "name": "resume_job",
        "description": (
            "Thaw a paused job's process tree (SIGCONT) back to running. Returns "
            "{job_id, state, error?} (error if the job isn't paused)."
        ),
        "method": "POST",
        "path": "/jobs/{id}/resume",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id to resume."}
            },
        },
    },
    {
        "name": "tail_logs",
        "description": (
            "SSE stream of a job's combined stdout+stderr from the start of the file; "
            "ends with a {ev:'eof', state} event once the job is terminal and drained, "
            "then closes the connection. There is no blocking server-side 'wait' "
            "endpoint — tail this, poll get_job, or use the client-side "
            "DaemonClient.wait / `make daemon-wait JOB=<id>`."
        ),
        "method": "GET",
        "path": "/jobs/{id}/logs",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to tail."}},
        },
    },
    {
        "name": "pause_queue",
        "description": "Hold the queue gate — queued jobs wait until start_queue. Submissions still accepted.",
        "method": "POST",
        "path": "/queue/pause",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "start_queue",
        "description": "Resume a paused queue — the worker launches queued jobs in order.",
        "method": "POST",
        "path": "/queue/start",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "health",
        "description": "Daemon liveness: {ok, pid, port, root, fingerprint, active_job, paused, worker_alive, worker_idle_for}. 'root' is the checkout it belongs to; 'fingerprint' is the daemon-source hash it booted with (stale if it differs from current on-disk source); worker_idle_for is seconds since the job worker last advanced.",
        "method": "GET",
        "path": "/health",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "shutdown",
        "description": "Stop the daemon. Destructive: kill_jobs=true (default) tree-kills the running job too.",
        "method": "POST",
        "path": "/shutdown",
        "input_schema": {
            "type": "object",
            "properties": {
                "kill_jobs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Kill the running job on the way down.",
                }
            },
        },
    },
]


class _Handler(BaseHTTPRequestHandler):
    server_version = "AnimaDaemon/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def manager(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, *, content_type: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _open_sse(self) -> None:
        """Open an SSE response — one response per connection, never keep-alive.

        An SSE body has no ``Content-Length``/chunked framing, so the socket
        closing is the client's only EOF signal. With keep-alive every consumer
        hangs after the ``eof`` event — keep ``Connection: close`` +
        ``close_connection``.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj) -> bool:
        """Write one SSE event and flush. Returns False on a dropped client —
        ``wfile`` buffers, so without the flush the client sees nothing until
        the buffer fills."""
        try:
            payload = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def log_message(self, fmt, *args) -> None:  # quieter than default stderr spam
        logger.debug("http: " + fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path in ("/", "/readme"):
            self._handle_readme()
        elif path == "/tools":
            self._send_json(TOOLS)
        elif path == "/health":
            self._handle_health()
        elif path == "/jobs":
            self._handle_list()
        elif path == "/events":
            self._handle_events()
        elif m := _JOB_LOGS_RE.match(path):
            self._handle_logs(m.group("id"))
        elif m := _JOB_PROGRESS_RE.match(path):
            self._handle_progress(m.group("id"), query)
        elif m := _JOB_RE.match(path):
            self._handle_get(m.group("id"))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/jobs":
            self._handle_submit()
        elif path == "/queue/start":
            self.manager.resume()
            self._send_json({"ok": True, "paused": False})
        elif path == "/queue/pause":
            self.manager.pause()
            self._send_json({"ok": True, "paused": True})
        elif path == "/shutdown":
            self._handle_shutdown()
        elif m := _JOB_STOP_RE.match(path):
            self._handle_stop(m.group("id"))
        elif m := _JOB_PAUSE_RE.match(path):
            self._handle_pause(m.group("id"))
        elif m := _JOB_RESUME_RE.match(path):
            self._handle_resume(m.group("id"))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def _handle_readme(self) -> None:
        try:
            text = _README.read_text(encoding="utf-8")
        except OSError:
            # README not shipped (e.g. a trimmed vendor tree): point at /tools.
            self._send_json({"error": "README.md not found", "tools": "/tools"}, 404)
            return
        self._send_text(text, content_type="text/markdown; charset=utf-8")

    def _handle_health(self) -> None:
        active = self.manager.active_job()
        self._send_json(
            {
                "ok": True,
                "pid": os.getpid(),
                "port": self.server.server_address[1],
                "root": str(config.ROOT),
                # Fingerprint we BOOTED with, not re-hashed live — see
                # config.source_fingerprint / client.daemon_is_stale.
                "fingerprint": getattr(self.server, "boot_fingerprint", None),
                "active_job": active.id if active else None,
                "paused": self.manager.is_paused(),
                "worker_alive": self.manager.worker_alive(),
                "worker_idle_for": self.manager.worker_idle_for(),
            }
        )

    def _handle_submit(self) -> None:
        body = self._read_json()
        # start: True → run now (resume queue), False → enqueue paused, None → as-is.
        start = body.get("start")
        if (body.get("kind") or "train") == "command":
            argv = body.get("argv")
            if not isinstance(argv, list) or not argv:
                self._send_json({"error": "missing 'argv' for command job"}, 400)
                return
            stall_timeout = body.get("stall_timeout")
            try:
                stall_timeout = (
                    float(stall_timeout) if stall_timeout is not None else None
                )
            except (TypeError, ValueError):
                self._send_json({"error": "'stall_timeout' must be a number"}, 400)
                return
            job = self.manager.submit_command(
                label=body.get("label") or "command",
                argv=[str(a) for a in argv],
                extra_env=body.get("extra_env") or {},
                chain_train=body.get("chain_train") or None,
                stall_timeout=stall_timeout,
                config_snapshot=body.get("config_snapshot") or None,
                config_file=body.get("config_file") or None,
                start=start,
                captured_env=body.get("captured_env") or {},
            )
            self._send_json({"job_id": job.id, "state": job.state}, 201)
            return
        method = body.get("method")
        if not method:
            self._send_json({"error": "missing 'method'"}, 400)
            return
        job = self.manager.submit(
            method=method,
            preset=body.get("preset") or "default",
            methods_subdir=body.get("methods_subdir"),
            config_snapshot=body.get("config_snapshot") or None,
            config_file=body.get("config_file") or None,
            overrides=body.get("overrides") or {},
            extra=body.get("extra") or [],
            start=start,
            captured_env=body.get("captured_env") or {},
        )
        self._send_json({"job_id": job.id, "state": job.state}, 201)

    def _handle_list(self) -> None:
        self._send_json([j.public() for j in self.manager.list_jobs()])

    def _handle_get(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        out = job.public()
        out["latest"] = tail.last_event(job.progress_path)
        out["stale_for"] = self.manager.stale_for(job)
        self._send_json(out)

    def _handle_progress(self, job_id: str, query: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        params = urllib.parse.parse_qs(query)

        def _int(name: str, default=None):
            raw = (params.get(name) or [None])[0]
            try:
                return int(raw) if raw is not None else default
            except ValueError:
                return default

        raw_events = (params.get("events") or [None])[0]
        kinds = (
            [s.strip() for s in raw_events.split(",") if s.strip()]
            if raw_events
            else None
        )
        evs = tail.read_events(
            job.progress_path,
            events=kinds,
            since_step=_int("since_step"),
            every_nth=_int("every_nth"),
            last_n=_int("last_n", 200),
        )
        self._send_json(
            {
                "job_id": job.id,
                "state": job.state,
                "progress_path": job.progress_path,
                "count": len(evs),
                "events": evs,
            }
        )

    def _handle_stop(self, job_id: str) -> None:
        job = self.manager.stop(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._send_json({"job_id": job.id, "state": job.state})

    def _handle_pause(self, job_id: str) -> None:
        result = self.manager.pause_job(job_id)
        if result is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        # A refusal rides an `error` field in a 200 body; only a missing job 404s.
        self._send_json(result)

    def _handle_resume(self, job_id: str) -> None:
        result = self.manager.resume_job(job_id)
        if result is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._send_json(result)

    def _handle_shutdown(self) -> None:
        body = self._read_json()
        kill = bool(body.get("kill_jobs", True))
        self._send_json({"ok": True, "kill_jobs": kill})
        # Trigger shutdown after the response is flushed, off the handler thread
        # (server.shutdown() must not run in a request thread).
        threading.Thread(
            target=self.server.request_shutdown,  # type: ignore[attr-defined]
            args=(kill,),
            daemon=True,
        ).start()

    def _handle_events(self) -> None:
        q = self.manager.subscribe()
        self._open_sse()
        try:
            if not self._sse({"ev": "hello", "ts": time.time()}):
                return
            while True:
                try:
                    event = q.get(timeout=15)
                except Exception:
                    # idle keepalive comment so proxies/clients don't time out
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    continue
                if not self._sse(event):
                    return
        finally:
            self.manager.unsubscribe(q)

    def _handle_logs(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._open_sse()
        path = Path(job.stdout_path) if job.stdout_path else None
        if path is None:
            self._sse({"error": "no stdout for job"})
            return
        for line in tail.follow(path, from_start=True):
            if line:
                if not self._sse(line.rstrip("\n")):
                    return
            else:
                # Empty tick = log drained; stop once the job has left
                # queued/running.
                cur = self.manager.get(job_id)
                if cur is not None and cur.state not in ("queued", "running"):
                    self._sse({"ev": "eof", "state": cur.state})
                    return


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR means "rebind a TIME_WAIT socket" on POSIX (safe) but
    # "double-bind a live in-use port" on Windows, defeating
    # serve_with_fallback's collision detection — so Windows must fail loudly.
    allow_reuse_address = os.name != "nt"

    def __init__(self, addr, manager: JobManager, *, fingerprint=None):
        super().__init__(addr, _Handler)
        self.manager = manager
        self.boot_fingerprint = fingerprint

    def request_shutdown(self, kill_jobs: bool) -> None:
        self.manager.shutdown(kill_jobs=kill_jobs)
        self.shutdown()  # unblocks serve_forever()


def serve(manager: JobManager, *, port: int, fingerprint=None) -> _Server:
    """Bind 127.0.0.1:port and return the server (call ``serve_forever``)."""
    return _Server((config.HOST, port), manager, fingerprint=fingerprint)


def serve_with_fallback(manager: JobManager, *, port: int, fingerprint=None) -> _Server:
    """Bind ``port``; if it's already taken, fall back to an OS-chosen free one.

    On ``EADDRINUSE`` we first probe the port: if an anima daemon already
    answers ``/health`` there (a startup-race sibling that won), we re-raise
    so the caller stands down instead of spinning up a second daemon that
    would overwrite the pidfile. Only a *stranger* holding the port sends us
    to an ephemeral one (``ensure_daemon`` re-resolves it from the pidfile).
    """
    try:
        return _Server((config.HOST, port), manager, fingerprint=fingerprint)
    except OSError:
        from .client import DaemonClient

        # A sibling may have bound the socket but not yet reached
        # serve_forever; probe a few times.
        for _ in range(3):
            if DaemonClient(port).health(timeout=0.5) is not None:
                raise  # an anima daemon owns it → let the caller stand down
            time.sleep(0.3)
        logger.warning(
            "127.0.0.1:%s held by a non-anima process; using an ephemeral port",
            port,
        )
        return _Server((config.HOST, 0), manager, fingerprint=fingerprint)
