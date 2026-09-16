"""Stdio MCP bridge for the daemon.

Exposes the daemon's HTTP surface as MCP tools, resolving the daemon via the
pidfile (``config.discover_pidfile``). Setup and the deviations from
``server.TOOLS``: ``anima_daemon/README.md`` "MCP bridge".

Transport: newline-delimited JSON-RPC 2.0 over stdio. Nothing but protocol
messages may touch stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # direct file execution: `python anima_daemon/mcp.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import anima_daemon  # noqa: F401  — make the package importable, then

    __package__ = "anima_daemon"  # re-anchor so the relative imports below work

import json
import urllib.error
import urllib.parse
from typing import Any, Callable, Optional

from . import config, tail
from .client import DaemonClient, ensure_daemon
from .server import TOOLS

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "anima-daemon", "version": "1.0"}

# SSE endpoints can't be tool calls (a call returns once); tail_log replaces it.
_SKIP = {"tail_logs"}

# Tools allowed to auto-start the daemon. Everything else attaches passively.
_ENSURE = {"submit_training", "submit_command"}

TAIL_LOG_TOOL = {
    "name": "tail_log",
    "description": (
        "Last N lines of a job's combined stdout+stderr plus its current state. "
        "Non-streaming substitute for the SSE /jobs/{id}/logs endpoint — poll "
        "this (or get_job) to follow a run. Works even when the daemon is down "
        "(falls back to the on-disk job.json record)."
    ),
    "input_schema": {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "string", "description": "Job id from submit_*."},
            "lines": {
                "type": "integer",
                "default": 80,
                "description": "How many trailing lines to return.",
            },
        },
    },
}


def _tail_lines(path: str, n: int, *, max_bytes: int = 262_144) -> list[str]:
    """Last ``n`` non-blank lines of a (possibly huge) log, decoded leniently.
    ``\\r``-separated tqdm redraws collapse to their final rendering."""
    try:
        p = Path(path)
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
    except OSError:
        return []
    lines: list[str] = []
    for raw in data.decode("utf-8", errors="replace").split("\n"):
        if "\r" in raw:
            raw = next((s for s in reversed(raw.split("\r")) if s.strip()), "")
        raw = raw.rstrip()
        if raw:
            lines.append(raw)
    return lines[-n:]


def _error_result(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": True}


class MCPServer:
    """Protocol state + tool dispatch, transport-free for testability.

    ``handle(msg)`` maps one JSON-RPC message to a response dict (or ``None``
    for notifications); the stdio loop in :func:`main` is just framing around
    it. ``client_factory`` / ``ensure`` are injectable so tests can point the
    bridge at an in-process daemon instead of the pidfile-resolved one.
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[], DaemonClient] = DaemonClient,
        ensure: Optional[Callable[[], DaemonClient]] = None,
    ) -> None:
        self._client_factory = client_factory
        self._ensure = ensure or (lambda: ensure_daemon(expected_root=config.ROOT))
        self._tools = [t for t in TOOLS if t["name"] not in _SKIP] + [TAIL_LOG_TOOL]
        self._by_name = {t["name"]: t for t in self._tools}

    def handle(self, msg: dict) -> Optional[dict]:
        method = msg.get("method")
        msg_id = msg.get("id")
        is_notification = "id" not in msg
        try:
            if method == "initialize":
                result = self._initialize(msg.get("params") or {})
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": t["name"],
                            "description": t["description"],
                            "inputSchema": t["input_schema"],
                        }
                        for t in self._tools
                    ]
                }
            elif method == "tools/call":
                result = self._call(msg.get("params") or {})
            elif is_notification:
                return None  # notifications/initialized, notifications/cancelled, …
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
        except Exception as e:  # noqa: BLE001 — protocol layer must not crash
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(e)},
            }
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _initialize(self, params: dict) -> dict:
        client_version = params.get("protocolVersion")
        return {
            "protocolVersion": (
                client_version if isinstance(client_version, str) else PROTOCOL_VERSION
            ),
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }

    def _call(self, params: dict) -> dict:
        name = params.get("name")
        args = dict(params.get("arguments") or {})
        tool = self._by_name.get(name)
        if tool is None:
            return _error_result(f"unknown tool: {name}")
        try:
            payload = self._invoke(name, tool, args)
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                detail = ""
            return _error_result(f"daemon returned HTTP {e.code}: {detail or e.reason}")
        except (urllib.error.URLError, OSError) as e:
            return _error_result(f"cannot reach the daemon: {e}")
        except RuntimeError as e:  # root mismatch / boot failure / bad args
            return _error_result(str(e))
        return {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
            "isError": False,
        }

    def _invoke(self, name: str, tool: dict, args: dict) -> Any:
        if name == "tail_log":
            return self._tail_log(args)
        if name == "get_progress":
            # Handled locally: the generic GET dispatch drops query args, and
            # reading progress.jsonl directly also works with the daemon down.
            return self._get_progress(args)
        if name in _ENSURE:
            client = self._ensure()
        else:
            client = self._client_factory()
            if client.health() is None:
                if name == "health":
                    return {"ok": False, "up": False, "detail": "no daemon is running"}
                raise RuntimeError(
                    "no daemon is running; submit a job (or run "
                    "`python tasks.py daemon`) to start one"
                )
        path = tool["path"]
        if "{id}" in path:
            job_id = str(args.pop("id", "") or "")
            if not job_id:
                raise RuntimeError("missing required argument: id")
            path = path.replace("{id}", urllib.parse.quote(job_id, safe=""))
        if name == "submit_command":
            args["kind"] = "command"  # the daemon defaults a bare /jobs POST to train
        if name in ("submit_training", "submit_command"):
            # The raw-POST path skips client.submit's auto-capture, so
            # snapshot the bridge's whitelisted env here (an explicit
            # captured_env in the call still wins).
            args.setdefault("captured_env", config.capture_env())
        if tool["method"] == "GET":
            return client._request("GET", path)
        return client._request("POST", path, args)

    def _job_record(self, job_id: str) -> Optional[dict]:
        """The job record via the daemon, or the on-disk ``job.json`` when the
        daemon is down (or doesn't know the id)."""
        client = self._client_factory()
        if client.health() is not None:
            try:
                return client.get(job_id)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
        try:
            return json.loads(
                (config.job_dir(job_id) / "job.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def _tail_log(self, args: dict) -> dict:
        job_id = str(args.get("id") or "")
        if not job_id:
            raise RuntimeError("missing required argument: id")
        n = max(1, int(args.get("lines") or 80))
        job = self._job_record(job_id)
        if job is None:
            return {"error": "no such job", "job_id": job_id}
        stdout_path = job.get("stdout_path")
        return {
            "job_id": job_id,
            "state": job.get("state"),
            "error": job.get("error"),
            "lines": _tail_lines(stdout_path, n) if stdout_path else [],
        }

    def _get_progress(self, args: dict) -> dict:
        job_id = str(args.get("id") or "")
        if not job_id:
            raise RuntimeError("missing required argument: id")
        job = self._job_record(job_id)
        if job is None:
            return {"error": "no such job", "job_id": job_id}
        raw_events = args.get("events")
        if isinstance(raw_events, str):
            kinds = [s.strip() for s in raw_events.split(",") if s.strip()] or None
        elif isinstance(raw_events, list):
            kinds = [str(s) for s in raw_events] or None
        else:
            kinds = None

        def _int(name: str, default=None):
            val = args.get(name)
            try:
                return int(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        evs = tail.read_events(
            job.get("progress_path"),
            events=kinds,
            since_step=_int("since_step"),
            every_nth=_int("every_nth"),
            last_n=_int("last_n", 200),
        )
        return {
            "job_id": job_id,
            "state": job.get("state"),
            "progress_path": job.get("progress_path"),
            "count": len(evs),
            "events": evs,
        }


def main() -> int:
    server = MCPServer()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            print(
                f"anima-daemon mcp: skipping non-JSON line: {raw[:80]}", file=sys.stderr
            )
            continue
        if not isinstance(msg, dict):
            continue
        resp = server.handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
