"""Structured training-progress sink.

Writes a JSONL event stream next to the checkpoint so any consumer (the GUI
progress bar, the daemon, an MCP client) can follow a run by tailing one
file. Append-only,
line-buffered, main-process only. One event per line:

    {"ev": "run_start", "ts": 0.0, "run": ..., "method": ..., "preset": ...,
     "total_steps": ..., "total_epochs": ..., "pid": ...}
    {"ev": "step", "ts": ..., "global_step": ..., "epoch": ..., "loss": ..., ...}
    {"ev": "val",  "ts": ..., "global_step": ..., "epoch": ..., "cmmd": ...}
    {"ev": "ckpt", "ts": ..., "global_step": ..., "path": ...}
    {"ev": "log",  "ts": ..., "level": "WARNING|ERROR|...", "logger": ...,
     "msg": ...}
    {"ev": "run_end", "ts": ..., "status": "ok|error|stopped", "final_step": ...,
     "error": ...}

``log`` events mirror WARNING+ records from the root logger (see
:meth:`ProgressSink.attach_log_mirror`), so the run's warnings/errors are in
this one structured file.

A reader tails the file: missing file = not started; last line ``run_end`` =
done. Every write is wrapped so a logging failure can never crash training.
:func:`read_status` is the read side — it digests a whole stream into one
status dict (step / rate / ETA / last ckpt / terminal status); ``make
run-status`` (``scripts/run_status.py``) is its CLI.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """``json.dumps`` ``default`` hook: coerce tensors / numpy scalars to
    plain Python numbers, falling back to ``str`` for anything exotic."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _flatten_logs(logs: dict) -> dict:
    """Keep only JSON-friendly scalar entries from a ``logs`` dict.

    The training ``logs`` dict carries floats, ints, bools and (occasionally)
    0-dim tensors. Drop anything that isn't scalar-shaped so the event line
    stays small and parseable.
    """
    out: dict[str, Any] = {}
    for key, val in logs.items():
        if isinstance(val, (int, float, bool, str)):
            out[key] = val
        else:
            item = getattr(val, "item", None)
            if callable(item):
                try:
                    out[key] = item()
                except Exception:
                    continue
    return out


def _find_cmmd(logs: dict) -> Optional[float]:
    """Pull the CMMD value out of a validation ``logs`` dict.

    CMMD validation logs a ``..._cmmd`` key (see
    ``library/training/validation.py``); return its scalar value if present.
    """
    for key, val in logs.items():
        if key.endswith("_cmmd"):
            item = getattr(val, "item", None)
            try:
                return item() if callable(item) else float(val)
            except Exception:
                return None
    return None


class _SinkLogHandler(logging.Handler):
    """Mirrors log records into a :class:`ProgressSink` as ``log`` events.

    Capped so a record emitted every step can't bloat the stream — after
    ``max_events`` a final notice is written and the rest are dropped (they
    still reach the normal console/stdout handlers).
    """

    def __init__(self, sink: "ProgressSink", *, max_events: int = 500) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink
        self._remaining = max_events

    def emit(self, record: logging.LogRecord) -> None:
        if self._remaining <= 0:
            return
        self._remaining -= 1
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        self._sink._emit("log", level=record.levelname, logger=record.name, msg=msg)
        if self._remaining == 0:
            self._sink._emit(
                "log",
                level="WARNING",
                logger=__name__,
                msg="log-event cap reached; further records go to stdout only",
            )


class ProgressSink:
    """Append-only JSONL progress writer. Construct on the main process only."""

    def __init__(
        self,
        path: str,
        *,
        run: str,
        method: Optional[str],
        preset: Optional[str],
        t0: Optional[float] = None,
    ) -> None:
        self._path = path
        self._run = run
        self._method = method
        self._preset = preset
        self._t0 = t0 if t0 is not None else time.time()
        self._fh = None
        self._closed = False
        self._log_handler: Optional[_SinkLogHandler] = None

    @staticmethod
    def resolve_path(args) -> Optional[str]:
        """Resolve the JSONL path from args, or ``None`` to disable.

        ``--progress_jsonl`` unset → derive
        ``<output_dir>/../logs/<output_name>.progress.jsonl`` (default on) — a
        sibling ``logs/`` dir so the checkpoint dir holds only model artifacts.
        Explicit empty / ``none`` / ``off`` → disabled. Any other value → that
        literal path. (The daemon always passes an explicit per-job path, so this
        derived default only governs inline CLI runs.)
        """
        explicit = getattr(args, "progress_jsonl", None)
        if explicit is not None:
            explicit = explicit.strip()
            if explicit.lower() in ("", "none", "off"):
                return None
            return explicit
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return None
        output_name = getattr(args, "output_name", None) or "run"
        # Sibling logs/ dir next to the checkpoint dir (parent of output_dir);
        # fall back to a logs/ subdir if output_dir has no parent component.
        parent = os.path.dirname(os.path.normpath(output_dir))
        logs_dir = os.path.join(parent or output_dir, "logs")
        return os.path.join(logs_dir, f"{output_name}.progress.jsonl")

    def _emit(self, ev: str, **fields: Any) -> None:
        if self._closed or self._fh is None:
            return
        try:
            rec = {"ev": ev, "ts": round(time.time() - self._t0, 3)}
            rec.update(fields)
            self._fh.write(json.dumps(rec, default=_jsonable) + "\n")
        except Exception as exc:  # progress logging must never crash training
            logger.debug("progress sink write failed (%s): %s", ev, exc)

    # region lifecycle events

    def run_start(
        self,
        *,
        total_steps: int,
        total_epochs: int,
        pid: int,
        log_dir: Optional[str] = None,
    ) -> None:
        """Open the file fresh (truncating any stale stream) and write the
        opening event."""
        if self._closed:
            return
        try:
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # line-buffered so tailing readers see each event immediately
            self._fh = open(self._path, "w", buffering=1, encoding="utf-8")
        except Exception as exc:
            logger.debug("progress sink open failed: %s", exc)
            self._fh = None
            return
        extra = {"log_dir": log_dir} if log_dir is not None else {}
        self._emit(
            "run_start",
            run=self._run,
            method=self._method,
            preset=self._preset,
            total_steps=total_steps,
            total_epochs=total_epochs,
            pid=pid,
            **extra,
        )

    def run_end(
        self,
        *,
        status: str,
        final_step: int,
        error: Optional[str] = None,
        **fields: Any,
    ) -> None:
        self._emit(
            "run_end", status=status, final_step=final_step, error=error, **fields
        )
        self.close()

    # endregion

    def log(
        self,
        logs: dict,
        *,
        global_step: int,
        epoch: int,
        val_step: Optional[int] = None,
    ) -> None:
        """Emit a ``step`` or ``val`` event from a training ``logs`` dict.

        A dict carrying a ``..._cmmd`` key (or an explicit ``val_step``) is a
        validation pass → ``val``; everything else → ``step``.
        """
        if self._fh is None:
            return
        cmmd = _find_cmmd(logs)
        if val_step is not None or cmmd is not None:
            fields = {"global_step": global_step, "epoch": epoch}
            if cmmd is not None:
                fields["cmmd"] = cmmd
            if val_step is not None:
                fields["val_step"] = val_step
            self._emit("val", **fields)
        else:
            fields = _flatten_logs(logs)
            fields["global_step"] = global_step
            fields["epoch"] = epoch
            self._emit("step", **fields)

    def ckpt(self, *, global_step: int, path: str) -> None:
        self._emit("ckpt", global_step=global_step, path=path)

    def attach_log_mirror(self, *, max_events: int = 500) -> None:
        """Mirror WARNING+ records from the root logger as ``log`` events.

        Call after :meth:`run_start` (no-op while the stream is unopened).
        The handler sits on the root logger so anything that propagates —
        ``library.*``, ``networks.*``, third-party warnings routed through
        ``logging`` — lands in the stream. Detached automatically on
        :meth:`close`.
        """
        if self._fh is None or self._closed or self._log_handler is not None:
            return
        self._log_handler = _SinkLogHandler(self, max_events=max_events)
        logging.getLogger().addHandler(self._log_handler)

    def close(self) -> None:
        if self._log_handler is not None:
            try:
                logging.getLogger().removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._closed = True


def _pid_alive(pid: Optional[int]) -> Optional[bool]:
    """``True``/``False`` if the pid is/isn't running, ``None`` if unknowable."""
    if not pid:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    except Exception:
        return None
    return True


def read_status(path: str, *, rate_window: int = 20) -> dict:
    """Digest a ``progress.jsonl`` stream into one run-status dict.

    Answers "what step is this run at, and is it still alive?" with a
    whole-file read of an append-only stream (a few thousand short lines even
    for a long run). Truncated / partially-written lines are skipped, so it is safe to
    call against a live run mid-write.

    Keys: ``run``/``method``/``preset``/``pid``/``log_dir`` (from ``run_start``),
    ``global_step``/``total_steps``/``pct``, ``elapsed``/``rate``/``eta``
    (seconds; ``rate`` is steps/sec over the last ``rate_window`` step events,
    ``None`` before two land), ``metrics`` (last ``step`` event's scalars),
    ``val`` (last ``val`` event), ``ckpt`` (last ``ckpt`` event), ``warnings``
    (``log`` event count) and ``status``:

    ``running`` · ``ok`` · ``error`` · ``stopped`` (the ``run_end`` statuses) ·
    ``dead`` (no ``run_end`` and the pid is gone — it was killed / OOMed) ·
    ``unknown`` (no ``run_end``, liveness unknowable).
    """
    events: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue  # torn last line on a live stream
    if not events:
        raise ValueError(f"no progress events in {path}")

    def _last(ev: str) -> Optional[dict]:
        for rec in reversed(events):
            if rec.get("ev") == ev:
                return rec
        return None

    start = _last("run_start") or {}
    end = _last("run_end")
    steps = [r for r in events if r.get("ev") == "step"]
    last_step = steps[-1] if steps else None
    last_val = _last("val")
    last_ckpt = _last("ckpt")

    global_step = None
    if end is not None:
        global_step = end.get("final_step")
    if global_step is None and last_step is not None:
        global_step = last_step.get("global_step")
    total_steps = start.get("total_steps")
    elapsed = events[-1].get("ts")

    # Rate over a trailing window of step events — the run-average would be
    # skewed by startup (model load, compile) on an otherwise steady run.
    rate = None
    window = steps[-rate_window:]
    if len(window) >= 2:
        d_step = window[-1].get("global_step", 0) - window[0].get("global_step", 0)
        d_ts = window[-1].get("ts", 0) - window[0].get("ts", 0)
        if d_step > 0 and d_ts > 0:
            rate = d_step / d_ts

    if end is not None:
        status = end.get("status") or "unknown"
    else:
        alive = _pid_alive(start.get("pid"))
        status = "running" if alive else ("dead" if alive is False else "unknown")

    remaining = (
        total_steps - global_step
        if (total_steps and global_step is not None and status == "running")
        else None
    )
    metrics = {
        k: v
        for k, v in (last_step or {}).items()
        if k not in ("ev", "ts", "global_step", "epoch")
    }
    return {
        "path": path,
        "run": start.get("run"),
        "method": start.get("method"),
        "preset": start.get("preset"),
        "pid": start.get("pid"),
        "log_dir": start.get("log_dir"),
        "status": status,
        "error": (end or {}).get("error"),
        "global_step": global_step,
        "epoch": (last_step or {}).get("epoch"),
        "total_steps": total_steps,
        "pct": (
            100.0 * global_step / total_steps
            if (total_steps and global_step is not None)
            else None
        ),
        "elapsed": elapsed,
        "rate": rate,
        "eta": (remaining / rate) if (rate and remaining and remaining > 0) else None,
        "metrics": metrics,
        "val": last_val,
        "ckpt": last_ckpt,
        "warnings": sum(1 for r in events if r.get("ev") == "log"),
    }


@contextmanager
def run_scope(
    sink: Optional[ProgressSink],
    *,
    final_step: Callable[[], int],
    extra_fields: Optional[Callable[[], dict]] = None,
):
    """Emit the matching ``run_end`` when the wrapped training block exits.

    ``run_start`` must already have fired (the sink is constructed earlier so it
    can be handed to the checkpoint saver). On block exit this maps the outcome
    to a status: normal return → ``ok``; ``KeyboardInterrupt`` → ``stopped``;
    any other exception → ``error`` (re-raised either way). ``final_step`` is
    read lazily at exit so the event records where training actually stopped;
    ``extra_fields`` likewise — its dict (e.g. the liveness summary) is merged
    into the ``run_end`` event, and a failure inside it is swallowed so it can
    neither crash a clean exit nor mask the real exception.
    A ``None`` sink makes this a transparent pass-through.
    """

    def _extra() -> dict:
        if extra_fields is None:
            return {}
        try:
            fields = dict(extra_fields() or {})
            for reserved in ("ev", "ts", "status", "final_step", "error"):
                fields.pop(reserved, None)
            return fields
        except Exception as exc:
            logger.debug("run_end extra_fields failed: %s", exc)
            return {}

    if sink is None:
        yield
        return
    try:
        yield
    except KeyboardInterrupt:
        sink.run_end(status="stopped", final_step=final_step(), **_extra())
        raise
    except BaseException as exc:
        sink.run_end(
            status="error",
            final_step=final_step(),
            error=f"{type(exc).__name__}: {exc}",
            **_extra(),
        )
        raise
    else:
        sink.run_end(status="ok", final_step=final_step(), **_extra())
