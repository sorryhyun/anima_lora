"""The training runs' log: a JSONL event stream and the console lines.

Every trainer (Anima, MOSS, the pixel heads, turbo) reports through one
:class:`ProgressSink`; none prints its own step, eval or checkpoint lines.

The stream sits next to the checkpoint (or in the daemon job dir) and the GUI,
the daemon and MCP clients tail it. Append-only, line-buffered, main-process
only. One event per line:

    {"ev": "run_start", "ts": 0.0, "run": ..., "method": ..., "preset": ...,
     "total_steps": ..., "total_epochs": ..., "pid": ..., "config": {...}}
    {"ev": "step", "ts": ..., "global_step": ..., "epoch": ..., "loss": ..., ...,
     "vram_gb": ..., "vram_peak_gb": ...}
    {"ev": "val",  "ts": ..., "global_step": ..., "epoch": ..., "cmmd": ..., ...}
    {"ev": "ckpt", "ts": ..., "global_step": ..., "path": ...}
    {"ev": "log",  "ts": ..., "level": "WARNING|ERROR|...", "logger": ...,
     "msg": ...}
    {"ev": "run_end", "ts": ..., "status": "ok|error|stopped|paused",
     "final_step": ..., "error": ...}

``log`` events mirror WARNING+ root-logger records
(:meth:`ProgressSink.attach_log_mirror`). Missing file = not started; last line
``run_end`` = done.

The console side: the settings and a start line at ``run_start``, a step line
at most every ``CONSOLE_EVERY_S`` (speed and ETA over the trailing
``RATE_WINDOW_S``, VRAM), every eval and checkpoint, the end line, and a
heartbeat after ``QUIET_S`` of silence (load, compile, an eval). Every write is wrapped so a logging failure cannot crash
training. :func:`read_status` is the read side; ``make run-status``
(``scripts/run_status.py``) is its CLI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
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
    """Keep only scalar entries (0-dim tensors unwrapped) from a ``logs`` dict."""
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


def _scalars(metrics: dict) -> dict:
    """:func:`_flatten_logs` without the keys an event sets itself (a wandb
    ``logs`` dict carries ``global_step`` / ``epoch`` / ``val_step``)."""
    reserved = ("ev", "ts", "global_step", "epoch", "val_step")
    return {k: v for k, v in _flatten_logs(metrics).items() if k not in reserved}


def _find_cmmd(logs: dict) -> Optional[float]:
    """The ``..._cmmd`` value from a validation ``logs`` dict, if present."""
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

    Capped at ``max_events`` so a per-step warning can't bloat the stream; the
    rest still reach the console handlers.
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


# Console cadence: one step line per CONSOLE_EVERY_S at most, a heartbeat after
# QUIET_S without a line, and speed / ETA over the trailing RATE_WINDOW_S (the
# run average carries the load and compile time).
CONSOLE_EVERY_S = 30.0
QUIET_S = 120.0
RATE_WINDOW_S = 300.0


def trailing_rate(
    samples: list[tuple[float, int]], window_s: float = RATE_WINDOW_S
) -> Optional[float]:
    """Steps/sec over the ``(time, step)`` samples that fall in the last
    ``window_s`` (at least the last two samples); ``None`` before two land."""
    if len(samples) < 2:
        return None
    t_end, s_end = samples[-1]
    i = len(samples) - 2
    while i > 0 and t_end - samples[i - 1][0] <= window_s:
        i -= 1
    t_0, s_0 = samples[i]
    if s_end > s_0 and t_end > t_0:
        return (s_end - s_0) / (t_end - t_0)
    return None


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _vram() -> dict[str, float]:
    """Allocated and peak-allocated GB on the current CUDA device, or ``{}``
    where torch or CUDA is not up yet (never initialises CUDA itself)."""
    import sys

    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return {}
        return {
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
    except Exception:
        return {}


def _config_lines(config: dict, width: int = 160) -> list[str]:
    """``key=value`` pairs packed into lines of at most ``width`` columns,
    in the dict's order; ``None`` values are left out."""
    lines: list[str] = []
    cur = ""
    for key, value in config.items():
        if value is None:
            continue
        item = f"{key}={value}"
        if cur and len(cur) + 2 + len(item) > width:
            lines.append(cur)
            cur = item
        else:
            cur = f"{cur}  {item}" if cur else item
    if cur:
        lines.append(cur)
    return lines


class ProgressSink:
    """The run's log: each event is one ``progress.jsonl`` line and, at the
    console cadence, one stdout line. Construct on the main process only.

    ``path=None`` keeps the console side alone. ``unit`` names a step on the
    console (``rollout``, ``step``); ``console_keys`` picks the metrics a
    console line shows (a key ending in ``/`` matches its prefix), all scalars
    when ``None``.
    """

    def __init__(
        self,
        path: Optional[str],
        *,
        run: str,
        method: Optional[str],
        preset: Optional[str],
        t0: Optional[float] = None,
        unit: str = "step",
        console: bool = True,
        console_keys: Optional[tuple[str, ...]] = None,
    ) -> None:
        self._path = path
        self._run = run
        self._method = method
        self._preset = preset
        self._t0 = t0 if t0 is not None else time.time()
        self._fh = None
        self._closed = False
        self._log_handler: Optional[_SinkLogHandler] = None
        self.unit = unit
        self._console = console
        self._console_keys = console_keys
        self._total: Optional[int] = None
        self._samples: list[tuple[float, int]] = []
        self._last_step: Optional[int] = None
        self._last_print = time.time()
        self._last_step_print = 0.0
        self._stop = threading.Event()

    @staticmethod
    def resolve_path(args) -> Optional[str]:
        """Resolve the JSONL path from args, or ``None`` to disable.

        ``--progress_jsonl`` unset → ``<output_dir>/../logs/<output_name>.progress.jsonl``
        (keeps the checkpoint dir to model artifacts); empty / ``none`` /
        ``off`` → disabled; else that path. The daemon always passes an
        explicit per-job path.
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
        # output_dir with no parent component: logs/ goes inside it
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

    def _print(self, line: str) -> None:
        if not self._console or self._closed:
            return
        try:
            print(line, flush=True)
        except Exception:
            pass
        self._last_print = time.time()

    def _heartbeat(self) -> None:
        """Prints a line after ``QUIET_S`` without one: load, compile, an eval
        or a save would otherwise leave the log silent."""
        while not self._stop.wait(5.0):
            quiet = time.time() - self._last_print
            if quiet < QUIET_S:
                continue
            since = fmt_duration(time.time() - self._t0)
            if self._last_step is None:
                self._print(
                    f"… waiting for the first {self.unit} ({since} since start)"
                )
            else:
                self._print(
                    f"… {self.unit} {self._last_step}: no line for "
                    f"{fmt_duration(quiet)} ({since} since start)"
                )

    # region lifecycle events

    def run_start(
        self,
        *,
        total_steps: int,
        total_epochs: int,
        pid: int,
        log_dir: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Open the file fresh (truncating any stale stream), write the opening
        event and print the run's settings. ``config`` is the resolved settings
        the run uses, every key, not the changed ones."""
        if self._closed:
            return
        self._total = total_steps
        if self._path:
            try:
                parent = os.path.dirname(self._path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                # line-buffered so tailing readers see each event immediately
                self._fh = open(self._path, "w", buffering=1, encoding="utf-8")
            except Exception as exc:
                logger.debug("progress sink open failed: %s", exc)
                self._fh = None
        extra: dict[str, Any] = {"log_dir": log_dir} if log_dir is not None else {}
        if config is not None:
            extra["config"] = config
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
        if config is not None:
            self._print(f"settings ({len(config)}):")
            for line in _config_lines(config):
                self._print(f"  {line}")
        self._print(
            f"start {self._run} [{self._method or '-'}]: {total_steps} {self.unit}s"
            + (f", progress {self._path}" if self._fh is not None else "")
        )
        if self._console:
            threading.Thread(
                target=self._heartbeat, name="progress-heartbeat", daemon=True
            ).start()

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
        peak = _vram().get("vram_peak_gb")
        self._print(
            f"end {status}: {self.unit} {final_step}/{self._total or '?'} in "
            f"{fmt_duration(time.time() - self._t0)}"
            + (f", peak vram {peak:.1f}GB" if peak is not None else "")
            + (f" — {error}" if error else "")
        )
        self.close()

    # endregion

    def step(self, metrics: dict, *, global_step: int, epoch: int = 0) -> None:
        """A ``step`` event with ``metrics``' scalars and the VRAM; a console
        line on the first call, then at most every ``CONSOLE_EVERY_S`` and at
        the last step."""
        now = time.time()
        first = self._last_step is None
        self._last_step = global_step
        self._samples.append((now, global_step))
        while len(self._samples) > 2 and now - self._samples[1][0] > RATE_WINDOW_S:
            self._samples.pop(0)
        fields = _scalars(metrics)
        vram = _vram()
        self._emit("step", **fields, **vram, global_step=global_step, epoch=epoch)
        last = self._total is not None and global_step >= self._total
        if not (first or last or now - self._last_step_print >= CONSOLE_EVERY_S):
            return
        self._last_step_print = now
        self._print(self._step_line(fields, global_step, vram, now))

    def _step_line(self, fields: dict, global_step: int, vram: dict, now: float) -> str:
        total = self._total
        head = f"{self.unit} {global_step}/{total or '?'}"
        if total:
            head += f" ({100.0 * global_step / total:.1f}%)"
        keys = self._console_keys
        shown = [
            f"{k} {_fmt_value(v)}"
            for k, v in fields.items()
            if keys is None
            or any(k == c or (c.endswith("/") and k.startswith(c)) for c in keys)
        ]
        rate = trailing_rate(self._samples)
        if rate is None:
            speed = f"first {self.unit} at {fmt_duration(now - self._t0)}"
        else:
            per = (
                f"{1 / rate:.2f} s/{self.unit}"
                if rate < 1
                else f"{rate:.2f} {self.unit}/s"
            )
            eta = (total - global_step) / rate if total else None
            speed = f"{per}  eta {fmt_duration(eta)}"
        parts = [head, "  ".join(shown), speed]
        if vram:
            parts.append(f"vram {vram['vram_gb']:.1f}/{vram['vram_peak_gb']:.1f}GB")
        parts.append(fmt_duration(now - self._t0))
        return "  |  ".join(p for p in parts if p)

    def val(
        self,
        metrics: dict,
        *,
        global_step: int,
        epoch: int = 0,
        val_step: Optional[int] = None,
        text: Optional[str] = None,
    ) -> None:
        """A ``val`` event with ``metrics``' scalars (``cmmd`` lifted from a
        ``..._cmmd`` key) and a console line: ``text``, else the scalars."""
        fields = _scalars(metrics)
        cmmd = _find_cmmd(metrics)
        if cmmd is not None:
            fields["cmmd"] = cmmd
        if val_step is not None:
            fields["val_step"] = val_step
        self._emit("val", **fields, global_step=global_step, epoch=epoch)
        body = text or "  ".join(f"{k} {_fmt_value(v)}" for k, v in fields.items())
        self._print(f"eval @ {self.unit} {global_step}: {body}")

    def log(
        self,
        logs: dict,
        *,
        global_step: int,
        epoch: int,
        val_step: Optional[int] = None,
    ) -> None:
        """A tracker-style ``logs`` dict: one carrying a ``..._cmmd`` key (or an
        explicit ``val_step``) is a validation pass → :meth:`val`; everything
        else → :meth:`step`."""
        if val_step is not None or _find_cmmd(logs) is not None:
            self.val(logs, global_step=global_step, epoch=epoch, val_step=val_step)
        else:
            self.step(logs, global_step=global_step, epoch=epoch)

    def ckpt(self, *, global_step: int, path: str) -> None:
        self._emit("ckpt", global_step=global_step, path=path)
        self._print(f"saved {path} @ {self.unit} {global_step}")

    def attach_log_mirror(self, *, max_events: int = 500) -> None:
        """Mirror WARNING+ records from the root logger as ``log`` events.

        Call after :meth:`run_start` (no-op before). Detached on :meth:`close`.
        """
        if self._fh is None or self._closed or self._log_handler is not None:
            return
        self._log_handler = _SinkLogHandler(self, max_events=max_events)
        logging.getLogger().addHandler(self._log_handler)

    def close(self) -> None:
        self._stop.set()
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


def command_job_path(fallback: Optional[str]) -> Optional[str]:
    """The stream path for a loop the daemon runs as a command job: the job
    dir's ``progress.jsonl`` (``ANIMA_DAEMON_JOB_DIR``, where the daemon and
    ``make run-status`` read it) under the daemon, else ``fallback``."""
    job_dir = os.environ.get("ANIMA_DAEMON_JOB_DIR")
    return os.path.join(job_dir, "progress.jsonl") if job_dir else fallback


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


def read_status(path: str, *, window_s: float = RATE_WINDOW_S) -> dict:
    """Digest a ``progress.jsonl`` stream into one run-status dict.

    Reads the whole file; torn lines are skipped, so it is safe on a live run.

    Keys: ``run``/``method``/``preset``/``pid``/``log_dir`` (from ``run_start``),
    ``global_step``/``total_steps``/``pct``, ``elapsed``/``rate``/``eta``
    (seconds; ``rate`` is steps/sec over the step events of the last
    ``window_s``, ``None`` before two land), ``metrics`` (last ``step`` event's
    scalars), ``val`` (last ``val`` event), ``vals`` (every ``val`` event, in
    order), ``config`` (the settings ``run_start`` carries), ``ckpt`` (last
    ``ckpt`` event), ``warnings``
    (``log`` event count) and ``status``:

    ``running`` · the ``run_end`` status (``ok`` / ``error`` / ``stopped`` /
    ``paused``) · ``dead`` (no ``run_end`` and the pid is gone: killed / OOMed)
    · ``unknown`` (no ``run_end``, liveness unknowable).
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

    # Trailing window: the run average is skewed by startup (load, compile).
    rate = trailing_rate(
        [(r.get("ts", 0), r.get("global_step", 0)) for r in steps], window_s
    )

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
        if k not in ("ev", "ts", "global_step", "epoch", "vram_gb", "vram_peak_gb")
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
        "vals": [r for r in events if r.get("ev") == "val"],
        "config": start.get("config"),
        "ckpt": last_ckpt,
        "vram_peak_gb": (last_step or {}).get("vram_peak_gb"),
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

    ``run_start`` must already have fired. Normal return → ``ok``;
    ``KeyboardInterrupt`` → ``stopped``; :class:`~library.runtime.pause.TrainingPaused`
    → ``paused`` (with ``state_dir``); any other exception → ``error``;
    exceptions are re-raised. ``final_step`` and ``extra_fields`` are read at
    exit; a failure in ``extra_fields`` is swallowed so it cannot mask the
    real exception. A ``None`` sink is a pass-through.
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
    from library.runtime.pause import TrainingPaused

    try:
        yield
    except KeyboardInterrupt:
        sink.run_end(status="stopped", final_step=final_step(), **_extra())
        raise
    except TrainingPaused as paused:
        sink.run_end(
            status="paused",
            final_step=final_step(),
            state_dir=paused.state_dir,
            **_extra(),
        )
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
