"""Stall watchdog: when has a running job's output frozen for good?

Past the output-freeze budget, a job is killed only if its process tree has also
gone CPU-idle, so a quiet-but-computing embed/eval loop survives — up to
``budget × _STALL_CPU_GRACE``, after which it's killed regardless so a
busy-spinning deadlock can't park the queue forever. Per-kind budgets are in
``config.py``.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from . import config, proc
from .jobs import Job

_CPU_SAMPLE_MIN_GAP = 5.0  # seconds between samples (a short gap measures noise)
_CPU_BUSY_FRAC = 0.05  # ≥5% of one core, averaged over the gap → "computing"
_STALL_CPU_GRACE = 8.0  # hard ceiling multiplier on the configured budget


class StallWatchdog:
    def __init__(self) -> None:
        # job_id → (wall, cpu_seconds, verdict).
        self._cpu_samples: dict[str, tuple[float, float, bool]] = {}

    def forget(self, job_id: str) -> None:
        self._cpu_samples.pop(job_id, None)

    def check(self, job: Job) -> Optional[str]:
        """If the job has produced no output (stdout.log or progress.jsonl
        mtime) for longer than its stall budget, return an actionable error
        naming where it wedged; otherwise ``None``."""
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
            self.forget(job.id)  # output flowing again → forget samples
            return None
        busy = self._tree_cpu_advancing(job)
        if busy and idle < timeout * _STALL_CPU_GRACE:
            return None
        where = _last_output_line(job)
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

    def _tree_cpu_advancing(self, job: Job) -> bool:
        """True iff the job's process tree has burned CPU since the last
        sample. Needs two samples ``_CPU_SAMPLE_MIN_GAP`` apart for a
        meaningful rate; the first look (or one inside the gap) reuses the
        previous verdict, starting optimistic. Unreadable CPU times (pid gone)
        return False, falling back to output-mtime-only behavior."""
        cpu = proc.tree_cpu_seconds(getattr(job, "pid", None))
        if cpu is None:
            return False
        now = time.time()
        prev = self._cpu_samples.get(job.id)
        if prev is None:
            self._cpu_samples[job.id] = (now, cpu, True)
            return True
        gap = now - prev[0]
        if gap < _CPU_SAMPLE_MIN_GAP:
            return prev[2]
        verdict = (cpu - prev[1]) / gap >= _CPU_BUSY_FRAC
        self._cpu_samples[job.id] = (now, cpu, verdict)
        return verdict


def _last_output_line(job: Job, *, max_bytes: int = 8192) -> Optional[str]:
    """Best-effort last non-empty stdout line (carriage-return aware, so a
    tqdm bar's latest redraw wins), for the stall error."""
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
    parts = [p.strip() for p in re.split(r"[\r\n]", blob.decode("utf-8", "replace"))]
    parts = [p for p in parts if p]
    return parts[-1] if parts else None
