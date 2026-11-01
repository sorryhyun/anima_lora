"""Job states and the read side of the on-disk ``job.json`` record.

The state vocabulary is shared by every daemon client, so it lives here rather
than beside ``jobs.Job``: the ComfyUI node vendors this module with
``client.py``, and the GUI and CLI read records while the daemon is down.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from . import config

RECORD_NAME = "job.json"

# queued → running → {done | error | stopped}; running ↔ paused (tree-freeze).
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"
TERMINAL_STATES = frozenset({STATE_DONE, STATE_ERROR, STATE_STOPPED})
# The job holds its VRAM slot and blocks the queue in both states — the worker
# stays parked in the monitor loop for it, running or frozen. A release-paused
# job (`Job.released`) is the exception: it is `paused` with no process.
ACTIVE_STATES = frozenset({STATE_RUNNING, STATE_PAUSED})
# Not over yet: it can still produce output, and it can still be stopped.
UNFINISHED_STATES = frozenset({STATE_QUEUED}) | ACTIVE_STATES
# On the card or on its way there; a paused job is neither.
RUNNABLE_STATES = frozenset({STATE_QUEUED, STATE_RUNNING})
ALL_STATES = TERMINAL_STATES | UNFINISHED_STATES


def read_record(job_dir: Path) -> Optional[dict]:
    """The ``job.json`` under ``job_dir``, or ``None`` if missing or unreadable."""
    try:
        data = json.loads((job_dir / RECORD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def iter_records() -> Iterator[dict]:
    """Every readable record under ``JOBS_DIR``, in job-id (submit) order."""
    try:
        dirs = sorted(config.JOBS_DIR.iterdir())
    except OSError:
        return
    for d in dirs:
        rec = read_record(d)
        if rec is not None and rec.get("id"):
            yield rec
