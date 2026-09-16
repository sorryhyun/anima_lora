"""Read helpers for the per-job ``progress.jsonl`` + ``stdout.log``.

The daemon never pipes a child's stdout — it tails files, so a re-adopted
orphan it didn't spawn can still be followed. These helpers swallow
exceptions; a missing or half-written line is normal while the trainer appends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional


def read_events(
    progress_path: Optional[str],
    *,
    events: Optional[list[str]] = None,
    since_step: Optional[int] = None,
    every_nth: Optional[int] = None,
    last_n: Optional[int] = None,
) -> list[dict]:
    """Parse + filter a ``progress.jsonl`` stream into a list of event dicts.

    The query surface behind ``GET /jobs/{id}/progress`` and the MCP
    ``get_progress`` tool.

    - ``events``: keep only these ``ev`` kinds (e.g. ``["step", "val"]``).
    - ``since_step``: keep events at/after this ``global_step``. Events that
      carry no ``global_step`` (``run_start``, ``log``, ``run_end``) inherit
      the most recent step seen before them in the stream.
    - ``every_nth``: thin ``step`` events to every n-th (the latest ``step``
      is always kept so the caller sees where the run currently is).
    - ``last_n``: final cap — keep only the trailing n events.
    """
    if not progress_path:
        return []
    p = Path(progress_path)
    if not p.is_file():
        return []
    wanted = set(events) if events else None
    picked: list[dict] = []
    step_seen = 0  # most recent global_step, inherited by step-less events
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # half-written tail line
                if not isinstance(rec, dict):
                    continue
                gs = rec.get("global_step")
                if isinstance(gs, (int, float)):
                    step_seen = int(gs)
                if wanted is not None and rec.get("ev") not in wanted:
                    continue
                if since_step is not None and step_seen < since_step:
                    continue
                picked.append(rec)
    except OSError:
        return []
    if every_nth and every_nth > 1:
        steps = [i for i, rec in enumerate(picked) if rec.get("ev") == "step"]
        keep_steps = set(steps[::every_nth])
        if steps:
            keep_steps.add(steps[-1])
        picked = [
            rec
            for i, rec in enumerate(picked)
            if rec.get("ev") != "step" or i in keep_steps
        ]
    if last_n is not None and last_n >= 0:
        picked = picked[len(picked) - min(last_n, len(picked)) :]
    return picked


def last_event(progress_path: Optional[str]) -> Optional[dict]:
    """Parse the last complete JSON line of ``progress.jsonl`` (or ``None``)."""
    if not progress_path:
        return None
    p = Path(progress_path)
    if not p.is_file():
        return None
    last = None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except ValueError:
                    continue  # half-written tail line
    except OSError:
        return None
    return last


def last_ckpt_path(progress_path: Optional[str]) -> Optional[str]:
    """The ``path`` of the most recent ``ckpt`` event, if any."""
    if not progress_path:
        return None
    p = Path(progress_path)
    if not p.is_file():
        return None
    found: Optional[str] = None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or '"ckpt"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("ev") == "ckpt" and rec.get("path"):
                    found = rec["path"]
    except OSError:
        return None
    return found


def follow(path: Path, *, from_start: bool = True) -> Iterator[str]:
    """Generator yielding lines as they're appended (``tail -f``).

    Yields existing content first when ``from_start`` is set, then blocks for
    new lines. The caller drives cadence (it sleeps between empty reads) and is
    responsible for stopping — this never returns on its own. Windows opens
    files shared-read by default so this works while the trainer writes.
    """
    import time

    while not path.exists():
        yield ""  # let the caller poll / decide to give up
        time.sleep(0.3)
    with open(path, "r", encoding="utf-8") as fh:
        if not from_start:
            fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                yield line
            else:
                yield ""  # heartbeat tick so the caller can check liveness
                time.sleep(0.3)
