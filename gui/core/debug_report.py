"""Build a copy-pasteable diagnostic bundle for bug reports.

Qt-free on purpose (only stdlib + ``anima_daemon.config`` paths) so it stays
unit-testable and can't drag torch into the GUI. The Settings dialog's "Copy
debug report" button calls :func:`build_debug_report`; users paste the result
into an issue / chat.

The bundle is tuned for the one failure that's invisible from the GUI: a job
that sits ``queued`` forever because the daemon worker wedged or died. That
state shows only as a spinner with no error in the UI — but the cause is always
on disk (``daemon.log`` holds a worker traceback; ``job.json`` holds the stuck
state). So we surface, in order of diagnostic value: the daemon pidfiles, the
tail of ``daemon.log``, and the most recent jobs' records + stdout tails.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from anima_daemon import config as _cfg

_DAEMON_LOG_LINES = 2000  # read window for daemon.log before filtering to today
_JOB_STDOUT_LINES = 200  # per-job stdout tail (only for non-done jobs)
_MAX_JOBS = 8  # newest N job dirs


def _tail_lines(path: Path, n: int) -> str:
    """Last ``n`` lines of ``path``, decoded leniently (``""`` if absent)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # 256 bytes/line is a generous average; read a bounded window.
            f.seek(max(0, size - n * 256))
            blob = f.read()
    except OSError:
        return ""
    text = blob.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-n:])


def _today_log_lines(path: Path, n: int, today: str) -> str:
    """Today's ``daemon.log`` records (``YYYY-MM-DD`` prefix == ``today``).

    The log accumulates across many daemon restarts, so a raw tail buries the
    current session under weeks of history. We keep only records stamped today,
    plus untimestamped continuation lines (e.g. traceback bodies) that follow a
    today record so a worker crash stays intact.
    """
    tail = _tail_lines(path, n)
    if not tail:
        return ""
    kept: list[str] = []
    in_today = False
    for line in tail.splitlines():
        stamped = len(line) >= 10 and line[:4].isdigit() and line[4] == "-"
        if stamped:
            in_today = line.startswith(today)
        if in_today:
            kept.append(line)
    return "\n".join(kept)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _version() -> str:
    """Best-effort release string: ``.anima_release.json`` → git → 'unknown'."""
    rel = _read_json(_cfg.ROOT / ".anima_release.json")
    if rel:
        for key in ("tag", "version", "name"):
            v = rel.get(key)
            if v:
                return str(v)
    head = _cfg.ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            sha = (_cfg.ROOT / ".git" / ref.split(" ", 1)[1]).read_text().strip()
            return sha[:10]
        return ref[:10]
    except OSError:
        return "unknown"


def _pidfile_block() -> list[str]:
    out = ["## daemon pidfiles"]
    for label, path in (
        ("in-repo", _cfg.PIDFILE),
        ("global", _cfg.global_pidfile()),
    ):
        data = _read_json(path)
        if data is None:
            out.append(f"- {label} ({path}): absent / unreadable")
        else:
            out.append(f"- {label} ({path}): {json.dumps(data)}")
    return out


def _recent_jobs() -> list[Path]:
    try:
        dirs = [d for d in _cfg.JOBS_DIR.iterdir() if d.is_dir()]
    except OSError:
        return []
    dirs.sort(key=lambda d: d.stat().st_mtime if d.exists() else 0, reverse=True)
    return dirs[:_MAX_JOBS]


def _job_block(job_dir: Path) -> list[str]:
    rec = _read_json(job_dir / "job.json") or {}
    state = rec.get("state", "?")
    summary = (
        f"### job {job_dir.name}  [{state}]  kind={rec.get('kind', '?')}  "
        f"method={rec.get('method', '?')}"
    )
    out = [summary]
    for key in ("error", "status_detail", "submitted_at", "started_at", "ended_at"):
        if rec.get(key) is not None:
            out.append(f"  {key}: {rec[key]}")
    # A done/clean job's stdout is rarely the problem; tail only the suspects.
    if state != "done":
        tail = _tail_lines(job_dir / "stdout.log", _JOB_STDOUT_LINES)
        if tail:
            out.append("  --- stdout.log (tail) ---")
            out.append(tail)
    return out


def build_debug_report() -> str:
    """A single text blob a user can copy into a bug report."""
    lines: list[str] = []
    lines.append("===== Anima GUI debug report =====")
    lines.append(f"generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"version:   {_version()}")
    lines.append(f"platform:  {platform.platform()}")
    lines.append(f"python:    {sys.version.split()[0]} ({sys.executable})")
    lines.append(f"repo root: {_cfg.ROOT}")
    lines.append(f"ANIMA_DEBUG={os.environ.get('ANIMA_DEBUG', '')!r}")
    lines.append("")
    lines.extend(_pidfile_block())
    lines.append("")

    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"## daemon.log (today, {today})")
    log_tail = _today_log_lines(_cfg.DAEMON_LOG, _DAEMON_LOG_LINES, today)
    lines.append(log_tail or "(no daemon.log entries for today)")
    lines.append("")

    jobs = _recent_jobs()
    lines.append(f"## recent jobs (newest {len(jobs)})")
    if not jobs:
        lines.append("(no jobs on disk)")
    for job_dir in jobs:
        lines.append("")
        lines.extend(_job_block(job_dir))

    lines.append("")
    lines.append("===== end debug report =====")
    return "\n".join(lines)
