"""The ``Job`` record + its on-disk persistence.

One JSON file per job under ``output/daemon/jobs/<id>/job.json``. The
in-memory table is authoritative while running; ``persist()`` mirrors each
state change to disk, and ``load_all()`` rebuilds it on boot. See
``anima_daemon/README.md`` for the job lifecycle and retention policy.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config

# queued → running → {done | error | stopped}; running ↔ paused (tree-freeze).
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"
TERMINAL_STATES = frozenset({STATE_DONE, STATE_ERROR, STATE_STOPPED})
# The job holds its VRAM slot and blocks the queue in both states — the worker
# stays parked in the monitor loop for it, running or frozen.
ACTIVE_STATES = frozenset({STATE_RUNNING, STATE_PAUSED})


def new_job_id() -> str:
    """Sortable, collision-resistant id: ``YYYYmmdd-HHMMSS-<6hex>``."""
    import secrets

    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


@dataclass
class Job:
    id: str
    method: str
    preset: str
    methods_subdir: Optional[str] = None
    config_file: Optional[str] = None
    overrides: dict = field(default_factory=dict)
    extra: list[str] = field(default_factory=list)

    # "train" (a train.py run) or "command" (plain ``python <argv>``, finalized
    # on exit code with no progress.jsonl). Default "train" keeps legacy
    # job.json records without this field loading.
    kind: str = "train"
    argv: list[str] = field(default_factory=list)
    extra_env: dict = field(default_factory=dict)

    # Whitelisted env snapshot of the submitter's shell (config.capture_env),
    # layered daemon-env ← captured_env ← extra_env at spawn. See README.
    captured_env: dict = field(default_factory=dict)

    # Auto-chain: a command job carrying a chain_train spec (README "Two job
    # kinds") makes the manager enqueue that train job on success.
    # chained_job_id records the follow-on it spawned.
    chain_train: Optional[dict] = None
    chained_job_id: Optional[str] = None

    # Per-job stall-watchdog budget in seconds, overriding the per-kind default.
    # 0 disables the watchdog for this job; None means "use the default".
    stall_timeout: Optional[float] = None

    # Set on a chain-spawned follow-on train job to skip the pre-launch GPU
    # guard: the daemon just ran the preceding step on this same serial queue,
    # so guarding here would only race that step's not-yet-released VRAM.
    from_chain: bool = False

    state: str = STATE_QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    # Wall-clock of the last running->paused tree-freeze; cleared on resume.
    # Observability only.
    paused_at: Optional[float] = None

    # Release-pause (README "pause with release_model"): `release_requested`
    # is set while train.py is asked to save + exit; `released` marks a paused
    # job with no process (GPU free, queue advances past it); `resume_state_dir`
    # is what `resume` appends as `--resume` on relaunch; `resume_count` counts
    # relaunches.
    release_requested: bool = False
    released: bool = False
    resume_state_dir: Optional[str] = None
    resume_count: int = 0

    # Whether the train tree was spawned under `accelerate launch` (multi-GPU).
    # pause_job refuses these: a frozen NCCL rank trips the collective heartbeat
    # timeout. Command jobs and the single-GPU direct-invoke path are False.
    accelerate_launched: bool = False

    # (pid, create_time) identifies the spawned process so a reused PID is
    # never mistaken for our job (see manager._gpu_guard).
    pid: Optional[int] = None
    create_time: Optional[float] = None

    progress_path: Optional[str] = None
    stdout_path: Optional[str] = None
    ckpt_path: Optional[str] = None

    # OS exit code, mirrored on terminal transition. None until the process
    # exits, or for an adopted orphan (psutil liveness gives no code).
    returncode: Optional[int] = None

    # Result-envelope lift (README "Result envelopes"): a GPU job that
    # writes a bench envelope drops a `result_path.json` pointer; the monitor
    # follows it and records the absolute path here plus a {label, metrics}
    # digest in result_summary. Both None for a job that wrote no envelope.
    result_path: Optional[str] = None
    result_summary: Optional[dict] = None

    error: Optional[str] = None
    # Free-text hint for terminal states ("orphaned", "gpu_held_by_unknown", …).
    status_detail: Optional[str] = None
    # Set by an explicit stop request so the monitor records `stopped`, not
    # `error`, when it sees the process vanish. Runtime + persisted.
    stop_requested: bool = False

    @property
    def dir(self) -> Path:
        return config.job_dir(self.id)

    def persist(self) -> None:
        d = self.dir
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(d / "job.json")  # atomic on POSIX + Windows

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def public(self) -> dict:
        """The dict shape returned over HTTP (all fields, stable order)."""
        return asdict(self)


def load_all() -> dict[str, Job]:
    """Rebuild the job table from ``jobs/*/job.json`` (boot reconciliation)."""
    out: dict[str, Job] = {}
    if not config.JOBS_DIR.exists():
        return out
    for d in sorted(config.JOBS_DIR.iterdir()):
        f = d / "job.json"
        if not f.is_file():
            continue
        try:
            out[d.name] = Job.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
    return out


def _dir_size(d: Path) -> int:
    """Bytes under ``d`` (best-effort — a file that vanishes mid-walk is 0)."""
    total = 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def prune_jobs(
    *,
    max_age_days: Optional[float] = None,
    keep_recent: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """Delete the job dirs of terminal jobs older than ``max_age_days``.

    Called once at boot, **before** ``load_all()`` — a pruned job must never
    enter the in-memory table, or a later ``persist()`` would resurrect the
    dir. Also exposed as ``python -m anima_daemon prune`` (dry-run by default).

    A dir is a candidate only when its ``job.json`` parses with a terminal
    state (queued/running/paused and unreadable records are always kept), it's
    older than ``max_age_days`` (by ``ended_at`` / ``submitted_at`` / mtime),
    and it's not among the ``keep_recent`` newest terminal jobs.
    ``max_age_days=0`` disables pruning. Returns
    ``{scanned, pruned: [ids], kept, freed_bytes, errors}``; never raises.
    """
    days = config.JOB_RETENTION_DAYS if max_age_days is None else max_age_days
    keep = config.JOB_RETENTION_KEEP if keep_recent is None else keep_recent
    summary: dict = {
        "scanned": 0,
        "pruned": [],
        "kept": 0,
        "freed_bytes": 0,
        "errors": 0,
        "max_age_days": days,
        "keep_recent": keep,
        "dry_run": dry_run,
    }
    if days <= 0 or not config.JOBS_DIR.exists():
        return summary

    now = time.time()
    cutoff = now - days * 86400.0
    candidates: list[tuple[float, Path]] = []  # (age-key timestamp, dir)
    for d in sorted(config.JOBS_DIR.iterdir()):
        f = d / "job.json"
        if not f.is_file():
            continue
        summary["scanned"] += 1
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("state") not in TERMINAL_STATES:
            continue
        ts = data.get("ended_at") or data.get("submitted_at")
        if not isinstance(ts, (int, float)):
            try:
                ts = f.stat().st_mtime
            except OSError:
                continue
        candidates.append((float(ts), d))

    # Newest first, then drop the protected head (terminal jobs only).
    candidates.sort(key=lambda t: t[0], reverse=True)
    for ts, d in candidates[keep:] if keep > 0 else candidates:
        if ts >= cutoff:
            continue
        size = _dir_size(d)
        if not dry_run:
            try:
                shutil.rmtree(d)
            except OSError:
                summary["errors"] += 1
                continue
        summary["pruned"].append(d.name)
        summary["freed_bytes"] += size
    summary["kept"] = len(candidates) - len(summary["pruned"])
    return summary
