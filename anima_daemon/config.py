"""Shared paths + constants for the local training daemon.

State lives under ``output/daemon/`` (already covered by the repo's
``output/`` gitignore). See ``anima_daemon/README.md`` for the on-disk layout
and job lifecycle.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# anima_daemon/config.py -> parents[1] == anima_lora/
ROOT = Path(__file__).resolve().parents[1]

STATE_DIR = ROOT / "output" / "daemon"
JOBS_DIR = STATE_DIR / "jobs"
PIDFILE = STATE_DIR / "daemon.json"
DAEMON_LOG = STATE_DIR / "daemon.log"

# Shortcuts to the job the daemon is currently driving (the last one launched
# when idle): ``output/daemon/current`` -> ``jobs/<id>/`` and
# ``output/daemon/current.log`` -> that job's ``stdout.log``. Relative symlinks,
# retargeted atomically by the manager at launch/adopt; best-effort on hosts
# without symlink privileges (Windows outside developer mode).
CURRENT_LINK_NAME = "current"
CURRENT_LOG_NAME = "current.log"


def point_current_job(job_id: str) -> None:
    """Retarget the ``current`` / ``current.log`` shortcuts at ``job_id``."""
    state_dir = Path(STATE_DIR)
    targets = {
        CURRENT_LINK_NAME: Path("jobs") / job_id,
        CURRENT_LOG_NAME: Path("jobs") / job_id / "stdout.log",
    }
    for name, target in targets.items():
        link = state_dir / name
        tmp = state_dir / f".{name}.tmp"
        try:
            tmp.unlink(missing_ok=True)
            os.symlink(target, tmp, target_is_directory=name == CURRENT_LINK_NAME)
            os.replace(tmp, link)
        except OSError:
            # No symlink privilege / exotic FS: the shortcut is a convenience,
            # never a contract — job.json + stdout.log stay the source of truth.
            tmp.unlink(missing_ok=True)
            return


# localhost-only by design (no remote, no auth). Port is not guaranteed to be
# 8765 — see discover_pidfile/serve_with_fallback; never hardcode it elsewhere.
DEFAULT_PORT = int(os.environ.get("ANIMA_DAEMON_PORT", "8765"))
HOST = "127.0.0.1"

# Pre-launch GPU guard (manager._gpu_guard): busy_frac = used/total above which
# the card is treated as busy; retries/delay = how long to wait before launching
# anyway. Loose by default so a partially-loaded desktop/ComfyUI doesn't stall
# every launch — VRAM leaked by our own dead jobs is reaped by pid regardless.
GPU_GUARD_BUSY_FRAC = float(os.environ.get("ANIMA_DAEMON_GPU_BUSY_FRAC", "0.85"))
GPU_GUARD_RETRIES = int(os.environ.get("ANIMA_DAEMON_GPU_RETRIES", "1"))
GPU_GUARD_DELAY = float(os.environ.get("ANIMA_DAEMON_GPU_DELAY", "2.0"))

# Stall watchdog (manager._stall_reason): kills a *running* job whose output has
# frozen for this many seconds, so a wedged download/deadlock fails loudly
# instead of parking the queue. Command jobs are watched by default; training is
# not (0 = off) because its first-step torch.compile trace is legitimately
# silent for minutes. A per-job `stall_timeout` on submit overrides both.
CMD_STALL_TIMEOUT = float(os.environ.get("ANIMA_DAEMON_CMD_STALL_TIMEOUT", "120"))
JOB_STALL_TIMEOUT = float(os.environ.get("ANIMA_DAEMON_JOB_STALL_TIMEOUT", "0"))

# Job-dir retention (jobs.prune_jobs): terminal jobs older than RETENTION_DAYS
# are deleted at boot, always keeping the newest RETENTION_KEEP regardless of
# age. Queued/running/paused dirs are never candidates. 0 days disables pruning.
JOB_RETENTION_DAYS = float(os.environ.get("ANIMA_DAEMON_JOB_RETENTION_DAYS", "30"))
JOB_RETENTION_KEEP = int(os.environ.get("ANIMA_DAEMON_JOB_RETENTION_KEEP", "200"))


# Daemon-source fingerprint: a resident daemon serving stale anima_daemon/*.py
# after an edit/pull is untrustworthy, so a client compares this against the
# fingerprint the daemon booted with and eagerly restarts on mismatch (see
# client.daemon_is_stale; "Disposable daemon" in README.md). Keys on file
# contents, not git HEAD, since the common case is a dirty working tree.
_SRC_DIR = Path(__file__).resolve().parent


def source_fingerprint() -> str:
    """Short content hash of ``anima_daemon/*.py``; unreadable files are
    skipped rather than raising (a partial hash still beats crashing boot)."""
    h = hashlib.sha256()
    for path in sorted(_SRC_DIR.glob("*.py")):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()[:16]


# Submit-time env capture: a queued job should run with the SUBMITTER's shell
# env, not the daemon's stale boot env (see README POST /jobs). PATH /
# VIRTUAL_ENV are deliberately NOT captured — the daemon must keep resolving its
# own venv interpreter, which a captured PATH from an odd shell would break.
CAPTURED_ENV_PREFIXES = ("ANIMA_", "CUDA_", "HF_", "PYTORCH_", "TORCH_", "NCCL_")
# ANIMA_DAEMON_* configures the daemon itself, not a job — never capture it.
_CAPTURED_ENV_SKIP_PREFIXES = ("ANIMA_DAEMON_",)


def capture_env(source: dict | None = None) -> dict:
    """Snapshot the whitelisted env vars from the *submitting* process (not the
    daemon's). ``source`` defaults to ``os.environ``."""
    env = os.environ if source is None else source
    return {
        k: v
        for k, v in env.items()
        if k.startswith(CAPTURED_ENV_PREFIXES)
        and not k.startswith(_CAPTURED_ENV_SKIP_PREFIXES)
    }


def ensure_state_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def global_pidfile() -> Path:
    """Stable, repo-independent pidfile mirror, at a fixed per-user location.

    A standalone install (e.g. a vendored ComfyUI node) can't compute the
    in-repo ``PIDFILE`` path, so the daemon mirrors here too — letting it
    discover a running daemon (and its possibly-ephemeral port) without
    knowing the checkout location. Override with ``$ANIMA_DAEMON_PIDFILE``
    (also honored by ``discover_pidfile``).
    """
    override = os.environ.get("ANIMA_DAEMON_PIDFILE")
    if override:
        return Path(override)
    return Path.home() / ".anima" / "daemon.json"


def discover_pidfile() -> Path:
    """Locate a running daemon's pidfile across install shapes: the in-repo
    ``PIDFILE``, then ``global_pidfile()`` (per-user mirror / env override),
    then ``$ANIMA_LORA_ROOT``, then an upward search for a repo root
    (``train.py`` + ``configs/``). Falls back to ``PIDFILE`` (nonexistent →
    the client uses ``DEFAULT_PORT``) when nothing is found."""
    if PIDFILE.exists():
        return PIDFILE

    mirror = global_pidfile()
    if mirror.exists():
        return mirror

    env_root = os.environ.get("ANIMA_LORA_ROOT")
    if env_root:
        cand = Path(env_root) / "output" / "daemon" / "daemon.json"
        if cand.exists():
            return cand

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "train.py").is_file() and (parent / "configs").is_dir():
            cand = parent / "output" / "daemon" / "daemon.json"
            if cand.exists():
                return cand
            break

    return PIDFILE
