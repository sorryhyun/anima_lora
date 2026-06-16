"""Shared paths + constants for the local training daemon.

Single localhost process; one job at a time. State lives under
``output/daemon/`` so it sits beside the checkpoints the jobs produce and is
already covered by the repo's ``output/`` gitignore.

    output/daemon/
      daemon.json           pidfile: {"pid", "create_time", "port"}
      daemon.log            the detached daemon's stdout/stderr
      jobs/<job_id>/
        job.json            persisted Job record (survives daemon restart)
        stdout.log          the training subprocess's captured stdout+stderr
        progress.jsonl      the Phase-0 structured progress stream (we point
                            train.py's --progress_jsonl here)
"""

from __future__ import annotations

import os
from pathlib import Path

# scripts/daemon/config.py -> parents[2] == anima_lora/
ROOT = Path(__file__).resolve().parents[2]

STATE_DIR = ROOT / "output" / "daemon"
JOBS_DIR = STATE_DIR / "jobs"
PIDFILE = STATE_DIR / "daemon.json"
DAEMON_LOG = STATE_DIR / "daemon.log"

# Fixed localhost port — the daemon binds 127.0.0.1 only (non-goal: remote /
# auth). Overridable for tests / odd setups via env.
DEFAULT_PORT = int(os.environ.get("ANIMA_DAEMON_PORT", "8765"))
HOST = "127.0.0.1"

# Pre-launch GPU guard (see manager._gpu_guard). Loose by default: this is a
# single-GPU *serial* queue, so the only thing the guard must catch is VRAM
# leaked by our own dead jobs (reaped by pid, independent of the threshold).
# The total-VRAM fraction is just a heuristic for "some *other* process owns the
# card"; keep it high so a loaded ComfyUI / browser / idle desktop doesn't trip
# it and stall every launch. All three are env-tunable for odd setups.
#   busy_frac: treat the card as busy only above this used/total fraction.
#   retries/delay: how long to wait for it to free up before launching anyway.
GPU_GUARD_BUSY_FRAC = float(os.environ.get("ANIMA_DAEMON_GPU_BUSY_FRAC", "0.85"))
GPU_GUARD_RETRIES = int(os.environ.get("ANIMA_DAEMON_GPU_RETRIES", "1"))
GPU_GUARD_DELAY = float(os.environ.get("ANIMA_DAEMON_GPU_DELAY", "2.0"))

# Stall watchdog: kill a *running* job that has emitted no output at all —
# neither stdout.log nor progress.jsonl has advanced — for this many seconds,
# then finalize it as an error naming its last output line. Turns a wedged
# download / symlink-cycle walk / deadlock into a loud, queue-freeing failure
# instead of an indefinite hang. 0 disables a bound (see manager._stall_reason).
#
# Only *command* jobs (preprocess / mask) are watched by default: they front-
# load a model load (~60-90s worst case) then emit a tqdm bar every ~10s, so a
# 120s silence already means wedged. Training is *not* watched by default
# (budget 0): its first step is legitimately silent for the whole torch.compile
# trace — the harness budgets "~30-60s" and warming every token family up front
# can push that to minutes on a slow card — and a false kill mid-compile costs
# far more than the rare genuinely-hung run (which a frozen step makes obvious).
# Opt training in by setting ANIMA_DAEMON_JOB_STALL_TIMEOUT to a generous value.
CMD_STALL_TIMEOUT = float(os.environ.get("ANIMA_DAEMON_CMD_STALL_TIMEOUT", "120"))
JOB_STALL_TIMEOUT = float(os.environ.get("ANIMA_DAEMON_JOB_STALL_TIMEOUT", "0"))


def ensure_state_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def global_pidfile() -> Path:
    """Stable, repo-independent pidfile location the daemon also mirrors to.

    The in-repo ``PIDFILE`` lives under *this checkout's* ``output/daemon/``; a
    ComfyUI node installed in a different directory tree (the published,
    standalone shape — this whole module is vendored into the node) can't
    compute that path. So the daemon mirrors its pidfile here too, at a fixed
    per-user location both sides derive without knowing each other's paths —
    letting the node discover a running daemon, and its possibly-ephemeral
    fallback port, on its own.

    Override with ``$ANIMA_DAEMON_PIDFILE`` (also honored by
    ``discover_pidfile``).
    """
    override = os.environ.get("ANIMA_DAEMON_PIDFILE")
    if override:
        return Path(override)
    return Path.home() / ".anima" / "daemon.json"


def discover_pidfile() -> Path:
    """Locate a running daemon's pidfile across install shapes.

    First existing candidate wins:

      1. the in-repo ``PIDFILE`` — authoritative when this module runs inside
         the anima_lora checkout (``parents[2]`` is the repo root);
      2. ``global_pidfile()`` — the per-user mirror (covers the
         ``$ANIMA_DAEMON_PIDFILE`` override and the standalone-node case where
         ``parents[2]`` points at the vendor dir and ``PIDFILE`` never exists);
      3. ``$ANIMA_LORA_ROOT/output/daemon/daemon.json`` — an explicit repo root;
      4. an upward search from this file for a repo root (``train.py`` +
         ``configs/``), then its ``output/daemon/daemon.json``.

    Falls back to ``PIDFILE`` (nonexistent → the client uses ``DEFAULT_PORT`` /
    ``$ANIMA_DAEMON_PORT``) when nothing is found.
    """
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
