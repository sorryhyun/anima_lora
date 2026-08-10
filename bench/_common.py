"""Shared helpers for bench/ scripts.

Goal: one consistent envelope so results from different methods can be
compared, indexed, and re-found. Each script picks its own metrics shape;
this module just standardizes the *outer* record.

Envelope (result.json) — schema_version=1::

    {
      "schema_version": 1,
      "script": "bench/<method>/<script>.py",
      "label": "lambda-sweep",
      "timestamp_utc": "2026-05-03T14:07:23Z",
      "git":  {"sha": "abc1234", "branch": "main", "dirty": false},
      "env":  {"python": "3.13.1", "torch": "2.6.0", "cuda": "12.4",
               "device": "cuda:0", "gpu_name": "RTX 5090"},
      "args": { ...vars(args)... },
      "metrics":   { ...script-specific... },
      "artifacts": ["per_step.csv", "gap_curves.png"]   # relative to run_dir
    }

Layout per run::

    bench/<method>/results/<YYYYMMDD-HHMM>[-<label>]/
        result.json
        <artifacts>

Usage::

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("spectrum", label="delta-sweep")
    # ... write artifacts into run_dir ...
    write_result(run_dir, script=__file__, args=args,
                 metrics={...}, artifacts=["per_step.csv", "curves.png"])

Cache-level data access (issues.md DX4)::

    For raw cache reads without the CachedDataset/blueprint layer, use
    ``library/io/cache.py`` directly:
        load_cached_latents(npz_path)      -> (latents, resolution, orig_h, orig_w)
        load_cached_text_features(te_path) -> (crossattn_emb, pooled)  # None,None if absent
        discover_latents_by_stem(root)     -> {stem: [LatentCacheFile, ...]}  # buckets/stem
    Don't hand-roll the ``next(k for k if k.startswith("latents_"))`` NPZ read —
    ``load_cached_latents`` already does it (and recovers original pixel size).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]  # anima_lora/


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git_info() -> dict[str, Any]:
    dirty = _git("status", "--porcelain")
    return {
        "sha": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty) if dirty is not None else None,
    }


def boot_fingerprint() -> dict[str, Any]:
    """Environment identity for raw-gradient-vector comparability
    (sigma_lowres roadmap §3 T0): vector reads between arm_sums stores
    are licensed only when boot_epoch matches — a reboot re-autotunes
    the kernel path and breaks cross-store vector cosines
    (project_crossboot_arm_store_break). Cache-dir fields are recorded
    for the T2 pinning experiment (a certified pinned cache would key
    comparability on the cache dir instead of the boot)."""
    fp: dict[str, Any] = {"boot_epoch": None, "driver": None}
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                fp["boot_epoch"] = int(line.split()[1])
                break
    except OSError:
        pass
    try:
        fp["driver"] = Path("/sys/module/nvidia/version").read_text().strip()
    except OSError:
        pass
    try:
        import torch

        fp["torch"] = torch.__version__
        fp["cuda"] = torch.version.cuda
    except ImportError:
        pass
    fp["inductor_cache_dir"] = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    fp["triton_cache_dir"] = os.environ.get("TRITON_CACHE_DIR")
    return fp


def _env_info(device: Any = None) -> dict[str, Any]:
    fp = boot_fingerprint()
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "boot_epoch": fp["boot_epoch"],
        "driver": fp["driver"],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if device is not None:
            dev_str = str(device)
            info["device"] = dev_str
            if dev_str.startswith("cuda") and torch.cuda.is_available():
                if dev_str == "cuda":
                    idx = torch.cuda.current_device()
                else:
                    try:
                        idx = int(dev_str.split(":", 1)[1])
                    except (IndexError, ValueError):
                        idx = torch.cuda.current_device()
                info["gpu_name"] = torch.cuda.get_device_name(idx)
    except ImportError:
        pass
    return info


def _serializable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Path):
        return str(v)
    if is_dataclass(v):
        return _serializable(asdict(v))
    if isinstance(v, Mapping):
        return {str(k): _serializable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_serializable(x) for x in v]
    return repr(v)


def start_heartbeat(interval: float = 45.0, *, label: str = "hb") -> None:
    """Print a keep-alive line every ``interval`` seconds (daemon-friendly).

    The daemon's stall watchdog kills a *command* job whose stdout and
    progress.jsonl both freeze for 120 s (``anima_daemon/config.py``), which is
    the normal shape of an embed/eval loop between prints — so every long-quiet
    bench script needs *something*. Two supported ways, in preference order:

    1. submit with a raised/disabled budget — ``make daemon-run
       ARGS="… --stall-timeout 0"`` / ``submit_command(stall_timeout=…)``. The
       daemon also cross-checks process-tree CPU before firing, so a genuinely
       computing job survives its budget.
    2. call this at the top of ``main()`` — one line, no submit-side knowledge
       needed, and the ticks double as a "still alive, N seconds in" log.

    Daemon thread, so it never keeps the process alive; unbuffered so the line
    actually reaches the job's stdout.log (which is what the watchdog stats).
    """
    import threading
    import time

    t0 = time.time()

    def beat() -> None:
        while True:
            time.sleep(interval)
            print(f"[{label}] {time.time() - t0:.0f}s", flush=True)

    threading.Thread(target=beat, daemon=True, name="bench-heartbeat").start()


def make_run_dir(
    method: str,
    label: str | None = None,
    when: datetime | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    """Create and return ``<root>/<YYYYMMDD-HHMM>[-<label>]/``.

    Default ``root`` is ``<repo>/bench/<method>/results/``. Pass an explicit
    ``root`` (e.g. a dir under ``post_image_dataset/``) to redirect — useful for
    calibration data that's logically a cache, not a published-bench artifact.
    """
    ts = (when or datetime.now()).strftime("%Y%m%d-%H%M")
    name = f"{ts}-{label}" if label else ts
    if root is not None:
        base = Path(root)
        if not base.is_absolute():
            base = REPO_ROOT / base
    else:
        base = REPO_ROOT / "bench" / method / "results"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_result(
    run_dir: Path,
    *,
    script: str | Path,
    args: Any,
    metrics: Mapping[str, Any],
    label: str | None = None,
    artifacts: Iterable[str | Path] = (),
    device: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the standard ``result.json`` envelope into ``run_dir``.

    Only ``run_dir`` is positional; everything else is keyword-only (issues.md
    DX5). Call as
    ``write_result(run_dir, script=__file__, args=args, metrics={...})`` — a
    positional ``write_result(run_dir, metrics)`` fails with an opaque
    "takes 1 positional argument" error.
    """
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    script_path = Path(script)
    if script_path.is_absolute():
        try:
            script_str = str(script_path.relative_to(REPO_ROOT))
        except ValueError:
            script_str = str(script_path)
    else:
        script_str = str(script_path)

    artifact_names: list[str] = []
    for a in artifacts:
        a_path = Path(a)
        if a_path.is_absolute():
            try:
                artifact_names.append(str(a_path.relative_to(run_dir)))
            except ValueError:
                artifact_names.append(a_path.name)
        else:
            artifact_names.append(str(a_path))

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "script": script_str,
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": _git_info(),
        "env": _env_info(device),
        "args": _serializable(args_dict),
        "metrics": _serializable(metrics),
        "artifacts": artifact_names,
    }
    if extra:
        record["extra"] = _serializable(extra)
    out = run_dir / "result.json"
    out.write_text(json.dumps(record, indent=2))

    # Daemon result-envelope lift (proposal Phase 1a): when this script runs as a
    # daemon job, the daemon exported ANIMA_DAEMON_JOB_DIR into our env. Drop a
    # pointer to the just-written envelope there so the monitor can lift its
    # abs path + {label, metrics} digest into the job record on the terminal
    # transition. Absent (a plain inline `python bench/.../run_bench.py`) → a
    # no-op, so bench scripts stay standalone with zero daemon coupling.
    job_dir = os.environ.get("ANIMA_DAEMON_JOB_DIR")
    if job_dir:
        try:
            pointer = Path(job_dir) / "result_path.json"
            pointer.write_text(json.dumps({"path": str(out.resolve())}))
        except OSError:
            pass  # best-effort; a missing job dir must never fail the bench run

    return out
