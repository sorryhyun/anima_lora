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

    run_dir = make_run_dir("dcw", label="lambda-sweep")
    # ... write artifacts into run_dir ...
    write_result(run_dir, script=__file__, args=args,
                 metrics={...}, artifacts=["per_step.csv", "gap_curves.png"])
"""

from __future__ import annotations

import json
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


def _env_info(device: Any = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
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


def make_run_dir(
    method: str,
    label: str | None = None,
    when: datetime | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    """Create and return ``<root>/<YYYYMMDD-HHMM>[-<label>]/``.

    Default ``root`` is ``<repo>/bench/<method>/results/``. Pass an explicit
    ``root`` (e.g. ``post_image_dataset/dcw``) to redirect — useful for
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
    """Write the standard ``result.json`` envelope into ``run_dir``."""
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
    return out
