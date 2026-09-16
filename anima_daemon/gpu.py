"""Best-effort GPU-occupancy probe for the serial dequeue guard.

Before launching the next job the manager asks who holds the GPU, to tell
"free" from "a known dead job leaked VRAM, reap it" from "an unknown process
is using the card, leave it alone".

pynvml first, ``nvidia-smi`` fallback; if neither is available (e.g. CPU-only
CI) the probes return ``None`` and the guard assumes the GPU is free.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Optional


def no_window_kwargs() -> dict:
    """``subprocess`` kwargs that suppress the Windows console-window flash.

    Inlined rather than imported from ``library.runtime.proc``: the daemon
    package must not import ``library`` / ``networks`` / ``torch``. Returns
    ``{"creationflags": CREATE_NO_WINDOW}`` on Windows, ``{}`` elsewhere.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def gpu_pids() -> Optional[set[int]]:
    """PIDs with a **compute** context on any visible GPU.

    ``None`` means "couldn't tell" (no NVML, no nvidia-smi) — distinct from an
    empty set, which means "queried successfully, no compute job is running".

    Compute contexts only: graphics contexts (desktop compositor, browsers)
    must not count, or on Windows WDDM the guard would stall every launch on
    innocent renderers.
    """
    pids = _gpu_pids_nvml()
    if pids is not None:
        return pids
    return _gpu_pids_smi()


def gpu_mem() -> Optional[tuple[int, int]]:
    """``(used_mib, total_mib)`` summed over visible GPUs, or ``None``.

    The busy/free signal the guard uses: on Windows WDDM per-process
    enumeration is unreliable but aggregate memory is accurate, and a training
    run holds GBs where an idle desktop holds a few hundred MiB.
    """
    mem = _gpu_mem_nvml()
    if mem is not None:
        return mem
    return _gpu_mem_smi()


def _gpu_mem_nvml() -> Optional[tuple[int, int]]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        used = total = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
            used += int(info.used)
            total += int(info.total)
        return (used // (1024 * 1024), total // (1024 * 1024))
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_mem_smi() -> Optional[tuple[int, int]]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    used = total = 0
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            used += int(parts[0])
            total += int(parts[1])
    return (used, total) if total > 0 else None


def _gpu_pids_nvml() -> Optional[set[int]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        out: set[int] = set()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Compute contexts only, mirroring nvidia-smi --query-compute-apps.
            try:
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                    out.add(int(proc.pid))
            except Exception:
                continue
        return out
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_pids_smi() -> Optional[set[int]]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [
                smi,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    pids: set[int] = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids
