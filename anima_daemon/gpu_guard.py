"""Pre-launch GPU guard: free the card, or at least find out it is busy."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

from . import config, gpu, proc
from .jobs import Job

logger = logging.getLogger("anima.daemon")


def wait_for_free_gpu(
    job: Job,
    *,
    jobs: Callable[[], Iterable[Job]],
    on_wait: Callable[[dict], None],
    retries: int = config.GPU_GUARD_RETRIES,
    delay: float = config.GPU_GUARD_DELAY,
    busy_frac: float = config.GPU_GUARD_BUSY_FRAC,
) -> bool:
    """Before launching, make sure the GPU is actually free. False means the
    wait ran out and the card still looks busy; the caller launches anyway,
    since nothing here kills what the daemon didn't start.

    Busy/free is decided from **total VRAM in use**, not the process list
    (on Windows WDDM desktop apps show up as compute processes). Process
    enumeration is used only to reap VRAM leaked by our *own* dead jobs,
    matched by **(pid, create_time)** — a bare pid match could kill a
    stranger that reused the pid (issue #83). If memory can't be probed,
    assume free. Tunable via ANIMA_DAEMON_GPU_{BUSY_FRAC,RETRIES,DELAY}.
    """
    # Ask a resident inference server to free VRAM before we launch — it
    # stays alive and reloads on its next request. Best-effort.
    evict_resident_inference()

    for attempt in range(retries):
        # Reap only holders matching a known job on (pid, create_time).
        holders = gpu.gpu_pids() or set()
        known = {
            j.pid: j
            for j in jobs()
            if j.pid in holders and proc.is_alive(j.pid, j.create_time)
        }
        reaped = False
        for pid, owner in known.items():
            if owner.id == job.id:
                continue
            logger.warning(
                "gpu_guard: reaping leaked VRAM from job %s (pid %s)", owner.id, pid
            )
            proc.kill_tree(pid)
            reaped = True
        if reaped:
            time.sleep(0.5)  # let the killed procs release VRAM

        mem = gpu.gpu_mem()
        if mem is None:  # can't tell → don't deadlock the queue
            return True
        used, total = mem
        if total <= 0 or used / total < busy_frac:
            return True  # GPU effectively free → go
        logger.warning(
            "gpu_guard: GPU busy — %d/%d MiB used (attempt %d/%d)",
            used,
            total,
            attempt + 1,
            retries,
        )
        on_wait(
            {
                "ev": "gpu_wait",
                "job_id": job.id,
                "used_mib": used,
                "total_mib": total,
            }
        )
        time.sleep(delay)
    return False


def evict_resident_inference() -> None:
    """Ask a resident inference server (if any) to free VRAM before launch.

    Discovery mirrors scripts/inference_server.py's pidfiles (inline, no
    import). Every failure is swallowed; the server's idle-TTL frees the
    card eventually anyway.
    """
    candidates = []
    override = os.environ.get("ANIMA_INFERENCE_PIDFILE")
    if override:
        candidates.append(Path(override))
    candidates += [
        config.ROOT / "output" / "inference" / "server.json",
        Path.home() / ".anima" / "inference.json",
    ]
    for pf in candidates:
        try:
            port = json.loads(pf.read_text()).get("port")
        except (OSError, ValueError):
            continue
        if not port:
            continue
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/unload", method="POST"
                ),
                timeout=5,
            ).read()
            logger.info("gpu_guard: inference server (port %s) unloaded", port)
            time.sleep(1.0)  # let VRAM release before we measure
        except Exception:  # noqa: BLE001 — best-effort
            pass
        return
