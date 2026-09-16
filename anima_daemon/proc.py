"""Process control for the daemon — spawn detached, kill trees, prove liveness.

A job is a **process tree** (``train.py`` → dataloader workers, plus an
``accelerate launch`` parent on multi-GPU runs), not one PID, and PIDs get
reused. Every spawn / kill / liveness check goes through psutil so the same
code works on Linux and Windows.

``Popen``-based sibling of ``gui/process.py`` (``QProcess``): same
snapshot-then-terminate-then-kill tree walk.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psutil


def create_time(pid: int) -> Optional[float]:
    """``psutil.Process(pid).create_time()`` or ``None`` if the PID is gone."""
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def is_alive(pid: Optional[int], ct: Optional[float], *, tol: float = 1.0) -> bool:
    """True iff ``pid`` exists *and* its create_time matches ``ct``.

    The create_time check is the only guard against PID reuse. ``tol`` absorbs
    sub-second create_time rounding differences between platforms.
    """
    if pid is None or ct is None:
        return False
    actual = create_time(pid)
    if actual is None:
        return False
    return abs(actual - ct) <= tol


def tree_cpu_seconds(pid: Optional[int]) -> Optional[float]:
    """Total CPU seconds (user+system) burned by ``pid`` and every descendant.

    The stall watchdog samples this twice and differences it: a quiet embed/eval
    loop still burns CPU, a wedged process does not. ``None`` when the tree
    can't be read (pid gone / no permission), so the caller falls back to its
    output-mtime-only verdict.
    """
    if pid is None:
        return None
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    total = 0.0
    seen_any = False
    family = [parent]
    try:
        family.extend(parent.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    for p in family:
        try:
            t = p.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        total += float(t.user) + float(t.system)
        seen_any = True
    return total if seen_any else None


def spawn_detached(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Spawn ``cmd`` detached from this process's console, stdout→file.

    Detaching keeps a console ctrl-C from reaching the child:
    ``start_new_session=True`` on POSIX, ``CREATE_NO_WINDOW`` on Windows.

    Windows: ``CREATE_NO_WINDOW``, not ``DETACHED_PROCESS``. A tree with no
    console at all makes every native compiler ``torch.compile`` shells out to
    (``ptxas.exe``, ``cl.exe``) allocate its own visible console window;
    ``CREATE_NO_WINDOW`` gives the tree a hidden console those grandchildren
    inherit, and it is still private, so a terminal CTRL_C can't reach it.
    Stdio has no usable inherited handles, so the redirect to a file is
    mandatory (done on both platforms).

    This flag does not reliably suppress the console of the uv venv
    ``python.exe`` trampoline; windowless callers launch under ``pythonw.exe``
    (``client.venv_python``).
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(stdout_path, "ab", buffering=0)
    kwargs: dict = {
        "cwd": str(cwd),
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kwargs)
    finally:
        log.close()  # the child has dup'd the fd; our handle is done


def kill_tree(pid: int, *, grace_seconds: float = 5.0) -> None:
    """Terminate ``pid`` and every descendant; SIGKILL survivors after grace.

    Snapshots descendants up-front — children of a dying process get reparented
    and would slip past a re-walk. Safe on an already-dead PID and on one we
    have no rights to: every psutil call here swallows ``AccessDenied``,
    including the reap-wait. Do not use ``psutil.wait_procs`` for that wait —
    it lets ``AccessDenied`` escape from ``Process.wait()`` and crashes the
    worker (issue #83). The open-coded wait shares one deadline, so one
    unwaitable member can't abort the reap or multiply the grace period.
    """
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    family = [parent]
    try:
        family.extend(parent.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    for p in family:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    deadline = time.monotonic() + grace_seconds
    alive = []
    for p in family:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except psutil.TimeoutExpired:
            alive.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # gone, or not ours to wait on — either way, don't escalate
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def suspend_tree(pid: int) -> None:
    """SIGSTOP ``pid`` and every descendant — freeze a whole job tree in place.

    Parent **first** so it can't fork a new child into the gap while we walk;
    descendants follow. SIGSTOP on Linux, ``NtSuspendProcess`` on Windows (via
    psutil). The CUDA context and VRAM survive. Pairs with
    :func:`resume_tree`. Safe on an already-dead PID.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        parent.suspend()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for p in children:
        try:
            p.suspend()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def resume_tree(pid: int) -> None:
    """SIGCONT ``pid`` and every descendant — the inverse of :func:`suspend_tree`.

    Children **first**, parent last, so the parent never observes a
    still-frozen child. Safe on an already-dead PID.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for p in children:
        try:
            p.resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        parent.resume()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


# pidfile — single-daemon lock keyed on (pid, create_time)
def write_pidfile(
    path: Path,
    *,
    pid: int,
    port: int,
    root: Optional[Path] = None,
    fingerprint: Optional[str] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ct = create_time(pid)
    data = {"pid": pid, "create_time": ct, "port": port}
    if root is not None:
        data["root"] = str(root)
    if fingerprint is not None:
        # Boot-time source fingerprint, readable without the HTTP port.
        data["fingerprint"] = fingerprint
    path.write_text(json.dumps(data), encoding="utf-8")


def read_pidfile(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def daemon_alive(path: Path) -> Optional[dict]:
    """Return the pidfile dict iff it points at a live daemon, else ``None``.

    A stale pidfile (process gone, or PID reused by a stranger) reads as not
    alive — the caller is then free to take over the port.
    """
    info = read_pidfile(path)
    if not info:
        return None
    if is_alive(info.get("pid"), info.get("create_time")):
        return info
    return None
