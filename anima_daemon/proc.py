"""Process control for the daemon — spawn detached, kill trees, prove liveness.

Every rule here exists because a training job is a **process tree**
(``accelerate launch → train.py → dataloader workers``), not one PID, and
because PIDs get reused. Route every spawn / kill / liveness check through
psutil so the same code works on Linux and Windows (the daemon must run on
both — ``python tasks.py daemon`` is the Windows alias for ``make daemon``).

This is the ``Popen``-flavored sibling of ``gui/process.py`` (which is
``QProcess``-bound): same snapshot-then-terminate-then-kill tree walk.
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

    The create_time check is the sole defense against PID reuse — without it a
    recycled PID looks like our still-running job. ``tol`` absorbs the
    sub-second rounding difference between platforms' create_time clocks.
    """
    if pid is None or ct is None:
        return False
    actual = create_time(pid)
    if actual is None:
        return False
    return abs(actual - ct) <= tol


def tree_cpu_seconds(pid: Optional[int]) -> Optional[float]:
    """Total CPU seconds (user+system) burned by ``pid`` and every descendant.

    The liveness signal for a job that legitimately writes nothing for minutes:
    an embed/eval loop is *quiet but computing*, while a wedged process (stalled
    socket, deadlock, symlink-cycle walk) burns no CPU. Sampled twice and
    differenced by the stall watchdog. ``None`` when the tree can't be read at
    all (pid gone / no permission), so the caller can fall back to its
    output-mtime-only verdict rather than treating "unknown" as "alive".
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

    Detaching is what lets a console ctrl-C miss the child:
    ``start_new_session=True`` on POSIX (new session/process group, terminal
    SIGINT only reaches the foreground group), ``CREATE_NO_WINDOW`` on Windows.

    Windows console nuance — why ``CREATE_NO_WINDOW`` *without*
    ``DETACHED_PROCESS``: detaching gives the whole training tree **no console
    at all**, so when ``torch.compile``'s inductor/Triton backend shells out to
    native compilers (``ptxas.exe`` per CUDA kernel, ``cl.exe`` for the C++
    wrapper) with no creation flags, Windows sees "parent has no console" and
    allocates a fresh **visible** console for each — a burst of terminal-window
    flashes on every compile-heavy training start. ``CREATE_NO_WINDOW`` instead
    gives the tree a console that *exists but is hidden*; those compiler
    grandchildren inherit it rather than popping their own. CTRL_C isolation is
    preserved regardless: the daemon runs under ``pythonw`` with no console of
    its own, and a ``CREATE_NO_WINDOW`` child gets its own private hidden
    console, so a stray terminal CTRL_C still can't reach it (and we kill jobs
    via ``kill_tree``, not console events). Stdio still has no usable inherited
    handles, so redirecting to a file stays mandatory — we do it on both
    platforms for uniformity.

    Window suppression on Windows is the *interpreter's* job, not a creation
    flag's: the uv venv ``python.exe`` is a trampoline that re-launches the real
    interpreter, so ``CREATE_NO_WINDOW`` set here doesn't reliably reach the
    child's console. Callers that must stay windowless (the long-lived daemon)
    launch under ``pythonw.exe`` instead (see ``client.venv_python``).
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
    and would slip past a re-walk. Safe to call on an already-dead PID, and on
    one we have no rights to: every psutil call here swallows ``AccessDenied``,
    including the reap-wait. ``psutil.wait_procs`` cannot be used for that wait
    because it lets ``AccessDenied`` escape from its inner ``Process.wait()`` —
    which crashed the daemon worker in issue #83 when the guard aimed at a
    system process. We open-code the wait on a shared deadline instead, so one
    unwaitable member can't abort the reap for the rest of the family (nor can a
    large family multiply the grace period).
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

    Parent **first** so it can't fork a new child into the gap while we walk
    (the same "snapshot before you act" reason ``kill_tree`` snapshots up
    front); dataloader workers and any compiler grandchildren follow. On Linux
    this is SIGSTOP, on Windows ``NtSuspendProcess`` — psutil abstracts both.
    The CUDA context and VRAM survive; only SM scheduling stops. Pairs with
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

    Reverse order: children (deepest last-suspended) **first**, parent last, so
    the parent never briefly observes a still-frozen child after it itself has
    unfrozen. Safe on an already-dead PID.
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
        # The daemon-source fingerprint it booted with — disk-observable so a
        # passive reader can flag stale code without the HTTP port (Phase 0a).
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
