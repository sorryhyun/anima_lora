"""Process-tree control helpers for the GUI.

QProcess.kill() only signals the immediate child. The GUI launches wrappers
like ``accelerate launch ... train.py`` and ``python tasks.py <task>``; the
actual training process is a grandchild that holds VRAM. Killing the wrapper
alone leaves it orphaned. Two helpers fix that:

* ``setup_kill_safe`` — on Unix, starts the child in a fresh session so the
  whole launcher subtree shares one process group.
* ``kill_process_tree`` — walks descendants via psutil, SIGTERMs them, then
  SIGKILLs anything still alive after a short grace period.
"""

from __future__ import annotations

import sys

import psutil
from PySide6.QtCore import QProcess


def setup_kill_safe(proc: QProcess) -> None:
    """Configure ``proc`` so its child can be killed as a tree.

    On Unix, the child is started as a session leader (``setsid()``) which
    makes it the head of a new process group — handy for ``os.killpg`` style
    teardown, though we let psutil handle the walk itself. On Windows this
    is a no-op; psutil's tree walk works there without extra setup.
    """
    if sys.platform == "win32":
        return
    params = QProcess.UnixProcessParameters()
    params.flags = QProcess.UnixProcessFlag.CreateNewSession
    proc.setUnixProcessParameters(params)


def kill_process_tree(proc: QProcess, *, grace_seconds: float = 3.0) -> None:
    """Terminate ``proc`` and every descendant.

    Sends SIGTERM (terminate on Windows) to the whole tree, waits up to
    ``grace_seconds`` for clean shutdown, then SIGKILLs the survivors. Safe
    to call when the process is already gone.
    """
    if proc.state() == QProcess.NotRunning:
        return

    pid = int(proc.processId())
    if pid <= 0:
        proc.kill()
        return

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Snapshot descendants up-front: children of dying processes can get
    # reparented and slip past a re-walk.
    family = [parent]
    try:
        family.extend(parent.children(recursive=True))
    except psutil.NoSuchProcess:
        pass

    for p in family:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(family, timeout=grace_seconds)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
