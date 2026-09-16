"""Windows-quiet subprocess helpers.

On Windows, ``subprocess`` launches of a *console* program (``git``,
``nvidia-smi``, ``powershell`` …) flash a console window on screen unless
``CREATE_NO_WINDOW`` is passed. The daemon's GPU-occupancy poll and the
per-checkpoint ModelSpec git query fire repeatedly, so without it a terminal
blinks several times whenever a checkpoint is written.

``CREATE_NO_WINDOW`` works for direct console executables like these, but not
through the uv venv ``python.exe`` trampoline re-exec — the daemon's job
launcher (``anima_daemon/proc.py``) uses ``pythonw.exe`` instead.

Usage::

    subprocess.run([...], **no_window_kwargs())

For library code that can't be patched (``torch.compile`` / Triton / Inductor
shelling out to ``ptxas.exe`` and ``cl.exe`` per kernel), call
``install_no_window_default()`` once at process start — it monkey-patches
``subprocess.Popen`` to default-on ``CREATE_NO_WINDOW`` for callers that
didn't pick a console-creation flag themselves.
"""

from __future__ import annotations

import subprocess
import sys


def no_window_kwargs() -> dict:
    """``subprocess`` kwargs that suppress the Windows console-window flash.

    Returns ``{"creationflags": CREATE_NO_WINDOW}`` on Windows, ``{}`` elsewhere
    (so it's a harmless no-op on Linux/macOS). Merge into an existing kwargs
    dict or splat directly into a ``subprocess`` call.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


_INSTALLED = False


def install_no_window_default() -> None:
    """Default-on ``CREATE_NO_WINDOW`` for every ``subprocess.Popen`` (Windows).

    No-op on Linux/macOS and on any second call. On Windows, wraps
    ``subprocess.Popen.__init__`` so that when the caller hasn't already
    specified one of ``CREATE_NEW_CONSOLE``/``CREATE_NO_WINDOW``/
    ``DETACHED_PROCESS``/``CREATE_NEW_PROCESS_GROUP``, we OR in
    ``CREATE_NO_WINDOW`` before delegating.

    ``torch.compile`` (inductor + Triton) shells out to ``ptxas.exe`` /
    ``cl.exe`` / ``cuobjdump.exe`` per generated kernel during the first
    training step, from call sites that can't take ``no_window_kwargs()``. If
    the Python parent has no inherited console (uv-venv ``python.exe``
    trampoline re-exec, pythonw.exe GUI parent, some double-click launchers),
    Windows allocates a fresh **visible** console for each grandchild.

    Safe defaults: ptxas / cl / cuobjdump never read from a console and their
    stdout/stderr are always captured via pipes by torch — losing the visible
    console loses no output. Callers that explicitly want a new visible window
    keep working because we only set the flag when no console-disposition flag
    is present.
    """
    global _INSTALLED
    if _INSTALLED or sys.platform != "win32":
        return

    flag = subprocess.CREATE_NO_WINDOW
    console_flags = (
        subprocess.CREATE_NEW_CONSOLE
        | subprocess.CREATE_NO_WINDOW
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
    )
    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        cf = kwargs.get("creationflags", 0) or 0
        if not (cf & console_flags):
            kwargs["creationflags"] = cf | flag
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init  # type: ignore[method-assign]
    _INSTALLED = True
