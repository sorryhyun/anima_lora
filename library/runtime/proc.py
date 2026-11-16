"""Windows-quiet subprocess helpers.

On Windows, launching a console program (``git``, ``nvidia-smi`` …) flashes a
console window unless ``CREATE_NO_WINDOW`` is passed; the daemon's GPU poll and
the per-checkpoint git query would blink a terminal repeatedly. The flag does
not survive the uv venv ``python.exe`` trampoline re-exec, so the daemon's job
launcher (``anima_daemon/proc.py``) uses ``pythonw.exe`` instead.

Usage::

    subprocess.run([...], **no_window_kwargs())

``install_no_window_default()`` covers call sites we can't patch
(``torch.compile`` / Triton shelling out to ``ptxas.exe`` / ``cl.exe``).
"""

from __future__ import annotations

import functools
import subprocess
import sys


def no_window_kwargs() -> dict:
    """``subprocess`` kwargs that suppress the Windows console-window flash.

    ``{"creationflags": CREATE_NO_WINDOW}`` on Windows, ``{}`` elsewhere.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


_INSTALLED = False


def install_no_window_default() -> None:
    """Default-on ``CREATE_NO_WINDOW`` for every ``subprocess.Popen`` (Windows).

    No-op off Windows and on a second call. Wraps ``subprocess.Popen.__init__``
    to OR in ``CREATE_NO_WINDOW`` unless the caller already set a
    console-disposition flag. Needed because ``torch.compile`` shells out to
    ``ptxas.exe`` / ``cl.exe`` per kernel, and a parent with no inherited
    console (venv trampoline, pythonw GUI) gets a visible console per
    grandchild. Those tools' output is piped, so nothing is lost.
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


def no_window_main(fn):
    """Decorator for an entry point's ``main()``: runs
    ``install_no_window_default()`` before the body."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        install_no_window_default()
        return fn(*args, **kwargs)

    return wrapper
