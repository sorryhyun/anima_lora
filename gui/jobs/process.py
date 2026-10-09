"""Process-tree control helpers for the GUI.

QProcess.kill() only signals the immediate child. The GUI's directly-spawned
processes are wrappers (``python tasks.py <task>``, the toolkit scripts) whose
real work — holding VRAM — runs in a grandchild, which killing the wrapper
alone leaves orphaned. Two helpers:

* ``setup_kill_safe`` — on Unix, starts the child in a fresh session so the
  whole launcher subtree shares one process group.
* ``kill_process_tree`` — walks descendants via psutil, SIGTERMs them, then
  SIGKILLs anything still alive after a short grace period.

``StreamingProcess`` wraps both around a ``QProcess`` that runs a Python script
from the repo root and hands its output to the host as decoded text.
"""

from __future__ import annotations

import codecs
import re
import sys
from collections.abc import Mapping, Sequence

import psutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from gui.core.paths import ROOT


def setup_kill_safe(proc: QProcess) -> None:
    """Configure ``proc`` so its child can be killed as a tree.

    On Unix, the child is started as a session leader (``setsid()``) which
    makes it the head of a new process group; psutil does the actual walk.
    On Windows this
    is a no-op; psutil's tree walk works there without extra setup.

    Also sets ``PYTHONUNBUFFERED=1`` on the child process environment so the
    GUI sees output (especially tqdm progress) in real-time instead of in
    block-buffered chunks. Pipes from QProcess aren't TTYs, and Python's
    default stdio is block-buffered when redirected to pipes.
    """
    env = make_subprocess_env()
    proc.setProcessEnvironment(env)
    if sys.platform == "win32":
        return
    params = QProcess.UnixProcessParameters()
    params.flags = QProcess.UnixProcessFlag.CreateNewSession
    proc.setUnixProcessParameters(params)


def make_subprocess_env(**extras: str) -> QProcessEnvironment:
    """Build a QProcessEnvironment from the system env + ``PYTHONUNBUFFERED=1``.

    Pass keyword args to add tab-specific vars (e.g. ``METHOD``,
    ``METHODS_SUBDIR``) without forgetting the unbuffered flag.
    """
    env = QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONUNBUFFERED", "1")
    for k, v in extras.items():
        env.insert(k, v)
    return env


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


class StreamingProcess(QObject):
    """A kill-safe ``python <args>`` child run from ``ROOT``, streamed as text.

    Signals (the bool is ``True`` for stderr):

    * ``chunk(text, is_stderr)`` — every decoded read, as it arrives.
    * ``line(text, is_stderr)`` — each complete line, split on ``separators``
      (a regex). A stream's unterminated tail is emitted as a last ``line``
      just before ``finished``.
    * ``finished(exit_code)``.

    A host connects the signals it needs. One run at a time: :meth:`start`
    returns ``False`` while a child is alive.
    """

    chunk = Signal(str, bool)
    line = Signal(str, bool)
    finished = Signal(int)

    def __init__(self, parent: QObject | None = None, *, separators: str = r"[\r\n]"):
        super().__init__(parent)
        self._sep = re.compile(separators)
        self._reset_streams()
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(lambda: self._read(False))
        self._proc.readyReadStandardError.connect(lambda: self._read(True))
        self._proc.finished.connect(self._on_finished)

    def is_running(self) -> bool:
        return self._proc.state() != QProcess.NotRunning

    def start(
        self, args: Sequence[str], *, env: Mapping[str, str | None] | None = None
    ) -> bool:
        """Run ``sys.executable *args``. ``env`` entries are set on top of the
        system environment (plus ``PYTHONUNBUFFERED=1``); a ``None`` value
        removes the variable."""
        if self.is_running():
            return False
        penv = make_subprocess_env()
        for k, v in (env or {}).items():
            if v is None:
                penv.remove(k)
            else:
                penv.insert(k, v)
        self._proc.setProcessEnvironment(penv)
        self._reset_streams()
        self._proc.start(sys.executable, list(args))
        return True

    def kill(self) -> None:
        """Terminate the child and its descendants (no-op when idle)."""
        kill_process_tree(self._proc)

    def _reset_streams(self) -> None:
        # Incremental decoders: a multi-byte character split across two reads
        # decodes whole instead of as two replacement characters.
        self._decoders = {
            err: codecs.getincrementaldecoder("utf-8")(errors="replace")
            for err in (False, True)
        }
        self._bufs = {False: "", True: ""}

    def _read(self, err: bool) -> None:
        raw = (
            self._proc.readAllStandardError()
            if err
            else self._proc.readAllStandardOutput()
        )
        text = self._decoders[err].decode(raw.data())
        if not text:
            return
        self.chunk.emit(text, err)
        *lines, self._bufs[err] = self._sep.split(self._bufs[err] + text)
        for ln in lines:
            self.line.emit(ln, err)

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        for err in (False, True):
            self._read(err)
            tail = self._bufs[err] + self._decoders[err].decode(b"", final=True)
            self._bufs[err] = ""
            if tail:
                self.line.emit(tail, err)
        self.finished.emit(exit_code)
