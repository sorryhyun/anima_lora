"""Resident Anima Tagger worker — the autotag subprocess-protocol state machine
split out of ``image_tab.py``.

``_AutotagWorker`` owns the ``QProcess`` lifetime, the stdio sentinel protocol,
and the idle/GPU-watch timers; it never touches a widget. ``ImageViewerTab``
owns one instance and wires its ``status``/``busy``/``result``/``error``
signals to the caption editor + autotag button, so this module stays UI-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

from anime_tools.contract import (
    AUTOTAG_ERROR_PREFIX,
    AUTOTAG_READY,
    AUTOTAG_RESULT_PREFIX,
)
from PySide6.QtCore import (
    QElapsedTimer,
    QObject,
    QProcess,
    QProcessEnvironment,
    QTimer,
    Signal,
)

from gui import DEFAULT_AUTOTAG_CONFIDENCE, ROOT, get_setting
from gui import daemon as gui_daemon

# Stdio protocol sentinels of the resident autotag worker
# (``anime_tools.tagger.cli.autotag_server``), from the package's stdlib-only
# contract module — the GUI stays torch-free by construction.
_AUTOTAG_READY = AUTOTAG_READY
_AUTOTAG_RESULT_PREFIX = AUTOTAG_RESULT_PREFIX
_AUTOTAG_ERROR_PREFIX = AUTOTAG_ERROR_PREFIX

# Free the resident tagger (VRAM) after this many ms with no autotag request.
_AUTOTAG_IDLE_MS = 10 * 60 * 1000
# Poll cadence (ms) for "did some other GPU job start?" while resident.
_AUTOTAG_GPU_WATCH_MS = 700
# How much of the worker's stderr to keep. A startup failure (missing or gated
# tagger weights) never reaches the stdout sentinel protocol — the process just
# dies — so without this tail the tab could only report a bare "exit".
_STDERR_TAIL_CHARS = 8000

# The catalog rows the resident worker needs on disk before it can serve
# anything: our small checkpoint plus the gated upstream backbone.
TAGGER_ASSET_GROUP = "tagger-model"

# Substrings that mark a worker crash as "the weights aren't there / aren't
# reachable" rather than a genuine tagging failure. Matched case-insensitively
# against the worker's last stderr line, which for the gated case is the
# ``FileNotFoundError`` anime_tools raises with the accept-terms recovery.
_MISSING_MODEL_MARKERS = (
    "gated",
    "hf auth login",
    "animetimm/",
    "repositorynotfounderror",
    "entrynotfounderror",
    "backbone",
    "tagger checkpoint",
    "anima-tagger",
    "401 client error",
    "403 client error",
)


def _tagger_assets() -> list:
    """The tagger's catalog rows, or ``[]`` if the catalog can't be read.

    Import is local so this module stays cheap for callers that never autotag,
    and the rows come from ``library.downloads`` rather than a path list here —
    see the catalog rule in ``gui/system_dialog.py``.
    """
    try:
        from library import downloads as DL

        return list(DL.resolve([TAGGER_ASSET_GROUP]))
    except Exception:  # noqa: BLE001 — a broken catalog must not block autotag
        return []


def missing_tagger_assets() -> list:
    """Tagger rows that are not installed yet (offline probe).

    Empty means the resident worker has everything it needs.
    """
    return [a for a in _tagger_assets() if not a.installed]


def gated_urls(assets) -> list[str]:
    """The accept-terms pages among ``assets`` (``Asset.gated`` is the URL)."""
    return [a.gated for a in assets if getattr(a, "gated", None)]


def tagger_gated_urls() -> list[str]:
    """Every accept-terms page the tagger needs, installed or not.

    The fallback when the rows probe as installed but the worker still failed
    on access — a revoked token reads as "present" to an offline cache probe,
    and the terms page is exactly what the user needs in that case.
    """
    return gated_urls(_tagger_assets())


def is_missing_model_error(message: str) -> bool:
    """True when ``message`` reads like a missing / gated-weights failure."""
    low = message.lower()
    return any(m in low for m in _MISSING_MODEL_MARKERS)


def _last_line(text: str) -> str:
    """Last non-empty line of ``text`` — the exception line of a traceback."""
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


# Status phases emitted on `status` — the tab maps these to translated text
# (kept plain-string here so this module never imports gui.i18n).
STATUS_LOADING = "loading"
STATUS_RUNNING = "running"
STATUS_READY = "ready"
STATUS_IDLE = ""


class _AutotagWorker(QObject):
    """Drives the resident ``autotag_server`` subprocess for one tab.

    First request spawns a torch subprocess that loads the model once; it then
    stays alive so later requests just stream an image path to it, freed after
    ``_AUTOTAG_IDLE_MS`` idle or as soon as another GPU job (train/preprocess/
    group) starts. Callers only see these signals — never the subprocess or
    the stdio protocol:

    - ``status(str)`` — one of the ``STATUS_*`` phases above.
    - ``busy(bool)`` — True while a request is in flight or the worker is
      loading; the tab disables the autotag button on True.
    - ``result(Path, str)`` — (image_path, predicted caption) once a reply
      lands; the tab applies it only if that image is still on screen.
    - ``error(str)`` — a raw error string: the worker's stderr sentinel, or,
      when it died before the protocol started, the last line of its stderr
      (``"exit"`` only if it printed nothing). The tab formats + shows it, and
      routes a missing/gated-weights message to the download prompt.
    """

    status = Signal(str)
    busy = Signal(bool)
    result = Signal(Path, str)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._ready = False
        self._buf = ""  # partial-line buffer for the worker's stdout
        self._err = ""  # tail of the worker's stderr (crash diagnosis)
        self._inflight_image: Path | None = None
        self._idle = QElapsedTimer()
        # Polls "did another GPU job start?" only while the worker is resident.
        self._gpu_watch_timer = QTimer(self)
        self._gpu_watch_timer.setInterval(_AUTOTAG_GPU_WATCH_MS)
        self._gpu_watch_timer.timeout.connect(self._gpu_watch_tick)

    def request(self, image_path: Path) -> None:
        """Tag ``image_path``. Silently ignored if a request is already in
        flight (double-click guard — the caller's button should already be
        disabled via ``busy``, this is a defensive second check)."""
        if self._inflight_image is not None:
            return
        self._inflight_image = image_path
        self._idle.restart()
        self.busy.emit(True)
        if self._proc is None:
            self._spawn()
            self.status.emit(STATUS_LOADING)
        elif self._ready:
            self.status.emit(STATUS_RUNNING)
            self._send(image_path)
        # else: worker still loading — _on_stdout sends it on READY.

    def kill(self) -> None:
        """Tear down the resident worker and free its VRAM. Idempotent."""
        self._gpu_watch_timer.stop()
        proc = self._proc
        self._proc = None
        self._ready = False
        self._buf = ""
        self._err = ""
        self._inflight_image = None
        self.busy.emit(False)
        self.status.emit(STATUS_IDLE)
        if proc is None:
            return
        try:
            proc.readyReadStandardOutput.disconnect()
            proc.readyReadStandardError.disconnect()
            proc.finished.disconnect()
            proc.errorOccurred.disconnect()
        except (RuntimeError, TypeError):
            pass
        if proc.state() != QProcess.NotRunning:
            proc.closeWriteChannel()  # EOF on stdin → worker exits cleanly
            proc.kill()
            proc.waitForFinished(2000)
        proc.deleteLater()

    def _spawn(self) -> None:
        """Launch the resident worker subprocess (torch lives here, not the GUI)."""
        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(["-m", "anime_tools.tagger.cli.autotag_server"])
        proc.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # stream sentinel lines live
        # anime_tools anchors bare defaults on ANIMA_HOME (else the CWD).
        env.insert("ANIMA_HOME", str(ROOT))
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(lambda _e: self._on_finished(-1, None))
        self._proc = proc
        self._ready = False
        self._buf = ""
        self._err = ""
        self._gpu_watch_timer.start()
        proc.start()

    def _send(self, image_path: Path) -> None:
        if self._proc is None:
            return
        # Read the confidence floor fresh each request so a settings change
        # applies without respawning the resident worker.
        try:
            conf = float(get_setting("autotag_confidence", DEFAULT_AUTOTAG_CONFIDENCE))
        except (TypeError, ValueError):
            conf = DEFAULT_AUTOTAG_CONFIDENCE
        conf = max(0.0, min(1.0, conf))
        self._proc.write(f"{conf}\t{image_path}\n".encode("utf-8"))

    def _on_stdout(self) -> None:
        if self._proc is None:
            return
        self._buf += bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", "replace"
        )
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                continue
            if line == _AUTOTAG_READY:
                self._ready = True
                self.status.emit(STATUS_READY)
                # Send the request that spawned the worker, now that it's loaded.
                if self._inflight_image is not None:
                    self.status.emit(STATUS_RUNNING)
                    self._send(self._inflight_image)
            elif line.startswith(_AUTOTAG_RESULT_PREFIX):
                self._finish_result(line[len(_AUTOTAG_RESULT_PREFIX) :])
            elif line.startswith(_AUTOTAG_ERROR_PREFIX):
                self._finish_error(line[len(_AUTOTAG_ERROR_PREFIX) :])

    def _on_stderr(self) -> None:
        """Keep the tail of the worker's stderr (logs + any traceback).

        Never parsed for the protocol — stdout owns that. This exists purely so
        a crash before ``READY`` (missing / gated weights being the common one)
        can be reported with its real reason instead of a bare exit code.
        """
        if self._proc is None:
            return
        self._err += bytes(self._proc.readAllStandardError()).decode("utf-8", "replace")
        if len(self._err) > _STDERR_TAIL_CHARS:
            self._err = self._err[-_STDERR_TAIL_CHARS:]

    def _finish_result(self, caption: str) -> None:
        image = self._inflight_image
        self._clear_inflight()
        if image is not None:
            self.result.emit(image, caption.strip())

    def _finish_error(self, message: str) -> None:
        self._clear_inflight()
        self.error.emit(message)

    def _clear_inflight(self) -> None:
        self._inflight_image = None
        self.busy.emit(False)
        self._idle.restart()
        if self._ready:
            self.status.emit(STATUS_READY)

    def _gpu_watch_tick(self) -> None:
        """While the worker is resident: free it on any other GPU job or idle."""
        if self._proc is None:
            self._gpu_watch_timer.stop()
            return
        if gui_daemon.active_job_id():  # train/preprocess/group grabbed the card
            self.kill()
            return
        if (
            self._inflight_image is None
            and self._idle.isValid()
            and self._idle.hasExpired(_AUTOTAG_IDLE_MS)
        ):
            self.kill()

    def _on_finished(self, _code, _status) -> None:
        """Worker exited (crash, kill, or EOF) — reset to the no-worker state."""
        was_inflight = self._inflight_image is not None
        self._on_stderr()  # drain whatever the crash wrote just before exiting
        reason = _last_line(self._err)
        self._gpu_watch_timer.stop()
        self._proc = None
        self._ready = False
        self._buf = ""
        self._err = ""
        self._inflight_image = None
        self.busy.emit(False)
        self.status.emit(STATUS_IDLE)
        if was_inflight:
            self.error.emit(reason or "exit")
