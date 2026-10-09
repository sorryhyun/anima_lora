"""Slim always-on bar showing live GPU utilisation / VRAM.

Qt-only by design: it shells out to ``nvidia-smi`` via a non-blocking
``QProcess`` rather than importing torch or pynvml, so it adds nothing to GUI
startup time (see gui/CLAUDE.md — the GUI must stay torch-free). On a CPU-only
or non-NVIDIA box ``nvidia-smi`` is absent; the bar then hides itself silently.
"""

from __future__ import annotations

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from gui.i18n import t
from gui.theme import tok

# Columns pulled per GPU, in order. Matches the parse in `_on_finished`.
_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu"
_INTERVAL_MS = 2000


class GpuStatusBar(QWidget):
    """A one-line footer polling ``nvidia-smi`` every couple of seconds.

    Each tick spawns a short-lived ``nvidia-smi`` query; the UI thread never
    blocks on it. If nvidia-smi can't be found or errors, the bar stops polling
    and removes itself so a CPU-only setup isn't bothered by a dead widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel(t("gpu_probing"))
        self._label.setStyleSheet(f"color:{tok('text_dim')};font-size:12px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 2, 10, 6)
        lay.addWidget(self._label)
        lay.addStretch()

        self._proc = QProcess(self)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

        self._timer = QTimer(self)
        self._timer.setInterval(_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    def _poll(self) -> None:
        # Skip if the previous probe is still in flight (a stalled nvidia-smi
        # shouldn't queue up overlapping calls).
        if self._proc.state() != QProcess.NotRunning:
            return
        self._proc.start(
            "nvidia-smi",
            [f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
        )

    def _on_error(self, _error) -> None:
        # nvidia-smi missing (FailedToStart) or crashed — drop the bar.
        self._timer.stop()
        self.hide()

    def _on_finished(self, code: int, _status) -> None:
        if code != 0:
            self._timer.stop()
            self.hide()
            return
        out = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        parts: list[str] = []
        for i, line in enumerate(ln.strip() for ln in out.splitlines() if ln.strip()):
            try:
                util, used, total, temp = (c.strip() for c in line.split(","))
                used_gb = float(used) / 1024
                total_gb = float(total) / 1024
            except (ValueError, IndexError):
                continue
            parts.append(
                t(
                    "gpu_stat",
                    i=i,
                    util=util,
                    used=f"{used_gb:.1f}",
                    total=f"{total_gb:.1f}",
                    temp=temp,
                )
            )
        if parts:
            self._label.setText("      ".join(parts))

    def cleanup(self) -> None:
        """Stop polling and kill any in-flight probe (called on window close)."""
        self._timer.stop()
        if self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
