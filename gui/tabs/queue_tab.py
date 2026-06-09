"""QueueTab — inspect daemon jobs and their captured stdout."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import LazyTabMixin
from gui import daemon as gui_daemon
from gui.i18n import t
from gui.progress import TQDM_RE, make_progress_bar

_LIVE_STATES = {"queued", "running"}
_MAX_LOG_BYTES = 2 * 1024 * 1024
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_TQDM_BAR_RE = re.compile(r"\d+%\|")


def _fmt_time(ts) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "-"


def _job_label(job: dict) -> str:
    state = job.get("state") or "unknown"
    kind = job.get("kind") or "train"
    method = job.get("method") or "-"
    chain = job.get("chain_train") or {}
    if kind == "command" and chain.get("method"):
        method = f"{method} -> {chain.get('method')}"
    return f"[{state}] {kind}: {method}  {job.get('id') or ''}"


def _job_detail(job: dict) -> str:
    lines = [
        t("queue_detail_id", job_id=job.get("id") or "-"),
        t("queue_detail_state", state=job.get("state") or "-"),
        t("queue_detail_kind", kind=job.get("kind") or "train"),
        t("queue_detail_method", method=job.get("method") or "-"),
        t("queue_detail_submitted", time=_fmt_time(job.get("submitted_at"))),
        t("queue_detail_started", time=_fmt_time(job.get("started_at"))),
        t("queue_detail_ended", time=_fmt_time(job.get("ended_at"))),
    ]
    if job.get("from_chain"):
        lines.append(t("queue_detail_from_chain"))
    chain = job.get("chain_train") or {}
    if chain.get("method"):
        lines.append(t("queue_detail_chain", method=chain.get("method")))
    if job.get("chained_job_id"):
        lines.append(t("queue_detail_chained_id", job_id=job.get("chained_job_id")))
    if job.get("pid"):
        lines.append(t("queue_detail_pid", pid=job.get("pid")))
    if job.get("error"):
        lines.append(t("queue_detail_error", error=job.get("error")))
    if job.get("status_detail"):
        lines.append(t("queue_detail_status_detail", detail=job.get("status_detail")))
    if job.get("config_file"):
        lines.append(t("queue_detail_config", path=job.get("config_file")))
    stdout = job.get("stdout_path")
    if stdout:
        lines.append(t("queue_detail_stdout", path=stdout))
    return "\n".join(lines)


def _read_log(path: str | None) -> str:
    if not path:
        return t("queue_log_missing")
    p = Path(path)
    if not p.is_file():
        return t("queue_log_missing")
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > _MAX_LOG_BYTES:
                f.seek(size - _MAX_LOG_BYTES)
            data = f.read()
    except OSError as exc:
        return t("queue_log_read_failed", err=str(exc))
    text = data.decode("utf-8", errors="replace")
    if size > _MAX_LOG_BYTES:
        text = t("queue_log_truncated", mb=_MAX_LOG_BYTES // (1024 * 1024)) + text
    return text


def _clean_log(text: str) -> str:
    """Render captured terminal output like the training tab log."""
    if not text:
        return text
    out: list[str] = []
    for raw in re.split(r"[\r\n]+", text):
        line = _ANSI_RE.sub("", raw).strip()
        if not line or (_TQDM_BAR_RE.search(line) and TQDM_RE.search(line)):
            continue
        out.append(line)
    return "\n".join(out)


class QueueTab(LazyTabMixin, QWidget):
    def __init__(self):
        super().__init__()
        self._jobs: list[dict] = []
        self._selected_job_id: str | None = None
        self._last_log_text: str | None = None

        outer = QVBoxLayout(self)

        top = QHBoxLayout()
        self.refresh_btn = QPushButton(t("queue_refresh"))
        self.refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self.refresh_btn)

        self.stop_btn = QPushButton(t("queue_stop_selected"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop_selected)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)

        self.copy_btn = QPushButton(t("queue_copy_output"))
        self.copy_btn.clicked.connect(self._copy_output)
        self.copy_btn.setEnabled(False)
        top.addWidget(self.copy_btn)

        top.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#aaa;")
        top.addWidget(self.status_label)
        outer.addLayout(top)

        split = QSplitter(Qt.Horizontal)

        self.job_list = QListWidget()
        self.job_list.currentItemChanged.connect(self._selection_changed)
        split.addWidget(self.job_list)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(8, 0, 0, 0)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.detail.setMaximumHeight(150)
        self.detail.setStyleSheet("font-family:monospace;font-size:11px;")
        self.detail.setPlaceholderText(t("queue_detail_placeholder"))
        rlay.addWidget(self.detail)

        self.progress = make_progress_bar()
        rlay.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("queue_log_placeholder"))
        rlay.addWidget(self.log, 1)

        split.addWidget(right)
        split.setSizes([360, 760])
        outer.addWidget(split, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)

    def _lazy_init(self) -> None:
        self.refresh()
        self._timer.start()

    def refresh(self) -> None:
        selected = self._selected_job_id
        try:
            # Passive: a monitor that polls on a timer must never spawn a daemon
            # just by being open, nor block the UI thread waiting for one.
            jobs = gui_daemon.list_jobs_passive()
        except Exception as exc:  # noqa: BLE001
            self._jobs = []
            self.job_list.clear()
            self.detail.clear()
            self.log.clear()
            self.status_label.setText(t("queue_daemon_unavailable"))
            self.stop_btn.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self.progress.setVisible(False)
            if self.isVisible():
                self.log.setPlainText(t("daemon_submit_failed", err=str(exc)))
            return

        self._jobs = list(reversed(jobs))
        self.job_list.blockSignals(True)
        self.job_list.clear()
        selected_row = -1
        for row, job in enumerate(self._jobs):
            item = QListWidgetItem(_job_label(job))
            item.setData(Qt.UserRole, job.get("id"))
            state = job.get("state")
            if state == "running":
                item.setForeground(Qt.GlobalColor.green)
            elif state == "queued":
                item.setForeground(Qt.GlobalColor.cyan)
            elif state in {"error", "stopped"}:
                item.setForeground(Qt.GlobalColor.red)
            self.job_list.addItem(item)
            if selected and job.get("id") == selected:
                selected_row = row
        if selected_row < 0 and self._jobs:
            selected_row = 0
        previous_selected = self._selected_job_id
        if selected_row >= 0:
            self.job_list.setCurrentRow(selected_row)
            self._selected_job_id = self.job_list.item(selected_row).data(Qt.UserRole)
        else:
            self._selected_job_id = None
        if self._selected_job_id != previous_selected:
            self._last_log_text = None
        self.job_list.blockSignals(False)

        live = sum(1 for job in self._jobs if job.get("state") in _LIVE_STATES)
        self.status_label.setText(t("queue_status", total=len(self._jobs), live=live))
        self._update_selected_view()

    def _selection_changed(self, current: QListWidgetItem | None, _prev) -> None:
        self._selected_job_id = current.data(Qt.UserRole) if current else None
        self._last_log_text = None
        self._update_selected_view()

    def _selected_job(self) -> dict | None:
        if not self._selected_job_id:
            return None
        for job in self._jobs:
            if job.get("id") == self._selected_job_id:
                return job
        return None

    def _update_selected_view(self) -> None:
        job = self._selected_job()
        if not job:
            self.detail.clear()
            self.log.clear()
            self.stop_btn.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self.progress.setVisible(False)
            return
        detail_text = _job_detail(job)
        if (
            detail_text != self.detail.toPlainText()
            and not self.detail.textCursor().hasSelection()
        ):
            self.detail.setPlainText(detail_text)
        self.stop_btn.setEnabled(job.get("state") in _LIVE_STATES)
        self.copy_btn.setEnabled(True)

        raw_log = _read_log(job.get("stdout_path"))
        self._update_progress(job, raw_log)
        log_text = _clean_log(raw_log)
        if log_text == self._last_log_text:
            return
        if self.log.textCursor().hasSelection():
            return
        bar = self.log.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        cursor = self.log.textCursor()
        cursor_at_end = cursor.position() == len(self.log.toPlainText())
        self.log.setPlainText(log_text)
        self._last_log_text = log_text
        if at_bottom or cursor_at_end:
            self.log.moveCursor(QTextCursor.End)

    def _update_progress(self, job: dict, raw_log: str) -> None:
        if self._progress_from_jsonl(job.get("progress_path")):
            return
        if self._progress_from_tqdm(raw_log):
            return
        if job.get("state") in _LIVE_STATES:
            self.progress.setRange(0, 0)
            self.progress.setFormat(t("starting"))
            self.progress.setVisible(True)
        else:
            self.progress.setVisible(False)

    def _progress_from_jsonl(self, path: str | None) -> bool:
        if not path:
            return False
        p = Path(path)
        if not p.is_file():
            return False
        total = 0
        cur = 0
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("ev") == "run_start":
                        total = int(ev.get("total_steps") or total or 0)
                    elif ev.get("ev") == "step":
                        cur = int(ev.get("global_step") or cur or 0)
        except OSError:
            return False
        if total <= 0 and cur <= 0:
            return False
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(min(cur, total))
            self.progress.setFormat(f"step {cur}/{total} (%p%)")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(f"step {cur}")
        self.progress.setVisible(True)
        return True

    def _progress_from_tqdm(self, raw_log: str) -> bool:
        last = None
        for line in re.split(r"[\r\n]+", raw_log or ""):
            line = _ANSI_RE.sub("", line)
            m = TQDM_RE.search(line)
            if m:
                last = m
        if not last:
            return False
        label = last.group("label").strip() or "progress"
        cur = int(last.group("cur"))
        total = int(last.group("tot"))
        self.progress.setRange(0, total)
        self.progress.setValue(min(cur, total))
        self.progress.setFormat(f"{label}: {cur}/{total} (%p%)")
        self.progress.setVisible(True)
        return True

    def _stop_selected(self) -> None:
        job = self._selected_job()
        if not job:
            return
        job_id = job.get("id")
        if not job_id:
            return
        try:
            gui_daemon.stop_job(job_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, t("error"), f"stop failed: {exc}")
            return
        self.refresh()

    def _copy_output(self) -> None:
        cursor = self.log.textCursor()
        text = cursor.selectedText().replace("\u2029", "\n")
        if not text:
            text = self.log.toPlainText()
        QApplication.clipboard().setText(text)

    def cleanup_subprocess(self) -> None:
        self._timer.stop()
