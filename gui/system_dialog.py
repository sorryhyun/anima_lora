"""Models / Update dialogs — wrappers around `python tasks.py download-* / update`.

Both dialogs share the same shape: a row of action buttons, a status area, and
a streaming log fed by ``QProcess`` (same pattern as MergeTab). Only one job
runs at a time per dialog — buttons disable while busy and re-enable on finish.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QProcess
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import ROOT
from gui.i18n import t
from gui.process import kill_process_tree, setup_kill_safe

# (task-key, display-label-i18n-key, [paths-relative-to-ROOT-that-must-all-exist])
# Status is "installed" iff every path resolves; otherwise "missing".
_MODEL_GROUPS: list[tuple[str, str, list[str]]] = [
    (
        "anima",
        "model_anima",
        [
            "models/diffusion_models/anima-preview3-base.safetensors",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "models/vae/qwen_image_vae.safetensors",
        ],
    ),
    ("sam3", "model_sam3", ["models/sam3/sam3.pt"]),
    ("mit", "model_mit", ["models/mit/model.pth"]),
    ("tipsv2", "model_tipsv2", ["models/tipsv2/config.json"]),
    ("pe", "model_pe", ["models/pe/PE-Core-L14-336.pt"]),
    ("pe-g", "model_pe_g", ["models/pe/PE-Core-G14-448.pt"]),
]


def _all_exist(paths: list[str]) -> bool:
    return all((ROOT / p).exists() for p in paths)


class _StreamingDialog(QDialog):
    """Base — owns the QProcess, log pane, and busy-state plumbing.

    Subclasses build the action UI in ``_build_actions(layout)`` and call
    ``self._run([...])`` to launch a ``python tasks.py ...`` invocation.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 520)

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(12, 12, 12, 12)

        self._actions_host = QWidget()
        actions_lay = QVBoxLayout(self._actions_host)
        actions_lay.setContentsMargins(0, 0, 0, 0)
        self._build_actions(actions_lay)
        self._lay.addWidget(self._actions_host)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self._lay.addWidget(self.log, 1)

        bottom = QHBoxLayout()
        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        bottom.addWidget(self.stop_btn)
        bottom.addStretch()
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        bottom.addWidget(bb)
        self._lay.addLayout(bottom)

        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)

    def _build_actions(self, layout: QVBoxLayout) -> None:  # override
        raise NotImplementedError

    def _set_busy(self, busy: bool) -> None:  # override to disable subclass buttons
        self.stop_btn.setEnabled(busy)

    def _run(self, args: list[str]) -> None:
        if self._proc.state() != QProcess.NotRunning:
            return
        cmd = [sys.executable, "tasks.py", *args]
        self._log(f"> {' '.join(cmd)}\n")
        self._set_busy(True)
        self._proc.start(cmd[0], cmd[1:])

    def _stop(self) -> None:
        kill_process_tree(self._proc)

    def _read_stdout(self):
        self._log(self._proc.readAllStandardOutput().data().decode(errors="replace"))

    def _read_stderr(self):
        self._log(self._proc.readAllStandardError().data().decode(errors="replace"))

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus):
        self._log(f"\n{t('finished', code=exit_code)}\n")
        self._set_busy(False)
        self._after_finished(exit_code)

    def _after_finished(self, exit_code: int) -> None:  # optional override
        pass

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def closeEvent(self, ev):
        kill_process_tree(self._proc)
        super().closeEvent(ev)


class ModelsDialog(_StreamingDialog):
    """One row per model group: label · status · download button.

    Download-all button at the top runs ``download-models`` (Anima + SAM3 +
    MIT + TIPSv2). Per-group buttons let users pick just one (re)download.
    """

    def __init__(self, parent=None):
        # Each entry: (status_label, paths, button) — populated in _build_actions
        # so _after_finished can refresh every row after a download-all run.
        self._rows: list[tuple[QLabel, list[str], QPushButton]] = []
        super().__init__(t("models_title"), parent)

    def _build_actions(self, layout: QVBoxLayout) -> None:
        intro = QLabel(t("models_intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#aaa;")
        layout.addWidget(intro)

        # Download-all button (Anima + SAM3 + MIT + TIPSv2).
        all_row = QHBoxLayout()
        self.all_btn = QPushButton(t("models_download_all"))
        self.all_btn.setStyleSheet(
            "background:#16a085;color:white;font-weight:bold;padding:6px 18px;"
        )
        self.all_btn.clicked.connect(lambda: self._run(["download-models"]))
        all_row.addWidget(self.all_btn)
        all_row.addStretch()
        layout.addLayout(all_row)

        # Per-group rows.
        for key, label_key, paths in _MODEL_GROUPS:
            row = QHBoxLayout()

            name = QLabel(t(label_key))
            name.setMinimumWidth(280)
            row.addWidget(name)

            installed = _all_exist(paths)
            status = QLabel(t("models_installed") if installed else t("models_missing"))
            status.setStyleSheet("color:#4ade80;" if installed else "color:#f87171;")
            status.setMinimumWidth(110)
            row.addWidget(status)

            row.addStretch()

            btn = QPushButton(
                t("models_redownload") if installed else t("models_download")
            )
            btn.clicked.connect(
                lambda _checked=False, k=key, s=status, p=paths, b=btn: self._download(
                    k, s, p, b
                )
            )
            self._rows.append((status, paths, btn))
            row.addWidget(btn)

            layout.addLayout(row)

    def _download(
        self,
        key: str,
        _status_lbl: QLabel,
        _paths: list[str],
        _btn: QPushButton,
    ) -> None:
        # _after_finished refreshes every row, so we don't need to track which
        # row was clicked.
        self._run([f"download-{key}"])

    def _set_busy(self, busy: bool) -> None:
        super()._set_busy(busy)
        self.all_btn.setEnabled(not busy)
        for _status, _paths, b in self._rows:
            b.setEnabled(not busy)

    def _after_finished(self, _exit_code: int) -> None:
        # Refresh every row's status — handles both per-group downloads and
        # download-models, which touches several groups in one run.
        for status_lbl, paths, btn in self._rows:
            installed = _all_exist(paths)
            status_lbl.setText(
                t("models_installed") if installed else t("models_missing")
            )
            status_lbl.setStyleSheet(
                "color:#4ade80;" if installed else "color:#f87171;"
            )
            btn.setText(t("models_redownload") if installed else t("models_download"))


class UpdateDialog(_StreamingDialog):
    """Run ``python tasks.py update`` with a confirmation + dry-run option.

    The update script in ``scripts/update.py`` preserves dataset/output/models
    and prompts on config conflicts, but it still rewrites the working tree —
    we surface that warning before kicking off.
    """

    def __init__(self, parent=None):
        super().__init__(t("update_title"), parent)

    def _build_actions(self, layout: QVBoxLayout) -> None:
        warn = QLabel(t("update_warning"))
        warn.setWordWrap(True)
        warn.setStyleSheet(
            "padding:8px; border-radius:3px; background:#3d2e0a; color:#fbbf24;"
        )
        layout.addWidget(warn)

        row = QHBoxLayout()
        self.dry_btn = QPushButton(t("update_dry_run"))
        self.dry_btn.clicked.connect(lambda: self._run(["update", "--dry-run"]))
        row.addWidget(self.dry_btn)

        self.run_btn = QPushButton(t("update_run"))
        self.run_btn.setStyleSheet(
            "background:#16a085;color:white;font-weight:bold;padding:6px 18px;"
        )
        self.run_btn.clicked.connect(self._confirm_and_run)
        row.addWidget(self.run_btn)
        row.addStretch()
        layout.addLayout(row)

    def _confirm_and_run(self):
        ok = QMessageBox.question(
            self,
            t("update_title"),
            t("update_confirm"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ok == QMessageBox.Yes:
            self._run(["update"])

    def _set_busy(self, busy: bool) -> None:
        super()._set_busy(busy)
        self.dry_btn.setEnabled(not busy)
        self.run_btn.setEnabled(not busy)


# Public helpers for app.py.

GITHUB_REPO_URL = "https://github.com/sorryhyun/anima_lora"
GITHUB_ISSUES_URL = f"{GITHUB_REPO_URL}/issues"


def open_models_dialog(parent=None):
    ModelsDialog(parent).exec()


def open_update_dialog(parent=None):
    UpdateDialog(parent).exec()


__all__ = [
    "GITHUB_ISSUES_URL",
    "GITHUB_REPO_URL",
    "ModelsDialog",
    "UpdateDialog",
    "open_models_dialog",
    "open_update_dialog",
]
