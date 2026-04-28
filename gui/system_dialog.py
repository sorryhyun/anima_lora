"""Models / Update dialogs — wrappers around `python tasks.py download-* / update`.

Both dialogs share the same shape: a row of action buttons, a status area, and
a streaming log fed by ``QProcess`` (same pattern as MergeTab). Only one job
runs at a time per dialog — buttons disable while busy and re-enable on finish.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from PySide6.QtCore import QProcess, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
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


GITHUB_REPO = "sorryhyun/anima_lora"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}"
GITHUB_ISSUES_URL = f"{GITHUB_REPO_URL}/issues"
RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
MANIFEST_FILE = ROOT / ".anima_release.json"


def _load_local_version() -> str | None:
    """Read the baseline tag from .anima_release.json, or None if absent."""
    if not MANIFEST_FILE.exists():
        return None
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("version")


class _UpdateCheckThread(QThread):
    """Fetch the latest release tag + body from GitHub on a worker thread.

    Emits ``finished_check`` with a dict: keys ``ok`` (bool), ``tag``,
    ``body``, ``html_url``, ``error`` (str, populated only on failure).
    The HTTP call is plain urllib so we don't pull in a new dependency.
    """

    finished_check = Signal(dict)

    def run(self) -> None:  # noqa: D401 — Qt override
        result: dict = {"ok": False, "tag": "", "body": "", "html_url": "", "error": ""}
        req = urllib.request.Request(
            RELEASE_API_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "anima-update-gui",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            result["ok"] = True
            result["tag"] = data.get("tag_name", "") or ""
            result["body"] = data.get("body", "") or ""
            result["html_url"] = data.get("html_url", "") or ""
        except urllib.error.HTTPError as e:
            result["error"] = f"HTTP {e.code} {e.reason}"
        except urllib.error.URLError as e:
            result["error"] = str(e.reason)
        except Exception as e:  # JSON / unexpected
            result["error"] = str(e)
        self.finished_check.emit(result)


class UpdateDialog(_StreamingDialog):
    """Run ``python tasks.py update`` with a confirmation + dry-run option.

    The update script in ``scripts/update.py`` preserves dataset/output/models
    and prompts on config conflicts, but it still rewrites the working tree —
    we surface that warning before kicking off.

    On open, fires a background GitHub API call to compare the locally
    pinned tag (from ``.anima_release.json``) against the latest release
    and renders the release body as markdown so users can see what's new
    before pulling.
    """

    def __init__(self, parent=None):
        self._check_thread: _UpdateCheckThread | None = None
        self._latest_url: str = ""
        super().__init__(t("update_title"), parent)
        self._kick_check()

    def _build_actions(self, layout: QVBoxLayout) -> None:
        # ── Version + status row ──
        version_row = QHBoxLayout()
        version_row.setSpacing(12)

        local = _load_local_version()
        local_str = local if local else t("update_no_baseline")
        self.current_lbl = QLabel(t("update_current_version", v=local_str))
        version_row.addWidget(self.current_lbl)

        self.latest_lbl = QLabel(t("update_latest_version", v="…"))
        version_row.addWidget(self.latest_lbl)

        self.status_lbl = QLabel(t("update_status_checking"))
        self.status_lbl.setStyleSheet("color:#9ca3af;font-weight:bold;")
        version_row.addWidget(self.status_lbl)

        version_row.addStretch()

        self.view_release_btn = QPushButton(t("update_view_release"))
        self.view_release_btn.setEnabled(False)
        self.view_release_btn.clicked.connect(self._open_release_page)
        version_row.addWidget(self.view_release_btn)

        self.check_btn = QPushButton(t("update_check_now"))
        self.check_btn.clicked.connect(self._kick_check)
        version_row.addWidget(self.check_btn)
        layout.addLayout(version_row)

        # ── Release notes panel ──
        notes_label = QLabel(t("update_release_notes"))
        notes_label.setStyleSheet("color:#aaa;margin-top:4px;")
        layout.addWidget(notes_label)

        self.notes_view = QTextBrowser()
        self.notes_view.setOpenExternalLinks(True)
        self.notes_view.setMaximumHeight(180)
        self.notes_view.document().setDefaultStyleSheet(
            "a { color: #ffb86b; text-decoration: underline; }"
            "code { background:#2a2a2a; padding:1px 4px; border-radius:3px; }"
            "pre { background:#2a2a2a; padding:6px; border-radius:4px; }"
        )
        self.notes_view.setStyleSheet(
            "QTextBrowser { background:#1e1e1e; color:#dcdcdc; "
            "border:1px solid #444; padding:8px; }"
        )
        self.notes_view.setPlaceholderText(t("update_status_checking"))
        layout.addWidget(self.notes_view)

        # ── Warning + run buttons (existing controls) ──
        warn = QLabel(t("update_warning"))
        warn.setWordWrap(True)
        warn.setStyleSheet(
            "padding:8px; border-radius:3px; background:#3d2e0a; color:#fbbf24;"
        )
        layout.addWidget(warn)

        row = QHBoxLayout()
        # Dry-run uses --keep-conflicts: stdin isn't a TTY under QProcess, so
        # without a non-interactive flag the script would block on input().
        self.dry_btn = QPushButton(t("update_dry_run"))
        self.dry_btn.clicked.connect(
            lambda: self._run(["update", "--dry-run", "--keep-conflicts"])
        )
        row.addWidget(self.dry_btn)

        # Two run buttons make the conflict policy explicit instead of an
        # invisible interactive prompt that the GUI can't service.
        self.run_keep_btn = QPushButton(t("update_run_keep"))
        self.run_keep_btn.setStyleSheet(
            "background:#0e7490;color:white;font-weight:bold;padding:6px 18px;"
        )
        self.run_keep_btn.clicked.connect(
            lambda: self._confirm_and_run("--keep-conflicts")
        )
        row.addWidget(self.run_keep_btn)

        self.run_overwrite_btn = QPushButton(t("update_run_overwrite"))
        self.run_overwrite_btn.setStyleSheet(
            "background:#16a085;color:white;font-weight:bold;padding:6px 18px;"
        )
        self.run_overwrite_btn.clicked.connect(
            lambda: self._confirm_and_run("--yes-overwrite")
        )
        row.addWidget(self.run_overwrite_btn)
        row.addStretch()
        layout.addLayout(row)

    def _kick_check(self) -> None:
        if self._check_thread is not None and self._check_thread.isRunning():
            return
        self.check_btn.setEnabled(False)
        self.view_release_btn.setEnabled(False)
        self.latest_lbl.setText(t("update_latest_version", v="…"))
        self.status_lbl.setText(t("update_status_checking"))
        self.status_lbl.setStyleSheet("color:#9ca3af;font-weight:bold;")
        self.notes_view.setMarkdown("")
        self.notes_view.setPlaceholderText(t("update_status_checking"))

        self._check_thread = _UpdateCheckThread(self)
        self._check_thread.finished_check.connect(self._on_check_result)
        self._check_thread.start()

    def _on_check_result(self, result: dict) -> None:
        self.check_btn.setEnabled(True)
        if not result.get("ok"):
            self.latest_lbl.setText(t("update_latest_version", v="?"))
            self.status_lbl.setText(t("update_status_failed"))
            self.status_lbl.setStyleSheet("color:#f87171;font-weight:bold;")
            self.notes_view.setPlainText(
                t("update_check_error", err=result.get("error", "")),
            )
            return

        latest = result.get("tag", "")
        self._latest_url = result.get("html_url", "")
        self.view_release_btn.setEnabled(bool(self._latest_url))
        self.latest_lbl.setText(t("update_latest_version", v=latest or "?"))

        local = _load_local_version()
        if local and latest and local == latest:
            self.status_lbl.setText(t("update_status_uptodate"))
            self.status_lbl.setStyleSheet("color:#4ade80;font-weight:bold;")
        elif local is None:
            # No manifest — can't tell if user is on this release or older.
            self.status_lbl.setText(t("update_status_unknown"))
            self.status_lbl.setStyleSheet("color:#fbbf24;font-weight:bold;")
        else:
            self.status_lbl.setText(t("update_status_available"))
            self.status_lbl.setStyleSheet("color:#fbbf24;font-weight:bold;")

        body = result.get("body", "").strip()
        if body:
            self.notes_view.setMarkdown(body)
        else:
            self.notes_view.setPlainText(t("update_no_release_notes"))

    def _open_release_page(self) -> None:
        if self._latest_url:
            QDesktopServices.openUrl(QUrl(self._latest_url))

    def _confirm_and_run(self, conflict_flag: str):
        ok = QMessageBox.question(
            self,
            t("update_title"),
            t("update_confirm"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ok == QMessageBox.Yes:
            self._run(["update", conflict_flag])

    def _set_busy(self, busy: bool) -> None:
        super()._set_busy(busy)
        self.dry_btn.setEnabled(not busy)
        self.run_keep_btn.setEnabled(not busy)
        self.run_overwrite_btn.setEnabled(not busy)
        self.check_btn.setEnabled(not busy)

    def closeEvent(self, ev):
        # Without wait(), Qt warns about destroying a running QThread.
        if self._check_thread is not None and self._check_thread.isRunning():
            self._check_thread.wait(2000)
        super().closeEvent(ev)


# Public helpers for app.py.


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
