"""Models / Update dialogs — wrappers around `python tasks.py download-* / update`.

Both dialogs share the same shape: a row of action buttons, a status area, and
a streaming log fed by ``QProcess`` (same pattern as MergeTab). Only one job
runs at a time per dialog — buttons disable while busy and re-enable on finish.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

from PySide6.QtCore import QProcess, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import ROOT
from library import downloads as DL
from gui._paths import get_setting, set_setting
from gui.i18n import t
from gui.process import kill_process_tree, setup_kill_safe
from gui.theme import tok
from gui.widgets import apply_variant

# Rows come from the catalog, never from a table here: ``library/downloads.py``
# (Anima weights) and ``anime_tools.downloads`` (curation weights) each carry an
# Asset's repo, files, destination and an offline installed-probe. A duplicate
# path list in the GUI is a Download button that reports the wrong state the
# moment a row moves — which is exactly how ``models/mit/model.pth`` came to
# read as MISSING while sitting on disk.
#
# Only the rows that already had translations keep an i18n key; anything newer
# shows the catalog's own English title, which is mostly a proper noun anyway.
_TITLE_KEYS: dict[str, str] = {
    "anima_dit": "model_anima_dit",
    "anima_te": "model_anima_te",
    "anima_vae": "model_anima_vae",
    "pe_core": "model_pe",
    "vocab_pack": "model_vocab_pack",
    "sam3": "model_sam3",
    "mit_text": "model_mit",
    "pe_spatial": "model_pe_spatial",
    "danbooru_tags": "model_danbooru_tags",
    "tagger_backbone": "model_tagger",
}


def _label(asset) -> str:
    key = _TITLE_KEYS.get(asset.id)
    return t(key) if key else asset.title


def _tooltip(asset) -> str:
    lines = [f"{asset.repo} → {asset.location}"]
    if asset.used_by:
        lines.append(t("models_used_by", what=asset.used_by))
    if asset.notes:
        lines.append(asset.notes)
    return "\n\n".join(lines)


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


class _HFLoginThread(QThread):
    """Validate + persist a HuggingFace token off the UI thread.

    Mirrors ``hf auth login`` non-interactively: ``huggingface_hub.login()``
    checks the token via ``whoami`` and writes it to the HF token cache, so
    every subsequent ``hf download`` invocation (which the model downloads
    shell out to) is authenticated without the user opening a terminal.

    Emits ``done`` with a dict: ``ok`` (bool), ``name`` (username on success),
    ``error`` (str, populated only on failure).
    """

    done = Signal(dict)

    def __init__(self, token: str, parent=None):
        super().__init__(parent)
        self._token = token

    def run(self) -> None:  # noqa: D401 — Qt override
        result = {"ok": False, "name": "", "error": ""}
        try:
            from huggingface_hub import login, whoami

            login(token=self._token, add_to_git_credential=False)
            info = whoami()
            result["ok"] = True
            result["name"] = info.get("name", "") if isinstance(info, dict) else ""
        except Exception as e:  # invalid token / network / SDK error
            result["error"] = str(e)
        self.done.emit(result)


class _CatalogPanel(QWidget):
    """One tab: a "fetch everything" button over one catalog half's rows.

    Rows scroll inside the tab, so a catalog that grows (the package's is at 15
    and counting) never pushes the log pane off the dialog. Downloads run on the
    owning dialog's single QProcess — one job at a time across both tabs.
    """

    def __init__(self, dialog: "ModelsDialog", intro_key: str, rows_fn, all_fn):
        super().__init__(dialog)
        self._dialog = dialog
        self._assets_fn = rows_fn
        self._all_fn = all_fn
        # (asset, status_label, button) — refreshed together after any run,
        # because one target installs several rows.
        self._rows: list[tuple[object, QLabel, QPushButton]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        intro = QLabel(t(intro_key))
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{tok('text_dim')};")
        outer.addWidget(intro)

        all_row = QHBoxLayout()
        self.all_btn = QPushButton()
        apply_variant(self.all_btn, "success")
        self.all_btn.clicked.connect(self._download_all)
        all_row.addWidget(self.all_btn)
        all_row.addStretch()
        outer.addLayout(all_row)

        body = QWidget()
        rows_lay = QVBoxLayout(body)
        rows_lay.setContentsMargins(0, 0, 0, 0)
        for asset in self.assets():
            rows_lay.addLayout(self._build_row(asset))
        rows_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        self.refresh()

    def assets(self) -> tuple:
        return self._assets_fn()

    def _build_row(self, asset) -> QHBoxLayout:
        row = QHBoxLayout()

        name = QLabel(("🔒 " if asset.gated else "") + _label(asset))
        name.setMinimumWidth(280)
        name.setToolTip(_tooltip(asset))
        row.addWidget(name)

        status = QLabel()
        status.setMinimumWidth(110)
        row.addWidget(status)
        row.addStretch()

        btn = QPushButton()
        btn.clicked.connect(
            lambda _checked=False, a=asset: self._dialog.run_download(
                ["download-model", a.id]
            )
        )
        row.addWidget(btn)

        self._rows.append((asset, status, btn))
        return row

    def refresh(self) -> None:
        """Re-probe every row. Offline (a path stat or a hub-cache lookup), so
        this is cheap enough to run after every job."""
        for asset, status, btn in self._rows:
            installed = asset.installed
            status.setText(t("models_installed") if installed else t("models_missing"))
            status.setStyleSheet(
                f"color:{tok('ok')};" if installed else f"color:{tok('err')};"
            )
            btn.setText(t("models_redownload") if installed else t("models_download"))
        label, args = self._all_fn(self.assets())
        self.all_btn.setText(label)
        self.all_btn.setEnabled(args is not None)

    def _download_all(self) -> None:
        _label_text, args = self._all_fn(self.assets())
        if args is not None:
            self._dialog.run_download(args)

    def set_busy(self, busy: bool) -> None:
        _label_text, args = self._all_fn(self.assets())
        self.all_btn.setEnabled(not busy and args is not None)
        for _asset, _status, btn in self._rows:
            btn.setEnabled(not busy)


def _anima_all(_assets) -> tuple[str, list[str] | None]:
    """The Anima tab's top button is the first-run set, not "every row": it
    deliberately also pulls the two curation rows a default preprocess needs
    (tagger checkpoint, tag KB) while SAM3, OCR and the vocab pack stay opt-in."""
    return t("models_download_all"), ["download-models"]


def _missing_all(assets) -> tuple[str, list[str] | None]:
    """The curation tab's: only what is missing. The full catalog is several GB
    and most of it is opt-in per stage."""
    missing = [a.id for a in assets if not a.installed]
    if not missing:
        return t("models_all_installed"), None
    return t("models_download_missing", n=len(missing)), ["download-model", *missing]


class ModelsDialog(_StreamingDialog):
    """Model downloads: a HuggingFace token field, two catalog tabs, one log.

    The tabs are the two halves of the catalog — ``library/downloads.py``'s
    Anima rows and ``anime_tools.downloads``' curation rows. Splitting them
    inside one modal keeps each list short and keeps the token field, the log
    and the single QProcess shared, which two dialogs could not do.
    """

    # Emitted after any successful (exit_code 0) download run so live tabs can
    # pick up freshly-installed assets — e.g. ImageViewerTab reloading the
    # danbooru tag KB — without an app restart.
    models_changed = Signal()

    def __init__(self, parent=None):
        self._panels: list[_CatalogPanel] = []
        self._login_thread: _HFLoginThread | None = None
        super().__init__(t("models_title"), parent)
        self.resize(780, 620)

    def run_download(self, args: list[str]) -> None:
        """Panels call this; the dialog owns the one QProcess."""
        self._run(args)

    def _build_actions(self, layout: QVBoxLayout) -> None:
        self._build_auth(layout)

        self.tabs = QTabWidget()
        for label_key, intro_key, rows_fn, all_fn in (
            ("models_tab_anima", "models_intro", DL.catalog, _anima_all),
            (
                "models_tab_curation",
                "curation_models_intro",
                DL.curation_catalog,
                _missing_all,
            ),
        ):
            panel = _CatalogPanel(self, intro_key, rows_fn, all_fn)
            self._panels.append(panel)
            self.tabs.addTab(panel, t(label_key))
        layout.addWidget(self.tabs, 1)

    # -- auth --------------------------------------------------------------

    def _build_auth(self, layout: QVBoxLayout) -> None:
        # Token persisted via huggingface_hub (same cache `hf auth login`
        # writes); the gated SAM3 and tagger-backbone repos need it.
        auth_row = QHBoxLayout()
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText(t("models_hf_token_placeholder"))
        self.token_edit.returnPressed.connect(self._authenticate)
        auth_row.addWidget(self.token_edit, 1)
        self.auth_btn = QPushButton(t("models_hf_authenticate"))
        self.auth_btn.clicked.connect(self._authenticate)
        auth_row.addWidget(self.auth_btn)
        layout.addLayout(auth_row)

        self.auth_status = QLabel()
        self.auth_status.setWordWrap(True)
        layout.addWidget(self.auth_status)

        hint = QLabel(t("models_hf_token_hint"))
        hint.setWordWrap(True)
        hint.setOpenExternalLinks(True)
        hint.setStyleSheet(f"color:{tok('text_dim')};font-size:11px;margin-bottom:6px;")
        layout.addWidget(hint)
        self._refresh_auth_status()

    def _refresh_auth_status(self) -> None:
        """Show whether a token is already cached (no network call)."""
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
        if token:
            self.auth_status.setText(t("models_hf_token_present"))
            self.auth_status.setStyleSheet(f"color:{tok('ok')};")
        else:
            self.auth_status.setText(t("models_hf_not_authenticated"))
            self.auth_status.setStyleSheet(f"color:{tok('text_dim')};")

    def _authenticate(self) -> None:
        token = self.token_edit.text().strip()
        if not token:
            self.auth_status.setText(t("models_hf_token_empty"))
            self.auth_status.setStyleSheet(f"color:{tok('err')};")
            return
        if self._login_thread is not None and self._login_thread.isRunning():
            return
        self.auth_btn.setEnabled(False)
        self.auth_status.setText(t("models_hf_authenticating"))
        self.auth_status.setStyleSheet(f"color:{tok('text_dim')};")
        self._login_thread = _HFLoginThread(token, self)
        self._login_thread.done.connect(self._on_login_result)
        self._login_thread.start()

    def _on_login_result(self, result: dict) -> None:
        self.auth_btn.setEnabled(True)
        if result.get("ok"):
            self.token_edit.clear()  # don't leave the secret sitting in the field
            self.auth_status.setText(
                t("models_hf_logged_in", name=result.get("name", ""))
            )
            self.auth_status.setStyleSheet(f"color:{tok('ok')};font-weight:bold;")
        else:
            self.auth_status.setText(
                t("models_hf_login_failed", err=result.get("error", ""))
            )
            self.auth_status.setStyleSheet(f"color:{tok('err')};")

    # -- run lifecycle -----------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        super()._set_busy(busy)
        for panel in self._panels:
            panel.set_busy(busy)

    def _after_finished(self, exit_code: int) -> None:
        # Both tabs: `download-models` spans them, and the Anima tab's button
        # installs curation rows.
        for panel in self._panels:
            panel.refresh()

        if exit_code != 0:
            QMessageBox.warning(
                self,
                t("models_failed_title"),
                t("models_failed_message", code=exit_code),
            )
        else:
            self.models_changed.emit()
            QMessageBox.information(
                self,
                t("models_done_title"),
                t("models_done_message"),
            )

    def closeEvent(self, ev):
        # Without wait(), Qt warns about destroying a running QThread.
        if self._login_thread is not None and self._login_thread.isRunning():
            self._login_thread.wait(2000)
        super().closeEvent(ev)


GITHUB_REPO = "sorryhyun/anima_lora"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}"
GITHUB_ISSUES_URL = f"{GITHUB_REPO_URL}/issues"
RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
MANIFEST_FILE = ROOT / ".anima_release.json"

# Release-tag cache (gui_settings.json) so the on-launch badge check doesn't hit GitHub every start; 6h balances freshness vs per-launch network cost.
UPDATE_CACHE_TTL_SECONDS = 6 * 3600
_UPDATE_CACHE_KEY = "update_check"


def _load_local_version() -> str | None:
    """Read the baseline tag from .anima_release.json, or None if absent."""
    if not MANIFEST_FILE.exists():
        return None
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("version")


def _load_cached_latest_tag(ttl: int = UPDATE_CACHE_TTL_SECONDS) -> str | None:
    entry = get_setting(_UPDATE_CACHE_KEY)
    if not isinstance(entry, dict):
        return None
    tag = entry.get("latest_tag")
    checked_at = entry.get("checked_at")
    if not isinstance(tag, str) or not isinstance(checked_at, (int, float)):
        return None
    if time.time() - float(checked_at) > ttl:
        return None
    return tag or None


def _save_cached_latest_tag(tag: str) -> None:
    if not tag:
        return
    set_setting(_UPDATE_CACHE_KEY, {"latest_tag": tag, "checked_at": int(time.time())})


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
            if result["tag"]:
                _save_cached_latest_tag(result["tag"])
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
        # Which run is in flight, so _after_finished can pick the right post-run feedback.
        self._last_run_kind: str | None = None  # "real" | "dry" | None
        super().__init__(t("update_title"), parent)
        self._kick_check()

    def _build_actions(self, layout: QVBoxLayout) -> None:
        version_row = QHBoxLayout()
        version_row.setSpacing(12)

        local = _load_local_version()
        local_str = local if local else t("update_no_baseline")
        self.current_lbl = QLabel(t("update_current_version", v=local_str))
        version_row.addWidget(self.current_lbl)

        self.latest_lbl = QLabel(t("update_latest_version", v="…"))
        version_row.addWidget(self.latest_lbl)

        self.status_lbl = QLabel(t("update_status_checking"))
        self.status_lbl.setStyleSheet(f"color:{tok('text_dim')};font-weight:bold;")
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

        notes_label = QLabel(t("update_release_notes"))
        notes_label.setStyleSheet(f"color:{tok('text_dim')};margin-top:4px;")
        layout.addWidget(notes_label)

        self.notes_view = QTextBrowser()
        self.notes_view.setOpenExternalLinks(True)
        self.notes_view.setMaximumHeight(180)
        self.notes_view.document().setDefaultStyleSheet(
            f"a {{ color: {tok('link')}; text-decoration: underline; }}"
            f"code {{ background:{tok('panel')}; padding:1px 4px; border-radius:3px; }}"
            f"pre {{ background:{tok('panel')}; padding:6px; border-radius:4px; }}"
        )
        self.notes_view.setStyleSheet(
            f"QTextBrowser {{ background:{tok('base')}; color:{tok('text')}; "
            f"border:1px solid {tok('border_dim')}; padding:8px; }}"
        )
        self.notes_view.setPlaceholderText(t("update_status_checking"))
        layout.addWidget(self.notes_view)

        warn = QLabel(t("update_warning"))
        warn.setWordWrap(True)
        warn.setStyleSheet(
            f"padding:8px; border-radius:3px; background:#3d2e0a; color:{tok('warn')};"
        )
        layout.addWidget(warn)

        row = QHBoxLayout()
        # Dry-run uses --keep-conflicts: stdin isn't a TTY under QProcess, so
        # without a non-interactive flag the script would block on input().
        self.dry_btn = QPushButton(t("update_dry_run"))
        self.dry_btn.clicked.connect(self._start_dry_run)
        row.addWidget(self.dry_btn)

        # Two run buttons make the conflict policy explicit instead of an
        # invisible interactive prompt that the GUI can't service.
        self.run_keep_btn = QPushButton(t("update_run_keep"))
        apply_variant(self.run_keep_btn, "info")
        self.run_keep_btn.clicked.connect(
            lambda: self._confirm_and_run("--keep-conflicts")
        )
        row.addWidget(self.run_keep_btn)

        self.run_overwrite_btn = QPushButton(t("update_run_overwrite"))
        apply_variant(self.run_overwrite_btn, "success")
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
        self.status_lbl.setStyleSheet(f"color:{tok('text_dim')};font-weight:bold;")
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
            self.status_lbl.setStyleSheet(f"color:{tok('err')};font-weight:bold;")
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
            self.status_lbl.setStyleSheet(f"color:{tok('ok')};font-weight:bold;")
        elif local is None:
            # No manifest — can't tell if user is on this release or older.
            self.status_lbl.setText(t("update_status_unknown"))
            self.status_lbl.setStyleSheet(f"color:{tok('warn')};font-weight:bold;")
        else:
            self.status_lbl.setText(t("update_status_available"))
            self.status_lbl.setStyleSheet(f"color:{tok('warn')};font-weight:bold;")

        body = result.get("body", "").strip()
        if body:
            self.notes_view.setMarkdown(body)
        else:
            self.notes_view.setPlainText(t("update_no_release_notes"))

    def _open_release_page(self) -> None:
        if self._latest_url:
            QDesktopServices.openUrl(QUrl(self._latest_url))

    def _start_dry_run(self) -> None:
        self._last_run_kind = "dry"
        self._run(["update", "--dry-run", "--keep-conflicts"])

    def _confirm_and_run(self, conflict_flag: str):
        ok = QMessageBox.question(
            self,
            t("update_title"),
            t("update_confirm"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ok == QMessageBox.Yes:
            self._last_run_kind = "real"
            self._run(["update", conflict_flag])

    def _after_finished(self, exit_code: int) -> None:
        # Promote post-run state into the persistent UI instead of burying it in the streaming log.
        kind = self._last_run_kind
        self._last_run_kind = None
        if exit_code != 0:
            QMessageBox.warning(
                self,
                t("update_failed_title"),
                t("update_failed_message", code=exit_code),
            )
            return
        if kind == "dry":
            QMessageBox.information(
                self,
                t("update_dryrun_done_title"),
                t("update_dryrun_done_message"),
            )
            return
        if kind == "real":
            new_version = _load_local_version() or "?"
            self.current_lbl.setText(t("update_current_version", v=new_version))
            self.status_lbl.setText(t("update_success_badge", v=new_version))
            self.status_lbl.setStyleSheet(f"color:{tok('ok')};font-weight:bold;")
            QMessageBox.information(
                self,
                t("update_success_title"),
                t("update_success_message", v=new_version),
            )

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


def open_models_dialog(parent=None, on_models_changed=None):
    dlg = ModelsDialog(parent)
    if on_models_changed is not None:
        dlg.models_changed.connect(on_models_changed)
    dlg.exec()


def open_update_dialog(parent=None):
    UpdateDialog(parent).exec()


def check_for_update_async(parent, on_available) -> QThread | None:
    """Fire a non-blocking update check used by the top-bar update badge.

    Skips entirely when ``.anima_release.json`` is missing — without a
    baseline we can't tell whether the user is already on the latest tag,
    and a false "update available" badge is worse than no badge.

    Uses the 6h ``gui_settings.json`` cache to avoid a network round-trip
    on every launch. ``on_available(latest_tag)`` is invoked only when a
    newer tag is detected; the caller is responsible for keeping the
    returned ``QThread`` alive (parent it on a widget) so Qt doesn't tear
    it down mid-fetch.
    """
    local = _load_local_version()
    if local is None:
        return None
    cached = _load_cached_latest_tag()
    if cached is not None:
        if cached != local:
            on_available(cached)
        return None

    thread = _UpdateCheckThread(parent)

    def _handler(result: dict) -> None:
        if not result.get("ok"):
            return
        latest = result.get("tag", "") or ""
        if latest and latest != local:
            on_available(latest)

    thread.finished_check.connect(_handler)
    thread.start()
    return thread


__all__ = [
    "GITHUB_ISSUES_URL",
    "GITHUB_REPO_URL",
    "ModelsDialog",
    "UpdateDialog",
    "check_for_update_async",
    "open_models_dialog",
    "open_update_dialog",
]
