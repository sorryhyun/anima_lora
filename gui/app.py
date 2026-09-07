"""Anima LoRA GUI — main window, dark theme, and entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import toml
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from gui import get_setting
from gui import daemon as gui_daemon
from gui import theme as gui_theme
from gui.dialogs import GuidebookDialog, _guidebook_path
from gui.gpu_status import GpuStatusBar
from gui.widgets import LazyTabHolder, action_button, wrap_tooltip
from gui.i18n import load_language, t
from gui.settings_dialog import SettingsDialog
from gui.tabs.easycontrol_tab import EasyControlTab
from gui.tabs.image_tab import ImageViewerTab
from gui.tabs.merge_tab import MergeTab
from gui.tabs.methods_tab import MethodsTab
from gui.tabs.preprocess import PreprocessingTab
from gui.tabs.queue_tab import QueueTab
from gui.tensorboard import TensorBoardTab
from gui.system_dialog import (
    GITHUB_ISSUES_URL,
    GITHUB_REPO_URL,
    check_for_update_async,
    open_models_dialog,
    open_update_dialog,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_PATH = Path(__file__).resolve().parent / "icon.ico"

# Keeps the live MainWindow alive across the in-place rebuild that applies a
# language change (main() seeds it; MainWindow._reload_ui swaps it).
_WINDOW: MainWindow | None = None


def _dark(app: QApplication):
    """Apply the user's chosen named theme (Dark / Light / Sepia)."""
    gui_theme.apply_theme(app)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_title"))
        self.resize(1100, 750)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        central = QWidget()
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        lang_bar = QHBoxLayout()
        if ICON_PATH.exists():
            icon_label = QLabel()
            pix = QPixmap(str(ICON_PATH))
            if not pix.isNull():
                icon_label.setPixmap(
                    pix.scaled(
                        QSize(28, 28),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            icon_label.setContentsMargins(4, 0, 6, 0)
            lang_bar.addWidget(icon_label)
        self.guide_btn = action_button(
            t("guidebook"),
            variant="success",
            tooltip=t("guidebook_tooltip"),
            on_click=self._open_guidebook,
        )
        lang_bar.addWidget(self.guide_btn)

        self.models_btn = QPushButton(t("models_btn"))
        self.models_btn.setToolTip(t("models_btn_tooltip"))
        self.models_btn.clicked.connect(
            lambda: open_models_dialog(
                self, on_models_changed=self._reload_image_tab_kb
            )
        )
        lang_bar.addWidget(self.models_btn)

        self.update_btn = QPushButton(t("update_btn"))
        self.update_btn.setToolTip(t("update_btn_tooltip"))
        self.update_btn.clicked.connect(lambda: open_update_dialog(self))
        lang_bar.addWidget(self.update_btn)
        # Background check; paints the button amber when a newer release exists.
        self._update_check_thread = check_for_update_async(
            self, self._show_update_available
        )

        self.issues_btn = QPushButton(t("report_issue"))
        self.issues_btn.setToolTip(t("report_issue_tooltip"))
        self.issues_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_ISSUES_URL))
        )
        lang_bar.addWidget(self.issues_btn)

        # Top-bar toggle, not a tab: the daemon job queue is global; overlays are mutually exclusive.
        self.queue_btn = QPushButton(t("tab_queue"))
        self.queue_btn.setCheckable(True)
        self._queue_idle_style = gui_theme.nav_button_qss("queue")
        self._queue_active_style = gui_theme.nav_button_qss("queue_on")
        self.queue_btn.toggled.connect(self._toggle_queue_view)
        lang_bar.addWidget(self.queue_btn)

        # Top-bar toggle, not a tab: the run list is global, shared across every method.
        self.tensorboard_btn = QPushButton(t("tab_tensorboard"))
        self.tensorboard_btn.setCheckable(True)
        self._tensorboard_idle_style = gui_theme.nav_button_qss("tensorboard")
        self._tensorboard_active_style = gui_theme.nav_button_qss("tensorboard_on")
        self.tensorboard_btn.toggled.connect(self._toggle_tensorboard)
        lang_bar.addWidget(self.tensorboard_btn)

        lang_bar.addStretch()
        self.settings_btn = QPushButton(t("settings_btn"))
        self.settings_btn.setToolTip(t("settings_btn_tooltip"))
        self.settings_btn.clicked.connect(self._open_settings)
        lang_bar.addWidget(self.settings_btn)
        main_lay.addLayout(lang_bar)

        # Overlays share a QStackedWidget with the tab set; all widgets stay alive
        # across switches so subprocess state and log buffers survive toggling.
        self._tb_tab = TensorBoardTab()

        # Built before ConfigTab so the Train auto-chain can flush this tab's
        # GUI preprocess settings to the selected method before it preprocesses.
        self._preprocess_tab = PreprocessingTab()

        self.tabs = QTabWidget()
        # Config = MethodsTab over the shipped LoRA family + Turbo distiller.
        self.tabs.addTab(
            MethodsTab(
                tb_panel=self._tb_tab.panel,
                flat_methods=("lora", "tlora", "hydralora"),
                distill_methods=("turbo",),
                preprocess_tab=self._preprocess_tab,
            ),
            t("tab_config"),
        )
        self.tabs.addTab(self._preprocess_tab, t("tab_preprocess"))
        # Every tab after Config is a LazyTabHolder: built on first open, keeping
        # the launch path to Config + Preprocess only.
        self._image_tab = LazyTabHolder(
            lambda: ImageViewerTab(preprocess_tab=self._preprocess_tab)
        )
        self.tabs.addTab(self._image_tab, t("tab_images"))
        self.tabs.addTab(LazyTabHolder(MergeTab), t("tab_merge"))
        # EasyControl keeps a dedicated tab (own preprocess/dataset lifecycle).
        self.tabs.addTab(
            LazyTabHolder(
                lambda: MethodsTab(
                    tb_panel=self._tb_tab.panel,
                    flat_methods=("chimera", "soft_tokens"),
                    distill_methods=("soup",),
                )
            ),
            t("tab_experimental"),
        )
        self.tabs.addTab(LazyTabHolder(EasyControlTab), t("tab_easycontrol"))

        self._queue_tab = QueueTab()

        self.tab_stack = QStackedWidget()
        self.tab_stack.addWidget(self.tabs)
        self.tab_stack.addWidget(self._tb_tab)
        self.tab_stack.addWidget(self._queue_tab)
        main_lay.addWidget(self.tab_stack)

        # Live GPU utilisation/VRAM footer, always visible below the tab set.
        self._gpu_bar = GpuStatusBar()
        main_lay.addWidget(self._gpu_bar)
        self.setCentralWidget(central)

        self._update_tensorboard_btn_style(False)
        self._update_queue_btn_style(False)

        # App-wide filter for uniform right-click + wrapped tooltips. Must install
        # LAST, after the tree is built: it only handles ContextMenu/ToolTip (can't
        # fire before the window is shown), but installed earlier it would also see
        # every construction-time event (ChildAdded/Polish/...) — ~82k Python
        # round-trips per launch, measurably slowing startup. Removed in closeEvent
        # so a _reload_ui rebuild doesn't stack filters.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def closeEvent(self, event):
        # Drop the app-wide filter so a _reload_ui rebuild doesn't leave a dead window filtering events.
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        # Without this, closing the window leaves training subprocesses orphaned, still holding VRAM.
        for i in range(self.tabs.count()):
            cleanup = getattr(self.tabs.widget(i), "cleanup_subprocess", None)
            if callable(cleanup):
                cleanup()
        self._tb_tab.cleanup_subprocess()
        self._queue_tab.cleanup_subprocess()
        self._gpu_bar.cleanup()
        super().closeEvent(event)

    def _reload_image_tab_kb(self) -> None:
        """Refresh the Images tab's tag KB after a Models-dialog download."""
        if self._image_tab.inner is not None:
            self._image_tab.inner.reload_tag_knowledge_base()

    def _show_update_available(self, latest_tag: str) -> None:
        self.update_btn.setText(t("update_btn_available"))
        self.update_btn.setToolTip(t("update_btn_available_tooltip", v=latest_tag))
        self.update_btn.setStyleSheet(gui_theme.nav_button_qss("update"))

    def _clear_overlay_toggle(self, btn: QPushButton, style_fn) -> None:
        """Silently un-check an overlay toggle without re-firing its handler."""
        if btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            style_fn(False)

    def _toggle_tensorboard(self, on: bool):
        if on:
            self._clear_overlay_toggle(self.queue_btn, self._update_queue_btn_style)
            self.tab_stack.setCurrentWidget(self._tb_tab)
        else:
            self.tab_stack.setCurrentWidget(self.tabs)
        self._update_tensorboard_btn_style(on)

    def _update_tensorboard_btn_style(self, on: bool):
        self.tensorboard_btn.setStyleSheet(
            self._tensorboard_active_style if on else self._tensorboard_idle_style
        )

    def _toggle_queue_view(self, on: bool):
        if on:
            self._clear_overlay_toggle(
                self.tensorboard_btn, self._update_tensorboard_btn_style
            )
            self.tab_stack.setCurrentWidget(self._queue_tab)
        else:
            self.tab_stack.setCurrentWidget(self.tabs)
        self._update_queue_btn_style(on)

    def _update_queue_btn_style(self, on: bool):
        self.queue_btn.setStyleSheet(
            self._queue_active_style if on else self._queue_idle_style
        )

    def eventFilter(self, obj, event):  # noqa: N802 — Qt event handler name
        """Intercept every right-click to show our menu instead of the target
        widget's default one, and re-show long tooltips wrapped to a bounded
        width."""
        if event.type() == QEvent.ContextMenu:
            self._show_context_menu(event.globalPos())
            return True
        if event.type() == QEvent.ToolTip:
            tip = obj.toolTip() if hasattr(obj, "toolTip") else ""
            wrapped = wrap_tooltip(tip)
            if wrapped is not None and wrapped != tip:
                QToolTip.showText(event.globalPos(), wrapped, obj)
                return True
        return super().eventFilter(obj, event)

    def _show_context_menu(self, global_pos):
        """Walk up from the widget under the cursor; the first ancestor exposing
        a callable ``app_context_menu(target, global_pos)`` supplies the menu,
        else fall back to the app default."""
        target = QApplication.widgetAt(global_pos)
        w = target
        while w is not None:
            provider = getattr(w, "app_context_menu", None)
            if callable(provider):
                menu = provider(target, global_pos)
                if menu is not None:
                    menu.exec(global_pos)
                    return
                break
            w = w.parentWidget()
        self._show_app_menu(global_pos)

    def _show_app_menu(self, global_pos):
        """The default right-click menu — currently just a link to the repo."""
        menu = QMenu(self)
        visit = menu.addAction(t("visit_github"))
        visit.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_REPO_URL)))
        menu.exec(global_pos)

    def _open_guidebook(self):
        path = _guidebook_path()
        if not path.exists():
            QMessageBox.warning(
                self, t("guidebook"), t("guidebook_missing", path=str(path))
            )
            return
        dlg = GuidebookDialog(path, self)
        dlg.show()

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()
        if dlg.reload_requested:
            self._reload_ui()

    def _reload_ui(self):
        """Rebuild the main window in place to apply a language/theme change.
        The daemon owns running jobs, so only local UI state resets. New window
        is shown before the old closes so quitOnLastWindowClosed never fires."""
        global _WINDOW
        new = MainWindow()
        new.setGeometry(self.geometry())
        new.show()
        _WINDOW = new
        self.close()


def _ensure_source_image_dir() -> None:
    """Create the training source dir on launch so first-time users hit an
    empty folder rather than a confusing "no images found" error."""
    src = "image_dataset"
    # Read order matches load_path_overrides' precedence: a legacy base.toml key (read second) wins.
    for fname in ("preprocess.toml", "base.toml"):
        cfg_path = _REPO_ROOT / "configs" / fname
        try:
            if cfg_path.exists():
                raw = toml.loads(cfg_path.read_text(encoding="utf-8"))
                cfg_src = raw.get("source_image_dir")
                if isinstance(cfg_src, str) and cfg_src.strip():
                    src = cfg_src
        except (OSError, toml.TomlDecodeError):
            pass
    src_path = Path(src)
    if not src_path.is_absolute():
        src_path = _REPO_ROOT / src_path
    try:
        src_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"warn: could not create {src_path}: {e}", file=sys.stderr)


def main():
    load_language()
    _ensure_source_image_dir()
    gui_theme._prefer_cleartype_font_engine()
    # Pass fractional display scaling through (tiny text on HiDPI Windows otherwise).
    # Must be set before QApplication is constructed.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    _dark(app)
    # Set before the daemon spawns below so a fresh daemon logs at DEBUG.
    if get_setting("debug_mode", False):
        os.environ["ANIMA_DEBUG"] = "1"
    global _WINDOW
    _WINDOW = MainWindow()
    _WINDOW.show()
    # Deferred so a cold-start daemon boot doesn't block the window from appearing.
    # Best-effort: the Train button's own ensure_daemon() does the real wait-for-health.
    QTimer.singleShot(0, gui_daemon.ensure_daemon_quietly)
    sys.exit(app.exec())
