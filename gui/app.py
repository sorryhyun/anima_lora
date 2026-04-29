"""Anima LoRA GUI — main window, dark theme, and entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui.adapter_tab import EasyControlTab, IPAdapterTab
from gui.config_tab import ConfigTab
from gui.i18n import (
    available_languages,
    current_language,
    load_language,
    save_language,
    t,
)
from gui.image_tab import ImageViewerTab
from gui.merge_tab import MergeTab
from gui.system_dialog import (
    GITHUB_ISSUES_URL,
    open_models_dialog,
    open_update_dialog,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDEBOOK_PATH = _REPO_ROOT / "docs" / "guidelines" / "가이드북.md"
ICON_PATH = _REPO_ROOT / "icon.ico"


LANG_NAMES = {"en": "English", "ko": "한국어"}


def _dark(app: QApplication):
    # Use a font that supports Korean glyphs on Windows
    font = QFont("Malgun Gothic", 9)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    p = QPalette()
    for role, color in [
        (QPalette.Window, QColor(30, 30, 30)),
        (QPalette.WindowText, QColor(220, 220, 220)),
        (QPalette.Base, QColor(25, 25, 25)),
        (QPalette.AlternateBase, QColor(35, 35, 35)),
        (QPalette.ToolTipBase, QColor(50, 50, 50)),
        (QPalette.ToolTipText, QColor(220, 220, 220)),
        (QPalette.Text, QColor(220, 220, 220)),
        (QPalette.Button, QColor(45, 45, 45)),
        (QPalette.ButtonText, QColor(220, 220, 220)),
        (QPalette.Highlight, QColor(60, 120, 200)),
        (QPalette.HighlightedText, QColor(255, 255, 255)),
        # Default Qt link blue (#0000ff-ish) is unreadable on dark bg.
        (QPalette.Link, QColor(0xFF, 0xB8, 0x6B)),
        (QPalette.LinkVisited, QColor(0xE6, 0x94, 0x4E)),
    ]:
        p.setColor(role, color)
    app.setPalette(p)
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold; border: 1px solid #444;
            border-radius: 4px; margin-top: 8px; padding-top: 16px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton { padding: 4px 12px; border: 1px solid #555; border-radius: 3px; }
        QPushButton:hover { background: #555; }
        QScrollArea { border: none; }
        QSplitter::handle { background: #444; }
        QLineEdit, QSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QListWidget {
            background: #2a2a2a; color: #dcdcdc; border: 1px solid #555; border-radius: 3px;
            padding: 2px 4px;
        }
        QComboBox QAbstractItemView {
            background: #2a2a2a; color: #dcdcdc; selection-background-color: #3c78c8;
        }
        QTabWidget::pane { border: 1px solid #444; }
        QTabBar::tab {
            background: #2a2a2a; color: #dcdcdc; border: 1px solid #444;
            padding: 4px 12px; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
        }
        QTabBar::tab:selected { background: #1e1e1e; }
        QTabBar::tab:hover { background: #3a3a3a; }
        QToolTip { max-width: 400px; }
    """)


class GuidebookDialog(QDialog):
    """In-app markdown viewer for the guidebook."""

    def __init__(self, md_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("guidebook"))
        self.resize(900, 720)
        self._md_path = md_path

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        # Resolve relative links/images against the markdown file's directory.
        self.browser.setSearchPaths([str(md_path.parent)])
        self.browser.document().setBaseUrl(
            QUrl.fromLocalFile(str(md_path.parent) + "/")
        )
        # Default anchor color is pure blue — illegible on the dark bg.
        self.browser.document().setDefaultStyleSheet(
            "a { color: #ffb86b; text-decoration: underline; }"
            "a:visited { color: #e6944e; }"
            "code { background:#2a2a2a; padding:1px 4px; border-radius:3px; }"
            "pre { background:#2a2a2a; padding:8px; border-radius:4px; }"
        )
        self.browser.setStyleSheet(
            "QTextBrowser { background:#1e1e1e; color:#dcdcdc; "
            "border:1px solid #444; padding:12px; }"
        )
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as e:
            text = f"# Error\n\nCould not read `{md_path}`:\n\n`{e}`"
        self.browser.setMarkdown(text)
        lay.addWidget(self.browser)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        open_ext = QPushButton(t("guidebook_open_external"))
        open_ext.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._md_path)))
        )
        close = QPushButton(t("guidebook_close"))
        close.clicked.connect(self.close)
        btn_bar.addWidget(open_ext)
        btn_bar.addWidget(close)
        lay.addLayout(btn_bar)


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

        # Language selector bar
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
        self.guide_btn = QPushButton(t("guidebook"))
        self.guide_btn.setToolTip(t("guidebook_tooltip"))
        self.guide_btn.setStyleSheet(
            "background:#16a085;color:white;font-weight:bold;padding:4px 12px;"
        )
        self.guide_btn.clicked.connect(self._open_guidebook)
        lang_bar.addWidget(self.guide_btn)

        self.models_btn = QPushButton(t("models_btn"))
        self.models_btn.setToolTip(t("models_btn_tooltip"))
        self.models_btn.clicked.connect(lambda: open_models_dialog(self))
        lang_bar.addWidget(self.models_btn)

        self.update_btn = QPushButton(t("update_btn"))
        self.update_btn.setToolTip(t("update_btn_tooltip"))
        self.update_btn.clicked.connect(lambda: open_update_dialog(self))
        lang_bar.addWidget(self.update_btn)

        self.issues_btn = QPushButton(t("report_issue"))
        self.issues_btn.setToolTip(t("report_issue_tooltip"))
        self.issues_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_ISSUES_URL))
        )
        lang_bar.addWidget(self.issues_btn)

        self.experimental_btn = QPushButton(t("experimental_features"))
        self.experimental_btn.setToolTip(t("experimental_features_tooltip"))
        self.experimental_btn.setCheckable(True)
        # Two visual states: idle (purple, advertises the toggle) vs active
        # (orange, signals "you're currently in experimental mode — click to
        # return"). Style is reapplied in `_update_experimental_btn_style`.
        self._experimental_idle_style = (
            "QPushButton { background:#8e44ad; color:white; "
            "font-weight:bold; padding:4px 12px; border:1px solid #8e44ad; "
            "border-radius:3px; }"
            "QPushButton:hover { background:#9b59b6; }"
        )
        self._experimental_active_style = (
            "QPushButton { background:#e67e22; color:white; "
            "font-weight:bold; padding:4px 12px; border:1px solid #e67e22; "
            "border-radius:3px; }"
            "QPushButton:hover { background:#f39c12; }"
        )
        self.experimental_btn.toggled.connect(self._toggle_experimental)
        lang_bar.addWidget(self.experimental_btn)

        lang_bar.addStretch()
        lang_bar.addWidget(QLabel(t("language")))
        self.lang_combo = QComboBox()
        for code in available_languages():
            self.lang_combo.addItem(LANG_NAMES.get(code, code), code)
        self.lang_combo.setCurrentIndex(available_languages().index(current_language()))
        self.lang_combo.currentIndexChanged.connect(self._change_lang)
        self.lang_combo.setFixedWidth(100)
        lang_bar.addWidget(self.lang_combo)
        main_lay.addLayout(lang_bar)

        # Two parallel tab sets share a QStackedWidget so the experimental
        # button swaps the visible tab bar in place — same window, no popup.
        # Both sets stay alive across switches so subprocess state and log
        # buffers survive toggling between modes.
        # Standard set: bakeable methods only — plain LoRA (with hardware
        # variants), OrthoLoRA, T-LoRA (incl. T-LoRA + OrthoLoRA combo). All
        # produce checkpoints that can be merged into the base DiT. Non-
        # mergeable methods (HydraLoRA / ReFT / Postfix) and the APEX
        # distillation flow live behind the experimental toggle.
        self.tabs = QTabWidget()
        self.tabs.addTab(
            ConfigTab(methods=["lora", "ortholora", "tlora"]), t("tab_config")
        )
        self.tabs.addTab(ImageViewerTab(), t("tab_images"))
        self.tabs.addTab(MergeTab(), t("tab_merge"))

        # Experimental set: non-mergeable methods + APEX distillation +
        # image-conditioning adapters. The first tab hosts a single ConfigTab
        # with a method picker spanning HydraLoRA / ReFT / Postfix / APEX —
        # these all share the same training UI so one tab with a picker keeps
        # the tab bar manageable. IP-Adapter and EasyControl have their own
        # preprocess/dataset lifecycles, so they keep dedicated tabs.
        self.experimental_tabs = QTabWidget()
        self.experimental_tabs.addTab(
            ConfigTab(methods=["hydralora", "reft", "postfix", "apex"]),
            t("tab_methods"),
        )
        self.experimental_tabs.addTab(IPAdapterTab(), t("tab_ip_adapter"))
        self.experimental_tabs.addTab(EasyControlTab(), t("tab_easycontrol"))

        self.tab_stack = QStackedWidget()
        self.tab_stack.addWidget(self.tabs)
        self.tab_stack.addWidget(self.experimental_tabs)
        main_lay.addWidget(self.tab_stack)
        self.setCentralWidget(central)

        self._update_experimental_btn_style(False)

    def closeEvent(self, event):
        # Without this, closing the window leaves training subprocesses
        # (accelerate → train.py) orphaned and still holding VRAM.
        for tabs in (self.tabs, self.experimental_tabs):
            for i in range(tabs.count()):
                cleanup = getattr(tabs.widget(i), "cleanup_subprocess", None)
                if callable(cleanup):
                    cleanup()
        super().closeEvent(event)

    def _toggle_experimental(self, on: bool):
        self.tab_stack.setCurrentWidget(
            self.experimental_tabs if on else self.tabs
        )
        self._update_experimental_btn_style(on)

    def _update_experimental_btn_style(self, on: bool):
        self.experimental_btn.setStyleSheet(
            self._experimental_active_style if on
            else self._experimental_idle_style
        )

    def _open_guidebook(self):
        if not GUIDEBOOK_PATH.exists():
            QMessageBox.warning(
                self, t("guidebook"), t("guidebook_missing", path=str(GUIDEBOOK_PATH))
            )
            return
        dlg = GuidebookDialog(GUIDEBOOK_PATH, self)
        dlg.show()

    def _change_lang(self, idx: int):
        lang = self.lang_combo.itemData(idx)
        save_language(lang)
        QMessageBox.information(
            self,
            "Language" if lang == "en" else "언어",
            "Please restart the application to apply the language change."
            if current_language() == "en"
            else "언어 변경을 적용하려면 앱을 다시 시작해주세요.",
        )


def main():
    load_language()
    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    _dark(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
