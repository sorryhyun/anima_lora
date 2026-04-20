"""Anima LoRA GUI — main window, dark theme, and entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.config_tab import ConfigTab
from gui.graft_tab import GraftTab
from gui.i18n import (
    available_languages,
    current_language,
    load_language,
    save_language,
    t,
)
from gui.image_tab import ImageViewerTab

GUIDEBOOK_PATH = Path(__file__).resolve().parent.parent / "docs" / "guidelines" / "가이드북.md"


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_title"))
        self.resize(1100, 750)

        central = QWidget()
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        # Language selector bar
        lang_bar = QHBoxLayout()
        self.guide_btn = QPushButton(t("guidebook"))
        self.guide_btn.setToolTip(t("guidebook_tooltip"))
        self.guide_btn.setStyleSheet(
            "background:#16a085;color:white;font-weight:bold;padding:4px 12px;"
        )
        self.guide_btn.clicked.connect(self._open_guidebook)
        lang_bar.addWidget(self.guide_btn)
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

        self.tabs = QTabWidget()
        self.tabs.addTab(ConfigTab(), t("tab_config"))
        self.tabs.addTab(GraftTab(), t("tab_graft"))
        self.tabs.addTab(ImageViewerTab(), t("tab_images"))
        main_lay.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _open_guidebook(self):
        if not GUIDEBOOK_PATH.exists():
            QMessageBox.warning(
                self, t("guidebook"), t("guidebook_missing", path=str(GUIDEBOOK_PATH))
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(GUIDEBOOK_PATH)))

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
    _dark(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
