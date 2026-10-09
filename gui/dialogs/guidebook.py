"""In-app guidebook viewer (top-bar Guidebook button, EasyControl adapter guide)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from gui.core.paths import ROOT
from gui.i18n import current_language, t

_GUIDELINES = ROOT / "docs" / "guidelines"
_GUIDEBOOK_BY_LANG: dict[str, Path] = {
    "en": _GUIDELINES / "guidebook.md",
    "ko": _GUIDELINES / "가이드북.md",
    "cn": _GUIDELINES / "指南书.md",
    "ja": _GUIDELINES / "ガイドブック.md",
}
_GUIDEBOOK_FALLBACK = _GUIDEBOOK_BY_LANG["en"]


def _guidebook_path() -> Path:
    return _GUIDEBOOK_BY_LANG.get(current_language(), _GUIDEBOOK_FALLBACK)


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
