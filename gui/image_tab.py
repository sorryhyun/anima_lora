"""ImageViewerTab — dataset image browser with caption display."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from gui import ScaledImageLabel, _image_dirs, _imgs
from gui.i18n import t


class ImageViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        self._images: list[Path] = []
        self._dirs = _image_dirs()
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(t("directory")))
        self.dc = QComboBox()
        self.dc.addItems(self._dirs)
        self.dc.currentTextChanged.connect(self._load_dir)
        top.addWidget(self.dc, 1)
        self.cnt = QLabel()
        top.addWidget(self.cnt)
        lay.addLayout(top)

        sp = QSplitter(Qt.Horizontal)
        self.fl = QListWidget()
        self.fl.currentRowChanged.connect(self._show)
        sp.addWidget(self.fl)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.img = ScaledImageLabel()
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setMinimumSize(400, 400)
        rl.addWidget(self.img, 1)
        rl.addWidget(QLabel(t("caption")))
        self.cap = QTextEdit()
        self.cap.setReadOnly(True)
        self.cap.setMaximumHeight(120)
        rl.addWidget(self.cap)
        sp.addWidget(right)
        sp.setSizes([220, 750])
        lay.addWidget(sp)

        QShortcut(QKeySequence("Right"), self, lambda: self._nav(1))
        QShortcut(QKeySequence("Left"), self, lambda: self._nav(-1))
        if self._dirs:
            self._load_dir(self.dc.currentText())

    def _load_dir(self, name: str):
        d = self._dirs.get(name)
        if not d:
            return
        self._images = _imgs(d)
        self.fl.clear()
        for p in self._images:
            self.fl.addItem(p.stem)
        self.cnt.setText(t("n_images", n=len(self._images)))
        if self._images:
            self.fl.setCurrentRow(0)

    def _show(self, row: int):
        if not 0 <= row < len(self._images):
            return
        p = self._images[row]
        pm = QPixmap(str(p))
        if not pm.isNull():
            self.img.set_source(pm)
        cp = p.with_suffix(".txt")
        self.cap.setPlainText(cp.read_text(encoding="utf-8") if cp.exists() else t("no_caption"))

    def _nav(self, d: int):
        r = self.fl.currentRow() + d
        if 0 <= r < self.fl.count():
            self.fl.setCurrentRow(r)
