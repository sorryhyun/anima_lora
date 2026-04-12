"""GraftTab — GRAFT iteration curation with thumbnail grid and preview."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import GRAFT_DIR, ScaledImageLabel, _imgs, _load, _read, _save, _widget
from gui.i18n import t


THUMB = 220


class Thumbnail(QLabel):
    clicked = Signal(object)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.selected = False
        self.setFixedSize(THUMB, THUMB)
        self.setAlignment(Qt.AlignCenter)
        pm = QPixmap(str(path))
        if not pm.isNull():
            self.setPixmap(
                pm.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self.setToolTip(path.name)
        self._style()

    def toggle(self):
        self.selected = not self.selected
        self._style()

    def _style(self):
        c = "#e74c3c" if self.selected else "transparent"
        self.setStyleSheet(f"border:3px solid {c};background:#1e1e1e;")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.toggle()
            self.clicked.emit(self)


class GraftTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thumbs: list[Thumbnail] = []
        lay = QVBoxLayout(self)
        sp = QSplitter(Qt.Horizontal)

        # Left panel: iteration list + GRAFT config
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        hl = QHBoxLayout()
        hl.addWidget(QLabel(t("iterations")))
        ref = QPushButton("\u21bb")
        ref.setFixedWidth(28)
        ref.setToolTip(t("refresh"))
        ref.clicked.connect(self._refresh)
        hl.addWidget(ref)
        ll.addLayout(hl)

        self.il = QListWidget()
        self.il.currentTextChanged.connect(self._load_iter)
        ll.addWidget(self.il)

        gc = _load(GRAFT_DIR / "graft_config.toml")
        self._gcw: dict[str, QWidget] = {}
        if gc:
            box = QGroupBox(t("graft_config"))
            form = QFormLayout()
            for k, v in gc.items():
                w = _widget(v)
                self._gcw[k] = w
                form.addRow(k, w)
            box.setLayout(form)
            ll.addWidget(box)
            sb = QPushButton(t("save_graft_config"))
            sb.clicked.connect(self._save_gc)
            ll.addWidget(sb)

        sp.addWidget(left)

        # Right panel: toolbar + thumbnail grid + preview
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        for lbl, fn in [
            (t("select_all"), self._sel_all),
            (t("invert"), self._inv),
            (t("deselect"), self._desel),
        ]:
            b = QPushButton(lbl)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        self.stat = QLabel(t("n_images", n=0))
        bar.addWidget(self.stat)
        db = QPushButton(t("delete_selected"))
        db.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 12px;"
        )
        db.clicked.connect(self._delete)
        bar.addWidget(db)
        rl.addLayout(bar)

        vs = QSplitter(Qt.Vertical)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._gw = QWidget()
        self._gl = QGridLayout(self._gw)
        self._gl.setSpacing(8)
        sc.setWidget(self._gw)
        vs.addWidget(sc)

        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        self.prev = ScaledImageLabel()
        self.prev.setMinimumHeight(200)
        self.prev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pl.addWidget(self.prev)
        self.prev_info = QLabel()
        self.prev_info.setWordWrap(True)
        pl.addWidget(self.prev_info)
        vs.addWidget(pw)
        vs.setSizes([450, 300])

        rl.addWidget(vs)
        sp.addWidget(right)
        sp.setSizes([250, 750])
        lay.addWidget(sp)

        QShortcut(QKeySequence.Delete, self, self._delete)
        QShortcut(QKeySequence("Ctrl+A"), self, self._sel_all)
        self._refresh()

    def _refresh(self):
        self.il.clear()
        cd = GRAFT_DIR / "candidates"
        if not cd.exists():
            return
        ds = sorted(d.name for d in cd.iterdir() if d.is_dir())
        self.il.addItems(ds)
        if ds:
            self.il.setCurrentRow(len(ds) - 1)

    def _load_iter(self, name: str):
        self.thumbs.clear()
        while self._gl.count():
            w = self._gl.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not name:
            return
        for i, p in enumerate(_imgs(GRAFT_DIR / "candidates" / name)):
            th = Thumbnail(p)
            th.clicked.connect(self._on_click)
            self.thumbs.append(th)
            self._gl.addWidget(th, i // 4, i % 4)
        self._upd()

    def _on_click(self, thumb: Thumbnail):
        self._upd()
        pm = QPixmap(str(thumb.path))
        if not pm.isNull():
            self.prev.set_source(pm)
        info = thumb.path.name
        jp = thumb.path.with_suffix(".json")
        if jp.exists():
            try:
                m = json.loads(jp.read_text(encoding="utf-8"))
                info += f"  |  seed: {m.get('seed', '?')}"
                cap = m.get("caption", "")
                if cap:
                    info += f"\n{cap[:200]}"
            except (json.JSONDecodeError, OSError):
                pass
        self.prev_info.setText(info)

    def _upd(self):
        n = len(self.thumbs)
        s = sum(th.selected for th in self.thumbs)
        self.stat.setText(t("n_images_selected", n=n, s=s))

    def _sel_all(self):
        for th in self.thumbs:
            if not th.selected:
                th.toggle()
        self._upd()

    def _desel(self):
        for th in self.thumbs:
            if th.selected:
                th.toggle()
        self._upd()

    def _inv(self):
        for th in self.thumbs:
            th.toggle()
        self._upd()

    def _delete(self):
        sel = [th for th in self.thumbs if th.selected]
        if not sel:
            return
        if (
            QMessageBox.question(
                self,
                t("delete"),
                t("delete_confirm", n=len(sel)),
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        for th in sel:
            th.path.unlink(missing_ok=True)
            th.path.with_suffix(".json").unlink(missing_ok=True)
        cur = self.il.currentItem()
        if cur:
            self._load_iter(cur.text())

    def _save_gc(self):
        gc = _load(GRAFT_DIR / "graft_config.toml")
        _save(
            GRAFT_DIR / "graft_config.toml",
            {k: _read(w, gc.get(k)) for k, w in self._gcw.items()},
        )
        QMessageBox.information(self, t("saved"), t("graft_saved"))
