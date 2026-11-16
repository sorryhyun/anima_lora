"""Aspect-preserving zoom/pan image view + its magnified single-image dialog."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QWidget


class ScaledImageLabel(QLabel):
    """Aspect-preserving image view with Ctrl+wheel magnify.

    Zoom 1.0 fits the widget; Ctrl+scroll zooms around the cursor and a left
    drag pans. Drawn in ``paintEvent`` (not ``setPixmap``) so a zoomed image can
    overflow and be clipped. Zoom resets when the source changes.
    """

    MIN_ZOOM = 1.0
    MAX_ZOOM = 8.0
    _ZOOM_STEP = 1.25

    def __init__(self):
        super().__init__()
        self._src: QPixmap | None = None
        self._zoom = 1.0
        # Top-left of the drawn image in widget coords; None = auto-centered.
        self._tl: QPointF | None = None
        self._pan_last: QPointF | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

    def set_source(self, pm: QPixmap):
        self._src = pm
        self._reset_zoom()
        self.update()

    def clear(self):
        self._src = None
        self._reset_zoom()
        super().clear()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._tl = None
        self._pan_last = None

    def _fit_scale(self) -> float:
        sw, sh = self._src.width(), self._src.height()
        if sw <= 0 or sh <= 0:
            return 1.0
        return min(self.width() / sw, self.height() / sh)

    def _display_size(self) -> QSize:
        scale = self._fit_scale() * self._zoom
        return QSize(
            max(1, round(self._src.width() * scale)),
            max(1, round(self._src.height() * scale)),
        )

    def _current_tl(self) -> QPointF:
        if self._tl is not None:
            return self._tl
        d = self._display_size()
        return QPointF((self.width() - d.width()) / 2, (self.height() - d.height()) / 2)

    def _clamp_tl(self) -> None:
        if self._tl is None:
            return
        d = self._display_size()
        x, y = self._tl.x(), self._tl.y()
        if d.width() <= self.width():
            x = (self.width() - d.width()) / 2
        else:
            x = min(0.0, max(float(self.width() - d.width()), x))
        if d.height() <= self.height():
            y = (self.height() - d.height()) / 2
        else:
            y = min(0.0, max(float(self.height() - d.height()), y))
        self._tl = QPointF(x, y)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._clamp_tl()
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._src is None or self._src.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        tl = self._current_tl()
        d = self._display_size()
        painter.drawPixmap(
            int(round(tl.x())), int(round(tl.y())), d.width(), d.height(), self._src
        )

    def wheelEvent(self, ev):
        if not (ev.modifiers() & Qt.ControlModifier):
            super().wheelEvent(ev)
            return
        if self._src is None or self._src.isNull():
            return
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        new_zoom = self._zoom * (self._ZOOM_STEP if delta > 0 else 1 / self._ZOOM_STEP)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        if new_zoom == self._zoom:
            ev.accept()
            return
        # Keep the image point under the cursor fixed across the zoom.
        cursor = ev.position()
        tl = self._current_tl()
        old_scale = self._fit_scale() * self._zoom
        img_x = (cursor.x() - tl.x()) / old_scale
        img_y = (cursor.y() - tl.y()) / old_scale
        self._zoom = new_zoom
        new_scale = self._fit_scale() * self._zoom
        self._tl = QPointF(
            cursor.x() - img_x * new_scale, cursor.y() - img_y * new_scale
        )
        self._clamp_tl()
        self.update()
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._zoom > 1.0:
            self._pan_last = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_last is not None:
            pos = ev.position()
            delta = pos - self._pan_last
            self._pan_last = pos
            tl = self._current_tl()
            self._tl = QPointF(tl.x() + delta.x(), tl.y() + delta.y())
            self._clamp_tl()
            self.update()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._pan_last is not None:
            self._pan_last = None
            self.unsetCursor()
        else:
            super().mouseReleaseEvent(ev)


class ImageViewerDialog(QDialog):
    """Magnified view of one gallery image, at native size clamped to ~90% of
    the screen. Non-modal; open with show().
    """

    def __init__(self, path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(getattr(path, "name", None) or str(path))
        self.setAttribute(Qt.WA_DeleteOnClose)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        pm = QPixmap(str(path))
        if pm.isNull():
            err = QLabel(str(path))
            err.setAlignment(Qt.AlignCenter)
            err.setMargin(20)
            lay.addWidget(err)
            return
        img = ScaledImageLabel()
        img.set_source(pm)
        lay.addWidget(img)
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        scale = min(
            1.0,
            avail.width() * 0.9 / pm.width(),
            avail.height() * 0.9 / pm.height(),
        )
        self.resize(
            max(1, round(pm.width() * scale)), max(1, round(pm.height() * scale))
        )
