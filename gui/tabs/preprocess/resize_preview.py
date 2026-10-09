"""Resize preview: which bucket and tier each source image lands in, and the
crop that gets it there, under the Preprocess tab's live resize settings.

Reads the tab's widgets (tiers, crop anchor + margins, free-fit clamp, source
dir, preprocess pattern) and never touches a file. Images either exclusion
ledger skips are left out, as the resize stage leaves them out.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from gui import ROOT, ScaledImageLabel, _imgs
from gui.i18n import t
from gui.tabs.preprocess.knobs import (
    DEFAULT_PREPROCESS_PATH_PATTERN,
    DEFAULT_SOURCE_IMAGE_DIR,
)
from library.datasets.curation_actions import (
    excluded_rels,
    rel_key,
    workspace_excluded_rels,
)
from library.datasets.path_filter import filter_paths_by_glob
from library.preprocess.resize_preview import (
    DEFAULT_FREEFIT_MAX_RATIO,
    ResizePreview,
    compute_resize_preview,
)

_KEPT_COLOR = QColor(40, 220, 120, 255)
_MARGIN_COLOR = QColor(255, 70, 60, 255)
_SHADE = QColor(0, 0, 0, 72)

# Columns: image, source size, bucket, tier, share of the source kept.
_COLS = 5
_PATH_ROLE = Qt.UserRole
_SORT_ROLE = Qt.UserRole + 1


class _SortItem(QTreeWidgetItem):
    """Sorts each column by the number stored under ``_SORT_ROLE`` when there
    is one (pixel counts, tier, kept share), else by text."""

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        a, b = self.data(col, _SORT_ROLE), other.data(col, _SORT_ROLE)
        if a is not None and b is not None:
            return a < b
        return self.text(col).lower() < other.text(col).lower()


def compose_overlay(source: QPixmap, preview: ResizePreview) -> QPixmap:
    """``source`` with everything the resize drops shaded, the kept rect in
    green and (when set) the crop-margin rect in red."""
    w, h = source.width(), source.height()

    def clamp(rect) -> tuple[int, int, int, int]:
        left = max(0, min(w, round(rect.left)))
        top = max(0, min(h, round(rect.top)))
        right = max(left, min(w, round(rect.left + rect.width)))
        bottom = max(top, min(h, round(rect.top + rect.height)))
        return left, top, right, bottom

    left, top, right, bottom = clamp(preview.kept_rect)
    result = QPixmap(source)
    painter = QPainter(result)
    try:
        painter.setPen(Qt.NoPen)
        painter.fillRect(0, 0, w, top, _SHADE)
        painter.fillRect(0, bottom, w, h - bottom, _SHADE)
        painter.fillRect(0, top, left, bottom - top, _SHADE)
        painter.fillRect(right, top, w - right, bottom - top, _SHADE)
        pen_width = max(2, round(min(w, h) * 0.004))
        painter.setBrush(Qt.NoBrush)
        if any(v > 0 for v in preview.crop_margins.values()):
            ml, mt, mr, mb = clamp(preview.margin_rect)
            painter.setPen(QPen(_MARGIN_COLOR, pen_width, Qt.SolidLine))
            painter.drawRect(QRect(ml, mt, mr - ml, mb - mt).adjusted(1, 1, -1, -1))
        painter.setPen(QPen(_KEPT_COLOR, pen_width, Qt.SolidLine))
        painter.drawRect(
            QRect(left, top, right - left, bottom - top).adjusted(1, 1, -1, -1)
        )
    finally:
        painter.end()
    return result


class ResizePreviewDialog(QDialog):
    """Non-modal: stays open beside the tab and recomputes when its image-prep
    section changes."""

    def __init__(self, tab) -> None:
        super().__init__(tab)
        self._tab = tab
        self._sizes: dict[Path, tuple[float, tuple[int, int]]] = {}
        self._source_pm: QPixmap | None = None
        self._source_path: Path | None = None
        self.setWindowTitle(t("preprocess_resize_preview_title"))
        self.resize(1100, 700)

        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        head.addWidget(self.summary, 1)
        refresh = QPushButton(t("preprocess_resize_preview_refresh"))
        refresh.clicked.connect(self.refresh)
        head.addWidget(refresh)
        lay.addLayout(head)

        split = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(_COLS)
        self.tree.setHeaderLabels(
            [
                t("preprocess_resize_preview_col_image"),
                t("preprocess_resize_preview_col_source"),
                t("preprocess_resize_preview_col_bucket"),
                t("preprocess_resize_preview_col_tier"),
                t("preprocess_resize_preview_col_kept"),
            ]
        )
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        self.tree.currentItemChanged.connect(self._on_current_changed)
        split.addWidget(self.tree)
        self.view = ScaledImageLabel()
        self.view.setMinimumWidth(360)
        split.addWidget(self.view)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)

        # A spin-box drag emits per step; recompute once it settles.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self.refresh)
        tab.image_section.changed.connect(self._debounce.start)

    # -- the tab's live settings --------------------------------------------

    def _config(self) -> dict:
        tab = self._tab
        try:
            target_res = tab.widget("target_res").value()
        except (KeyError, TypeError, ValueError):
            target_res = None
        try:
            spin = tab.widget("freefit_max_ratio")
        except KeyError:
            spin = None
        return {
            "target_res": target_res,
            "crop_anchor": tab.widget("resize_crop_anchor").value(),
            "crop_margins": tab.widget("resize_crop_margins").margins(),
            "max_ratio": float(spin.value()) if spin else DEFAULT_FREEFIT_MAX_RATIO,
        }

    def _source_dir(self) -> Path:
        raw = (
            str(self._tab.values().get("source_image_dir") or "").strip()
            or DEFAULT_SOURCE_IMAGE_DIR
        )
        p = Path(raw).expanduser()
        return p if p.is_absolute() else ROOT / p

    def _sources(self, src: Path) -> tuple[list[Path], int]:
        """Images the resize would visit, and how many the ledgers skip."""
        paths = _imgs(src)
        pattern = (
            str(self._tab.values().get("preprocess_path_pattern") or "").strip()
            or DEFAULT_PREPROCESS_PATH_PATTERN
        )
        if pattern != DEFAULT_PREPROCESS_PATH_PATTERN:
            keep = filter_paths_by_glob([str(p) for p in paths], str(src), pattern)
            paths = [p for p, k in zip(paths, keep) if k]
        skipped = set(excluded_rels()) | set(workspace_excluded_rels())
        kept = [p for p in paths if rel_key(p, src) not in skipped]
        return kept, len(paths) - len(kept)

    def _size(self, path: Path) -> tuple[int, int]:
        """Header-only read, cached by mtime so a refresh re-reads only what
        changed."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return (0, 0)
        cached = self._sizes.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        size = QImageReader(str(path)).size()
        wh = (max(0, size.width()), max(0, size.height()))
        self._sizes[path] = (mtime, wh)
        return wh

    # -- build --------------------------------------------------------------

    def refresh(self) -> None:
        src = self._source_dir()
        current = self._source_path
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            paths, n_skipped = self._sources(src)
            cfg = self._config()
            tiers: Counter[int] = Counter()
            buckets: set[tuple[int, int]] = set()
            items: list[QTreeWidgetItem] = []
            unreadable = 0
            for p in paths:
                w, h = self._size(p)
                try:
                    pv = compute_resize_preview(w, h, **cfg)
                except (KeyError, TypeError, ValueError):
                    unreadable += 1
                    continue
                tiers[pv.target_edge] += 1
                buckets.add(pv.bucket_size)
                items.append(self._item(p, src, pv))
        finally:
            QApplication.restoreOverrideCursor()

        self.tree.setSortingEnabled(False)
        self.tree.clear()
        self.tree.addTopLevelItems(items)
        self.tree.setSortingEnabled(True)
        for col in range(1, _COLS):
            self.tree.resizeColumnToContents(col)

        parts = [
            t("preprocess_resize_preview_summary", n=len(items), buckets=len(buckets))
        ]
        parts += [f"{edge}: {tiers[edge]}" for edge in sorted(tiers)]
        if n_skipped:
            parts.append(t("preprocess_resize_preview_skipped", n=n_skipped))
        if unreadable:
            parts.append(t("preprocess_resize_preview_unreadable", n=unreadable))
        if not items and not n_skipped:
            parts = [t("preprocess_resize_preview_empty", path=str(src))]
        self.summary.setText("  ·  ".join(parts))

        self._select(current)

    def _item(self, path: Path, src: Path, pv: ResizePreview) -> QTreeWidgetItem:
        sw, sh = pv.source_size
        bw, bh = pv.bucket_size
        kept = pv.kept_rect.width * pv.kept_rect.height / max(1, sw * sh)
        item = _SortItem(
            [
                rel_key(path, src),
                f"{sw}×{sh}",
                f"{bw}×{bh}",
                str(pv.target_edge),
                f"{kept:.0%}",
            ]
        )
        item.setData(0, _PATH_ROLE, str(path))
        item.setData(1, _SORT_ROLE, sw * sh)
        item.setData(2, _SORT_ROLE, bw * bh)
        item.setData(3, _SORT_ROLE, pv.target_edge)
        item.setData(4, _SORT_ROLE, kept)
        for col in range(1, _COLS):
            item.setTextAlignment(col, Qt.AlignRight | Qt.AlignVCenter)
        return item

    def _select(self, path: Path | None) -> None:
        if path is not None:
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                if item.data(0, _PATH_ROLE) == str(path):
                    self.tree.setCurrentItem(item)
                    self._show(path)
                    return
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        else:
            self._source_pm = self._source_path = None
            self.view.clear()

    # -- the image pane -----------------------------------------------------

    def _on_current_changed(self, current, _previous) -> None:
        if current is not None:
            self._show(Path(current.data(0, _PATH_ROLE)))

    def _show(self, path: Path) -> None:
        if path != self._source_path or self._source_pm is None:
            pm = QPixmap(str(path))
            self._source_pm = None if pm.isNull() else pm
            self._source_path = path
        if self._source_pm is None:
            self.view.clear()
            return
        try:
            pv = compute_resize_preview(
                self._source_pm.width(), self._source_pm.height(), **self._config()
            )
        except (KeyError, TypeError, ValueError):
            self.view.set_source(self._source_pm)
            return
        self.view.set_source(compose_overlay(self._source_pm, pv))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()
