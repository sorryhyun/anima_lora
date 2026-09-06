"""Off-thread image decode + mask / resize-preview overlay composers.

Pure leaf helpers split out of ``image_tab.py``: none of them reference the
owning ``ImageViewerTab``. They take a ``QPixmap`` (plus a little preprocess
config) and return a new ``QPixmap`` with the overlay painted on.
"""

from __future__ import annotations

from pathlib import Path

import toml
from PySide6.QtCore import QObject, QRect, QRunnable, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap

from gui import ROOT, default_mask_dir
from library.preprocess.resize_preview import (
    DEFAULT_FIT_MODE,
    DEFAULT_FREEFIT_MAX_RATIO,
    compute_resize_preview,
)


class _DecodeSignals(QObject):
    """GUI-thread sink for off-thread image decodes (QRunnable can't carry
    signals, so the signal lives on its own QObject)."""

    done = Signal(str, object)  # (path_str, QImage | None)


class _DecodeTask(QRunnable):
    """Decode one image to a QImage on a worker thread. QPixmap is GUI-thread
    only, so we hand back a QImage and convert on the main thread."""

    def __init__(self, path_str: str, signals: _DecodeSignals) -> None:
        super().__init__()
        self._path = path_str
        self._signals = signals

    def run(self) -> None:  # noqa: D401 — QRunnable API
        img = QImage(self._path)
        self._signals.done.emit(self._path, None if img.isNull() else img)


# Tint over the *masked-out* (inverted) region; alpha driven from the mask +
# QPainter.setOpacity rather than baked into the color.
_MASK_OVERLAY_COLOR_OPAQUE = QColor(255, 60, 60, 255)
_MASK_OVERLAY_OPACITY = 0.55
_RESIZE_PREVIEW_COLOR = QColor(40, 220, 120, 255)
_RESIZE_MARGIN_COLOR = QColor(255, 70, 60, 255)
_RESIZE_PREVIEW_SHADE = QColor(0, 0, 0, 72)


def _format_file_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(max(0, size))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024


def _resolve_mask_path(image_path: Path, current_dir: Path | None) -> Path | None:
    """Locate the merged mask PNG for ``image_path``.

    Mirrors the trainer's mask layout: ``<mask_dir>/<rel>/<stem>_mask.png``
    where ``mask_dir`` is the configured mask root (configs/preprocess.toml)
    and ``rel`` is the image's parent relative to ``current_dir``. Falls back
    to the legacy ``masks/merged/...`` tree before giving up.
    """
    if current_dir is None:
        return None
    try:
        rel = image_path.relative_to(current_dir)
    except ValueError:
        return None
    rel_parent = rel.parent
    name = f"{image_path.stem}_mask.png"
    for root in (default_mask_dir(), ROOT / "masks" / "merged"):
        candidate = root / rel_parent / name
        if candidate.is_file():
            return candidate
    return None


def _compose_mask_overlay(source: QPixmap, mask_path: Path) -> QPixmap:
    """Return ``source`` with a red translucent tint over the masked-out region.

    Convention from ``anime_tools.masking.cli.merge_masks``: **white = "train here",
    black = ignored (text bubble / artifact)**. We invert so the tint lands
    on the *ignored* region — that's the half users want to see at a glance
    ("did the detector catch every bubble?").

    Implementation note: ``convertToFormat(Alpha8)`` does **not** repurpose a
    grayscale channel as alpha — Qt fills it with the source's actual alpha
    (which is opaque-255 for Grayscale8), giving a uniform tint. Use
    ``setAlphaChannel`` instead: when given a grayscale image, it copies the
    luminance into the alpha channel of an ARGB32 layer.

    Alignment: masks are generated at the **bucket** resolution
    (``post_image_dataset/resized/`` = scale-to-cover + center-crop of the
    original in ``image_dataset/``). A plain ``IgnoreAspectRatio`` rescale
    onto the source would (a) stretch non-uniformly when ARs differ and
    (b) ignore the cropped-out margins — both contribute visible drift on
    the original-image view. Invert the bucket transform: scale the mask
    uniformly to match the appropriate axis, then letterbox the other axis
    so masked features land where the trainer actually saw them.
    """
    mask_img = QImage(str(mask_path))
    if mask_img.isNull():
        return source
    gray = mask_img.convertToFormat(QImage.Format_Grayscale8)
    gray.invertPixels()  # bubble (was 0) → 255, train-here (was 255) → 0

    src_w, src_h = source.width(), source.height()
    mask_w, mask_h = gray.width(), gray.height()
    if (src_w, src_h) == (mask_w, mask_h):
        aligned = gray
    elif src_w * mask_h >= src_h * mask_w:
        # ar_src >= ar_mask: bucket cropped left/right; match height, letterbox width.
        scaled_w = max(1, round(mask_w * src_h / mask_h))
        scaled = gray.scaled(
            scaled_w, src_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        aligned = QImage(src_w, src_h, QImage.Format_Grayscale8)
        aligned.fill(0)  # 0 = no tint on the cropped-out bars
        offset_x = max(0, (src_w - scaled_w) // 2)
        painter = QPainter(aligned)
        try:
            painter.drawImage(offset_x, 0, scaled)
        finally:
            painter.end()
    else:
        # ar_src < ar_mask: bucket cropped top/bottom; match width, letterbox height.
        scaled_h = max(1, round(mask_h * src_w / mask_w))
        scaled = gray.scaled(
            src_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        aligned = QImage(src_w, src_h, QImage.Format_Grayscale8)
        aligned.fill(0)
        offset_y = max(0, (src_h - scaled_h) // 2)
        painter = QPainter(aligned)
        try:
            painter.drawImage(0, offset_y, scaled)
        finally:
            painter.end()

    layer = QImage(source.size(), QImage.Format_ARGB32)
    layer.fill(_MASK_OVERLAY_COLOR_OPAQUE)
    layer.setAlphaChannel(aligned)

    result = QPixmap(source)
    p = QPainter(result)
    try:
        p.setOpacity(_MASK_OVERLAY_OPACITY)
        p.drawImage(0, 0, layer)
    finally:
        p.end()
    return result


def _load_preprocess_toml_data():
    path = ROOT / "configs" / "preprocess.toml"
    if not path.is_file():
        return {}
    try:
        return toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}


def _load_resize_preview_target_res():
    return _load_preprocess_toml_data().get("target_res")


def _compose_resize_preview_overlay(
    source: QPixmap,
    target_res,
    crop_anchor=None,
    bucket_resos=None,
    crop_margins=None,
    fit_mode=DEFAULT_FIT_MODE,
    max_ratio=DEFAULT_FREEFIT_MAX_RATIO,
) -> QPixmap:
    try:
        preview = compute_resize_preview(
            source.width(),
            source.height(),
            target_res,
            crop_anchor=crop_anchor,
            bucket_resos=bucket_resos,
            crop_margins=crop_margins,
            fit_mode=fit_mode,
            max_ratio=max_ratio,
        )
    except (KeyError, TypeError, ValueError):
        return source

    rect = preview.kept_rect
    left = max(0, min(source.width(), round(rect.left)))
    top = max(0, min(source.height(), round(rect.top)))
    right = max(left, min(source.width(), round(rect.left + rect.width)))
    bottom = max(top, min(source.height(), round(rect.top + rect.height)))
    margin = preview.margin_rect
    margin_left = max(0, min(source.width(), round(margin.left)))
    margin_top = max(0, min(source.height(), round(margin.top)))
    margin_right = max(
        margin_left,
        min(source.width(), round(margin.left + margin.width)),
    )
    margin_bottom = max(
        margin_top,
        min(source.height(), round(margin.top + margin.height)),
    )

    result = QPixmap(source)
    painter = QPainter(result)
    try:
        painter.setPen(Qt.NoPen)
        painter.fillRect(0, 0, source.width(), top, _RESIZE_PREVIEW_SHADE)
        painter.fillRect(
            0,
            bottom,
            source.width(),
            source.height() - bottom,
            _RESIZE_PREVIEW_SHADE,
        )
        painter.fillRect(0, top, left, bottom - top, _RESIZE_PREVIEW_SHADE)
        painter.fillRect(
            right,
            top,
            source.width() - right,
            bottom - top,
            _RESIZE_PREVIEW_SHADE,
        )

        pen_width = max(2, round(min(source.width(), source.height()) * 0.004))
        if any(value > 0 for value in preview.crop_margins.values()):
            painter.setPen(QPen(_RESIZE_MARGIN_COLOR, pen_width, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                QRect(
                    margin_left,
                    margin_top,
                    margin_right - margin_left,
                    margin_bottom - margin_top,
                ).adjusted(1, 1, -1, -1)
            )

        painter.setPen(QPen(_RESIZE_PREVIEW_COLOR, pen_width, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(
            QRect(left, top, right - left, bottom - top).adjusted(1, 1, -1, -1)
        )

    finally:
        painter.end()
    return result
