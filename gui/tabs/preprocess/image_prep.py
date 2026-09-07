"""Image-prep section: the trainer's dataset roots / scope / pattern / low-res
switch on top of the ``anime_tools`` **resize** stage form (min_pixels,
tiers, crop anchor + margins, free-fit clamp, overwrite, workers).

The three domain widgets (tier row, 3×3 anchor picker, four margin spins)
are kept and mapped onto the stage's dests — the schema's ``list`` kind is
too weak to draw them."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from gui.i18n import t
from gui.tabs.preprocess._section import checkbox, line, no_wheel, spin
from gui.tabs.preprocess.knobs import (
    DEFAULT_PREPROCESS_PATH_PATTERN,
    DEFAULT_SOURCE_IMAGE_DIR,
)
from gui.tabs.preprocess.stage_form import StageFormSection
from gui.theme import tok
from gui.widgets import _TargetResWidget
from library.preprocess.resize_preview import (
    DEFAULT_RESIZE_CROP_ANCHOR,
    normalize_crop_margins,
)

_MARGIN_SIDES = ("top", "right", "bottom", "left")


class _ResizeCropAnchorWidget(QWidget):
    """3×3 anchor picker (which side survives when a resize has to crop)."""

    changed = Signal()

    _LAYOUT = (
        ("top_left", "↖", 0, 0),
        ("top", "↑", 0, 1),
        ("top_right", "↗", 0, 2),
        ("left", "←", 1, 0),
        ("center", "●", 1, 1),
        ("right", "→", 1, 2),
        ("bottom_left", "↙", 2, 0),
        ("bottom", "↓", 2, 1),
        ("bottom_right", "↘", 2, 2),
    )

    def __init__(self) -> None:
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)
        self._buttons: dict[str, QPushButton] = {}
        for key, text, row, col in self._LAYOUT:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedSize(32, 28)
            btn.setStyleSheet(
                "QPushButton { padding:0; } "
                f"QPushButton:checked {{ background:{tok('accent')}; color:#ffffff; "
                "font-weight:bold; }"
            )
            btn.setToolTip(t(f"resize_crop_anchor_{key}"))
            btn.clicked.connect(lambda _checked, value=key: self.set_value(value))
            grid.addWidget(btn, row, col)
            self._buttons[key] = btn
        self.set_value(DEFAULT_RESIZE_CROP_ANCHOR, emit=False)
        self.setFixedSize(32 * 3 + 2 * 2, 28 * 3 + 2 * 2)

    def value(self) -> str:
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return DEFAULT_RESIZE_CROP_ANCHOR

    def set_value(self, value, *, emit: bool = True) -> None:
        anchor = str(value or DEFAULT_RESIZE_CROP_ANCHOR)
        if anchor not in self._buttons:
            anchor = DEFAULT_RESIZE_CROP_ANCHOR
        for key, btn in self._buttons.items():
            btn.blockSignals(True)
            btn.setChecked(key == anchor)
            btn.blockSignals(False)
        if emit:
            self.changed.emit()


class _CropMarginsWidget(QWidget):
    """Four percent spins (top/right/bottom/left) — the crop-exclusion margins.

    ``value()`` is the stage's spelling: a ``[top, right, bottom, left]`` list,
    or ``None`` when every side is zero (the request default, so nothing is
    spelled on the argv)."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.spins: dict[str, QDoubleSpinBox] = {}
        for side in _MARGIN_SIDES:
            lbl = QLabel(t(f"resize_crop_margin_{side}"))
            lbl.setMinimumWidth(24)
            lbl.setStyleSheet(f"QLabel {{ color:{tok('text')}; }}")
            row.addWidget(lbl)
            s = self._margin_spin()
            row.addWidget(s)
            self.spins[side] = s
        row.addStretch(1)

    def _margin_spin(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.0, 95.0)
        s.setDecimals(1)
        s.setSingleStep(1.0)
        s.setSuffix("%")
        s.setStyleSheet(
            "QDoubleSpinBox { padding-right: 4px; }"
            "QDoubleSpinBox::up-button { width: 16px; }"
            "QDoubleSpinBox::down-button { width: 16px; margin-right: 16px; }"
        )
        # After the stylesheet: setAlignment on the spinbox doesn't survive the styled rebuild.
        s.lineEdit().setTextMargins(0, 0, 0, 0)
        s.setFixedWidth(84)
        no_wheel(s)
        s.valueChanged.connect(lambda _v: self.changed.emit())
        return s

    def margins(self) -> dict[str, float]:
        return {side: float(s.value()) for side, s in self.spins.items()}

    def value(self) -> list[float] | None:
        m = self.margins()
        values = [m[side] for side in _MARGIN_SIDES]
        return values if any(v > 0 for v in values) else None

    def set_value(self, value) -> None:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            value = dict(zip(_MARGIN_SIDES, value))
        elif isinstance(value, str) and value.strip():
            value = dict(zip(_MARGIN_SIDES, value.split(",")))
        margins = normalize_crop_margins(value if isinstance(value, dict) else None)
        for side, s in self.spins.items():
            s.blockSignals(True)
            s.setValue(float(margins[side]))
            s.blockSignals(False)


class ImagePrepSection(StageFormSection):
    """The ``resize`` stage form under the trainer's dataset rows."""

    def __init__(self, schema: dict, help_cb, *, pp_cfg: dict, defaults: dict):
        self._pp = pp_cfg
        super().__init__(
            schema,
            help_cb,
            title=t("preprocess_image_prep"),
            defaults=defaults,
            gated_by={"min_pixels": "drop_lowres_images"},
        )

    def _build_prefix(self) -> None:
        pp = self._pp
        self.add_trainer_knob(
            "source_image_dir",
            line(
                str(pp.get("source_image_dir", DEFAULT_SOURCE_IMAGE_DIR)),
                placeholder=DEFAULT_SOURCE_IMAGE_DIR,
            ),
            t("preprocess_source_image_dir"),
            tooltip=t("preprocess_source_image_dir_tip"),
        )
        self.add_trainer_knob(
            "path_scope", line(placeholder="data_group1"), t("path_scope")
        )
        self.add_trainer_knob(
            "preprocess_path_pattern",
            line(
                str(pp.get("preprocess_path_pattern", DEFAULT_PREPROCESS_PATH_PATTERN)),
                placeholder="*",
            ),
            t("preprocess_path_pattern"),
            tooltip=t("preprocess_path_pattern_tip"),
        )
        drop = checkbox(t("preprocess_drop_lowres"))
        drop.setChecked(bool(pp.get("drop_lowres_images", True)))
        # min_pixels (a stage field) only applies when the filter is on — the
        # CLI's drop_lowres=false → --min_pixels 0; wired via ``gated_by``.
        self.add_trainer_knob(
            "drop_lowres_images",
            drop,
            t("preprocess_drop_lowres"),
            tooltip=t("preprocess_drop_lowres_tip"),
        )

    def _make_widget(self, fd: dict) -> QWidget:
        dest = fd["dest"]
        if dest == "min_pixels":
            min_px = spin(0, 100_000_000, int(fd.get("default") or 0), step=50_000)
            min_px.setGroupSeparatorShown(True)
            return min_px
        if dest == "target_res":
            # Dual-use: preprocess resizes to these tiers, and the tab's status
            # / the Dataset tab's resize preview read the same widget.
            self.target_res = _TargetResWidget(fd.get("default") or [1024])
            return self.target_res
        if dest == "resize_crop_anchor":
            return _ResizeCropAnchorWidget()
        if dest == "resize_crop_margins":
            return _CropMarginsWidget()
        if dest == "freefit_max_ratio":
            w = super()._make_widget(fd)
            w.setRange(1.0, 4.0)
            w.setDecimals(2)
            w.setSingleStep(0.25)
            return w
        return super()._make_widget(fd)

    def set_values(self, values: dict) -> None:
        # The tier row and the domain widgets take their own shapes, not the
        # csv text the base would hand a ``list`` field.
        rest = dict(values)
        if "target_res" in rest:
            self.set_target_res(rest.pop("target_res"))
        if "resize_crop_margins" in rest:
            self.widgets["resize_crop_margins"].set_value(
                rest.pop("resize_crop_margins")
            )
        if "resize_crop_anchor" in rest:
            self.widgets["resize_crop_anchor"].set_value(
                rest.pop("resize_crop_anchor"), emit=False
            )
        super().set_values(rest)

    def set_target_res(self, values) -> None:
        """Set the tier checkboxes without emitting per-box change signals."""
        if values is None:
            selected = {1024}
        elif isinstance(values, (list, tuple, set)):
            selected = {int(v) for v in values}
        else:
            selected = {int(values)}
        for edge, box in self.target_res._boxes.items():
            box.blockSignals(True)
            box.setChecked(edge in selected)
            box.blockSignals(False)
