"""``KnobSection`` — one ``QGroupBox`` form over a slice of the knob table.

A section subclass declares *which* widgets it builds (``_build``); the base
handles the clickable help label, the change→dirty wiring, the ``enabled_by`` gating,
and the generic ``values()`` / ``set_values()`` read/write keyed by knob.

Read/write is dispatched on the *widget type*, not the knob kind, because two
``float`` knobs may legitimately differ in editor (a ``QDoubleSpinBox`` for
``freefit_max_ratio``, a free-text ``QLineEdit`` for the dropout rate whose raw
text must reach the env export untouched — see ``knobs.to_env``).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from gui.explanations import preprocess_field_help
from gui.tabs.preprocess.knobs import KNOBS_BY_KEY, Knob
from gui.theme import tok
from gui.widgets import ClickableLabel, make_field_label


def connect_change(widget: QWidget, slot) -> None:
    """Wire a widget's value-changed signal to ``slot`` (domain widgets expose
    a ``changed`` signal; the Qt primitives each have their own name)."""
    sig = getattr(widget, "changed", None)
    if sig is None:
        if isinstance(widget, QComboBox):
            sig = widget.currentIndexChanged
        elif isinstance(widget, QCheckBox):
            sig = widget.toggled
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            sig = widget.valueChanged
        elif isinstance(widget, QLineEdit):
            sig = widget.textChanged
        else:
            return
    # The Qt primitives pass their new value; the slot is arity-free.
    sig.connect(lambda *_: slot())


def no_wheel(widget: QWidget) -> QWidget:
    """Block scroll-wheel edits (a scrolled form must not change values)."""
    widget.wheelEvent = lambda e: e.ignore()
    return widget


def read_widget(knob: Knob, widget: QWidget):
    """Raw widget state in the shape ``knobs`` expects (strings stripped,
    free-text numerics left as text, combo → item *data*)."""
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox):
        return int(widget.value())
    if isinstance(widget, QDoubleSpinBox):
        return float(widget.value())
    if isinstance(widget, QComboBox):
        return str(widget.currentData() or knob.default)
    if isinstance(widget, QLineEdit):
        return widget.text().strip()
    return widget.value()  # domain widgets: value()/set_value()


def set_widget(knob: Knob, widget: QWidget, value) -> None:
    """Inverse of :func:`read_widget`; empty text falls back to the hardcoded
    default (``"*"`` patterns, the source dir)."""
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(value))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
    elif isinstance(widget, QComboBox):
        index = widget.findData(str(value or knob.default))
        if index < 0:
            index = widget.findData(knob.default)
        widget.setCurrentIndex(max(index, 0))
    elif isinstance(widget, QLineEdit):
        if knob.kind == "float":
            widget.setText(f"{float(value):g}")
        elif value is None or value == "":
            widget.setText(str(knob.default))
        else:
            widget.setText(str(value))
    else:
        widget.set_value(value)


class KnobSection(QGroupBox):
    """A titled form of knob widgets. Subclasses implement ``_build`` using
    :meth:`add_knob`; the tab composes sections and reads them back with
    :meth:`values`."""

    changed = Signal()

    def __init__(self, title: str, help_cb):
        super().__init__(title)
        self._help_cb = help_cb
        self.widgets: dict[str, QWidget] = {}
        self.form = QFormLayout()
        self.setLayout(self.form)
        self._build()
        self._wire_enabled_by()

    # -- building -----------------------------------------------------------

    def _build(self) -> None:  # pragma: no cover — abstract
        raise NotImplementedError

    def field_label(self, key: str, text: str) -> ClickableLabel:
        """Dotted-underline label that routes the field's help into the tab's
        explanation panel (tooltip/help keyed by the knob key)."""
        help_text = preprocess_field_help(key)
        return make_field_label(
            text,
            style=f"color:{tok('text')}; text-decoration: underline dotted;",
            on_click=lambda _t=text, _h=help_text: self._help_cb(_t, _h),
        )

    def add_knob(
        self,
        key: str,
        widget: QWidget,
        label: str,
        *,
        tooltip: str | None = None,
        field=None,
    ) -> QWidget:
        """Register ``widget`` as the editor for knob ``key`` and add its form
        row. ``field`` (a layout or wrapper widget) replaces ``widget`` in the
        row when the editor is embedded in a composite; ``tooltip`` defaults
        to the per-field help text."""
        if key not in KNOBS_BY_KEY:
            raise KeyError(f"{key}: not in the preprocess knob table")
        widget.setToolTip(
            tooltip if tooltip is not None else (preprocess_field_help(key) or "")
        )
        self.form.addRow(
            self.field_label(key, label), field if field is not None else widget
        )
        self.widgets[key] = widget
        connect_change(widget, self.changed.emit)
        return widget

    def _wire_enabled_by(self) -> None:
        for key, widget in self.widgets.items():
            gate = KNOBS_BY_KEY[key].enabled_by
            if gate and gate in self.widgets:
                self.widgets[gate].toggled.connect(widget.setEnabled)
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        for key, widget in self.widgets.items():
            gate = KNOBS_BY_KEY[key].enabled_by
            if gate and gate in self.widgets:
                widget.setEnabled(self.widgets[gate].isChecked())

    # -- values -------------------------------------------------------------

    def keys(self) -> tuple[str, ...]:
        return tuple(self.widgets)

    def values(self) -> dict[str, object]:
        return {
            key: read_widget(KNOBS_BY_KEY[key], w) for key, w in self.widgets.items()
        }

    def set_values(self, values: dict) -> None:
        """Push ``values`` (any superset of this section's keys) into the
        widgets. Callers hold the tab's ``_loading_variant`` flag so the
        resulting ``changed`` emissions don't mark the form dirty."""
        for key, widget in self.widgets.items():
            if key in values:
                set_widget(KNOBS_BY_KEY[key], widget, values[key])
        self._sync_enabled()


def checkbox(text: str) -> QCheckBox:
    return QCheckBox(text)


def spin(lo: int, hi: int, value: int, *, step: int | None = None) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    if step is not None:
        w.setSingleStep(step)
    w.setValue(value)
    return no_wheel(w)


def dspin(
    lo: float, hi: float, value: float, *, step: float, decimals: int
) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(value)
    return no_wheel(w)


def line(text: str = "", *, placeholder: str | None = None) -> QLineEdit:
    w = QLineEdit(text)
    if placeholder is not None:
        w.setPlaceholderText(placeholder)
    return w


__all__ = [
    "KnobSection",
    "checkbox",
    "connect_change",
    "dspin",
    "line",
    "no_wheel",
    "read_widget",
    "set_widget",
    "spin",
]
