"""Action-button variants + the split-button proxy style."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QProxyStyle,
    QPushButton,
    QStyle,
    QStyleOptionToolButton,
    QToolButton,
    QWidget,
)


def apply_variant(w: QWidget, variant: str | None) -> None:
    """(Re)assign an action-button style ``variant`` and repolish so it repaints.

    The saturated action colors live in the global stylesheet as
    ``[variant="…"]`` rules (see :data:`gui.theme.ACTION_COLORS`). Setting the
    dynamic property after a widget is realized does **not** restyle it until the
    style is re-polished — hence the unpolish/polish dance. This is what a state
    flip uses (idle→busy swaps the variant in one line instead of swapping a
    whole stylesheet string). Pass ``None``/"" to clear back to the neutral
    default button look.
    """
    w.setProperty("variant", variant or "")
    style = w.style()
    style.unpolish(w)
    style.polish(w)


def action_button(
    text: str = "",
    *,
    variant: str = "primary",
    tooltip: str | None = None,
    on_click=None,
    tool_button: bool = False,
) -> QPushButton | QToolButton:
    """Build a saturated action button styled by its global-stylesheet *variant*.

    Creates the button, sets its variant / tooltip and connects ``on_click``.
    The color lives in :data:`gui.theme.ACTION_COLORS`; flip it on a state
    change via :func:`apply_variant`.

    Variants: ``primary`` (green go), ``secondary`` (purple alt), ``info``
    (blue), ``danger`` (red stop), ``warning`` (orange/dirty), ``busy`` (gray).
    Pass ``tool_button=True`` for a :class:`QToolButton` (the action buttons that
    flip busy state are tool buttons); default is a :class:`QPushButton`.
    """
    btn: QPushButton | QToolButton = QToolButton() if tool_button else QPushButton()
    if text:
        btn.setText(text)
    apply_variant(btn, variant)
    if tooltip:
        btn.setToolTip(tooltip)
    if on_click is not None:
        btn.clicked.connect(on_click)
    return btn


class SplitButtonStyle(QProxyStyle):
    """Widen a ``QToolButton``'s dropdown indicator and paint a divider + tint.

    Styling ``QToolButton::menu-button`` via a stylesheet disables Qt's
    reservation of the arrow region for layout, which snaps the label to the
    *full*-button centre. Driving the indicator width from the style metric
    instead keeps Qt's reservation intact, so the label stays centred in the
    action (non-arrow) segment — what we actually want for a split button —
    while still giving a wide, visually distinct dropdown half. The divider +
    subtle dark tint are painted over the menu sub-control (a stylesheet rule
    there would re-break the centring).

    Apply with ``button.setStyle(style)`` and keep a reference alive (the widget
    does not take ownership). Set the style BEFORE the stylesheet.
    """

    INDICATOR = 22

    def pixelMetric(self, metric, option=None, widget=None):
        if metric == QStyle.PM_MenuButtonIndicator:
            return self.INDICATOR
        return super().pixelMetric(metric, option, widget)

    def drawComplexControl(self, control, option, painter, widget=None):
        super().drawComplexControl(control, option, painter, widget)
        if (
            control == QStyle.CC_ToolButton
            and isinstance(option, QStyleOptionToolButton)
            and option.features & QStyleOptionToolButton.HasMenu
        ):
            rect = self.subControlRect(
                QStyle.CC_ToolButton, option, QStyle.SC_ToolButtonMenu, widget
            )
            painter.save()
            painter.fillRect(rect, QColor(0, 0, 0, 46))
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawLine(
                rect.left(), rect.top() + 3, rect.left(), rect.bottom() - 3
            )
            painter.restore()
