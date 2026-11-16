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

    The colors are ``[variant="…"]`` rules in the global stylesheet
    (:data:`gui.theme.ACTION_COLORS`). A dynamic property set after the widget
    is realized does not restyle it until re-polished. ``None``/"" clears to the
    neutral button.
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

    Variants: ``primary`` (green go), ``secondary`` (purple alt), ``info``
    (blue), ``danger`` (red stop), ``warning`` (orange/dirty), ``busy`` (gray);
    flip one later with :func:`apply_variant`. ``tool_button=True`` builds a
    :class:`QToolButton` instead of a :class:`QPushButton`.
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

    A stylesheet rule on ``QToolButton::menu-button`` drops Qt's layout
    reservation for the arrow and centres the label on the whole button; the
    style metric keeps it centred in the action segment. The divider and tint
    are painted for the same reason.

    Apply with ``button.setStyle(style)`` before the stylesheet, and keep a
    reference alive (the widget does not take ownership).
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
