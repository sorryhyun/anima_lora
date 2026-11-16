"""Tiny Qt-only helpers shared by sibling widget modules.

A leaf below ``fields.py`` and ``sample_prompts.py``, which both use it and
may not import each other.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


def _no_wheel(w: QWidget) -> QWidget:
    """Ignore the mouse wheel on a combo/spin, so scrolling the form does not
    silently edit whichever one the cursor passes over."""
    w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    w.wheelEvent = lambda e: e.ignore()
    return w
