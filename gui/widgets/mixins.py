"""Shared tab mixins: lazy first-show init + Save-button dirty-state tracking."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import t
from gui.widgets.buttons import apply_variant
from gui.widgets.sample_prompts import _SamplePromptsLauncher, _SamplePromptsWidget
from gui.widgets.target_res import _TargetResWidget


class LazyTabMixin:
    """Defer a tab's first expensive scan until the tab is actually opened.

    Several tabs walk dataset/checkpoint directories (and the Merge tab reads
    safetensors keys). Mixing this in keeps construction cheap: the heavy
    work runs on the first ``showEvent`` — i.e. when the user selects the tab —
    and exactly once thereafter. Subclasses override ``_lazy_init``.

    Mix in BEFORE ``QWidget`` so ``super().showEvent`` resolves to Qt's.
    """

    _lazy_done = False

    def showEvent(self, event):  # noqa: N802 — Qt event handler name
        super().showEvent(event)
        if not self._lazy_done:
            self._lazy_done = True
            self._lazy_init()

    def _lazy_init(self) -> None:
        """Run the tab's first directory scan / classification. Override."""


class LazyTabHolder(QWidget):
    """Defer a tab's entire *construction* until it is first opened.

    :class:`LazyTabMixin` only defers a tab's first directory scan — the
    widget tree itself is still built up front, and with the app-wide QSS
    every widget creation pays a style-resolution tax. For tabs that are not
    visible at launch even that is wasted time, so this holder sits in the
    ``QTabWidget`` as an empty page and builds the real tab (via ``factory``)
    on the first ``showEvent``. The inner tab's own ``LazyTabMixin`` scan then
    fires naturally as it becomes visible.

    ``inner`` is ``None`` until built — callers reaching into the tab from
    outside (e.g. the Models dialog refreshing the Images tab's tag KB) must
    treat a not-yet-built tab as "nothing to update": it will construct from
    fresh on-disk state when opened. ``cleanup_subprocess`` forwards to the
    inner tab so ``MainWindow.closeEvent``'s teardown loop keeps working.
    """

    def __init__(self, factory):
        super().__init__()
        self._factory = factory
        self.inner: QWidget | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

    def showEvent(self, event):  # noqa: N802 — Qt event handler name
        super().showEvent(event)
        if self.inner is None:
            self.inner = self._factory()
            self.layout().addWidget(self.inner)

    def cleanup_subprocess(self) -> None:
        if self.inner is not None:
            cleanup = getattr(self.inner, "cleanup_subprocess", None)
            if callable(cleanup):
                cleanup()


class DirtyTrackingMixin:
    """Save-button dirty-state tracking shared by the config-style tabs.

    A tab is *dirty* when its form has edits not yet written back to the config
    file on disk. Used by ConfigTab (+ its EasyControl subclass), the distill
    editors, and PreprocessingTab.

    Host requirements:
      * ``self._dirty: bool`` initialised in ``__init__``;
      * a Save button exposed as ``self._save_btn`` (override
        :meth:`_dirty_save_button` if it lives under a different name);
      * optionally ``self._save_btn_dirty_variant`` — the action-button variant
        flipped on while dirty (defaults to ``"warning"``, the orange look);
      * optionally ``self._loading_variant`` — when truthy ``_mark_dirty`` is a
        no-op so a bulk reload doesn't trip the flag.

    Override :meth:`_update_save_button` when the button's text/tooltip differ
    from the default (PreprocessingTab does — its label is the localized
    "Save settings").

    Mix in BEFORE ``QWidget`` (and any other event-handling mixin).
    """

    _dirty: bool = False

    def _dirty_save_button(self) -> QPushButton | None:
        return getattr(self, "_save_btn", None)

    def _connect_dirty_signal(self, w: QWidget) -> None:
        """Wire a form widget's change signal to :meth:`_mark_dirty`.

        Connect AFTER the widget's value has been seeded, so the initial
        setValue/addItems calls don't trip the flag. The branch set is a
        superset of what any single tab needs — widget types a given tab never
        builds simply never match.
        """
        if isinstance(
            w, (_TargetResWidget, _SamplePromptsWidget, _SamplePromptsLauncher)
        ):
            w.changed.connect(self._mark_dirty)
        elif isinstance(w, QComboBox):
            w.currentTextChanged.connect(self._mark_dirty)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(self._mark_dirty)
        elif isinstance(w, QDoubleSpinBox):
            w.valueChanged.connect(self._mark_dirty)
        elif isinstance(w, QSpinBox):
            w.valueChanged.connect(self._mark_dirty)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(self._mark_dirty)
        elif isinstance(w, QPlainTextEdit):
            w.textChanged.connect(self._mark_dirty)

    def _mark_dirty(self, *_) -> None:
        if getattr(self, "_loading_variant", False) or self._dirty:
            return
        self._dirty = True
        self._update_save_button()

    def _clear_dirty(self) -> None:
        self._dirty = False
        self._update_save_button()

    # The dirty look is the centralized "warning" action variant (orange),
    # flipped on/off via apply_variant — no per-tab style strings.
    _save_btn_dirty_variant: str = "warning"

    def _update_save_button(self) -> None:
        btn = self._dirty_save_button()
        if btn is None:
            return
        if self._dirty:
            btn.setText(t("save") + " *")
            btn.setToolTip(t("save_dirty_tooltip"))
            apply_variant(btn, self._save_btn_dirty_variant)
        else:
            btn.setText(t("save"))
            btn.setToolTip("")
            apply_variant(btn, None)
