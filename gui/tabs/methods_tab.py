"""MethodsTab — one Method dropdown over a mix of flat-config + distill methods.

Trainable methods come in two editor flavours: the LoRA family edits a *flat*
``gui-methods/<variant>.toml`` and submits a ``train.py --method`` training job,
while the distill methods (Turbo) edit a *sectioned*
``configs/methods/turbo.toml`` and submit a bespoke ``tasks.py`` distill
command job (no ``train.py`` path).

This wrapper hides that split behind a single Method picker. A ``QStackedWidget``
holds the two editor kinds; selecting a flat method drives the embedded
ConfigTab (its own method picker is hidden so the two don't compete), while
selecting a distill method swaps in its editor.

The same wrapper backs two tabs (see ``gui/app.py``): the **main Config tab**
(``lora`` / ``tlora`` / ``hydralora`` + the promoted ``turbo`` distiller) and the
**Experimental tab** (``soft_tokens`` + the ``soup`` pipeline).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.tabs.config_tab import ConfigTab
from gui.tabs.distill_tab import TurboTrainTab
from gui.tabs.soup_tab import SoupTrainTab


class MethodsTab(QWidget):
    """Unified method picker (flat-config + distill editors)."""

    # Distill methods → their bespoke sectioned-config editor class. `soup` is
    # not a distiller but shares the same shape (a bespoke pipeline submitting a
    # daemon command job, no train.py path), so it rides the same picker slot.
    _DISTILL_EDITORS = {"turbo": TurboTrainTab, "soup": SoupTrainTab}

    def __init__(
        self,
        tb_panel=None,
        *,
        flat_methods=("soft_tokens",),
        distill_methods=(),
        preprocess_tab=None,
    ):
        super().__init__()
        self._flat_methods = tuple(flat_methods)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._config = ConfigTab(
            methods=list(self._flat_methods),
            tb_panel=tb_panel,
            preprocess_tab=preprocess_tab,
        )
        # The wrapper owns method selection, so hide ConfigTab's own method picker
        # (its inline variant picker stays).
        self._config._method_label.setVisible(False)
        self._config.method_combo.setVisible(False)
        # Distill tabs are LazyTabMixin — their TOML scan is deferred until picked.
        self._distill: dict[str, QWidget] = {
            name: self._DISTILL_EDITORS[name]() for name in distill_methods
        }

        # The Method picker rides inside whichever editor page is active, mounted at
        # the front of that page's _top_bar; insertWidget reparents it on page switch.
        self._picker = QWidget()
        picker_row = QHBoxLayout(self._picker)
        picker_row.setContentsMargins(0, 0, 0, 0)
        # "Method" matches ConfigTab's own (hardcoded) picker label.
        picker_row.addWidget(QLabel("Method"))
        self._combo = QComboBox()
        self._combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._combo.setMinimumContentsLength(
            max(len(m) for m in (*self._flat_methods, *self._distill))
        )
        self._combo.addItems([*self._flat_methods, *self._distill])
        self._combo.currentTextChanged.connect(self._on_method)
        picker_row.addWidget(self._combo)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._config)  # one page serves all flat methods
        for w in self._distill.values():
            self._stack.addWidget(w)
        lay.addWidget(self._stack)

        self._on_method(self._combo.currentText())

    def _on_method(self, method: str) -> None:
        editor = self._distill.get(method)
        page = editor if editor is not None else self._config
        self._stack.setCurrentWidget(page)
        page._top_bar.insertWidget(0, self._picker)
        if editor is not None:
            return
        # Drive the embedded ConfigTab's hidden combo (still emits → _reload).
        if self._config.method_combo.currentText() != method:
            self._config.method_combo.setCurrentText(method)

    def cleanup_subprocess(self) -> None:
        """App-shutdown hook — forward to every embedded editor (each leaves its
        detached daemon job alive; this only stops the observers)."""
        for w in (self._config, *self._distill.values()):
            cleanup = getattr(w, "cleanup_subprocess", None)
            if callable(cleanup):
                cleanup()
