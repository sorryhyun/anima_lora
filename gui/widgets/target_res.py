"""The multi-scale ``target_res`` tier checkbox row."""

from __future__ import annotations

import functools

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from gui.i18n import t


@functools.cache
def _target_res_tiers() -> tuple[tuple[int, ...], dict[int, int]]:
    """``(allowed tiers, {edge: max token count})`` sourced from
    ``library.datasets.buckets`` (the single source of truth) instead of a
    hardcoded mirror. Lazy + cached so importing ``widgets`` stays cheap; the
    bucket module is torch-free so this never drags the training stack in.

    A tier is "dangerous" (extra compiled block graph + VRAM) when its per-image
    token count exceeds the canonical 1024 tier — which reproduces the previous
    ``{1280: 6300, 1536: 8640}`` flag set, but recomputed from the tables.
    """
    from library.datasets.buckets import ALLOWED_TARGET_RES, EDGE_TOKEN_BANDS

    # A tier's max token count is the high end of its free-fit band.
    max_tok = {edge: hi for edge, (lo, hi) in EDGE_TOKEN_BANDS.items()}
    canonical = max_tok.get(1024, 4200)
    danger = {edge: tok for edge, tok in max_tok.items() if tok > canonical}
    return tuple(ALLOWED_TARGET_RES), danger


class _TargetResWidget(QWidget):
    """Horizontal row of tier checkboxes for the multi-scale ``target_res`` knob.

    Reads/writes a list of edge ints (e.g. ``[1024, 1536]``). Never returns an
    empty list — unchecking everything falls back to ``[1024]`` (the legacy
    single ~1MP tier) so preprocess/train always have a valid tier.

    The 1280/1536 tiers are visually flagged as "dangerous" (high token count
    + extra compile graph / VRAM) via colour + an i18n tooltip. Free-fit is the
    only resize mode, so there is no per-tier bucket allow-list any more (the
    snap-era ``resize_bucket_resos`` popup was removed 2026-09-07).
    """

    changed = Signal()

    def __init__(self, selected) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        sel = {int(e) for e in selected} if selected else set()
        tiers, danger = _target_res_tiers()
        self._boxes: dict[int, QCheckBox] = {}
        for edge in tiers:
            edge_box = QWidget()
            edge_lay = QHBoxLayout(edge_box)
            edge_lay.setContentsMargins(0, 0, 0, 0)
            edge_lay.setSpacing(2)
            cb = QCheckBox()
            cb.setChecked(edge in sel)
            cb.toggled.connect(lambda _on: self.changed.emit())
            label = QLabel(str(edge))
            if edge in danger:
                tip = t("target_res_danger_tooltip", edge=edge, tokens=danger[edge])
                cb.setStyleSheet("QCheckBox { color: #d9822b; font-weight: bold; }")
                cb.setToolTip(tip)
                label.setStyleSheet("QLabel { color: #d9822b; font-weight: bold; }")
                label.setToolTip(tip)
            edge_lay.addWidget(cb)
            edge_lay.addWidget(label)
            lay.addWidget(edge_box)
            self._boxes[edge] = cb
        lay.addStretch(1)

    def value(self) -> list[int]:
        out = [e for e, cb in self._boxes.items() if cb.isChecked()]
        return out or [1024]

    def set_value(self, values) -> None:
        if values is None:
            selected = {1024}
        elif isinstance(values, (list, tuple, set)):
            selected = {int(v) for v in values}
        elif isinstance(values, str):
            selected = {int(v) for v in values.replace(",", " ").split() if v}
        else:
            selected = {int(values)}
        for edge, cb in self._boxes.items():
            cb.blockSignals(True)
            cb.setChecked(edge in selected)
            cb.blockSignals(False)
