"""Structured editor for train.py's one-line sample-prompt syntax."""

from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import t
from gui.theme import tok
from gui.widgets._qt_utils import _no_wheel


class _SamplePromptRow(QFrame):
    """One sample prompt, read back in train.py's one-line syntax."""

    changed = Signal()

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ border:1px solid {tok('border_dim')}; border-radius:4px; }} "
            "QLabel, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, "
            "QCheckBox { border:0; }"
        )

        lay = QGridLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setHorizontalSpacing(12)
        lay.setVerticalSpacing(6)

        self.select_box = QCheckBox(t("sample_prompt_select"))
        lay.addWidget(self.select_box, 0, 0, 1, 2)

        prompt_stack = QVBoxLayout()
        prompt_stack.setContentsMargins(0, 0, 0, 0)
        prompt_stack.setSpacing(4)
        lay.addLayout(prompt_stack, 1, 0)

        prompt_label = QLabel(t("sample_prompt_col_prompt"))
        prompt_stack.addWidget(prompt_label)
        self.prompt_edit = QPlainTextEdit(str(data.get("prompt", "")))
        self.prompt_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.prompt_edit.setMinimumHeight(118)
        self.prompt_edit.setPlaceholderText(t("sample_prompt_prompt_placeholder"))
        self.prompt_edit.textChanged.connect(self.changed.emit)
        prompt_stack.addWidget(self.prompt_edit)

        negative_label = QLabel(t("sample_prompt_col_negative"))
        negative_label.setToolTip(t("sample_prompt_tip_negative"))
        prompt_stack.addWidget(negative_label)
        self.negative = QPlainTextEdit(str(data.get("negative", "")))
        self.negative.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.negative.setMinimumHeight(72)
        self.negative.setMaximumHeight(96)
        self.negative.setPlaceholderText(t("sample_prompt_default_negative"))
        self.negative.setToolTip(t("sample_prompt_tip_negative"))
        self.negative.textChanged.connect(self.changed.emit)
        prompt_stack.addWidget(self.negative)

        opts = QGridLayout()
        opts.setContentsMargins(0, 0, 0, 0)
        opts.setHorizontalSpacing(8)
        opts.setVerticalSpacing(4)
        lay.addLayout(opts, 1, 1)

        self.width = self._int_spin(
            data.get("width", 0),
            default_label=t("sample_prompt_default_width"),
            tooltip=t("sample_prompt_tip_width"),
        )
        self.height = self._int_spin(
            data.get("height", 0),
            default_label=t("sample_prompt_default_height"),
            tooltip=t("sample_prompt_tip_height"),
        )
        self.steps = self._int_spin(
            data.get("steps", 0),
            maximum=1000,
            default_label=t("sample_prompt_default_steps"),
            tooltip=t("sample_prompt_tip_steps"),
        )
        self.seed = self._seed_spin(data.get("seed"))
        self.scale = self._float_spin(
            data.get("scale", 0.0),
            default_label=t("sample_prompt_default_cfg"),
            tooltip=t("sample_prompt_tip_cfg"),
        )
        self.guidance = self._float_spin(
            data.get("guidance", 0.0),
            default_label=t("sample_prompt_default_guidance"),
            tooltip=t("sample_prompt_tip_guidance"),
        )
        self.flow_shift = self._float_spin(
            data.get("flow_shift", 0.0),
            default_label=t("sample_prompt_default_shift"),
            tooltip=t("sample_prompt_tip_shift"),
        )
        self.extra = QLineEdit(str(data.get("extra", "")))
        self.extra.setToolTip(t("sample_prompt_tip_extra"))
        self.extra.textChanged.connect(lambda *_: self.changed.emit())

        for row, (label_text, widget, tooltip) in enumerate(
            (
                (
                    t("sample_prompt_col_width"),
                    self.width,
                    t("sample_prompt_tip_width"),
                ),
                (
                    t("sample_prompt_col_height"),
                    self.height,
                    t("sample_prompt_tip_height"),
                ),
                (
                    t("sample_prompt_col_steps"),
                    self.steps,
                    t("sample_prompt_tip_steps"),
                ),
                (t("sample_prompt_col_seed"), self.seed, t("sample_prompt_tip_seed")),
                (t("sample_prompt_col_cfg"), self.scale, t("sample_prompt_tip_cfg")),
                (
                    t("sample_prompt_col_guidance"),
                    self.guidance,
                    t("sample_prompt_tip_guidance"),
                ),
                (
                    t("sample_prompt_col_shift"),
                    self.flow_shift,
                    t("sample_prompt_tip_shift"),
                ),
                (
                    t("sample_prompt_col_extra"),
                    self.extra,
                    t("sample_prompt_tip_extra"),
                ),
            )
        ):
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            opts.addWidget(label, row, 0)
            opts.addWidget(widget, row, 1)

        lay.setColumnStretch(0, 4)
        lay.setColumnStretch(1, 2)
        self.setMinimumHeight(300)

    def _int_spin(
        self,
        value: int = 0,
        *,
        maximum: int = 8192,
        default_label: str,
        tooltip: str = "",
        unset: int = 0,
    ) -> QSpinBox:
        w = QSpinBox()
        w.setRange(unset, maximum)
        w.setSpecialValueText(default_label)
        w.setValue(int(value or unset))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        if tooltip:
            w.setToolTip(tooltip)
        return _no_wheel(w)

    def _seed_spin(self, value: int | None = None) -> QSpinBox:
        w = QSpinBox()
        w.setRange(-1, 2_147_483_647)
        w.setSpecialValueText(t("sample_prompt_default_seed"))
        w.setValue(-1 if value is None else int(value))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        w.setToolTip(t("sample_prompt_tip_seed"))
        return _no_wheel(w)

    def _float_spin(
        self, value: float = 0.0, *, default_label: str, tooltip: str = ""
    ) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(0.0, 100.0)
        w.setDecimals(2)
        w.setSingleStep(0.5)
        w.setSpecialValueText(default_label)
        w.setValue(float(value or 0.0))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        if tooltip:
            w.setToolTip(tooltip)
        return _no_wheel(w)

    @staticmethod
    def _single_line(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def value(self) -> str | None:
        prompt = self._single_line(self.prompt_edit.toPlainText())
        if not prompt:
            return None
        parts = [prompt]
        for widget, flag in (
            (self.width, "w"),
            (self.height, "h"),
            (self.steps, "s"),
        ):
            val = widget.value()
            if val > 0:
                parts.append(f"--{flag} {val}")
        seed = self.seed.value()
        if seed >= 0:
            parts.append(f"--d {seed}")
        for widget, flag in (
            (self.scale, "l"),
            (self.guidance, "g"),
            (self.flow_shift, "fs"),
        ):
            val = widget.value()
            if val > 0:
                parts.append(f"--{flag} {val:g}")
        negative = self._single_line(self.negative.toPlainText())
        if negative:
            parts.append(f"--n {negative}")
        extra = self._single_line(self.extra.text())
        if extra:
            parts.append(extra if extra.startswith("--") else "--" + extra)
        return " ".join(parts)


class _SamplePromptsWidget(QWidget):
    """Structured editor for train.py's one-line sample prompt syntax."""

    changed = Signal()

    _COLLAPSED_HEIGHT = 300
    _EXPANDED_HEIGHT = 650

    def __init__(self, prompts, fill: bool = False) -> None:
        super().__init__()
        # ``fill``: grow with the host (the dialog). Otherwise a fixed height
        # with an expand/collapse toggle.
        self._fill = fill
        self._expanded = False
        self._rows: list[_SamplePromptRow] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        hint = QLabel(t("sample_prompt_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{tok('text_dim')};")
        lay.addWidget(hint)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding if fill else QSizePolicy.Fixed,
        )
        self.scroll.setStyleSheet(
            f"""
            QScrollBar:vertical {{
                background: {tok("scroll_bg")};
                width: 14px;
                margin: 0;
            }}
            QScrollBar:horizontal {{
                background: {tok("scroll_bg")};
                height: 14px;
                margin: 0;
            }}
            QScrollBar::handle {{
                background: {tok("scroll_handle")};
                border-radius: 5px;
                min-height: 24px;
                min-width: 24px;
            }}
            QScrollBar::handle:hover {{
                background: {tok("scroll_handle_hover")};
            }}
            QScrollBar::add-line,
            QScrollBar::sub-line {{
                width: 0;
                height: 0;
            }}
            """
        )
        self._rows_widget = QWidget()
        self._row_layout = QVBoxLayout(self._rows_widget)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(8)
        self._row_layout.addStretch(1)
        self.scroll.setWidget(self._rows_widget)
        lay.addWidget(self.scroll)

        row_lay = QHBoxLayout()
        row_lay.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton(t("sample_prompt_add"))
        add_btn.clicked.connect(lambda: self._add_row({}))
        row_lay.addWidget(add_btn)
        select_all_btn = QPushButton(t("sample_prompt_select_all"))
        select_all_btn.clicked.connect(self._select_all)
        row_lay.addWidget(select_all_btn)
        remove_btn = QPushButton(t("sample_prompt_remove"))
        remove_btn.clicked.connect(self._remove_selected)
        row_lay.addWidget(remove_btn)
        self.expand_btn: QPushButton | None = None
        if not fill:
            self.expand_btn = QPushButton(t("sample_prompt_expand"))
            self.expand_btn.clicked.connect(self._toggle_expanded)
            row_lay.addWidget(self.expand_btn)
        row_lay.addStretch(1)
        lay.addLayout(row_lay)

        rows = self._parse_prompts(prompts)
        if rows:
            for row in rows:
                self._add_row(row)
        else:
            self._add_row({})
        self._apply_height()

    def _apply_height(self) -> None:
        if self._fill or self.expand_btn is None:
            return
        height = self._EXPANDED_HEIGHT if self._expanded else self._COLLAPSED_HEIGHT
        self.scroll.setMinimumHeight(height)
        self.scroll.setMaximumHeight(height)
        self.expand_btn.setText(
            t("sample_prompt_collapse") if self._expanded else t("sample_prompt_expand")
        )

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._apply_height()

    @staticmethod
    def _parse_prompts(prompts) -> list[dict[str, Any]]:
        if isinstance(prompts, (list, tuple)):
            lines = [str(p).strip() for p in prompts]
        elif prompts is None:
            lines = []
        else:
            lines = [ln.strip() for ln in str(prompts).splitlines()]
        return [
            _SamplePromptsWidget._parse_line(ln)
            for ln in lines
            if ln and not ln.startswith("#")
        ]

    @staticmethod
    def _parse_line(line: str) -> dict[str, Any]:
        parts = line.split(" --")
        out: dict[str, Any] = {"prompt": parts[0].strip()}
        extra: list[str] = []
        for part in parts[1:]:
            try:
                if m := re.match(r"w (\d+)$", part, re.IGNORECASE):
                    out["width"] = int(m.group(1))
                elif m := re.match(r"h (\d+)$", part, re.IGNORECASE):
                    out["height"] = int(m.group(1))
                elif m := re.match(r"s (\d+)$", part, re.IGNORECASE):
                    out["steps"] = int(m.group(1))
                elif m := re.match(r"d (\d+)$", part, re.IGNORECASE):
                    out["seed"] = int(m.group(1))
                elif m := re.match(r"l ([\d.]+)$", part, re.IGNORECASE):
                    out["scale"] = float(m.group(1))
                elif m := re.match(r"g ([\d.]+)$", part, re.IGNORECASE):
                    out["guidance"] = float(m.group(1))
                elif m := re.match(r"fs ([\d.]+)$", part, re.IGNORECASE):
                    out["flow_shift"] = float(m.group(1))
                elif m := re.match(r"n (.+)$", part, re.IGNORECASE):
                    out["negative"] = m.group(1).strip()
                else:
                    extra.append("--" + part.strip())
            except ValueError:
                extra.append("--" + part.strip())
        if extra:
            out["extra"] = " ".join(extra)
        return out

    def _add_row(self, data: dict[str, Any]) -> None:
        row = _SamplePromptRow(data)
        row.changed.connect(self.changed.emit)
        self._rows.append(row)
        self._row_layout.insertWidget(max(0, self._row_layout.count() - 1), row)
        self.changed.emit()

    def _select_all(self) -> None:
        for row in self._rows:
            row.select_box.setChecked(True)

    def _remove_selected(self) -> None:
        rows = [row for row in self._rows if row.select_box.isChecked()]
        if not rows:
            return
        answer = QMessageBox.question(
            self,
            t("sample_prompt_remove_confirm_title"),
            t("sample_prompt_remove_confirm_body", n=len(rows)),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        for row in rows:
            if row in self._rows:
                self._rows.remove(row)
                self._row_layout.removeWidget(row)
                row.setParent(None)
                row.deleteLater()
        if not self._rows:
            self._add_row({})
        self.changed.emit()

    def value(self) -> list[str]:
        lines: list[str] = []
        for row in self._rows:
            value = row.value()
            if value:
                lines.append(value)
        return lines


def _normalize_prompt_lines(prompts) -> list[str]:
    """Coerce a stored sample_prompts value (list or multiline string) into a
    clean list of non-empty, non-comment one-line prompts."""
    if isinstance(prompts, (list, tuple)):
        lines = [str(p).strip() for p in prompts]
    elif prompts is None:
        lines = []
    else:
        lines = [ln.strip() for ln in str(prompts).splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


class SamplePromptsDialog(QDialog):
    """Popup hosting the structured sample-prompt editor at full size."""

    def __init__(self, prompts, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("sample_prompt_dialog_title"))
        self.setModal(True)
        self.resize(960, 720)
        lay = QVBoxLayout(self)
        self._editor = _SamplePromptsWidget(prompts, fill=True)
        lay.addWidget(self._editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def value(self) -> list[str]:
        return self._editor.value()


class _SamplePromptsLauncher(QWidget):
    """Compact form field: a button + summary that edits the prompt list in
    :class:`SamplePromptsDialog`."""

    changed = Signal()

    def __init__(self, prompts) -> None:
        super().__init__()
        self._prompts = _normalize_prompt_lines(prompts)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._edit_btn = QPushButton(t("sample_prompt_edit_button"))
        self._edit_btn.clicked.connect(self._open_dialog)
        lay.addWidget(self._edit_btn)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color:{tok('text_dim')};")
        lay.addWidget(self._summary, 1)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n = len(self._prompts)
        if not n:
            self._summary.setText(t("sample_prompt_summary_none"))
            self._summary.setToolTip("")
            return
        first = self._prompts[0].split(" --", 1)[0].strip()
        if len(first) > 60:
            first = first[:57] + "…"
        self._summary.setText(t("sample_prompt_summary_count", n=n, first=first))
        self._summary.setToolTip("\n".join(self._prompts))

    def _open_dialog(self) -> None:
        dlg = SamplePromptsDialog(self._prompts, self)
        if dlg.exec():
            self._prompts = dlg.value()
            self._refresh_summary()
            self.changed.emit()

    def value(self) -> list[str]:
        return list(self._prompts)
