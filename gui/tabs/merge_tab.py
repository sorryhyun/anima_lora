"""MergeTab — bake a LoRA adapter into the base DiT.

Layout mirrors ImageViewerTab: top directory combo, left file list, right
details panel (file stats + bakeability scan + merge options + log).

Runs ``scripts/toolkits/merge_to_dit.py`` (and the merge/extract toolkits) via
``QProcess`` and streams stdout/stderr into the log pane.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import ROOT, LazyTabMixin, _adapter_dirs, _safetensors_in
from gui.i18n import t
from gui.theme import tok
from gui.widgets import action_button, apply_variant
from gui.process import kill_process_tree, setup_kill_safe

_DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"


class PickerLineEdit(QLineEdit):
    """Read-only line edit that opens a file/dir picker on click."""

    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setReadOnly(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"QLineEdit {{ background: {tok('input_bg')}; color: {tok('text')}; "
            f"border: 1px solid {tok('border')}; border-radius: 3px; padding: 2px 6px; }}"
            f"QLineEdit:hover {{ border-color: #3c78c8; background: {tok('input_hover')}; }}"
            f"QLineEdit:disabled {{ color: {tok('text_dim')}; background: {tok('base')}; }}"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(ev)


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _scan_adapter(path: Path) -> dict:
    """Read safetensors keys and classify the adapter's bakeability.

    Returns a dict with verdict (str), severity ('ok'|'partial'|'block'|'unknown'),
    details (list of short strings), and counts per key family.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return {
            "verdict": "safetensors not installed",
            "severity": "unknown",
            "details": [],
            "counts": {},
        }

    counts = {
        "lora_down": 0,
        "lora_up_weight": 0,  # hydra stacked
        "lora_ups": 0,  # hydra split
        "postfix": 0,
        "ortho_sp": 0,  # .S_p keys (OrthoLoRA / OrthoHydraLoRA)
        "other": 0,
    }
    # Postfix mode lives in metadata — its key names share no greppable prefix.
    metadata_mode: str | None = None
    try:
        # framework="np" — keys + metadata only; "pt" would drag torch in (~1.4s).
        with safe_open(str(path), framework="np") as f:
            keys = list(f.keys())
            try:
                meta = f.metadata() or {}
                metadata_mode = meta.get("ss_mode")
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 — surface any read error in UI
        return {
            "verdict": f"unreadable: {exc!s}",
            "severity": "unknown",
            "details": [],
            "counts": {},
        }

    is_postfix = metadata_mode in {
        "postfix",
        "postfix_exp",
        "postfix_func",
        "cond",
    }

    for k in keys:
        if k.endswith(".lora_up_weight"):
            counts["lora_up_weight"] += 1
        elif ".lora_ups." in k:
            counts["lora_ups"] += 1
        elif k.endswith(".lora_down.weight"):
            counts["lora_down"] += 1
        elif k.endswith(".S_p"):
            counts["ortho_sp"] += 1
        elif is_postfix:
            counts["postfix"] += 1
        else:
            counts["other"] += 1

    details = []
    if counts["lora_down"]:
        details.append(f"{counts['lora_down']} LoRA keys")
    if counts["ortho_sp"]:
        details.append(f"{counts['ortho_sp']} OrthoLoRA keys")
    if counts["lora_up_weight"] or counts["lora_ups"]:
        n = counts["lora_up_weight"] + counts["lora_ups"]
        details.append(f"{n} HydraLoRA keys")
    if counts["postfix"]:
        details.append(f"{counts['postfix']} {metadata_mode} keys")

    is_hydra = bool(counts["lora_up_weight"] or counts["lora_ups"])
    is_postfix_only = is_postfix and not (counts["lora_down"] or counts["ortho_sp"])
    has_lora_like = bool(counts["lora_down"] or counts["ortho_sp"])

    if is_hydra:
        verdict = t("merge_verdict_hydra")
        severity = "block"
    elif is_postfix_only:
        verdict = t("merge_verdict_postfix_only")
        severity = "block"
    elif has_lora_like:
        verdict = t("merge_verdict_ready")
        severity = "ok"
    else:
        verdict = t("merge_verdict_unknown")
        severity = "unknown"

    return {
        "verdict": verdict,
        "severity": severity,
        "details": details,
        "counts": counts,
    }


class MergeTab(LazyTabMixin, QWidget):
    def __init__(self):
        super().__init__()

        self._dirs = _adapter_dirs()
        self._files: list[Path] = []
        self._current_scan: dict | None = None
        self._stdout_buf = ""  # line buffer for ANALYZE_RESULT marker detection

        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(t("directory")))
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(list(self._dirs))
        self.dir_combo.currentTextChanged.connect(self._load_dir)
        top.addWidget(self.dir_combo, 1)
        refresh_btn = QPushButton(t("refresh"))
        refresh_btn.clicked.connect(self._refresh_dirs)
        top.addWidget(refresh_btn)
        self.count_label = QLabel()
        top.addWidget(self.count_label)
        lay.addLayout(top)

        sp = QSplitter(Qt.Horizontal)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._show_file)
        self.file_list.itemSelectionChanged.connect(self._on_lora_selection)
        sp.addWidget(self.file_list)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(8, 0, 0, 0)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-weight: bold;")
        rlay.addWidget(self.path_label)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(f"color:{tok('text_dim')};")
        rlay.addWidget(self.stats_label)

        # Mode picker: bake one adapter into the DiT, or fuse N LoRAs into one LoRA.
        # Dropdown (not checkboxes) — the two modes are mutually exclusive and
        # drive quite different option panels below.
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(t("merge_mode")))
        self.mode_combo = QComboBox()
        # Index 0 = Into DiT (default), 1 = Merge LoRAs, 2 = Extract LoRA.
        # _lora_mode()/_extract_mode() key off this index.
        self.mode_combo.addItem(t("merge_mode_dit"))
        self.mode_combo.addItem(t("merge_mode_loras"))
        self.mode_combo.addItem(t("merge_mode_extract"))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        rlay.addLayout(mode_row)

        self.verdict_label = QLabel()
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet(
            f"padding:6px; border-radius:3px; background:{tok('panel')};"
        )
        rlay.addWidget(self.verdict_label)

        opt_box = QGroupBox(t("merge_options"))
        opt_lay = QFormLayout(opt_box)

        self.dit_edit = PickerLineEdit(_DEFAULT_DIT)
        self.dit_edit.clicked.connect(self._browse_dit)
        opt_lay.addRow(t("merge_base_dit"), self.dit_edit)

        self.mult_spin = QDoubleSpinBox()
        self.mult_spin.setRange(-2.0, 2.0)
        self.mult_spin.setSingleStep(0.1)
        self.mult_spin.setDecimals(3)
        self.mult_spin.setValue(1.0)
        self.mult_spin.setToolTip(t("merge_multiplier_tip"))
        opt_lay.addRow(t("merge_multiplier"), self.mult_spin)

        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["bf16", "fp16", "fp32"])
        opt_lay.addRow(t("merge_dtype"), self.dtype_combo)

        self.out_edit = PickerLineEdit()
        self.out_edit.setPlaceholderText(t("merge_out_placeholder"))
        self.out_edit.clicked.connect(self._browse_out)
        opt_lay.addRow(t("merge_out"), self.out_edit)

        self.allow_partial = QCheckBox(t("merge_allow_partial"))
        self.allow_partial.setToolTip(t("merge_allow_partial_tip"))
        opt_lay.addRow("", self.allow_partial)

        self.opt_box = opt_box
        rlay.addWidget(opt_box)

        # LoRA ⊕ LoRA → LoRA options (merge_loras.py); hidden until that mode.
        lora_box = QGroupBox(t("merge_lora_options"))
        lora_lay = QFormLayout(lora_box)

        self.lora_hint = QLabel(t("merge_lora_select_hint"))
        self.lora_hint.setWordWrap(True)
        self.lora_hint.setStyleSheet(f"color:{tok('text_dim')};")
        lora_lay.addRow(self.lora_hint)

        self.lora_selected_label = QLabel()
        self.lora_selected_label.setWordWrap(True)
        self.lora_selected_label.setStyleSheet(f"color:{tok('text_dim')};")
        lora_lay.addRow(t("merge_lora_selected"), self.lora_selected_label)

        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText(t("merge_weights_placeholder"))
        self.weights_edit.setToolTip(t("merge_weights_tip"))
        lora_lay.addRow(t("merge_weights"), self.weights_edit)

        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(["global", "per_module", "off"])
        self.normalize_combo.setToolTip(t("merge_normalize_tip"))
        lora_lay.addRow(t("merge_normalize"), self.normalize_combo)

        self.lora_dtype_combo = QComboBox()
        self.lora_dtype_combo.addItems(["bf16", "fp16", "fp32"])
        lora_lay.addRow(t("merge_dtype"), self.lora_dtype_combo)

        self.lora_out_edit = PickerLineEdit()
        self.lora_out_edit.setPlaceholderText(t("merge_lora_out_placeholder"))
        self.lora_out_edit.clicked.connect(self._browse_out)
        lora_lay.addRow(t("merge_out"), self.lora_out_edit)

        self.lora_box = lora_box
        lora_box.setVisible(False)
        rlay.addWidget(lora_box)

        # ΔW extraction: two full DiT checkpoints → a plain LoRA
        # (scripts/toolkits/extract_delta_lora.py). Inputs are full checkpoints,
        # not adapters, so this mode ignores the left file list entirely.
        extract_box = QGroupBox(t("merge_extract_options"))
        extract_lay = QFormLayout(extract_box)

        self.extract_hint = QLabel(t("merge_extract_hint"))
        self.extract_hint.setWordWrap(True)
        self.extract_hint.setStyleSheet(f"color:{tok('text_dim')};")
        extract_lay.addRow(self.extract_hint)

        self.extract_base_edit = PickerLineEdit(_DEFAULT_DIT)
        self.extract_base_edit.clicked.connect(
            lambda: self._browse_ckpt(self.extract_base_edit)
        )
        extract_lay.addRow(t("merge_extract_base"), self.extract_base_edit)

        self.extract_tuned_edit = PickerLineEdit()
        self.extract_tuned_edit.setPlaceholderText(t("merge_extract_tuned_placeholder"))
        self.extract_tuned_edit.clicked.connect(
            lambda: self._browse_ckpt(self.extract_tuned_edit)
        )
        extract_lay.addRow(t("merge_extract_tuned"), self.extract_tuned_edit)

        self.extract_rank_spin = QSpinBox()
        self.extract_rank_spin.setRange(1, 512)
        self.extract_rank_spin.setValue(96)
        self.extract_rank_spin.setToolTip(t("merge_extract_rank_tip"))
        extract_lay.addRow(t("merge_extract_rank"), self.extract_rank_spin)

        self.extract_adaln_check = QCheckBox(t("merge_extract_include_adaln"))
        self.extract_adaln_check.setToolTip(t("merge_extract_include_adaln_tip"))
        extract_lay.addRow("", self.extract_adaln_check)

        self.extract_adaln_layout_combo = QComboBox()
        self.extract_adaln_layout_combo.addItems(["runtime", "comfy"])
        self.extract_adaln_layout_combo.setToolTip(t("merge_extract_adaln_layout_tip"))
        self.extract_adaln_layout_combo.setEnabled(False)
        self.extract_adaln_check.toggled.connect(
            self.extract_adaln_layout_combo.setEnabled
        )
        extract_lay.addRow(
            t("merge_extract_adaln_layout"), self.extract_adaln_layout_combo
        )

        self.extract_dtype_combo = QComboBox()
        self.extract_dtype_combo.addItems(["bf16", "fp16", "fp32"])
        extract_lay.addRow(t("merge_dtype"), self.extract_dtype_combo)

        self.extract_out_edit = PickerLineEdit()
        self.extract_out_edit.setPlaceholderText(t("merge_extract_out_placeholder"))
        self.extract_out_edit.clicked.connect(self._browse_out)
        extract_lay.addRow(t("merge_out"), self.extract_out_edit)

        self.extract_box = extract_box
        extract_box.setVisible(False)
        rlay.addWidget(extract_box)

        bar = QHBoxLayout()
        self.merge_btn = action_button(
            t("merge_button"), variant="success", on_click=self._start_merge
        )
        bar.addWidget(self.merge_btn)

        # LoRA-mode only: dry-run weight-space interference report (no file written).
        self.analyze_btn = action_button(
            t("merge_analyze_button"), variant="info", on_click=self._start_analyze
        )
        self.analyze_btn.setToolTip(t("merge_analyze_tip"))
        self.analyze_btn.setVisible(False)
        bar.addWidget(self.analyze_btn)

        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_merge)
        bar.addWidget(self.stop_btn)
        bar.addStretch()
        rlay.addLayout(bar)

        # Styled pass/warning banner for the interference analysis — populated
        # from the ANALYZE_RESULT marker that merge_loras.py --analyze emits.
        # The full report still streams into the log below; this is the verdict.
        self.analysis_label = QLabel()
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setVisible(False)
        rlay.addWidget(self.analysis_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("merge_log_placeholder"))
        rlay.addWidget(self.log, 1)

        sp.addWidget(right)
        sp.setSizes([260, 760])
        lay.addWidget(sp, 1)

        # Kill-safe setup so Stop kills the whole subtree (frees the loaded DiT weights).
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)

        self._clear_details()

    def _lazy_init(self) -> None:
        # Deferred to first show so the dir scan + adapter classify don't block window startup.
        if self._dirs:
            self._load_dir(self.dir_combo.currentText())
        else:
            self.count_label.setText(t("merge_no_adapter"))

    def _refresh_dirs(self):
        previous = self.dir_combo.currentText()
        self._dirs = _adapter_dirs()
        self.dir_combo.blockSignals(True)
        self.dir_combo.clear()
        self.dir_combo.addItems(list(self._dirs))
        if previous in self._dirs:
            self.dir_combo.setCurrentText(previous)
        self.dir_combo.blockSignals(False)
        if self._dirs:
            self._load_dir(self.dir_combo.currentText())
        else:
            self.file_list.clear()
            self._files = []
            self._clear_details()
            self.count_label.setText(t("merge_no_adapter"))

    def _load_dir(self, name: str):
        d = self._dirs.get(name)
        if not d:
            self.file_list.clear()
            self._files = []
            self._clear_details()
            return
        self._files = _safetensors_in(d)
        self.file_list.clear()
        for p in self._files:
            self.file_list.addItem(p.name)
        self.count_label.setText(t("n_files", n=len(self._files)))
        if self._files:
            self.file_list.setCurrentRow(0)
        else:
            self._clear_details()

    def _current_file(self) -> Path | None:
        r = self.file_list.currentRow()
        if 0 <= r < len(self._files):
            return self._files[r]
        return None

    def _show_file(self, row: int):
        if not 0 <= row < len(self._files):
            self._clear_details()
            return
        p = self._files[row]
        try:
            rel = p.resolve().relative_to(ROOT)
            shown_path = str(rel)
        except ValueError:
            shown_path = str(p)
        self.path_label.setText(shown_path)

        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        self.stats_label.setText(f"{_format_size(st.st_size)} · {mtime}")

        if self._extract_mode():
            # Clicking a checkpoint in the list picks it as the tuned input
            # (the file picker stays available for dirs not in the combo). Skip
            # the adapter scan — these are full DiT checkpoints, not adapters.
            self.extract_tuned_edit.setText(str(p))
            self._update_extract_enabled()
            return

        scan = _scan_adapter(p)
        self._current_scan = scan
        colors = {
            "ok": ("#0a3d2a", tok("ok")),  # bg (darkened tint, kept), text
            "partial": ("#3d2e0a", tok("warn")),
            "block": ("#3d0a0a", tok("err")),
            "unknown": (tok("panel"), tok("text_dim")),
        }
        bg, fg = colors.get(scan["severity"], colors["unknown"])
        bits = scan["verdict"]
        if scan["details"]:
            bits += "\n" + " · ".join(scan["details"])
        self.verdict_label.setText(bits)
        self.verdict_label.setStyleSheet(
            f"padding:8px; border-radius:3px; background:{bg}; color:{fg};"
        )
        # Match allow-partial to this file so the previous selection's state can't linger.
        self.allow_partial.setChecked(scan["severity"] == "partial")
        if self._lora_mode():
            # LoRA-merge enablement is driven by the multi-selection, not this scan.
            self._on_lora_selection()
            return
        # Only "ok" / "partial" are mergeable.
        self.merge_btn.setEnabled(
            self._proc.state() == QProcess.NotRunning
            and scan["severity"] in ("ok", "partial")
        )

    def _clear_details(self):
        self.path_label.setText("")
        self.stats_label.setText("")
        self.verdict_label.setText(t("merge_no_selection"))
        self.verdict_label.setStyleSheet(
            f"padding:8px; border-radius:3px; background:{tok('panel')}; color:{tok('text_dim')};"
        )
        self._current_scan = None
        self.merge_btn.setEnabled(False)

    def _browse_dit(self):
        start = str(ROOT / "models" / "diffusion_models")
        f, _ = QFileDialog.getOpenFileName(
            self, t("merge_pick_dit"), start, "Safetensors (*.safetensors)"
        )
        if f:
            self.dit_edit.setText(f)

    def _browse_ckpt(self, target: "PickerLineEdit"):
        """Pick a full DiT checkpoint (extract mode base/tuned inputs)."""
        start = str(ROOT / "models" / "diffusion_models")
        f, _ = QFileDialog.getOpenFileName(
            self, t("merge_extract_pick_ckpt"), start, "Safetensors (*.safetensors)"
        )
        if f:
            target.setText(f)
            self._update_extract_enabled()

    def _browse_out(self):
        start = str(ROOT / "output" / "ckpt")
        f, _ = QFileDialog.getSaveFileName(
            self, t("merge_pick_out"), start, "Safetensors (*.safetensors)"
        )
        if f:
            if self._extract_mode():
                target = self.extract_out_edit
            elif self._lora_mode():
                target = self.lora_out_edit
            else:
                target = self.out_edit
            target.setText(f)
            if self._extract_mode():
                self._update_extract_enabled()

    def _lora_mode(self) -> bool:
        return self.mode_combo.currentIndex() == 1

    def _extract_mode(self) -> bool:
        return self.mode_combo.currentIndex() == 2

    def _selected_loras(self) -> list[Path]:
        """Selected adapters in list order (LoRA-merge mode uses multi-select).

        `_files` and `file_list` can be momentarily out of sync — `_load_dir`
        sets `_files` before `file_list.clear()`, and the clear emits
        `itemSelectionChanged`, re-entering here while the widget is empty. Bound
        the scan by the widget's own count so `item(i)` is never None.
        """
        n = min(len(self._files), self.file_list.count())
        return [self._files[i] for i in range(n) if self.file_list.item(i).isSelected()]

    def _on_mode_changed(self):
        lora = self._lora_mode()
        extract = self._extract_mode()
        dit = not lora and not extract
        self.opt_box.setVisible(dit)
        self.lora_box.setVisible(lora)
        self.extract_box.setVisible(extract)
        # Extract has no adapter to scan — hide the verdict banner in that mode.
        self.verdict_label.setVisible(not extract)
        self.file_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
            if lora
            else QAbstractItemView.SingleSelection
        )
        if extract:
            self.merge_btn.setText(t("merge_extract_button"))
        elif lora:
            self.merge_btn.setText(t("merge_lora_button"))
        else:
            self.merge_btn.setText(t("merge_button"))
        self.analyze_btn.setVisible(lora)
        if not lora:
            # The interference banner only applies to LoRA-merge mode.
            self.analysis_label.setVisible(False)
        if extract:
            self._update_extract_enabled()
        elif lora:
            self._on_lora_selection()
        else:
            # Restore single-file enablement from the current scan.
            self._show_file(self.file_list.currentRow())

    def _on_lora_selection(self):
        if not self._lora_mode():
            return
        sel = self._selected_loras()
        self.lora_selected_label.setText(
            "\n".join(f"{i + 1}. {p.name}" for i, p in enumerate(sel))
            or t("merge_lora_need_two")
        )
        enabled = self._proc.state() == QProcess.NotRunning and len(sel) >= 2
        self.merge_btn.setEnabled(enabled)
        self.analyze_btn.setEnabled(enabled)

    def _start_merge(self):
        if self._proc.state() != QProcess.NotRunning:
            return
        if self._extract_mode():
            self._start_extract()
        elif self._lora_mode():
            self._start_lora_merge()
        else:
            self._start_dit_merge()

    # Torch save_dtype names (extract_delta_lora.py does getattr(torch, ...)).
    _EXTRACT_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

    def _update_extract_enabled(self):
        """Extract button gates on its own pickers, independent of the file list."""
        if not self._extract_mode():
            return
        ok = (
            self._proc.state() == QProcess.NotRunning
            and bool(self.extract_tuned_edit.text().strip())
            and bool(self.extract_out_edit.text().strip())
        )
        self.merge_btn.setEnabled(ok)

    def _start_extract(self):
        tuned = self.extract_tuned_edit.text().strip()
        out = self.extract_out_edit.text().strip()
        if not tuned:
            QMessageBox.warning(self, t("error"), t("merge_extract_no_tuned"))
            return
        if not out:
            QMessageBox.warning(self, t("error"), t("merge_extract_no_out"))
            return
        args = [
            "scripts/toolkits/extract_delta_lora.py",
            "--base",
            self.extract_base_edit.text().strip() or _DEFAULT_DIT,
            "--tuned",
            tuned,
            "--rank",
            str(self.extract_rank_spin.value()),
            "--out",
            out,
            "--save_dtype",
            self._EXTRACT_DTYPE[self.extract_dtype_combo.currentText()],
        ]
        if self.extract_adaln_check.isChecked():
            args += [
                "--include_adaln",
                "--adaln_layout",
                self.extract_adaln_layout_combo.currentText(),
            ]
        self._launch(args)

    def _start_dit_merge(self):
        adapter = self._current_file()
        if adapter is None or not adapter.exists():
            QMessageBox.warning(self, t("error"), t("merge_no_adapter_msg"))
            return

        args = [
            "scripts/toolkits/merge_to_dit.py",
            "--adapter",
            str(adapter),
            "--dit",
            self.dit_edit.text().strip() or _DEFAULT_DIT,
            "--multiplier",
            f"{self.mult_spin.value():g}",
            "--dtype",
            self.dtype_combo.currentText(),
        ]
        out = self.out_edit.text().strip()
        if out:
            args += ["--out", out]
        if self.allow_partial.isChecked():
            args.append("--allow-partial")
        self._launch(args)

    def _start_lora_merge(self):
        sel = self._selected_loras()
        if len(sel) < 2:
            QMessageBox.warning(self, t("error"), t("merge_lora_need_two"))
            return

        args = ["scripts/toolkits/merge_loras.py"]
        args += [str(p) for p in sel]
        args += [
            "--normalize",
            self.normalize_combo.currentText(),
            "--dtype",
            self.lora_dtype_combo.currentText(),
        ]
        # Empty → merge_loras.py defaults to <first-stem>_merged<N>.safetensors.
        out = self.lora_out_edit.text().strip()
        if out:
            args += ["--out", out]
        weights = self.weights_edit.text().strip()
        if weights:
            n = len(weights.split(","))
            if n != len(sel):
                QMessageBox.warning(
                    self, t("error"), t("merge_weights_mismatch", n=n, m=len(sel))
                )
                return
            args += ["--weights", weights]
        self._launch(args)

    def _start_analyze(self):
        """Dry-run interference report for the selected LoRAs (writes nothing)."""
        if self._proc.state() != QProcess.NotRunning:
            return
        sel = self._selected_loras()
        if len(sel) < 2:
            QMessageBox.warning(self, t("error"), t("merge_lora_need_two"))
            return
        args = ["scripts/toolkits/merge_loras.py", *[str(p) for p in sel], "--analyze"]
        weights = self.weights_edit.text().strip()
        if weights:
            n = len(weights.split(","))
            if n != len(sel):
                QMessageBox.warning(
                    self, t("error"), t("merge_weights_mismatch", n=n, m=len(sel))
                )
                return
            args += ["--weights", weights]
        self._launch(args)

    def _launch(self, args: list[str]):
        import sys as _sys

        self.log.clear()
        self._stdout_buf = ""
        self.analysis_label.setVisible(False)
        self._log(f"> {_sys.executable} {' '.join(args)}\n")

        self.merge_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        apply_variant(self.merge_btn, "busy")
        self.merge_btn.setText(self.merge_btn.text() + " ...")
        self.stop_btn.setEnabled(True)
        self.dir_combo.setEnabled(False)
        self.mode_combo.setEnabled(False)

        self._proc.start(_sys.executable, args)

    def _stop_merge(self):
        kill_process_tree(self._proc)

    def cleanup_subprocess(self):
        """Hook for app shutdown — kill any running launcher + descendants."""
        kill_process_tree(self._proc)

    def _read_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode(errors="replace")
        # Scan complete lines for the ANALYZE_RESULT marker; route it to the
        # styled banner and keep it out of the visible log. Anything before the
        # last newline is a complete line; the remainder stays buffered.
        self._stdout_buf += data
        *lines, self._stdout_buf = self._stdout_buf.split("\n")
        visible = []
        for ln in lines:
            if ln.startswith("ANALYZE_RESULT "):
                self._apply_analysis_marker(ln[len("ANALYZE_RESULT ") :])
            else:
                visible.append(ln)
        if visible:
            self._log("\n".join(visible) + "\n")

    # Banner severity grades on the strongest pairwise |cos| (operator alignment),
    # NOT the verdict word or the N-inflated energy ratio. Near-orthogonal LoRAs
    # (the common case for distinct artists, ~0.06) are safe to merge — and
    # normalize=global tames summed magnitude regardless. These bands are heuristic.
    _SAFE_COS = 0.15
    _STRONG_COS = 0.35
    # A near-zero cosine can still hide a subspace collision (both LoRAs writing
    # the same weight subspaces in different directions) — escalate the "safe"
    # banner when the worst pair's output-overlap ×random factor is colliding.
    # Mirrors merge_analysis.OVERLAP_COLLIDING_X (not imported: that module pulls
    # torch, which the GUI must not).
    _OVERLAP_WARN_X = 8.0

    def _apply_analysis_marker(self, payload: str):
        """Render the interference result as a colored safe/caution/warn banner."""
        import json as _json

        try:
            data = _json.loads(payload)
        except ValueError:
            return
        ratio = data.get("ratio", 0.0)
        shared = data.get("shared", 0)
        modules = data.get("modules", 0)
        strongest = data.get("strongest")
        if strongest and len(strongest) == 3:
            a, b, cos = strongest[0], strongest[1], float(strongest[2])
        else:
            a = b = None
            cos = ratio - 1.0  # fall back to ratio deviation if no pair
        mag = abs(cos)
        overlap = data.get("overlap")
        if overlap and len(overlap) == 4:
            ov_a, ov_b = overlap[0], overlap[1]
            ov_out, ov_x = float(overlap[2]), float(overlap[3])
        else:
            ov_a = ov_b = None
            ov_out = ov_x = 0.0

        if mag < self._SAFE_COS and ov_x >= self._OVERLAP_WARN_X:
            # Directions near-orthogonal but the pair still writes into the
            # same weight subspaces — the cosine-based "safe" would be false
            # comfort, so surface the collision as a caution.
            text = t(
                "merge_analysis_overlap",
                a=ov_a,
                b=ov_b,
                overlap=f"{ov_out:.3f}",
                x=f"{ov_x:.0f}",
                ratio=f"{ratio:.3f}",
                shared=shared,
                modules=modules,
            )
            sev, bg = "warn", "#3d2e0a"
        elif mag < self._SAFE_COS:
            # Near-orthogonal — safe to merge regardless of sign.
            text = t(
                "merge_analysis_safe",
                cos=f"{mag:.3f}",
                ratio=f"{ratio:.3f}",
                shared=shared,
                modules=modules,
            )
            sev, bg = "ok", "#0a3d2a"
        else:
            strong = mag >= self._STRONG_COS
            strength = t(
                "merge_analysis_strong" if strong else "merge_analysis_moderate"
            )
            reinforcing = cos >= 0
            key = "merge_analysis_reinforce" if reinforcing else "merge_analysis_cancel"
            text = t(
                key,
                strength=strength,
                a=a,
                b=b,
                cos=f"{cos:+.3f}",
                ratio=f"{ratio:.3f}",
                shared=shared,
                modules=modules,
            )
            # Reinforcement is benign — the default normalize=global rescales the
            # aligned sum back to single-LoRA magnitude, so there is no overdrive
            # (it only signals near-duplicate ingredients ⇒ caution, never a red
            # alert; common for soup-of-soups off a shared init). Only genuine
            # cancellation, where the deltas erase each other, goes red when
            # strong. Mirrors the sign-aware CLI band in library.anima.merge_analysis.
            if reinforcing:
                sev, bg = "warn", "#3d2e0a"
            else:
                sev, bg = ("err", "#3d0a0a") if strong else ("warn", "#3d2e0a")

        self.analysis_label.setText(text)
        self.analysis_label.setStyleSheet(
            f"padding:8px; border-radius:3px; background:{bg}; color:{tok(sev)};"
        )
        self.analysis_label.setVisible(True)

    def _read_stderr(self):
        data = self._proc.readAllStandardError().data().decode(errors="replace")
        self._log(data)

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus):
        # Flush any trailing partial line left in the stdout buffer.
        if self._stdout_buf:
            tail, self._stdout_buf = self._stdout_buf, ""
            if tail.startswith("ANALYZE_RESULT "):
                self._apply_analysis_marker(tail[len("ANALYZE_RESULT ") :])
            else:
                self._log(tail)
        self._log(f"\n{t('finished', code=exit_code)}\n")
        apply_variant(self.merge_btn, "success")
        self.stop_btn.setEnabled(False)
        self.dir_combo.setEnabled(True)
        self.mode_combo.setEnabled(True)
        if self._extract_mode():
            self.merge_btn.setText(t("merge_extract_button"))
            self._update_extract_enabled()
        elif self._lora_mode():
            self.merge_btn.setText(t("merge_lora_button"))
            self._on_lora_selection()
        else:
            self.merge_btn.setText(t("merge_button"))
            self.merge_btn.setEnabled(
                self._current_scan is not None
                and self._current_scan.get("severity") in ("ok", "partial")
            )
        # A merged file may have been created — refresh so it shows up.
        if exit_code == 0:
            self._refresh_dirs()

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
