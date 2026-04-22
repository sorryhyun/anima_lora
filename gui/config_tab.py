"""ConfigTab — training config editor with field tooltips and LoRA variant guide."""

from __future__ import annotations

import re
import shutil
import sys
import time
from typing import Any

import html

import toml
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import (
    CONFIGS_DIR,
    GUI_METHODS_DIR,
    IMAGE_EXTS,
    PRESETS_FILE,
    ROOT,
    _GROUPS,
    _K2G,
    _SKIP,
    _load,
    _load_all_presets,
    _read,
    _save,
    _widget,
    list_gui_variants,
    list_methods,
    list_presets,
    merged_gui_variant_preset,
)
from gui.explanations import field_help, method_guide
from gui.i18n import t

# Matches tqdm lines like:
#   "Denoising steps:  40%|####      | 12/30 [00:12<00:34,  2.50it/s]"
# The trailing "[...]" block carries the rate as either "X.XXit/s" or
# "X.XXs/it"; both are captured optionally so non-timed bars still parse.
_TQDM_RE = re.compile(
    r"^(?P<label>.*?):?\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r"(?:[^\[]*\[[^\]]*?(?P<rate>[\d.]+)(?P<unit>it/s|s/it)[^\]]*\])?"
)


class ClickableLabel(QLabel):
    """QLabel that emits `clicked` on left-click."""

    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class ConfigTab(QWidget):
    def __init__(self):
        super().__init__()
        self._w: dict[str, QWidget] = {}
        self._preprocessed = (ROOT / "post_image_dataset").exists()
        # (monotonic_anchor_time, anchor_step, label, total) — we measure
        # s/step from the first step *completion*, not process launch, so
        # warmup (model load, compilation) doesn't inflate the reported rate.
        self._rate_anchor: tuple[float, int, str, int] | None = None
        lay = QVBoxLayout(self)

        # Top bar: method + preset + save + preprocess + train + stop
        top = QHBoxLayout()
        top.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(list_methods())
        self.method_combo.currentTextChanged.connect(lambda _: self._on_method_changed())
        top.addWidget(self.method_combo, 1)

        top.addWidget(QLabel(t("preset")))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list_presets())
        default_idx = self.preset_combo.findText("default")
        if default_idx >= 0:
            self.preset_combo.setCurrentIndex(default_idx)
        self.preset_combo.currentTextChanged.connect(lambda _: self._reload())
        top.addWidget(self.preset_combo, 1)

        save_btn = QPushButton(t("save"))
        save_btn.clicked.connect(self._save_preset)
        top.addWidget(save_btn)

        self.preprocess_btn = QPushButton(t("preprocess"))
        self._preprocess_idle_style = (
            "background:#2980b9;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._preprocess_busy_style = (
            "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.preprocess_btn.setStyleSheet(self._preprocess_idle_style)
        self.preprocess_btn.clicked.connect(self._start_preprocess)
        top.addWidget(self.preprocess_btn)

        self.train_btn = QPushButton(t("train"))
        self._train_idle_style = (
            "background:#27ae60;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._train_busy_style = (
            "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.clicked.connect(self._start_training)
        self.train_btn.setEnabled(self._preprocessed)
        top.addWidget(self.train_btn)

        self.test_btn = QPushButton(t("test"))
        self._test_idle_style = (
            "background:#8e44ad;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._test_busy_style = (
            "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.test_btn.setStyleSheet(self._test_idle_style)
        self.test_btn.clicked.connect(self._start_test)
        self.test_btn.setEnabled(self._has_lora_output())
        top.addWidget(self.test_btn)

        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)

        lay.addLayout(top)

        # Variant picker — each entry is a self-contained file under
        # configs/gui-methods/. Selecting a variant immediately reloads the
        # form from that file; saves and training invocations target the same
        # file. No toggle-block overlays — what you see is what runs.
        self.variant_row = QWidget()
        vrow = QHBoxLayout(self.variant_row)
        vrow.setContentsMargins(0, 0, 0, 0)
        vrow.addWidget(QLabel(t("variant")))
        self.variant_combo = QComboBox()
        self.variant_combo.currentTextChanged.connect(lambda _: self._reload())
        vrow.addWidget(self.variant_combo, 1)
        self.show_guide_btn = QPushButton(t("show_guide"))
        self.show_guide_btn.setToolTip(t("show_guide_tooltip"))
        self.show_guide_btn.clicked.connect(self._show_explain_placeholder)
        vrow.addWidget(self.show_guide_btn)
        lay.addWidget(self.variant_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.progress.setVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar { border: 1px solid #444; border-radius: 3px;"
            " text-align: center; padding: 1px; font-size: 11px; }"
            "QProgressBar::chunk { background: #27ae60; }"
        )
        lay.addWidget(self.progress)

        # Vertical splitter: config form on top, log on bottom
        vsplit = QSplitter(Qt.Vertical)

        # Horizontal splitter: form on left, explanation panel on right
        hsplit = QSplitter(Qt.Horizontal)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._form = QWidget()
        self._fl = QVBoxLayout(self._form)
        sc.setWidget(self._form)
        hsplit.addWidget(sc)

        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            "QTextBrowser { font-size: 13px; padding: 12px; background: #2b2b2b; color: #e0e0e0; }"
        )
        self._explain.setMinimumWidth(320)
        self._show_explain_placeholder()
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([720, 420])

        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("log_placeholder"))
        vsplit.addWidget(self.log)

        vsplit.setSizes([500, 200])
        lay.addWidget(vsplit)

        # QProcess for training
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)
        self._stdout_buf = ""
        self._stderr_buf = ""

        self._origin: dict[str, str] = {}
        self._reload()

    def _current(self) -> tuple[str, str]:
        return self.method_combo.currentText(), self.preset_combo.currentText()

    def _current_variant(self) -> str:
        """gui-methods variant for the selected method. Falls back to the
        method name itself when no variants are registered (apex, graft)."""
        v = self.variant_combo.currentText()
        return v or self.method_combo.currentText()

    def _on_method_changed(self):
        self._reload()

    def _refresh_variant_row(self, method: str) -> None:
        variants = list_gui_variants(method)
        current = [self.variant_combo.itemText(i) for i in range(self.variant_combo.count())]
        # Rebuilding the combo resets currentText to the first item, which
        # would clobber the user's selection on every _reload. Only rebuild
        # when the variant list actually changed (i.e. method family switched).
        if current != variants:
            self.variant_combo.blockSignals(True)
            self.variant_combo.clear()
            if variants:
                self.variant_combo.addItems(variants)
            self.variant_combo.blockSignals(False)
        # Single-variant families (apex, graft) don't need a picker.
        self.variant_row.setVisible(len(variants) > 1)

    def _reload(self):
        method, preset = self._current()
        if not method or not preset:
            return
        self._refresh_variant_row(method)
        variant = self._current_variant()
        merged, origin = merged_gui_variant_preset(variant, preset)
        cfg = {k: v for k, v in merged.items() if k not in _SKIP}

        self._origin = origin

        if hasattr(self, "_explain"):
            self._show_explain_placeholder()

        self._w.clear()
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # Grouped training config fields
        groups: dict[str, dict] = {g: {} for g in _GROUPS}
        groups["Other"] = {}
        for k, v in cfg.items():
            groups.setdefault(_K2G.get(k, "Other"), {})[k] = v

        origin_style = {
            "base": (
                "color:#888; text-decoration: underline dotted;",
                "from base.toml",
            ),
            "preset": (
                "color:#6aa4d8; text-decoration: underline dotted;",
                f"from presets/{preset}.toml",
            ),
            "method": (
                "color:#f0f0f0; text-decoration: underline dotted;",
                f"from gui-methods/{variant}.toml",
            ),
        }

        for gn, flds in groups.items():
            if not flds:
                continue
            box = QGroupBox(gn)
            form = QFormLayout()
            for k in sorted(flds):
                w = _widget(flds[k], key=k)
                self._w[k] = w
                lbl = ClickableLabel(k)

                help_text = field_help(k)
                notes: list[str] = []
                style, note = origin_style.get(
                    self._origin.get(k, "base"), origin_style["base"]
                )
                lbl.setStyleSheet(style)
                notes.append(note)

                lbl.clicked.connect(
                    lambda _k=k, _h=help_text, _n=tuple(notes): self._show_explain(
                        _k, _h, _n
                    )
                )

                form.addRow(lbl, w)
            box.setLayout(form)
            self._fl.addWidget(box)

        self._fl.addStretch()

    # ── Explanation panel ──

    def _show_explain_placeholder(self) -> None:
        # When the current method ships variant presets, the right-panel
        # default is the variant guide + Apply-semantics callout (replacing
        # the old collapsible box on the left-side form).
        method = self.method_combo.currentText() if hasattr(self, "method_combo") else ""
        guide = method_guide(method)
        if guide:
            self._explain.setHtml(guide)
            return
        self._explain.setHtml(
            f"<p style='color:#888; font-style:italic;'>{html.escape(t('click_field_for_help'))}</p>"
        )

    def _show_test_output(self) -> None:
        d = ROOT / "output" / "tests"
        imgs: list = []
        if d.is_dir():
            imgs = sorted(
                (p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:4]
        title = html.escape(t("test_output_title"))
        if not imgs:
            self._explain.setHtml(
                f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{title}</h2>"
                f"<p style='color:#888; font-style:italic;'>{html.escape(t('test_output_empty'))}</p>"
            )
            return
        parts = [f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{title}</h2>"]
        for p in imgs:
            url = p.resolve().as_uri()
            parts.append(
                f"<p style='margin:0 0 10px 0;'>"
                f"<img src='{url}' style='max-width:100%;'/><br/>"
                f"<span style='color:#aaa; font-size:11px;'>{html.escape(p.name)}</span>"
                f"</p>"
            )
        self._explain.setHtml("".join(parts))

    def _show_explain(
        self, field: str, help_text: str | None, notes: tuple[str, ...]
    ) -> None:
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{html.escape(field)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:14px; line-height:1.6;'>{html.escape(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:#888; font-style:italic;'>{html.escape(t('no_help_available'))}</p>"
            )
        for note in notes:
            parts.append(
                f"<p style='color:#aaa; font-style:italic; margin-top:12px;'>• {html.escape(note)}</p>"
            )
        self._explain.setHtml("".join(parts))

    # ── Save ──

    def _route_key(self, key: str) -> str:
        """Decide which file (method|preset) a key's edit should be written to."""
        origin = self._origin.get(key)
        if origin in ("method", "preset"):
            return origin
        # New key (from base or default widgets): route by group.
        if _K2G.get(key) == "Performance":
            return "preset"
        return "method"

    def _save_preset(self, *, silent: bool = False):
        _, preset = self._current()
        variant = self._current_variant()
        method_path = GUI_METHODS_DIR / f"{variant}.toml"

        method_orig = _load(method_path)
        all_presets = _load_all_presets()
        preset_orig = dict(all_presets.get(preset, {}))
        base = _load(CONFIGS_DIR / "base.toml")

        method_out: dict[str, Any] = dict(method_orig)
        preset_out: dict[str, Any] = dict(preset_orig)

        for k, w in self._w.items():
            v = _read(w, (method_orig.get(k) or preset_orig.get(k) or base.get(k)))
            target = self._route_key(k)
            if target == "method":
                effective = method_orig.get(k, base.get(k))
                if k in method_orig or effective != v:
                    method_out[k] = v
                preset_out.pop(k, None)
            else:
                effective = preset_orig.get(k, base.get(k))
                if k in preset_orig or effective != v:
                    preset_out[k] = v
                method_out.pop(k, None)

        _save(method_path, method_out)
        all_presets[preset] = preset_out
        PRESETS_FILE.write_text(toml.dumps(all_presets), encoding="utf-8")
        if not silent:
            QMessageBox.information(
                self,
                t("saved"),
                f"Saved gui-methods/{method_path.name} + presets.toml[{preset}]",
            )

    # ── Training ──

    def _has_lora_output(self) -> bool:
        out = ROOT / "output" / "ckpt"
        return out.is_dir() and any(out.glob("*.safetensors"))

    def _start_test(self):
        if not self._has_lora_output():
            QMessageBox.warning(self, t("error"), t("no_lora_for_test"))
            return

        python = sys.executable
        args = ["tasks.py", "test"]

        self.log.clear()
        self._reset_progress()
        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "test"
        self._proc.start(python, args)
        self.test_btn.setText(t("test") + " ...")
        self.test_btn.setStyleSheet(self._test_busy_style)
        self.test_btn.setEnabled(False)
        self.preprocess_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.method_combo.setEnabled(False)
        self.preset_combo.setEnabled(False)

    def _start_preprocess(self):
        python = sys.executable
        args = ["tasks.py", "preprocess"]

        self.log.clear()
        self._reset_progress()
        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "preprocess"
        self._proc.start(python, args)
        self.preprocess_btn.setText(t("preprocess") + " ...")
        self.preprocess_btn.setStyleSheet(self._preprocess_busy_style)
        self.preprocess_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.method_combo.setEnabled(False)
        self.preset_combo.setEnabled(False)

    def _start_training(self):
        if not self._preprocessed:
            QMessageBox.warning(self, t("error"), t("preprocess_required"))
            return

        accelerate = shutil.which("accelerate")
        if not accelerate:
            QMessageBox.warning(self, t("error"), t("accelerate_not_found"))
            return

        _, preset = self._current()
        variant = self._current_variant()
        args = [
            "launch",
            "--num_cpu_threads_per_process",
            "3",
            "--mixed_precision",
            "bf16",
            "train.py",
            "--method",
            variant,
            "--methods_subdir",
            "gui-methods",
            "--preset",
            preset,
        ]

        self.log.clear()
        self._reset_progress()
        self._log(f"> accelerate {' '.join(args)}\n")
        self._running_mode = "train"
        self._proc.start(accelerate, args)
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(self._train_busy_style)
        self.train_btn.setEnabled(False)
        self.preprocess_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.method_combo.setEnabled(False)
        self.preset_combo.setEnabled(False)

    def _stop_training(self):
        if self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    def _read_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode(errors="replace")
        self._stdout_buf = self._handle_stream(self._stdout_buf + data)

    def _read_stderr(self):
        data = self._proc.readAllStandardError().data().decode(errors="replace")
        self._stderr_buf = self._handle_stream(self._stderr_buf + data)

    def _handle_stream(self, buf: str) -> str:
        # Split on \n and \r so tqdm carriage-return updates work too.
        parts = re.split(r"[\r\n]", buf)
        tail = parts[-1]  # incomplete trailing fragment — keep buffered
        for line in parts[:-1]:
            m = _TQDM_RE.search(line)
            if m:
                cur = int(m.group("cur"))
                tot = int(m.group("tot"))
                label = m.group("label").strip() or "progress"
                rate_str = self._update_rate(label, cur, tot)
                if tot > 0:
                    self.progress.setMaximum(tot)
                    self.progress.setValue(cur)
                    self.progress.setFormat(f"{label}: {cur}/{tot} (%p%){rate_str}")
                    if not self.progress.isVisible():
                        self.progress.setVisible(True)
                continue
            if line:
                self._log(line + "\n")
        return tail

    def _update_rate(self, label: str, cur: int, tot: int) -> str:
        """Return a ' — X.XXs/step' suffix measured from the first completed
        step of this bar. The first-step timing is excluded so model-load and
        compile overhead don't skew the rate."""
        now = time.monotonic()
        anchor = self._rate_anchor
        # New bar (label/total changed, or progress rewound) → drop anchor.
        if anchor is None or anchor[2] != label or anchor[3] != tot or cur < anchor[1]:
            if cur >= 1:
                self._rate_anchor = (now, cur, label, tot)
            else:
                self._rate_anchor = None
            return ""
        anchor_time, anchor_step, _, _ = anchor
        steps = cur - anchor_step
        if steps <= 0:
            return ""
        spi = (now - anchor_time) / steps
        return f" — {spi:.2f}s/step"

    def _reset_progress(self):
        self._stdout_buf = ""
        self._stderr_buf = ""
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.progress.setVisible(False)
        self._rate_anchor = None

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus):
        # Flush any buffered partial lines before the finish banner.
        for buf_name in ("_stdout_buf", "_stderr_buf"):
            leftover = getattr(self, buf_name, "")
            if leftover and not _TQDM_RE.search(leftover):
                self._log(leftover + "\n")
            setattr(self, buf_name, "")
        self.progress.setVisible(False)
        self._log(f"\n{t('finished', code=exit_code)}\n")
        mode = getattr(self, "_running_mode", "train")
        if mode == "preprocess" and exit_code == 0:
            self._preprocessed = True
        if mode == "test" and exit_code == 0:
            self._show_test_output()
        self.preprocess_btn.setText(t("preprocess"))
        self.preprocess_btn.setStyleSheet(self._preprocess_idle_style)
        self.preprocess_btn.setEnabled(True)
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.setEnabled(self._preprocessed)
        self.test_btn.setText(t("test"))
        self.test_btn.setStyleSheet(self._test_idle_style)
        self.test_btn.setEnabled(self._has_lora_output())
        self.stop_btn.setEnabled(False)
        self.method_combo.setEnabled(True)
        self.preset_combo.setEnabled(True)

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
