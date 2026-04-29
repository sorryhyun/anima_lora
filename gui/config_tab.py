"""ConfigTab — training config editor with field tooltips and LoRA variant guide."""

from __future__ import annotations

import re
import sys
import time
from typing import Any

import html

import toml
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
    IMAGE_EXTS,
    ROOT,
    _GROUPS,
    _K2G,
    _SKIP,
    _load,
    _read,
    _save,
    _widget,
    is_basic_field,
    list_gui_variants,
    list_methods,
    merged_gui_variant_preset,
    variant_path,
)
from gui.explanations import field_help, method_guide
from gui.i18n import t
from gui.process import kill_process_tree, make_subprocess_env, setup_kill_safe

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
    def __init__(self, methods: list[str] | None = None):
        super().__init__()
        self._w: dict[str, QWidget] = {}
        self._preprocessed = (ROOT / "post_image_dataset").exists()
        # Advanced section starts collapsed; user's expand/collapse state
        # persists across _reload (variant switches, save round-trips).
        self._advanced_expanded = False
        # (monotonic_anchor_time, anchor_step, label, total) — we measure
        # s/step from the first step *completion*, not process launch, so
        # warmup (model load, compilation) doesn't inflate the reported rate.
        self._rate_anchor: tuple[float, int, str, int] | None = None
        lay = QVBoxLayout(self)

        # Top bar: method + save + preprocess + train + stop
        # The preset combo is intentionally absent — gui-methods variants
        # (lora-fast, lora-8gb, etc.) already encode the hardware/perf knobs
        # users used to pick via presets, and all saves now write directly to
        # the current variant file (no preset/variant routing distinction).
        # `methods=` lets callers restrict the picker (e.g. the standard tab
        # shows only lora; the experimental tab mounts a multi-method picker
        # spanning hydralora / reft / postfix / apex). When only one method is
        # allowed, the picker hides itself.
        top = QHBoxLayout()
        method_items = methods if methods is not None else list_methods()
        self._method_label = QLabel("Method")
        top.addWidget(self._method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(method_items)
        self.method_combo.currentTextChanged.connect(
            lambda _: self._on_method_changed()
        )
        top.addWidget(self.method_combo)
        if len(method_items) <= 1:
            self._method_label.setVisible(False)
            self.method_combo.setVisible(False)

        # Variant picker sits inline next to the method picker — selecting a
        # variant swaps the gui-methods/<variant>.toml file the form is bound
        # to. "+ New" creates a custom variant under gui-methods/custom/;
        # "Guide" replays the method-level help in the right panel.
        self._variant_label = QLabel(t("variant"))
        top.addWidget(self._variant_label)
        self.variant_combo = QComboBox()
        self.variant_combo.currentTextChanged.connect(lambda _: self._reload())
        top.addWidget(self.variant_combo, 1)
        self.new_variant_btn = QPushButton(t("new_variant"))
        self.new_variant_btn.setToolTip(t("new_variant_tooltip"))
        self.new_variant_btn.clicked.connect(self._create_variant)
        top.addWidget(self.new_variant_btn)
        self.show_guide_btn = QPushButton(t("show_guide"))
        self.show_guide_btn.setToolTip(t("show_guide_tooltip"))
        self.show_guide_btn.clicked.connect(self._show_explain_placeholder)
        top.addWidget(self.show_guide_btn)

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
        outer = QVBoxLayout(self._form)
        outer.setContentsMargins(0, 0, 0, 0)

        # Inner container holds the dynamically-rebuilt grouped form fields
        # (cleared on every _reload). The extra-args button and textarea sit
        # below it inside the same scroll area, but outside the cleared layout
        # so they persist across reloads.
        self._form_inner = QWidget()
        self._fl = QVBoxLayout(self._form_inner)
        self._fl.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._form_inner)

        self.extra_args_btn = QPushButton(t("extra_args_toggle"))
        self.extra_args_btn.setCheckable(True)
        self.extra_args_btn.setToolTip(t("extra_args_tooltip"))
        self.extra_args_btn.clicked.connect(self._toggle_extra_args)
        outer.addWidget(self.extra_args_btn)
        self.extra_args_edit = QPlainTextEdit()
        self.extra_args_edit.setPlaceholderText(t("extra_args_placeholder"))
        self.extra_args_edit.setToolTip(t("extra_args_tooltip"))
        self.extra_args_edit.setMaximumHeight(120)
        self.extra_args_edit.setVisible(False)
        outer.addWidget(self.extra_args_edit)
        outer.addStretch()

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

        # QProcess for training. The launchers we spawn (``accelerate launch``,
        # ``python tasks.py …``) fork the real training process, which is what
        # holds VRAM. Run the child in its own session so kill_process_tree
        # can take down the whole subtree on Stop / window close.
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)
        self._stdout_buf = ""
        self._stderr_buf = ""

        self._origin: dict[str, str] = {}
        self._reload()

    # Preset selection is no longer surfaced in the GUI — variants encode the
    # hardware/perf knobs that used to live in presets. The merge still uses
    # 'default' under the hood so the form shows reasonable effective values
    # when a variant file is sparse. All saves write to the variant file.
    _IMPLICIT_PRESET = "default"

    def _current_variant(self) -> str:
        """gui-methods variant for the selected method. Falls back to the
        method name itself when no variants are registered (apex, ip_adapter,
        easycontrol)."""
        v = self.variant_combo.currentText()
        return v or self.method_combo.currentText()

    def _on_method_changed(self):
        self._reload()

    def _refresh_variant_row(self, method: str) -> None:
        variants = list_gui_variants(method)
        current = [
            self.variant_combo.itemText(i) for i in range(self.variant_combo.count())
        ]
        # Rebuilding the combo resets currentText to the first item, which
        # would clobber the user's selection on every _reload. Only rebuild
        # when the variant list actually changed (i.e. method family switched).
        if current != variants:
            self.variant_combo.blockSignals(True)
            self.variant_combo.clear()
            if variants:
                self.variant_combo.addItems(variants)
            self.variant_combo.blockSignals(False)

    def _reload(self):
        method = self.method_combo.currentText()
        if not method:
            return
        self._refresh_variant_row(method)
        variant = self._current_variant()
        merged, origin = merged_gui_variant_preset(variant, self._IMPLICIT_PRESET)
        cfg = {k: v for k, v in merged.items() if k not in _SKIP}

        self._origin = origin

        if hasattr(self, "_explain"):
            self._show_explain_placeholder()

        self._w.clear()
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # Partition fields by Basic vs Advanced first, then by sub-group
        # (Architecture/Training/Performance/Paths/Other). Basic stays
        # always-visible; Advanced is wrapped in a collapsible container.
        basic: dict[str, dict] = {g: {} for g in _GROUPS}
        basic["Other"] = {}
        advanced: dict[str, dict] = {g: {} for g in _GROUPS}
        advanced["Other"] = {}
        for k, v in cfg.items():
            sub = _K2G.get(k, "Other")
            (basic if is_basic_field(k) else advanced)[sub][k] = v

        # "preset" origin shows where the value comes from today, but on Save
        # everything routes to the variant file — no preset/variant split.
        variant_label = f"gui-methods/{variant}.toml"
        origin_style = {
            "base": (
                "color:#888; text-decoration: underline dotted;",
                "from base.toml",
            ),
            "preset": (
                "color:#6aa4d8; text-decoration: underline dotted;",
                f"from presets.toml[{self._IMPLICIT_PRESET}] (saves to {variant_label})",
            ),
            "method": (
                "color:#f0f0f0; text-decoration: underline dotted;",
                f"from {variant_label}",
            ),
        }

        def _build_subgroup_box(gn: str, flds: dict) -> QGroupBox:
            box = QGroupBox(gn)
            form = QFormLayout()
            for k in sorted(flds):
                w = _widget(flds[k], key=k)
                self._w[k] = w
                lbl = ClickableLabel(k)

                help_text = field_help(k)
                style, note = origin_style.get(
                    self._origin.get(k, "base"), origin_style["base"]
                )
                lbl.setStyleSheet(style)
                notes = (note,)

                lbl.clicked.connect(
                    lambda _k=k, _h=help_text, _n=notes: self._show_explain(
                        _k, _h, _n
                    )
                )
                form.addRow(lbl, w)
            box.setLayout(form)
            return box

        # Basic — flat list of sub-group boxes.
        basic_box = QGroupBox(t("basic_section"))
        basic_layout = QVBoxLayout()
        basic_layout.setContentsMargins(8, 12, 8, 8)
        for gn, flds in basic.items():
            if not flds:
                continue
            basic_layout.addWidget(_build_subgroup_box(gn, flds))
        basic_box.setLayout(basic_layout)
        self._fl.addWidget(basic_box)

        # Advanced — collapsible. QGroupBox.setCheckable + a child container
        # whose visibility is bound to the checkbox gives a free toggle UI.
        advanced_box = QGroupBox(t("advanced_section"))
        advanced_box.setCheckable(True)
        advanced_box.setChecked(self._advanced_expanded)
        adv_outer = QVBoxLayout()
        adv_outer.setContentsMargins(8, 12, 8, 8)
        adv_inner = QWidget()
        adv_inner_layout = QVBoxLayout(adv_inner)
        adv_inner_layout.setContentsMargins(0, 0, 0, 0)
        for gn, flds in advanced.items():
            if not flds:
                continue
            adv_inner_layout.addWidget(_build_subgroup_box(gn, flds))
        adv_inner.setVisible(self._advanced_expanded)
        adv_outer.addWidget(adv_inner)
        advanced_box.setLayout(adv_outer)

        def _on_advanced_toggled(checked: bool, _inner=adv_inner):
            self._advanced_expanded = checked
            _inner.setVisible(checked)

        advanced_box.toggled.connect(_on_advanced_toggled)
        self._fl.addWidget(advanced_box)

        self._fl.addStretch()

    # ── Explanation panel ──

    def _show_explain_placeholder(self) -> None:
        # When the current method ships variant presets, the right-panel
        # default is the variant guide + Apply-semantics callout (replacing
        # the old collapsible box on the left-side form).
        method = (
            self.method_combo.currentText() if hasattr(self, "method_combo") else ""
        )
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

    def _save_preset(self, *, silent: bool = False):
        """Write the form (and any extra-args TOML) into the current variant
        file. No preset/variant routing — the variant file is the single
        source of truth for the GUI."""
        variant = self._current_variant()
        path = variant_path(variant)

        method_orig = _load(path)
        base = _load(CONFIGS_DIR / "base.toml")
        # Default-preset overlay is the implicit baseline used by _reload, so
        # we treat it as part of the "effective baseline" when deciding which
        # form values are worth writing to disk (skips redundant entries).
        from gui import _load_all_presets  # local import: only needed for save

        implicit_pset = _load_all_presets().get(self._IMPLICIT_PRESET, {})

        out: dict[str, Any] = dict(method_orig)

        for k, w in self._w.items():
            baseline = method_orig.get(k, implicit_pset.get(k, base.get(k)))
            v = _read(w, baseline)
            if k in method_orig or v != baseline:
                out[k] = v

        # Extra-args textarea: parse as TOML and merge in. Textarea overrides
        # the form for any duplicate key (it's the more explicit signal).
        extra_text = self.extra_args_edit.toPlainText().strip()
        extras: dict[str, Any] = {}
        if extra_text:
            try:
                parsed = toml.loads(extra_text)
            except toml.TomlDecodeError as e:
                QMessageBox.warning(self, t("invalid_toml"), str(e))
                return
            extras = {k: v for k, v in parsed.items() if not isinstance(v, dict)}
            out.update(extras)

        path.parent.mkdir(parents=True, exist_ok=True)
        _save(path, out)

        if extras:
            self.extra_args_edit.clear()
            self._reload()
        if not silent:
            try:
                rel = path.relative_to(CONFIGS_DIR.parent)
            except ValueError:
                rel = path
            QMessageBox.information(self, t("saved"), f"Saved {rel}")

    def _create_variant(self):
        name, ok = QInputDialog.getText(self, t("new_variant"), t("new_variant_prompt"))
        if not ok:
            return
        name = (name or "").strip()
        if not name or not re.match(r"^[A-Za-z0-9_\-]+$", name):
            QMessageBox.warning(self, t("error"), t("new_variant_invalid"))
            return
        full = f"custom/{name}"
        new_path = variant_path(full)
        if new_path.exists():
            QMessageBox.warning(self, t("error"), t("new_variant_exists", name=name))
            return
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text("", encoding="utf-8")
        # Rebuild combo and select the new entry. _reload fires via the
        # currentTextChanged signal once we set the index.
        method = self.method_combo.currentText()
        variants = list_gui_variants(method)
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        self.variant_combo.addItems(variants)
        self.variant_combo.blockSignals(False)
        idx = self.variant_combo.findText(full)
        if idx >= 0:
            self.variant_combo.setCurrentIndex(idx)
        else:
            self._reload()

    def _toggle_extra_args(self):
        self.extra_args_edit.setVisible(self.extra_args_btn.isChecked())

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
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)

    def _start_preprocess(self):
        python = sys.executable
        args = ["tasks.py", "preprocess"]

        # Point tasks.py at the same variant training will use, so any
        # source_image_dir / resized_image_dir / lora_cache_dir override the
        # user wrote into the variant file is honored by preprocess too.
        variant = self._current_variant()
        self._proc.setProcessEnvironment(
            make_subprocess_env(
                METHOD=variant,
                METHODS_SUBDIR="gui-methods",
                PRESET=self._IMPLICIT_PRESET,
            )
        )

        self.log.clear()
        self._reset_progress()
        self._log(
            f"> METHOD={variant} METHODS_SUBDIR=gui-methods python {' '.join(args)}\n"
        )
        self._running_mode = "preprocess"
        self._proc.start(python, args)
        self.preprocess_btn.setText(t("preprocess") + " ...")
        self.preprocess_btn.setStyleSheet(self._preprocess_busy_style)
        self.preprocess_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)

    def _start_training(self):
        if not self._preprocessed:
            QMessageBox.warning(self, t("error"), t("preprocess_required"))
            return

        # Flip button visuals to busy + repaint BEFORE the slow accelerate
        # import and QProcess.start, otherwise Qt's event loop is blocked
        # long enough for Windows to flag the GUI as "Not Responding".
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(self._train_busy_style)
        self.train_btn.setEnabled(False)
        self.preprocess_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)
        self.log.clear()
        self._reset_progress()
        QApplication.processEvents()

        try:
            import accelerate.commands.accelerate_cli  # noqa: F401
        except ImportError:
            QMessageBox.warning(self, t("error"), t("accelerate_not_found"))
            self._restore_train_idle()
            return

        variant = self._current_variant()
        # Route through tasks.py rather than spawning accelerate directly:
        # tasks.py uses python.exe + CREATE_NO_WINDOW for its subprocess calls,
        # which keeps tqdm output flowing back to the GUI. If we spawned
        # accelerate from this process (sys.executable = pythonw.exe under the
        # desktop shortcut), accelerate's workers would inherit pythonw and
        # their stdio would silently drop.
        args = ["tasks.py", "lora-gui", variant]

        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "train"
        self._proc.start(sys.executable, args)
        self.stop_btn.setEnabled(True)

    def _restore_train_idle(self):
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.setEnabled(self._preprocessed)
        self.preprocess_btn.setEnabled(True)
        self.test_btn.setEnabled(self._has_lora_output())
        self.method_combo.setEnabled(True)
        self.variant_combo.setEnabled(True)
        self.new_variant_btn.setEnabled(True)

    def _stop_training(self):
        kill_process_tree(self._proc)

    def cleanup_subprocess(self):
        """Hook for app shutdown — kill any running launcher + descendants."""
        kill_process_tree(self._proc)

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
        self.variant_combo.setEnabled(True)
        self.new_variant_btn.setEnabled(True)

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
