"""ConfigTab — training config editor with field tooltips and LoRA variant guide."""

from __future__ import annotations

import re
import shutil
import sys
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

# Matches tqdm lines like: "Denoising steps:  40%|####      | 12/30 [..]"
_TQDM_RE = re.compile(
    r"^(?P<label>.*?):?\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
)

from gui import (
    CONFIGS_DIR,
    IMAGE_EXTS,
    PRESETS,
    ROOT,
    _GROUPS,
    _K2G,
    _LOCKED_PERFORMANCE,
    _SKIP,
    _load,
    _merged,
    _read,
    _save,
    _widget,
)
from gui.explanations import field_help, lora_guide
from gui.i18n import t


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
        self._vkeys: set[str] = set()
        self._ds_edit: QPlainTextEdit | None = None
        self._preprocessed = (ROOT / "post_image_dataset").exists()
        lay = QVBoxLayout(self)

        # Top bar: preset + save + preprocess + train + stop
        top = QHBoxLayout()
        top.addWidget(QLabel(t("preset")))
        self.combo = QComboBox()
        self.combo.addItems(PRESETS)
        self.combo.currentTextChanged.connect(self._load_preset)
        top.addWidget(self.combo, 1)

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

        self._load_preset(self.combo.currentText())

    def _load_preset(self, name: str):
        f = PRESETS[name]
        var = _load(CONFIGS_DIR / f)
        self._vkeys = set(var) - _SKIP
        cfg = {k: v for k, v in _merged(f).items() if k not in _SKIP}

        if hasattr(self, "_explain"):
            self._show_explain_placeholder()

        self._w.clear()
        self._ds_edit = None
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # LoRA variant guide (collapsible)
        guide_box = QGroupBox(t("lora_variants"))
        guide_box.setCheckable(True)
        guide_box.setChecked(False)
        guide_lay = QVBoxLayout()
        guide_label = QLabel(lora_guide())
        guide_label.setWordWrap(True)
        guide_label.setTextFormat(Qt.RichText)
        guide_label.setStyleSheet("font-size: 12px; padding: 4px;")
        guide_lay.addWidget(guide_label)
        guide_box.setLayout(guide_lay)
        self._fl.addWidget(guide_box)

        # Grouped training config fields
        groups: dict[str, dict] = {g: {} for g in _GROUPS}
        groups["Other"] = {}
        for k, v in cfg.items():
            groups.setdefault(_K2G.get(k, "Other"), {})[k] = v

        lock_perf = name in _LOCKED_PERFORMANCE

        for gn, flds in groups.items():
            if not flds:
                continue
            locked = lock_perf and gn == "Performance"
            box = QGroupBox(gn)
            if locked:
                box.setToolTip(t("locked_by_preset"))
            form = QFormLayout()
            for k in sorted(flds):
                w = _widget(flds[k], key=k)
                if locked:
                    w.setEnabled(False)
                self._w[k] = w
                lbl = ClickableLabel(k)

                help_text = field_help(k)
                notes: list[str] = []
                if locked:
                    lbl.setStyleSheet("color:#666; text-decoration: underline dotted;")
                    notes.append(t("locked_by_preset"))
                elif k not in self._vkeys:
                    lbl.setStyleSheet("color:#888; text-decoration: underline dotted;")
                    notes.append(t("from_base"))
                else:
                    lbl.setStyleSheet("text-decoration: underline dotted;")

                lbl.clicked.connect(
                    lambda _k=k, _h=help_text, _n=tuple(notes): self._show_explain(
                        _k, _h, _n
                    )
                )

                form.addRow(lbl, w)
            box.setLayout(form)
            self._fl.addWidget(box)

        # Dataset config (raw TOML editor)
        ds_path = CONFIGS_DIR / "dataset_config.toml"
        if ds_path.exists():
            box = QGroupBox(t("dataset_config"))
            bl = QVBoxLayout()
            ds_edit = QPlainTextEdit(ds_path.read_text(encoding="utf-8"))
            ds_edit.setStyleSheet("font-family:monospace;")
            ds_edit.setMaximumHeight(180)
            bl.addWidget(ds_edit)
            self._ds_edit = ds_edit
            dsb = QPushButton(t("save_dataset_config"))
            dsb.clicked.connect(self._save_ds)
            bl.addWidget(dsb)
            box.setLayout(bl)
            self._fl.addWidget(box)

        self._fl.addStretch()

    # ── Explanation panel ──

    def _show_explain_placeholder(self) -> None:
        self._explain.setHtml(
            f"<p style='color:#888; font-style:italic;'>{html.escape(t('click_field_for_help'))}</p>"
        )

    def _show_test_output(self) -> None:
        d = ROOT / "test_output"
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

    def _build_save_data(self) -> tuple:
        f = PRESETS[self.combo.currentText()]
        p = CONFIGS_DIR / f
        orig = _load(p)
        bf = orig.get("base_config")
        base = _load(CONFIGS_DIR / bf) if bf else {}
        merged = {**base, **orig}

        out: dict[str, Any] = {}
        if bf:
            out["base_config"] = bf
        ds = orig.get("dataset_config")
        if ds:
            out["dataset_config"] = ds

        for k, w in self._w.items():
            v = _read(w, merged.get(k))
            if k in self._vkeys or (k in base and base[k] != v):
                out[k] = v
        return p, out

    def _save_preset(self):
        p, out = self._build_save_data()
        _save(p, out)
        QMessageBox.information(self, t("saved"), t("saved_file", name=p.name))

    def _save_ds(self):
        if not self._ds_edit:
            return
        p = CONFIGS_DIR / "dataset_config.toml"
        text = self._ds_edit.toPlainText()
        try:
            toml.loads(text)
        except toml.TomlDecodeError as e:
            QMessageBox.warning(self, t("invalid_toml"), str(e))
            return
        p.write_text(text, encoding="utf-8")
        QMessageBox.information(self, t("saved"), t("dataset_saved"))

    # ── Training ──

    def _has_lora_output(self) -> bool:
        out = ROOT / "output"
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
        self.combo.setEnabled(False)

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
        self.combo.setEnabled(False)

    def _start_training(self):
        if not self._preprocessed:
            QMessageBox.warning(self, t("error"), t("preprocess_required"))
            return

        # Auto-save config before training
        p, out = self._build_save_data()
        _save(p, out)

        accelerate = shutil.which("accelerate")
        if not accelerate:
            QMessageBox.warning(self, t("error"), t("accelerate_not_found"))
            return

        f = PRESETS[self.combo.currentText()]
        args = [
            "launch",
            "--num_cpu_threads_per_process",
            "3",
            "--mixed_precision",
            "bf16",
            "train.py",
            "--config_file",
            f"configs/{f}",
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
        self.combo.setEnabled(False)

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
                if tot > 0:
                    self.progress.setMaximum(tot)
                    self.progress.setValue(cur)
                    self.progress.setFormat(f"{label}: {cur}/{tot} (%p%)")
                    if not self.progress.isVisible():
                        self.progress.setVisible(True)
                continue
            if line:
                self._log(line + "\n")
        return tail

    def _reset_progress(self):
        self._stdout_buf = ""
        self._stderr_buf = ""
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.progress.setVisible(False)

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
        self.combo.setEnabled(True)

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
