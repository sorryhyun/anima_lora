"""Qwen-Image-2.1 LoRA window — Preprocess + Train over the daemon.

Two forms rendered from ``library.qwen21.requests`` (one widget per request
field, advanced ones folded), a shared log and progress bar, and daemon command
jobs running the sidecars ``scripts/qwen21/{cache,train}.py``. No torch here.

"Preprocess, then train" chains on the GUI side: the train job is submitted
when the cache job succeeds, so a failed cache pass never trains on a stale
folder. Closing the window mid-chain drops the pending train job (the running
cache job keeps going — the daemon outlives the GUI).
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import daemon as gui_daemon
from gui import i18n as main_i18n
from gui import theme as gui_theme
from gui._job_mixin import DaemonJobMixin
from gui._paths import ROOT, get_setting, set_setting
from gui.progress import TQDM_RE
from gui.qwen21.strings import FIELDS_CN, FIELDS_CN_GENERATE, LANGUAGES, UI
from gui.widgets import action_button, apply_variant
from library.env import resolve_under_home
from library.qwen21.requests import (
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    resolve_model_dir,
)
from library.qwen21.scan import cache_counts, scan

SETTINGS_KEY = "qwen21_gui"
LANGUAGE_KEY = "qwen21_language"
LABEL_CACHE = "qwen21-cache"
LABEL_TRAIN = "qwen21-train"
LABEL_TEST = "qwen21-test"
LABELS = (LABEL_CACHE, LABEL_TRAIN, LABEL_TEST)
TEST_THUMB_PX = 360
# Loading the 17.5 GB text encoder can be silent for a while; the default
# command-job watchdog (120 s) would kill it.
STALL_TIMEOUT = 900.0

_DIR_FIELDS = {"src", "out", "cache", "model_dir", "out_dir"}
_FILE_FIELDS = {"output"}  # save dialog
_OPEN_FIELDS = {"lora", "prompts_file"}  # open dialog
_PROGRESS_RE = re.compile(r"^\s+(progress|step|text|latents|image) (\d+)/(\d+)(.*)")

_lang = "en"


def tr(key: str, **kwargs) -> str:
    s = UI.get(_lang, UI["en"]).get(key) or UI["en"].get(key, key)
    return s.format(**kwargs) if kwargs else s


def field_text(f: dataclasses.Field, cls) -> tuple[str, str]:
    if _lang == "cn":
        if cls is GenerateRequest and f.name in FIELDS_CN_GENERATE:
            return FIELDS_CN_GENERATE[f.name]
        if f.name in FIELDS_CN:
            return FIELDS_CN[f.name]
    return f.name, f.metadata.get("help") or ""


def _base_type(f: dataclasses.Field) -> type:
    annotation = str(f.type)
    for kind in (bool, int, float):
        if kind.__name__ in annotation:
            return kind
    return str


class RequestForm(QWidget):
    """One widget per request field; ``request()`` reads them back typed."""

    def __init__(self, cls, parent=None):
        super().__init__(parent)
        self.cls = cls
        self.widgets: dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        basic = QFormLayout()
        advanced_box = QGroupBox(tr("advanced"))
        advanced_box.setCheckable(True)
        advanced_box.setChecked(False)
        advanced_inner = QWidget()
        advanced = QFormLayout(advanced_inner)
        QVBoxLayout(advanced_box).addWidget(advanced_inner)
        advanced_inner.setVisible(False)
        advanced_box.toggled.connect(advanced_inner.setVisible)

        for f in dataclasses.fields(cls):
            label, help_text = field_text(f, cls)
            widget, row = self._make(f)
            widget.setToolTip(help_text)
            name = QLabel(label)
            name.setToolTip(help_text)
            (advanced if f.metadata.get("advanced") else basic).addRow(name, row)
            self.widgets[f.name] = widget
        layout.addLayout(basic)
        layout.addWidget(advanced_box)

    def _make(self, f: dataclasses.Field) -> tuple[QWidget, QWidget]:
        kind = _base_type(f)
        default = None if f.default is dataclasses.MISSING else f.default
        if kind is bool:
            w = QCheckBox()
            w.setChecked(bool(default))
            return w, w
        if f.metadata.get("choices"):
            w = QComboBox()
            w.addItems(list(f.metadata["choices"]))
            w.setCurrentText(str(default))
            return w, w
        if kind is int and default is not None:
            w = QSpinBox()
            w.setRange(0, 1_000_000)
            w.setValue(default)
            return w, w
        if f.metadata.get("multiline"):
            w = QPlainTextEdit(str(default or ""))
            w.setFixedHeight(96)
            return w, w
        w = QLineEdit("" if default is None else str(default))
        if default is None:
            w.setPlaceholderText(tr("auto"))
        if f.name == "model_dir":
            w.setPlaceholderText(str(resolve_model_dir(None)))
        if f.name not in _DIR_FIELDS | _FILE_FIELDS | _OPEN_FIELDS:
            return w, w
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w, 1)
        browse = QPushButton(tr("browse"))
        browse.clicked.connect(lambda _=False, n=f.name, e=w: self._browse(n, e))
        h.addWidget(browse)
        return w, row

    def _browse(self, name: str, edit: QLineEdit) -> None:
        start = str(resolve_under_home(edit.text() or edit.placeholderText() or "."))
        if name in _FILE_FIELDS:
            path, _ = QFileDialog.getSaveFileName(
                self, name, start, "safetensors (*.safetensors)"
            )
        elif name in _OPEN_FIELDS:
            path, _ = QFileDialog.getOpenFileName(self, name, start)
        else:
            path = QFileDialog.getExistingDirectory(self, name, start)
        if path:
            edit.setText(path)

    def text_of(self, name: str) -> str:
        w = self.widgets[name]
        if isinstance(w, QPlainTextEdit):
            return w.toPlainText().strip()
        return w.text().strip() if isinstance(w, QLineEdit) else ""

    def request(self):
        """The typed request, or ``None`` after warning about a bad number."""
        values = {}
        for f in dataclasses.fields(self.cls):
            w = self.widgets[f.name]
            if isinstance(w, QCheckBox):
                values[f.name] = w.isChecked()
            elif isinstance(w, QComboBox):
                values[f.name] = w.currentText()
            elif isinstance(w, QSpinBox):
                values[f.name] = w.value()
            elif isinstance(w, QPlainTextEdit):
                values[f.name] = w.toPlainText().strip()
            else:
                text = w.text().strip()
                kind = _base_type(f)
                if kind in (int, float):
                    if not text:
                        values[f.name] = None
                        continue
                    try:
                        values[f.name] = kind(text)
                    except ValueError:
                        QMessageBox.warning(
                            self,
                            tr("window_title"),
                            tr("bad_value", field=f.name, value=text),
                        )
                        return None
                else:
                    values[f.name] = text or (None if f.default is None else "")
        return self.cls(**values)

    def set_values(self, values: dict) -> None:
        for name, value in values.items():
            w = self.widgets.get(name)
            if w is None:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QPlainTextEdit):
                w.setPlainText(str(value))
            else:
                w.setText("" if value is None else str(value))

    def changed_values(self) -> dict:
        """Fields that differ from the request defaults — what gets persisted."""
        req = self.request()
        if req is None:
            return {}
        out = {}
        for f in dataclasses.fields(self.cls):
            value = getattr(req, f.name)
            if f.default is dataclasses.MISSING or value != f.default:
                out[f.name] = value
        return out

    def watch(self, callback) -> None:
        for w in self.widgets.values():
            if isinstance(w, (QLineEdit, QPlainTextEdit)):
                w.textChanged.connect(callback)
            elif isinstance(w, QCheckBox):
                w.toggled.connect(callback)


class _LineProgress:
    """Drives the bar from the sidecars' ``  step|text|latents i/n`` lines.

    Per-image cache lines are swallowed (the bar shows them); training step
    lines stay in the log — they carry loss and memory.
    """

    def __init__(self, bar: QProgressBar):
        self.bar = bar

    def feed(self, line: str) -> bool:
        m = _PROGRESS_RE.match(line)
        if not m:
            # tqdm bars (weight loading, the pipeline's denoise loop) drive the
            # bar too, instead of one log line per redraw.
            t = TQDM_RE.search(line)
            if not t:
                return False
            cur, tot = int(t.group("cur")), int(t.group("tot"))
            if tot > 0:
                self.bar.setVisible(True)
                self.bar.setRange(0, tot)
                self.bar.setValue(cur)
                self.bar.setFormat(f"{t.group('label').strip() or 'progress'} %v/%m")
            return True
        kind, cur, tot = m.group(1), int(m.group(2)), int(m.group(3))
        self.bar.setVisible(True)
        self.bar.setRange(0, tot)
        self.bar.setValue(cur)
        if kind == "progress":
            # "  progress i/n epoch e/E loss x eta h:mm:ss" — the tail rides along.
            self.bar.setFormat(f"step %v/%m {m.group(4).strip()}")
            return True
        if kind == "step":
            # The every-20th detail line: keep it in the log, leave the bar to
            # the per-step progress lines.
            return False
        self.bar.setFormat(f"{kind} %v/%m")
        return True

    def starting(self, label: str) -> None:
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.bar.setFormat(label)

    def reset(self) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setVisible(False)


class QwenWindow(DaemonJobMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(900, 820)
        self._init_job_observer()
        self._job_label: str | None = None
        self._pending_train = None

        central = QWidget()
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.model_label = QLabel()
        self.model_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.model_label, 1)
        top.addWidget(QLabel(tr("language")))
        self.lang_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.lang_combo.addItem(name, code)
        self.lang_combo.setCurrentIndex(list(LANGUAGES).index(_lang))
        self.lang_combo.currentIndexChanged.connect(self._change_language)
        top.addWidget(self.lang_combo)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.cache_form = RequestForm(CacheRequest)
        self.train_form = RequestForm(TrainRequest)
        saved = get_setting(SETTINGS_KEY, {}) or {}
        self.cache_form.set_values(saved.get("cache", {}))
        self.train_form.set_values(saved.get("train", {}))

        self.cache_summary = QLabel()
        self.cache_summary.setWordWrap(True)
        self.btn_cache = action_button(tr("run_cache"), on_click=self._run_cache)
        self.tabs.addTab(
            self._page(self.cache_form, self.cache_summary, [self.btn_cache]),
            tr("tab_preprocess"),
        )

        self.train_summary = QLabel()
        self.train_summary.setWordWrap(True)
        self.btn_train = action_button(tr("run_train"), on_click=self._run_train)
        self.btn_chain = action_button(
            tr("run_chain"), variant="secondary", on_click=self._run_chain
        )
        self.test_form = RequestForm(GenerateRequest)
        self.test_form.set_values(saved.get("test", {}))
        self.btn_test = action_button(
            tr("run_test"), variant="info", on_click=self._run_test
        )
        self.test_images = QHBoxLayout()
        self._test_out: Path | None = None
        test_box = QGroupBox(tr("test_group"))
        tb = QVBoxLayout(test_box)
        hint = QLabel(tr("test_hint"))
        hint.setWordWrap(True)
        tb.addWidget(hint)
        tb.addWidget(self.test_form)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_test)
        tb.addLayout(row)
        tb.addLayout(self.test_images)
        self.tabs.addTab(
            self._page(
                self.train_form,
                self.train_summary,
                [self.btn_train, self.btn_chain],
                extra=test_box,
            ),
            tr("tab_train"),
        )

        self.status = QLabel(tr("idle"))
        self.btn_stop = action_button(tr("stop"), variant="danger", on_click=self._stop)
        self.btn_stop.setEnabled(False)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.btn_stop)

        bar = QProgressBar()
        bar.setVisible(False)
        self._progress_tracker = _LineProgress(bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addLayout(status_row)
        bl.addWidget(bar)
        bl.addWidget(self.log, 1)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)
        self.setCentralWidget(central)

        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setInterval(300)
        self._rescan_timer.timeout.connect(self._rescan)
        self.cache_form.watch(self._rescan_timer.start)
        self.train_form.watch(self._rescan_timer.start)
        self.test_form.watch(self._rescan_timer.start)
        self._set_busy(False)
        self._rescan()
        self._show_test(self._latest_test_dir())
        QTimer.singleShot(0, self._reattach)

    def _page(
        self, form: QWidget, summary: QLabel, buttons: list, extra=None
    ) -> QWidget:
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(form)
        v.addWidget(summary)
        row = QHBoxLayout()
        row.addStretch(1)
        for b in buttons:
            row.addWidget(b)
        v.addLayout(row)
        if extra is not None:
            v.addWidget(extra)
        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    # ── state ────────────────────────────────────────────────────────────

    def _model_dir(self) -> Path:
        return resolve_model_dir(
            self.cache_form.text_of("model_dir") or self.train_form.text_of("model_dir")
        )

    def _rescan(self) -> None:
        model = self._model_dir()
        ok = (model / "model_index.json").exists()
        self.model_label.setText(
            tr("model", path=model) if ok else tr("model_missing", path=model)
        )
        src = self.cache_form.text_of("src")
        out = self.cache_form.text_of("out")
        lines = []
        if src:
            s = scan(resolve_under_home(src), resolve_under_home(out) if out else None)
            lines.append(
                tr(
                    "scan",
                    pairs=s.pairs,
                    images=s.images,
                    text=s.text_cached,
                    latents=s.latents_cached,
                )
            )
            if s.stale_text:
                lines.append(tr("scan_stale", stale=s.stale_text))
            if s.duplicates:
                lines.append(tr("scan_dupes", n=s.duplicates))
        else:
            lines.append(tr("scan_no_src"))
        if out:
            c = cache_counts(resolve_under_home(out))
            lines.append(
                tr("cache_counts", samples=c.samples, text=c.text, latents=c.latents)
            )
        self.cache_summary.setText("\n".join(lines))
        cache = self.train_form.text_of("cache")
        n = cache_counts(resolve_under_home(cache)).samples if cache else 0
        self.train_summary.setText(tr("train_ready", n=n) if n else tr("train_empty"))
        self.test_form.widgets["lora"].setPlaceholderText(
            tr("test_lora_placeholder", path=self._train_output())
        )

    def _train_output(self) -> Path:
        return resolve_under_home(
            self.train_form.text_of("output") or TrainRequest().output
        )

    def _persist(self) -> None:
        set_setting(
            SETTINGS_KEY,
            {
                "cache": self.cache_form.changed_values(),
                "train": self.train_form.changed_values(),
                "test": self.test_form.changed_values(),
            },
        )

    def _set_busy(self, busy: bool) -> None:
        for b in (self.btn_cache, self.btn_train, self.btn_chain, self.btn_test):
            b.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        apply_variant(self.btn_stop, "danger" if busy else "busy")

    # ── jobs ─────────────────────────────────────────────────────────────

    def _model_ok(self) -> bool:
        model = self._model_dir()
        if (model / "model_index.json").exists():
            return True
        QMessageBox.warning(self, tr("window_title"), tr("model_missing", path=model))
        return False

    def _src_ok(self, req: CacheRequest) -> bool:
        if req.src and resolve_under_home(req.src).is_dir():
            return True
        QMessageBox.warning(self, tr("window_title"), tr("scan_no_src"))
        return False

    def _submit(self, req, label: str) -> bool:
        argv = [req.SCRIPT, *req.to_argv()]
        job_id = self._submit_job(
            lambda: gui_daemon.ensure_daemon().submit_command(
                label=label, argv=argv, stall_timeout=STALL_TIMEOUT, start=True
            ),
            on_fail=lambda: self._set_busy(False),
        )
        if job_id is None:
            return False
        self._job_label = label
        self.log.appendPlainText(f"$ python {' '.join(argv)}")
        self.status.setText(tr("running", label=label, job=job_id))
        self._set_busy(True)
        self._progress_tracker.starting(label)
        self._watch_job(job_id, replay_log=False)
        return True

    def _run_cache(self) -> None:
        req = self.cache_form.request()
        if req is None or not self._src_ok(req) or not self._model_ok():
            return
        self._persist()
        self._submit(req, LABEL_CACHE)

    def _run_train(self) -> None:
        req = self.train_form.request()
        if req is None or not self._model_ok():
            return
        self._persist()
        self._submit(req, LABEL_TRAIN)

    def _run_chain(self) -> None:
        cache_req = self.cache_form.request()
        if cache_req is None or not self._src_ok(cache_req) or not self._model_ok():
            return
        # Training reads what this preprocessing run writes.
        self.train_form.set_values({"cache": cache_req.out})
        train_req = self.train_form.request()
        if train_req is None:
            return
        self._persist()
        self.log.appendPlainText(
            tr("chain_cache", path=resolve_under_home(cache_req.out))
        )
        if self._submit(cache_req, LABEL_CACHE):
            self._pending_train = train_req

    def _run_test(self) -> None:
        req = self.test_form.request()
        if req is None or not self._model_ok():
            return
        lora = resolve_under_home(req.lora) if req.lora else self._train_output()
        if not lora.is_file():
            QMessageBox.warning(self, tr("window_title"), tr("no_lora", path=lora))
            return
        self._persist()
        # One folder per run, so earlier tests stay on disk.
        out = resolve_under_home(req.out_dir) / time.strftime("%Y%m%d-%H%M%S")
        req = dataclasses.replace(req, lora=str(lora), out_dir=str(out))
        if self._submit(req, LABEL_TEST):
            self._test_out = out

    def _show_test(self, out: Path | None) -> None:
        while self.test_images.count():
            item = self.test_images.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        manifest = out / "manifest.json" if out else None
        if manifest is None or not manifest.exists():
            return
        for entry in json.loads(manifest.read_text(encoding="utf-8"))["images"]:
            path = out / entry["file"]
            m = entry["multiplier"]
            cell = QWidget()
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            image = QLabel()
            image.setPixmap(
                QPixmap(str(path)).scaled(
                    TEST_THUMB_PX,
                    TEST_THUMB_PX,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            image.setToolTip(f"{path}\n{entry['prompt']}")
            caption = QLabel(tr("test_lora", m=f"{m:g}") if m else tr("test_base"))
            caption.setAlignment(Qt.AlignCenter)
            v.addWidget(image)
            v.addWidget(caption)
            self.test_images.addWidget(cell)
        self.test_images.addStretch(1)

    def _latest_test_dir(self) -> Path | None:
        base = resolve_under_home(
            self.test_form.text_of("out_dir") or GenerateRequest().out_dir
        )
        runs = sorted(p for p in base.glob("*/manifest.json")) if base.is_dir() else []
        return runs[-1].parent if runs else None

    def _stop(self) -> None:
        self._pending_train = None
        self._stop_job()

    def _on_job_finished(self, state: str) -> None:
        self._job_timer.stop()
        self._drain_job_stdout()
        label = self._job_label or "job"
        self.log.appendPlainText(tr("finished", label=label, state=state))
        self._job_id = None
        self._progress_tracker.reset()
        self._set_busy(False)
        self.status.setText(tr("finished", label=label, state=state))
        self._rescan()
        if label == LABEL_TEST and gui_daemon.is_success(state):
            self._show_test(self._test_out or self._latest_test_dir())
            self._test_out = None
        pending, self._pending_train = self._pending_train, None
        if pending is not None and gui_daemon.is_success(state):
            self.log.appendPlainText(tr("chain_next"))
            self._submit(pending, LABEL_TRAIN)

    def _reattach(self) -> None:
        job_id = gui_daemon.active_job_id()
        if not job_id:
            return
        label = gui_daemon.read_job_label(job_id)
        if label not in LABELS:
            return
        self._job_label = label
        self.status.setText(tr("running", label=label, job=job_id))
        self.log.appendPlainText(tr("reattached", job=job_id))
        self._set_busy(True)
        self._watch_job(job_id, replay_log=True)

    # ── language ─────────────────────────────────────────────────────────

    def _change_language(self) -> None:
        code = self.lang_combo.currentData()
        if code == _lang:
            return
        self._persist()
        set_setting(LANGUAGE_KEY, code)
        _set_language(code)
        # Rebuild in place: every label is set at construction.
        global _WINDOW
        geometry = self.saveGeometry()
        self._job_timer.stop()
        _WINDOW = QwenWindow()
        _WINDOW._pending_train = self._pending_train
        _WINDOW.restoreGeometry(geometry)
        _WINDOW.show()
        self.close()


def _set_language(code: str) -> None:
    global _lang
    _lang = code if code in LANGUAGES else "en"
    # The job mixin's warnings use the main GUI's table.
    main_i18n.set_language(_lang)


_WINDOW: QwenWindow | None = None


def main() -> None:
    main_lang = main_i18n.load_language()
    _set_language(get_setting(LANGUAGE_KEY) or ("cn" if main_lang == "cn" else "en"))
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    icon = ROOT / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    gui_theme.apply_theme(app)
    global _WINDOW
    _WINDOW = QwenWindow()
    _WINDOW.show()
    QTimer.singleShot(0, gui_daemon.ensure_daemon_quietly)
    sys.exit(app.exec())
