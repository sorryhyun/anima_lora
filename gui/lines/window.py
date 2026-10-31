"""Sidecar-line window — Preprocess + Train + Test over the daemon.

One window class for the non-Anima lines (``make gui-qwen``, ``make gui-moss``).
A ``LineSpec`` names the line's torch-free request classes, scan helpers and
strings; the forms are rendered from the requests (one widget per field,
advanced ones folded), with a shared log and progress bar, and every action is
a daemon command job running the line's sidecar (``req.SCRIPT``). No torch here,
and nothing line-specific: ``gui/<line>/app.py`` builds the spec.

Chains run on the GUI side: "Preprocess, then train" submits the train job when
the cache job succeeds, so a failed cache pass never trains on a stale folder,
and a Test with several passes (``LineSpec.test_runs``) submits them one after
another. Closing the window mid-chain drops the pending jobs (the running one
keeps going — the daemon outlives the GUI).

A line can add its own tabs after Train (``LineSpec.extra_pages``, each a
``LinePage``): they submit through the panel (``LinePanel.submit``), so they
share its log, progress bar, Stop button and one-job-at-a-time state.
"""

from __future__ import annotations

import dataclasses
import html
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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

from gui.jobs import daemon as gui_daemon
from gui import i18n as main_i18n
from gui import theme as gui_theme
from gui.jobs.mixin import DaemonJobMixin
from gui.core.paths import ROOT, get_setting, set_setting
from gui.jobs.progress import TQDM_RE
from gui.lines.strings import LANGUAGES, UI
from gui.theme import Pad, rich_text_pt, tok
from gui.widgets import (
    ExplainPanel,
    _no_wheel,
    action_button,
    apply_variant,
    make_field_label,
)
from library.env import resolve_under_home

TEST_THUMB_PX = 360
# Loading a multi-GB text encoder can be silent for a while; the default
# command-job watchdog (120 s) would kill it.
STALL_TIMEOUT = 900.0


@dataclass(frozen=True)
class LineSpec:
    """Everything line-specific the window needs. All callables are torch-free."""

    key: str  # settings keys ``<key>_gui`` / ``<key>_language``, job labels ``<key>-cache|train|test``
    cache_cls: type
    train_cls: type
    test_cls: type
    # (model_dir field text, variant) → the ``model_index.json`` that must exist.
    model_index: Callable[[str | None, str], Path]
    # Folder shown in the help panel's guide for a model_dir field text.
    resolve_model_dir: Callable[[str | None], Path]
    # (src, out) → object with images / pairs / text_cached / latents_cached /
    # stale_text / duplicates; cache folder → object with text / latents / samples.
    scan: Callable[[Path | None, Path | None], Any]
    cache_counts: Callable[[Path], Any]
    # Test request (lora and out_dir already resolved) → the jobs to submit, in order.
    test_runs: Callable[[Any], list]
    # Run folder → [(image, multiplier, prompt)] to show; [] while it holds none.
    test_images: Callable[[Path], list[tuple[Path, float, str]]]
    ui: dict[str, dict[str, str]]  # line keys over gui.lines.strings.UI
    # lang → field name → (label, help); English falls back to the field's metadata.
    fields: dict[str, dict[str, tuple[str, str]]] = dataclasses.field(
        default_factory=dict
    )
    # Test-request fields whose name means something else on another request.
    fields_test: dict[str, dict[str, tuple[str, str]]] = dataclasses.field(
        default_factory=dict
    )
    # Panel → the ``LinePage`` tabs to add after Train.
    extra_pages: Callable[[Any], list] | None = None
    # Called with the Train and Test ``RequestForm``s, before saved values load.
    setup_forms: Callable[[Any, Any], None] | None = None

    @property
    def labels(self) -> tuple[str, str, str]:
        return (f"{self.key}-cache", f"{self.key}-train", f"{self.key}-test")

    @property
    def languages(self) -> list[str]:
        """The codes this line has strings for, in ``LANGUAGES`` order."""
        return [code for code in LANGUAGES if code in self.ui]


_DIR_FIELDS = {"src", "out", "cache", "model_dir", "out_dir"}
_FILE_FIELDS = {"output"}  # save dialog
_OPEN_FIELDS = {"lora", "prompts_file"}  # open dialog
_PROGRESS_RE = re.compile(r"^\s+(progress|step|text|latents|image) (\d+)/(\d+)(.*)")

_lang = "en"
_spec: LineSpec | None = None


@dataclass
class LineJob:
    """One daemon command job: ``req.SCRIPT`` + ``req.to_argv()`` under ``label``,
    with ``env`` added to the job's environment; ``note`` is logged when it starts
    as the next job of a chain."""

    req: Any
    label: str
    env: dict | None = None
    note: str = ""


def tr(key: str, **kwargs) -> str:
    tables = [_spec.ui.get(_lang, {}), _spec.ui.get("en", {})] if _spec else []
    tables += [UI.get(_lang, {}), UI["en"]]
    s = next((t[key] for t in tables if key in t), key)
    return s.format(**kwargs) if kwargs else s


def field_text(f: dataclasses.Field, cls) -> tuple[str, str]:
    if _spec is not None:
        test = _spec.fields_test.get(_lang, {})
        if cls is _spec.test_cls and f.name in test:
            return test[f.name]
        fields = _spec.fields.get(_lang, {})
        if f.name in fields:
            return fields[f.name]
    return f.name, f.metadata.get("help") or ""


def _base_type(f: dataclasses.Field) -> type:
    annotation = str(f.type)
    for kind in (bool, int, float):
        if kind.__name__ in annotation:
            return kind
    return str


class RequestForm(QWidget):
    """One widget per request field; ``request()`` reads them back typed.
    Clicking a field label calls ``help_cb(label, help_text)``."""

    def __init__(self, cls, help_cb: Callable[[str, str], None], parent=None):
        super().__init__(parent)
        self.cls = cls
        self.widgets: dict[str, QWidget] = {}
        self._help: dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        basic = self._basic = QFormLayout()
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
            name = make_field_label(
                label,
                tooltip=help_text,
                on_click=lambda _l=label, _n=f.name: help_cb(_l, self._help[_n]),
            )
            self._help[f.name] = help_text
            (advanced if f.metadata.get("advanced") else basic).addRow(name, row)
            self.widgets[f.name] = widget
        layout.addLayout(basic)
        layout.addWidget(advanced_box)

    def set_help(self, field: str, help_text: str) -> None:
        """Replace basic field ``field``'s tooltip and click-through help."""
        widget = self.widgets[field]
        self._help[field] = help_text
        widget.setToolTip(help_text)
        self._basic.labelForField(widget).setToolTip(help_text)

    def _make(self, f: dataclasses.Field) -> tuple[QWidget, QWidget]:
        kind = _base_type(f)
        default = None if f.default is dataclasses.MISSING else f.default
        if kind is bool:
            w = QCheckBox()
            w.setChecked(bool(default))
            return w, w
        if f.metadata.get("choices"):
            w = _no_wheel(QComboBox())
            w.addItems(list(f.metadata["choices"]))
            w.setCurrentText(str(default))
            return w, w
        if kind is int and default is not None:
            w = _no_wheel(QSpinBox())
            # -1 is a sentinel on some fields ("auto", "steps - 3").
            w.setRange(min(-1, default), 1_000_000)
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
            w.setPlaceholderText(str(_spec.resolve_model_dir(None)))
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
                # An item's data is the field value when its text is not.
                values[f.name] = w.currentData() or w.currentText()
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
                i = w.findData(value)
                w.setCurrentIndex(i) if i >= 0 else w.setCurrentText(str(value))
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
            # the progress lines.
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


class LinePage(QWidget):
    """A tab after Train (``LineSpec.extra_pages``). It submits with
    ``panel.submit`` and gets the panel's job events for its own labels."""

    title = ""
    labels: tuple[str, ...] = ()  # job labels this page submits
    scripts: dict[str, str] = {}  # sidecar script → label, for a job started elsewhere

    def set_busy(self, busy: bool) -> None:
        pass

    def job_finished(self, label: str, ok: bool) -> None:
        """A job of this page's ``labels`` ended and no chained job follows it."""

    def persist(self) -> None:
        pass


class LinePanel(DaemonJobMixin, QWidget):
    """The line's Preprocess / Train / Test forms, log and job observer.

    ``LineWindow`` wraps it standalone; the main GUI's MOSS mode embeds it
    (``build_panel``) and adds its own tabs beside Preprocess / Train
    (``add_tab``). ``on_language`` adds the language picker (standalone only).
    """

    def __init__(
        self, spec: LineSpec, on_language: Callable[[str], None] | None = None
    ):
        super().__init__()
        self.spec = spec
        self.label_cache, self.label_train, self.label_test = spec.labels
        self._init_job_observer()
        self._job_label: str | None = None
        # Submitted one by one as each job succeeds.
        self._pending: list[LineJob] = []

        root = QVBoxLayout(self)

        if on_language is not None:
            top = QHBoxLayout()
            top.addStretch(1)
            top.addWidget(QLabel(tr("language")))
            self.lang_combo = QComboBox()
            for code in spec.languages:
                self.lang_combo.addItem(LANGUAGES[code], code)
            self.lang_combo.setCurrentIndex(spec.languages.index(_lang))
            self.lang_combo.currentIndexChanged.connect(
                lambda: on_language(self.lang_combo.currentData())
            )
            top.addWidget(self.lang_combo)
            root.addLayout(top)

        self.tabs = QTabWidget()
        self.cache_explain = ExplainPanel()
        self.train_explain = ExplainPanel()
        self.cache_form = RequestForm(
            spec.cache_cls, self.cache_explain.show_field_help
        )
        self.train_form = RequestForm(
            spec.train_cls, self.train_explain.show_field_help
        )
        saved = get_setting(self._settings_key, {}) or {}
        self.cache_form.set_values(saved.get("cache", {}))

        self.cache_summary = QLabel()
        self.cache_summary.setWordWrap(True)
        self.btn_cache = action_button(tr("run_cache"), on_click=self._run_cache)
        self.tabs.addTab(
            self._page(
                self.cache_form,
                self.cache_summary,
                [self.btn_cache],
                self.cache_explain,
            ),
            tr("tab_preprocess"),
        )

        self.train_summary = QLabel()
        self.train_summary.setWordWrap(True)
        self.btn_train = action_button(tr("run_train"), on_click=self._run_train)
        self.btn_chain = action_button(
            tr("run_chain"), variant="secondary", on_click=self._run_chain
        )
        self.test_form = RequestForm(spec.test_cls, self.train_explain.show_field_help)
        if spec.setup_forms:
            spec.setup_forms(self.train_form, self.test_form)
        self.train_form.set_values(saved.get("train", {}))
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
        tb.addLayout(self.test_images)
        self.tabs.addTab(
            self._page(
                self.train_form,
                self.train_summary,
                [self.btn_train, self.btn_chain, self.btn_test],
                self.train_explain,
                extra=test_box,
            ),
            tr("tab_train"),
        )

        self._pages: list[LinePage] = spec.extra_pages(self) if spec.extra_pages else []
        for page in self._pages:
            self.tabs.addTab(page, page.title)

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

        self._bottom = bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addLayout(status_row)
        bl.addWidget(bar)
        bl.addWidget(self.log, 1)

        self._line_pages = self.tabs.count()
        self.tabs.currentChanged.connect(
            lambda i: bottom.setVisible(i < self._line_pages)
        )
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

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
        # While idle and shown, pick up a line job that started elsewhere (the
        # CLI's ``make daemon-run``, a queued job reaching the GPU).
        self._claim_timer = QTimer(self)
        self._claim_timer.setInterval(3000)
        self._claim_timer.timeout.connect(self._claim_tick)
        self._claim_timer.start()

    def _page(
        self,
        form: QWidget,
        summary: QLabel,
        buttons: list,
        explain: ExplainPanel,
        extra=None,
    ) -> QWidget:
        """The form scrolls under a header that keeps ``buttons`` top-right."""
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(form)
        v.addWidget(summary)
        if extra is not None:
            v.addWidget(extra)
        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll)
        split.addWidget(explain)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([620, 380])
        header = QHBoxLayout()
        header.setContentsMargins(Pad.SM, Pad.SM, Pad.SM, Pad.XS)
        header.addStretch(1)
        for b in buttons:
            header.addWidget(b)
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(Pad.NONE, Pad.NONE, Pad.NONE, Pad.NONE)
        root.addLayout(header)
        root.addWidget(split, 1)
        return page

    def _guide_html(self, guide_key: str, model: Path, model_ok: bool) -> str:
        """The tab's default help: what the tab does, the model folder (red
        when it is missing), and the click-a-label hint."""
        dim = tok("text_dim")
        status = (
            f"<p style='color:{dim};'>{html.escape(tr('model', path=model))}</p>"
            if model_ok
            else f"<p style='color:{tok('err')};'>"
            + html.escape(tr("model_missing", path=model)).replace("\n", "<br/>")
            + "</p>"
        )
        title = "tab_preprocess" if guide_key == "guide_preprocess" else "tab_train"
        return (
            f"<h2 style='margin:0 0 10px 0; font-size:{rich_text_pt(18)};'>"
            f"{html.escape(tr(title))}</h2>"
            f"<p style='font-size:{rich_text_pt(15)}; line-height:1.6;'>"
            f"{html.escape(tr(guide_key))}</p>"
            f"{status}"
            f"<p style='color:{dim}; font-style:italic;'>"
            f"{html.escape(tr('click_field'))}</p>"
        )

    # ── state ────────────────────────────────────────────────────────────

    @property
    def _settings_key(self) -> str:
        return f"{self.spec.key}_gui"

    def _model_dir_text(self) -> str:
        return self.cache_form.text_of("model_dir") or self.train_form.text_of(
            "model_dir"
        )

    def _rescan(self) -> None:
        model = self.spec.resolve_model_dir(self._model_dir_text())
        ok = self.spec.model_index(self._model_dir_text(), "base").exists()
        # Refresh the guide only; a field's help stays up while it is edited.
        for explain, key in (
            (self.cache_explain, "guide_preprocess"),
            (self.train_explain, "guide_train"),
        ):
            if explain.mode is None:
                explain.show_guide(self._guide_html(key, model, ok))
        scan, cache_counts = self.spec.scan, self.spec.cache_counts
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
        cache_dir = resolve_under_home(cache) if cache else None
        n = cache_counts(cache_dir).samples if cache_dir else 0
        subfolders = self._cache_subfolders(cache_dir) if cache_dir and not n else []
        if n:
            summary = tr("train_ready", n=n)
        elif subfolders:
            summary = tr("train_subfolders", folders=" · ".join(subfolders))
        else:
            summary = tr("train_empty")
        self.train_summary.setText(summary)
        self.test_form.widgets["lora"].setPlaceholderText(
            tr("test_lora_placeholder", path=self._train_output())
        )

    def _cache_subfolders(self, cache: Path) -> list[str]:
        """``name (samples)`` of ``cache``'s subfolders that hold samples: a
        cache is flat, so training does not read them from ``cache``."""
        if not cache.is_dir():
            return []
        counts = (
            (p.name, self.spec.cache_counts(p).samples)
            for p in sorted(cache.iterdir())
            if p.is_dir()
        )
        return [f"{name} ({n})" for name, n in counts if n]

    def _train_output(self) -> Path:
        return resolve_under_home(
            self.train_form.text_of("output") or self.spec.train_cls().output
        )

    def _persist(self) -> None:
        set_setting(
            self._settings_key,
            {
                "cache": self.cache_form.changed_values(),
                "train": self.train_form.changed_values(),
                "test": self.test_form.changed_values(),
            },
        )
        for page in self._pages:
            page.persist()

    def _set_busy(self, busy: bool) -> None:
        for b in (self.btn_cache, self.btn_train, self.btn_chain, self.btn_test):
            b.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        apply_variant(self.btn_stop, "danger" if busy else "busy")
        for page in self._pages:
            page.set_busy(busy)

    # ── jobs ─────────────────────────────────────────────────────────────

    def _model_ok(self, req) -> bool:
        index = self.spec.model_index(req.model_dir, getattr(req, "model", "base"))
        if index.exists():
            return True
        QMessageBox.warning(
            self, tr("window_title"), tr("model_missing", path=index.parent)
        )
        return False

    def _src_ok(self, req) -> bool:
        if req.src and resolve_under_home(req.src).is_dir():
            return True
        QMessageBox.warning(self, tr("window_title"), tr("scan_no_src"))
        return False

    def submit(self, jobs: list[LineJob]) -> bool:
        """Submit ``jobs[0]``; the rest follow one by one while each succeeds."""
        first, *rest = jobs
        if not self._submit(first.req, first.label, first.env):
            return False
        self._pending = rest
        return True

    def _submit(self, req, label: str, env: dict | None = None) -> bool:
        argv = [req.SCRIPT, *req.to_argv()]
        job_id = self._submit_job(
            lambda: gui_daemon.ensure_daemon().submit_command(
                label=label,
                argv=argv,
                extra_env=env,
                stall_timeout=STALL_TIMEOUT,
                start=True,
            ),
            on_fail=lambda: self._set_busy(False),
        )
        if job_id is None:
            self._pending = []
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
        if req is None or not self._src_ok(req) or not self._model_ok(req):
            return
        self._persist()
        self._submit(req, self.label_cache)

    def _run_train(self) -> None:
        req = self.train_form.request()
        if req is None or not self._model_ok(req):
            return
        self._persist()
        self._submit(req, self.label_train)

    def _run_chain(self) -> None:
        cache_req = self.cache_form.request()
        if (
            cache_req is None
            or not self._src_ok(cache_req)
            or not self._model_ok(cache_req)
        ):
            return
        # Training reads what this preprocessing run writes.
        self.train_form.set_values({"cache": cache_req.out})
        train_req = self.train_form.request()
        if train_req is None or not self._model_ok(train_req):
            return
        self._persist()
        self.log.appendPlainText(
            tr("chain_cache", path=resolve_under_home(cache_req.out))
        )
        self.submit(
            [
                LineJob(cache_req, self.label_cache),
                LineJob(train_req, self.label_train, note=tr("chain_next")),
            ]
        )

    def _run_test(self) -> None:
        req = self.test_form.request()
        if req is None or not self._model_ok(req):
            return
        lora = resolve_under_home(req.lora) if req.lora else self._train_output()
        if not lora.is_file():
            QMessageBox.warning(self, tr("window_title"), tr("no_lora", path=lora))
            return
        self._persist()
        # One folder per run, so earlier tests stay on disk.
        out = resolve_under_home(req.out_dir) / time.strftime("%Y%m%d-%H%M%S")
        req = dataclasses.replace(req, lora=str(lora), out_dir=str(out))
        first, *rest = self.spec.test_runs(req)
        if self.submit(
            [
                LineJob(first, self.label_test),
                *(LineJob(r, self.label_test, note=tr("test_next")) for r in rest),
            ]
        ):
            self._test_out = out

    def _show_test(self, out: Path | None) -> None:
        while self.test_images.count():
            item = self.test_images.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for path, m, prompt in self.spec.test_images(out) if out else []:
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
            image.setToolTip(f"{path}\n{prompt}")
            caption = QLabel(tr("test_lora", m=f"{m:g}") if m else tr("test_base"))
            caption.setAlignment(Qt.AlignCenter)
            v.addWidget(image)
            v.addWidget(caption)
            self.test_images.addWidget(cell)
        self.test_images.addStretch(1)

    def _latest_test_dir(self) -> Path | None:
        base = resolve_under_home(
            self.test_form.text_of("out_dir") or self.spec.test_cls().out_dir
        )
        runs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        return next((p for p in reversed(runs) if self.spec.test_images(p)), None)

    def _stop(self) -> None:
        self._pending = []
        self._stop_job()

    def _on_job_finished(self, state: str) -> None:
        self._end_job_watch()
        label = self._job_label or "job"
        self.log.appendPlainText(tr("finished", label=label, state=state))
        self._progress_tracker.reset()
        self._set_busy(False)
        self.status.setText(tr("finished", label=label, state=state))
        self._rescan()
        ok = gui_daemon.is_success(state)
        nxt = self._pending.pop(0) if ok and self._pending else None
        if not ok:
            self._pending = []
        if label == self.label_test and ok and nxt is None:
            self._show_test(self._test_out or self._latest_test_dir())
            self._test_out = None
        if nxt is None:
            for page in self._pages:
                if label in page.labels:
                    page.job_finished(label, ok)
            return
        if nxt.note:
            self.log.appendPlainText(nxt.note)
        self._submit(nxt.req, nxt.label, nxt.env)

    def _job_line_label(self, job_id: str) -> str | None:
        """This line's label for ``job_id``: the label the panel submitted it
        under, or the one its sidecar script maps to (a CLI submit labels it
        ``train`` / ``generate``). ``None`` for another line's job."""
        label = gui_daemon.read_job_label(job_id)
        if label in self.spec.labels or any(label in p.labels for p in self._pages):
            return label
        argv = gui_daemon.read_job_argv(job_id)
        scripts = {
            c.SCRIPT: lb
            for c, lb in zip(
                (self.spec.cache_cls, self.spec.train_cls, self.spec.test_cls),
                self.spec.labels,
            )
        }
        for page in reversed(self._pages):
            scripts = {**page.scripts, **scripts}
        if argv:
            script = Path(argv[0]).as_posix()
            for s, lb in scripts.items():
                if script == s or script.endswith("/" + s):
                    return lb
        return None

    def _claim_tick(self) -> None:
        if self._job_id is None and self.isVisible():
            self._reattach()

    def _reattach(self) -> None:
        job_id = gui_daemon.active_job_id()
        if not job_id:
            return
        label = self._job_line_label(job_id)
        if label is None:
            return
        self._job_label = label
        self.status.setText(tr("running", label=label, job=job_id))
        self.log.appendPlainText(tr("reattached", job=job_id))
        self._set_busy(True)
        self._watch_job(job_id, replay_log=True)

    def add_tab(self, widget: QWidget, label: str) -> None:
        """A tab after Preprocess / Train; the status and log hide on it."""
        self.tabs.addTab(widget, label)

    def cleanup_subprocess(self) -> None:
        """Stop observing on close; the job itself keeps running in the daemon."""
        self._job_timer.stop()
        self._claim_timer.stop()
        for i in range(self._line_pages, self.tabs.count()):
            cleanup = getattr(self.tabs.widget(i), "cleanup_subprocess", None)
            if callable(cleanup):
                cleanup()


class LineWindow(QMainWindow):
    """``make gui-qwen`` / ``make gui-moss``: one ``LinePanel`` in its own window."""

    def __init__(self, spec: LineSpec):
        super().__init__()
        self.spec = spec
        self.setWindowTitle(tr("window_title"))
        self.resize(900, 820)
        self.panel = LinePanel(spec, on_language=self._change_language)
        self.setCentralWidget(self.panel)

    def _change_language(self, code: str) -> None:
        if code == _lang:
            return
        self.panel._persist()
        set_setting(f"{self.spec.key}_language", code)
        _set_language(code)
        # Rebuild in place: every label is set at construction.
        global _WINDOW
        geometry = self.saveGeometry()
        self.panel.cleanup_subprocess()
        _WINDOW = LineWindow(self.spec)
        _WINDOW.panel._pending = self.panel._pending
        _WINDOW.restoreGeometry(geometry)
        _WINDOW.show()
        self.close()


def _set_language(code: str) -> None:
    global _lang
    _lang = code if _spec is not None and code in _spec.languages else "en"
    # The job mixin's warnings use the main GUI's table.
    main_i18n.set_language(_lang)


_WINDOW: LineWindow | None = None


def build_panel(spec: LineSpec, main_lang: str) -> LinePanel:
    """The line's panel for embedding in the main GUI, in the main GUI's
    language when the line has it, else en. Leaves the main GUI's i18n table
    alone."""
    global _spec, _lang
    _spec = spec
    _lang = main_lang if main_lang in spec.languages else "en"
    return LinePanel(spec)


def run(spec: LineSpec) -> None:
    global _spec
    _spec = spec
    main_lang = main_i18n.load_language()
    _set_language(get_setting(f"{spec.key}_language") or main_lang)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    icon = ROOT / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    gui_theme.apply_theme(app)
    global _WINDOW
    _WINDOW = LineWindow(spec)
    _WINDOW.show()
    QTimer.singleShot(0, gui_daemon.ensure_daemon_quietly)
    sys.exit(app.exec())
