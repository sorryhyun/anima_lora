"""Turbo distillation config tab.

Turbo trains through a bespoke distill loop (``make turbo`` →
``scripts/distill_turbo/``), NOT ``train.py``, and reads a *sectioned* TOML
(``configs/methods/turbo.toml``) — nested ``[network]`` / ``[schedule]`` /
``[optim]`` / … tables — instead of the flat method+preset config the ConfigTab
edits. It also has no dataset of its own: it reuses the ordinary LoRA cache
under ``post_image_dataset/lora``.

The tab is a structured config editor in the method tabs' layout: a
per-section form on top, a log on the bottom, and daemon-backed Preprocess /
Train / Stop in the top bar.

Editing uses ``tomlkit`` so the configs' extensive inline comments survive a
Save round-trip (a plain ``toml.dumps`` would strip them); those comments
double as the form's field tooltips.
"""

from __future__ import annotations

import html

import tomlkit
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from tomlkit.items import Table, Whitespace

from gui import ROOT, CONFIGS_DIR, LazyTabMixin, _read, _widget
from gui import daemon as gui_daemon
from gui.explanations import method_overview
from gui.i18n import t
from gui._job_mixin import DaemonJobMixin
from gui.progress import TqdmProgressTracker, make_progress_bar
from gui.theme import rich_text_pt as _explain_pt, tok
from gui.widgets import (
    DirtyTrackingMixin,
    action_button,
    apply_variant,
    make_field_label,
)


class _DistillConfigTab(DaemonJobMixin, DirtyTrackingMixin, LazyTabMixin, QWidget):
    """Structured editor for a bespoke sectioned distill config.

    Subclasses set the config file, the ``tasks.py`` train target, and a label.
    The form is rebuilt from the TOML on first show; Save writes edits back
    through ``tomlkit`` (comments preserved); Train/Preprocess submit the make
    target to the local daemon and observe it via the per-job ``stdout.log``.
    """

    CONFIG_PATH: str = ""  # repo-relative, e.g. "configs/methods/turbo.toml"
    TRAIN_TASK: str = ""  # tasks.py target, e.g. "turbo"
    METHOD_LABEL: str = ""

    def __init__(self):
        super().__init__()
        self._config_path = ROOT / self.CONFIG_PATH
        self._doc: tomlkit.TOMLDocument | None = None
        # (section_or_None, key, widget, original_python_value)
        self._fields: list[tuple[str | None, str, QWidget, object]] = []
        self._dirty = False
        # Leading file-header comment block → the right panel's default guide.
        self._guide_lines: list[str] = []

        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        # Exposed so MethodsTab can mount its Method picker inline at the front.
        self._top_bar = top
        title = QLabel(self.METHOD_LABEL)
        title.setStyleSheet("font-weight:bold;font-size:14px;")
        top.addWidget(title)
        path_lbl = QLabel(self.CONFIG_PATH)
        path_lbl.setStyleSheet(f"color:{tok('text_dim')};")
        top.addWidget(path_lbl)
        top.addStretch()

        self._save_btn = QPushButton(t("save"))
        self._save_btn.clicked.connect(lambda: self._save())
        top.addWidget(self._save_btn)

        self.train_btn = action_button(
            t("train"), variant="primary", on_click=self._start_train
        )
        top.addWidget(self.train_btn)

        self.stop_btn = action_button(t("stop"), variant="danger", on_click=self._stop)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)
        lay.addLayout(top)

        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._form = QWidget()
        self._fl = QVBoxLayout(self._form)
        self._fl.setContentsMargins(0, 0, 0, 0)
        sc.setWidget(self._form)
        hsplit.addWidget(sc)

        # Right-side explanation panel — mirrors the ConfigTab method tabs.
        # Defaults to the config's file-header block (a method overview); a
        # field label click swaps in that field's full doc comment.
        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            "QTextBrowser { font-size: 120%; padding: 12px; "
            f"background: {tok('panel')}; color: {tok('text')}; }}"
        )
        self._explain.setMinimumWidth(300)
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([640, 360])
        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("log_placeholder"))
        vsplit.addWidget(self.log)
        vsplit.setSizes([500, 200])
        lay.addWidget(vsplit)
        self._show_explain_placeholder()

        # Command job → no progress.jsonl; tqdm parsing drives the bar.
        self._init_job_observer()

    def _lazy_init(self) -> None:
        self._load_and_build()
        self._try_reattach()

    def _load_and_build(self) -> None:
        try:
            text = self._config_path.read_text(encoding="utf-8")
        except OSError as e:
            self._fl.addWidget(QLabel(t("distill_config_missing", err=str(e))))
            return
        self._doc = tomlkit.parse(text)
        self._guide_lines = self._header_comment_lines()

        self._fields = []
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # Top-level scalars first (everything that isn't a [section] table),
        # then one box per table — preserving the file's order via doc.body.
        top_body = [(k, item) for k, item in self._doc.body]
        general = self._build_box(t("distill_general_section"), top_body, None)
        if general is not None:
            self._fl.addWidget(general)
        for key, item in self._doc.body:
            if key is not None and isinstance(item, Table):
                box = self._build_box(
                    str(key).strip(), item.value.body, str(key).strip()
                )
                if box is not None:
                    self._fl.addWidget(box)

        self._fl.addStretch()
        for _s, _k, w, _o in self._fields:
            self._connect_dirty_signal(w)
        self._clear_dirty()
        self._show_explain_placeholder()

    def _build_box(self, title: str, body, section: str | None) -> QGroupBox | None:
        """Build a group box of scalar fields from a tomlkit container body.

        Preceding ``# comment`` lines become the field's tooltip (a blank line
        resets the accumulator so a header block doesn't bleed onto the next
        key). Nested tables are skipped here — the top-level loop renders each
        as its own box. Returns ``None`` when the body has no scalar fields.
        """
        box = QGroupBox(title)
        form = QFormLayout()
        pending: list[str] = []
        added = 0
        for key, item in body:
            if key is None:
                if isinstance(item, Whitespace):
                    pending = []  # blank line separates unrelated comment blocks
                else:  # Comment
                    txt = self._comment_text(item)
                    if txt:
                        pending.append(txt)
                continue
            if isinstance(item, Table):
                pending = []
                continue
            name = str(key).strip()
            orig = item.unwrap() if hasattr(item, "unwrap") else item
            w = _widget(orig, key=name)
            if isinstance(w, QSpinBox):
                # _widget caps ints at 10k; iterations / step counts run higher.
                w.setRange(0, 100_000_000)
                w.setValue(int(orig))
            # Tooltip = preceding comment block + this line's inline trailing comment.
            parts = [" ".join(pending).strip(), self._comment_text(item)]
            tip = "  ".join(p for p in parts if p)
            pending = []
            if tip:
                w.setToolTip(tip)
            lbl = make_field_label(
                name,
                style="text-decoration: underline dotted; color:#ddd;",
                tooltip=tip or None,
                on_click=lambda _n=name, _t=tip: self._show_explain(_n, _t),
            )
            form.addRow(lbl, w)
            self._fields.append((section, name, w, orig))
            added += 1
        box.setLayout(form)
        return box if added else None

    @staticmethod
    def _comment_text(item) -> str:
        """The text of a tomlkit Comment item, stripped of its leading ``#``."""
        if isinstance(item, Whitespace):
            return ""
        trivia = getattr(item, "trivia", None)
        raw = getattr(trivia, "comment", "") if trivia else ""
        return raw.lstrip("#").strip()

    def _header_comment_lines(self) -> list[str]:
        """The config file's leading ``#`` block (up to the first key / blank
        line) — used as the right panel's default method overview. Bare ``#``
        separator lines come through as ``""`` so they split into paragraphs."""
        lines: list[str] = []
        for key, item in self._doc.body:
            if key is not None or isinstance(item, Whitespace):
                break
            lines.append(self._comment_text(item))
        return lines

    def _show_explain_placeholder(self) -> None:
        # Prefer the localized HTML guide when registered; else fall back to the
        # config's English file-header comment block built below.
        guide = method_overview(self._config_path.stem)
        if guide:
            self._explain.setHtml(guide)
            return
        title = html.escape(self.METHOD_LABEL)
        paras: list[str] = []
        cur: list[str] = []
        for ln in self._guide_lines:
            if ln:
                cur.append(ln)
            elif cur:
                paras.append(" ".join(cur))
                cur = []
        if cur:
            paras.append(" ".join(cur))
        if paras:
            body = "".join(
                f"<p style='font-size:{_explain_pt(14)}; line-height:1.6;'>{html.escape(p)}</p>"
                for p in paras
            )
        else:
            body = (
                f"<p style='color:#888; font-style:italic;'>"
                f"{html.escape(t('click_field_for_help'))}</p>"
            )
        self._explain.setHtml(
            f"<h2 style='margin:0 0 10px 0; font-size:{_explain_pt(18)};'>{title}</h2>{body}"
        )

    def _show_explain(self, field: str, help_text: str) -> None:
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:{_explain_pt(18)};'>{html.escape(field)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:{_explain_pt(15)}; line-height:1.6;'>"
                f"{html.escape(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:#888; font-style:italic;'>"
                f"{html.escape(t('no_help_available'))}</p>"
            )
        self._explain.setHtml("".join(parts))

    # Dirty tracking (_connect_dirty_signal / _mark_dirty / _clear_dirty /
    # _update_save_button) is inherited from DirtyTrackingMixin.

    def _save(self, *, silent: bool = False) -> bool:
        if self._doc is None:
            return False
        for section, name, w, orig in self._fields:
            container = self._doc if section is None else self._doc[section]
            container[name] = _read(w, orig)
        try:
            self._config_path.write_text(tomlkit.dumps(self._doc), encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, t("error"), str(e))
            return False
        self._clear_dirty()
        if not silent:
            try:
                rel = self._config_path.relative_to(CONFIGS_DIR.parent)
            except ValueError:
                rel = self._config_path
            QMessageBox.information(self, t("saved"), f"Saved {rel}")
        return True

    def _start_train(self):
        self._launch(self.TRAIN_TASK, ["tasks.py", self.TRAIN_TASK])

    def _launch(self, label: str, argv: list[str]) -> None:
        if self._job_id:
            QMessageBox.information(self, "", t("distill_job_running"))
            return
        # The distill script re-reads the TOML from disk, so flush unsaved edits.
        if self._dirty and not self._save(silent=True):
            return
        self._set_busy()
        self.log.clear()
        self.progress.setVisible(True)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_submitting") + "\n")
        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(label=label, argv=argv),
            on_fail=self._restore_idle,
        )
        if not job_id:
            return
        self._log(t("daemon_queued", job_id=job_id))
        self._attach(job_id, replay=False)

    def _try_reattach(self) -> None:
        """Re-bind to our own train job still running from a previous session.

        Only re-claims a *command* job whose label matches this tab's train
        target — a shared ``preprocess`` job (also submittable from other tabs)
        is intentionally left for whoever owns it."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "command":
            return
        if gui_daemon.read_job_label(job_id) != self.TRAIN_TASK:
            return
        self.log.clear()
        self.progress.setVisible(True)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_reattached", job_id=job_id))
        self._attach(job_id, replay=True)

    def _attach(self, job_id: str, *, replay: bool) -> None:
        self._set_busy()
        self.stop_btn.setEnabled(True)
        self._watch_job(job_id, replay_log=replay)

    def _emit_log_line(self, line: str) -> None:
        # _log uses insertPlainText (no auto-newline), so add one here.
        self._log(line + "\n")

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        self._drain_job_stdout()
        if self._stdout_buf:
            self._log(self._stdout_buf + "\n")
        self._stdout_buf = ""
        job_id = self._job_id
        self._job_id = None
        self._stdout_tailer.reset()
        self.progress.setVisible(False)
        self._log("\n" + gui_daemon.format_finish_banner(job_id, state) + "\n")
        self._restore_idle()

    def _stop(self):
        self._stop_job()

    def cleanup_subprocess(self):
        """App-shutdown hook. Stops observing but leaves the daemon job alive —
        it runs detached so training survives the GUI closing (re-attached on
        next launch)."""
        self._job_timer.stop()

    def _set_busy(self):
        self.train_btn.setText(t("train") + " ...")
        apply_variant(self.train_btn, "busy")
        self.train_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _restore_idle(self):
        self.train_btn.setText(t("train"))
        apply_variant(self.train_btn, "primary")
        self.train_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)


class TurboTrainTab(_DistillConfigTab):
    # Turbo few-step LoRA student via DP-DMD through the bespoke loop `make turbo` (NOT train.py). See docs/methods/turbo.md.
    CONFIG_PATH = "configs/methods/turbo.toml"
    TRAIN_TASK = "turbo"
    METHOD_LABEL = "Turbo"
