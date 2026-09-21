"""ConfigTab — training config editor with field tooltips and LoRA variant guide."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path
from typing import Any

import html

import toml
from PySide6.QtCore import QEvent, QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QToolButton,
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
    _VIRTUAL_KEYS,
    _load,
    _load_base,
    _read,
    _base_folder_repeats,
    _save,
    _widget,
    apply_folder_repeats_choice,
    apply_validation_choice,
    confirm_existing_caches,
    confirm_resumable_checkpoint,
    confirm_train_using_cache,
    default_lora_cache_dir,
    get_setting,
    is_basic_field,
    lint_variant_configs,
    list_gui_variants,
    list_hardware_presets,
    list_methods,
    merged_gui_variant_preset,
    set_setting,
    remove_unknown_dataset_keys,
    variant_path,
)
from gui import daemon as gui_daemon
from gui._job_mixin import DaemonJobMixin
from gui.theme import action_button_qss, rich_text_pt as _explain_pt, tok
from gui.explanations import field_help, field_help_html, method_guide
from gui.i18n import t
from gui.process import kill_process_tree, setup_kill_safe
from gui.widgets import (
    ClickableLabel,  # noqa: F401 — re-exported
    DirtyTrackingMixin,
    ImageViewerDialog,
    SplitButtonStyle,  # noqa: F401 — re-exported; preprocess_tab imports it from here
    action_button,
    apply_variant,
    make_field_label,
)
from gui.progress import (
    TQDM_RE,
    JsonlProgressReader,
    TqdmProgressTracker,
    make_progress_bar,
)

_GUI_PATH_SCOPE_KEY = "path_scope"
# gui_settings.json key holding the Hardware preset picked in the top bar.
_HW_PRESET_SETTING = "hardware_preset"
_FIELD_ORDER = {
    _GUI_PATH_SCOPE_KEY: 10,
    "source_image_dir": 11,
    "resized_image_dir": 12,
    "lora_cache_dir": 13,
    "output_dir": 14,
    "output_name": 15,
    "save_model_as": 16,
    "path_pattern": 20,
    "pretrained_model_name_or_path": 30,
    "qwen3": 31,
    "vae": 32,
    # Pins must stay BELOW the unpinned default (100) or alphabetical sort
    # interleaves the block with the rest of its group box.
    "use_repa": 80,
    "repa_target_dog": 81,
    "train_adaln": 82,
    "adaln_rank": 83,
    "adaln_alpha": 84,
    "sigma_lowres": 85,
    "sigma_lowres_route": 86,
    "sigma_lowres_threshold": 87,
    "sigma_lowres_threshold_max": 88,
    "sigma_lowres_yarnsig": 89,
    "sigma_lowres_span": 90,
    "sigma_lowres_route2": 91,
    "sigma_lowres_threshold2": 92,
    "sigma_lowres_threshold2_max": 93,
    "sigma_lowres_span2": 94,
    "sample_prompts": 10,
    "sample_every_n_epochs": 11,
    "sample_at_first": 12,
    "sample_decode_inline": 13,
}


class ConfigTab(DaemonJobMixin, DirtyTrackingMixin, QWidget):
    def __init__(
        self, methods: list[str] | None = None, tb_panel=None, preprocess_tab=None
    ):
        super().__init__()
        self._tb_panel = tb_panel  # reference only, to sync log dir / current run
        self._preprocess_tab = preprocess_tab
        self._w: dict[str, QWidget] = {}
        self._preprocessed = (ROOT / "post_image_dataset").exists()
        self._advanced_expanded = False
        self._dirty = False
        lay = QVBoxLayout(self)

        # `methods=` lets callers restrict the picker; a single method hides it.
        top = QHBoxLayout()
        # Exposed so MethodsTab can mount its own Method picker at the front of
        # this row when it embeds this tab.
        self._top_bar = top
        method_items = methods if methods is not None else list_methods()
        self._method_label = QLabel("Method")
        top.addWidget(self._method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(method_items)
        self.method_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.method_combo.setMinimumContentsLength(
            max((len(m) for m in method_items), default=10)
        )
        self.method_combo.currentTextChanged.connect(
            lambda _: self._on_method_changed()
        )
        top.addWidget(self.method_combo)
        if len(method_items) <= 1:
            self._method_label.setVisible(False)
            self.method_combo.setVisible(False)

        # Selecting a variant swaps the gui-methods/<variant>.toml file the form
        # is bound to. "+ New" creates a custom variant under gui-methods/custom/.
        self._variant_label = QLabel(t("variant"))
        top.addWidget(self._variant_label)
        self.variant_combo = QComboBox()
        self.variant_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.variant_combo.setMinimumContentsLength(20)
        self.variant_combo.currentTextChanged.connect(lambda _: self._reload())
        top.addWidget(self.variant_combo, 1)
        self.new_variant_btn = QPushButton(t("new_variant"))
        self.new_variant_btn.setToolTip(t("new_variant_tooltip"))
        self.new_variant_btn.clicked.connect(self._create_variant)
        top.addWidget(self.new_variant_btn)

        # Options are presets.toml sections tagged [<name>.gui] group="hardware".
        # The choice is a machine property, persisted in gui_settings.json (not
        # the variant file) and fed into every base->preset->variant merge.
        self._preset_label = QLabel(t("hardware_preset"))
        top.addWidget(self._preset_label)
        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for name, meta in list_hardware_presets():
            self.preset_combo.addItem(str(meta.get("label") or name), name)
            desc = meta.get("description")
            if desc:
                self.preset_combo.setItemData(
                    self.preset_combo.count() - 1, str(desc), Qt.ToolTipRole
                )
        saved_idx = self.preset_combo.findData(
            str(get_setting(_HW_PRESET_SETTING, "default"))
        )
        if saved_idx >= 0:
            self.preset_combo.setCurrentIndex(saved_idx)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        top.addWidget(self.preset_combo)

        self._save_btn = QPushButton(t("save"))
        self._save_btn.clicked.connect(self._save_preset)
        top.addWidget(self._save_btn)

        # Train is a split button: the main action trains now; the dropdown
        # queues it on the daemon instead.
        self.train_btn = QToolButton()
        # SplitButtonStyle owns the arrow geometry; keep a ref (widget doesn't own it).
        self._split_style = SplitButtonStyle()
        self.train_btn.setStyle(self._split_style)
        self.train_btn.setText(t("train"))
        self.train_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.train_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        # Split buttons need a per-widget stylesheet — the global [variant] rule
        # bypasses the proxy style and miscentres the label.
        self.train_btn.setStyleSheet(action_button_qss("primary"))
        self.train_btn.setToolTip(t("train_tooltip"))
        self.train_btn.clicked.connect(self._start_training)
        queue_menu = QMenu(self.train_btn)
        train_preprocess_action = queue_menu.addAction(t("queue_train_preprocess"))
        train_preprocess_action.triggered.connect(
            lambda _checked=False: self._queue_preprocess(train_after=True)
        )
        train_only_action = queue_menu.addAction(t("queue_train_only"))
        train_only_action.triggered.connect(lambda _checked=False: self._queue_train())
        self.train_btn.setMenu(queue_menu)
        # Always enabled — the dropdown can keep queuing variants while a job is
        # attached (the main action is guarded in _start_training).
        self.train_btn.setEnabled(True)
        top.addWidget(self.train_btn)

        self.test_btn = action_button(
            t("test"), variant="secondary", on_click=self._start_test
        )
        self.test_btn.setEnabled(self._has_lora_output())
        top.addWidget(self.test_btn)

        self.stop_btn = action_button(
            t("stop"), variant="danger", on_click=self._stop_training
        )
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)

        # Exposed so subclasses (EasyControlTab) can splice extra buttons in.
        self._top_bar = top
        lay.addLayout(top)

        # Config-health banner: flags dataset-blueprint keys the trainer will
        # reject before the run dies in the daemon. Rebuilt on every _reload.
        self._config_warning_box = QWidget()
        self._config_warning_box.setStyleSheet(
            "background:#5c1a1a;border:1px solid #a33;border-radius:4px;"
        )
        _cwl = QHBoxLayout(self._config_warning_box)
        _cwl.setContentsMargins(10, 8, 10, 8)
        self._config_warning = QLabel()
        self._config_warning.setWordWrap(True)
        self._config_warning.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._config_warning.setStyleSheet("color:#ffd9d9;border:0;font-size:12px;")
        _cwl.addWidget(self._config_warning, 1)
        self._config_warning_btn = QPushButton(t("config_remove_keys_btn"))
        self._config_warning_btn.clicked.connect(self._remove_unknown_keys)
        _cwl.addWidget(self._config_warning_btn, 0, Qt.AlignTop)
        self._config_warning_box.setVisible(False)
        lay.addWidget(self._config_warning_box)

        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        # Tails progress.jsonl and takes over the bar once events appear; tqdm
        # parsing above is the fallback.
        self._jsonl_reader = JsonlProgressReader(
            self.progress, on_run_start=self._on_run_start_event
        )
        self._jsonl_timer = QTimer(self)
        self._jsonl_timer.setInterval(400)
        self._jsonl_timer.timeout.connect(self._jsonl_reader.poll)
        lay.addWidget(self.progress)

        vsplit = QSplitter(Qt.Vertical)

        hsplit = QSplitter(Qt.Horizontal)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._form = QWidget()
        outer = QVBoxLayout(self._form)
        outer.setContentsMargins(0, 0, 0, 0)

        # Cleared on every _reload; extra-args button/textarea sit below it but
        # outside the cleared layout so they persist across reloads.
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
        self.extra_args_edit.textChanged.connect(self._mark_dirty)
        outer.addWidget(self.extra_args_edit)
        outer.addStretch()

        sc.setWidget(self._form)
        hsplit.addWidget(sc)

        self._explain = QTextBrowser()
        # Links are dispatched manually in _on_explain_anchor instead.
        self._explain.setOpenLinks(False)
        self._explain.anchorClicked.connect(self._on_explain_anchor)
        self._explain.setStyleSheet(
            f"QTextBrowser {{ font-size: 120%; padding: 12px; background: {tok('panel')}; color: {tok('text')}; }}"
        )
        self._explain.setMinimumWidth(320)
        # Identity of the gallery render currently showing (None = not a
        # gallery); lets the poll skip setHtml when nothing changed.
        self._gallery_sig: tuple | None = None
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

        self._log_copy_btn = QToolButton(self.log)
        self._log_copy_btn.setText(t("copy_log"))
        self._log_copy_btn.setToolTip(t("copy_log_tooltip"))
        self._log_copy_btn.setCursor(Qt.PointingHandCursor)
        self._log_copy_btn.setStyleSheet(
            f"QToolButton {{ background:{tok('surface')}; color:{tok('text')}; border:1px solid {tok('border')};"
            " border-radius:4px; padding:2px 8px; font-size:11px; }"
            f"QToolButton:hover {{ background:{tok('surface_hover')}; }}"
        )
        self._log_copy_btn.clicked.connect(self._copy_log)
        self.log.installEventFilter(self)
        self._reposition_log_copy_btn()

        vsplit.setSizes([500, 200])
        lay.addWidget(vsplit)

        # Run the child in its own session so kill_process_tree can take down
        # the whole subtree (it forks a real training process) on Stop / close.
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)
        self._stdout_buf = ""
        self._stderr_buf = ""

        # Submitted to the local daemon (not a child of this QProcess) so it
        # survives the GUI closing; observed by polling on-disk job files.
        self._job_id: str | None = None
        # "train" or "preprocess" (auto-chain cache build); drives the
        # chain-to-train decision in _on_job_finished and the busy-button label.
        self._job_kind: str | None = None
        self._stdout_tailer = gui_daemon.FileTailer()
        self._job_timer = QTimer(self)
        self._job_timer.setInterval(400)
        self._job_timer.timeout.connect(self._poll_job)

        self._origin: dict[str, str] = {}
        self._reload()
        self._try_reattach()

    def _current_preset(self) -> str:
        """Hardware preset selected in the top bar ('default' before the combo
        exists — subclasses may call this during ``__init__``)."""
        combo = getattr(self, "preset_combo", None)
        data = combo.currentData() if combo is not None else None
        return str(data) if data else "default"

    def _on_preset_changed(self, *_) -> None:
        set_setting(_HW_PRESET_SETTING, self._current_preset())
        self._reload()

    def _current_variant(self) -> str:
        """gui-methods variant for the selected method. Falls back to the
        method name itself when no variants are registered (easycontrol)."""
        v = self.variant_combo.currentText()
        return v or self.method_combo.currentText()

    def _on_method_changed(self):
        self._reload()

    def _refresh_variant_row(self, method: str) -> None:
        variants = list_gui_variants(method)
        current = [
            self.variant_combo.itemText(i) for i in range(self.variant_combo.count())
        ]
        # Rebuilding resets currentText to the first item, clobbering the
        # selection — only rebuild when the variant list actually changed.
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
        merged, origin = merged_gui_variant_preset(variant, self._current_preset())
        cfg = {k: v for k, v in merged.items() if k not in _SKIP}
        if self._preprocess_tab is not None:
            self._preprocess_tab.set_variant(variant, method=method)

        self._origin = origin

        logging_dir = merged.get("logging_dir")
        if logging_dir and self._tb_panel is not None:
            self._tb_panel.set_log_dir(logging_dir)

        if hasattr(self, "_explain"):
            self._show_explain_placeholder()

        self._w.clear()
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        basic: dict[str, dict] = {g: {} for g in _GROUPS}
        basic["Other"] = {}
        advanced: dict[str, dict] = {g: {} for g in _GROUPS}
        advanced["Other"] = {}
        for k, v in cfg.items():
            sub = _K2G.get(k, "Other")
            (basic if is_basic_field(k) else advanced)[sub][k] = v

        # Origin shows where the value comes from today, but Save always writes
        # to the variant file — no preset/variant split.
        variant_label = f"gui-methods/{variant}.toml"
        origin_style = {
            "base": (
                f"color:{tok('text_dim')}; text-decoration: underline dotted;",
                "from base.toml",
            ),
            "preset": (
                f"color:{tok('link')}; text-decoration: underline dotted;",
                f"from presets.toml[{self._current_preset()}] (saves to {variant_label})",
            ),
            "method": (
                f"color:{tok('text')}; text-decoration: underline dotted;",
                f"from {variant_label}",
            ),
        }

        def _build_subgroup_box(gn: str, flds: dict) -> QGroupBox:
            box = QGroupBox(gn)
            form = QFormLayout()
            for k in sorted(flds, key=lambda key: (_FIELD_ORDER.get(key, 100), key)):
                w = _widget(flds[k], key=k)
                self._w[k] = w
                help_text = field_help(k)
                style, note = origin_style.get(
                    self._origin.get(k, "base"), origin_style["base"]
                )
                notes = (note,)
                lbl = make_field_label(
                    k,
                    style=style,
                    on_click=lambda _k=k, _h=help_text, _n=notes: self._show_explain(
                        _k, _h, _n
                    ),
                )
                form.addRow(lbl, w)
            box.setLayout(form)
            return box

        basic_box = QGroupBox(t("basic_section"))
        basic_layout = QVBoxLayout()
        basic_layout.setContentsMargins(8, 12, 8, 8)
        for gn, flds in basic.items():
            if not flds:
                continue
            basic_layout.addWidget(_build_subgroup_box(gn, flds))
        basic_box.setLayout(basic_layout)
        self._fl.addWidget(basic_box)

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

        # Connect change signals AFTER the values are seeded by _widget, so the
        # initial setValue/addItems calls don't trip the dirty flag.
        for w in self._w.values():
            self._connect_dirty_signal(w)

        self._wire_validation_widgets(int(merged.get("validation_split_num") or 0))

        self._clear_dirty()

        self._refresh_config_warnings(variant)

    def _refresh_config_warnings(self, variant: str) -> None:
        """Show/hide the config-health banner from a scan of the active
        dataset-blueprint sections."""
        try:
            issues = lint_variant_configs(variant)
        except Exception:
            # Linting must never break the form; a config that won't parse is
            # surfaced elsewhere (Save / load chain).
            self._config_warning_box.setVisible(False)
            return
        if not issues:
            self._config_warning_box.setVisible(False)
            return
        lines = "<br>".join(
            f"&nbsp;&nbsp;• <b>{html.escape(i.key)}</b> in "
            f"<code>[{html.escape(i.section)}]</code> "
            f"({html.escape(i.location)})"
            for i in issues
        )
        self._config_warning.setText(f"⚠ {t('config_bad_keys_header')}<br>{lines}")
        self._config_warning_box.setVisible(True)

    def _remove_unknown_keys(self) -> None:
        """Delete the flagged dataset-blueprint keys from their source files
        (they aren't form-editable — see ``remove_unknown_dataset_keys``)."""
        variant = self._current_variant()
        issues = lint_variant_configs(variant)
        if not issues:
            self._refresh_config_warnings(variant)
            return
        listing = "\n".join(f"  • {i.key}  ({i.location})" for i in issues)
        if (
            QMessageBox.question(
                self,
                t("config_remove_keys_btn"),
                t("config_remove_keys_confirm", n=len(issues), keys=listing),
            )
            != QMessageBox.Yes
        ):
            return
        try:
            removed = remove_unknown_dataset_keys(variant)
        except Exception as e:
            QMessageBox.warning(self, t("error"), str(e))
            return
        self._reload()
        if not removed:
            QMessageBox.warning(self, t("error"), t("config_remove_keys_none"))

    def _wire_validation_widgets(self, current_split_num: int) -> None:
        """Keep ``use_valid`` and ``validation_split_num`` in sync: the spinbox
        is the source of truth for the count, the checkbox its on/off mirror.
        Ticking surfaces a positive default up front rather than coercing a 0
        count at save time."""
        from PySide6.QtWidgets import QCheckBox, QSpinBox

        from gui.validation import _DEFAULT_VALIDATION_SPLIT_NUM

        use_valid_w = self._w.get("use_valid")
        vsn_w = self._w.get("validation_split_num")
        if not isinstance(use_valid_w, QCheckBox) or not isinstance(vsn_w, QSpinBox):
            return
        # Count to restore when (re-)ticked: variant/base value if positive,
        # else the historical default.
        default_split = (
            current_split_num
            if current_split_num > 0
            else _DEFAULT_VALIDATION_SPLIT_NUM
        )

        def _on_use_valid(checked: bool) -> None:
            vsn_w.blockSignals(True)
            if checked and vsn_w.value() == 0:
                vsn_w.setValue(default_split)
            elif not checked:
                vsn_w.setValue(0)
            vsn_w.blockSignals(False)

        def _on_split_changed(value: int) -> None:
            want = value > 0
            if use_valid_w.isChecked() != want:
                use_valid_w.blockSignals(True)
                use_valid_w.setChecked(want)
                use_valid_w.blockSignals(False)

        use_valid_w.toggled.connect(_on_use_valid)
        vsn_w.valueChanged.connect(_on_split_changed)

    def _show_explain_placeholder(self) -> None:
        self._explain_mode = None
        method = (
            self.method_combo.currentText() if hasattr(self, "method_combo") else ""
        )
        # Prefer a variant-specific guide (e.g. easycontrol vs colorize, which
        # share the "easycontrol" method); fall back to the method-family guide.
        variant = self._current_variant() if hasattr(self, "variant_combo") else ""
        guide = method_guide(variant) or method_guide(method)
        if guide:
            self._set_explain_html(guide)
            return
        self._set_explain_html(
            f"<p style='color:{tok('text_dim')}; font-style:italic;'>{html.escape(t('click_field_for_help'))}</p>"
        )

    def _set_explain_html(
        self, content: str, *, gallery_sig: tuple | None = None
    ) -> None:
        """Chokepoint for writing the explanation panel; records which gallery
        render (if any) is showing so _render_image_gallery can tell a
        poll-driven refresh from a real content change."""
        self._gallery_sig = gallery_sig
        self._explain.setHtml(content)

    def _on_explain_anchor(self, url: QUrl) -> None:
        """``magnify:`` is the gallery zoom scheme (a file URI with the scheme
        swapped); in-document fragments scroll, everything else opens externally."""
        if url.scheme() == "magnify":
            fileurl = QUrl(url)
            fileurl.setScheme("file")
            ImageViewerDialog(Path(fileurl.toLocalFile()), self.window()).show()
        elif url.isRelative() and url.hasFragment():
            self._explain.scrollToAnchor(url.fragment())
        else:
            QDesktopServices.openUrl(url)

    def _render_image_gallery(self, title_key: str, empty_key: str, imgs: list) -> None:
        """Render the newest few images as an HTML ``<img>`` stack (shared by
        test-output and training-sample views). Polled every 400ms, so an
        unchanged image set skips setHtml (which resets scroll to top); a real
        refresh restores the previous scroll offset after rendering."""

        def _mtime(p: Path):
            try:
                return p.stat().st_mtime_ns
            except OSError:
                return None

        sig = (title_key, tuple((str(p), _mtime(p)) for p in imgs))
        if sig == self._gallery_sig:
            return
        title = html.escape(t(title_key))
        if not imgs:
            self._set_explain_html(
                f"<h2 style='margin:0 0 10px 0; font-size:{_explain_pt(18)};'>{title}</h2>"
                f"<p style='color:{tok('text_dim')}; font-style:italic;'>{html.escape(t(empty_key))}</p>",
                gallery_sig=sig,
            )
            return
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:{_explain_pt(18)};'>{title}</h2>"
        ]
        for p in imgs:
            url = p.resolve().as_uri()
            magnify = "magnify" + url[len("file") :]
            parts.append(
                f"<p style='margin:0 0 10px 0;'>"
                f"<a href='{magnify}'><img src='{url}' style='max-width:100%;'/></a><br/>"
                f"<span style='color:{tok('text_dim')}; font-size:{_explain_pt(11)};'>{html.escape(p.name)}</span> "
                f"<a href='{magnify}' style='text-decoration:none; font-size:{_explain_pt(12)};'>🔍</a>"
                f"</p>"
            )
        sb = self._explain.verticalScrollBar()
        pos = sb.value()
        self._set_explain_html("".join(parts), gallery_sig=sig)
        sb.setValue(min(pos, sb.maximum()))

    @staticmethod
    def _newest_images(d: Path, limit: int = 4, *, since: float | None = None) -> list:
        """Newest images in ``d`` by mtime. ``since`` (epoch seconds) drops any
        written before it — the training-sample gallery passes the job's start
        time so a fresh run never shows the previous run's stale samples."""
        if not d.is_dir():
            return []
        dated: list[tuple[float, Path]] = []
        for p in d.iterdir():
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                mt = p.stat().st_mtime
            except OSError:  # file vanished mid-scan (e.g. a clobbering re-run)
                continue
            if since is not None and mt < since:
                continue
            dated.append((mt, p))
        dated.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in dated[:limit]]

    def _show_test_output(self) -> None:
        self._explain_mode = "test"
        imgs = self._newest_images(ROOT / "output" / "tests")
        self._render_image_gallery("test_output_title", "test_output_empty", imgs)

    def _resolve_sample_dir(self) -> Path:
        """Absolute ``<output_dir>/sample`` for the current variant."""
        try:
            merged, _ = merged_gui_variant_preset(
                self._current_variant(), self._current_preset()
            )
            merged = self._gui_scoped_paths(merged)
            out = merged.get("output_dir") or "output/ckpt"
        except Exception:
            out = "output/ckpt"
        d = Path(out)
        if not d.is_absolute():
            d = ROOT / d
        return d / "sample"

    def _show_sample_output(self, *, announce: bool = False) -> None:
        """Show the newest training sample previews. ``announce=False`` is a
        no-op when no samples exist yet (the live poll never clobbers field
        help with an empty placeholder); ``announce=True`` always renders,
        used when a training job finishes."""
        sample_dir = getattr(self, "_sample_dir", None) or self._resolve_sample_dir()
        imgs = self._newest_images(
            sample_dir, since=getattr(self, "_sample_floor", None)
        )
        if not imgs and not announce:
            return
        self._explain_mode = "sample"
        self._render_image_gallery("sample_output_title", "sample_output_empty", imgs)

    def _show_explain(
        self, field: str, help_text: str | None, notes: tuple[str, ...]
    ) -> None:
        self._explain_mode = "help"
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:{_explain_pt(18)};'>{html.escape(field)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:{_explain_pt(15)}; line-height:1.6;'>{field_help_html(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:{tok('text_dim')}; font-style:italic;'>{html.escape(t('no_help_available'))}</p>"
            )
        for note in notes:
            parts.append(
                f"<p style='color:{tok('text_dim')}; font-style:italic; margin-top:12px;'>• {html.escape(note)}</p>"
            )
        self._set_explain_html("".join(parts))

    def _save_preset(self, *, silent: bool = False):
        """Write the form (and any extra-args TOML) into the current variant
        file — the single source of truth for the GUI."""
        variant = self._current_variant()
        path = variant_path(variant)

        method_orig = _load(path)
        base = _load_base()
        # A value the hardware preset already provides must NOT be baked into
        # the variant file, or it would pin the key against future preset
        # switches (method wins over preset in the merge).
        from gui import _load_all_presets  # local import: only needed for save

        preset_overlay = _load_all_presets().get(self._current_preset(), {})

        out: dict[str, Any] = dict(method_orig)

        for k, w in self._w.items():
            if k in _VIRTUAL_KEYS:
                # Not real flat TOML keys; writeback handled below via per-key apply helpers.
                continue
            if k == _GUI_PATH_SCOPE_KEY:
                scope = str(_read(w, "") or "").strip()
                meta = out.get("variant")
                if not isinstance(meta, dict):
                    meta = {}
                if scope:
                    meta[_GUI_PATH_SCOPE_KEY] = scope
                    out["variant"] = meta
                else:
                    meta.pop(_GUI_PATH_SCOPE_KEY, None)
                    if meta:
                        out["variant"] = meta
                    else:
                        out.pop("variant", None)
                out.pop(_GUI_PATH_SCOPE_KEY, None)
                continue
            baseline = method_orig.get(k, preset_overlay.get(k, base.get(k)))
            v = _read(w, baseline)
            if k in method_orig or v != baseline:
                out[k] = v

        use_valid_w = self._w.get("use_valid")
        if use_valid_w is not None:
            vsn_w = self._w.get("validation_split_num")
            vsn_val: int | None = None
            if vsn_w is not None:
                try:
                    vsn_val = int(_read(vsn_w))
                except (TypeError, ValueError):
                    vsn_val = None
            base_vsn = None
            base_datasets = base.get("datasets")
            if isinstance(base_datasets, list) and base_datasets:
                first = base_datasets[0]
                if isinstance(first, dict):
                    raw = first.get("validation_split_num")
                    if raw is not None:
                        try:
                            base_vsn = int(raw)
                        except (TypeError, ValueError):
                            base_vsn = None
            apply_validation_choice(
                out,
                bool(_read(use_valid_w)),
                split_num=vsn_val,
                base_split_num=base_vsn,
            )

        rbf_w = self._w.get("repeat_by_folder_name")
        if rbf_w is not None:
            apply_folder_repeats_choice(
                out,
                bool(_read(rbf_w)),
                base_enabled=_base_folder_repeats(base),
            )

        # Parse as TOML and merge in (overrides the form on duplicate keys).
        # Bare backslashes (Windows path paste) break TOML escapes — try
        # verbatim, then retry after \->/ before surfacing the error.
        extra_text = self.extra_args_edit.toPlainText().strip()
        extras: dict[str, Any] = {}
        if extra_text:
            try:
                parsed = toml.loads(extra_text)
            except toml.TomlDecodeError as e:
                if "\\" in extra_text:
                    try:
                        parsed = toml.loads(extra_text.replace("\\", "/"))
                    except toml.TomlDecodeError:
                        QMessageBox.warning(self, t("invalid_toml"), str(e))
                        return
                else:
                    QMessageBox.warning(self, t("invalid_toml"), str(e))
                    return
            extras = {k: v for k, v in parsed.items() if not isinstance(v, dict)}
            out.update(extras)

        path.parent.mkdir(parents=True, exist_ok=True)
        _save(path, out)

        if extras:
            self.extra_args_edit.clear()
            self._reload()  # _reload calls _clear_dirty itself
        else:
            self._clear_dirty()
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
        # Seed from the selected variant so the form has all method-specific
        # knobs (network_dim/network_alpha live only in the variant file); an
        # empty seed would fall back to argparse defaults on train.
        seed: dict[str, Any] = {}
        current = self.variant_combo.currentText()
        if current:
            seed_path = variant_path(current)
            if seed_path.is_file():
                seed = _load(seed_path)
                seed.pop("variant", None)
        if seed:
            _save(new_path, seed)
        else:
            new_path.write_text("", encoding="utf-8")
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
        self._progress_tracker.mark_starting(t("starting"))
        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "test"
        self._proc.start(python, args)
        self.test_btn.setText(t("test") + " ...")
        apply_variant(self.test_btn, "busy")
        self.test_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)
        self.preset_combo.setEnabled(False)

    def _resolve_cache_dir(self, variant: str) -> Path:
        """Absolute lora_cache_dir for the given variant."""
        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        merged = self._gui_scoped_paths(merged)
        cache_rel = merged.get("lora_cache_dir")
        if not cache_rel:
            return default_lora_cache_dir()
        cache_dir = Path(cache_rel)
        if not cache_dir.is_absolute():
            cache_dir = ROOT / cache_dir
        return cache_dir

    def _preprocess_env(self, variant: str) -> dict[str, str]:
        env = {
            "METHOD": variant,
            "METHODS_SUBDIR": "gui-methods",
            "PRESET": self._current_preset(),
        }
        if self._preprocess_tab is not None:
            env.update(self._preprocess_tab.preprocess_env())
        return env

    def _chain_train_spec(
        self, variant: str, *, config_snapshot: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "method": variant,
            "preset": self._current_preset(),
            "methods_subdir": "gui-methods",
        }
        if config_snapshot is not None:
            spec["config_snapshot"] = config_snapshot
        return spec

    @staticmethod
    def _normalize_path_scope(scope: Any) -> str | None:
        """A safe relative GUI path scope like ``data_group1``."""
        if not isinstance(scope, str):
            return None
        value = scope.strip().replace("\\", "/").strip("/")
        if not value:
            return None
        if value.endswith("/*"):
            value = value[:-2].strip("/")
        if not value or "|" in value or any(ch in value for ch in "*?[]:"):
            return None
        parts = value.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            return None
        return "/".join(parts)

    @staticmethod
    def _append_scope(path_value: Any, scope: str) -> str:
        base = str(path_value).strip() if path_value is not None else ""
        if not base:
            return scope
        norm = base.replace("\\", "/").rstrip("/")
        if norm == scope or norm.endswith("/" + scope):
            return base
        return f"{norm}/{scope}"

    @staticmethod
    def _gui_scoped_paths(merged: dict[str, Any]) -> dict[str, Any]:
        """Apply GUI-only path_scope to concrete run paths. ``path_pattern``
        keeps its training-filter meaning, evaluated relative to the scoped
        image/cache directories."""
        scope = ConfigTab._normalize_path_scope(merged.get(_GUI_PATH_SCOPE_KEY))
        if not scope:
            return merged
        out = copy.deepcopy(merged)
        defaults = {
            "source_image_dir": "image_dataset",
            "resized_image_dir": "post_image_dataset/resized",
            "lora_cache_dir": "post_image_dataset/lora",
            "output_dir": "output/ckpt",
        }
        for key, default in defaults.items():
            out[key] = ConfigTab._append_scope(out.get(key) or default, scope)
        out.pop(_GUI_PATH_SCOPE_KEY, None)
        out.pop("variant", None)
        return out

    def _queue_config_snapshot(
        self, variant: str, merged: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Full config snapshot captured at GUI submit time."""
        from library.config.io import load_dataset_config_from_base

        snapshot = copy.deepcopy(
            merged
            if merged is not None
            else merged_gui_variant_preset(variant, self._current_preset())[0]
        )
        snapshot = self._gui_scoped_paths(snapshot)
        if self._preprocess_tab is not None:
            snapshot.update(self._preprocess_tab.preprocess_overrides())
        # Preprocess-only knobs leak into `merged`/`preprocess_overrides()` but
        # must NOT ride into the training config: `caption_tag_dropout_rate`
        # collides with a real train arg meaning *live* dataloader tag dropout,
        # and tag dropout is already baked into cached caption variants at
        # preprocess time — running it live too trips the TE-cache assertion.
        from gui.tabs.preprocess.knobs import PREPROCESS_ONLY_KEYS

        for key in (
            "base_config",
            "dataset_config",
            "variant",
            "method",
            "preset",
            "methods_subdir",
            _GUI_PATH_SCOPE_KEY,
            "preprocess_path_pattern",
            "caption_tag_randomize_rate",
            *PREPROCESS_ONLY_KEYS,
            *_VIRTUAL_KEYS,
        ):
            snapshot.pop(key, None)

        dataset_cfg = load_dataset_config_from_base(
            overrides=snapshot,
            method=variant,
            methods_subdir="gui-methods",
        )
        if dataset_cfg:
            snapshot["general"] = dataset_cfg.get("general", {})
            snapshot["datasets"] = dataset_cfg.get("datasets", [])

        def _clean(value):
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items() if v is not None}
            if isinstance(value, list):
                return [_clean(v) for v in value if v is not None]
            if isinstance(value, Path):
                return str(value)
            return value

        return _clean(snapshot)

    def _preprocess_config_snapshot(self, variant: str) -> dict[str, Any]:
        """Config snapshot for the queued preprocess command. Training
        snapshots strip preprocess-only keys (incl. source_image_dir), so use
        the Preprocess tab's own snapshot when available."""
        if self._preprocess_tab is not None:
            return self._preprocess_tab.preprocess_config_snapshot()
        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        return self._gui_scoped_paths(copy.deepcopy(merged))

    def _launch_preprocess(self, variant: str) -> None:
        """Submit the auto-chain preprocess step to the daemon. Only caller is
        the Train auto-chain (PreprocessingTab owns the standalone UI). Runs as
        a daemon "command" job, like training, so it survives the GUI closing
        and shares the daemon's serial queue with the training run that follows.
        On success _on_job_finished chains into training; on failure/cancel it
        stays idle so we never train over a broken cache.

        Geometry knobs (target_res / freefit_max_ratio) can't ride the
        CONFIG_FILE snapshot (it strips preprocess-only keys), so
        _preprocess_env forwards them as env instead, read by tasks.py with
        priority over the snapshot."""
        chain_after = getattr(self, "_chain_train_after_preprocess", False)
        if chain_after:
            self.train_btn.setText(t("train_preprocessing"))
            self.train_btn.setStyleSheet(action_button_qss("busy"))
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)
        self.preset_combo.setEnabled(False)
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        if chain_after:
            self._log(t("train_autopreprocess_log"))
        self._log(t("daemon_submitting") + "\n")
        QApplication.processEvents()

        # Remember the variant to train so a re-attach after a GUI reopen shows
        # the right one even if the combo is touched.
        self._chain_variant = variant
        # Hand the daemon a chain_train spec so IT enqueues the follow-on
        # training the moment preprocess succeeds — completes even if the GUI
        # closes mid-cache, and tags the job as this tab's so it's re-claimed on reopen.
        train_snapshot = self._queue_config_snapshot(variant)
        chain_train = (
            self._chain_train_spec(variant, config_snapshot=train_snapshot)
            if chain_after
            else None
        )

        def _on_fail():
            self._chain_train_after_preprocess = False
            self._restore_idle_ui()

        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(
                label="preprocess",
                argv=["tasks.py", "preprocess"],
                extra_env=self._preprocess_env(variant),
                chain_train=chain_train,
                config_snapshot=self._preprocess_config_snapshot(variant),
                start=True,  # main Train auto-chain: run now
            ),
            on_fail=_on_fail,
        )
        if not job_id:
            return

        self._log(t("daemon_queued", job_id=job_id))
        self._attach_to_job(job_id, replay_log=False, kind="preprocess")

    def _start_training(self):
        # Guard the foreground action while a job is attached (a second attach
        # would hijack the bar); the button stays enabled for queuing.
        if self._job_id is not None:
            QMessageBox.information(self, "", t("train_busy_use_queue"))
            return

        # train.py reads the variant file from disk, so unsaved edits would
        # otherwise be ignored.
        if self._dirty:
            self._save_preset(silent=True)

        # Auto-chain preprocess reads tiers/filter/caption-shuffle knobs from
        # env (_preprocess_env), not directly from widgets.
        if self._preprocess_tab is not None:
            if not self._preprocess_tab.persist_preprocess_inputs():
                return

        variant = self._current_variant()
        cache_dir = self._resolve_cache_dir(variant)

        # Resolve use_repa before the cache-state branch: a config preprocessed
        # before REPA was enabled must re-run preprocess to build the missing PE
        # sidecars rather than launching a silent no-op REPA run.
        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        merged = self._gui_scoped_paths(merged)
        _use_repa = merged.get("use_repa")
        require_pe = _use_repa is True or str(_use_repa).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        # PE sidecar suffix is encoder-specific ({stem}_anima_{encoder}.…), so
        # the probe must look for the encoder REPA will actually read.
        pe_encoder = str(merged.get("repa_encoder") or "pe_spatial").strip() or None

        # Three-way: cache exists -> confirm reuse; missing -> auto-chain
        # Preprocess -> Train. With use_repa on, a cache lacking PE sidecars
        # counts as "missing" (require_pe) so it's rebuilt instead of training
        # REPA against an absent target.
        decision = confirm_train_using_cache(
            self, cache_dir, require_pe=require_pe, pe_encoder=pe_encoder
        )
        if decision is False:
            return

        # Resume prompt up-front for both paths: the daemon owns the
        # preprocess->train chain and can't pause to ask later (GUI may be
        # closed), so the choice is captured and baked in here.
        if not confirm_resumable_checkpoint(self, merged):
            return

        if decision is None:
            # Cache missing -> auto-chain; the daemon enqueues training itself
            # when preprocess succeeds (see _launch_preprocess chain_train).
            self._chain_train_after_preprocess = True
            self._launch_preprocess(variant)
            return

        self._launch_training(variant)

    def _queue_train(self):
        """Enqueue training only (no preprocess) for the current variant.

        Held until the Queue tab's "Start Queue"; assumes the cache is already
        built — use "Train + Preprocess" when it isn't."""
        if self._dirty:
            self._save_preset(silent=True)

        variant = self._current_variant()
        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        merged = self._gui_scoped_paths(merged)
        if not confirm_resumable_checkpoint(self, merged):
            return

        self._log(t("queue_submitting", variant=variant) + "\n")
        QApplication.processEvents()

        job_id = self._submit_job(
            lambda: gui_daemon.submit_training(
                method=variant,
                preset=self._current_preset(),
                methods_subdir="gui-methods",
                config_snapshot=self._queue_config_snapshot(variant, merged),
                start=False,  # queue dropdown: add to queue, don't start now
            )
        )
        if not job_id:
            return

        self._log(t("queue_added_train", variant=variant, job_id=job_id))

    def _queue_preprocess(self, *, train_after: bool):
        """Enqueue preprocess for the current variant, optionally chaining train."""
        if self._dirty:
            self._save_preset(silent=True)
        if self._preprocess_tab is not None:
            if not self._preprocess_tab.persist_preprocess_inputs():
                return

        variant = self._current_variant()
        cache_dir = self._resolve_cache_dir(variant)
        if not confirm_existing_caches(self, cache_dir):
            return

        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        merged = self._gui_scoped_paths(merged)
        if train_after and not confirm_resumable_checkpoint(self, merged):
            return

        submit_key = (
            "queue_submitting_train_preprocess"
            if train_after
            else "queue_submitting_preprocess"
        )
        self._log(t(submit_key, variant=variant) + "\n")
        QApplication.processEvents()

        queued_key = (
            "queue_added_preprocess" if train_after else "queue_added_preprocess_only"
        )
        train_snapshot = self._queue_config_snapshot(variant, merged)
        preprocess_snapshot = self._preprocess_config_snapshot(variant)
        chain_train = (
            self._chain_train_spec(variant, config_snapshot=train_snapshot)
            if train_after
            else None
        )

        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(
                label="preprocess",
                argv=["tasks.py", "preprocess"],
                extra_env=self._preprocess_env(variant),
                chain_train=chain_train,
                config_snapshot=preprocess_snapshot,
                start=False,  # queue dropdown: add to queue, don't start now
            )
        )
        if not job_id:
            return

        self._log(t(queued_key, variant=variant, job_id=job_id))
        # Deliberately don't attach the main tab's bar — the job is paused
        # until the Queue tab's "Start Queue", and attaching now would show a
        # perpetual "starting…" spinner. Watched and started from the Queue tab.

    def _launch_training(self, variant: str) -> None:
        """Submit a training job to the local daemon (runs ``train.py``
        detached, so training survives the GUI closing). The caller owns all
        pre-launch confirmations."""
        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        merged = self._gui_scoped_paths(merged)
        logging_dir = merged.get("logging_dir")
        if logging_dir and self._tb_panel is not None:
            self._tb_panel.set_log_dir(logging_dir)

        # Flip to busy before the submit (cold-start daemon /health wait can
        # take a moment) so the UI feels responsive.
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(action_button_qss("busy"))
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.method_combo.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.new_variant_btn.setEnabled(False)
        self.preset_combo.setEnabled(False)
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_submitting") + "\n")
        QApplication.processEvents()

        job_id = self._submit_job(
            lambda: gui_daemon.submit_training(
                method=variant,
                preset=self._current_preset(),
                methods_subdir="gui-methods",
                config_snapshot=self._queue_config_snapshot(variant, merged),
                start=True,  # main Train button: run now
            ),
            on_fail=self._restore_idle_ui,
        )
        if not job_id:
            return

        self._log(t("daemon_queued", job_id=job_id))
        self._attach_to_job(job_id, replay_log=False)

    def _try_reattach(self) -> None:
        """Bind to a daemon job still running when this tab is constructed —
        makes "close GUI mid-train -> reopen -> re-attach" work, and surfaces a
        job the CLI / ComfyUI node submitted."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable → nothing to attach
            return
        if not job_id:
            return
        kind = gui_daemon.read_job_kind(job_id)
        if kind != "train":
            # A command job is ours only if it's the auto-chain preprocess this
            # tab submitted (tagged ANIMA_CHAIN_TRAIN); a standalone
            # preprocess/mask belongs to the PreprocessingTab.
            chain_variant = gui_daemon.read_job_chain_variant(job_id)
            if not chain_variant:
                return
            self._chain_train_after_preprocess = True
            self._chain_variant = chain_variant
            reattach_kind = "preprocess"
        else:
            reattach_kind = "train"
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_reattached", job_id=job_id))
        self._attach_to_job(job_id, replay_log=True, kind=reattach_kind)

    def _attach_to_job(
        self, job_id: str, *, replay_log: bool, kind: str = "train"
    ) -> None:
        """Point the bar + log at a daemon job's on-disk files and start
        polling. ``replay_log`` reads ``stdout.log`` from the top (re-attach
        after a GUI restart); otherwise a fresh launch shows only new lines.
        ``kind`` "preprocess" jobs emit no progress.jsonl, so the bar falls
        back to tqdm parsing in _drain_job_stdout."""
        self._job_id = job_id
        self._job_kind = kind
        self._running_mode = kind
        # Cache the sample dir once so the 400ms poll doesn't re-merge the
        # config chain every tick.
        self._sample_dir = self._resolve_sample_dir() if kind == "train" else None
        # Floor the gallery at this job's start so a fresh run never surfaces
        # the previous run's stale previews (never cleared on disk).
        self._sample_floor = (
            gui_daemon.read_job_started_at(job_id) if kind == "train" else None
        )
        self._stdout_buf = ""
        self._jsonl_reader.watch(gui_daemon.progress_path(job_id))
        self._stdout_tailer.watch(gui_daemon.stdout_path(job_id))
        if not replay_log:
            self._stdout_tailer.read_new()  # discard backlog
        chain_after = getattr(self, "_chain_train_after_preprocess", False)
        if kind == "preprocess":
            self.train_btn.setText(
                t("train_preprocessing") if chain_after else t("train")
            )
        else:
            self.train_btn.setText(t("train_running_daemon"))
        self.train_btn.setStyleSheet(action_button_qss("busy"))
        # Keep pickers live so the user can Queue another variant behind the
        # running one; the running job uses an immutable snapshot so editing
        # the form afterward can't disturb it.
        self.train_btn.setEnabled(True)
        self.test_btn.setEnabled(False)
        self.method_combo.setEnabled(True)
        self.variant_combo.setEnabled(True)
        self.new_variant_btn.setEnabled(True)
        self.preset_combo.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self._job_timer.start()

    def _drain_job_stdout(self) -> None:
        """Append new stdout.log lines to the log widget. When progress.jsonl
        drives the bar (training), tqdm lines are swallowed; otherwise tqdm
        drives the bar instead — mirrors the QProcess _handle_stream path."""
        chunk = self._stdout_tailer.read_new()
        if not chunk:
            return
        parts = re.split(r"[\r\n]", self._stdout_buf + chunk)
        self._stdout_buf = parts[-1]  # incomplete trailing fragment
        for line in parts[:-1]:
            if self._jsonl_reader.active:
                if TQDM_RE.search(line):
                    continue
            elif self._progress_tracker.feed(line):
                continue
            if line:
                self._log(line + "\n")

    def _poll_job(self) -> None:
        if not self._job_id:
            return
        self._jsonl_reader.poll()
        self._drain_job_stdout()
        # Refresh the gallery as samples land, but only while the panel isn't
        # pinned to field help.
        if self._job_kind == "train" and getattr(self, "_explain_mode", None) in (
            None,
            "sample",
        ):
            self._show_sample_output()
        state = gui_daemon.read_job_state(self._job_id)
        if gui_daemon.is_terminal(state):
            self._on_job_finished(state)

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        self._jsonl_reader.poll()
        self._drain_job_stdout()
        if self._stdout_buf:
            self._log(self._stdout_buf + "\n")
        self._stdout_buf = ""
        job_id = self._job_id
        kind = self._job_kind
        self._job_id = None
        self._job_kind = None
        self._jsonl_timer.stop()
        self._jsonl_reader.reset()
        self._stdout_tailer.reset()
        self.progress.setVisible(False)
        self._log("\n" + gui_daemon.format_finish_banner(job_id, state) + "\n")

        # Auto-chain Train after a successful preprocess: the daemon already
        # enqueued the training job (chained_job_id), so just hop the UI onto
        # it; on failure/Stop there's no chained job, so stay idle.
        if kind == "preprocess":
            chain = getattr(self, "_chain_train_after_preprocess", False)
            self._chain_train_after_preprocess = False
            self._chain_variant = None
            if state == "done":
                self._preprocessed = True
            if chain and state == "done":
                chained = gui_daemon.read_job_chained_id(job_id)
                if chained:
                    # Defer so this poll callback finishes before attaching.
                    QTimer.singleShot(
                        0,
                        lambda jid=chained: self._reattach_chained_training(jid),
                    )
                    return  # stay busy — training is starting
        if kind == "train" and state == "done":
            self._show_sample_output(announce=True)
        # Follow the serial queue: if another train job we own is
        # running/queued behind this one, hop the bar onto it instead of
        # snapping to idle.
        if self._follow_queue_successor(finished_id=job_id):
            return
        self._restore_idle_ui()

    def _follow_queue_successor(self, *, finished_id: str) -> bool:
        """Re-attach the UI to the queue's next train job, if any. The daemon
        promotes the next queued job to running the instant this one ends, but
        the GUI only consults ``active_job_id`` once (at construction) — so
        without this the bar vanishes even though a queued run is now live.
        Returns True when it hopped onto a successor."""
        try:
            jobs = gui_daemon.list_jobs_passive()
        except Exception:  # noqa: BLE001 — daemon down/unreachable → go idle
            return False
        successor: tuple[tuple[int, float], str] | None = None
        for job in jobs:
            jid = job.get("id")
            if not jid or jid == finished_id:
                continue
            if job.get("state") not in ("running", "queued"):
                continue
            if (job.get("kind") or "train") != "train":
                continue
            # Running beats queued; within a tier earliest submit wins (FIFO).
            rank = (
                0 if job.get("state") == "running" else 1,
                float(job.get("submitted_at") or 0.0),
            )
            if successor is None or rank < successor[0]:
                successor = (rank, jid)
        if successor is None:
            return False
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_next_queued", job_id=successor[1]))
        self._attach_to_job(successor[1], replay_log=True, kind="train")
        return True

    def _reattach_chained_training(self, job_id: str) -> None:
        """Bind the UI to a training job the daemon auto-chained off a
        preprocess. ``replay_log=False`` since the training stdout is fresh."""
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._attach_to_job(job_id, replay_log=False, kind="train")

    def _restore_idle_ui(self):
        """Return every control to its idle state."""
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(action_button_qss("primary"))
        self.train_btn.setEnabled(True)
        self.test_btn.setText(t("test"))
        apply_variant(self.test_btn, "secondary")
        self.test_btn.setEnabled(self._has_lora_output())
        self.stop_btn.setEnabled(False)
        self.method_combo.setEnabled(True)
        self.variant_combo.setEnabled(True)
        self.new_variant_btn.setEnabled(True)
        self.preset_combo.setEnabled(True)
        if self._tb_panel is not None:
            self._tb_panel.clear_current_run()

    def _stop_training(self):
        # A daemon job is aborted via the daemon; a QProcess test run is killed directly.
        if self._job_id:
            try:
                gui_daemon.stop_job(self._job_id)
            except Exception as e:  # noqa: BLE001
                self._log(f"stop failed: {e}\n")
            return
        kill_process_tree(self._proc)

    def _on_run_start_event(self, ev: dict) -> None:
        """Called by JsonlProgressReader on a run_start event; highlights the
        current run's ``log_dir`` in the TensorBoard panel."""
        log_dir = ev.get("log_dir")
        if log_dir and self._tb_panel is not None:
            self._tb_panel.set_current_run(log_dir)

    def cleanup_subprocess(self):
        """App-shutdown hook. Kills a running test subprocess, but deliberately
        leaves a daemon training job alive — it runs detached and survives."""
        self._job_timer.stop()
        kill_process_tree(self._proc)

    def _read_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode(errors="replace")
        self._stdout_buf = self._handle_stream(self._stdout_buf + data)

    def _read_stderr(self):
        data = self._proc.readAllStandardError().data().decode(errors="replace")
        self._stderr_buf = self._handle_stream(self._stderr_buf + data)

    def _handle_stream(self, buf: str) -> str:
        parts = re.split(r"[\r\n]", buf)
        tail = parts[-1]  # incomplete trailing fragment — keep buffered
        for line in parts[:-1]:
            if self._jsonl_reader.active:
                # JSONL drives the bar; swallow tqdm lines so they don't move it.
                if TQDM_RE.search(line):
                    continue
            elif self._progress_tracker.feed(line):
                continue
            if line:
                self._log(line + "\n")
        return tail

    def _reset_progress(self):
        self._stdout_buf = ""
        self._stderr_buf = ""
        self._progress_tracker.reset()
        self._jsonl_timer.stop()
        self._jsonl_reader.reset()

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus):
        # QProcess backs only the Test button (training/preprocess are daemon jobs).
        for buf_name in ("_stdout_buf", "_stderr_buf"):
            leftover = getattr(self, buf_name, "")
            if leftover and not TQDM_RE.search(leftover):
                self._log(leftover + "\n")
            setattr(self, buf_name, "")
        self._jsonl_timer.stop()
        self._jsonl_reader.poll()
        self._jsonl_reader.reset()
        self.progress.setVisible(False)
        self._log(f"\n{t('finished', code=exit_code)}\n")
        if getattr(self, "_running_mode", "test") == "test" and exit_code == 0:
            self._show_test_output()
        self._restore_idle_ui()

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def eventFilter(self, obj, event):
        if obj is self.log and event.type() == QEvent.Resize:
            self._reposition_log_copy_btn()
        return super().eventFilter(obj, event)

    def _reposition_log_copy_btn(self):
        """Keep the Copy button pinned to the top-right of the log viewport."""
        btn = getattr(self, "_log_copy_btn", None)
        if btn is None:
            return
        btn.adjustSize()
        margin = 6
        # Account for a visible vertical scrollbar so the button doesn't overlap it.
        sb = self.log.verticalScrollBar()
        sb_w = sb.width() if sb is not None and sb.isVisible() else 0
        x = self.log.width() - btn.width() - sb_w - margin
        btn.move(max(margin, x), margin)
        btn.raise_()

    def _copy_log(self):
        QApplication.clipboard().setText(self.log.toPlainText())
        btn = self._log_copy_btn
        btn.setText(t("copy_log_done"))
        QTimer.singleShot(1200, lambda: btn.setText(t("copy_log")))
