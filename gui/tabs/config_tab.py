"""ConfigTab — training config editor with field tooltips and LoRA variant guide."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import html

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QTextCursor
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import (
    CONFIGS_DIR,
    ROOT,
    _SKIP,
    _load,
    _load_base,
    _read,
    _save,
    _widget,
    confirm_existing_caches,
    confirm_resumable_checkpoint,
    confirm_train_using_cache,
    get_setting,
    lint_variant_configs,
    list_gui_variants,
    list_hardware_presets,
    list_methods,
    merged_gui_variant_preset,
    set_setting,
    remove_unknown_dataset_keys,
    variant_path,
)
from gui.core import submit, variant_form
from gui.jobs import daemon as gui_daemon
from gui.jobs.mixin import DaemonJobMixin
from gui.theme import action_button_qss, tok
from gui.explanations import field_help, method_guide
from gui.i18n import t
from gui.jobs.process import StreamingProcess
from gui.widgets import (
    DirtyTrackingMixin,
    ExplainPanel,
    SplitButtonStyle,
    action_button,
    apply_variant,
    make_field_label,
    newest_images,
)
from gui.jobs.progress import (
    TQDM_RE,
    JsonlProgressReader,
    TqdmProgressTracker,
    make_progress_bar,
)

# gui_settings.json key holding the Hardware preset picked in the top bar.
_HW_PRESET_SETTING = "hardware_preset"


class ConfigTab(DaemonJobMixin, DirtyTrackingMixin, QWidget):
    """Variant form + Train / Test / Queue for the ``train.py --method`` methods.

    Override points (EasyControlTab uses these): ``_reload`` / ``_save_preset``
    (its descriptor form), ``_refresh_variant_row``, ``_set_pickers_enabled``
    (extra controls that lock with the pickers), ``_attach_to_job`` /
    ``_try_reattach`` (+ ``_reattach``), and ``_submit_training`` / ``_proc``
    as building blocks for its own launches.
    """

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

        self._explain = ExplainPanel()
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

        # Test (and EasyControl's preprocess) run as a direct child; Stop / close
        # kill its whole subtree (it forks a real inference process).
        self._proc = StreamingProcess(self)
        self._proc.line.connect(lambda line, _err: self._route_line(line))
        self._proc.finished.connect(self._on_finished)

        # Training / auto-chain preprocess are daemon jobs (not children of this
        # QProcess), so they survive the GUI closing; observed via on-disk files.
        self._init_job_observer()
        # "train" or "preprocess" (auto-chain cache build); drives the
        # chain-to-train decision in _on_job_finished and the busy-button label.
        self._job_kind: str | None = None

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

        self._clear_form()
        basic, advanced = variant_form.group_fields(cfg)
        styles = self._origin_styles(variant)
        self._fl.addWidget(self._basic_section(basic, styles))
        self._fl.addWidget(self._advanced_section(advanced, styles))
        self._fl.addStretch()

        # Connect change signals AFTER the values are seeded by _widget, so the
        # initial setValue/addItems calls don't trip the dirty flag.
        for w in self._w.values():
            self._connect_dirty_signal(w)

        self._wire_validation_widgets(int(merged.get("validation_split_num") or 0))

        self._clear_dirty()

        self._refresh_config_warnings(variant)

    def _clear_form(self) -> None:
        self._w.clear()
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def _origin_styles(self, variant: str) -> dict[str, tuple[str, str]]:
        """Origin → (label style, note). The origin says where a value comes
        from today; Save always writes to the variant file."""
        variant_label = f"gui-methods/{variant}.toml"
        return {
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

    def _field_group_box(
        self, title: str, fields: dict, styles: dict[str, tuple[str, str]]
    ) -> QGroupBox:
        """One group of fields; registers each widget in ``self._w``."""
        box = QGroupBox(title)
        form = QFormLayout()
        for k in sorted(fields, key=variant_form.field_sort_key):
            w = _widget(fields[k], key=k)
            self._w[k] = w
            help_text = field_help(k)
            style, note = styles.get(self._origin.get(k, "base"), styles["base"])
            lbl = make_field_label(
                k,
                style=style,
                on_click=lambda _k=k, _h=help_text, _n=(note,): self._show_explain(
                    _k, _h, _n
                ),
            )
            form.addRow(lbl, w)
        box.setLayout(form)
        return box

    def _basic_section(
        self, groups: dict[str, dict], styles: dict[str, tuple[str, str]]
    ) -> QGroupBox:
        box = QGroupBox(t("basic_section"))
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 12, 8, 8)
        for title, fields in groups.items():
            if fields:
                lay.addWidget(self._field_group_box(title, fields, styles))
        box.setLayout(lay)
        return box

    def _advanced_section(
        self, groups: dict[str, dict], styles: dict[str, tuple[str, str]]
    ) -> QGroupBox:
        """The checkable "Advanced" fold; its open state survives reloads."""
        box = QGroupBox(t("advanced_section"))
        box.setCheckable(True)
        box.setChecked(self._advanced_expanded)
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 12, 8, 8)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        for title, fields in groups.items():
            if fields:
                inner_lay.addWidget(self._field_group_box(title, fields, styles))
        inner.setVisible(self._advanced_expanded)
        outer.addWidget(inner)
        box.setLayout(outer)

        def _on_toggled(checked: bool) -> None:
            self._advanced_expanded = checked
            inner.setVisible(checked)

        box.toggled.connect(_on_toggled)
        return box

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

        from gui.core.validation import _DEFAULT_VALIDATION_SPLIT_NUM

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
        method = (
            self.method_combo.currentText() if hasattr(self, "method_combo") else ""
        )
        # Prefer a variant-specific guide (e.g. easycontrol vs colorize, which
        # share the "easycontrol" method); fall back to the method-family guide.
        variant = self._current_variant() if hasattr(self, "variant_combo") else ""
        self._explain.show_guide(method_guide(variant) or method_guide(method))

    def _show_explain(
        self, field: str, help_text: str | None, notes: tuple[str, ...]
    ) -> None:
        self._explain.show_field_help(field, help_text, notes)

    def _show_test_output(self) -> None:
        imgs = newest_images(ROOT / "output" / "tests")
        self._explain.show_gallery(
            "test", "test_output_title", "test_output_empty", imgs
        )

    def _resolve_sample_dir(self) -> Path:
        """Absolute ``<output_dir>/sample`` for the current variant."""
        try:
            out = (
                self._scoped_merged(self._current_variant()).get("output_dir")
                or "output/ckpt"
            )
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
        imgs = newest_images(sample_dir, since=getattr(self, "_sample_floor", None))
        if not imgs and not announce:
            return
        self._explain.show_gallery(
            "sample", "sample_output_title", "sample_output_empty", imgs
        )

    def _save_preset(self, *, silent: bool = False):
        """Write the form (and any extra-args TOML) into the current variant
        file — the single source of truth for the GUI."""
        try:
            extras = variant_form.parse_extra_args(self.extra_args_edit.toPlainText())
        except variant_form.ExtraArgsError as e:
            QMessageBox.warning(self, t("invalid_toml"), str(e))
            return
        from gui import _load_all_presets  # local import: only needed for save

        path = variant_path(self._current_variant())
        out = variant_form.variant_from_form(
            _load(path),
            self._w,
            lambda k, baseline: _read(self._w[k], baseline),
            base=_load_base(),
            preset_overlay=_load_all_presets().get(self._current_preset(), {}),
            extras=extras,
        )
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

        args = ["tasks.py", "test"]

        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "test"
        self._proc.start(args)
        self.test_btn.setText(t("test") + " ...")
        apply_variant(self.test_btn, "busy")
        self.test_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_pickers_enabled(False)

    # -- submit plan (pure logic lives in gui.core.submit) --------------------

    def _scoped_merged(self, variant: str) -> dict[str, Any]:
        """The variant's merged chain under the current preset, ``path_scope`` applied."""
        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        return submit.scoped_paths(merged)

    def _resolve_cache_dir(self, variant: str) -> Path:
        return submit.cache_dir(self._scoped_merged(variant))

    def _preprocess_env(self, variant: str) -> dict[str, str]:
        tab_env = (
            self._preprocess_tab.preprocess_env()
            if self._preprocess_tab is not None
            else None
        )
        return submit.preprocess_env(variant, self._current_preset(), tab_env)

    def _chain_train_spec(
        self, variant: str, *, config_snapshot: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return submit.chain_train_spec(
            variant, self._current_preset(), config_snapshot=config_snapshot
        )

    def _queue_config_snapshot(
        self, variant: str, merged: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Full training config snapshot captured at GUI submit time."""
        if merged is None:
            merged = merged_gui_variant_preset(variant, self._current_preset())[0]
        overrides = (
            self._preprocess_tab.preprocess_overrides()
            if self._preprocess_tab is not None
            else {}
        )
        return submit.training_snapshot(variant, merged, overrides)

    def _preprocess_config_snapshot(self, variant: str) -> dict[str, Any]:
        """Config snapshot for the queued preprocess command. Training
        snapshots strip preprocess-only keys (incl. source_image_dir), so use
        the Preprocess tab's own snapshot when available."""
        if self._preprocess_tab is not None:
            return self._preprocess_tab.preprocess_config_snapshot()
        return self._scoped_merged(variant)

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
        self._set_pickers_enabled(False)
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
        merged = self._scoped_merged(variant)
        require_pe, pe_encoder = submit.repa_requirements(merged)

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
        merged = self._scoped_merged(variant)
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

        merged = self._scoped_merged(variant)
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
        merged = self._scoped_merged(variant)
        self._submit_training(
            lambda: gui_daemon.submit_training(
                method=variant,
                preset=self._current_preset(),
                methods_subdir="gui-methods",
                config_snapshot=self._queue_config_snapshot(variant, merged),
                start=True,  # main Train button: run now
            ),
            logging_dir=merged.get("logging_dir"),
        )

    def _submit_training(self, submit_fn, *, logging_dir: str | None) -> None:
        """Busy UI → ``submit_fn`` (a ``gui_daemon.submit_training`` call) →
        attach to the job. Every Train launch goes through here."""
        if logging_dir and self._tb_panel is not None:
            self._tb_panel.set_log_dir(logging_dir)

        # Flip to busy before the submit (cold-start daemon /health wait can
        # take a moment) so the UI feels responsive.
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(action_button_qss("busy"))
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self._set_pickers_enabled(False)
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_submitting") + "\n")
        QApplication.processEvents()

        job_id = self._submit_job(submit_fn, on_fail=self._restore_idle_ui)
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
        self._reattach(job_id, kind=reattach_kind)

    def _reattach(self, job_id: str, *, kind: str) -> None:
        """Attach to an already-running job, replaying its log from the top."""
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_reattached", job_id=job_id))
        self._attach_to_job(job_id, replay_log=True, kind=kind)

    def _attach_to_job(
        self, job_id: str, *, replay_log: bool, kind: str = "train"
    ) -> None:
        """Point the bar + log at a daemon job's on-disk files and start
        polling. ``replay_log`` reads ``stdout.log`` from the top (re-attach
        after a GUI restart); otherwise a fresh launch shows only new lines.
        ``kind`` "preprocess" jobs emit no progress.jsonl, so the bar falls
        back to tqdm parsing in _drain_job_stdout."""
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
        self._jsonl_reader.watch(gui_daemon.progress_path(job_id))
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
        self._set_pickers_enabled(True)
        self.stop_btn.setEnabled(True)
        self._watch_job(job_id, replay_log=replay_log)

    def _route_progress_line(self, line: str) -> bool:
        # Once progress.jsonl drives the bar (training), tqdm lines are only
        # swallowed; before that (preprocess, Test) tqdm drives the bar.
        if self._jsonl_reader.active:
            return bool(TQDM_RE.search(line))
        return self._progress_tracker.feed(line)

    def _emit_log_line(self, line: str) -> None:
        self._log(line + "\n")

    def _on_job_tick(self) -> None:
        self._jsonl_reader.poll()
        # Refresh the gallery as samples land, but only while the panel isn't
        # pinned to field help.
        if self._job_kind == "train" and self._explain.mode in (None, "sample"):
            self._show_sample_output()

    def _on_job_finished(self, state: str | None) -> None:
        self._jsonl_reader.poll()
        job_id = self._end_job_watch()
        kind, self._job_kind = self._job_kind, None
        self._jsonl_reader.reset()
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

    def _set_pickers_enabled(self, enabled: bool) -> None:
        """Method / variant / + New / Hardware pickers. Locked while a launch
        is in flight; left live while attached so another variant can be queued."""
        for w in (
            self.method_combo,
            self.variant_combo,
            self.new_variant_btn,
            self.preset_combo,
        ):
            w.setEnabled(enabled)

    def _restore_idle_ui(self):
        """Return every control to its idle state."""
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(action_button_qss("primary"))
        self.train_btn.setEnabled(True)
        self.test_btn.setText(t("test"))
        apply_variant(self.test_btn, "secondary")
        self.test_btn.setEnabled(self._has_lora_output())
        self.stop_btn.setEnabled(False)
        self._set_pickers_enabled(True)
        if self._tb_panel is not None:
            self._tb_panel.clear_current_run()

    def _stop_training(self):
        # A daemon job is aborted via the daemon; a direct child is killed as a tree.
        if self._job_id:
            self._stop_job()
            return
        self._proc.kill()

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
        self._proc.kill()

    def _reset_progress(self):
        self._stdout_buf = ""
        self._progress_tracker.reset()
        self._jsonl_reader.reset()

    def _on_finished(self, exit_code: int):
        # The direct child: Test, or EasyControl's preprocess (training and the
        # auto-chain preprocess are daemon jobs).
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
