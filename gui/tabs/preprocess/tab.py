"""PreprocessingTab — the cache builder. Composes the knob sections (image
prep / text caching / caption mirror) over the Qt-free knob table, and owns
what isn't a knob: the method/variant bar, Save + the Run split buttons, the
status row, the explanation panel, the log, and the daemon job observer.

Layout mirrors ConfigTab: top action bar, form+explain split, log panel.
Surfaces the knobs the bare ``make preprocess`` path hardcodes.

Curation (autotag, position clauses, SAM masks) is the ``anime_tools``
panel's; this tab never runs it — ``knobs.CURATION_GATES_OFF`` rides in every
env it builds, the Train auto-chain's included.

Settings persist to the selected ``configs/gui-methods/<variant>.toml``
``[variant]`` table: the trainer-native knobs as flat keys, each
``anime_tools`` stage form under ``[variant.stages.<stage_id>]``. At submit the
stage forms travel as ``PREPROCESS_STAGES_JSON`` (``stage_form.STAGE_VALUES_ENV``)
and ``tasks.py`` builds each request through the package's ``build_argv``.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

import toml
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
    QHBoxLayout,
    QWidget,
)

from gui import (
    IMAGE_EXTS,
    ROOT,
    LazyTabMixin,
    _load,
    _save,
    count_preprocess_caches,
    default_lora_cache_dir,
    default_mask_dir,
    default_resized_dir,
    list_gui_variants,
    merged_gui_variant_preset,
    variant_path,
)
from gui.core import anime_tools_panel
from gui.jobs import daemon as gui_daemon
from gui.jobs.mixin import DaemonJobMixin
from gui.core.paths import read_gui_settings
from gui.explanations import field_help_html, preprocess_guide
from gui.i18n import t
from gui.jobs.progress import TqdmProgressTracker, make_progress_bar
from gui.core import submit
from gui.tabs.preprocess.captions import CaptionEditingSection
from gui.tabs.preprocess.image_prep import ImagePrepSection
from gui.tabs.preprocess.knobs import (
    DEFAULT_PREPROCESS_PATH_PATTERN,
    DEFAULT_TE_TAG_DROPOUT,
    PREPROCESS_ONLY_KEYS,
    load_values,
    merge_into_meta,
    resolved_defaults,
    to_env,
    to_overrides,
)
from gui.tabs.preprocess.stage_form import (
    STAGE_IDS,
    STAGE_VALUES_ENV,
    argv_for,
    load_stage_schemas,
    load_stage_values,
    merge_stages_into_meta,
    seeded_defaults,
)
from gui.tabs.preprocess.text_caching import TextCachingSection
from gui.theme import action_button_qss, rich_text_pt as _explain_pt, tok
from gui.widgets import (
    DirtyTrackingMixin,
    SplitButtonStyle,
    action_button,
    apply_variant,
)
from library.datasets.path_filter import filter_paths_by_glob

PREPROCESS_TOML = ROOT / "configs" / "preprocess.toml"

PREPROCESS_METHODS = ["lora", "tlora", "hydralora"]

# Sourced from base.toml via gui.core.config_io (shared with the Config/EasyControl tabs);
# fallback only, when the variant doesn't override the path.
RESIZED_DIR = default_resized_dir()
LORA_CACHE_DIR = default_lora_cache_dir()
MASK_DIR = default_mask_dir()

# Legacy widget attribute names → (section attribute, key). Kept for
# one release so tests and the resize preview that reach into ``tab.<widget>``
# stay valid; new code should go through ``tab.values()`` / ``tab.stage_values()``
# or the owning section instead. A key is a knob-table key for a trainer row
# (``knob_widgets``) or a stage dest (``widgets``).
_WIDGET_ALIASES: dict[str, tuple[str, str]] = {
    "source_dir_edit": ("image_section", "source_image_dir"),
    "path_scope_edit": ("image_section", "path_scope"),
    "preprocess_path_pattern_edit": ("image_section", "preprocess_path_pattern"),
    "target_res_widget": ("image_section", "target_res"),
    "resize_crop_anchor_widget": ("image_section", "resize_crop_anchor"),
    "resize_crop_margins_widget": ("image_section", "resize_crop_margins"),
    "freefit_max_ratio_spin": ("image_section", "freefit_max_ratio"),
    "shuffle_spin": ("text_section", "caption_shuffle_variants"),
    "dropout_edit": ("text_section", "caption_tag_dropout_rate"),
    "caption_no_correct_chk": ("caption_section", "no_correct"),
    "caption_insert_no_artist_chk": ("caption_section", "caption_insert_no_artist"),
    "caption_trigger_word_edit": ("caption_section", "caption_trigger_word"),
    "caption_trigger_at_front_chk": ("caption_section", "caption_trigger_at_front"),
    "caption_drop_groups_edit": ("caption_section", "caption_drop_groups"),
}

# Placeholder roots for validating a stage form at Save / Run (the real
# roots are filled by ``tasks.py`` at submit); ``src`` / ``dst`` are required
# by the ``correct`` request, so they must be non-empty here.
_VALIDATION_ROOTS = {
    "src": "image_dataset",
    "dst": "post_image_dataset/resized",
    "masks": "post_image_dataset/masks",
}


def _load_preprocess_toml() -> dict:
    """Read configs/preprocess.toml, {} if absent/unparseable. CLI-default
    fallback only; GUI edits are stored on the selected gui-method variant."""
    if not PREPROCESS_TOML.exists():
        return {}
    try:
        return toml.loads(PREPROCESS_TOML.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}


def _filtered_files(root: Path, pattern: str | None, predicate) -> list[Path]:
    if not root.is_dir():
        return []
    paths = [p for p in root.rglob("*") if p.is_file() and predicate(p)]
    if pattern and pattern != "*":
        keep = filter_paths_by_glob([str(p) for p in paths], str(root), pattern)
        paths = [p for p, k in zip(paths, keep) if k]
    return paths


def _count_masks(mask_dir: Path, path_pattern: str | None = None) -> int:
    # rglob picks up nested `<rel>/` subtrees from `make mask`; flat trees still count correctly.
    return len(
        _filtered_files(mask_dir, path_pattern, lambda p: p.name.endswith("_mask.png"))
    )


def _count_resized(resized_dir: Path, path_pattern: str | None = None) -> int:
    # rglob picks up nested `<rel>/` subtrees from the resize stage; flat trees still count correctly.
    return len(
        _filtered_files(
            resized_dir, path_pattern, lambda p: p.suffix.lower() in IMAGE_EXTS
        )
    )


def _count_tree_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for p in path.rglob("*") if p.is_file())


class PreprocessingTab(DaemonJobMixin, DirtyTrackingMixin, LazyTabMixin, QWidget):
    def __init__(self):
        super().__init__()
        # Each Run submits a detached daemon "command" job (mirrors ConfigTab's
        # Train button) so it survives the GUI closing and queues behind training.
        self._init_job_observer()
        self._run_buttons: list[QToolButton] = []
        # Kept alive here because setStyle() does not take ownership.
        self._split_styles: list[SplitButtonStyle] = []
        self._variant: str | None = None
        self._resize_preview = None  # ResizePreviewDialog, built on first open
        self._loading_variant = False
        self._dirty = False

        outer = QVBoxLayout(self)
        outer.addLayout(self._build_top_bar())

        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        outer.addWidget(self.progress)
        outer.addLayout(self._build_status_row())

        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)
        hsplit.addWidget(self._build_form())

        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            f"QTextBrowser {{ font-size: 120%; padding: 12px; "
            f"background: {tok('panel')}; color: {tok('text')}; }}"
        )
        self._explain.setMinimumWidth(320)
        self._show_default_explain()
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([720, 420])
        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("preprocess_log_placeholder"))
        vsplit.addWidget(self.log)
        vsplit.setSizes([520, 200])
        outer.addWidget(vsplit, 1)

        for section in self.sections:
            section.changed.connect(self._mark_dirty)
        self._clear_dirty()

    # -- construction -------------------------------------------------------

    def _build_top_bar(self) -> QHBoxLayout:
        top = QHBoxLayout()
        # Split buttons need a per-widget stylesheet (a global [variant] rule
        # bypasses SplitButtonStyle and miscentres the label).
        run_step_style = action_button_qss("info")

        self._method_label = QLabel("Method")
        top.addWidget(self._method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(PREPROCESS_METHODS)
        self.method_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.method_combo.setMinimumContentsLength(
            max((len(m) for m in PREPROCESS_METHODS), default=10)
        )
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        top.addWidget(self.method_combo)

        self._variant_label = QLabel(t("variant"))
        top.addWidget(self._variant_label)
        self.variant_combo = QComboBox()
        self.variant_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.variant_combo.setMinimumContentsLength(20)
        self.variant_combo.currentTextChanged.connect(self._on_variant_changed)
        top.addWidget(self.variant_combo, 1)
        self._refresh_variant_row(self.method_combo.currentText())

        self.save_btn = QPushButton(t("preprocess_save_settings"))
        self.save_btn.setToolTip(t("preprocess_save_settings_tip"))
        self.save_btn.clicked.connect(self._save_all_clicked)
        top.addWidget(self.save_btn)

        # Save is implicit on each Run (matches ConfigTab's auto-save before Train).
        self.run_te_btn = self._make_run_button(
            t("preprocess_run_te"), run_step_style, self._run_te
        )
        top.addWidget(self.run_te_btn)
        # Standalone PE (vision-encoder) caching — refresh PE sidecars without
        # re-running VAE+text; encoder follows the variant's repa_encoder.
        self.run_pe_btn = self._make_run_button(
            t("preprocess_run_pe"), run_step_style, self._run_pe
        )
        top.addWidget(self.run_pe_btn)

        top.addStretch()
        self.stop_btn = action_button(t("stop"), variant="danger", on_click=self._stop)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)
        return top

    def _build_status_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{tok('text')}; padding: 2px 0;")
        row.addWidget(self.status_lbl)
        row.addStretch()
        self.resize_preview_btn = QToolButton()
        self.resize_preview_btn.setText("🔍 " + t("preprocess_resize_preview"))
        self.resize_preview_btn.setToolTip(t("preprocess_resize_preview_tooltip"))
        self.resize_preview_btn.clicked.connect(self._open_resize_preview)
        row.addWidget(self.resize_preview_btn)
        self.open_dataset_btn = QToolButton()
        self.open_dataset_btn.setText("📂 " + t("preprocess_open_dataset_dir"))
        self.open_dataset_btn.setToolTip(t("preprocess_open_dataset_dir_tooltip"))
        self.open_dataset_btn.clicked.connect(self._open_dataset_dir)
        row.addWidget(self.open_dataset_btn)
        self.clear_scope_cache_btn = QToolButton()
        self.clear_scope_cache_btn.setText(t("preprocess_clear_scope_cache"))
        self.clear_scope_cache_btn.setToolTip(t("preprocess_clear_scope_cache_tooltip"))
        self.clear_scope_cache_btn.clicked.connect(self._clear_scope_preprocess_files)
        row.addWidget(self.clear_scope_cache_btn)
        return row

    def _build_form(self) -> QScrollArea:
        """The section stack. Each section seeds its widgets from the same
        default sources ``_resolved_defaults`` / ``_stage_defaults`` read, so
        a fresh tab (no variant loaded yet) already shows the effective
        defaults."""
        settings = read_gui_settings()
        pp_cfg = _load_preprocess_toml()
        help_cb = self._show_field_help
        self._schemas = load_stage_schemas()
        stage_defaults = self._stage_defaults(pp_cfg)

        self.image_section = ImagePrepSection(
            self._schemas["resize"],
            help_cb,
            pp_cfg=pp_cfg,
            defaults=stage_defaults["resize"],
        )
        self.text_section = TextCachingSection(help_cb, settings=settings)
        self.caption_section = CaptionEditingSection(
            self._schemas["correct"],
            help_cb,
            defaults=stage_defaults["correct"],
        )
        self.sections = (
            self.image_section,
            self.text_section,
            self.caption_section,
        )
        # stage id → the section holding its form.
        self.stage_sections = {
            "resize": self.image_section,
            "correct": self.caption_section,
        }

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        for section in self.sections:
            layout.addWidget(section)
        layout.addStretch()
        scroll.setWidget(host)
        return scroll

    def _stage_defaults(self, pp_cfg: dict) -> dict[str, dict]:
        """``{stage_id: {dest: default}}`` — the schema defaults with
        ``preprocess.toml`` seeded on top (``stage_form.seeded_defaults``)."""
        return {sid: seeded_defaults(self._schemas[sid], pp_cfg) for sid in STAGE_IDS}

    def __getattr__(self, name: str):
        # Legacy ``tab.<widget>`` access → the owning section's widget (see
        # _WIDGET_ALIASES). Only reached when normal lookup fails, and never
        # before the sections exist.
        alias = _WIDGET_ALIASES.get(name)
        if alias is not None and self.__dict__.get("sections"):
            section = self.__dict__.get(alias[0])
            if section is not None:
                key = alias[1]
                if key in section.widgets:
                    return section.widgets[key]
                knobs = getattr(section, "knob_widgets", {})
                if key in knobs:
                    return knobs[key]
        raise AttributeError(name)

    def _lazy_init(self) -> None:
        self._refresh_status()
        self._try_reattach()

    # -- variant selection --------------------------------------------------

    def _refresh_variant_row(self, method: str) -> None:
        variants = list_gui_variants(method)
        current = [
            self.variant_combo.itemText(i) for i in range(self.variant_combo.count())
        ]
        if current == variants:
            return
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        if variants:
            self.variant_combo.addItems(variants)
        self.variant_combo.blockSignals(False)

    def _on_method_changed(self, method: str) -> None:
        if self._loading_variant:
            return
        self._refresh_variant_row(method)
        self.set_variant(self.variant_combo.currentText(), method=method)

    def _on_variant_changed(self, variant: str) -> None:
        if self._loading_variant:
            return
        self.set_variant(variant, method=self.method_combo.currentText())

    def set_variant(self, variant: str, *, method: str | None = None) -> None:
        """Load GUI preprocess controls for the selected training variant."""
        if not variant:
            return
        if method:
            self._loading_variant = True
            try:
                if self.method_combo.currentText() != method:
                    self.method_combo.setCurrentText(method)
                self._refresh_variant_row(method)
                if self.variant_combo.currentText() != variant:
                    self.variant_combo.setCurrentText(variant)
            finally:
                self._loading_variant = False
        self._variant = variant
        meta = self._variant_preprocess_meta(variant)
        values = load_values(meta, self._resolved_defaults())
        self.set_values(values)
        self.set_stage_values(self._load_stage_values(meta))
        if hasattr(self, "status_lbl"):
            self._refresh_status()
        if self._resize_preview is not None and self._resize_preview.isVisible():
            self._resize_preview.refresh()
        self._clear_dirty()

    @staticmethod
    def _variant_preprocess_meta(variant: str) -> dict:
        try:
            data = _load(variant_path(variant))
        except Exception:
            return {}
        meta = data.get("variant")
        if not isinstance(meta, dict):
            return {}
        return {k: meta[k] for k in PREPROCESS_ONLY_KEYS if k in meta}

    def _load_stage_values(self, meta: dict) -> dict[str, object]:
        """``{stage_id: values}`` for a variant: its ``[variant.stages.*]``
        over the seeded defaults."""
        defaults = self._stage_defaults(_load_preprocess_toml())
        return {sid: load_stage_values(meta, sid, defaults[sid]) for sid in STAGE_IDS}

    # -- values (the section surface) ---------------------------------------

    def values(self) -> dict[str, object]:
        """Raw widget state of the trainer-native knobs (``knobs.KNOBS``),
        merged across sections. Free-text numerics (``caption_tag_dropout_rate``)
        stay as the line-edit text — validated where they're persisted."""
        out: dict[str, object] = {}
        for section in self.sections:
            reader = getattr(section, "knob_values", None)
            out.update(reader() if reader is not None else section.values())
        return out

    _widget_values = values  # legacy name

    def stage_values(self) -> dict[str, object]:
        """``{stage_id: {dest: value}}`` for every stage form — what the job
        receives and the variant keeps."""
        out: dict[str, object] = {}
        for sid, section in self.stage_sections.items():
            reader = getattr(section, "stage_values", None)
            out[sid] = reader() if reader is not None else section.values()
        return out

    def set_values(self, values: dict) -> None:
        """Push trainer-knob values into every section without tripping the
        dirty flag."""
        self._loading_variant = True
        try:
            for section in self.sections:
                setter = getattr(section, "set_knob_values", None)
                (setter or section.set_values)(values)
        finally:
            self._loading_variant = False

    def set_stage_values(self, stage_values: dict) -> None:
        self._loading_variant = True
        try:
            for sid, section in self.stage_sections.items():
                if sid not in stage_values:
                    continue
                setter = getattr(section, "set_stage_values", None)
                (setter or section.set_values)(stage_values[sid])
        finally:
            self._loading_variant = False

    # Thin delegates kept for the tests that call them directly.
    def _set_target_res_widget(self, values) -> None:
        self.image_section.set_target_res(values)

    def _set_resize_crop_anchor(self, value) -> None:
        self.resize_crop_anchor_widget.set_value(value, emit=False)

    def _set_resize_crop_margins(self, value) -> None:
        self.resize_crop_margins_widget.set_value(value)

    def _resize_crop_margins(self) -> dict[str, float]:
        return self.resize_crop_margins_widget.margins()

    # -- dirty / help / status ----------------------------------------------

    def _update_save_button(self):
        if not hasattr(self, "save_btn"):
            return
        if self._dirty:
            self.save_btn.setText(t("preprocess_save_settings") + " *")
            apply_variant(self.save_btn, "warning")
            self.save_btn.setToolTip(t("save_dirty_tooltip"))
        else:
            self.save_btn.setText(t("preprocess_save_settings"))
            apply_variant(self.save_btn, None)
            self.save_btn.setToolTip(t("preprocess_save_settings_tip"))

    def _show_default_explain(self) -> None:
        self._explain.setHtml(preprocess_guide())

    def _show_field_help(self, field_label: str, help_text: str | None) -> None:
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:{_explain_pt(18)};'>"
            f"{html.escape(field_label)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:{_explain_pt(15)}; line-height:1.6;'>"
                f"{field_help_html(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:{tok('text_dim')}; font-style:italic;'>"
                f"{html.escape(t('no_help_available'))}</p>"
            )
        self._explain.setHtml("".join(parts))

    def _refresh_status(self) -> None:
        snapshot = self.preprocess_config_snapshot()
        preprocess_pattern = (
            self.preprocess_path_pattern_edit.text().strip()
            or DEFAULT_PREPROCESS_PATH_PATTERN
        )
        path_pattern = (
            preprocess_pattern
            if preprocess_pattern != DEFAULT_PREPROCESS_PATH_PATTERN
            else str(snapshot.get("path_pattern") or DEFAULT_PREPROCESS_PATH_PATTERN)
        )
        n_resized = _count_resized(
            self._snapshot_path(snapshot, "resized_image_dir", RESIZED_DIR),
            path_pattern,
        )
        caches = count_preprocess_caches(
            self._snapshot_path(snapshot, "lora_cache_dir", LORA_CACHE_DIR),
            path_pattern,
            pe_encoder=str(snapshot.get("repa_encoder") or "pe_spatial").strip()
            or None,
        )
        mask_n = _count_masks(
            self._snapshot_path(snapshot, "mask_dir", MASK_DIR), path_pattern
        )
        if n_resized == 0:
            self.status_lbl.setText(t("preprocess_status_no_resized"))
            return
        lines = [
            t("preprocess_status_resized", n=n_resized),
            t(
                "preprocess_status_caches",
                lat=caches["latents"],
                te=caches["te"],
                pe=caches["pe"],
            ),
            t("preprocess_status_masks", masks=mask_n),
        ]
        if anime_tools_panel.export_is_stale():
            lines.append(t("preprocess_status_export_stale"))
        self.status_lbl.setText("  |  ".join(lines))

    def _open_dataset_dir(self) -> None:
        """Open the post_image_dataset/ folder (resized + caches) in the OS file manager."""
        snapshot = self.preprocess_config_snapshot()
        resized = self._snapshot_path(snapshot, "resized_image_dir", RESIZED_DIR)
        # post_image_dataset/ is the parent of resized/ (and lora/, masks/).
        target = resized.parent
        if not target.is_dir():
            target = ROOT
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _open_resize_preview(self) -> None:
        if self._resize_preview is None:
            from gui.tabs.preprocess.resize_preview import ResizePreviewDialog

            self._resize_preview = ResizePreviewDialog(self)
        self._resize_preview.show()
        self._resize_preview.raise_()
        self._resize_preview.activateWindow()

    @staticmethod
    def _snapshot_path(snapshot: dict[str, object], key: str, default: Path) -> Path:
        raw = snapshot.get(key)
        if not raw:
            return default
        p = Path(str(raw))
        return p if p.is_absolute() else ROOT / p

    def _normalize_scope_or_warn(self) -> str | None:
        raw = self.path_scope_edit.text().strip()
        if not raw:
            return ""
        scope = submit.normalize_path_scope(raw)
        if scope is None:
            QMessageBox.warning(
                self, t("error"), t("preprocess_invalid_path_scope", value=raw)
            )
            return None
        return scope

    def _clear_scope_preprocess_files(self) -> None:
        scope = self._normalize_scope_or_warn()
        if scope is None:
            return
        snapshot = self.preprocess_config_snapshot()
        resized = self._snapshot_path(snapshot, "resized_image_dir", RESIZED_DIR)
        lora = self._snapshot_path(snapshot, "lora_cache_dir", LORA_CACHE_DIR)
        targets = [resized, lora]

        for target in targets:
            outside = target == ROOT
            if not outside:
                try:
                    target.relative_to(ROOT)
                except ValueError:
                    outside = True
            if outside:
                QMessageBox.warning(
                    self,
                    t("error"),
                    t("preprocess_clear_scope_cache_outside_root", path=str(target)),
                )
                return

        resized_count = _count_tree_files(resized)
        lora_count = _count_tree_files(lora)
        if resized_count == 0 and lora_count == 0:
            QMessageBox.information(
                self,
                t("preprocess_clear_scope_cache"),
                t("preprocess_clear_scope_cache_empty"),
            )
            self._refresh_status()
            return

        scope_label = scope or t("preprocess_clear_scope_cache_all_scope")
        answer = QMessageBox.question(
            self,
            t("preprocess_clear_scope_cache"),
            t(
                "preprocess_clear_scope_cache_confirm",
                scope=scope_label,
                resized=str(resized),
                resized_count=resized_count,
                lora=str(lora),
                lora_count=lora_count,
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        removed = 0
        for target in targets:
            if not target.exists():
                continue
            removed += _count_tree_files(target)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        QMessageBox.information(
            self,
            t("preprocess_clear_scope_cache"),
            t("preprocess_clear_scope_cache_done", count=removed),
        )
        self._refresh_status()

    def _parse_float(self, text: str, field_label: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            QMessageBox.warning(
                self,
                t("error"),
                t("preprocess_invalid_float", field=field_label, value=text),
            )
            return None

    # -- the ConfigTab contract -------------------------------------------

    @staticmethod
    def _resolved_defaults() -> dict:
        """Effective per-knob defaults from the two sources the trainer knobs
        consult (``preprocess.toml`` / ``gui_settings.json``)."""
        return resolved_defaults(_load_preprocess_toml(), read_gui_settings())

    def preprocess_env(self) -> dict[str, str]:
        """Environment values consumed by ``tasks.py preprocess`` / ``mask``:
        the trainer knobs (``knobs.to_env``) plus every stage form as JSON
        (``PREPROCESS_STAGES_JSON``), from which ``tasks.py`` builds the
        ``anime_tools`` requests through ``build_argv``."""
        env = to_env(self.values(), self._resolved_defaults())
        env[STAGE_VALUES_ENV] = json.dumps(self.stage_values(), ensure_ascii=False)
        return env

    def preprocess_overrides(self) -> dict[str, object]:
        """Flat config overrides that should be captured in preprocess snapshots."""
        return to_overrides(self.values())

    def preprocess_config_snapshot(self) -> dict[str, object]:
        """Full preprocess config snapshot captured at GUI submit time. Paths
        come from the selected GUI method plus ``path_scope``.
        ``preprocess_path_pattern`` is not written into the flat config (training
        must not see this GUI-only filter) — forwarded via ``PREPROCESS_PATH_PATTERN`` instead."""
        variant = self._variant or "lora"
        merged, _ = merged_gui_variant_preset(variant, "default")
        # Seed source dir from the editable field before scoping, so path_scope
        # appends onto the user-chosen root, not the hard default.
        source_dir = self.source_dir_edit.text().strip()
        if source_dir:
            merged["source_image_dir"] = source_dir
        path_scope = self.path_scope_edit.text().strip()
        if path_scope:
            merged["path_scope"] = path_scope
        else:
            merged.pop("path_scope", None)
        return submit.preprocess_snapshot(merged, self.preprocess_overrides())

    def persist_target_res(self) -> None:
        """Mark dirty on tier change; ConfigTab's auto-chain/queue calls
        ``persist_preprocess_inputs`` before submit to capture the latest tiers
        without silently saving mid-edit."""
        self._mark_dirty()

    def persist_preprocess_inputs(self) -> bool:
        """Persist cache-building inputs for ConfigTab's auto-chain/queue."""
        return self._save_variant_preprocess_meta(validate_dropout=True)

    # -- save ---------------------------------------------------------------

    def _validate_stages(self) -> bool:
        """Run every stage form through the package's ``build_argv`` (the
        request's own ``__post_init__``): a bad value surfaces as a dialog
        here, not minutes into a cache build."""
        for sid, values in self.stage_values().items():
            try:
                argv_for(
                    self._schemas[sid],
                    values,
                    roots=_VALIDATION_ROOTS,
                    settings={"path_pattern": "*"},
                    mask_root="post_image_dataset",
                )
            except ValueError as exc:
                label = self._schemas[sid].get("title") or sid
                QMessageBox.warning(
                    self,
                    t("error"),
                    t("preprocess_invalid_stage", stage=label, err=str(exc)),
                )
                return False
        return True

    def _save_variant_preprocess_meta(self, *, validate_dropout: bool) -> bool:
        if not self._variant:
            return True
        dropout_text = self.dropout_edit.text().strip()
        if validate_dropout:
            dropout = self._parse_float(
                dropout_text, t("preprocess_caption_tag_dropout_rate")
            )
            if dropout is None:
                return False
        else:
            try:
                dropout = float(dropout_text)
            except ValueError:
                dropout = DEFAULT_TE_TAG_DROPOUT
        if not self._validate_stages():
            return False

        path = variant_path(self._variant)
        data = _load(path)
        meta = data.get("variant")
        if not isinstance(meta, dict):
            meta = {}

        scope = self._normalize_scope_or_warn()
        if scope is None:
            return False

        values = self.values()
        values["path_scope"] = scope
        values["caption_tag_dropout_rate"] = float(dropout)
        # Elision (pop-if-default, with the preprocess.toml-resolved comparison
        # for the caption-master stages) is the knob table's job; the stage
        # forms are elided against their seeded defaults by stage_form.
        merge_into_meta(meta, values, self._resolved_defaults())
        # Stage tables this tab no longer draws (an older GUI's ``autotag`` /
        # ``masks_sam``) are left as they are.
        merge_stages_into_meta(
            meta,
            self.stage_values(),
            self._stage_defaults(_load_preprocess_toml()),
            include_mask=False,
        )

        if meta:
            data["variant"] = meta
        else:
            data.pop("variant", None)
        _save(path, data)
        return True

    def _save_all(self) -> bool:
        """Validate and persist every form value. Returns True on success."""
        dropout = self._parse_float(
            self.dropout_edit.text().strip(), t("preprocess_caption_tag_dropout_rate")
        )
        if dropout is None:
            return False
        if not self._save_variant_preprocess_meta(validate_dropout=False):
            return False
        self._clear_dirty()
        return True

    def _save_all_clicked(self) -> None:
        if self._save_all():
            QMessageBox.information(self, t("saved"), t("preprocess_settings_saved"))

    # -- run ----------------------------------------------------------------

    def _is_running(self) -> bool:
        return self._job_id is not None

    def _make_run_button(self, label: str, style: str, run_cb) -> QToolButton:
        """Split Run button: main action runs now, dropdown queues it
        (``run_cb(queue=True)``, submit without attaching)."""
        btn = QToolButton()
        # SplitButtonStyle must outlive the button (setStyle doesn't take ownership), so stash a ref.
        split_style = SplitButtonStyle()
        self._split_styles.append(split_style)
        btn.setStyle(split_style)
        btn.setText(label)
        btn.setStyleSheet(style)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setPopupMode(QToolButton.MenuButtonPopup)
        btn.clicked.connect(lambda _checked=False: run_cb())
        menu = QMenu(btn)
        queue_action = menu.addAction(t("preprocess_add_to_queue"))
        queue_action.triggered.connect(lambda _checked=False: run_cb(queue=True))
        btn.setMenu(menu)
        self._run_buttons.append(btn)
        return btn

    def _run_te(self, *, queue: bool = False) -> None:
        # `tasks.py preprocess` chains resize → VAE-latent cache → text-embedding cache.
        if not self._save_all():
            return
        self._submit(
            label="preprocess",
            argv=["tasks.py", "preprocess"],
            extra_env=self.preprocess_env(),
            config_snapshot=self.preprocess_config_snapshot(),
            attach=not queue,
        )

    def _run_pe(self, *, queue: bool = False) -> None:
        # Encoder follows the variant's `repa_encoder` (pe_spatial default; `pe` = PE-Core for CMMD).
        if not self._save_all():
            return
        variant = self._variant or "lora"
        merged, _ = merged_gui_variant_preset(variant, "default")
        encoder = (
            str(merged.get("repa_encoder") or "pe_spatial").strip() or "pe_spatial"
        )
        task = "preprocess-pe-spatial" if encoder == "pe_spatial" else "preprocess-pe"
        self._submit(
            label="preprocess-pe",
            argv=["tasks.py", task],
            extra_env=self.preprocess_env(),
            config_snapshot=self.preprocess_config_snapshot(),
            attach=not queue,
        )

    def _submit(
        self,
        *,
        label: str,
        argv: list[str],
        extra_env: dict,
        config_snapshot: dict | None = None,
        attach: bool = True,
    ) -> None:
        """Submit a preprocess job to the daemon (spawns ``python <argv>``
        detached, serialized behind any running training job). Pre-launch
        validation is the caller's job. ``attach=True`` (main Run) takes over
        the log/bar and blocks Run until the job finishes; ``attach=False``
        ("add to queue") submits silently and the job is watched from the
        Queue tab."""
        if attach and self._is_running():
            QMessageBox.information(self, "", t("preprocess_already_running"))
            return
        if attach:
            # Repaint before submit so the tab stays responsive during a cold-start daemon /health wait.
            self._set_busy_ui(True)
            self.log.clear()
            self._stdout_buf = ""
            self._progress_tracker.reset()
            self._progress_tracker.mark_starting(t("starting"))
            self.log.appendPlainText("> " + " ".join([sys.executable, *argv]))
            self.log.appendPlainText(t("daemon_submitting"))
            QApplication.processEvents()

        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(
                label=label,
                argv=argv,
                extra_env=extra_env,
                config_snapshot=config_snapshot,
                start=attach,
            ),
            on_fail=(self._restore_idle_ui if attach else None),
        )
        if not job_id:
            return
        if attach:
            self.log.appendPlainText(t("daemon_queued", job_id=job_id).rstrip("\n"))
            self._attach_to_job(job_id, replay_log=False)
        else:
            self.log.appendPlainText(t("preprocess_queued", label=label, job_id=job_id))

    # -- job observer -------------------------------------------------------

    def _try_reattach(self) -> None:
        """Bind to a preprocess job still running when the tab first opens
        (so close-mid-preprocess → reopen re-attaches). Skips a training job
        (belongs to ConfigTab) and stays idle when the daemon is down."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable → nothing to attach
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "command":
            return
        # An auto-chain preprocess belongs to ConfigTab, which re-claims it so the chain into training stays there.
        if gui_daemon.read_job_chain_variant(job_id):
            return
        self.log.clear()
        self._stdout_buf = ""
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self.log.appendPlainText(t("daemon_reattached", job_id=job_id).rstrip("\n"))
        self._attach_to_job(job_id, replay_log=True)

    def _attach_to_job(self, job_id: str, *, replay_log: bool) -> None:
        """Point the log + bar at a daemon job's on-disk files and start polling.
        ``replay_log`` reads ``stdout.log`` from the top (re-attach case);
        otherwise a fresh launch shows only new lines."""
        self._set_busy_ui(True)
        self._watch_job(job_id, replay_log=replay_log)

    def _on_job_finished(self, state: str | None) -> None:
        job_id = self._end_job_watch()
        self._progress_tracker.reset()
        self.log.appendPlainText(gui_daemon.format_finish_banner(job_id, state))
        self._restore_idle_ui()
        self._refresh_status()

    def _set_busy_ui(self, busy: bool) -> None:
        for btn in self._run_buttons:
            btn.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _restore_idle_ui(self) -> None:
        self._set_busy_ui(False)

    def _stop(self) -> None:
        self._stop_job()

    def cleanup_subprocess(self) -> None:
        """App-shutdown hook. Stops observing but leaves the daemon job alive —
        it runs detached, so a cache build survives GUI close
        (re-attached on next launch)."""
        self._job_timer.stop()
