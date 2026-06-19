"""PreprocessingTab — caption shuffle/dropout + SAM3/MIT mask config.

Layout mirrors ConfigTab:
- Top action bar (refresh + per-step Run buttons + Save + Stop)
- Horizontal split: form on left (clickable labels show help on the right),
  explanation panel on the right
- Log panel below in a vertical splitter

Surfaces the knobs that the bare ``make preprocess`` / ``make mask`` paths
hardcode: caption shuffle variant count, per-tag dropout rate, SAM prompt
list / threshold / dilate, MIT text-threshold / dilate.

Settings persist to:
- the selected ``configs/gui-methods/<variant>.toml`` ``[variant]`` table —
  GUI-profile preprocess knobs such as preprocess path filter, target_res,
  low-res filtering, caption shuffle/dropout, and mask settings.
- ``configs/sam_mask.yaml`` — SAM prompts / threshold / dilate (existing
  canonical CLI fallback, read directly by ``scripts/preprocess/generate_masks.py``).
  GUI Save no longer writes this file; direct terminal ``make mask`` will not
  see GUI-profile mask settings unless the user edits the YAML manually.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
import copy
from pathlib import Path

import toml
import yaml
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import (
    IMAGE_EXTS,
    ROOT,
    LazyTabMixin,
    _TargetResWidget,
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
from gui import daemon as gui_daemon
from gui._job_mixin import DaemonJobMixin
from gui.explanations import preprocess_field_help, preprocess_guide
from gui.i18n import t
from gui.progress import TQDM_RE, TqdmProgressTracker, make_progress_bar
from gui.theme import action_button_qss, rich_text_pt as _explain_pt, tok
from gui.tabs.config_tab import ConfigTab, SplitButtonStyle
from gui.widgets import (
    ClickableLabel,
    DirtyTrackingMixin,
    action_button,
    apply_variant,
    make_field_label,
)
from library.datasets.buckets import DEFAULT_TARGET_RES as _LIB_DEFAULT_TARGET_RES
from library.datasets.path_filter import filter_paths_by_glob
from library.preprocess.resize_preview import (
    DEFAULT_FREEFIT_MAX_RATIO,
    DEFAULT_RESIZE_CROP_ANCHOR,
    normalize_crop_margins,
)

SAM_YAML = ROOT / "configs" / "sam_mask.yaml"
PREPROCESS_TOML = ROOT / "configs" / "preprocess.toml"
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "gui_settings.json"

# Defaults match the historical hardcoded values in scripts/tasks/preprocess.py
# and scripts/preprocess/generate_masks_mit.py so a freshly installed GUI runs the
# same pipeline as the bare CLI.
DEFAULT_SOURCE_IMAGE_DIR = "image_dataset"
DEFAULT_PREPROCESS_PATH_PATTERN = "*"
DEFAULT_DROP_LOWRES_IMAGES = True
DEFAULT_MIN_PIXELS = 500000
DEFAULT_TARGET_RES = list(_LIB_DEFAULT_TARGET_RES)
DEFAULT_RESIZE_BUCKET_RESOS: list[str] = []
DEFAULT_RESIZE_CROP_MARGINS = {"top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0}
DEFAULT_TE_SHUFFLE_VARIANTS = 4
DEFAULT_TE_TAG_DROPOUT = 0.1
DEFAULT_CAPTION_CORRECT_ORDER = False
DEFAULT_CAPTION_INSERT_NO_ARTIST = False
DEFAULT_CAPTION_TRIGGER_WORD = ""
DEFAULT_CAPTION_TRIGGER_AT_FRONT = False
DEFAULT_SAM_PROMPTS = ("speech bubble", "text bubble")
DEFAULT_SAM_THRESHOLD = 0.5
DEFAULT_SAM_DILATE = 5
DEFAULT_MASK_PATH_PATTERN = "*"
DEFAULT_MIT_TEXT_THRESHOLD = 0.8
DEFAULT_MIT_DILATE = 5
DEFAULT_RUN_SAM_MASK = True
DEFAULT_RUN_MIT_MASK = True
PREPROCESS_METHODS = ["lora", "tlora", "hydralora"]
_GUI_PREPROCESS_KEYS = {
    "path_scope",
    "source_image_dir",
    "preprocess_path_pattern",
    "drop_lowres_images",
    "min_pixels",
    "target_res",
    "resize_bucket_resos",
    "resize_crop_anchor",
    "resize_crop_margins",
    "freefit_max_ratio",
    "caption_shuffle_variants",
    "caption_tag_dropout_rate",
    "caption_correct_order",
    "caption_insert_no_artist",
    "caption_trigger_word",
    "caption_trigger_at_front",
    "run_sam_mask",
    "run_mit_mask",
    "mask_path_pattern",
    "mask_rules",
    "mit_text_threshold",
    "mit_dilate",
}

# Default cache/output dirs — sourced from base.toml (the canonical path keys
# the trainer reads) via gui.config_io so the Preprocess tab can't drift from
# the Config/EasyControl tabs or training. Used only as the fallback when the
# active variant's merged config doesn't override the path.
RESIZED_DIR = default_resized_dir()
LORA_CACHE_DIR = default_lora_cache_dir()
MASK_DIR = default_mask_dir()


class _ResizeCropAnchorWidget(QWidget):
    changed = Signal()

    _LAYOUT = (
        ("top_left", "↖", 0, 0),
        ("top", "↑", 0, 1),
        ("top_right", "↗", 0, 2),
        ("left", "←", 1, 0),
        ("center", "●", 1, 1),
        ("right", "→", 1, 2),
        ("bottom_left", "↙", 2, 0),
        ("bottom", "↓", 2, 1),
        ("bottom_right", "↘", 2, 2),
    )

    def __init__(self) -> None:
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)
        self._buttons: dict[str, QPushButton] = {}
        for key, text, row, col in self._LAYOUT:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedSize(32, 28)
            btn.setStyleSheet(
                "QPushButton { padding:0; } "
                f"QPushButton:checked {{ background:{tok('accent')}; color:#ffffff; "
                "font-weight:bold; }"
            )
            btn.setToolTip(t(f"resize_crop_anchor_{key}"))
            btn.clicked.connect(lambda _checked, value=key: self.set_value(value))
            grid.addWidget(btn, row, col)
            self._buttons[key] = btn
        self.set_value(DEFAULT_RESIZE_CROP_ANCHOR, emit=False)
        self.setFixedSize(32 * 3 + 2 * 2, 28 * 3 + 2 * 2)

    def value(self) -> str:
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return DEFAULT_RESIZE_CROP_ANCHOR

    def set_value(self, value, *, emit: bool = True) -> None:
        anchor = str(value or DEFAULT_RESIZE_CROP_ANCHOR)
        if anchor not in self._buttons:
            anchor = DEFAULT_RESIZE_CROP_ANCHOR
        for key, btn in self._buttons.items():
            btn.blockSignals(True)
            btn.setChecked(key == anchor)
            btn.blockSignals(False)
        if emit:
            self.changed.emit()


def _load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_preprocess_toml() -> dict:
    """Read configs/preprocess.toml (the preprocess-only knobs split out of
    base.toml). Returns {} if absent/unparseable so callers fall back to
    defaults. The GUI uses this only as the CLI-default fallback; GUI edits are
    stored on the selected gui-method variant."""
    if not PREPROCESS_TOML.exists():
        return {}
    try:
        return toml.loads(PREPROCESS_TOML.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}


def _load_sam_yaml() -> dict:
    if not SAM_YAML.exists():
        return {}
    try:
        return yaml.safe_load(SAM_YAML.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _load_rules(sam_yaml: dict) -> list[dict]:
    """Normalize either schema into a list of per-card rule dicts.

    A ``rules:`` array is returned card-for-card (per-rule threshold/dilate
    fall back to the top-level values). A flat config (no ``rules:`` key)
    collapses to one catch-all card carrying the top-level prompts.
    """
    default_threshold = float(sam_yaml.get("threshold", DEFAULT_SAM_THRESHOLD))
    default_dilate = int(sam_yaml.get("dilate", DEFAULT_SAM_DILATE))
    raw = sam_yaml.get("rules")
    if raw is None:
        return [
            {
                "path_pattern": "",
                "prompts": sam_yaml.get("prompts") or list(DEFAULT_SAM_PROMPTS),
                "focus_prompts": sam_yaml.get("focus_prompts") or [],
                "threshold": default_threshold,
                "dilate": default_dilate,
            }
        ]
    return [
        {
            "path_pattern": r.get("path_pattern") or "",
            "prompts": r.get("prompts") or [],
            "focus_prompts": r.get("focus_prompts") or [],
            "threshold": float(r.get("threshold", default_threshold)),
            "dilate": int(r.get("dilate", default_dilate)),
        }
        for r in raw
    ]


def _filtered_files(root: Path, pattern: str | None, predicate) -> list[Path]:
    if not root.is_dir():
        return []
    paths = [p for p in root.rglob("*") if p.is_file() and predicate(p)]
    if pattern and pattern != "*":
        keep = filter_paths_by_glob([str(p) for p in paths], str(root), pattern)
        paths = [p for p, k in zip(paths, keep) if k]
    return paths


def _count_masks(mask_dir: Path, path_pattern: str | None = None) -> int:
    if not mask_dir.is_dir():
        return 0
    # rglob picks up the nested `<rel>/` subtrees produced by `make mask`; legacy flat trees still count correctly.
    return len(
        _filtered_files(
            mask_dir,
            path_pattern,
            lambda p: p.name.endswith("_mask.png"),
        )
    )


def _count_resized(resized_dir: Path, path_pattern: str | None = None) -> int:
    if not resized_dir.is_dir():
        return 0
    # rglob picks up the nested `<rel>/` subtrees produced by recursive resize_images.py; flat trees still count correctly.
    return len(
        _filtered_files(
            resized_dir,
            path_pattern,
            lambda p: p.suffix.lower() in IMAGE_EXTS,
        )
    )


def _count_tree_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for p in path.rglob("*") if p.is_file())


class _RuleCard(QGroupBox):
    """One SAM mask rule editor: path_pattern + prompts/focus + threshold/dilate.

    ``prompts`` mask OUT (ignored in the loss); ``focus_prompts`` keep ONLY
    that subject (reversed polarity). An empty / ``*`` path_pattern is a
    catch-all. Emits ``removed(self)`` when its Remove button is clicked.
    """

    removed = Signal(object)
    changed = Signal()

    def __init__(self, rule: dict, help_cb):
        super().__init__(t("preprocess_sam_rule"))
        self._help_cb = help_cb
        form = QFormLayout(self)

        self.path_pattern_edit = QLineEdit(rule.get("path_pattern", ""))
        self.path_pattern_edit.setPlaceholderText("*")
        self.path_pattern_edit.setToolTip(t("preprocess_sam_rule_path_pattern_tip"))
        self.path_pattern_edit.textChanged.connect(lambda *_: self.changed.emit())
        form.addRow(
            self._label("sam_rule_path_pattern", t("preprocess_sam_rule_path_pattern")),
            self.path_pattern_edit,
        )

        self.prompts_edit = QPlainTextEdit("\n".join(rule.get("prompts") or []))
        self.prompts_edit.setMaximumHeight(70)
        self.prompts_edit.setStyleSheet("font-family:monospace;")
        self.prompts_edit.setToolTip(t("preprocess_sam_prompts_tip"))
        self.prompts_edit.textChanged.connect(lambda: self.changed.emit())
        form.addRow(
            self._label("sam_prompts", t("preprocess_sam_prompts")), self.prompts_edit
        )

        self.focus_prompts_edit = QPlainTextEdit(
            "\n".join(rule.get("focus_prompts") or [])
        )
        self.focus_prompts_edit.setMaximumHeight(70)
        self.focus_prompts_edit.setStyleSheet("font-family:monospace;")
        self.focus_prompts_edit.setToolTip(t("preprocess_sam_focus_prompts_tip"))
        self.focus_prompts_edit.textChanged.connect(lambda: self.changed.emit())
        form.addRow(
            self._label("sam_focus_prompts", t("preprocess_sam_focus_prompts")),
            self.focus_prompts_edit,
        )

        self.threshold_edit = QLineEdit(
            f"{float(rule.get('threshold', DEFAULT_SAM_THRESHOLD)):g}"
        )
        self.threshold_edit.setToolTip(t("preprocess_sam_threshold_tip"))
        self.threshold_edit.textChanged.connect(lambda *_: self.changed.emit())
        form.addRow(
            self._label("sam_threshold", t("preprocess_sam_threshold")),
            self.threshold_edit,
        )

        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 64)
        self.dilate_spin.setValue(int(rule.get("dilate", DEFAULT_SAM_DILATE)))
        self.dilate_spin.wheelEvent = lambda e: e.ignore()
        self.dilate_spin.valueChanged.connect(lambda *_: self.changed.emit())
        form.addRow(self._label("sam_dilate", t("preprocess_dilate")), self.dilate_spin)

        self.remove_btn = QPushButton(t("preprocess_sam_remove_rule"))
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        form.addRow("", self.remove_btn)

    def _label(self, key: str, text: str) -> ClickableLabel:
        """Clickable field label that routes this field's help to the panel."""
        help_text = preprocess_field_help(key)
        return make_field_label(
            text,
            style=f"color:{tok('text')}; text-decoration: underline dotted;",
            on_click=lambda _t=text, _h=help_text: self._help_cb(_t, _h),
        )

    def to_dict(self) -> dict:
        """Serialize to a rule dict. Raises ValueError on an unparseable threshold."""
        text = self.threshold_edit.text().strip()
        try:
            threshold = float(text)
        except ValueError as exc:
            raise ValueError(text) from exc
        rule: dict = {}
        pattern = self.path_pattern_edit.text().strip()
        if pattern and pattern != "*":
            rule["path_pattern"] = pattern
        prompts = [
            line.strip()
            for line in self.prompts_edit.toPlainText().splitlines()
            if line.strip()
        ]
        if prompts:
            rule["prompts"] = prompts
        focus = [
            line.strip()
            for line in self.focus_prompts_edit.toPlainText().splitlines()
            if line.strip()
        ]
        if focus:
            rule["focus_prompts"] = focus
        rule["threshold"] = threshold
        rule["dilate"] = int(self.dilate_spin.value())
        return rule


class PreprocessingTab(DaemonJobMixin, DirtyTrackingMixin, LazyTabMixin, QWidget):
    def __init__(self):
        super().__init__()
        # Daemon-backed preprocessing (mirrors ConfigTab's Train button): each Run submits a detached "command" job to the local daemon so it survives the GUI closing and shares the daemon's serial single-GPU queue with training.
        self._init_job_observer()
        self._run_buttons: list[QToolButton] = []
        # Kept alive here because setStyle() does not take ownership.
        self._split_styles: list[SplitButtonStyle] = []
        self._variant: str | None = None
        self._loading_variant = False
        self._dirty = False

        outer = QVBoxLayout(self)

        top = QHBoxLayout()

        # Split buttons keep a per-widget stylesheet (the global [variant] rule
        # bypasses SplitButtonStyle and miscentres the label) — colour from the
        # central ACTION_COLORS "info" via action_button_qss.
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

        # Standalone PE (vision-encoder) caching, letting the user refresh PE sidecars without re-running the whole VAE+text pass; encoder follows the variant's repa_encoder.
        self.run_pe_btn = self._make_run_button(
            t("preprocess_run_pe"), run_step_style, self._run_pe
        )
        top.addWidget(self.run_pe_btn)

        self.run_mask_btn = self._make_run_button(
            t("preprocess_run_mask"), run_step_style, self._run_mask
        )
        top.addWidget(self.run_mask_btn)

        top.addStretch()
        self.stop_btn = action_button(t("stop"), variant="danger", on_click=self._stop)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)
        outer.addLayout(top)

        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        outer.addWidget(self.progress)

        status_row = QHBoxLayout()
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{tok('text')}; padding: 2px 0;")
        status_row.addWidget(self.status_lbl)
        status_row.addStretch()
        self.open_dataset_btn = QToolButton()
        self.open_dataset_btn.setText("📂 " + t("preprocess_open_dataset_dir"))
        self.open_dataset_btn.setToolTip(t("preprocess_open_dataset_dir_tooltip"))
        self.open_dataset_btn.clicked.connect(self._open_dataset_dir)
        status_row.addWidget(self.open_dataset_btn)
        self.clear_scope_cache_btn = QToolButton()
        self.clear_scope_cache_btn.setText(t("preprocess_clear_scope_cache"))
        self.clear_scope_cache_btn.setToolTip(t("preprocess_clear_scope_cache_tooltip"))
        self.clear_scope_cache_btn.clicked.connect(self._clear_scope_preprocess_files)
        status_row.addWidget(self.clear_scope_cache_btn)
        outer.addLayout(status_row)

        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form_layout = QVBoxLayout(form_host)
        form_layout.setContentsMargins(0, 0, 0, 0)

        settings = _load_settings()
        pp_cfg = _load_preprocess_toml()
        sam_yaml = _load_sam_yaml()
        sam_rules = _load_rules(sam_yaml)
        mask_path_pattern = sam_yaml.get("path_pattern") or DEFAULT_MASK_PATH_PATTERN

        # GUI cache knobs are stored on the selected gui-method variant; configs/preprocess.toml remains the CLI default/fallback.
        img_box = QGroupBox(t("preprocess_image_prep"))
        img_form = QFormLayout()

        self.source_dir_edit = QLineEdit(
            str(pp_cfg.get("source_image_dir", DEFAULT_SOURCE_IMAGE_DIR))
        )
        self.source_dir_edit.setPlaceholderText(DEFAULT_SOURCE_IMAGE_DIR)
        self.source_dir_edit.setToolTip(t("preprocess_source_image_dir_tip"))
        img_form.addRow(
            self._field_label("source_image_dir", t("preprocess_source_image_dir")),
            self.source_dir_edit,
        )

        self.path_scope_edit = QLineEdit()
        self.path_scope_edit.setPlaceholderText("data_group1")
        self.path_scope_edit.setToolTip(preprocess_field_help("path_scope") or "")
        img_form.addRow(
            self._field_label("path_scope", t("path_scope")),
            self.path_scope_edit,
        )

        self.preprocess_path_pattern_edit = QLineEdit(
            str(pp_cfg.get("preprocess_path_pattern", DEFAULT_PREPROCESS_PATH_PATTERN))
        )
        self.preprocess_path_pattern_edit.setPlaceholderText("*")
        self.preprocess_path_pattern_edit.setToolTip(t("preprocess_path_pattern_tip"))
        img_form.addRow(
            self._field_label("preprocess_path_pattern", t("preprocess_path_pattern")),
            self.preprocess_path_pattern_edit,
        )

        self.drop_lowres_chk = QCheckBox(t("preprocess_drop_lowres"))
        self.drop_lowres_chk.setToolTip(t("preprocess_drop_lowres_tip"))
        self.drop_lowres_chk.setChecked(
            bool(pp_cfg.get("drop_lowres_images", DEFAULT_DROP_LOWRES_IMAGES))
        )
        img_form.addRow(
            self._field_label("drop_lowres_images", t("preprocess_drop_lowres")),
            self.drop_lowres_chk,
        )

        self.min_pixels_spin = QSpinBox()
        self.min_pixels_spin.setRange(0, 100_000_000)
        self.min_pixels_spin.setSingleStep(50_000)
        self.min_pixels_spin.setGroupSeparatorShown(True)
        self.min_pixels_spin.setValue(int(pp_cfg.get("min_pixels", DEFAULT_MIN_PIXELS)))
        self.min_pixels_spin.wheelEvent = lambda e: e.ignore()
        # min_pixels only applies when the filter is on (mirrors the CLI: drop_lowres=false → --min_pixels 0).
        self.min_pixels_spin.setEnabled(self.drop_lowres_chk.isChecked())
        self.drop_lowres_chk.toggled.connect(self.min_pixels_spin.setEnabled)
        img_form.addRow(
            self._field_label("min_pixels", t("preprocess_min_pixels")),
            self.min_pixels_spin,
        )

        # Dual-use: preprocess resizes to these tiers and train.py reads the same value back (via load_method_preset) to size the compile cache — this widget is the single source of truth.
        self.target_res_widget = _TargetResWidget(
            pp_cfg.get("target_res", DEFAULT_TARGET_RES)
        )
        # Mark dirty on every toggle so the Config tab's Train auto-chain doesn't preprocess at the stale tier when the user changed tiers without clicking Save.
        self.target_res_widget.changed.connect(self.persist_target_res)
        img_form.addRow(
            self._field_label("target_res", t("preprocess_target_res")),
            self.target_res_widget,
        )
        self.resize_crop_anchor_widget = _ResizeCropAnchorWidget()
        self.resize_crop_anchor_widget.setToolTip(t("resize_crop_anchor_tip"))
        self.resize_crop_anchor_widget.changed.connect(self.persist_target_res)
        img_form.addRow(
            self._field_label("resize_crop_anchor", t("resize_crop_anchor")),
            self.resize_crop_anchor_widget,
        )
        margins_row = QHBoxLayout()
        margins_row.setContentsMargins(0, 0, 0, 0)
        margins_row.setSpacing(6)
        self.resize_margin_top_spin = self._margin_spin()
        self.resize_margin_right_spin = self._margin_spin()
        self.resize_margin_bottom_spin = self._margin_spin()
        self.resize_margin_left_spin = self._margin_spin()
        for label, spin in (
            (t("resize_crop_margin_top"), self.resize_margin_top_spin),
            (t("resize_crop_margin_right"), self.resize_margin_right_spin),
            (t("resize_crop_margin_bottom"), self.resize_margin_bottom_spin),
            (t("resize_crop_margin_left"), self.resize_margin_left_spin),
        ):
            lbl = QLabel(label)
            lbl.setMinimumWidth(24)
            lbl.setStyleSheet(f"QLabel {{ color:{tok('text')}; }}")
            margins_row.addWidget(lbl)
            margins_row.addWidget(spin)
        margins_row.addStretch(1)
        img_form.addRow(
            self._field_label("resize_crop_margins", t("resize_crop_margins")),
            margins_row,
        )

        # Free-fit is the only resize mode: images keep their native aspect and the
        # patch-grid token count lands anywhere inside the tier's band (no discrete
        # (W, H) bucket snap). Only the max-ratio clamp is user-tunable.
        self.freefit_max_ratio_spin = QDoubleSpinBox()
        self.freefit_max_ratio_spin.setRange(1.0, 4.0)
        self.freefit_max_ratio_spin.setSingleStep(0.25)
        self.freefit_max_ratio_spin.setDecimals(2)
        self.freefit_max_ratio_spin.setValue(
            float(pp_cfg.get("freefit_max_ratio", DEFAULT_FREEFIT_MAX_RATIO))
        )
        self.freefit_max_ratio_spin.wheelEvent = lambda e: e.ignore()
        self.freefit_max_ratio_spin.setToolTip(t("preprocess_freefit_max_ratio_tip"))
        img_form.addRow(
            self._field_label("freefit_max_ratio", t("preprocess_freefit_max_ratio")),
            self.freefit_max_ratio_spin,
        )

        img_box.setLayout(img_form)
        form_layout.addWidget(img_box)

        text_box = QGroupBox(t("preprocess_text_caching"))
        text_form = QFormLayout()
        self.shuffle_spin = QSpinBox()
        self.shuffle_spin.setRange(0, 64)
        self.shuffle_spin.setValue(
            int(settings.get("caption_shuffle_variants", DEFAULT_TE_SHUFFLE_VARIANTS))
        )
        # Block scroll-wheel changes (matches gui/__init__.py::_widget convention).
        self.shuffle_spin.wheelEvent = lambda e: e.ignore()
        text_form.addRow(
            self._field_label(
                "caption_shuffle_variants",
                t("preprocess_caption_shuffle_variants"),
            ),
            self.shuffle_spin,
        )

        self.dropout_edit = QLineEdit(
            f"{float(settings.get('caption_tag_dropout_rate', DEFAULT_TE_TAG_DROPOUT)):g}"
        )
        text_form.addRow(
            self._field_label(
                "caption_tag_dropout_rate",
                t("preprocess_caption_tag_dropout_rate"),
            ),
            self.dropout_edit,
        )

        self.caption_correct_order_chk = QCheckBox(t("preprocess_caption_correct_order"))
        self.caption_correct_order_chk.setToolTip(t("preprocess_caption_correct_order_tip"))
        self.caption_correct_order_chk.setChecked(DEFAULT_CAPTION_CORRECT_ORDER)
        text_form.addRow(
            self._field_label(
                "caption_correct_order",
                t("preprocess_caption_correct_order"),
            ),
            self.caption_correct_order_chk,
        )

        self.caption_insert_no_artist_chk = QCheckBox(
            t("preprocess_caption_insert_no_artist")
        )
        self.caption_insert_no_artist_chk.setToolTip(
            t("preprocess_caption_insert_no_artist_tip")
        )
        self.caption_insert_no_artist_chk.setChecked(DEFAULT_CAPTION_INSERT_NO_ARTIST)
        text_form.addRow(
            self._field_label(
                "caption_insert_no_artist",
                t("preprocess_caption_insert_no_artist"),
            ),
            self.caption_insert_no_artist_chk,
        )

        self.caption_trigger_word_edit = QLineEdit(DEFAULT_CAPTION_TRIGGER_WORD)
        self.caption_trigger_word_edit.setPlaceholderText("@trigger")
        self.caption_trigger_word_edit.setToolTip(t("preprocess_caption_trigger_word_tip"))
        text_form.addRow(
            self._field_label(
                "caption_trigger_word",
                t("preprocess_caption_trigger_word"),
            ),
            self.caption_trigger_word_edit,
        )

        self.caption_trigger_at_front_chk = QCheckBox(
            t("preprocess_caption_trigger_at_front")
        )
        self.caption_trigger_at_front_chk.setToolTip(
            t("preprocess_caption_trigger_at_front_tip")
        )
        self.caption_trigger_at_front_chk.setChecked(DEFAULT_CAPTION_TRIGGER_AT_FRONT)
        text_form.addRow(
            self._field_label(
                "caption_trigger_at_front",
                t("preprocess_caption_trigger_at_front"),
            ),
            self.caption_trigger_at_front_chk,
        )
        text_box.setLayout(text_form)
        form_layout.addWidget(text_box)

        sam_box = QGroupBox(t("preprocess_masking_sam"))
        sam_outer = QVBoxLayout()

        sam_form = QFormLayout()
        self.run_sam_mask_chk = QCheckBox(t("preprocess_run_sam_mask"))
        self.run_sam_mask_chk.setToolTip(t("preprocess_run_sam_mask_tip"))
        self.run_sam_mask_chk.setChecked(
            bool(settings.get("run_sam_mask", DEFAULT_RUN_SAM_MASK))
        )
        sam_form.addRow(
            self._field_label("run_sam_mask", t("preprocess_run_sam_mask")),
            self.run_sam_mask_chk,
        )
        # Stored in sam_mask.yaml but scopes BOTH backends — masking.py reads
        # it and forwards --path-pattern to SAM and MIT alike.
        self.mask_path_pattern_edit = QLineEdit(mask_path_pattern)
        self.mask_path_pattern_edit.setPlaceholderText("*")
        self.mask_path_pattern_edit.setToolTip(t("preprocess_mask_path_pattern_tip"))
        sam_form.addRow(
            self._field_label("mask_path_pattern", t("preprocess_mask_path_pattern")),
            self.mask_path_pattern_edit,
        )
        sam_outer.addLayout(sam_form)

        # One card per rule: routes a subset of images (by path_pattern) to its own prompt set; matching rules compose.
        self._rule_cards: list[_RuleCard] = []
        self._rules_layout = QVBoxLayout()
        self._rules_layout.setContentsMargins(0, 0, 0, 0)
        sam_outer.addLayout(self._rules_layout)
        for rule in sam_rules:
            self._add_rule_card(rule)

        self.add_rule_btn = QPushButton(t("preprocess_sam_add_rule"))
        self.add_rule_btn.setToolTip(t("preprocess_sam_add_rule_tip"))
        self.add_rule_btn.clicked.connect(lambda: self._add_rule_card())
        sam_outer.addWidget(self.add_rule_btn)

        sam_box.setLayout(sam_outer)
        form_layout.addWidget(sam_box)

        mit_box = QGroupBox(t("preprocess_masking_mit"))
        mit_form = QFormLayout()
        self.run_mit_mask_chk = QCheckBox(t("preprocess_run_mit_mask"))
        self.run_mit_mask_chk.setToolTip(t("preprocess_run_mit_mask_tip"))
        self.run_mit_mask_chk.setChecked(
            bool(settings.get("run_mit_mask", DEFAULT_RUN_MIT_MASK))
        )
        mit_form.addRow(
            self._field_label("run_mit_mask", t("preprocess_run_mit_mask")),
            self.run_mit_mask_chk,
        )

        self.mit_threshold_edit = QLineEdit(
            f"{float(settings.get('mit_text_threshold', DEFAULT_MIT_TEXT_THRESHOLD)):g}"
        )
        mit_form.addRow(
            self._field_label("mit_text_threshold", t("preprocess_mit_threshold")),
            self.mit_threshold_edit,
        )

        self.mit_dilate_spin = QSpinBox()
        self.mit_dilate_spin.setRange(0, 64)
        self.mit_dilate_spin.setValue(
            int(settings.get("mit_dilate", DEFAULT_MIT_DILATE))
        )
        self.mit_dilate_spin.wheelEvent = lambda e: e.ignore()
        mit_form.addRow(
            self._field_label("mit_dilate", t("preprocess_dilate")),
            self.mit_dilate_spin,
        )
        mit_box.setLayout(mit_form)
        form_layout.addWidget(mit_box)

        form_layout.addStretch()
        scroll.setWidget(form_host)
        hsplit.addWidget(scroll)

        # Same QTextBrowser style as ConfigTab's explain panel so the look matches across tabs.
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

        self._connect_dirty_signals()
        self._clear_dirty()

    def _lazy_init(self) -> None:
        self._refresh_status()
        # Re-bind to a preprocess/mask job still running from a previous GUI session or the CLI so closing+reopening re-attaches.
        self._try_reattach()

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
        settings = _load_settings()
        pp_cfg = _load_preprocess_toml()

        # path_scope is layered on top at submit time by _gui_scoped_paths, so this field shows/edits the *unscoped* root, not the scoped run path.
        source_dir = meta.get(
            "source_image_dir",
            pp_cfg.get("source_image_dir", DEFAULT_SOURCE_IMAGE_DIR),
        )
        path_scope = meta.get("path_scope", "")

        target_res = meta.get(
            "target_res", pp_cfg.get("target_res", DEFAULT_TARGET_RES)
        )
        resize_bucket_resos = meta.get(
            "resize_bucket_resos",
            pp_cfg.get("resize_bucket_resos", DEFAULT_RESIZE_BUCKET_RESOS),
        )
        resize_crop_anchor = meta.get(
            "resize_crop_anchor",
            pp_cfg.get("resize_crop_anchor", DEFAULT_RESIZE_CROP_ANCHOR),
        )
        resize_crop_margins = meta.get(
            "resize_crop_margins",
            pp_cfg.get("resize_crop_margins", DEFAULT_RESIZE_CROP_MARGINS),
        )
        freefit_max_ratio = meta.get(
            "freefit_max_ratio",
            pp_cfg.get("freefit_max_ratio", DEFAULT_FREEFIT_MAX_RATIO),
        )
        path_pattern = meta.get(
            "preprocess_path_pattern", DEFAULT_PREPROCESS_PATH_PATTERN
        )
        drop_lowres = meta.get(
            "drop_lowres_images",
            pp_cfg.get("drop_lowres_images", DEFAULT_DROP_LOWRES_IMAGES),
        )
        min_pixels = meta.get(
            "min_pixels", pp_cfg.get("min_pixels", DEFAULT_MIN_PIXELS)
        )
        shuffle_variants = meta.get(
            "caption_shuffle_variants",
            settings.get("caption_shuffle_variants", DEFAULT_TE_SHUFFLE_VARIANTS),
        )
        tag_dropout = meta.get(
            "caption_tag_dropout_rate",
            settings.get("caption_tag_dropout_rate", DEFAULT_TE_TAG_DROPOUT),
        )
        caption_correct_order = meta.get(
            "caption_correct_order", DEFAULT_CAPTION_CORRECT_ORDER
        )
        caption_insert_no_artist = meta.get(
            "caption_insert_no_artist", DEFAULT_CAPTION_INSERT_NO_ARTIST
        )
        caption_trigger_word = meta.get(
            "caption_trigger_word", DEFAULT_CAPTION_TRIGGER_WORD
        )
        caption_trigger_at_front = meta.get(
            "caption_trigger_at_front", DEFAULT_CAPTION_TRIGGER_AT_FRONT
        )
        run_sam_mask = meta.get(
            "run_sam_mask",
            settings.get("run_sam_mask", DEFAULT_RUN_SAM_MASK),
        )
        run_mit_mask = meta.get(
            "run_mit_mask",
            settings.get("run_mit_mask", DEFAULT_RUN_MIT_MASK),
        )
        mask_path_pattern = meta.get(
            "mask_path_pattern",
            _load_sam_yaml().get("path_pattern") or DEFAULT_MASK_PATH_PATTERN,
        )
        mask_rules = meta.get("mask_rules")
        if not isinstance(mask_rules, list):
            mask_rules = _load_rules(_load_sam_yaml())
        mit_threshold = meta.get(
            "mit_text_threshold",
            settings.get("mit_text_threshold", DEFAULT_MIT_TEXT_THRESHOLD),
        )
        mit_dilate = meta.get(
            "mit_dilate",
            settings.get("mit_dilate", DEFAULT_MIT_DILATE),
        )

        self._loading_variant = True
        try:
            self.source_dir_edit.setText(str(source_dir or DEFAULT_SOURCE_IMAGE_DIR))
            self.path_scope_edit.setText(str(path_scope or ""))
            self.preprocess_path_pattern_edit.setText(str(path_pattern or "*"))
            self.drop_lowres_chk.setChecked(bool(drop_lowres))
            self.min_pixels_spin.setValue(int(min_pixels))
            self.min_pixels_spin.setEnabled(self.drop_lowres_chk.isChecked())
            self._set_target_res_widget(target_res)
            self.target_res_widget.set_bucket_resos(resize_bucket_resos)
            self._set_resize_crop_anchor(resize_crop_anchor)
            self._set_resize_crop_margins(resize_crop_margins)
            self.freefit_max_ratio_spin.setValue(float(freefit_max_ratio))
            self.shuffle_spin.setValue(int(shuffle_variants))
            self.dropout_edit.setText(f"{float(tag_dropout):g}")
            self.caption_correct_order_chk.setChecked(bool(caption_correct_order))
            self.caption_insert_no_artist_chk.setChecked(bool(caption_insert_no_artist))
            self.caption_trigger_word_edit.setText(str(caption_trigger_word or ""))
            self.caption_trigger_at_front_chk.setChecked(bool(caption_trigger_at_front))
            self.run_sam_mask_chk.setChecked(bool(run_sam_mask))
            self.mask_path_pattern_edit.setText(str(mask_path_pattern or "*"))
            self._set_rule_cards(mask_rules)
            self.run_mit_mask_chk.setChecked(bool(run_mit_mask))
            self.mit_threshold_edit.setText(f"{float(mit_threshold):g}")
            self.mit_dilate_spin.setValue(int(mit_dilate))
        finally:
            self._loading_variant = False
        if hasattr(self, "status_lbl"):
            self._refresh_status()
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
        return {k: meta[k] for k in _GUI_PREPROCESS_KEYS if k in meta}

    def _set_target_res_widget(self, values) -> None:
        if values is None:
            selected = {1024}
        elif isinstance(values, (list, tuple, set)):
            selected = {int(v) for v in values}
        else:
            selected = {int(values)}
        for edge, checkbox in self.target_res_widget._boxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(edge in selected)
            checkbox.blockSignals(False)
        self.target_res_widget.refresh_bucket_enabled()

    def _set_resize_crop_anchor(self, value) -> None:
        self.resize_crop_anchor_widget.set_value(value, emit=False)

    def _margin_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 95.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix("%")
        # Tighter arrow zone than the global theme so the compact field stays snug.
        spin.setStyleSheet(
            "QDoubleSpinBox { padding-right: 4px; }"
            "QDoubleSpinBox::up-button { width: 16px; }"
            "QDoubleSpinBox::down-button { width: 16px; margin-right: 16px; }"
        )
        # Right-align on the line-edit, after the stylesheet (setAlignment on the spinbox doesn't survive the styled rebuild).
        line_edit = spin.lineEdit()
        line_edit.setTextMargins(0, 0, 0, 0)
        spin.setFixedWidth(84)
        spin.wheelEvent = lambda e: e.ignore()
        spin.valueChanged.connect(lambda _value: self.persist_target_res())
        return spin

    def _set_resize_crop_margins(self, value) -> None:
        margins = normalize_crop_margins(value)
        for key, spin in (
            ("top", self.resize_margin_top_spin),
            ("right", self.resize_margin_right_spin),
            ("bottom", self.resize_margin_bottom_spin),
            ("left", self.resize_margin_left_spin),
        ):
            spin.blockSignals(True)
            spin.setValue(float(margins[key]))
            spin.blockSignals(False)

    def _resize_crop_margins(self) -> dict[str, float]:
        return {
            "top": float(self.resize_margin_top_spin.value()),
            "right": float(self.resize_margin_right_spin.value()),
            "bottom": float(self.resize_margin_bottom_spin.value()),
            "left": float(self.resize_margin_left_spin.value()),
        }

    def _set_rule_cards(self, rules: list[dict]) -> None:
        for card in list(self._rule_cards):
            self._rules_layout.removeWidget(card)
            card.deleteLater()
        self._rule_cards.clear()
        for rule in rules or [{}]:
            self._add_rule_card(rule)
        self._update_remove_buttons()

    # _update_save_button is overridden below because the Save button has a different name + localized label.
    def _connect_dirty_signals(self) -> None:
        for widget in (
            self.source_dir_edit,
            self.path_scope_edit,
            self.preprocess_path_pattern_edit,
            self.drop_lowres_chk,
            self.min_pixels_spin,
            self.target_res_widget,
            self.resize_crop_anchor_widget,
            self.resize_margin_top_spin,
            self.resize_margin_right_spin,
            self.resize_margin_bottom_spin,
            self.resize_margin_left_spin,
            self.freefit_max_ratio_spin,
            self.shuffle_spin,
            self.dropout_edit,
            self.caption_correct_order_chk,
            self.caption_insert_no_artist_chk,
            self.caption_trigger_word_edit,
            self.caption_trigger_at_front_chk,
            self.run_sam_mask_chk,
            self.mask_path_pattern_edit,
            self.run_mit_mask_chk,
            self.mit_threshold_edit,
            self.mit_dilate_spin,
        ):
            self._connect_dirty_signal(widget)

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

    def _field_label(self, key: str, text_str: str) -> ClickableLabel:
        """Build a ClickableLabel that shows this field's help when clicked."""
        help_text = preprocess_field_help(key)
        return make_field_label(
            text_str,
            style=f"color:{tok('text')}; text-decoration: underline dotted;",
            on_click=lambda _k=key, _h=help_text, _t=text_str: self._show_field_help(
                _t, _h
            ),
        )

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
                f"{html.escape(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:{tok('text_dim')}; font-style:italic;'>"
                f"{html.escape(t('no_help_available'))}</p>"
            )
        self._explain.setHtml("".join(parts))

    def _refresh_status(self) -> None:
        snapshot = self.preprocess_config_snapshot()

        def _path(key: str, default: Path) -> Path:
            raw = snapshot.get(key)
            if not raw:
                return default
            p = Path(str(raw))
            return p if p.is_absolute() else ROOT / p

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
            _path("resized_image_dir", RESIZED_DIR),
            path_pattern,
        )
        caches = count_preprocess_caches(
            _path("lora_cache_dir", LORA_CACHE_DIR),
            path_pattern,
            pe_encoder=str(snapshot.get("repa_encoder") or "pe_spatial").strip()
            or None,
        )
        mask_n = _count_masks(_path("mask_dir", MASK_DIR), path_pattern)
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
        self.status_lbl.setText("  |  ".join(lines))

    def _open_dataset_dir(self) -> None:
        """Open the post_image_dataset/ folder (resized + caches) in the OS file manager."""
        snapshot = self.preprocess_config_snapshot()
        raw = snapshot.get("resized_image_dir")
        resized = (
            (Path(str(raw)) if Path(str(raw)).is_absolute() else ROOT / str(raw))
            if raw
            else RESIZED_DIR
        )
        # post_image_dataset/ is the parent of resized/ (and lora/, masks/).
        target = resized.parent
        if not target.is_dir():
            target = ROOT
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _snapshot_path(
        self, snapshot: dict[str, object], key: str, default: Path
    ) -> Path:
        raw = snapshot.get(key)
        if not raw:
            return default
        p = Path(str(raw))
        return p if p.is_absolute() else ROOT / p

    def _normalize_scope_or_warn(self) -> str | None:
        raw = self.path_scope_edit.text().strip()
        if not raw:
            return ""
        scope = ConfigTab._normalize_path_scope(raw)
        if scope is None:
            QMessageBox.warning(
                self,
                t("error"),
                t("preprocess_invalid_path_scope", value=raw),
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
            try:
                target.relative_to(ROOT)
            except ValueError:
                QMessageBox.warning(
                    self,
                    t("error"),
                    t("preprocess_clear_scope_cache_outside_root", path=str(target)),
                )
                return
            if target == ROOT:
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

    def _add_rule_card(self, rule: dict | None = None) -> None:
        card = _RuleCard(rule or {}, self._show_field_help)
        card.changed.connect(self._mark_dirty)
        card.removed.connect(self._remove_rule_card)
        self._rule_cards.append(card)
        self._rules_layout.addWidget(card)
        self._update_remove_buttons()
        if not self._loading_variant:
            self._mark_dirty()

    def _remove_rule_card(self, card: _RuleCard) -> None:
        if len(self._rule_cards) <= 1:
            return  # keep at least one rule
        self._rule_cards.remove(card)
        self._rules_layout.removeWidget(card)
        card.deleteLater()
        self._update_remove_buttons()
        self._mark_dirty()

    def _update_remove_buttons(self) -> None:
        # A lone rule can't be removed (would leave an empty config).
        sole = len(self._rule_cards) <= 1
        for card in self._rule_cards:
            card.remove_btn.setEnabled(not sole)

    def _collect_rules(self) -> list[dict] | None:
        """Serialize every rule card, or None if one fails validation."""
        rules: list[dict] = []
        for card in self._rule_cards:
            try:
                rules.append(card.to_dict())
            except ValueError as bad_threshold:
                QMessageBox.warning(
                    self,
                    t("error"),
                    t(
                        "preprocess_invalid_float",
                        field=t("preprocess_sam_threshold"),
                        value=str(bad_threshold),
                    ),
                )
                return None
        return rules

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

    def preprocess_env(self) -> dict[str, str]:
        """Environment values consumed by ``tasks.py preprocess``."""
        return {
            "CAPTION_SHUFFLE_VARIANTS": str(int(self.shuffle_spin.value())),
            "CAPTION_TAG_DROPOUT_RATE": self.dropout_edit.text().strip(),
            "CAPTION_CORRECT_ORDER": (
                "1" if self.caption_correct_order_chk.isChecked() else "0"
            ),
            "CAPTION_INSERT_NO_ARTIST": (
                "1" if self.caption_insert_no_artist_chk.isChecked() else "0"
            ),
            "CAPTION_TRIGGER_WORD": self.caption_trigger_word_edit.text().strip(),
            "CAPTION_TRIGGER_AT_FRONT": (
                "1" if self.caption_trigger_at_front_chk.isChecked() else "0"
            ),
            "PREPROCESS_PATH_PATTERN": (
                self.preprocess_path_pattern_edit.text().strip()
                or DEFAULT_PREPROCESS_PATH_PATTERN
            ),
        }

    def preprocess_overrides(self) -> dict[str, object]:
        """Flat config overrides that should be captured in preprocess snapshots."""
        return {
            "drop_lowres_images": self.drop_lowres_chk.isChecked(),
            "min_pixels": int(self.min_pixels_spin.value()),
            "target_res": self.target_res_widget.value(),
            "resize_bucket_resos": self.target_res_widget.bucket_resos(),
            "resize_crop_anchor": self.resize_crop_anchor_widget.value(),
            "resize_crop_margins": self._resize_crop_margins(),
            # Free-fit is the only resize mode; only the max-ratio clamp is tunable.
            "freefit_max_ratio": float(self.freefit_max_ratio_spin.value()),
            "caption_correct_order": self.caption_correct_order_chk.isChecked(),
            "caption_insert_no_artist": self.caption_insert_no_artist_chk.isChecked(),
            "caption_trigger_word": self.caption_trigger_word_edit.text().strip(),
            "caption_trigger_at_front": self.caption_trigger_at_front_chk.isChecked(),
        }

    def preprocess_config_snapshot(self) -> dict[str, object]:
        """Full preprocess config snapshot captured at GUI submit time.

        The concrete paths come from the selected GUI method plus ``path_scope``.
        ``preprocess_path_pattern`` is not written into the flat config because
        training should not see that GUI-only preprocess filter; it is forwarded
        to tasks.py via ``PREPROCESS_PATH_PATTERN`` instead.
        """
        variant = self._variant or "lora"
        merged, _ = merged_gui_variant_preset(variant, "default")
        # Seed the base source dir from the (editable) field before scoping so
        # path_scope is appended onto the user-chosen root, not the hard default.
        source_dir = self.source_dir_edit.text().strip()
        if source_dir:
            merged["source_image_dir"] = source_dir
        path_scope = self.path_scope_edit.text().strip()
        if path_scope:
            merged["path_scope"] = path_scope
        else:
            merged.pop("path_scope", None)
        snapshot = ConfigTab._gui_scoped_paths(copy.deepcopy(merged))
        snapshot.update(self.preprocess_overrides())
        for key in (
            "base_config",
            "dataset_config",
            "variant",
            "method",
            "preset",
            "methods_subdir",
            "path_scope",
            "preprocess_path_pattern",
        ):
            snapshot.pop(key, None)

        def _clean(value):
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items() if v is not None}
            if isinstance(value, list):
                return [_clean(v) for v in value if v is not None]
            if isinstance(value, Path):
                return str(value)
            return value

        return _clean(snapshot)

    def persist_target_res(self) -> None:
        """Mark the GUI profile dirty when the tier selection changes.

        ConfigTab auto-chain/queue calls ``persist_preprocess_inputs`` before it
        submits, so it still captures the latest tiers without silently saving
        them while the user is editing.
        """
        if not self._loading_variant:
            self._mark_dirty()

    def persist_preprocess_inputs(self) -> bool:
        """Persist cache-building inputs used by ConfigTab's auto-chain/queue.

        This intentionally excludes mask-only settings so an invalid mask
        threshold cannot block a plain cache build.
        """
        return self._save_variant_preprocess_meta(validate_dropout=True)

    def _save_variant_preprocess_meta(
        self,
        *,
        validate_dropout: bool,
        include_mask: bool = False,
        rules: list[dict] | None = None,
        mit_threshold: float | None = None,
    ) -> bool:
        if not self._variant:
            return True
        dropout = (
            self._parse_float(
                self.dropout_edit.text().strip(),
                t("preprocess_caption_tag_dropout_rate"),
            )
            if validate_dropout
            else None
        )
        if validate_dropout and dropout is None:
            return False
        if dropout is None:
            try:
                dropout = float(self.dropout_edit.text().strip())
            except ValueError:
                dropout = DEFAULT_TE_TAG_DROPOUT

        path = variant_path(self._variant)
        data = _load(path)
        meta = data.get("variant")
        if not isinstance(meta, dict):
            meta = {}

        # Persist on the variant only when it diverges from the preprocess.toml default, so a plain checkout keeps an empty meta.
        pp_default = str(
            _load_preprocess_toml().get("source_image_dir", DEFAULT_SOURCE_IMAGE_DIR)
        )
        source_dir = self.source_dir_edit.text().strip() or pp_default
        if source_dir == pp_default:
            meta.pop("source_image_dir", None)
        else:
            meta["source_image_dir"] = source_dir

        scope = self._normalize_scope_or_warn()
        if scope is None:
            return False
        if scope:
            meta["path_scope"] = scope
        else:
            meta.pop("path_scope", None)

        path_pattern = (
            self.preprocess_path_pattern_edit.text().strip()
            or DEFAULT_PREPROCESS_PATH_PATTERN
        )
        if path_pattern == DEFAULT_PREPROCESS_PATH_PATTERN:
            meta.pop("preprocess_path_pattern", None)
        else:
            meta["preprocess_path_pattern"] = path_pattern

        drop_lowres = self.drop_lowres_chk.isChecked()
        if drop_lowres == DEFAULT_DROP_LOWRES_IMAGES:
            meta.pop("drop_lowres_images", None)
        else:
            meta["drop_lowres_images"] = drop_lowres

        min_pixels = int(self.min_pixels_spin.value())
        if min_pixels == DEFAULT_MIN_PIXELS:
            meta.pop("min_pixels", None)
        else:
            meta["min_pixels"] = min_pixels

        target_res = self.target_res_widget.value()
        # Kept explicit even when it matches the default: it also affects train-time compile-cache sizing, so the GUI profile must show the exact tiers it will use.
        meta["target_res"] = target_res

        bucket_resos = self.target_res_widget.bucket_resos()
        if bucket_resos:
            meta["resize_bucket_resos"] = bucket_resos
        else:
            meta.pop("resize_bucket_resos", None)

        crop_anchor = self.resize_crop_anchor_widget.value()
        if crop_anchor == DEFAULT_RESIZE_CROP_ANCHOR:
            meta.pop("resize_crop_anchor", None)
        else:
            meta["resize_crop_anchor"] = crop_anchor

        crop_margins = normalize_crop_margins(self._resize_crop_margins())
        if any(value > 0 for value in crop_margins.values()):
            meta["resize_crop_margins"] = crop_margins
        else:
            meta.pop("resize_crop_margins", None)

        # Free-fit is the only resize mode; max_ratio is preprocess-only and
        # persisted only when it differs from the default.
        freefit_max_ratio = float(self.freefit_max_ratio_spin.value())
        if freefit_max_ratio != float(DEFAULT_FREEFIT_MAX_RATIO):
            meta["freefit_max_ratio"] = freefit_max_ratio
        else:
            meta.pop("freefit_max_ratio", None)

        shuffle = int(self.shuffle_spin.value())
        if shuffle == DEFAULT_TE_SHUFFLE_VARIANTS:
            meta.pop("caption_shuffle_variants", None)
        else:
            meta["caption_shuffle_variants"] = shuffle

        if float(dropout) == float(DEFAULT_TE_TAG_DROPOUT):
            meta.pop("caption_tag_dropout_rate", None)
        else:
            meta["caption_tag_dropout_rate"] = float(dropout)

        caption_correct_order = self.caption_correct_order_chk.isChecked()
        if caption_correct_order == DEFAULT_CAPTION_CORRECT_ORDER:
            meta.pop("caption_correct_order", None)
        else:
            meta["caption_correct_order"] = caption_correct_order

        caption_insert_no_artist = self.caption_insert_no_artist_chk.isChecked()
        if caption_insert_no_artist == DEFAULT_CAPTION_INSERT_NO_ARTIST:
            meta.pop("caption_insert_no_artist", None)
        else:
            meta["caption_insert_no_artist"] = caption_insert_no_artist

        trigger_word = self.caption_trigger_word_edit.text().strip()
        if trigger_word == DEFAULT_CAPTION_TRIGGER_WORD:
            meta.pop("caption_trigger_word", None)
        else:
            meta["caption_trigger_word"] = trigger_word

        trigger_at_front = self.caption_trigger_at_front_chk.isChecked()
        if trigger_at_front == DEFAULT_CAPTION_TRIGGER_AT_FRONT:
            meta.pop("caption_trigger_at_front", None)
        else:
            meta["caption_trigger_at_front"] = trigger_at_front

        if include_mask:
            mask_path_pattern = (
                self.mask_path_pattern_edit.text().strip() or DEFAULT_MASK_PATH_PATTERN
            )
            meta["run_sam_mask"] = self.run_sam_mask_chk.isChecked()
            meta["run_mit_mask"] = self.run_mit_mask_chk.isChecked()
            meta["mask_path_pattern"] = mask_path_pattern
            meta["mask_rules"] = list(rules or [])
            meta["mit_text_threshold"] = float(
                DEFAULT_MIT_TEXT_THRESHOLD if mit_threshold is None else mit_threshold
            )
            meta["mit_dilate"] = int(self.mit_dilate_spin.value())

        if meta:
            data["variant"] = meta
        else:
            data.pop("variant", None)
        _save(path, data)
        return True

    def _save_all(self) -> bool:
        """Validate and persist every form value. Returns True on success."""
        dropout = self._parse_float(
            self.dropout_edit.text().strip(),
            t("preprocess_caption_tag_dropout_rate"),
        )
        if dropout is None:
            return False
        mit_threshold = self._parse_float(
            self.mit_threshold_edit.text().strip(),
            t("preprocess_mit_threshold"),
        )
        if mit_threshold is None:
            return False

        rules = self._collect_rules()
        if rules is None:
            return False

        # The mask path pattern is read + persisted inside _save_variant_preprocess_meta (include_mask=True); nothing to pass here.
        if not self._save_variant_preprocess_meta(
            validate_dropout=False,
            include_mask=True,
            rules=rules,
            mit_threshold=mit_threshold,
        ):
            return False
        self._clear_dirty()
        return True

    def _save_all_clicked(self) -> None:
        if self._save_all():
            QMessageBox.information(self, t("saved"), t("preprocess_settings_saved"))

    def _is_running(self) -> bool:
        return self._job_id is not None

    def _make_run_button(self, label: str, style: str, run_cb) -> QToolButton:
        """Build a split Run button: main action runs now, dropdown queues it.

        ``run_cb`` is a ``_run_*`` handler taking a keyword-only ``queue`` flag;
        the dropdown calls it with ``queue=True`` (submit without attaching).
        """
        btn = QToolButton()
        # SplitButtonStyle (set before the stylesheet) widens the dropdown indicator + paints its divider/tint. The style must outlive the button, so stash a ref.
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
        # Unified caching step — `tasks.py preprocess` chains resize → VAE-latent cache → text-embedding cache. Only the TE knobs (shuffle/dropout) are GUI-tunable, so the form stays TE-only.
        if not self._save_all():
            return
        snapshot = self.preprocess_config_snapshot()
        self._submit(
            label="preprocess",
            argv=["tasks.py", "preprocess"],
            extra_env=self.preprocess_env(),
            config_snapshot=snapshot,
            attach=not queue,
        )

    def _run_pe(self, *, queue: bool = False) -> None:
        # Cache vision-encoder (PE) features; the encoder follows the variant's `repa_encoder` (pe_spatial default; `pe` = PE-Core for CMMD / DCW v4) so the button matches whatever a use_repa run reads.
        if not self._save_all():
            return
        variant = self._variant or "lora"
        merged, _ = merged_gui_variant_preset(variant, "default")
        encoder = (
            str(merged.get("repa_encoder") or "pe_spatial").strip() or "pe_spatial"
        )
        task = "preprocess-pe-spatial" if encoder == "pe_spatial" else "preprocess-pe"
        snapshot = self.preprocess_config_snapshot()
        self._submit(
            label="preprocess-pe",
            argv=["tasks.py", task],
            extra_env=self.preprocess_env(),
            config_snapshot=snapshot,
            attach=not queue,
        )

    def _run_mask(self, *, queue: bool = False) -> None:
        # GUI mask settings ride as env snapshots so queued jobs keep their queued-with values; direct CLI usage still falls back to ``configs/sam_mask.yaml``.
        if not self._save_all():
            return
        rules = self._collect_rules()
        if rules is None:
            return
        mask_path_pattern = (
            self.mask_path_pattern_edit.text().strip() or DEFAULT_MASK_PATH_PATTERN
        )
        run_sam = self.run_sam_mask_chk.isChecked()
        run_mit = self.run_mit_mask_chk.isChecked()
        if not (run_sam or run_mit):
            QMessageBox.warning(self, t("error"), t("preprocess_mask_nothing_enabled"))
            return
        # Carry the scoped paths so masking only scans the configured subfolder — without the snapshot the task falls back to the unscoped resized/ and re-masks every group each run.
        snapshot = self.preprocess_config_snapshot()
        self._submit(
            label="mask",
            argv=["tasks.py", "mask"],
            extra_env={
                "MIT_TEXT_THRESHOLD": self.mit_threshold_edit.text().strip(),
                "MIT_DILATE": str(int(self.mit_dilate_spin.value())),
                "RUN_SAM_MASK": "1" if run_sam else "0",
                "RUN_MIT_MASK": "1" if run_mit else "0",
                "SAM_MASK_CONFIG_JSON": json.dumps(
                    {
                        "rules": rules,
                        "path_pattern": mask_path_pattern,
                    },
                    ensure_ascii=False,
                ),
            },
            config_snapshot=snapshot,
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
        """Submit a preprocess/mask job to the daemon.

        The daemon spawns ``python <argv>`` detached and serializes it behind
        any running training job (single GPU). Pre-launch validation
        (``_save_all`` + per-step gating) is the caller's job.

        With ``attach=True`` (the main Run action) this tab takes over its
        log/bar and blocks the Run buttons until the job finishes. With
        ``attach=False`` (the "add to queue" dropdown) the job is submitted
        silently — the Run buttons stay live so the next step / variant can be
        queued, and the job is watched from the Queue tab."""
        if attach and self._is_running():
            QMessageBox.information(self, "", t("preprocess_already_running"))
            return
        if attach:
            # Busy UI + repaint before submit so the tab stays responsive during the daemon auto-start + /health wait on a cold start.
            for btn in self._run_buttons:
                btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
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
                # Main Run starts now; the "add to queue" dropdown holds it paused until the Queue tab's "Start Queue".
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

    def _try_reattach(self) -> None:
        """Bind to a preprocess/mask job still running when the tab first opens.

        Makes "close GUI mid-preprocess → reopen → re-attach" work. Skips a
        training job (that one belongs to the ConfigTab) and stays idle when the
        daemon is down."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable → nothing to attach
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "command":
            return
        # An auto-chain preprocess (tagged ANIMA_CHAIN_TRAIN) belongs to the ConfigTab, which re-claims it so the chain into training stays there. Leave it alone here.
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

        ``replay_log`` reads ``stdout.log`` from the top (re-attach after a GUI
        restart); otherwise pre-existing output is skipped so a fresh launch
        shows only new lines."""
        for btn in self._run_buttons:
            btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._watch_job(job_id, replay_log=replay_log)

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        # Drain trailing stdout before the finish banner; a half-written tqdm fragment is dropped (the bar already reflected it).
        self._drain_job_stdout()
        if self._stdout_buf and not TQDM_RE.search(self._stdout_buf):
            self.log.appendPlainText(self._stdout_buf)
        self._stdout_buf = ""
        job_id = self._job_id
        self._job_id = None
        self._stdout_tailer.reset()
        self._progress_tracker.reset()
        self.log.appendPlainText(gui_daemon.format_finish_banner(job_id, state))
        self._restore_idle_ui()
        self._refresh_status()

    def _restore_idle_ui(self) -> None:
        for btn in self._run_buttons:
            btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop(self) -> None:
        # Abort the daemon job; the poll loop observes 'stopped' and restores the UI. The daemon stays up and advances its queue.
        self._stop_job()

    def cleanup_subprocess(self) -> None:
        """App-shutdown hook. Stops observing but deliberately leaves the daemon
        job alive — it runs detached so a cache build / mask pass survives the
        GUI closing (re-attached on next launch)."""
        self._job_timer.stop()
