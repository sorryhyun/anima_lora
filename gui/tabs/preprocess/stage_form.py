"""``KnobSection``s drawn from ``anime_tools`` stage schemas.

The package already turns each stage's request dataclass into a JSON form
schema (``anime_tools.gui.stages.schema`` — kind / default / choices / help /
group / advanced / gate / bound roots) and a form payload back into the
request's argv with the request's own validation (``build_argv``). This
module is the trainer-side half of that seam
(``docs/proposal/gui_preprocess_from_anime_tools.md``):

- **Qt-free** (top of the file): :func:`load_stage_schemas`,
  :func:`visible_fields`, :func:`argv_for`, the field → :class:`Knob` adapter
  (:func:`knob_for`), the ``preprocess.toml`` seeding
  (:func:`seeded_defaults`), the ``[variant.stages.*]`` persistence helpers
  (:func:`persistable_values` / :func:`merge_stages_into_meta` /
  :func:`load_stage_values`) and the i18n hooks (:func:`label_for` /
  :func:`help_for` / :func:`choice_label_for`) reading the overlay in
  ``gui/explanations/guides/<lang>/_stage_fields.json`` keyed
  ``<stage_id>.<dest>``, falling back to the schema's English.
- **Qt** (bottom, guarded): :class:`StageFormSection`, a ``KnobSection`` that
  renders one schema — bool → checkbox, int/float → spin (free text when the
  default is ``None``), str → line edit, enum → combo, list → csv line edit,
  path → line edit + ``…`` chooser (dir vs file by ``path_kind``), ``gate`` →
  ``enabled_by``, ``advanced`` fields inside a fold — plus the trainer-native
  rows a section carries beside the stage's own (the chain gate, the
  dataset roots): those are ``knobs.py`` rows, kept in ``knob_widgets`` and
  read back by :meth:`StageFormSection.knob_values`.

The trainer owns a few dests per stage that its chain fills at run time
(:data:`TRAINER_FIELDS`: ``recursive``, the variant sidecar knobs, the
tokenizer dirs, a replay report); they are hidden like the bound roots.
Everything the form shows travels to ``tasks.py`` as ``{dest: value}`` under
:data:`STAGE_VALUES_ENV`, where ``scripts/tasks/_common.request_from_form``
fills the roots and builds the request through ``build_argv``.
"""

from __future__ import annotations

from typing import Any

from gui.tabs.preprocess.knobs import KNOBS_BY_KEY, STAGES_KEY, Knob

__all__ = [
    "ALWAYS_PERSIST",
    "STAGE_IDS",
    "STAGE_VALUES_ENV",
    "TRAINER_FIELDS",
    "argv_for",
    "choice_label_for",
    "help_for",
    "is_visible",
    "knob_for",
    "label_for",
    "load_stage_schemas",
    "load_stage_values",
    "merge_stages_into_meta",
    "persistable_values",
    "seeded_defaults",
    "visible_fields",
]

STAGE_IDS: tuple[str, ...] = ("resize", "autotag", "correct", "masks_sam")
"""The stages the Preprocessing tab draws a form for."""

STAGE_VALUES_ENV = "PREPROCESS_STAGES_JSON"
"""The env var a preprocess / mask job receives its stage forms in: a JSON
``{stage_id: {dest: value}}`` (``masks_sam``: a *list* of dicts, one per rule
card). Read by ``scripts/tasks/_common.gui_stage_values``."""

# ``--apply`` is the run bar's Dry-run / Apply choice, not a form row: the
# package's ``build_argv`` sets it from its ``apply`` keyword regardless of
# what the form sent, so a checkbox for it would be dead weight.
_RUN_BAR_FIELDS: frozenset[str] = frozenset({"apply"})

# ``path`` fields that are globs, not files — no ``…`` chooser.
_NO_BROWSE: frozenset[str] = frozenset({"path_pattern"})

TRAINER_FIELDS: dict[str, frozenset[str]] = {
    # stage id → dests the trainer's chain fills itself (hidden from the form,
    # like a bound root). ``scripts/tasks/preprocess.py`` / ``masking.py`` set
    # them on the request after ``build_argv``.
    "resize": frozenset({"recursive", "copy_captions", "skip"}),
    # The in-pipeline autotag always applies; a replay is a CLI affair.
    "autotag": frozenset({"from_report"}),
    # The variant sidecar knobs are the TextCachingSection's (trainer-side TE
    # cache), the tokenizer dirs are resolved on the trainer side.
    "correct": frozenset(
        {
            "recursive",
            "caption_shuffle_variants",
            "caption_tag_dropout_rate",
            "caption_tag_randomize_rate",
            "qwen3",
            "t5_tokenizer_path",
        }
    ),
    "masks_sam": frozenset({"recursive"}),
}

SHOWN_BOUND: dict[str, frozenset[str]] = {
    # stage id → bound dests this trainer form shows anyway. A SAM rule card
    # carries its own scope: ``make mask`` runs each card as one request with
    # the card's ``path_pattern`` as the run's setting.
    "masks_sam": frozenset({"path_pattern"}),
}

FIELD_ORDER: dict[str, tuple[str, ...]] = {
    # Rows listed here come first, in this order; the rest follow in schema order.
    "resize": (
        "min_pixels",
        "target_res",
        "resize_crop_anchor",
        "resize_crop_margins",
        "freefit_max_ratio",
        "overwrite",
        "workers",
    ),
    "masks_sam": ("path_pattern", "prompts", "focus_prompts", "threshold", "dilate"),
}

ALWAYS_PERSIST: dict[str, frozenset[str]] = {
    # Written to ``[variant.stages.<id>]`` even at the seeded default: the
    # tiers are what a user looks for in the profile (see
    # ``test_preprocess_tab_persists_default_target_res_to_variant``).
    "resize": frozenset({"target_res"}),
}

_PP_SEEDS: dict[str, dict[str, str]] = {
    # stage id → {dest: configs/preprocess.toml key}. The user-owned TOML is
    # the lowest-priority default for these (the CLI's ``make preprocess``
    # reads the same keys), so the form opens on what the CLI would run.
    "resize": {
        d: d
        for d in (
            "target_res",
            "min_pixels",
            "freefit_max_ratio",
            "resize_crop_anchor",
            "resize_crop_margins",
        )
    },
    "autotag": {
        "mode": "caption_autotag_mode",
        "min_confidence": "caption_autotag_min_confidence",
    },
    "correct": {
        d: d
        for d in (
            "caption_insert_no_artist",
            "caption_trigger_word",
            "caption_trigger_at_front",
            "caption_drop_groups",
        )
    },
}


# -- Qt-free ---------------------------------------------------------------


def load_stage_schemas() -> dict[str, dict]:
    """Every stage's schema keyed by id, built in-process from the request
    classes (~0.1 s, torch-free). Lazy import so the module stays cheap for a
    GUI that never opens a stage form."""
    from anime_tools.gui.stages import load_schemas

    return load_schemas()


def is_visible(field: dict, stage_id: str | None = None) -> bool:
    """Whether a schema field is the form's to show. Bound fields (dataset
    roots, ⚙ Settings values, report / mask tails) are filled by ``build_argv``
    from the trainer's roots; ``auto`` fields (``--device``) are never sent;
    a ``PANEL_FIELDS`` dest (``overridable``) is bound *and* shown; the
    trainer-owned dests (:data:`TRAINER_FIELDS`) are filled by the chain."""
    dest = field["dest"]
    if field.get("auto") or dest in _RUN_BAR_FIELDS:
        return False
    sid = stage_id or ""
    if dest in TRAINER_FIELDS.get(sid, frozenset()):
        return False
    if field.get("overridable") or dest in SHOWN_BOUND.get(sid, frozenset()):
        return True
    return not (
        field.get("root")
        or field.get("setting")
        or field.get("report")
        or field.get("mask")
    )


def visible_fields(schema: dict) -> list[dict]:
    """The schema's form rows — :data:`FIELD_ORDER` first, then declaration
    order — ``advanced`` and ``gate`` kept on each dict for the renderer."""
    sid = schema.get("id", "")
    fields = [f for f in schema.get("fields") or () if is_visible(f, sid)]
    order = FIELD_ORDER.get(sid)
    if not order:
        return fields
    rank = {dest: i for i, dest in enumerate(order)}
    return sorted(fields, key=lambda f: rank.get(f["dest"], len(rank)))


def argv_for(
    schema: dict,
    values: dict[str, Any],
    *,
    roots: dict[str, str] | None = None,
    settings: dict[str, str] | None = None,
    report_root: str | None = None,
    mask_root: str | None = None,
    apply: bool = False,
) -> list[str]:
    """``{dest: value}`` → the stage's argv, via the package's ``build_argv``:
    coerced field by field, read into the request (its ``__post_init__``
    raises ``ValueError`` here, before any job is submitted) and spelled back
    with ``to_argv()`` (a value at the request default is omitted).

    ``roots`` maps the package's root names (``src`` / ``dst`` / ``masks`` /
    ``master`` / ``out``) to the trainer's directories; ``settings`` carries
    ``path_pattern`` / ``tagger_dir`` / ``checkpoint`` / ``prompt_embed``;
    ``report_root`` / ``mask_root`` are joined to each stage's own tail.
    """
    from anime_tools.gui.stages import build_argv

    return build_argv(
        schema,
        values,
        apply=apply,
        roots=roots,
        settings=settings,
        report_root=report_root,
        mask_root=mask_root,
    )


# -- i18n (overlay keyed <stage_id>.<dest>, English from the schema) ---------


def _overlay(field: dict, stage_id: str) -> dict:
    if not stage_id:
        return {}
    from gui.explanations import stage_field

    return stage_field(f"{stage_id}.{field['dest']}") or {}


def label_for(field: dict, stage_id: str = "") -> str:
    """The label: the overlay's for the current language, else the schema's
    ``label`` (the flag, or the dest for a ``store_false`` switch so a ticked
    box always means *on*) humanised."""
    text = _overlay(field, stage_id).get("label")
    if text:
        return str(text)
    raw = str(field.get("label") or field["dest"]).lstrip("-")
    return raw.replace("-", " ").replace("_", " ")


def help_for(field: dict, stage_id: str = "") -> str:
    """The help: the overlay's, else the flag's ``--help`` text."""
    text = _overlay(field, stage_id).get("help")
    return str(text) if text else str(field.get("help") or "")


def choice_label_for(field: dict, choice: Any, stage_id: str = "") -> str:
    """An enum item's label: the overlay's ``choices`` map, else the value."""
    choices = _overlay(field, stage_id).get("choices") or {}
    return str(choices.get(str(choice)) or choice)


# -- field → Knob ------------------------------------------------------------


def knob_for(field: dict, section: str = "") -> Knob:
    """A schema field as the ``Knob`` row ``_section.read_widget`` /
    ``set_widget`` dispatch on.

    ``kind`` maps onto the knob table's vocabulary; an int/float whose default
    is ``None`` becomes a ``str`` knob because it is rendered as free text and
    ``build_argv`` coerces the text (blank → the request default). The knob
    ``default`` is what an empty editor falls back to, so ``None`` is spelled
    as the empty string. An enum / list is a ``str`` knob here (the combo's
    item data / the csv text); the section keeps the field's own kind.
    """
    kind = field["kind"]
    default = field.get("default")
    gate = field.get("gate")
    if kind == "bool":
        knob_kind, knob_default = "bool", bool(default)
    elif kind in ("int", "float") and default is not None:
        knob_kind, knob_default = kind, default
    elif kind == "list":
        knob_kind, knob_default = "str", ""
    else:
        knob_kind, knob_default = "str", "" if default is None else str(default)
    return Knob(
        field["dest"],
        section or "stage",
        knob_kind,  # type: ignore[arg-type]
        knob_default,
        enabled_by=gate if gate and gate != field["dest"] else None,
    )


def _csv_to_list(text: Any) -> list[str]:
    if isinstance(text, (list, tuple)):
        return [str(x).strip() for x in text if str(x).strip()]
    return [s.strip() for s in str(text or "").split(",") if s.strip()]


# -- defaults / persistence --------------------------------------------------


def _normalise_seed(dest: str, value: Any) -> Any:
    """``preprocess.toml`` spellings → form values (margins dict → 4-list)."""
    if dest == "resize_crop_margins" and isinstance(value, dict):
        from library.preprocess.resize_preview import normalize_crop_margins

        m = normalize_crop_margins(value)
        return [float(m[s]) for s in ("top", "right", "bottom", "left")]
    if dest == "target_res":
        return [
            int(v) for v in (value if isinstance(value, (list, tuple)) else [value])
        ]
    return value


def seeded_defaults(schema: dict, pp_cfg: dict | None = None) -> dict[str, Any]:
    """``{dest: default}`` for the form: the schema's defaults (argv
    spellings, ``None`` for a required / unset field) with
    ``configs/preprocess.toml`` on top for the dests it names
    (:data:`_PP_SEEDS`). ``correct.no_correct`` seeds from the TOML's
    ``caption_correct_order`` inverted, so the GUI default stays the CLI's
    (no reordering unless the TOML says so)."""
    sid = schema.get("id", "")
    pp = pp_cfg or {}
    out: dict[str, Any] = {f["dest"]: f.get("default") for f in visible_fields(schema)}
    for dest, key in _PP_SEEDS.get(sid, {}).items():
        if dest in out and pp.get(key) is not None:
            out[dest] = _normalise_seed(dest, pp[key])
    if sid == "correct" and "no_correct" in out:
        out["no_correct"] = not bool(pp.get("caption_correct_order", False))
    return out


def persistable_values(
    stage_id: str, values: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    """What ``[variant.stages.<stage_id>]`` keeps: the form's values minus the
    ones at their seeded default (:data:`ALWAYS_PERSIST` excepted) and minus
    ``None`` (TOML has no null). A saved table therefore reads as "what this
    variant changed", like the flat ``[variant]`` keys."""
    keep = ALWAYS_PERSIST.get(stage_id, frozenset())
    out: dict[str, Any] = {}
    for dest, value in values.items():
        if value is None:
            continue
        if dest not in keep and _same(value, defaults.get(dest)):
            continue
        out[dest] = value
    return out


def _same(a: Any, b: Any) -> bool:
    if a in ("", None) and b in ("", None):
        return True  # a blank editor and an unset default both mean "default"
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def load_stage_values(meta: dict, stage_id: str, defaults: dict[str, Any]):
    """The form's values for a variant: its ``[variant.stages.<stage_id>]``
    over the seeded defaults. For ``masks_sam`` a list of card dicts (the
    saved ``[[variant.stages.masks_sam]]`` rows, or ``None`` when the variant
    has none — the caller seeds the cards then)."""
    stages = meta.get(STAGES_KEY) if isinstance(meta, dict) else None
    saved = stages.get(stage_id) if isinstance(stages, dict) else None
    if stage_id == "masks_sam":
        if not isinstance(saved, list):
            return None
        return [{**defaults, **card} for card in saved if isinstance(card, dict)]
    out = dict(defaults)
    if isinstance(saved, dict):
        out.update(saved)
    return out


def merge_stages_into_meta(
    meta: dict,
    stage_values: dict[str, Any],
    defaults: dict[str, dict[str, Any]],
    *,
    include_mask: bool,
) -> dict:
    """Write every stage form into ``meta["stages"]`` (elided per
    :func:`persistable_values`); a stage whose table ends up empty is
    dropped, and so is the ``stages`` key when nothing is left. ``masks_sam``
    only moves when ``include_mask``. Returns ``meta``."""
    stages = meta.get(STAGES_KEY)
    if not isinstance(stages, dict):
        stages = {}
    for sid, values in stage_values.items():
        if sid == "masks_sam":
            if not include_mask:
                continue
            cards = [
                persistable_values(sid, card, defaults.get(sid, {}))
                for card in (values or [])
            ]
            # A card at every default is still a card (its prompts are the
            # default's); keep the list even when each row is empty.
            stages[sid] = cards
            continue
        kept = persistable_values(sid, values or {}, defaults.get(sid, {}))
        if kept:
            stages[sid] = kept
        else:
            stages.pop(sid, None)
    if stages:
        meta[STAGES_KEY] = stages
    else:
        meta.pop(STAGES_KEY, None)
    return meta


# -- Qt ----------------------------------------------------------------------

try:
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QPushButton,
        QWidget,
    )
except ImportError:  # pragma: no cover — headless import of the Qt-free half
    StageFormSection = None  # type: ignore[assignment]
else:
    from gui._paths import ROOT
    from gui.i18n import t
    from gui.tabs.preprocess._section import (
        KnobSection,
        checkbox,
        connect_change,
        dspin,
        line,
        no_wheel,
        read_widget,
        set_widget,
        spin,
    )
    from gui.theme import tok
    from gui.widgets import ClickableLabel, make_field_label

    __all__ += ["StageFormSection"]

    _INT_RANGE = (-(2**31), 2**31 - 1)
    _FLOAT_RANGE = (-1e9, 1e9)

    class StageFormSection(KnobSection):
        """One ``QGroupBox`` form over a stage schema.

        Same surface as every other section — :meth:`add_knob`, ``widgets``,
        :meth:`values` / :meth:`set_values`, the ``changed`` signal and the
        ``enabled_by`` gate — but keyed by the schema's dests, with a private
        knob table (:func:`knob_for`) in place of ``KNOBS_BY_KEY``. A field's
        help comes from the i18n overlay, else the schema.

        Trainer-native rows (a ``knobs.py`` key) sit in the same form through
        :meth:`add_trainer_knob`; they are kept in ``knob_widgets`` and read by
        :meth:`knob_values`, so the stage's ``values()`` stays exactly what
        ``build_argv`` takes. ``gate`` names a bool trainer knob rendered as
        the first row that enables / disables every stage row (a chain gate);
        ``gated_by`` maps a stage dest onto a trainer knob that enables it.
        """

        def __init__(
            self,
            schema: dict,
            help_cb,
            *,
            title: str | None = None,
            values: dict | None = None,
            defaults: dict | None = None,
            gate: str | None = None,
            gated_by: dict[str, str] | None = None,
        ):
            self.schema = schema
            self.stage_id: str = schema["id"]
            self._fields: dict[str, dict] = {
                f["dest"]: f for f in visible_fields(schema)
            }
            self._defaults: dict[str, Any] = dict(defaults or {})
            self._knobs: dict[str, Knob] = {}
            self.knob_widgets: dict[str, QWidget] = {}
            self._gate_key = gate
            self._gated_by: dict[str, str] = dict(gated_by or {})
            self._current_form: QFormLayout | None = None
            self.advanced_toggle: QCheckBox | None = None
            self.advanced_box: QGroupBox | None = None
            self.browse_buttons: dict[str, QPushButton] = {}
            super().__init__(title or schema.get("title") or self.stage_id, help_cb)
            if self._defaults:
                self.set_values(self._defaults)
            if values:
                self.set_values(values)

        # -- i18n hooks -----------------------------------------------------

        def label_for(self, field: dict) -> str:
            return label_for(field, self.stage_id)

        def help_for(self, field: dict) -> str:
            return help_for(field, self.stage_id)

        # -- building ------------------------------------------------------

        def field_label(self, key: str, text: str) -> ClickableLabel:
            if key in self._fields:
                help_text = self.help_for(self._fields[key])
            else:
                from gui.explanations import preprocess_field_help

                help_text = preprocess_field_help(key) or ""
            return make_field_label(
                text,
                style=f"color:{tok('text')}; text-decoration: underline dotted;",
                on_click=lambda _t=text, _h=help_text: self._help_cb(_t, _h),
            )

        def add_knob(
            self,
            key: str,
            widget: QWidget,
            label: str,
            *,
            tooltip: str | None = None,
            field=None,
        ) -> QWidget:
            """Register ``widget`` as the editor for schema dest ``key`` and
            add its row to the current form (the Advanced fold while it is
            being built). ``KeyError`` for a dest the schema does not show,
            mirroring the base's knob-table check."""
            if key not in self._fields:
                raise KeyError(f"{key}: not a visible field of stage {self.stage_id}")
            fd = self._fields[key]
            self._knobs[key] = knob_for(fd, self.stage_id)
            widget.setToolTip(tooltip if tooltip is not None else self.help_for(fd))
            form = self._current_form or self.form
            form.addRow(
                self.field_label(key, label), field if field is not None else widget
            )
            self.widgets[key] = widget
            connect_change(widget, self.changed.emit)
            return widget

        def add_trainer_knob(
            self,
            key: str,
            widget: QWidget,
            label: str,
            *,
            tooltip: str | None = None,
            field=None,
        ) -> QWidget:
            """A ``knobs.py`` row beside the stage's own (a chain gate, a
            dataset root). Lives in ``knob_widgets`` / :meth:`knob_values`."""
            if key not in KNOBS_BY_KEY:
                raise KeyError(f"{key}: not in the preprocess knob table")
            from gui.explanations import preprocess_field_help

            widget.setToolTip(
                tooltip if tooltip is not None else (preprocess_field_help(key) or "")
            )
            form = self._current_form or self.form
            form.addRow(
                self.field_label(key, label), field if field is not None else widget
            )
            self.knob_widgets[key] = widget
            connect_change(widget, self.changed.emit)
            return widget

        def _build(self) -> None:
            if self._gate_key:
                gate = checkbox(t(f"preprocess_{self._gate_key}"))
                gate.setChecked(bool(KNOBS_BY_KEY[self._gate_key].default))
                self.add_trainer_knob(
                    self._gate_key,
                    gate,
                    t(f"preprocess_{self._gate_key}"),
                    tooltip=t(f"preprocess_{self._gate_key}_tip"),
                )
            self._build_prefix()
            basic = [f for f in self._fields.values() if not f.get("advanced")]
            advanced = [f for f in self._fields.values() if f.get("advanced")]
            for fd in basic:
                self._add_field(fd)
            if advanced:
                self.advanced_toggle = QCheckBox(t("advanced_section"))
                self.advanced_box = QGroupBox()
                self.advanced_box.setFlat(True)
                adv_form = QFormLayout()
                self.advanced_box.setLayout(adv_form)
                self._current_form = adv_form
                try:
                    for fd in advanced:
                        self._add_field(fd)
                finally:
                    self._current_form = None
                self.advanced_box.setVisible(False)
                self.advanced_toggle.toggled.connect(self.advanced_box.setVisible)
                self.form.addRow(self.advanced_toggle)
                self.form.addRow(self.advanced_box)
            self._build_suffix()

        def _build_prefix(self) -> None:
            """Hook: trainer rows before the stage's fields."""

        def _build_suffix(self) -> None:
            """Hook: trainer rows after the stage's fields."""

        def _add_field(self, fd: dict) -> None:
            widget = self._make_widget(fd)
            browse = fd.get("path") and fd["dest"] not in _NO_BROWSE
            composite = self._with_browse(widget, fd) if browse else None
            self.add_knob(fd["dest"], widget, self.label_for(fd), field=composite)

        def _make_widget(self, fd: dict) -> QWidget:
            """The editor for one field; a subclass overrides this for a
            domain widget (the tier row, the crop-anchor picker)."""
            kind, default = fd["kind"], fd.get("default")
            if kind == "bool":
                w = checkbox(self.label_for(fd))
                w.setChecked(bool(default))
                return w
            if kind == "int":
                if default is None:
                    return line(placeholder="auto")
                return spin(*_INT_RANGE, int(default))
            if kind == "float":
                if default is None:
                    return line(placeholder="auto")
                return dspin(*_FLOAT_RANGE, float(default), step=0.05, decimals=4)
            if kind == "enum":
                combo = QComboBox()
                for choice in fd.get("choices") or ():
                    combo.addItem(
                        choice_label_for(fd, choice, self.stage_id), str(choice)
                    )
                if default is not None:
                    combo.setCurrentIndex(max(combo.findData(str(default)), 0))
                return no_wheel(combo)
            if kind == "list":
                text = ", ".join(str(x) for x in default) if default else ""
                return line(text, placeholder="a, b, c")
            return line("" if default is None else str(default))

        def _with_browse(self, edit: QWidget, fd: dict) -> QWidget:
            """The editor plus a ``…`` chooser (dir vs file per ``path_kind``).
            A neutral ``QPushButton`` — not an action button, no inline color."""
            wrap = QWidget()
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(edit, 1)
            btn = QPushButton("…")
            btn.setFixedWidth(28)
            btn.setToolTip(self.label_for(fd))
            btn.clicked.connect(lambda: self._browse(edit, fd))
            row.addWidget(btn)
            self.browse_buttons[fd["dest"]] = btn
            return wrap

        def _browse(self, edit, fd: dict) -> None:
            start = edit.text().strip() or str(ROOT)
            title = self.label_for(fd)
            if fd.get("path_kind") == "dir":
                chosen = QFileDialog.getExistingDirectory(self, title, start)
            else:
                chosen, _ = QFileDialog.getOpenFileName(self, title, start)
            if chosen:
                edit.setText(chosen)

        # -- gating (private knob table + trainer gates) --------------------

        def _gate_widget(self, key: str):
            """The checkbox gating ``key``: a stage dest's schema gate, or the
            trainer knob ``gated_by`` names."""
            if key in self._knobs:
                gate = self._knobs[key].enabled_by
                if gate and gate in self.widgets:
                    return self.widgets[gate]
            knob_gate = self._gated_by.get(key)
            if knob_gate and knob_gate in self.knob_widgets:
                return self.knob_widgets[knob_gate]
            return None

        def _wire_enabled_by(self) -> None:
            for key, widget in self.widgets.items():
                gate = self._gate_widget(key)
                if gate is not None:
                    gate.toggled.connect(widget.setEnabled)
            if self._gate_key and self._gate_key in self.knob_widgets:
                self.knob_widgets[self._gate_key].toggled.connect(
                    lambda _on: self._sync_enabled()
                )
            self._sync_enabled()

        def _sync_enabled(self) -> None:
            chain_on = True
            if self._gate_key and self._gate_key in self.knob_widgets:
                chain_on = self.knob_widgets[self._gate_key].isChecked()
            for key, widget in self.widgets.items():
                gate = self._gate_widget(key)
                on = chain_on and (gate is None or gate.isChecked())
                widget.setEnabled(on)
                btn = self.browse_buttons.get(key)
                if btn is not None:
                    btn.setEnabled(on)
            if self.advanced_toggle is not None:
                self.advanced_toggle.setEnabled(chain_on)
            for key, widget in self.knob_widgets.items():
                if key == self._gate_key:
                    continue
                widget.setEnabled(chain_on)

        # -- values ----------------------------------------------------------

        def fields(self) -> dict[str, dict]:
            return dict(self._fields)

        def field_kind(self, key: str) -> str:
            return self._fields[key]["kind"]

        def values(self) -> dict[str, object]:
            """``{dest: value}`` in the shape ``argv_for`` takes and
            ``[variant.stages.<id>]`` stores: a ``list`` field's csv editor
            becomes a list, a free-text numeric stays text (``build_argv``
            coerces it; blank → the request default), everything else the
            base's widget read."""
            out: dict[str, object] = {}
            for key, widget in self.widgets.items():
                knob = self._knobs[key]
                value = read_widget(knob, widget)
                if self.field_kind(key) == "list" and isinstance(value, str):
                    value = _csv_to_list(value)
                out[key] = value
            return out

        def set_values(self, values: dict) -> None:
            for key, widget in self.widgets.items():
                if key not in values:
                    continue
                knob = self._knobs[key]
                value = values[key]
                if value is None:
                    value = knob.default
                elif isinstance(value, (list, tuple)):
                    value = ", ".join(str(x) for x in value)
                set_widget(knob, widget, value)
            self._sync_enabled()

        def knob_values(self) -> dict[str, object]:
            """The trainer-native rows (:meth:`add_trainer_knob`)."""
            return {
                key: read_widget(KNOBS_BY_KEY[key], w)
                for key, w in self.knob_widgets.items()
            }

        def set_knob_values(self, values: dict) -> None:
            for key, widget in self.knob_widgets.items():
                if key in values:
                    set_widget(KNOBS_BY_KEY[key], widget, values[key])
            self._sync_enabled()

        def argv(
            self,
            *,
            roots: dict[str, str] | None = None,
            settings: dict[str, str] | None = None,
            report_root: str | None = None,
            mask_root: str | None = None,
            apply: bool = False,
        ) -> list[str]:
            """This form's current values as the stage's argv (:func:`argv_for`)."""
            return argv_for(
                self.schema,
                self.values(),
                roots=roots,
                settings=settings,
                report_root=report_root,
                mask_root=mask_root,
                apply=apply,
            )
