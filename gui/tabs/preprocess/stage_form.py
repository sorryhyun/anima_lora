"""A ``KnobSection`` drawn from an ``anime_tools`` stage schema (P0 pilot).

The package already turns each stage's request dataclass into a JSON form
schema (``anime_tools.gui.stages.schema`` — kind / default / choices / help /
group / advanced / gate / bound roots) and a form payload back into the
request's argv with the request's own validation (``build_argv``). This
module is the trainer-side half of that seam, proposed in
``docs/proposal/gui_preprocess_from_anime_tools.md``:

- **Qt-free** (top of the file): :func:`load_stage_schemas`,
  :func:`visible_fields`, :func:`argv_for`, plus the field → :class:`Knob`
  adapter (:func:`knob_for`) and the ``label_for`` / ``help_for`` defaults an
  i18n overlay will replace. Importable headless, like ``knobs.py``.
- **Qt** (bottom, guarded): :class:`StageFormSection`, a ``KnobSection`` that
  renders one schema — bool → checkbox, int/float → spin (free text when the
  default is ``None``), str → line edit, enum → combo, list → csv line edit,
  path → line edit + ``…`` chooser (dir vs file by ``path_kind``), ``gate`` →
  ``enabled_by``, ``advanced`` fields inside a fold — and reads/writes values
  through the base class's widget-type dispatch.

Not wired into ``tab.py``; the live tab still runs off ``knobs.py``.
"""

from __future__ import annotations

from typing import Any

from gui.tabs.preprocess.knobs import Knob

__all__ = [
    "argv_for",
    "help_for",
    "is_visible",
    "knob_for",
    "label_for",
    "load_stage_schemas",
    "visible_fields",
]

# ``--apply`` is the run bar's Dry-run / Apply choice, not a form row: the
# package's ``build_argv`` sets it from its ``apply`` keyword regardless of
# what the form sent, so a checkbox for it would be dead weight.
_RUN_BAR_FIELDS: frozenset[str] = frozenset({"apply"})


# -- Qt-free ---------------------------------------------------------------


def load_stage_schemas() -> dict[str, dict]:
    """Every stage's schema keyed by id, built in-process from the request
    classes (~0.1 s, torch-free). Lazy import so the module stays cheap for a
    GUI that never opens a stage form."""
    from anime_tools.gui.stages import load_schemas

    return load_schemas()


def is_visible(field: dict) -> bool:
    """Whether a schema field is the form's to show. Bound fields (dataset
    roots, ⚙ Settings values, report / mask tails) are filled by ``build_argv``
    from the trainer's roots; ``auto`` fields (``--device``) are never sent;
    a ``PANEL_FIELDS`` dest (``overridable``) is bound *and* shown."""
    if field.get("auto") or field["dest"] in _RUN_BAR_FIELDS:
        return False
    if field.get("overridable"):
        return True
    return not (
        field.get("root")
        or field.get("setting")
        or field.get("report")
        or field.get("mask")
    )


def visible_fields(schema: dict) -> list[dict]:
    """The schema's form rows in declaration order, ``advanced`` and ``gate``
    kept on each dict for the renderer."""
    return [f for f in schema.get("fields") or () if is_visible(f)]


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


def label_for(field: dict) -> str:
    """The English label: the schema's ``label`` (the flag, or the dest for a
    ``store_false`` switch so a ticked box always means *on*) humanised. The
    i18n overlay slots in here — keyed ``<stage_id>.<dest>``."""
    raw = str(field.get("label") or field["dest"]).lstrip("-")
    return raw.replace("-", " ").replace("_", " ")


def help_for(field: dict) -> str:
    """The English help (the flag's ``--help`` text). Same overlay hook."""
    return str(field.get("help") or "")


def knob_for(field: dict, section: str = "") -> Knob:
    """A schema field as the ``Knob`` row ``_section.read_widget`` /
    ``set_widget`` dispatch on.

    ``kind`` maps onto the knob table's vocabulary (enum → ``choice``, list →
    ``str_list``); an int/float whose default is ``None`` becomes a ``str``
    knob because it is rendered as free text and ``build_argv`` coerces the
    text (blank → the request default). The knob ``default`` is what an empty
    editor falls back to, so ``None`` is spelled as the empty string.
    """
    kind = field["kind"]
    default = field.get("default")
    gate = field.get("gate")
    if kind == "bool":
        knob_kind, knob_default = "bool", bool(default)
    elif kind in ("int", "float"):
        if default is None:
            knob_kind, knob_default = "str", ""
        else:
            knob_kind, knob_default = kind, default
    elif kind == "enum":
        knob_kind, knob_default = "choice", "" if default is None else str(default)
    elif kind == "list":
        knob_kind, knob_default = "str_list", ""
    else:
        knob_kind, knob_default = "str", "" if default is None else str(default)
    return Knob(
        field["dest"],
        section or "stage",
        knob_kind,  # type: ignore[arg-type]
        knob_default,
        choices=tuple(str(c) for c in (field.get("choices") or ())),
        enabled_by=gate if gate and gate != field["dest"] else None,
    )


def _csv_to_list(text: Any) -> list[str]:
    if isinstance(text, (list, tuple)):
        return [str(x).strip() for x in text if str(x).strip()]
    return [s.strip() for s in str(text or "").split(",") if s.strip()]


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
        help comes from the schema, not ``_preprocess_fields.json``.
        """

        def __init__(
            self,
            schema: dict,
            help_cb,
            *,
            title: str | None = None,
            values: dict | None = None,
        ):
            self.schema = schema
            self.stage_id: str = schema["id"]
            self._fields: dict[str, dict] = {
                f["dest"]: f for f in visible_fields(schema)
            }
            self._knobs: dict[str, Knob] = {}
            self._current_form: QFormLayout | None = None
            self.advanced_toggle: QCheckBox | None = None
            self.advanced_box: QGroupBox | None = None
            super().__init__(title or schema.get("title") or self.stage_id, help_cb)
            if values:
                self.set_values(values)

        # -- i18n hooks (English today; an overlay keyed stage_id.dest later) --

        def label_for(self, field: dict) -> str:
            return label_for(field)

        def help_for(self, field: dict) -> str:
            return help_for(field)

        # -- building ------------------------------------------------------

        def field_label(self, key: str, text: str) -> ClickableLabel:
            help_text = self.help_for(self._fields[key])
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

        def _build(self) -> None:
            basic = [f for f in self._fields.values() if not f.get("advanced")]
            advanced = [f for f in self._fields.values() if f.get("advanced")]
            for fd in basic:
                self._add_field(fd)
            if not advanced:
                return
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

        def _add_field(self, fd: dict) -> None:
            widget = self._make_widget(fd)
            composite = self._with_browse(widget, fd) if fd.get("path") else None
            self.add_knob(fd["dest"], widget, self.label_for(fd), field=composite)

        def _make_widget(self, fd: dict) -> QWidget:
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
                    combo.addItem(str(choice), str(choice))
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
            self.browse_buttons = getattr(self, "browse_buttons", {})
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

        # -- gating (private knob table) -----------------------------------

        def _wire_enabled_by(self) -> None:
            for key, widget in self.widgets.items():
                gate = self._knobs[key].enabled_by
                if gate and gate in self.widgets:
                    self.widgets[gate].toggled.connect(widget.setEnabled)
            self._sync_enabled()

        def _sync_enabled(self) -> None:
            for key, widget in self.widgets.items():
                gate = self._knobs[key].enabled_by
                if gate and gate in self.widgets:
                    widget.setEnabled(self.widgets[gate].isChecked())

        # -- values ----------------------------------------------------------

        def fields(self) -> dict[str, dict]:
            return dict(self._fields)

        def values(self) -> dict[str, object]:
            """``{dest: value}`` in the shape ``argv_for`` takes: csv list
            editors become lists, everything else the base's widget read."""
            out: dict[str, object] = {}
            for key, widget in self.widgets.items():
                knob = self._knobs[key]
                value = read_widget(knob, widget)
                out[key] = _csv_to_list(value) if knob.kind == "str_list" else value
            return out

        def set_values(self, values: dict) -> None:
            for key, widget in self.widgets.items():
                if key not in values:
                    continue
                knob = self._knobs[key]
                value = values[key]
                if knob.kind == "str_list" and isinstance(value, (list, tuple)):
                    value = ", ".join(str(x) for x in value)
                set_widget(knob, widget, value)
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
