"""Caption sections drawn from the ``anime_tools`` stage schemas: the
auto-tag box (the ``autotag`` stage behind its chain gate — it creates a
caption from nothing and runs right after resize) and the caption-editing
box (the ``correct`` stage: order correction, @no-artist, trigger word, tag
groups to drop — plus the ``position`` chain gate, whose stage has no form
here yet)."""

from __future__ import annotations

from gui.i18n import t
from gui.tabs.preprocess._section import checkbox
from gui.tabs.preprocess.stage_form import StageFormSection


class AutotagSection(StageFormSection):
    def __init__(self, schema: dict, help_cb, *, defaults: dict, gate_on: bool):
        self._gate_on = gate_on
        super().__init__(
            schema,
            help_cb,
            title=t("preprocess_caption_autotag_box"),
            defaults=defaults,
            gate="caption_autotag",
        )

    def _build(self) -> None:
        super()._build()
        self.knob_widgets["caption_autotag"].setChecked(self._gate_on)

    def mode(self) -> str:
        return str(self.widgets["mode"].currentData() or "missing")


class CaptionEditingSection(StageFormSection):
    def __init__(self, schema: dict, help_cb, *, defaults: dict, position_on: bool):
        self._position_on = position_on
        super().__init__(
            schema,
            help_cb,
            title=t("preprocess_caption_editing"),
            defaults=defaults,
        )

    def _build_suffix(self) -> None:
        # Unlike its neighbours this is a GPU stage (SAM3 + tagger) that rewrites
        # the derived caption in place, so it's off by default and its default
        # resolves through preprocess.toml (Knob.default_from). A chain gate,
        # not a `correct` field — the position stage's own knobs stay at the
        # package default here.
        pos = checkbox(t("preprocess_caption_position_clauses"))
        pos.setChecked(self._position_on)
        self.add_trainer_knob(
            "caption_position_clauses",
            pos,
            t("preprocess_caption_position_clauses"),
            tooltip=t("preprocess_caption_position_clauses_tip"),
        )
