"""Masking section: the ``run_sam_mask`` chain gate + one ``masks_sam`` stage
form per rule card. Each card is a complete ``SamMaskRequest`` minus the
roots ``make mask`` fills (the resized tree, a tempdir per card, the merge
into ``mask_dir``); its ``path_pattern`` is the card's own scope. Persists as
``[[variant.stages.masks_sam]]``."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QVBoxLayout

from gui.i18n import t
from gui.tabs.preprocess._section import KnobSection, checkbox
from gui.tabs.preprocess.knobs import DEFAULT_RUN_SAM_MASK
from gui.tabs.preprocess.stage_form import StageFormSection


class _RuleCard(StageFormSection):
    """One SAM mask rule: the ``masks_sam`` form with a Remove button."""

    def __init__(self, schema: dict, help_cb, *, defaults: dict, values: dict | None):
        super().__init__(
            schema,
            help_cb,
            title=t("preprocess_sam_rule"),
            defaults=defaults,
            values=values,
        )

    def _build_suffix(self) -> None:
        self.remove_btn = QPushButton(t("preprocess_sam_remove_rule"))
        self.form.addRow("", self.remove_btn)


class SamMaskSection(KnobSection):
    """Run-SAM toggle + one ``_RuleCard`` per rule; ``make mask`` runs each
    card as its own SAM pass and unions them (pixel-min merge)."""

    def __init__(
        self,
        schema: dict,
        help_cb,
        *,
        settings: dict,
        defaults: dict,
        initial_rules: list[dict],
    ):
        self.schema = schema
        self._settings = settings
        self._defaults = defaults
        self._initial_rules = initial_rules
        super().__init__(t("preprocess_masking_sam"), help_cb)

    def _build(self) -> None:
        on = checkbox(t("preprocess_run_sam_mask"))
        on.setChecked(bool(self._settings.get("run_sam_mask", DEFAULT_RUN_SAM_MASK)))
        self.add_knob(
            "run_sam_mask",
            on,
            t("preprocess_run_sam_mask"),
            tooltip=t("preprocess_run_sam_mask_tip"),
        )
        on.toggled.connect(self._sync_cards_enabled)
        # Rule cards sit below the form in a vertical stack.
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self._rule_cards: list[_RuleCard] = []
        self._rules_layout = QVBoxLayout()
        self._rules_layout.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._rules_layout)
        self.add_rule_btn = QPushButton(t("preprocess_sam_add_rule"))
        self.add_rule_btn.setToolTip(t("preprocess_sam_add_rule_tip"))
        self.add_rule_btn.clicked.connect(lambda: self.add_rule_card())
        outer.addWidget(self.add_rule_btn)
        self.form.addRow(outer)
        for rule in self._initial_rules:
            self.add_rule_card(rule)

    # -- rule cards ---------------------------------------------------------

    @property
    def rule_cards(self) -> list[_RuleCard]:
        return self._rule_cards

    def add_rule_card(self, values: dict | None = None) -> _RuleCard:
        card = _RuleCard(
            self.schema, self._help_cb, defaults=self._defaults, values=values
        )
        card.changed.connect(self.changed.emit)
        card.remove_btn.clicked.connect(
            lambda _c=False, c=card: self.remove_rule_card(c)
        )
        self._rule_cards.append(card)
        self._rules_layout.addWidget(card)
        self._update_remove_buttons()
        self._sync_cards_enabled()
        self.changed.emit()
        return card

    def remove_rule_card(self, card: _RuleCard) -> None:
        if len(self._rule_cards) <= 1:
            return  # keep at least one rule
        self._rule_cards.remove(card)
        self._rules_layout.removeWidget(card)
        card.deleteLater()
        self._update_remove_buttons()
        self.changed.emit()

    def set_rule_cards(self, rules: list[dict]) -> None:
        for card in list(self._rule_cards):
            self._rules_layout.removeWidget(card)
            card.deleteLater()
        self._rule_cards.clear()
        for rule in rules or [None]:
            self.add_rule_card(rule)
        self._update_remove_buttons()

    def _update_remove_buttons(self) -> None:
        # A lone rule can't be removed (would leave an empty config).
        sole = len(self._rule_cards) <= 1
        for card in self._rule_cards:
            card.remove_btn.setEnabled(not sole)

    def _sync_cards_enabled(self, *_args) -> None:
        on = self.widgets["run_sam_mask"].isChecked()
        for card in self._rule_cards:
            card.setEnabled(on)
        self.add_rule_btn.setEnabled(on)

    # -- values -------------------------------------------------------------

    def stage_values(self) -> list[dict]:
        """One ``masks_sam`` value dict per card, in order."""
        return [card.values() for card in self._rule_cards]

    def set_stage_values(self, rules: list[dict] | None) -> None:
        self.set_rule_cards(rules or [])
