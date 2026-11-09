"""The ``model`` dropdown's items on MOSS forms (variant × int8 / bf16 plan) and the
Train form's recipe dropdown (``configs/moss/*.toml``).

An item's data is the variant, the request field. Changing the plan writes the fields
below; the item follows the int8 field, so tuning a field keeps the item. The recipe
dropdown works the same way over the recipe configs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QSpinBox

from gui.lines.window import tr
from library.env import resolve_under_home
from library.moss.requests import GenerateRequest, TrainRequest

# (item, config, the fields that tell the recipe apart). The first item is the
# fallback when no other recipe's fields match.
RECIPES = (
    ("lora", "configs/moss/lora.toml", ()),
    (
        "trigger_registers",
        "configs/moss/lora_trigger_registers.toml",
        ("delta_rows", "trigger_registers"),
    ),
    ("selfdistill", "configs/moss/selfdistill.toml", ("distill",)),
)
# Kept when a recipe is picked: paths, the variant, and the plan's fields.
_RECIPE_KEEP = {"cache", "output", "model", "model_dir", "trigger"}


# Reference tokens: swap 0 at budget 0.3 OOMs on sanitize's largest sample, this peaks
# at 12.33 GB (six steps, 2026-10-31).
_I2I_INT8 = {"blocks_to_swap": 20, "activation_memory_budget": 0.1}


def train_plans(seed: dict | None = None) -> dict[str, dict]:
    """int8: the request defaults, with ``_I2I_INT8`` on i2i. bf16:
    ``configs/moss/lora_bf16.toml``, under the task's ``seed`` on i2i. Over the keys
    that file sets."""
    config = resolve_under_home("configs/moss/lora_bf16.toml")
    bf16 = tomllib.loads(config.read_text(encoding="utf-8"))
    defaults = TrainRequest()
    int8 = {k: getattr(defaults, k) for k in bf16}
    if seed is not None:
        int8 |= _I2I_INT8
    return {
        "int8": int8,
        "bf16": {k: (seed or {}).get(k, v) for k, v in bf16.items()},
    }


def generate_plans(seed: dict | None = None) -> dict[str, dict]:
    """int8: the request defaults (docs/moss/optimization/int8.md's plan, no block
    swap). bf16: the bf16 DiT and stream, under the task's ``seed`` on i2i."""
    seed = seed or {}
    defaults = GenerateRequest()
    bf16 = {"int8": "off", "stream": "bf16", "blocks_to_swap": 0}
    return {
        "int8": {k: getattr(defaults, k) for k in bf16},
        "bf16": {k: seed.get(k, v) for k, v in bf16.items()},
    }


def model_items(form, plans: dict[str, dict], help_key: str) -> None:
    combo, int8 = form.widgets["model"], form.widgets["int8"]
    variants = [combo.itemText(i) for i in range(combo.count())]
    combo.clear()
    item_plan = {}
    for variant in variants:
        for plan in plans:
            combo.addItem(f"{variant} ({plan})", variant)
            item_plan[f"{variant} ({plan})"] = plan

    def current_plan() -> str:
        return "bf16" if int8.currentText() == "off" else "int8"

    def follow() -> None:
        combo.setCurrentText(f"{combo.currentData()} ({current_plan()})")

    def picked() -> None:
        plan = item_plan[combo.currentText()]
        if plan != current_plan():
            form.set_values(plans[plan])

    follow()
    int8.currentTextChanged.connect(follow)
    combo.activated.connect(picked)
    form.set_help("model", tr(help_key))


def _widget_value(w):
    if isinstance(w, QComboBox):
        return w.currentData() or w.currentText()
    if isinstance(w, QSpinBox):
        return w.value()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    return w.text().strip()


def recipe_knob(form) -> None:
    """A Recipe dropdown on top of the Train form. Picking one writes its config's
    values, and the field default for every key another recipe sets; paths, the
    model and the int8 / bf16 plan stay. The item follows the recipe's own fields."""
    defaults = TrainRequest()
    configs = {
        key: TrainRequest.load_config(resolve_under_home(path))
        for key, path, _ in RECIPES
    }
    keep = _RECIPE_KEEP | set(train_plans()["bf16"])
    keys = {k for c in configs.values() for k in c} - keep

    combo = QComboBox()
    for key, path, _ in RECIPES:
        combo.addItem(f"{tr('recipe_' + key)} ({Path(path).name})", key)

    def follow() -> None:
        current = RECIPES[0][0]
        for key, _, markers in RECIPES[1:]:
            if markers and all(
                _widget_value(form.widgets[m]) == configs[key][m] for m in markers
            ):
                current = key
        combo.setCurrentIndex(combo.findData(current))

    def picked() -> None:
        config = configs[combo.currentData()]
        form.set_values({k: config.get(k, getattr(defaults, k)) for k in keys})
        follow()

    markers = {m for *_, ms in RECIPES for m in ms}
    for m in markers:
        w = form.widgets[m]
        if isinstance(w, QComboBox):
            w.currentTextChanged.connect(follow)
        elif isinstance(w, QSpinBox):
            w.valueChanged.connect(follow)
    combo.activated.connect(picked)
    form.add_row(tr("recipe"), combo, tr("recipe_help"))
    follow()
