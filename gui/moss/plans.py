"""The ``model`` dropdown's items on MOSS forms: variant × int8 / bf16 plan.

An item's data is the variant, the request field. Changing the plan writes the fields
below; the item follows the int8 field, so tuning a field keeps the item.
"""

from __future__ import annotations

import tomllib

from gui.lines.window import tr
from library.env import resolve_under_home
from library.moss.requests import GenerateRequest, TrainRequest


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
    """int8: docs/moss/optimization/int8.md's plan, which runs no block swap. bf16:
    the request defaults, under the task's ``seed`` on i2i."""
    seed = seed or {}
    defaults = GenerateRequest()
    keys = ("int8", "stream", "blocks_to_swap")
    return {
        "int8": {"int8": "on", "stream": "fp32", "blocks_to_swap": 0},
        "bf16": {k: seed.get(k, getattr(defaults, k)) for k in keys},
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
