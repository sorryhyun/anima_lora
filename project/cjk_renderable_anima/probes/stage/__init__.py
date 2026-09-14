"""stage — one module per wake_probe stage; the shared plumbing is ``wake/``.

salad     Probe 0: does the base model's text salad hold real units?
data      glyph training set + eval prompt set
train     frozen DiT, rectified-flow loss on glyph crops → trained.pt
classify  same-noise diffusion classifiers (``classify``, ``classify_str``)
eval      T2I floor vs trained on the eval set (``eval``) and scene prompts (``native``)
scenes    self-generated EN-anchored scenes for the S-line composites (plan_synth)
"""

from __future__ import annotations

import importlib

# stage name → (module, function); modules import lazily so --stage data never loads torch
STAGES = {
    "salad": ("salad", "stage_salad"),
    "data": ("data", "stage_data"),
    "train": ("train", "stage_train"),
    "eval": ("eval", "stage_eval"),
    "classify": ("classify", "stage_classify"),
    "classify_str": ("classify", "stage_classify_str"),
    "native": ("eval", "stage_native"),
    "scenes": ("scenes", "stage_scenes"),
}
ALL = ["salad", "data", "train", "eval"]


def run(name: str, a) -> None:
    module, fn = STAGES[name]
    getattr(importlib.import_module(f".{module}", __package__), fn)(a)
