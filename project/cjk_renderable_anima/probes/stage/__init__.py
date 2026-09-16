"""stage — one module per wake_probe stage; the shared plumbing is ``wake/``.

salad     Probe 0: does the base model's text salad hold real units?
data      glyph training set + eval prompt set (S-line mix in ``synth.py``)
train     frozen DiT, rectified-flow loss on glyph crops → trained.pt
classify  same-noise diffusion classifiers (``classify``, ``classify_str``)
eval      T2I floor vs trained on the eval set (``eval``) and scene prompts
          (``native``); also ``enref`` (the EN-reference renders the ruler
          scores against) and ``native_rescore`` (re-read an existing run)
scenes    self-generated EN-anchored scenes for the S-line composites (plan_synth)

``STAGES`` below is the registry ``--stage`` accepts; ``ALL`` is what
``--stage all`` expands to.
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
    "enref": ("eval", "stage_enref"),
    "native_rescore": ("eval", "stage_native_rescore"),
    "scenes": ("scenes", "stage_scenes"),
}
ALL = ["salad", "data", "train", "eval"]


def run(name: str, a) -> None:
    module, fn = STAGES[name]
    getattr(importlib.import_module(f".{module}", __package__), fn)(a)
