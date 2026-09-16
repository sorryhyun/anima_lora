"""stages — the ``--stage`` registry; each stage lives in the package of its role.

salad     Probe 0: does the base model's text salad hold real units?  (eval/salad)
data      glyph training set + eval prompt set (S-line mix in data/synth)
train     frozen DiT, rectified-flow loss on glyph crops → trained.pt
classify  same-noise diffusion classifiers (``classify``, ``classify_str``)  (eval/classify)
eval      T2I floor vs trained on the eval set (eval/stage)
native    scene prompts + a kana clause; also ``enref`` (the EN-reference
          renders the ruler scores against) and ``native_rescore`` (re-read
          an existing run)  (eval/native)
scenes    self-generated EN-anchored scenes for the S-line composites
          (generation in scenes/stage, keep / reject in scenes/judge)

``STAGES`` below is the registry ``--stage`` accepts; ``ALL`` is what
``--stage all`` expands to.
"""

from __future__ import annotations

import importlib

# stage name → (module, function); modules import lazily so --stage data never loads torch
STAGES = {
    "salad": ("eval.salad", "stage_salad"),
    "data": ("data.stage", "stage_data"),
    "train": ("train.stage", "stage_train"),
    "eval": ("eval.stage", "stage_eval"),
    "classify": ("eval.classify", "stage_classify"),
    "classify_str": ("eval.classify", "stage_classify_str"),
    "native": ("eval.native", "stage_native"),
    "enref": ("eval.native", "stage_enref"),
    "native_rescore": ("eval.native", "stage_native_rescore"),
    "scenes": ("scenes.stage", "stage_scenes"),
}
ALL = ["salad", "data", "train", "eval"]


def run(name: str, a) -> None:
    module, fn = STAGES[name]
    getattr(importlib.import_module(module), fn)(a)
