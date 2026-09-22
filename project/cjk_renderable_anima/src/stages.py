"""stages — the ``--stage`` registry; each stage lives in the package of its role.

salad     Probe 0: does the base model's text salad hold real units?  (eval/salad)
data      glyph training set + eval prompt set (S-line mix in data/synth)
train     frozen DiT, rectified-flow loss on glyph crops → trained.pt
classify  same-noise diffusion classifiers (``classify``, ``classify_str``)  (eval/classify)
cf_sense  idea.md Gate 0: caption leverage on a B-rendered input, by σ  (eval/cf_sense)
eval      T2I floor vs trained on the eval set (eval/stage)
native    scene prompts + a kana clause; also ``enref`` (the EN-reference
          renders the ruler scores against) and ``native_rescore`` (re-read
          an existing run)  (eval/native)
target    the user's own captions (``--target_prompts``) rendered verbatim at
          --eval_shape, floor vs trained, readers only  (eval/native)
summary   eval_summary.png — headline numbers + a hit and a miss per group
          from eval / native / target; auto-written when each of those ends
          (eval/summary)
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
    "cf_sense": ("eval.cf_sense", "stage_cf_sense"),
    "native": ("eval.native", "stage_native"),
    "enref": ("eval.native", "stage_enref"),
    "native_rescore": ("eval.native", "stage_native_rescore"),
    "target": ("eval.native", "stage_target"),
    "summary": ("eval.summary", "stage_summary"),
    "scenes": ("scenes.stage", "stage_scenes"),
}
ALL = ["salad", "data", "train", "eval"]


def run(name: str, a) -> None:
    module, fn = STAGES[name]
    getattr(importlib.import_module(module), fn)(a)
