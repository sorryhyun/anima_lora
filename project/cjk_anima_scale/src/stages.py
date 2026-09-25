"""stages — the ``--stage`` registry the line calls; each stage lives in the
package of its role. Vendored into ``project/cjk_anima_scale/src/``
(2026-09-25) and trimmed to what the line runs:

eval      the eval set rendered with the table and read (``--no_floor``: the
          line renders its floor as its own arm, ``load(seed)``)  (eval/stage)
native    scene prompts + a vocab clause; ``--eval_tag sent`` reads the run's
          ``read`` strings  (eval/native)
target    the user's own captions (``--target_prompts``) rendered verbatim at
          --eval_shape, readers only  (eval/native)
cf_sense  caption leverage on a B-rendered input, by σ  (eval/cf_sense)
summary   eval_summary.png — headline numbers + a hit and a miss per group
          from eval / native / target; auto-written when each of those ends
          (eval/summary)
scenes    self-generated EN-anchored scenes — the scene pools the recipes
          draw on (generation in scenes/stage, keep / reject in scenes/judge)
"""

from __future__ import annotations

import importlib

# stage name → (module, function); modules import lazily so nothing loads torch early
STAGES = {
    "eval": ("eval.stage", "stage_eval"),
    "cf_sense": ("eval.cf_sense", "stage_cf_sense"),
    "native": ("eval.native", "stage_native"),
    "target": ("eval.native", "stage_target"),
    "summary": ("eval.summary", "stage_summary"),
    "scenes": ("scenes.stage", "stage_scenes"),
}


def run(name: str, a) -> None:
    module, fn = STAGES[name]
    getattr(importlib.import_module(module), fn)(a)
