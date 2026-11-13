"""The dialogue ruler (``criteria.md``): real JA dialogue in a speech bubble,
read per render against a floor rendered once. The CLI is the line's
``ruler.py``.

build    the string set → ``output/cjk_anima_reseed/ruler/ruler.json`` (CPU)
arms     the arms: run dirs, ``--pack`` arms, derived rows, ext-id tables
render   ``Renderer`` (one ``ExtDelta``, routed) and the render pass (GPU)
score    one render's scores: text, glyphs / regions, EN-ref page sims
stats    tallies per group and the paired tests
read     reads every render, scores, pairs → ``results/<ts>-ruler-<label>/``;
         ``sample`` sheets

Here: the paths, the shared constants, what a run renders / reads (``VIEW``)
and the render files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from reseed import OUT

RULER = OUT / "ruler"
BINS = {"short": (2, 4), "mid": (5, 9), "long": (10, 20)}
# the prompt sets; renders under RULER / mode. `captioned`: the image's own
# caption as the tagger wrote it. Renders under RULER/<arm> are the 10-05
# first read's (an older string set) and RULER/sensitive/ the 10-05..11-13
# reads' (a rewritten-rating prompt set, retired) — records only
MODES = ("captioned",)
EN = "en"  # the EN references: no ext row in the caption, the delta off
FLOOR = ("en", "retrain_kana", "seed_retrain_0930")  # rendered once (criteria.md)
STEPS, CFG, SEED_RENDER = 28, 4.0, 0  # the reads of record's sampler
# ``X#<model>``: arm X rendered on another generator (DiT / LoRA / sampler) —
# its own dir, read against ``en#<model>``, the same model's EN refs. A bare
# arm is the reads of record's base at STEPS / CFG. Paths relative to the repo.
M14 = "models/diffusion_models/anima_merged14.safetensors"
BASE_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
MODELS = {
    # the base on today's code (the bare arms' renders are older code)
    "base": {"dit": BASE_DIT, "steps": STEPS, "cfg": CFG},
    "m14": {"dit": M14, "steps": STEPS, "cfg": CFG},  # the soup teacher
    "m14v3": {  # merged14 + Hub turbo v3 at 1.0, untrained on m14
        "dit": M14,
        "lora": "models/loras/turbo_hub/anima_turbo_4step_v3.safetensors",
        "steps": 4,
        "cfg": 1.0,
    },
}
MARK_CHARS = set("～〜~…‥♡♥、。，")


@dataclass
class View:
    """What a run renders / reads; ``ruler.py`` sets it from its flags."""

    mode: str = MODES[0]  # ``--prompts``: the prompt set rendered / read
    # ``--pack``: the base pack a render sits on (``reseed.config.PACKS``), its
    # own arms (a run with ``pack``); ``X@<pack>`` = arm X's rows on that pack
    # (its routing), rendered beside X's on the raw pack
    pack: str = ""
    # ``--marks``: only the strings holding a punct run's mark row (``〜`` /
    # ``～`` / ``~``, a dot run, ``♡♥``, ``、。，``) — the rest encode as on the floor
    marks_only: bool = False
    only: set = field(default_factory=set)  # ``--only``: these indices (a look)


VIEW = View()


def has_mark(t: str) -> bool:
    return any(c in MARK_CHARS for c in t) or "・・" in t or ".." in t


def items() -> list:
    its = json.loads((RULER / "ruler.json").read_text(encoding="utf-8"))["items"]
    if VIEW.only:
        its = [m for m in its if m["i"] in VIEW.only]
    return [m for m in its if has_mark(m["text"])] if VIEW.marks_only else its


def mode_dir() -> Path:
    return RULER / VIEW.mode


def model_of(arm: str) -> str:
    return arm.split("#")[1] if "#" in arm else ""


def en_of(arm: str) -> str:
    """The EN refs an arm reads against: its own model's."""
    m = model_of(arm)
    return f"{EN}#{m}" if m else EN


def render_file(arm: str, i: int) -> Path:
    # the EN refs keep eval.enref's naming (pi = the ruler index)
    if arm.split("#")[0] == EN:
        return mode_dir() / arm / f"enref_p{i:02d}_s{SEED_RENDER}.png"
    return mode_dir() / arm / f"r{i:02d}_s{SEED_RENDER}.png"
