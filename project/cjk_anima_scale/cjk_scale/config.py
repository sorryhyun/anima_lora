"""Stage configs — ``configs/<stage>.toml`` → a validated ``StageConfig``.

One file per stage; the stages differ in ``band``, ``warm_from`` and the
data mix, and share the trainer surface (design § 5). Keys:

    stage = "stage0709"
    band = [0.7, 0.9]
    gate = "contain"          # contain | overlap | none (stage0309: mixed, no gate)
    min_overlap = 0.8         # the overlap gate's threshold
    warm_from = "stage0507"   # a stage name (same tag → its trained.pt) or a path

    [data]                    # inventory + pools, shared by every recipe
    seed, n_items, units, pieces, scenes, scene_one_bubble, single_scenes,
    single_max_ar, shapes, phrase_file, phrase_min_pieces, phrase_max_pieces

    [[data.mix]]              # the recipes and their shares (sum to 1)
    recipe = "scene_single"; share = 0.5; <recipe params>

    [train]                   # the whole trainer surface
    steps_per_row | train_steps, batch, lr_rows, lr_decay, lr_warmup,
    init_anchor, box_share, box_share_cap, compile, seed, save_every

    [eval]
    groups, native_chars, native_clauses, seeds, cf_sense (bool), regress (list of stages)
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .paths import CONFIGS, REPO, arm_dir

GATES = ("contain", "overlap", "none")
RECIPES = (
    "scene_single",
    "grid_single",
    "scene_piece",
    "scene_short",
    "scene_sentence",
    "grid_string",
)

_TRAIN_DEFAULTS = {
    "steps_per_row": 30,
    "train_steps": 0,  # 0 = steps_per_row × rows
    "batch": 4,
    "lr_rows": 1e-3,
    "lr_decay": "cosine",
    "lr_warmup": 500,
    "init_anchor": 0.0,
    "free_residual": 1e-3,  # μ‖f‖² on rows the warm table never had (a constant, not a lever)
    "box_share": 0.25,  # in-box share at one glyph (loss.py: log in the glyph count)
    "box_share_cap": 0.5,  # the ceiling …
    "box_share_glyphs": 8,  # … reached at this many glyphs
    "compile": 1,
    "seed": 0,
    "save_every": 5000,
    "steps": 28,  # generation settings the probe's helpers want
    "cfg": 4.0,
}
_EVAL_DEFAULTS = {
    "groups": "single,single_ext,single_small,single_kanji,single_extra,piece,short,short_held,phrase,phrase_held,en",
    "native_chars": "あ,か,す,日",
    "native_clauses": "en,swap",
    "seeds": 2,
    "cf_sense": True,
    "cf_rows": "single",
    "regress": [],
}
_DATA_DEFAULTS = {
    "seed": 0,
    "n_items": 40000,
    "units": ["kana"],
    "pieces": "",
    "scenes": "s1,s1w,sl1w,ja_comic",
    "scene_one_bubble": "ja_comic",
    "single_scenes": "s1,s1w",
    "single_max_ar": 2.0,
    "shapes": "448,512:2,448x512,512x448",
    "phrase_file": "",
    "phrase_min_pieces": 2,
    "phrase_max_pieces": 10,
    "phrase_norm": True,
    "phrase_held_books": 2,
    "n_phrase_eval": 8,
    "n_piece_eval": 18,
    "vertical": True,
    "stroke": 0.0,
}


@dataclass
class Recipe:
    name: str
    share: float
    params: dict = field(default_factory=dict)


@dataclass
class StageConfig:
    stage: str
    band: tuple
    gate: str
    min_overlap: float
    warm_from: str
    data: dict
    mix: list
    train: dict
    eval: dict
    path: Path

    def warm_table(self, tag: str) -> Path | None:
        """``warm_from`` → the table to start from: a stage name is that
        stage's ``trained.pt`` under the same tag, a path is taken as typed
        (``""`` = cold)."""
        w = self.warm_from
        if not w:
            return None
        if "/" not in w and not w.endswith(".pt"):
            return arm_dir(w, tag) / "trained.pt"
        p = Path(os.path.expanduser(os.path.expandvars(w)))
        return p if p.is_absolute() else REPO / p

    def train_steps(self, n_rows: int) -> int:
        t = self.train
        return int(t["train_steps"]) or int(t["steps_per_row"]) * n_rows


def stage_names() -> list[str]:
    return sorted(p.stem for p in CONFIGS.glob("stage*.toml"))


def load(stage: str) -> StageConfig:
    path = CONFIGS / f"{stage}.toml" if "/" not in stage else Path(stage)
    assert path.is_file(), f"no stage config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    name = raw.get("stage", path.stem)
    band = tuple(float(x) for x in raw["band"])
    assert len(band) == 2 and 0.0 <= band[0] < band[1] <= 1.0, f"band {band}"
    gate = raw.get("gate", "contain")
    assert gate in GATES, f"gate {gate!r}: one of {GATES}"
    data = {
        **_DATA_DEFAULTS,
        **{k: v for k, v in raw.get("data", {}).items() if k != "mix"},
    }
    if "$" in data["phrase_file"]:
        from library.env import load_dotenv

        load_dotenv()  # MANGA109S lives in the repo's .env (never overrides a real var)
        if "$" in os.path.expandvars(data["phrase_file"]):
            raise SystemExit(
                f"{data['phrase_file']}: env var unset — put MANGA109S=<root> in .env"
            )
    data["phrase_file"] = (
        os.path.expanduser(os.path.expandvars(data["phrase_file"]))
        if data["phrase_file"]
        else ""
    )
    mix = [
        Recipe(
            m["recipe"],
            float(m["share"]),
            {k: v for k, v in m.items() if k not in ("recipe", "share")},
        )
        for m in raw.get("data", {}).get("mix", [])
    ]
    assert mix, f"{path}: [[data.mix]] is empty"
    for m in mix:
        assert m.name in RECIPES, f"{path}: recipe {m.name!r}: one of {RECIPES}"
    assert abs(sum(m.share for m in mix) - 1.0) < 1e-6, (
        f"{path}: recipe shares sum to {sum(m.share for m in mix)}"
    )
    train = {**_TRAIN_DEFAULTS, **raw.get("train", {})}
    unknown = set(train) - set(_TRAIN_DEFAULTS)
    assert not unknown, (
        f"{path}: [train] keys not on the trainer surface: {sorted(unknown)}"
    )
    ev = {**_EVAL_DEFAULTS, **raw.get("eval", {})}
    return StageConfig(
        stage=name,
        band=band,
        gate=gate,
        min_overlap=float(raw.get("min_overlap", 0.8)),
        warm_from=raw.get("warm_from", ""),
        data=data,
        mix=mix,
        train=train,
        eval=ev,
        path=path,
    )
