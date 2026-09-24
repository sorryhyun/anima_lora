"""Stage configs and run configs — two files, one ``StageConfig``.

A **stage file** (``configs/<stage>.toml``) is the band recipe: band, gate,
the recipe mix, the trainer surface, its place in the chain. It never says
which rows. A **run file** (``configs/runs/<run>.toml``) says the rest —
the inventory, the seed table the first stage warms from, the steps per
row per stage, the eval rows — the things that differ between a 16-row
read and the 2 300-row production chain. ``load(stage, run)`` merges them,
**run over stage** on every key both name. The run's name is the chain's tag.

Stage file keys:

    stage = "stage0709"
    band = [0.7, 0.9]
    gate = "contain"          # contain | overlap | none (stage0309: mixed, no gate)
    min_overlap = 0.8         # the overlap gate's threshold
    warm_from = "stage0507"   # the previous stage (same tag → its trained.pt); "" = the run's seed_table

    [data]                    # pools shared by every recipe — never units / pieces / phrase_file / n_items / seed
    scenes, scene_one_bubble, single_scenes, single_max_ar, horizontal_scenes, shapes, short_pieces, …

    [[data.mix]]              # the recipes and their shares (sum to 1)
    recipe = "scene_single"; share = 0.5; <recipe params>

    [train]                   # the trainer surface (steps_per_row comes from the run's [budget])
    batch, lr_rows, lr_decay, lr_warmup_ratio, init_anchor, box_share, box_share_cap, compile, save_every

    [eval]
    groups, native_chars, native_clauses, seeds, cf_sense (bool), cf_rows, regress (list of stages)

Run file keys:

    run = "run0923_micro"     # = the tag
    seed_table = "output/…/trained.pt"   # what a stage with warm_from = "" starts from
    seed = 0                  # data draw + train seed

    [data]                    # units, pieces, phrase_file, n_items (+ any [data] key, over the stage's)
    [budget]                  # <stage> = steps per row
    [train]                   # over the stage's [train] (surface keys only)
    [eval]                    # over the stage's [eval]
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .paths import CONFIGS, REPO, arm_dir

RUNS = CONFIGS / "runs"
GATES = ("contain", "overlap", "none")
RECIPES = (
    "scene_single",
    "grid_single",
    "scene_piece",
    "scene_short",
    "scene_sentence",
    "grid_string",
)
# [data] keys a stage file may not carry: they say which rows / how many, i.e. the run
RUN_DATA_KEYS = ("units", "pieces", "phrase_file", "n_items", "seed")

_TRAIN_DEFAULTS = {
    "steps_per_row": 30,
    "train_steps": 0,  # 0 = steps_per_row × rows
    "batch": 4,
    "lr_rows": 1e-3,
    "lr_decay": "cosine",
    "lr_warmup_ratio": 0.1,  # linear warmup over this fraction of the stage's steps
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
    "horizontal_scenes": "sl1w",  # the pools a left-to-right scene item may go to ("" = every pool)
    "shapes": "448,512:2,448x512,512x448",
    "phrase_file": "",
    "phrase_min_pieces": 2,
    "phrase_max_pieces": 10,
    "phrase_norm": True,
    "phrase_held_books": 2,
    "n_phrase_eval": 8,
    "n_piece_eval": 18,
    "vertical": True,  # no horizontal fit fallback: orientation is a draw (below)
    "stroke": 0.0,
    "horizontal_frac": 0.3,  # share of multi-glyph items (scene) / cells (grid) drawn as lines, marked
}


@dataclass
class Recipe:
    name: str
    share: float
    params: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    name: str
    path: Path
    seed_table: str  # "" = cold
    seed: int | None
    data: dict
    budget: dict  # stage → steps per row
    train: dict
    eval: dict


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
    run: RunConfig | None = None

    def warm_table(self, tag: str) -> Path | None:
        """``warm_from`` → the table to start from: a stage name is that
        stage's ``trained.pt`` under the same tag; a path is taken as typed;
        ``""`` is the run's ``seed_table`` (cold without a run)."""
        w = self.warm_from
        if not w:
            w = self.run.seed_table if self.run else ""
            if not w:
                return None
        if "/" not in w and not w.endswith(".pt"):
            return arm_dir(w, tag) / "trained.pt"
        p = Path(os.path.expanduser(os.path.expandvars(w)))
        return p if p.is_absolute() else REPO / p

    def train_steps(self, n_rows: int) -> int:
        t = self.train
        return int(t["train_steps"]) or int(t["steps_per_row"]) * n_rows

    def warmup_steps(self, steps: int) -> int:
        return int(round(float(self.train["lr_warmup_ratio"]) * steps))


def stage_names() -> list[str]:
    return sorted(p.stem for p in CONFIGS.glob("stage*.toml"))


def run_names() -> list[str]:
    return sorted(p.stem for p in RUNS.glob("*.toml"))


def _expand_phrase_file(p: str) -> str:
    if "$" in p:
        from library.env import load_dotenv

        load_dotenv()  # MANGA109S lives in the repo's .env (never overrides a real var)
        if "$" in os.path.expandvars(p):
            raise SystemExit(f"{p}: env var unset — put MANGA109S=<root> in .env")
    return os.path.expanduser(os.path.expandvars(p)) if p else ""


def load_run(run: str) -> RunConfig:
    path = RUNS / f"{run}.toml" if "/" not in run else Path(run)
    assert path.is_file(), f"no run config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    name = raw.get("run", path.stem)
    train = dict(raw.get("train", {}))
    unknown = set(train) - set(_TRAIN_DEFAULTS)
    assert not unknown, (
        f"{path}: [train] keys not on the trainer surface: {sorted(unknown)}"
    )
    assert "steps_per_row" not in train, f"{path}: steps per row go in [budget]"
    budget = {k: int(v) for k, v in raw.get("budget", {}).items()}
    seed = raw.get("seed")
    return RunConfig(
        name=name,
        path=path,
        seed_table=str(raw.get("seed_table", "")),
        seed=None if seed is None else int(seed),
        data=dict(raw.get("data", {})),
        budget=budget,
        train=train,
        eval=dict(raw.get("eval", {})),
    )


def load(stage: str, run: str | RunConfig | None = None) -> StageConfig:
    path = CONFIGS / f"{stage}.toml" if "/" not in stage else Path(stage)
    assert path.is_file(), f"no stage config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    name = raw.get("stage", path.stem)
    band = tuple(float(x) for x in raw["band"])
    assert len(band) == 2 and 0.0 <= band[0] < band[1] <= 1.0, f"band {band}"
    gate = raw.get("gate", "contain")
    assert gate in GATES, f"gate {gate!r}: one of {GATES}"
    rc = load_run(run) if isinstance(run, str) else run
    stage_data = {k: v for k, v in raw.get("data", {}).items() if k != "mix"}
    carried = sorted(set(stage_data) & set(RUN_DATA_KEYS))
    assert not carried, f"{path}: [data] {carried} belong to a run file"
    data = {**_DATA_DEFAULTS, **stage_data, **(rc.data if rc else {})}
    data["phrase_file"] = _expand_phrase_file(str(data["phrase_file"]))
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
    stage_train = dict(raw.get("train", {}))
    unknown = set(stage_train) - set(_TRAIN_DEFAULTS)
    assert not unknown, (
        f"{path}: [train] keys not on the trainer surface: {sorted(unknown)}"
    )
    assert "lr_warmup" not in stage_train, f"{path}: lr_warmup is lr_warmup_ratio now"
    train = {**_TRAIN_DEFAULTS, **stage_train}
    ev = {**_EVAL_DEFAULTS, **raw.get("eval", {})}
    if rc:
        train.update(rc.train)
        if name in rc.budget:
            train["steps_per_row"] = rc.budget[name]
        if rc.seed is not None:
            data["seed"] = rc.seed
            train["seed"] = rc.seed
        ev.update(rc.eval)
    assert 0.0 <= float(train["lr_warmup_ratio"]) < 1.0, (
        f"{path}: lr_warmup_ratio {train['lr_warmup_ratio']} is a fraction of the run"
    )
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
        run=rc,
    )
