"""Imports, the four stage configs and the run files, the chain's warm
resolution, the front door's parser, and the probe primitives the recipes lean on."""

from __future__ import annotations

import importlib

import pytest

from cjk_scale import config
from cjk_scale.paths import LINE, PROBE_SRC, arm_dir, data_dir, run_tag

MODULES = [
    "paths",
    "windows",
    "config",
    "recipes",
    "builder",
    "rows",
    "train",
    "loss",
    "boxprobe",
    "eval",
    "bake",
    "ledger",
]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    if name in ("rows", "train", "boxprobe"):
        pytest.importorskip("torch")
    mod = importlib.import_module(f"cjk_scale.{name}")
    assert mod.__file__.startswith(str(LINE))


def test_probe_packages_are_the_probe_line():
    """The names the recipes import resolve to the probe's ``src/``, not to
    anything of ours — the reason this package is not called ``src``."""
    for name in (
        "common.render.scene",
        "common.render.flat",
        "data.grid",
        "data.stage",
        "data.synth",
        "train.stage",
    ):
        mod = importlib.import_module(name)
        assert mod.__file__.startswith(str(PROBE_SRC)), (name, mod.__file__)


def test_output_root_is_ours_and_the_probe_reads_it():
    """Runs land under ``output/cjk_anima_scale/`` and the probe modules that
    bind ``OUT`` at import (``data.synth`` → the scene pools, ``eval.enref``
    → the EN reference cache) see the same root — the redirect in
    ``bootstrap()`` ran before they loaded."""
    import common.paths as probe_paths
    from cjk_scale.paths import OUT, REPO

    assert OUT == REPO / "output" / "cjk_anima_scale"
    assert probe_paths.OUT == OUT
    assert data_dir("stage0709", "t").parent == OUT
    assert importlib.import_module("data.synth").OUT == OUT
    assert importlib.import_module("eval.enref").OUT == OUT
    assert probe_paths.arm_dir(
        type(
            "A",
            (),
            {"arm": "rows", "data_tag": run_tag("stage0709", "t"), "arm_tag": ""},
        )()
    ) == arm_dir("stage0709", "t")


def test_box_share_curve():
    from cjk_scale.loss import box_share_of, glyph_count

    assert glyph_count("じゃ ない") == 4 and glyph_count("") == 1
    assert box_share_of(1, 0.25, 0.5, 8) == 0.25
    assert abs(box_share_of(2, 0.25, 0.5, 8) - (0.25 + 0.25 / 3)) < 1e-9
    assert abs(box_share_of(4, 0.25, 0.5, 8) - (0.25 + 0.5 / 3)) < 1e-9
    assert box_share_of(8, 0.25, 0.5, 8) == 0.5 == box_share_of(40, 0.25, 0.5, 8)
    assert box_share_of(5, 0.25, 0.5, 1) == 0.25  # n_cap 1: flat


def test_primitives_exist():
    from common.render.scene import H_PITCH, V_PITCH, region_capacity, render_into_scene  # noqa: F401
    from data.grid import _Deck, render_grid  # noqa: F401


def test_stage_configs_load_and_chain():
    names = config.stage_names()
    assert names == ["stage0305", "stage0507", "stage0709", "stage0309"] or set(
        names
    ) == {
        "stage0305",
        "stage0507",
        "stage0709",
        "stage0309",
    }
    cfgs = {n: config.load(n) for n in names}
    assert cfgs["stage0709"].band == (0.7, 0.9)
    assert cfgs["stage0507"].band == (0.5, 0.7)
    assert cfgs["stage0305"].band == (0.3, 0.5)
    assert cfgs["stage0309"].band == (0.3, 0.9) and cfgs["stage0309"].gate == "none"
    # the warm chain: each stage starts from the previous one under the same tag
    assert cfgs["stage0507"].warm_table("t") == arm_dir("stage0709", "t") / "trained.pt"
    assert cfgs["stage0305"].warm_table("t") == arm_dir("stage0507", "t") / "trained.pt"
    assert cfgs["stage0309"].warm_table("t") == arm_dir("stage0305", "t") / "trained.pt"
    # the first stage has no warm_from of its own: cold without a run, the
    # run's seed_table with one
    assert (
        cfgs["stage0709"].warm_from == "" and cfgs["stage0709"].warm_table("t") is None
    )
    w = config.load("stage0709", "run_full").warm_table("t")
    assert (
        w is not None and w.name == "trained.pt" and "rows_step1_0921_merged" in str(w)
    )
    for c in cfgs.values():
        assert abs(sum(m.share for m in c.mix) - 1) < 1e-9
        assert c.train_steps(2300) == 30 * 2300
        assert "lr_warmup" not in c.train and 0 < c.train["lr_warmup_ratio"] < 1
        # a stage file never says which rows
        assert not (set(c.data) & set(config.RUN_DATA_KEYS)) - set(
            config._DATA_DEFAULTS
        )


def test_run_files_overlay_the_stage():
    """A run file carries rows, seed table and budgets; run wins over stage."""
    assert set(config.run_names()) >= {"run_full", "run0923_micro"}
    full = config.load_run("run_full")
    assert (
        full.data["pieces"] == "ja_cold_0001_1900.txt"
        and full.budget["stage0709"] == 30
    )
    micro = config.load("stage0507", "run0923_micro")
    assert micro.run is not None and micro.run.name == "run0923_micro"
    assert micro.data["units"][0].startswith("chars:") and micro.data["pieces"] == ""
    assert micro.data["phrase_file"] == "" and micro.data["n_items"] == 4000
    assert micro.train["steps_per_row"] == 30 and micro.train_steps(24) == 720
    assert micro.warmup_steps(720) == 72
    assert micro.eval["groups"] == "single,word,en" and micro.eval["cf_rows"] == "piece"
    assert micro.data["seed"] == 0 and micro.train["seed"] == 0
    # the chain resolves under the run's name; the mix is the stage's
    assert (
        micro.warm_table("run0923_micro")
        == arm_dir("stage0709", "run0923_micro") / "trained.pt"
    )
    assert [m.name for m in micro.mix] == [m.name for m in config.load("stage0507").mix]
    # stage files may not carry run keys
    bad = LINE / "configs" / "_bad_stage.toml"
    bad.write_text(
        'stage = "bad"\nband = [0.5, 0.7]\n[data]\nunits = ["kana"]\n'
        '[[data.mix]]\nrecipe = "scene_single"\nshare = 1.0\n',
        encoding="utf-8",
    )
    try:
        with pytest.raises(AssertionError, match="belong to a run file"):
            config.load(str(bad))
    finally:
        bad.unlink()


def test_every_recipe_in_the_mixes_is_registered():
    from cjk_scale.recipes import RECIPES

    for n in config.stage_names():
        for m in config.load(n).mix:
            assert m.name in RECIPES, (n, m.name)


def test_run_dirs_follow_the_probe_layout():
    assert run_tag("stage0709", "t1") == "scale_stage0709_t1"
    assert data_dir("stage0709", "t1").name == "data_scale_stage0709_t1"
    assert arm_dir("stage0709", "t1").name == "rows_scale_stage0709_t1"
    with pytest.raises(AssertionError):
        run_tag("stage0709", "a b")


def test_front_door_parser():
    import scale

    a = scale.build_parser().parse_args(
        ["--stage", "stage0709", "--tag", "t1", "--steps", "data", "train"]
    )
    assert a.command == "run" and a.steps == ["data", "train"] and a.run is None
    a = scale.build_parser().parse_args(
        ["--run", "run0923_micro", "--stage", "stage0709", "--steps", "eval"]
    )
    assert a.run == "run0923_micro" and a.tag is None
    a = scale.build_parser().parse_args(["windows"])
    assert a.command == "windows"


def test_horizontal_marker_and_grid_share():
    """30 % of multi-glyph items are drawn as lines and say so: the scene
    caption's marker per frame, the grid's per-cell draw."""
    import random

    from common.prompts import grid_caption
    from common.render.flat import find_fonts
    from data.grid import render_grid
    from data.synth import scene_caption

    sc = {"head": ["manga"], "generals": ["english text"], "clause_tpl": None}
    reads = {**sc, "clause_tpl": 'English text reads as "{a}".'}
    said = {**sc, "clause_tpl": 'She is saying "{a}".'}
    assert (
        scene_caption(reads, "って")
        == 'manga, japanese text. Japanese text reads as "って".'
    )
    assert (
        scene_caption(reads, "って", horizontal=True)
        == 'manga, japanese text. horizontal Japanese text reads as "って".'
    )
    assert (
        scene_caption(said, "って", horizontal=True)
        == 'manga, japanese text. She is saying "って", written horizontally.'
    )
    assert scene_caption(sc, "あ", horizontal=True).endswith(
        'horizontal Japanese text reads as "あ".'
    )
    fonts = find_fonts()
    for frac, want in ((0.0, set()), (1.0, {0, 1, 3})):
        lines: list = []
        render_grid(
            ["って", "んだ", "あ", "先生"],
            2,
            2,
            (256, 256),
            fonts,
            random.Random(0),
            False,
            (0.3, 0.3),
            lines=lines,
            horizontal_frac=frac,
        )
        assert set(lines) == want, (frac, lines)  # the single glyph あ never
    cap = grid_caption("flat", 2, 2, ["って", "んだ", "あ", "先生"], horizontal={0, 3})
    assert cap.count("horizontal Japanese text reads as") == 2
    for c in config.stage_names():
        assert config.load(c).data["horizontal_frac"] == 0.3


def test_probe_eval_namespace_builds():
    from cjk_scale.eval import probe_args

    cfg = config.load("stage0507")
    a = probe_args(cfg, "t1", ["eval", "native", "cf_sense"], ["--eval_limit", "3"])
    assert a.data_tag == "scale_stage0507_t1" and a.arm == "rows" and a.no_floor
    assert a.cf_lang == "ja" and a.cf_rows == "piece" and a.eval_limit == 3
    assert a.native_clauses == "en,swap"
