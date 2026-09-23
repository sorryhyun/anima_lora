"""Imports, the four stage configs, the chain's warm resolution, the front
door's parser, and the probe primitives the recipes lean on."""

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
    w = cfgs["stage0709"].warm_table("t")
    assert (
        w is not None and w.name == "trained.pt" and "rows_step1_0921_merged" in str(w)
    )
    for c in cfgs.values():
        assert abs(sum(m.share for m in c.mix) - 1) < 1e-9
        assert c.train_steps(2300) == 30 * 2300


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
    assert a.command == "run" and a.steps == ["data", "train"]
    a = scale.build_parser().parse_args(["windows"])
    assert a.command == "windows"


def test_probe_eval_namespace_builds():
    from cjk_scale.eval import probe_args

    cfg = config.load("stage0507")
    a = probe_args(cfg, "t1", ["eval", "native", "cf_sense"], ["--eval_limit", "3"])
    assert a.data_tag == "scale_stage0507_t1" and a.arm == "rows" and a.no_floor
    assert a.cf_lang == "ja" and a.cf_rows == "piece" and a.eval_limit == 3
    assert a.native_clauses == "en,swap"
