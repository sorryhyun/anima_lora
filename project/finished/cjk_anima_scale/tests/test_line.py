"""Imports and the line's boundary (its stage packages are its own ``src/``),
the one-file run, the recipe table by kind, the run layout, the front door,
the eval arms (the merged trained rows + the floor) and the one-sheet compose."""

from __future__ import annotations

import importlib
import json
import os
import random

import pytest

from cjk_scale import config
from cjk_scale.paths import (
    LINE,
    OUT,
    REPO,
    SEED_ROWS,
    SEED_ROWS_0921,
    SRC,
    data_dir,
    floor_dir,
    run_dir,
    trained_path,
)

MODULES = [
    "paths",
    "windows",
    "config",
    "recipes",
    "builder",
    "rows",
    "train",
    "loss",
    "eval",
    "conflict",
    "ledger",
    "budget",
    "merge",
    "reads",
]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    if name in ("rows", "train", "conflict"):
        pytest.importorskip("torch")
    mod = importlib.import_module(f"cjk_scale.{name}")
    assert mod.__file__.startswith(str(LINE))


# ---------------------------------------------------------------------------
# the boundary: the stage packages are this line's own src/


STAGE_MODULES = (
    "stages",
    "cli",
    "common.paths",
    "common.models",
    "common.prompts",
    "common.readers",
    "common.hooks",
    "common.render.scene",
    "common.render.flat",
    "common.render.ink",
    "data.grid",
    "data.stage",
    "data.synth",
    "data.inventory",
    "train.stage",
    "eval.stage",
    "eval.native",
    "eval.enref",
    "eval.cf_sense",
    "scenes.stage",
    "probe.merge_tables",
)


@pytest.mark.parametrize("name", STAGE_MODULES)
def test_stage_packages_are_the_lines_own(name):
    """After ``paths.bootstrap()`` every stage module the line imports
    resolves under ``project/cjk_anima_scale/src/`` — the probe line is
    independent and never on the path."""
    mod = importlib.import_module(name)
    assert mod.__file__.startswith(str(SRC)), (name, mod.__file__)


def test_nothing_names_the_probe_line():
    """No code or config of the line (outside ``_archive/``) names the probe
    line — no import, no sys.path entry, no Path constant."""
    probe = "cjk_renderable" + "_anima"
    hits = [
        str(p.relative_to(LINE))
        for p in sorted(LINE.rglob("*"))
        if p.suffix in (".py", ".toml")
        and "_archive" not in p.relative_to(LINE).parts
        and "__pycache__" not in p.parts
        and probe in p.read_text(encoding="utf-8")
    ]
    assert not hits, hits


def test_stage_registry_is_trimmed():
    from stages import STAGES

    assert set(STAGES) == {"eval", "native", "target", "cf_sense", "summary", "scenes"}
    for module, fn in STAGES.values():
        assert callable(getattr(importlib.import_module(module), fn))


def test_output_root_is_ours():
    """Runs land under ``output/cjk_anima_scale/`` and the stage modules that
    bind ``OUT`` at import (``data.synth`` → the scene pools, ``eval.enref`` →
    the EN reference cache) see the same root; assets resolve in the line."""
    import common.paths as stage_paths
    from common.prompts import TARGET_PROMPTS
    from data.vocabs import VOCABS_DIR as STAGE_VOCABS

    from cjk_scale.paths import VOCABS_DIR

    assert OUT == REPO / "output" / "cjk_anima_scale"
    assert stage_paths.OUT == OUT and stage_paths.REPO == REPO
    assert importlib.import_module("data.synth").OUT == OUT
    assert importlib.import_module("eval.enref").OUT == OUT
    assert stage_paths.FONT_DIR == LINE / "assets" / "fonts"
    assert STAGE_VOCABS == VOCABS_DIR == LINE / "assets" / "vocabs"
    assert TARGET_PROMPTS == LINE / "assets" / "target_prompts.txt"
    assert (
        TARGET_PROMPTS.is_file() and (VOCABS_DIR / "ja_pieces_0925_300.txt").is_file()
    )


def test_run_layout():
    assert run_dir("r1") == OUT / "r1"
    assert data_dir("r1") == OUT / "r1" / "data"
    assert trained_path("r1") == OUT / "r1" / "trained.pt"
    assert SEED_ROWS == OUT / "seed_retrain_0930" / "trained.pt"
    assert floor_dir() == SEED_ROWS.parent  # the shared floor cache
    assert floor_dir(SEED_ROWS_0921) == OUT / "rows_step1_0921_merged"
    with pytest.raises(AssertionError):
        run_dir("a b")


def test_stage_paths_take_the_run_dirs():
    """The vendored ``data_dir`` / ``arm_dir`` are ``--data_path`` /
    ``--arm_path``; there is no tag-layout fallback."""
    from types import SimpleNamespace as NS

    import common.paths as stage_paths

    a = NS(arm="rows", data_path="/d", arm_path="/a")
    assert str(stage_paths.data_dir(a)) == "/d" and str(stage_paths.arm_dir(a)) == "/a"
    b = NS(arm="rows", data_path="", arm_path="")
    with pytest.raises(AssertionError):
        stage_paths.data_dir(b)
    with pytest.raises(AssertionError):
        stage_paths.arm_dir(b)


# ---------------------------------------------------------------------------
# a run is one file


def test_run_file_is_vocabs_and_read(tmp_path):
    assert "run0925_300f" in config.run_names()
    rc = config.load_run("run0925_300f")
    assert rc.vocabs == "ja_pieces_0925_300.txt"
    assert rc.vocab_specs() == ["list:@ja_pieces_0925_300.txt"]
    assert rc.vocabs_file().is_file()
    assert rc.read == ("はい", "おしい", "やったネ", "ちょっと来い", "こんにちは")
    spec = tmp_path / "specs.toml"
    spec.write_text('vocabs = ["kana", "kanji:200"]\n', encoding="utf-8")
    rs = config.load_run(str(spec))
    assert rs.vocab_specs() == ["kana", "kanji:200"] and rs.read == ()
    for body in (
        'vocabs = "ja_pieces_0925_300.txt"\nseed = 0\n',
        'vocabs = "ja_pieces_0925_300.txt"\n[budget]\njoint = 90\n',
        'vocabs = "ja_pieces_0925_300.txt"\nwarm_from = "seed"\n',
    ):
        bad = tmp_path / "bad.toml"
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(AssertionError, match="rules in code"):
            config.load_run(str(bad))
    missing = tmp_path / "missing.toml"
    missing.write_text('vocabs = "no_such_file.txt"\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="does not exist"):
        config.load_run(str(missing))
    # the pre-collapse stage-shaped run files are records: on disk, not loadable
    with pytest.raises(AssertionError, match="rules in code"):
        config.load_run("run0923_micro")
    # context: a run name, the chain nearest first; no context = the seed rows
    assert rs.context is None and rs.context_rows() == SEED_ROWS
    b3 = config.load_run("retrain_kanji_b3")
    assert b3.context == "retrain_kanji_b2"
    assert b3.context_chain() == [
        "retrain_kanji_b2",
        "retrain_kanji_b1",
        "retrain_kana",
    ]
    for body in ('context = "a/b"\n', 'context = "self"\n'):
        bad = tmp_path / "self.toml"
        bad.write_text('vocabs = ["kana"]\n' + body, encoding="utf-8")
        with pytest.raises(AssertionError, match="context"):
            config.load_run(str(bad))


def test_front_door_parser():
    import scale

    p = scale.build_parser()
    a = p.parse_args(["run0925_300f", "data", "--workers", "3"])
    assert a.run == "run0925_300f" and a.verb == "data" and a.workers == 3
    a = p.parse_args(["run0925_300f", "train", "--submit", "--queue"])
    assert a.verb == "train" and a.submit and a.queue
    assert p.parse_args(["windows"]).verb is None
    for gone in (
        ["--run", "run0925_300f"],
        ["run0925_300f", "bake"],
        ["run0925_300f", "train", "--init_anchor", "0.1"],
        ["run0925_300f", "train", "--steps_per_row", "30"],
        ["run0925_300f", "data", "--n_items", "10"],
        ["run0925_300f", "eval", "--seed_only"],
        ["run0925_300f", "train", "--tag", "x"],
    ):
        with pytest.raises(SystemExit):
            p.parse_args(gone)
    a = p.parse_args(["rows_2900_merged", "merge", "run_a", "run_b"])
    assert (
        a.verb == "merge"
        and a.run == "rows_2900_merged"
        and a.runs == ["run_a", "run_b"]
    )
    with pytest.raises(AssertionError, match="takes no run list"):
        scale.main(["run0925_300f", "train", "run_b"])


# ---------------------------------------------------------------------------
# the recipe table by kind


def test_recipe_table_by_kind():
    from cjk_scale.builder import ITEMS_PER_VOCAB, TABLE, TIER_PX, plan_groups, tiers
    from cjk_scale.recipes import RECIPES
    from cjk_scale.windows import ROWS

    law = {(k, (r.lo, r.hi)) for r in ROWS for k in r.kinds}
    for g in TABLE:
        assert (g.kind, g.band) in law, (g.label, g.kind, g.band)
        for t in g.tiers:
            assert t.recipe in RECIPES, (t.name, t.recipe)
    # a tier is named <form>_<px>, the px its items were built at; a name is one tier
    names = [n for g in TABLE for t in g.tiers for n in t.names]
    assert len(set(names)) == len(names)
    for n in names:
        assert abs(int(n.rsplit("_", 1)[1]) - TIER_PX[n][1]) <= 1, n
    assert tiers("bubbleN_18", "bubble1_32")[1].recipe == "bubble1"
    assert {(g.kind, g.band): g.label for g in TABLE} == {
        ("single", (0.7, 0.9)): "bubble1_52 + grid_82 + lone_190",
        ("single", (0.5, 0.7)): "bubbleN_34 + bubble1_32",
        ("single", (0.3, 0.5)): "bubbleN_18",
        ("piece", (0.5, 0.7)): "piece_bubble_38 + piece_grid_29 + line_bubble_32",
        ("piece", (0.3, 0.5)): "line_bubble_19 + piece_bubble_19 + piece_grid_17",
    }
    single = [t.recipe for g in TABLE if g.kind == "single" for t in g.tiers]
    piece = [t.recipe for g in TABLE if g.kind == "piece" for t in g.tiers]
    # the lone group + the in-word groups (retrain_experiments § 3): windows + the count tier
    assert single == ["bubble1", "grid", "bubbleN", "bubble1", "bubbleN"]
    assert sorted(set(piece)) == [
        "grid_string",
        "scene_piece",
        "scene_sentence",
        "scene_short",
    ]
    assert piece.count("scene_piece") == 2  # the two px tiers
    # the single kind draws 1.5 × its items (lone 0.5 : in-word 1.0, P1b)
    for kind, total in (("single", 1.5), ("piece", 1.0)):
        assert abs(sum(g.share for g in TABLE if g.kind == kind) - total) < 1e-9
    # which groups run is which kinds the vocabs hold; the volume is one rule
    pieces_only = plan_groups({"single": [], "piece": ["p"] * 300, "multi": []})
    assert [(g.band, n) for g, n in pieces_only] == [
        ((0.5, 0.7), 10000),
        ((0.3, 0.5), 10000),
    ]
    both = plan_groups({"single": ["s"] * 3, "piece": ["p"] * 6, "multi": []})
    assert [(g.kind, g.band, n) for g, n in both] == [
        ("single", (0.7, 0.9), round(1.5 * ITEMS_PER_VOCAB)),
        ("single", (0.5, 0.7), round(1.5 * ITEMS_PER_VOCAB)),
        ("single", (0.3, 0.5), round(1.5 * ITEMS_PER_VOCAB)),
        ("piece", (0.5, 0.7), round(3 * ITEMS_PER_VOCAB)),
        ("piece", (0.3, 0.5), round(3 * ITEMS_PER_VOCAB)),
    ]


def test_tier_of_reads_the_records_of_record():
    """A record built before 2026-10-02 carries its band group and recipe;
    ``tier_of`` gives the tier that pair became."""
    from cjk_scale.builder import tier_of

    old = {"group": "b0709", "recipe": "grid_single", "band": [0.7, 0.9]}
    assert tier_of(old | {"layout": "grid"}) == "grid_82"
    assert tier_of(old | {"layout": "flat"}) == "lone_190"
    assert tier_of(old | {"recipe": "scene_single", "layout": "scene"}) == "bubble1_52"
    mid = {"group": "b0507", "band": [0.5, 0.7], "layout": "scene"}
    assert tier_of(mid | {"recipe": "scene_window"}) == "bubbleN_34"
    assert tier_of(mid | {"recipe": "scene_single_small"}) == "bubble1_32"
    assert tier_of(mid | {"recipe": "scene_piece"}) == "piece_bubble_38"
    # the experiments' small tiers (grid_small, grid_lone)
    assert (
        tier_of({"group": "l0305", "recipe": "grid_single", "band": [0.3, 0.5]})
        == "lone_16"
    )
    # a pre-collapse stage record: no group, its stage was its band
    assert tier_of({"kind": "scene_sentence", "band": [0.3, 0.5]}) == "line_bubble_19"
    # an experiment's own table, and a record that says its tier
    assert tier_of(mid | {"recipe": "scene_spelled"}) == "b0507_scene_spelled"
    assert tier_of(mid | {"tier": "grid_29"}) == "grid_29"


def test_counts_split_a_group_by_weight():
    from cjk_scale.builder import _counts

    # stage0507's live mix under run0925_300f: 5 334 / 2 000 / 2 666 of 10 000
    assert _counts(
        [("scene_piece", 0.4), ("grid_string", 0.15), ("scene_short", 0.2)], 10000
    ) == {
        "scene_piece": 5334,
        "grid_string": 2000,
        "scene_short": 2666,
    }
    assert sum(_counts([("a", 0.6), ("b", 0.4)], 4001).values()) == 4001


def test_restart_puts_the_draw_state_back():
    """Every group restarts from the pools' post-build state — the rng
    a stage build of record continued with, the canvas rng, empty counters."""
    from collections import Counter
    from types import SimpleNamespace as NS

    from cjk_scale.builder import _restart

    rng, shapes = random.Random(0), NS(rng=random.Random(17))
    pools = NS(shapes=shapes, used=Counter(), decks={}, balanced={})
    snap = (rng.getstate(), shapes.rng.getstate())
    first = (rng.random(), shapes.rng.random())
    pools.used[3] += 1
    pools.decks["x"] = 1
    pools.balanced["short"] = Counter({"a": 1})
    _restart(pools, rng, snap)
    assert (rng.random(), shapes.rng.random()) == first
    assert not pools.used and not pools.decks and not pools.balanced


def test_missing_source_reads_the_pools():
    from types import SimpleNamespace as NS

    from cjk_scale.recipes import missing_source

    empty = NS(pieces=[], singles=list("あい"), digraphs=[], phrase={})
    full = NS(
        pieces=["それを", "はじ", "やはり", "すご"],
        singles=list("あいおなアナラル人日口女精聞動願"),
        digraphs=[],
        phrase={"short": ["それを はじ"], "sentence": []},
    )
    assert missing_source("scene_short", {}, empty) == "no short lines"
    assert missing_source("scene_sentence", {}, full) == "no sentence lines"
    assert missing_source("scene_piece", {}, empty) == "no pieces"
    nosingle = NS(pieces=["それを"], singles=[], digraphs=[], phrase={})
    assert missing_source("bubble1", {}, nosingle) == "no singles"
    assert missing_source("bubble1", {}, empty) is None
    assert missing_source("scene_piece", {}, full) is None
    assert missing_source("grid_string", {"source": "both"}, full) is None
    assert missing_source("grid_string", {"source": "short"}, empty)
    assert missing_source("grid", {"grids": "3x3"}, empty)
    assert missing_source("grid", {"grids": "1x1,3x3"}, empty) is None


def test_fit_px_is_the_bubble_capacity():
    from cjk_scale.recipes import _fill_for_px, _fit_px

    # 100 wide × 210 tall: two glyphs down one column → 210 / (2 × 1.05) = 100
    assert _fit_px([0, 0, 100, 210], 2, True) == pytest.approx(100.0)
    # a 20 px target fills a fifth of it; the fill is that ratio
    assert _fill_for_px([0, 0, 100, 210], 2, 20, True, 0.9) == pytest.approx(0.2)


def test_primitives_exist():
    from common.render.scene import H_PITCH, V_PITCH, region_capacity, render_into_scene  # noqa: F401
    from data.grid import _Deck, render_grid  # noqa: F401


def test_horizontal_marker_and_grid_share():
    """30 % of multi-glyph items are drawn as lines and say so: the scene
    caption's marker per frame, the grid's per-cell draw."""
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
    assert any(str(LINE / "assets" / "fonts") in f for f in fonts), "the line's fonts"
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
    assert config.DATA["horizontal_frac"] == 0.3
    assert config.DATA["horizontal_scenes"] == "sl1w"


# ---------------------------------------------------------------------------
# train


def test_vocab_idx_is_the_tokenizer_map():
    """The vocabs file is the inventory, the tokenizer maps it (plan.md § 4)."""
    from cjk_scale.train import vocab_idx

    class Tok:
        ids = {"あ": [186], "ちょっと": [901], "それを": [55]}

        def encode(self, text, add_special_tokens=False):
            return self.ids[text]

        def decode(self, ids):
            return {186: "あ", 901: "ちょっと", 55: "それを"}[ids[0]]

    qmap = {186: 186, 901: 40001, 55: 26585}
    assert vocab_idx(["あ", "ちょっと", "それを"], (Tok(), qmap)) == {186, 40001, 26585}


def test_items_carry_their_band(tmp_path):
    from cjk_scale.train import load_items

    rec = {
        "file": "x.png",
        "text": "あ",
        "caption": "c",
        "src": "scene",
        "shape": [512, 512],
    }
    (tmp_path / "eval.json").write_text("[]", encoding="utf-8")
    (tmp_path / "vocabs.json").write_text('["あ"]', encoding="utf-8")
    (tmp_path / "train.jsonl").write_text(
        json.dumps({**rec, "band": [0.7, 0.9]}) + "\n", encoding="utf-8"
    )
    recs, ev, vocabs = load_items(tmp_path)
    assert recs[0]["band"] == [0.7, 0.9] and ev == [] and vocabs == ["あ"]
    (tmp_path / "train.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="band"):
        load_items(tmp_path)


def test_box_share_curve():
    from cjk_scale.loss import box_share_of, glyph_count

    assert glyph_count("じゃ ない") == 4 and glyph_count("") == 1
    assert box_share_of(1, 0.25, 0.5, 8) == 0.25
    assert abs(box_share_of(2, 0.25, 0.5, 8) - (0.25 + 0.25 / 3)) < 1e-9
    assert abs(box_share_of(4, 0.25, 0.5, 8) - (0.25 + 0.5 / 3)) < 1e-9
    assert box_share_of(8, 0.25, 0.5, 8) == 0.5 == box_share_of(40, 0.25, 0.5, 8)
    assert box_share_of(5, 0.25, 0.5, 1) == 0.25  # n_cap 1: flat


def test_grid_box_union_mask():
    """grid_box: a grid item's cells become the loss box (union); off, and for
    a flat 1×1, the item is the plain canvas mean."""
    import torch

    from cjk_scale.loss import box_mask, box_share_fm_loss, item_boxes

    scene = {"layout": "scene", "src": "scene", "text": "聞", "box": [64, 64, 128, 128]}
    grid = {
        "layout": "grid",
        "src": "grid",
        "text": "はじ ファ",
        "boxes": [[0, 0, 64, 32], [128, 128, 192, 160]],
    }
    flat = {
        "layout": "flat",
        "src": "font",
        "text": "口",
        "boxes": [[32, 32, 224, 224]],
    }
    assert item_boxes(scene, False) == [[64, 64, 128, 128]] == item_boxes(scene, True)
    assert item_boxes(grid, False) == [] and item_boxes(grid, True) == grid["boxes"]
    assert item_boxes(flat, False) == [] == item_boxes(flat, True)
    m = box_mask((3, 4, 32, 32), [scene, grid, flat], "cpu", grid_box=True)
    assert m[0].sum() == 8 * 8 and m[1].sum() == 8 * 4 + 8 * 4 and m[2].sum() == 0
    assert (
        box_mask((3, 4, 32, 32), [scene, grid, flat], "cpu", grid_box=False)[1].sum()
        == 0
    )
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(1, 4, 32, 32, generator=g)
    target = torch.zeros_like(pred)
    plain = ((pred - target) ** 2).mean()
    off = box_share_fm_loss(pred, target, [grid], 0.25, 0.5, 8, grid_box=False)
    on = box_share_fm_loss(pred, target, [grid], 0.25, 0.5, 8, grid_box=True)
    assert torch.isclose(off, plain)
    se = ((pred - target) ** 2).mean(dim=1)[0]
    m1 = box_mask((1, 4, 32, 32), [grid], "cpu", grid_box=True)[0, 0].bool()
    s = 0.25 + 0.25 * (torch.log(torch.tensor(4.0)) / torch.log(torch.tensor(8.0)))
    want = s * se[m1].mean() + (1 - s) * se[~m1].mean()
    assert torch.isclose(on, want, atol=1e-6)
    assert torch.isclose(
        box_share_fm_loss(pred, target, [flat], 0.25, 0.5, 8, grid_box=True), plain
    )


# ---------------------------------------------------------------------------
# eval


def _rc(name="t1", read=("はい",)):
    return config.RunConfig(name=name, path=LINE / "x.toml", vocabs="v.txt", read=read)


def test_stage_eval_namespace_builds():
    import common.paths as stage_paths
    from cjk_scale import eval as ev

    rc = _rc()
    a = ev.probe_args(rc, ev.FLOOR_ARM, ["eval", "native"], ["--eval_limit", "3"])
    assert stage_paths.data_dir(a) == data_dir("t1")
    assert stage_paths.arm_dir(a) == floor_dir()
    assert a.arm == "rows" and a.no_floor and a.seeds == 2 and a.eval_limit == 3
    assert a.native_chars == "あ,い" and a.native_clauses == "en,swap"
    assert a.steps == 28 and a.cfg == 4.0 and a.seed == 0
    s = ev.ruler_args(rc, ev.TRAINED_ARM, "sent")
    assert (
        s.eval_tag == "sent" and s.native_chars == "はい" and s.native_clauses == "en"
    )
    # the trained arm is the run dir itself — no ctx sidecar
    assert stage_paths.arm_dir(s) == run_dir("t1")
    assert ev.rulers(rc) == ["eval", "native", "sent", "target"]
    assert ev.rulers(_rc(read=())) == ["eval", "native", "target"]


def test_piece_ruler(tmp_path, monkeypatch):
    """A run whose vocabs hold a multi-glyph vocab gets the piece ruler — a
    trained piece alone in a native scene, en + swap (reports/piece_2026_09_25.md:
    the one ruler that sees piece identity); a singles-only run does not."""
    from cjk_scale import eval as ev
    from cjk_scale import paths

    monkeypatch.setattr(paths, "OUT", tmp_path)
    rc = _rc()
    d = paths.data_dir("t1")
    d.mkdir(parents=True)
    (d / "vocabs.json").write_text(json.dumps(["あ"]), encoding="utf-8")
    assert ev.rulers(rc) == ["eval", "native", "single", "sent", "target"]
    (d / "vocabs.json").write_text(json.dumps(["すごい"]), encoding="utf-8")
    assert ev.rulers(rc) == ["eval", "native", "piece", "sent", "target"]
    (d / "vocabs.json").write_text(json.dumps(["あ", "すごい"]), encoding="utf-8")
    assert ev.rulers(rc) == ["eval", "native", "piece", "single", "sent", "target"]
    monkeypatch.setattr(ev, "piece_vocabs", lambda rc: ("すごい", "った"))
    a = ev.ruler_args(rc, ev.TRAINED_ARM, "piece")
    assert a.eval_tag == "piece" and a.native_chars == "すごい,った"
    assert a.native_clauses == "en,swap"
    assert ev.floor_keys(rc, "piece") == {
        "すごい|en",
        "すごい|swap",
        "った|en",
        "った|swap",
    }


def test_merge_seed(tmp_path):
    """The merge at save: the run's rows kept, every seed row the run lacks
    appended × (seed row_scale / the run's), ids sorted."""
    import torch

    from cjk_scale.rows import merge_seed

    seed = tmp_path / "seed.pt"
    torch.save(
        {
            "delta": {
                "ext_ids": [10, 11, 12],
                "raw": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "row_scale": 1.0,
            }
        },
        seed,
    )
    delta = {"ext_ids": [11], "raw": torch.tensor([[9.0, 9.0]]), "row_scale": 2.0}
    merged, n = merge_seed(delta, seed)
    assert n == 2 and merged["ext_ids"] == [10, 11, 12]
    assert torch.equal(
        merged["raw"], torch.tensor([[0.5, 1.0], [9.0, 9.0], [2.5, 3.0]])
    )
    assert merged["row_scale"] == 2.0


def test_floor_cache_and_merged_guard(tmp_path, monkeypatch):
    """The floor arm is the seed rows' dir — one read cache for every run
    (2026-09-26): a key it holds is never re-rendered, another run's keys stay
    out of this run's floor reads, and an older per-run floor/ folds in by
    copy. Eval refuses a pre-merge vocabs-only trained.pt (no ``seed_merged``)."""
    import torch
    from PIL import Image

    from cjk_scale import eval as ev
    from cjk_scale import paths

    monkeypatch.setattr(paths, "OUT", tmp_path)
    rc = _rc()
    assert ev.arm_out(rc, ev.FLOOR_ARM) == tmp_path / "seed_retrain_0930"
    assert ev.arm_out(rc, ev.TRAINED_ARM) == tmp_path / "t1"

    def rec(text, clause, d):
        f = d / "img" / f"trained_p00_{text}_{clause}_s0.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(f)
        return {
            "file": str(f),
            "cond": "trained",
            "seed": 0,
            "pi": 0,
            "text": text,
            "clause": clause,
            "reads": [],
        }

    # an old per-run floor/ with a sent read → folded into the cache (copied)
    old = tmp_path / "old_run" / "floor"
    (old / "native_sent").mkdir(parents=True)
    (old / "native_sent" / "native_reads.json").write_text(
        json.dumps(
            [
                rec("はい", "en", old / "native_sent"),
                rec("おしい", "en", old / "native_sent"),
            ]
        ),
        encoding="utf-8",
    )
    assert ev.import_floor(old) == {"sent": 2}
    assert ev.import_floor(old) == {"sent": 0}  # keys the cache holds win
    assert (old / "native_sent" / "img" / "trained_p00_はい_en_s0.png").exists()
    cached = paths.floor_dir() / "native_sent" / "img" / "trained_p00_はい_en_s0.png"
    assert cached.exists()
    # cached → no render (the stage is never reached); this run sees its key only
    monkeypatch.setattr(
        "stages.run", lambda *a, **k: pytest.fail("rendered a cached key")
    )
    assert ev.ensure_floor(rc, "sent") == 0
    got = ev._reads(rc, ev.FLOOR_ARM, "sent")
    assert [m["text"] for m in got] == ["はい"] and got[0]["file"] == str(cached)

    (tmp_path / "t1").mkdir()
    torch.save(
        {
            "delta": {
                "ext_ids": [12, 20],
                "raw": torch.full((2, 2), 7.0),
                "row_scale": 3.0,
            },
            "arm": "rows",
            "killed": "",
        },
        tmp_path / "t1" / "trained.pt",
    )
    with pytest.raises(AssertionError, match="seed_merged"):
        ev.run(rc)


def test_routed_run_reads_routed(tmp_path, monkeypatch):
    """A data dir built with windows (``build.json`` ``glyph_route``) reads
    routed: eval sets ``ANIMA_VOCAB_GLYPH_ROUTE=1`` in-process and the floor
    arm is the seed rows' ``routed/`` cache (the seed's trained.pt linked in),
    never the unrouted cache of record."""
    import torch

    from cjk_scale import eval as ev
    from cjk_scale import paths

    monkeypatch.setattr(paths, "OUT", tmp_path)
    monkeypatch.delenv("ANIMA_VOCAB_GLYPH_ROUTE", raising=False)
    rc = _rc()
    seed = paths.floor_dir()
    seed.mkdir(parents=True)
    torch.save({"seed": 1}, seed / "trained.pt")
    d = paths.data_dir("t1")
    d.mkdir(parents=True)
    assert not ev.routed(rc) and ev.arm_out(rc, ev.FLOOR_ARM) == seed
    (d / "build.json").write_text(json.dumps({"glyph_route": True}), encoding="utf-8")
    assert ev.routed(rc)
    assert ev.arm_out(rc, ev.FLOOR_ARM) == seed / "routed"
    assert torch.load(seed / "routed" / "trained.pt") == {"seed": 1}
    assert ev.floor_arm_dir(rc) == seed / "routed"  # idempotent
    with pytest.raises(AssertionError):  # no trained.pt: stops after the env is set
        ev.run(rc)
    assert os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] == "1"


def test_compose_one_sheet(tmp_path, monkeypatch):
    """Both arms' reads → reads.json (official / loose / contained per string
    and per group) + sheet.png; the floor side is the cache restricted to
    the run's keys."""
    from PIL import Image

    from cjk_scale import eval as ev
    from cjk_scale import paths

    monkeypatch.setattr(paths, "OUT", tmp_path)
    rc = _rc()
    img = tmp_path / "x.png"
    Image.new("RGB", (32, 32), "white").save(img)
    d = paths.data_dir("t1")
    d.mkdir(parents=True)
    (d / "eval.json").write_text(
        json.dumps([{"group": "word", "text": "すごい", "caption": "c"}]),
        encoding="utf-8",
    )

    def box(sfx, vl):
        return [{"box": [0, 0, 8, 8], "whole": False, "sfx": sfx, "vl": vl}]

    for arm, word_read, sent_read in (
        ("floor", "すずごい", "は"),
        ("trained", "すごい", "はい"),
    ):
        d = ev.arm_out(rc, arm)
        (d / "native_sent").mkdir(parents=True)
        (d / "eval_reads.json").write_text(
            json.dumps(
                [
                    {
                        "file": str(img),
                        "cond": "trained",
                        "seed": s,
                        "group": "word",
                        "text": "すごい",
                        "reads": box(word_read, word_read),
                        "exact": word_read == "すごい",
                        "cer_vl": 0.0 if word_read == "すごい" else 0.5,
                    }
                    for s in (0, 1)
                ]
            ),
            encoding="utf-8",
        )
        (d / "native_sent" / "native_reads.json").write_text(
            json.dumps(
                [
                    {
                        "file": str(img),
                        "cond": "trained",
                        "seed": s,
                        "pi": pi,
                        "text": "はい",
                        "clause": "en",
                        "reads": box(sent_read, "はい"),
                        "hit_sfx": sent_read == "はい",
                        "hit_vl": True,
                    }
                    for pi in (0, 1)
                    for s in (0, 1)
                ]
            ),
            encoding="utf-8",
        )
    out = ev.compose(rc)
    r = json.loads((out / "reads.json").read_text(encoding="utf-8"))
    word = r["rulers"]["eval"]["totals"]["word"]
    assert word["floor"] == {"n": 2, "official": 0, "loose": 0, "contained": 0}
    assert word["trained"] == {"n": 2, "official": 2, "loose": 2, "contained": 2}
    sent = r["rulers"]["sent"]["strings"]["はい|en"]
    assert sent["floor"] == {"n": 4, "official": 0, "loose": 4, "contained": 4}
    assert sent["trained"] == {"n": 4, "official": 4, "loose": 4, "contained": 4}
    assert (out / "sheet.png").exists()
    assert ev.floor_keys(rc, "sent") == {"はい|en"}
    assert ev.floor_keys(rc, "target") is None


def test_ext_delta_rows_and_no_line():
    """``ExtDelta`` adds each ext id's row × row_scale, round-trips its
    state, and refuses a state carrying ``line`` (line mode removed
    2026-09-28)."""
    import torch
    from types import SimpleNamespace

    from common.hooks import ExtDelta

    T = 32128
    emb = torch.nn.Embedding(T + 10, 4)
    torch.nn.init.zeros_(emb.weight)
    anima = SimpleNamespace(llm_adapter=SimpleNamespace(embed=emb))
    d = ExtDelta(anima, [1, 2, 3], 4, "cpu", row_scale=2.0)
    with torch.no_grad():
        d.raw.copy_(torch.eye(3, 4))
    ids = torch.tensor([[5, T + 1, 7, T + 2, T + 3, 9]])
    rows = torch.zeros(6, 4)
    for p, e in enumerate(ids[0].tolist()):
        if e >= T:
            rows[p] = d.raw.detach()[d.index[e - T]] * 2.0
    assert torch.allclose(emb(ids)[0], rows)
    sd = d.state_dict()
    assert "line" not in sd
    for h in d.handles:
        h.remove()
    d2 = ExtDelta.from_state(anima, sd, "cpu")
    assert torch.allclose(emb(ids)[0], rows)
    for h in d2.handles:
        h.remove()
    with pytest.raises(AssertionError, match="line mode was removed"):
        ExtDelta.from_state(anima, {**sd, "line": torch.zeros(4)}, "cpu")


class _Tok:
    """One Qwen token per vocab here, except the two-token digraph あっ."""

    ids = {"精": [1], "山": [2], "すごい": [3], "って": [4], "あっ": [5, 6]}

    def encode(self, text, add_special_tokens=False):
        return self.ids[text]

    def decode(self, ids):
        return {1: "精", 2: "山", 3: "すごい", 4: "って", 5: "あ", 6: "っ"}[ids[0]]


_QMAP = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106}


def test_budget_rule():
    """Singles start cold (plan_retrain): cold kanji split by ink (< 10 →
    225 / row, ≥ 10 → 337.5 with the mix), cold kana P1b's (the base),
    4–5-glyph pieces the long_b0 row, everything else the base; singles may
    mix budgets in a run (weighted draws), other kinds may not."""
    from cjk_scale import budget

    assert budget.factor("single", 1, False, "kanji", 7.4) == 150 / budget.BASE_STEPS
    assert budget.factor("single", 1, False, "kanji", 10.0) == 225 / budget.BASE_STEPS
    with pytest.raises(AssertionError, match="ink"):
        budget.factor("single", 1, False, "kanji")
    assert budget.glyph_ink("山") < 10 <= budget.glyph_ink("精")  # the pinned table
    assert budget.factor("single", 1, False, "kana") == 1.0
    assert budget.factor("single", 1, True) == 1.0
    assert budget.script_of("精") == "kanji" and budget.script_of("ー") == "kana"
    assert budget.factor("piece", 3, True) == 1.0
    assert budget.factor("piece", 4, True) == 3.0
    assert budget.factor("piece", 5, False) == 3.0
    tokq, seeds = (_Tok(), _QMAP), frozenset({101, 103, 105, 106})
    f = budget.vocab_factors(["精", "山", "すごい", "って", "あっ"], tokq, seeds)
    # 精 has a seed row and still starts cold; 精 is ink-dense, 山 is not
    assert f == {
        "精": 225 / 90,
        "山": 150 / 90,
        "すごい": 1.0,
        "って": 1.0,
        "あっ": 1.0,
    }
    assert budget.run_factor(["すごい", "って", "あっ"], tokq, seeds) == 1.0
    assert budget.run_factor(["山"], tokq, seeds) == 150 / 90
    with pytest.raises(AssertionError, match="different budgets"):
        budget.run_factor(["精", "山"], tokq, seeds)
    # singles mix: 2 : 3 in the draw pools; a one-budget run draws unweighted
    b = budget.run_budget(["精", "山"], tokq, seeds)
    assert b == {"精": 225 / 90, "山": 150 / 90}
    assert budget.draw_weights(b) == {"精": 3, "山": 2}
    assert budget.draw_weights({"す": 1.0, "ご": 1.0}) == {"す": 1, "ご": 1}
    assert budget.mix_factor({"single"}) == 1.5
    assert budget.mix_factor({"piece"}) == 1.0


def test_pieces_char_rows():
    """A byte-split glyph (緒: two row-less fragment tokens) comes back as
    itself with its char row on a ``GlyphRows`` map, as fragments on the
    plain map; a fragment run with no char row stays fragments."""
    from data.inventory import GlyphRows, pieces

    class Tok:
        enc = {"緒": [7, 8], "戻る": [7, 9, 3], "謎": [7, 10]}
        dec = {3: "る", 7: "\ufffd", 8: "\ufffd", 9: "\ufffd", 10: "\ufffd"}
        joint = {(7, 8): "緒", (7, 9): "戻", (7, 10): "謎"}

        def encode(self, text, add_special_tokens=False):
            return self.enc[text]

        def decode(self, ids):
            if len(ids) == 1:
                return self.dec[ids[0]]
            return self.joint.get(tuple(ids), "\ufffd")

    plain = {3: 179}
    q = GlyphRows(plain)
    q.char = {"緒": 37919, "戻": 33742}
    assert pieces(Tok(), plain, "緒") == [("\ufffd", None), ("\ufffd", None)]
    assert pieces(Tok(), q, "緒") == [("緒", 37919)]
    assert pieces(Tok(), q, "戻る") == [("戻", 33742), ("る", 179)]
    assert pieces(Tok(), q, "謎") == [("\ufffd", None), ("\ufffd", None)]


def test_plan_groups_take_the_budget():
    from cjk_scale.builder import ITEMS_PER_VOCAB, plan_groups

    got = plan_groups({"single": ["s"] * 3, "piece": [], "multi": []}, budget=150 / 90)
    n = round(3 * ITEMS_PER_VOCAB * 150 / 90 * 0.5)
    assert [(g.band, n) for g, n in got] == [
        ((0.7, 0.9), n),
        ((0.5, 0.7), n),
        ((0.3, 0.5), n),
    ]
    # per vocab: a kind's items are Σ of its vocabs' factors
    got = plan_groups(
        {"single": ["a", "b"], "piece": [], "multi": []},
        budget={"a": 150 / 90, "b": 225 / 90},
    )
    n = round(ITEMS_PER_VOCAB * (150 + 225) / 90 * 0.5)
    assert {m for _g, m in got} == {n}


def test_ruler_sample():
    """All up to RULER_N; past it, equal per glyph-count stratum (a short
    stratum's share to the rest), seeded."""
    from cjk_scale.eval import RULER_N, ruler_sample

    small = [f"あ{i}" for i in range(5)]
    assert ruler_sample(small) == small
    by = {
        2: [f"{c}い" for c in "かきくけこさしすせそたちつてと"],  # 15
        3: [f"{c}いう" for c in "かきくけこさしすせそたちつてと"],  # 15
        4: ["かいうえ", "きいうえ"],  # 2
    }
    pool = [v for g in (2, 3, 4) for v in by[g]]
    got = ruler_sample(pool)
    assert len(got) == RULER_N == 24 and len(set(got)) == 24
    n = {g: sum(v in by[g] for v in got) for g in by}
    assert n == {2: 11, 3: 11, 4: 2}
    assert ruler_sample(pool) == got  # seeded


def test_single_ruler(tmp_path, monkeypatch):
    from cjk_scale import eval as ev
    from cjk_scale import paths

    monkeypatch.setattr(paths, "OUT", tmp_path)
    rc = _rc()
    monkeypatch.setattr(ev, "single_vocabs", lambda rc: ("精", "輩"))
    a = ev.ruler_args(rc, ev.TRAINED_ARM, "single")
    assert a.eval_tag == "single" and a.native_chars == "精,輩"
    assert a.native_clauses == "en"
    assert ev.floor_keys(rc, "single") == {"精|en", "輩|en"}
    assert ev.READ_FILES["single"] == "native_single/native_reads.json"


def _rows_file(path, ids, raw, scale, **extra):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta": {
                "ext_ids": ids,
                "raw": torch.tensor(raw),
                "row_scale": scale,
                **extra,
            },
            "arm": "rows",
            "seed_merged": "seed.pt",
        },
        path,
    )


def test_merge_rows(tmp_path, monkeypatch):
    """The base whole; each other run's own idx over it, × (its row_scale /
    the base's), appended (ids sorted) where the base lacks one; overlap and
    a line mode refused."""
    import torch

    from cjk_scale import merge as mg

    monkeypatch.setattr(mg, "run_dir", lambda r: tmp_path / r)
    monkeypatch.setattr(mg, "trained_path", lambda r: tmp_path / r / "trained.pt")
    monkeypatch.setattr("data.inventory.qwen_pieces", lambda **k: None)
    idx = {"a": {10}, "b": {11, 13}, "c": {10}, "l": {12}}
    monkeypatch.setattr(mg, "run_idx", lambda r, tokq: idx[r])
    monkeypatch.setattr(mg, "idx_source", lambda r: tmp_path / r / "vocabs.json")
    _rows_file(tmp_path / "a" / "trained.pt", [10, 11, 12], [[1.0], [2.0], [3.0]], 1.0)
    _rows_file(
        tmp_path / "b" / "trained.pt",
        [10, 11, 12, 13],
        [[0.0], [5.0], [0.0], [7.0]],
        2.0,
    )
    out = mg.merge("m", ["a", "b"])
    sd = torch.load(out / "trained.pt", weights_only=False)
    assert sd["delta"]["ext_ids"] == [10, 11, 12, 13]
    assert torch.equal(sd["delta"]["raw"], torch.tensor([[1.0], [10.0], [3.0], [14.0]]))
    assert sd["delta"]["row_scale"] == 1.0 and sd["seed_merged"] == "seed.pt"
    rec = json.loads((out / "merge.json").read_text())
    assert rec["sources"][0]["rescale"] == 2.0 and rec["sources"][0]["added"] == 1
    with pytest.raises(AssertionError, match="already holds rows"):
        mg.merge("m", ["a", "b"])
    _rows_file(tmp_path / "c" / "trained.pt", [10], [[9.0]], 1.0)
    with pytest.raises(AssertionError, match="both trained"):
        mg.merge("m2", ["a", "c"])
    _rows_file(tmp_path / "l" / "trained.pt", [12], [[9.0]], 1.0, line=torch.zeros(1))
    with pytest.raises(AssertionError, match="line mode"):
        mg.merge("m3", ["a", "l"])
    # a cut run: refused as trained.pt, taken as <run>@partial (step recorded)
    (tmp_path / "p").mkdir()
    for f in ("trained.pt", "trained_partial.pt"):
        torch.save(
            {
                "delta": {
                    "ext_ids": [12],
                    "raw": torch.tensor([[4.0]]),
                    "row_scale": 1.0,
                },
                "arm": "rows",
                "seed_merged": "seed.pt",
                "step": 5,
                "args": {"train_steps": 9},
            },
            tmp_path / "p" / f,
        )
    idx["p"] = {12}
    with pytest.raises(AssertionError, match="stopped early"):
        mg.merge("m4", ["a", "p"])
    with pytest.raises(AssertionError, match="no rows"):
        mg.merge("m4", ["a", "b@partial"])
    out = mg.merge("m4", ["a", "p@partial"])
    sd = torch.load(out / "trained.pt", weights_only=False)
    assert sd["delta"]["raw"][2].item() == 4.0 and "step" not in sd
    src = json.loads((out / "merge.json").read_text())["sources"][0]
    assert src["run"] == "p" and (src["step"], src["train_steps"]) == (5, 9)
    assert src["path"].endswith("trained_partial.pt")


def test_window_pool():
    """Letters only, no repeated glyph, the read strings held out by trigram
    (a 2-glyph read: the string itself); windows cross word boundaries."""
    from cjk_scale.recipes import window_glyphs, window_pool

    g = window_glyphs(list("あいうえおかきゲーム。"))
    assert "。" not in g and "ー" in g
    ws = window_pool(g, ["あいう。えお", "かあかき"], held=())
    assert "あいう" in ws and "えお" in ws
    assert not any("。" in w for w in ws)
    assert "かあか" not in ws and "あかき" in ws  # no repeat inside a window
    ws = window_pool(g, ["あいうえお"], held=("いうえ", "かき"))
    assert "あいう" in ws and "いうえ" not in ws and "あいうえ" not in ws
    assert window_pool(g, ["かき"], held=("かき",)) == []
