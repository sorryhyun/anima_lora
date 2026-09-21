"""Imports, the CLI golden dump, and the pure helpers the data stage's
bit-identity rests on."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from cli_dump import SRC, dump

MODULES = sorted(
    ".".join(p.relative_to(SRC).with_suffix("").parts).removesuffix(".__init__")
    for p in SRC.rglob("*.py")
    # probe/ bench/ are scripts that do work at import time (quote_dir_save)
    if p.parts[len(SRC.parts)] not in ("probe", "bench") and p.name != "wake_probe.py"
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_from_src(name):
    mod = importlib.import_module(name)
    assert Path(mod.__file__).resolve().is_relative_to(SRC), mod.__file__


def test_stages_resolve():
    from stages import ALL, STAGES

    for stage, (module, fn) in STAGES.items():
        assert callable(getattr(importlib.import_module(module), fn)), stage
    assert set(ALL) <= set(STAGES)


def test_cli_golden():
    golden = Path(__file__).parent / "fixtures" / "cli_golden.json"
    assert dump() == json.loads(golden.read_text(encoding="utf-8"))


def test_parse_units_canonical_order():
    from data.units import KINDS, parse_units

    typed = ["list:、,。", "words:100/held=8", "kanji:200", "kana_ext", "kana"]
    a = parse_units(typed)
    b = parse_units(list(reversed(typed)))
    assert [s.spec() for s in a] == [s.spec() for s in b]
    assert [s.kind for s in a] == sorted((s.kind for s in a), key=KINDS.index)
    assert [s.spec() for s in parse_units(None)] == ["kana"]
    assert parse_units(["list:、,！！*3"])[0].units == ["、", "！！"]


def test_pool_small_kana_mass():
    """`--units small`: every small kana carries the pool mass of one weight-1
    unit, and a recipe without the source keeps its pool byte for byte."""
    from data.units import SMALL_PER, Inventory, parse_units

    inv = Inventory(sources=parse_units(["kana", "kana_ext*1"]))
    inv.kana, inv.kana_ext = ["あ", "き"], ["が", "ゃ"]
    assert inv.pool() == ["あ", "き", "が"]
    inv.sources = parse_units(["kana", "kana_ext*1", "small"])
    inv.small_of = {"ゃ": ["きゃ"] * SMALL_PER}
    pool = inv.pool()
    assert pool.count("あ") == pool.count("が") == SMALL_PER
    assert sum("ゃ" in u for u in pool) == SMALL_PER and "ゃ" not in pool


def test_parse_shapes():
    from common.shapes import parse_shape, parse_shapes, wh

    assert parse_shape("512") == (512, 512)
    assert parse_shape(" 384X512 ") == (384, 512)
    assert parse_shapes("448,512:2,448x512") == [
        (448, 448, 1.0),
        (512, 512, 2.0),
        (448, 512, 1.0),
    ]
    assert wh(512) == (512, 512) and wh((384, 512)) == (384, 512)
    with pytest.raises(AssertionError):
        parse_shapes("500")


def test_norm_cer_punctuation():
    from common.text import cer, norm

    assert norm("「あ、い。」！？ ・・・") == "あい"
    assert norm("ＡＢ c") == "abc"
    assert cer("あい！", "「あい」") == 0.0
    assert cer("あう", "あい") == 0.5
    assert cer("x", "・・・") == 1.0
    assert cer("あいうえお", "あ") == 1.0


def test_split_lines_kinsoku():
    from common.render.scene import NO_HEAD, NO_TAIL, split_lines

    for text, k in [
        ("こんにちは、せかい", 2),
        ("まって「ほんとうに」だよ", 2),
        ("ちょっとまってくださいね", 3),
        ("あっ、そうだったのか？", 2),
    ]:
        lines = split_lines(text, k)
        assert lines and "".join(lines) == text and len(lines) == k
        for prev, line in zip(lines, lines[1:]):
            assert line[0] not in NO_HEAD, lines
            assert prev[-1] not in NO_TAIL, lines
    assert split_lines("あい", 3) is None
    assert split_lines("あいうえ", 2, cuts=[2]) == ["あい", "うえ"]
    # every later cut violates: the nearest cut stands rather than refusing
    assert split_lines("あ・・・", 2) is not None


def test_render_into_scene_sibling(tmp_path):
    """ΔFM (plan_synth2): the sibling is drawn by the same fit and differs
    from the item only inside the union text box; the reference never carries
    the target's romaji letters."""
    import random

    import numpy as np
    from common.render.flat import find_fonts
    from common.render.scene import render_into_scene
    from data.pair import RefPool, romaji_letters
    from PIL import Image

    fonts = find_fonts()
    assert fonts, "assets/fonts is empty"
    Image.new("RGB", (512, 512), "white").save(tmp_path / "scene.png")
    scene = {
        "i": 0,
        "file": str(tmp_path / "scene.png"),
        "shape": [512, 512],
        "regions": [[96, 96, 416, 416]],
        "region": [96, 96, 416, 416],
        "boxes_anchor": [[160, 200, 352, 312]],
        "bubbles": [None],
    }
    for text in ("す", "がんばれ"):
        rng = random.Random(3)
        ref = RefPool(4, rng).draw(0, text)
        assert len(ref) == len(text) and not (set(ref) & romaji_letters(text))
        drawn = render_into_scene(
            scene, text, fonts[0], rng, min_glyph=28, tilt_frac=1.0, ref_text=ref
        )
        assert drawn is not None and len(drawn) == 4
        im_b, box_b, im_a, box_a = drawn
        diff = (np.array(im_b) != np.array(im_a)).any(axis=2)
        assert diff.any()  # the glyphs differ ...
        u = [
            min(box_b[0], box_a[0]),
            min(box_b[1], box_a[1]),
            max(box_b[2], box_a[2]),
            max(box_b[3], box_a[3]),
        ]
        diff[u[1] : u[3], u[0] : u[2]] = False
        assert not diff.any()  # ... and nothing else does
        # same draw without the sibling is bit-identical to the item
        rng = random.Random(3)
        RefPool(4, rng).draw(0, text)
        alone, box_alone = render_into_scene(
            scene, text, fonts[0], rng, min_glyph=28, tilt_frac=1.0
        )
        assert box_alone == box_b and np.array_equal(np.array(alone), np.array(im_b))


def test_render_string_flat_sibling():
    """ΔFM flat sibling: two strings rendered with each other as ``fit_text``
    under one layout share canvas, bubble and block centre — they differ
    under the glyphs only; without ``fit_text`` the render is unchanged."""
    import random

    import numpy as np
    from common.render.flat import find_fonts, render_string, sample_layout

    fonts = find_fonts()
    for text, ref, mode, seed in (
        ("が", "K", "v1", 1),
        ("こんにちは", "QXZPL", "jitter", 2),
        ("日", "M", "jitter", 4),
    ):
        rng = random.Random(seed)
        lay = sample_layout(len(text), rng, (448, 576), mode, 0.6)
        lay["rot"] = None  # the extent check below is axis-aligned
        im_b, bub_b = render_string(
            text, fonts[0], rng, size=(448, 576), layout=lay, fit_text=ref
        )
        im_a, bub_a = render_string(
            ref, fonts[0], rng, size=(448, 576), layout=lay, fit_text=text
        )
        assert bub_a == bub_b
        diff = (np.array(im_b) != np.array(im_a)).any(axis=2)
        ys, xs = np.nonzero(diff)
        assert len(ys)  # the glyphs differ ...
        # ... inside one text block, not across the canvas (a moved bubble
        # or centre would spread the difference)
        n = len(text)
        long_side = 1.2 * lay["fs"] * n + 3 * lay.get("stroke", 0) + 8
        assert max(ys.max() - ys.min(), xs.max() - xs.min()) <= long_side
    rng_a, rng_b = random.Random(9), random.Random(9)
    lay = sample_layout(1, rng_a, 512, "jitter", 0.6)
    sample_layout(1, rng_b, 512, "jitter", 0.6)
    one, _ = render_string("が", fonts[0], rng_a, size=512, layout=lay)
    same, _ = render_string("が", fonts[0], rng_b, size=512, layout=lay, fit_text="が")
    assert np.array_equal(np.array(one), np.array(same))


def test_pair_en_frame_inverts_scene_caption():
    """``--pair_ref_frame en``: the sibling caption goes back under the EN
    frame the scene was rendered with; a frame that names no language only
    loses the ``japanese text`` tag."""
    from data.pair import en_frame
    from data.synth import scene_caption

    sc = {
        "head": ["safe", "1girl"],
        "generals": ["english text", "speech bubble", "smile"],
        "clause_tpl": 'English text reads as "{a}".',
    }
    cap = scene_caption(sc, "K")
    assert "japanese text" in cap and 'Japanese text reads as "K"' in cap
    en = en_frame(cap)
    assert "apanese" not in en
    assert ", english text," in en and en.endswith('English text reads as "K".')
    said = scene_caption({**sc, "clause_tpl": 'She is saying "{a}".'}, "K")
    assert en_frame(said) == said.replace("japanese text", "english text")
    # last tag before the clause
    assert en_frame('1girl, japanese text. She is saying "K".') == (
        '1girl, english text. She is saying "K".'
    )


def test_init_rows_converts_row_scale(tmp_path):
    """``--init_rows`` across inventories (S2a pre-condition, 2026-09-18):
    ``raw`` is in the source run's row-norm units, so a row must be rescaled
    by ``src_row_scale / this_row_scale`` — the delta applied stays
    bit-for-bit the source's. Rows the source lacks stay zero; a source with
    no ``row_scale`` is taken as-is."""
    import types

    import torch
    from train.trainables import Trainables

    src_rs, dst_rs = 232.9, 197.0
    src_ids = [100, 101, 103]
    src_raw = torch.randn(3, 8)
    torch.save(
        {
            "arm": "rows",
            "delta": {"ext_ids": src_ids, "raw": src_raw, "row_scale": src_rs},
        },
        tmp_path / "src.pt",
    )
    torch.save(
        {"arm": "rows", "delta": {"ext_ids": src_ids, "raw": src_raw}},
        tmp_path / "old.pt",
    )

    def fresh():
        tr = Trainables.__new__(Trainables)
        tr.a = types.SimpleNamespace(c_flat_cap=0.0, init_anchor=0.0)
        tr.device = "cpu"
        tr.row_scale = dst_rs
        tr.c_flat = None
        tr.delta = types.SimpleNamespace(
            ext_ids=[100, 101, 102, 103], raw=torch.nn.Parameter(torch.zeros(4, 8))
        )
        return tr

    tr = fresh()
    tr._init_rows_from(str(tmp_path / "src.pt"))
    got = tr.delta.raw.detach() * dst_rs  # the delta this run applies
    want = src_raw * src_rs  # the delta the source run applied
    assert torch.allclose(got[[0, 1, 3]], want, atol=1e-4)
    assert torch.equal(got[2], torch.zeros(8))
    assert tr.warm_mask.tolist() == [True, True, False, True]

    tr = fresh()
    tr._init_rows_from(str(tmp_path / "old.pt"))
    assert torch.allclose(tr.delta.raw.detach()[[0, 1, 3]], src_raw)


def test_box_share_loss_is_area_independent():
    """--box_share (plan_synth4 R4.5): the in-box share of the loss is ρ_g per
    glyph whatever the box area, capped; 0 falls back to --box_weight."""
    import torch
    from train.stage import BOX_SHARE_CAP, weighted_fm_loss

    def loss(box, text="が", share=0.25, e_in=1.0, e_out=0.0):
        target = torch.zeros(1, 4, 64, 64)
        pred = torch.full_like(target, e_out**0.5)
        x0, y0, x1, y1 = (v // 8 for v in box)
        pred[:, :, y0:y1, x0:x1] = e_in**0.5
        rec = [{"box": box, "text": text}]
        return float(weighted_fm_loss(pred, target, rec, 4.0, share))

    small, big = [64, 64, 96, 96], [64, 64, 192, 192]  # 16 vs 256 cells
    assert loss(small) == pytest.approx(0.25)
    assert loss(big) == pytest.approx(0.25)
    assert loss(small, e_in=0.0, e_out=1.0) == pytest.approx(0.75)
    assert loss(big, text="がぎ") == pytest.approx(0.5)  # per glyph
    assert loss(big, text="が ぎ ぐ げ ご") == pytest.approx(BOX_SHARE_CAP)
    # --box_share_cap: a lower ceiling bites the many-glyph item only
    rec5 = [{"box": big, "text": "がぎぐげご"}]
    pred = torch.zeros(1, 4, 64, 64)
    pred[:, :, 8:24, 8:24] = 1.0
    capped = weighted_fm_loss(pred, torch.zeros_like(pred), rec5, 4.0, 0.25, 0.4)
    assert float(capped) == pytest.approx(0.4)
    # share 0 = the old weight-sum form, which does follow the area
    assert loss(small, share=0.0) < loss(big, share=0.0)


def test_box_split_separates_in_and_out_of_box():
    """BoxSplit: per-item in-box / out-of-box means, the σ split, and the reset."""
    import torch
    from train.stage import BoxSplit

    target = torch.zeros(2, 4, 64, 64)
    pred = torch.full_like(target, 0.5)  # out-of-box se 0.25
    pred[0, :, 8:12, 8:12] = 1.0  # item 0: in-box se 1, σ 0.8
    pred[1, :, 8:24, 8:24] = 2.0  # item 1: in-box se 4, σ 0.6, 16× the area
    recs = [{"box": [64, 64, 96, 96]}, {"box": [64, 64, 192, 192]}]
    sp = BoxSplit()
    sp.add(pred, target, recs, torch.tensor([0.8, 0.6]))
    out = sp.pop()
    assert out["in_box"] == pytest.approx(2.5)  # per item, not per cell
    assert out["out_box"] == pytest.approx(0.25)
    assert out["in_box_hi"] == pytest.approx(1.0)
    assert out["in_box_lo"] == pytest.approx(4.0)
    assert sp.pop() == {}


def test_phrase_norm_and_balanced_draw(tmp_path):
    """--phrase_norm respells ellipses / bangs and merges the spellings;
    --text_draw balanced lands every fitting string the same count ± 1."""
    import random
    from data.inventory import norm_phrase, phrase_file_lines
    from data.synth import _LenPool

    assert norm_phrase("すまん・・・・寝てたか！！！") == "すまん・・・寝てたか！！"
    assert norm_phrase("え…") == norm_phrase("え･･･") == "え・・・"
    assert norm_phrase("あ・い") == "あ・い"  # a lone separator stays
    f = tmp_path / "p.tsv"
    f.write_text("え…\tb1\t2\nえ・・・・\tb2\t2\nはい\tb1\t2\n", encoding="utf-8")
    assert phrase_file_lines(f, 2, 10) == [("え…", "b1", 2), ("え・・・・", "b2", 2), ("はい", "b1", 2)]
    assert phrase_file_lines(f, 2, 10, norm=True) == [("え・・・", "b1", None), ("はい", "b1", 2)]

    texts = ["ab", "cd", "ef", "ghijklmnopqrs"]  # one length-13 string
    rng = random.Random(0)
    by_len = _LenPool(texts)
    n = sum(by_len.draw(rng, 18) == texts[3] for _ in range(400))
    assert n > 150  # the by-length draw: half the items on one string
    bal = _LenPool(texts, balanced=True)
    draws = [bal.draw(rng, 18) for _ in range(402)]
    counts = sorted(draws.count(t) for t in texts)
    assert counts[-1] - counts[0] <= 1
    assert all(len(bal.draw(rng, 2)) == 2 for _ in range(9))  # the cap still binds



def test_grid_deck_deals_rows_evenly():
    """`--grid`: units are dealt, so draw counts differ by at most one deck
    pass and an item never repeats a unit."""
    import random
    from collections import Counter

    from data.grid import GRIDS, _Deck, parse_grid

    assert parse_grid("2x2,3x3:2") == [("2x2", 1.0), ("3x3", 2.0)]
    assert {c * r for c, r, _ in GRIDS.values()} == {4, 6, 9}
    assert all(W % 16 == 0 and H % 16 == 0 for _, _, (W, H) in GRIDS.values())
    deck = _Deck([str(i) for i in range(23)], random.Random(0))
    n = Counter()
    for k in [4, 9, 6, 9, 4, 6] * 20:
        got = deck.deal(k)
        assert len(set(got)) == k
        n.update(got)
    assert max(n.values()) - min(n.values()) <= 1
    # pool weights: a unit repeated w times is dealt w times per pass
    deck = _Deck(["a", "b", "c", "d"] * 6 + ["x", "y"], random.Random(0))
    n = Counter(u for _ in range(130) for u in deck.deal(4))
    assert 5 <= n["a"] / n["x"] <= 7


def test_grid_word_deck_by_row():
    """`--grid_words`: no 3x3, a bubble cell holds one glyph fewer, and the
    by-row draw keeps every word under the cap, spreads a row over its words
    and returns None (uses undone) once the supply is spent."""
    import random

    from data.grid import _WordDeck, max_glyphs

    assert max_glyphs("3x3", False, 56) == 0
    assert [max_glyphs(g, b, 56) for g in ("2x2", "2x3") for b in (False, True)] == [
        4, 3, 3, 2,
    ]  # fmt: skip
    words = {"あか": ["あ", "か"], "あす": ["あ", "す"], "かす": ["か", "す"],
             "すあか": ["す", "あ", "か"], "あかすか": ["あ", "か", "す", "か"]}  # fmt: skip
    deck = _WordDeck(words, 3, random.Random(0))
    assert deck.rows == ["あ", "か", "す"]
    got = deck.deal(4, 3)
    assert len(set(got)) == 4 and "あかすか" not in got
    while (got := deck.deal(2, 4)) is not None:
        assert len(set(got)) == 2
    assert max(deck.used.values()) <= 3
    assert sum(deck.used.values()) >= 3 * len(words) - 1  # one word cannot pair up
    before = dict(deck.used)
    assert deck.deal(4, 4) is None and dict(deck.used) == before


def test_grid_caption_headers():
    from common.prompts import grid_caption, grid_cell_header

    assert [grid_cell_header(2, 3, i) for i in range(6)] == [
        "On the top left",
        "On the top right",
        "On the middle left",
        "On the middle right",
        "On the bottom left",
        "On the bottom right",
    ]
    assert grid_cell_header(3, 3, 4) == "In the center"
    assert grid_cell_header(3, 2, 1) == "On the top middle"
    cap = grid_caption("flat", 2, 2, ["あ", "か", "す", "日"])
    assert cap.endswith('On the bottom right, Japanese text reads as "日".')
    assert cap.count("Japanese text reads as") == 4
