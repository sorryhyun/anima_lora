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
        {"arm": "rows", "delta": {"ext_ids": src_ids, "raw": src_raw, "row_scale": src_rs}},
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
