"""``captions = "control"`` draws of the MOSS colorize cache (library/moss/colorize.py)."""

from __future__ import annotations

import json
import random

from library.moss import colorize as C


def _is_color():
    return C._color_caption().is_color_tag


def test_character_colour_keeps_hair_eyes_and_clothes():
    is_color = _is_color()
    keep = ["pink hair", "blue eyes", "red pupils", "blue skirt", "light blue dress"]
    drop = ["white background", "dark skin", "dark-skinned female", "blue sky", "blush"]
    assert all(C.is_character_colour(t, is_color) for t in keep)
    assert not any(C.is_character_colour(t, is_color) for t in drop)


def test_keep_draw_never_empty_on_a_pool():
    rng = random.Random(0)
    atoms = list(range(3))
    for _ in range(200):
        assert C.keep_draw(atoms, 0.99, rng)
    assert C.keep_draw([], 0.8, rng) == []


def test_keep_draw_rate():
    rng = random.Random(1)
    atoms = list(range(1000))
    kept = C.keep_draw(atoms, 0.8, rng)
    assert 150 < len(kept) < 250


def test_manga_draws_keep_every_text_box_and_drop_colours():
    chars = [
        ([0, 0, 100, 200], ["pink hair", "white shirt"]),
        ([300, 0, 400, 200], ["black hair"]),
    ]
    texts = [[100, 0, 200, 100], [200, 100, 300, 200]]
    draws = C.manga_draws(chars, texts, (400, 200), 80, 0.8, random.Random(0))
    seen = set()
    for d in draws:
        lay = json.loads(d)
        assert lay["summary"] == {"caption": "comic"}
        assert lay["bbox_format"] == "yxyx"
        chars_in = [it for it in lay["items"] if it["label"] == "character"]
        texts_in = [it for it in lay["items"] if it["label"] == "text"]
        assert chars_in, "one colour at least"
        assert [it["bbox"] for it in texts_in] == [[-100, -100, 0, 0], [0, 0, 100, 100]]
        assert all(set(it) == {"label", "bbox"} for it in texts_in)
        for it in chars_in:
            assert it["bbox"] in ([-100, -200, 100, -100], [-100, 100, 100, 200])
            seen.update(t.strip() for t in it["caption"].split(","))
    assert seen == {"pink hair", "white shirt", "black hair"}


def test_manga_bare_is_comic_and_text_boxes():
    lay = json.loads(C.manga_layout([], [[100, 0, 200, 100]], (400, 200)))
    assert lay["summary"] == {"caption": "comic"}
    assert lay["items"] == [{"label": "text", "bbox": [-100, -100, 0, 0]}]
    assert C.manga_layout([], [], (64, 64)) == '{"summary":{"caption":"comic"}}'


def test_illustration_draws():
    from library.moss import cache as moss_cache

    kb = moss_cache.tag_knowledge_base()
    pool = ["genshin impact", "blonde hair", "blue skirt", "white shirt", "red ribbon"]
    draws = C.illustration_draws(pool, 20, 0.8, random.Random(0), kb)
    for d in draws:
        lay = json.loads(d)
        text = ", ".join([*lay["summary"].get("series", []), lay["summary"]["caption"]])
        assert any(t in text for t in pool)
    assert C.illustration_draws([], 3, 0.8, random.Random(0), kb) == ["", "", ""]
