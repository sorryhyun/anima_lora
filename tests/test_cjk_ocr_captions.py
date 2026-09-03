"""OCR lines -> arm-C caption (project/cjk_aware_anima/datasets/cache_te_ext.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    path = ROOT / "project" / "cjk_aware_anima" / "datasets" / "cache_te_ext.py"
    spec = importlib.util.spec_from_file_location("cache_te_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_order_format_is_one_phrase_in_record_order_and_round_trips():
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    m = _mod()
    cap = "1girl, smile. On the left, akita neru, yellow eyes."
    out = m.append_tags(cap, ["温水くん、私と", "あっ…うっ…"], "order")
    assert out == (
        '1girl, smile, Japanese text in following order: "温水くん、私と", '
        '"あっ…うっ…". On the left, akita neru, yellow eyes.'
    )
    # Grammar-safe: parse/compose leaves it byte-identical, the clause intact.
    p = parse_caption(out)
    assert compose_caption(p.flat_tags, p.clauses) == out
    assert p.clauses[0].tags == ("akita neru", "yellow eyes")
    # No separate presence tag: the phrase opens with it.
    assert "japanese text," not in out


def test_order_format_escapes_the_grammar_delimiters():
    m = _mod()
    out = m.append_tags("1girl", ['say "hi", now'], "order")
    assert out == '1girl, Japanese text in following order: "say ”hi”、 now"'


def test_tags_format_is_the_c2_shape():
    m = _mod()
    assert (
        m.append_tags("1girl, smile", ["あっ"], "tags")
        == "1girl, smile, japanese text, 「あっ」"
    )
    # An existing text tag suppresses the presence tag.
    assert (
        m.append_tags("1girl, speech bubble", ["あっ"], "tags")
        == "1girl, speech bubble, 「あっ」"
    )
