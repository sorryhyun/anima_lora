"""Pack digest stability, and the quote-aware caption grammar (anime_tools)."""

from __future__ import annotations

import pytest
import torch

from library.anima import ext_vocab as ev


@pytest.mark.core
def test_pack_digest_is_stable_and_regeneration_invariant():
    spec = ev.IsoSpec(seed=0, n_rows=4, dim=8, norm=1.0, start=4)
    trained = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    mapping = {
        "qwen": {"5": 0},
        "char": {"猫": 1},
        "rows": 8,
        "iso": spec.to_json(),
        "route": ev.Route(quotes=ev.DEFAULT_QUOTES).to_json(),
        "training": {"note": "provenance never changes the digest"},
    }
    seed_only = ev.pack_digest(trained, mapping)
    shipped = ev.pack_digest(ev.materialize_iso(trained, mapping), mapping)
    assert seed_only == shipped and len(seed_only) == 64
    assert ev.pack_digest(trained, {**mapping, "training": {}}) == seed_only
    # Rows, ids or the quote rule change → a different pack.
    assert ev.pack_digest(trained + 1, mapping) != seed_only
    assert ev.pack_digest(trained, {**mapping, "char": {"犬": 1}}) != seed_only
    no_quotes = {**mapping, "route": ev.Route().to_json()}
    assert ev.pack_digest(trained, no_quotes) != seed_only


# ---------------------------------------------------------------------------
# Caption grammar (anime_tools) — a quoted line is one tag
# ---------------------------------------------------------------------------


@pytest.mark.core
def test_caption_grammar_keeps_a_quoted_line_whole():
    from anime_tools.captions import position_clauses as pc

    if not hasattr(pc, "quoted_spans"):
        pytest.skip("installed anime_tools predates the quote-aware grammar")
    cap = 'safe, 1girl, english text, "Are you okay? I\'m fine, really.", speech bubble'
    parsed = pc.parse_caption(cap)
    assert parsed.flat_tags[3] == '"Are you okay? I\'m fine, really."'
    assert pc.compose_caption(parsed.flat_tags, parsed.clauses) == cap
    assert pc.parse_caption("a, 「大丈夫、本当に」, b").flat_tags == (
        "a",
        "「大丈夫、本当に」",
        "b",
    )
