"""The band law as code: the rows reproduce the reads, the gate reads them."""

from __future__ import annotations

import pytest

from cjk_scale.windows import (
    KINDS,
    ROWS,
    SIGMA_MAX,
    covers,
    kind_of,
    overlap,
    vocab_kind,
    window,
)

# a stand-in tokenizer: one token per vocab except the small-kana digraphs
_TOKENS = {"あっ": 2, "きゃ": 2, "ってる": 1, "って": 1, "先生": 1}


def n_tokens(u: str) -> int:
    return _TOKENS.get(u, 1 if len(u) <= 2 else len(u))


@pytest.mark.parametrize(
    "kind,px,layout,band",
    [
        ("single", 50, "scene", (0.7, 0.9)),  # B.1: bubble-fit composites (median 51)
        ("single", 42, "scene", (0.7, 0.9)),  # …down to the draw's p10 (40 px)
        ("single", 130, "grid", (0.7, 0.9)),  # step1_0921 grid cells
        ("single", 200, "flat", (0.7, 0.9)),  # the flat half
        ("single", 28, "scene", (0.5, 0.7)),  # design stage0507
        ("single", 39, "flat", (0.5, 0.7)),
        ("piece", 35, "scene", (0.5, 0.7)),  # micro_cf_0922
        ("piece", 48, "scene", (0.5, 0.7)),
        ("multi", 32, "scene", (0.5, 0.7)),  # short lines, design stage0507
        ("multi", 16, "scene", (0.3, 0.5)),  # A.2
        ("multi", 14, "grid", (0.3, 0.5)),
        ("piece", 16, "grid", (0.3, 0.5)),  # by design, unread
    ],
)
def test_rows_reproduce_the_reads(kind, px, layout, band):
    w = window(kind, px, layout)
    assert w is not None and w.band == band
    assert w.source


def test_unread_cells_have_no_window():
    assert (
        window("single", 16, "scene") is None
    )  # a glyph never renders that small in a bubble
    assert window("multi", 80, "grid") is None  # strings above 64 px are not a cell
    assert window("piece", 8, "flat") is None


def test_nothing_above_sigma_max():
    assert all(r.hi <= SIGMA_MAX for r in ROWS)
    assert all(set(r.kinds) <= set(KINDS) for r in ROWS)


def test_kind_is_tokens_then_glyphs():
    assert vocab_kind("あ", 1) == "single"
    assert vocab_kind("って", 1) == "piece"  # one token, two glyphs
    assert vocab_kind("あっ", 2) == "multi"  # host row + small row
    assert kind_of(["あ", "日", "！"], n_tokens) == "single"  # a grid of singles
    assert kind_of(["って", "先生"], n_tokens) == "piece"  # a grid of pieces
    assert kind_of(["あっ"], n_tokens) == "multi"
    assert kind_of(["ありがとうございます"], n_tokens) == "multi"  # a line
    assert kind_of(["って", "あっ"], n_tokens) == "multi"  # mixed grid: the heavier
    assert kind_of(["あ", "って"], n_tokens) == "piece"


def test_gate_contain_and_overlap():
    w = window("single", 50, "scene")  # 0.7–0.9
    assert covers((0.7, 0.9), w)
    assert not covers((0.5, 0.7), w)
    assert overlap((0.6, 0.9), w) == pytest.approx(2 / 3)
    assert not covers((0.6, 0.9), w, 0.8)
    assert covers((0.6, 0.9), w, 0.6)
    assert not covers((0.3, 0.9), w)  # the consolidation band fits no row: gate = none
    assert not covers((0.7, 0.9), None)
