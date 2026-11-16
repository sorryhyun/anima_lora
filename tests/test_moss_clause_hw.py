"""``clause_hw``: position-clause tokens on RoPE's h / w axes (training only)."""

import torch

from library.moss.model import _RoPE
from library.moss.training.grounding import clause_cells, clause_spans


def test_rope_text_hw_moves_only_text_hw_bands():
    rope = _RoPE(128, 10000.0)
    plain = rope(4, 6, 5, "cpu")
    assert torch.equal(plain, rope(4, 6, 5, "cpu", text_hw=None))
    hw = torch.zeros(5, 2)
    hw[2] = torch.tensor([1.5, -2.0])
    moved = rope(4, 6, 5, "cpu", text_hw=hw)
    assert torch.equal(plain, rope(4, 6, 5, "cpu", text_hw=torch.zeros(5, 2)))
    diff = (moved != plain)[0]
    assert diff.any(1).nonzero().flatten().tolist() == [4 * 6 + 2]
    # The sin half, then the cos half, each (r, f, h, w) × bands: r and f hold.
    for half in diff[4 * 6 + 2].chunk(2):
        r, f, h, w = half.view(4, -1)
        assert not r.any() and not f.any() and h.any() and w.any()


def test_clause_cells():
    assert clause_cells(["left", "right"]) == [(0.0, -1.0), (0.0, 1.0)]
    assert clause_cells(["top right", "bottom", "middle"]) == [
        (-1.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ]
    cols = clause_cells(["leftmost", "second from left", "rightmost"])
    assert [w for _, w in cols] == [-1.0, 0.0, 1.0]
    assert clause_cells(["top rightmost"]) == [(-1.0, 1.0)]


def test_clause_spans_skip_the_bag_and_text_clauses():
    caption = (
        "safe, 2girls. On the left, red hair, smile. On the top right, hat. "
        'Japanese text reads as "あ" [0,0,1,1].'
    )
    spans = clause_spans(caption)
    assert [(caption[a:b], oh, ow) for a, b, oh, ow in spans] == [
        ("On the left, red hair, smile", 0.0, -1.0),
        ("On the top right, hat", -1.0, 1.0),
    ]
