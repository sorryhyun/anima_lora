"""``clause_hw`` / ``text_box_hw``: prompt tokens on RoPE's h / w axes (training only)."""

import torch

from library.moss.model import _RoPE
from library.moss.training.grounding import box_spans, clause_cells, clause_spans


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
    from library.moss.training.grounding import clause_ranges

    assert [(caption[a:b], p) for a, b, p in clause_ranges(caption, header=False)] == [
        ("red hair, smile", "left"),
        ("hat", "top right"),
    ]


def test_box_spans_cover_the_quoted_string_at_the_box_centre():
    caption = (
        "2girls. On the left, red hair. Japanese text reads as "
        '"後藤\\n156cm" [-490,-420,-410,-120], "" [0,0,16,16], "あ" [0,0,32,64].'
    )
    spans = box_spans(caption)
    assert [(caption[a:b], h, w) for a, b, h, w in spans] == [
        ("後藤\\n156cm", -28.125, -16.875),
        ("あ", 1.0, 2.0),
    ]


def test_subject_centres_and_text_hw_follow_the_mask(tmp_path):
    import json

    import numpy as np
    from PIL import Image

    from library.moss.training.grounding import subject_centres, text_hw

    analysis = tmp_path / "analysis" / "artist"
    analysis.mkdir(parents=True)
    labels = np.zeros((40, 80), dtype=np.uint8)
    labels[10:30, 0:20] = 1  # centroid (20, 10) of 40 × 80
    Image.fromarray(labels).save(analysis / "a.png")
    record = {
        "labels": "instances",
        "instances": [
            {"position": "left", "box": [0, 10, 20, 30]},
            {"position": "right", "box": [40, 0, 80, 20]},  # no mask: box centre
        ],
    }
    (analysis / "a.json").write_text(json.dumps(record))
    centres = subject_centres(tmp_path)
    assert centres == {"a": {"left": (0.0, -0.375), "right": (-0.25, 0.25)}}

    class Tok:
        def apply_chat_template(self, messages, **_):
            return "<|u|>" + messages[-1]["content"] + "<|a|>"

        def __call__(self, text, return_offsets_mapping=False):
            class Out:
                offset_mapping = [(i, i + 1) for i in range(len(text))]

            return Out()

    caption = "2girls. On the left, red hair. On the top right, hat."
    cells = text_hw(Tok(), caption, (30, 60), clause=1.0)
    moved = text_hw(Tok(), caption, (30, 60), clause=1.0, centres=centres["a"])

    def at(hw, word):
        return hw[len("<|u|>") + caption.index(word)].tolist()

    assert at(cells, "red") == [0.0, -20.0]
    assert at(moved, "red") == [0.0, -22.5]
    # ``top right`` has no record: it keeps its cell.
    assert at(moved, "hat") == at(cells, "hat") == [-10.0, 20.0]
    assert at(moved, "2girls") == [0.0, 0.0]
    # The header stays at (0, 0) unless asked to move with its tags.
    assert at(cells, "On the left") == [0.0, 0.0]
    with_header = text_hw(Tok(), caption, (30, 60), clause=1.0, header=True)
    assert at(with_header, "On the left") == [0.0, -20.0]
