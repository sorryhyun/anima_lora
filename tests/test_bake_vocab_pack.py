"""``scripts/toolkits/bake_vocab_pack.py`` — folding a wake-line ext-row delta
into a vocab pack.

Invariants:

* **Bake equals hook** (the baked pack serves what the base pack + the
  ``ExtDelta`` hook serve) is checked beside that hook, in
  ``project/cjk_anima_reseed/tests/test_bake_equals_hook.py``.
* **Untouched rows are byte-identical**, the routing maps are unchanged, and
  only the summed rows are re-tiered in ``provenance``.
"""

from __future__ import annotations

import torch

from library.anima.ext_vocab import pack_digest


from scripts.toolkits import bake_vocab_pack as bvp

DIM = 16
ROWS = 8


def _base():
    g = torch.Generator().manual_seed(0)
    table = torch.randn(ROWS, DIM, generator=g) * 200.0
    mapping = {
        "rows": ROWS,
        "qwen": {str(100 + i): i for i in range(ROWS)},
        "route": {"ranges": [[0x3040, 0x30FF]]},
        "provenance": ["mapped"] * ROWS,
    }
    return table, mapping


def _delta(ids, row_scale=197.0):
    g = torch.Generator().manual_seed(1)
    return {
        "path": "synthetic",
        "arm": "rows",
        "ext_ids": list(ids),
        "raw": torch.randn(len(ids), DIM, generator=g),
        "row_scale": row_scale,
        "arm_tag": "t",
        "data_tag": "d",
        "train_steps": 1,
        "units": ["kana"],
        "init_rows": None,
    }


def test_bake_formula_and_provenance():
    table, mapping = _base()
    delta = _delta([1, 5])
    baked, m, summary = bvp.bake(table, mapping, delta, scale=0.5)
    idx = torch.tensor([1, 5])
    expect = table[idx] + delta["raw"] * (197.0 * 0.5)
    assert torch.equal(baked[idx], expect)
    others = torch.tensor([0, 2, 3, 4, 6, 7])
    assert torch.equal(baked[others], table[others])
    assert m["provenance"] == [
        "mapped",
        "render",
        "mapped",
        "mapped",
        "mapped",
        "render",
        "mapped",
        "mapped",
    ]
    assert m["qwen"] == mapping["qwen"] and m["route"] == mapping["route"]
    assert mapping["provenance"] == ["mapped"] * ROWS, "base mapping mutated"
    assert summary["rows"] == 2 and summary["scale"] == 0.5
    assert pack_digest(baked, m) != pack_digest(table, mapping)


def test_glyph_stage_save_is_baked_at_scale_one(tmp_path):
    """A ``project/cjk_rl/src/g1.py`` save holds the run's whole table in
    effective units: its rows are added as they are, and a row at zero (the
    pack's own row in that run) is not touched or re-tiered."""
    table, mapping = _base()
    g = torch.Generator().manual_seed(2)
    raw = torch.zeros(4, DIM)
    raw[[0, 2]] = torch.randn(2, DIM, generator=g) * 40.0
    path = tmp_path / "g1run" / "rows.pt"
    path.parent.mkdir()
    torch.save(
        {
            "ext_ids": [1, 3, 5, 6],
            "rows": {"あ": 0},
            "raw": raw,
            "init": torch.zeros(4, DIM),
            "row_scale": 206.0,
            "steps": {"あ": 6},
            "cfg": {"rows": "seed"},
        },
        path,
    )
    delta = bvp.load_delta(path)
    assert delta["ext_ids"] == [1, 5] and delta["row_scale"] == 1.0
    assert delta["arm_tag"] == "g1run" and delta["train_steps"] == 6
    baked, m, _ = bvp.bake(table, mapping, delta)
    assert torch.equal(baked[[1, 5]], table[[1, 5]] + raw[[0, 2]])
    others = torch.tensor([0, 2, 3, 4, 6, 7])
    assert torch.equal(baked[others], table[others])
    assert [i for i, p in enumerate(m["provenance"]) if p == "render"] == [1, 5]
