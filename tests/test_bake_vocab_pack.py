"""``scripts/toolkits/bake_vocab_pack.py`` — folding a wake-line ext-row delta
into a vocab pack.

Invariants:

* **Bake equals hook.** A pack with the delta summed in serves, through the
  plain ``attach_vocab_pack`` hooks, exactly what the base pack + the render
  line's ``ExtDelta`` hook serve (deploy_plan G0, at the table level).
* **Untouched rows are byte-identical**, the routing maps are unchanged, and
  only the summed rows are re-tiered in ``provenance``.
* An id outside the base table refuses instead of growing the table.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.anima import vocab_pack as vp
from library.anima.ext_vocab import T5_TABLE_SIZE, pack_digest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "toolkits"))
sys.path.insert(0, str(REPO / "project" / "cjk_renderable_anima" / "src"))

import bake_vocab_pack as bvp  # noqa: E402

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


def _adapter():
    torch.manual_seed(2)
    return SimpleNamespace(embed=torch.nn.Embedding(T5_TABLE_SIZE, DIM))


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


def test_bake_refuses_ids_outside_table():
    table, mapping = _base()
    with pytest.raises(ValueError, match="outside"):
        bvp.bake(table, mapping, _delta([2, ROWS]))


def test_bake_equals_hook():
    from common.hooks import ExtDelta

    table, mapping = _base()
    delta = _delta([0, 3, 7])
    ids = torch.tensor(
        [[5, T5_TABLE_SIZE + 3, 9, T5_TABLE_SIZE + 0, T5_TABLE_SIZE + 4]]
    )

    # base pack + ExtDelta hook (what eval / native render)
    ad = _adapter()
    base = vp.VocabPack(prefix=Path("base"), table=table, mapping=mapping, digest="b")
    vp.attach_vocab_pack(ad, base)
    hook = ExtDelta(
        SimpleNamespace(llm_adapter=ad),
        delta["ext_ids"],
        DIM,
        "cpu",
        delta["row_scale"],
    )
    hook.raw.data.copy_(delta["raw"])
    with torch.no_grad():
        want = ad.embed(ids).clone()
    for h in hook.handles:
        h.remove()
    vp.detach_vocab_pack(ad)

    # baked pack alone
    baked, m, _ = bvp.bake(table, mapping, delta)
    ad2 = _adapter()
    vp.attach_vocab_pack(
        ad2, vp.VocabPack(prefix=Path("baked"), table=baked, mapping=m, digest="k")
    )
    with torch.no_grad():
        got = ad2.embed(ids)
    assert torch.allclose(got, want, atol=1e-4, rtol=0), (got - want).abs().max()
    # stock ids untouched, an ext id with no delta serves the base row
    assert torch.equal(got[0, 0], ad2.embed.weight[5])
    assert torch.equal(got[0, 4], table[4])
