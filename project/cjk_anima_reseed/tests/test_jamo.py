"""The jamo factor rows (``reseed/jamo.py``): the syllable code, KS X 1001,
and the composition the trainer writes into ``Rows.delta.raw``."""

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

from reseed import jamo as J  # noqa: E402


def test_code_and_sets():
    assert J.decompose("가") == (0, 0, 0)
    assert J.decompose("없") == (11, 4, 18)  # ㅇ ㅓ ㅄ
    assert all(J.compose(*J.decompose(s)) == s for s in J.ALL)
    assert len(J.ALL) == 11172 and len(J.KSX1001) == 2350
    assert "쀼" in J.KSX1001 and "뷁" not in J.KSX1001
    assert Counter(map(J.layout, J.KSX1001)) == {
        "VF": 1069,
        "HF": 585,
        "CF": 347,
        "V": 149,
        "C": 109,
        "H": 91,
    }
    assert len(set().union(*map(J.cells, J.KSX1001))) == 112
    assert [J.layout(s) for s in "가고과각곡곽"] == ["V", "H", "C", "VF", "HF", "CF"]


def _rows(ids, frozen, d=8):
    delta = SimpleNamespace(
        ext_ids=ids, raw=torch.nn.Parameter(torch.randn(len(ids), d))
    )
    return SimpleNamespace(
        delta=delta, frozen_mask=torch.tensor([i in frozen for i in ids]), params=None
    )


def test_compose_and_grad():
    syl = {10: "가", 11: "각", 12: "고"}
    rows = _rows([5, 10, 11, 12], frozen={5})
    held = rows.delta.raw.detach()[0].clone()
    jm = J.Jamo(rows, syl, lr=1e-3)
    assert rows.params[0]["params"] == [jm.b, jm.C, jm.V, jm.F]
    assert J.Jamo.N_VECTORS == 106
    with torch.no_grad():
        for t in (jm.b, jm.C, jm.V, jm.F):
            t.normal_()
    jm.apply()
    raw = rows.delta.raw
    assert torch.equal(raw[0], held)  # the frozen context row kept
    F = torch.cat([torch.zeros_like(jm.F[:1]), jm.F])
    # 가 = ㄱ (vertical) + ㅏ + no final; 각 adds the final ㄱ; 고 is horizontal
    want = jm.b[0] + jm.C[0, 0] + jm.V[0]
    assert torch.allclose(raw[1], want)
    assert torch.allclose(raw[2], want + F[1])
    assert torch.allclose(raw[3], jm.b[0] + jm.C[0, 1] + jm.V[8])
    raw[1:].sum().backward()
    assert jm.b.grad is not None and float(jm.b.grad.abs().sum()) > 0
    assert float(jm.C.grad[0, 2].abs().sum()) == 0  # no compound-vowel syllable
    assert rows.delta.raw is not jm.free and not jm.free.requires_grad


def test_merge_all():
    syl = {10: "가", 11: "각"}
    rows = _rows([5, 10, 11], frozen={5})
    jm = J.Jamo(rows, syl, lr=1e-3)
    with torch.no_grad():
        jm.V.normal_()
        jm.apply()
    sd = {
        "delta": {
            "ext_ids": [5, 10, 11],
            "raw": rows.delta.raw.detach().clone(),
            "row_scale": 1.0,
        }
    }
    n = jm.merge_all(sd, {"가": 10, "각": 11, "나": 20, "갸": 3})
    assert n == 2 and sd["delta"]["ext_ids"] == [3, 5, 10, 11, 20]
    raw = sd["delta"]["raw"]
    assert torch.allclose(raw[4], raw[2])  # 나 = 가 (C[ㄴ] still 0)
    assert torch.allclose(raw[0], jm.V[2].detach())  # 갸: ㅑ
