"""_make_dynamic_seq_forward — which seq-axis mark each attention backend gets.

GH #107: under SDPA (attn_mode="torch", forced on ROCm) inductor guards the
backward logsumexp stride on ``seq % 8``, so a strict ``mark_dynamic`` range
raises ConstraintViolationError at the first compiled step. SDPA must get soft
``maybe_mark_dynamic`` marks; flash keeps the strict per-band range. The compile
itself needs a GPU; the dispatch is pure and is pinned here with recorders.
"""

import pytest
import torch

from library.anima.models import _make_dynamic_seq_forward
from networks.attention_dispatch import AttentionParams

BANDS = [(992, 3100)]


@pytest.fixture
def marks(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch._dynamo,
        "mark_dynamic",
        lambda t, dim, **kw: calls.append(("strict", dim, kw)),
    )
    monkeypatch.setattr(
        torch._dynamo,
        "maybe_mark_dynamic",
        lambda t, dim: calls.append(("maybe", dim, {})),
    )
    return calls


def _call(attn_params, seq=1023):
    fwd = _make_dynamic_seq_forward(lambda *a: a[0], BANDS)
    x = torch.zeros(1, 1, seq, 1, 4)
    rope = (torch.zeros(seq, 1, 1, 4), torch.zeros(seq, 1, 1, 4))
    out = fwd(x, None, None, attn_params, rope, None)
    assert out is x


@pytest.mark.parametrize("params", [AttentionParams("torch"), None])
def test_sdpa_gets_soft_marks(marks, params):
    _call(params)
    assert [(kind, dim) for kind, dim, _ in marks] == [
        ("maybe", 2),
        ("maybe", 0),
        ("maybe", 0),
    ]


def test_flash_keeps_strict_band(marks):
    _call(AttentionParams("flash"))
    assert [(kind, dim) for kind, dim, _ in marks] == [
        ("strict", 2),
        ("strict", 0),
        ("strict", 0),
    ]
    assert all(kw == {"min": 992, "max": 3100} for _, _, kw in marks)
