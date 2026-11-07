"""``model._attention``'s ``ref_gate`` against a dense masked SDPA, and the hole sampler."""

import torch
import torch.nn.functional as F

from library.moss import model as mm
from library.moss.training.holes import HoleSampler, cells

B, HEADS, D = 1, 2, 8
N_IMG, N_TXT, N_REF = 12, 3, 12


def _rope(n):
    return torch.cat((torch.zeros(B, n, D // 2), torch.ones(B, n, D // 2)), -1)


def _dense(q, k, v, allow):
    """``(B, L, H·D)`` with ``allow`` ``(Lq, Lk)`` bool."""
    shape = lambda t: t.reshape(B, t.shape[1], HEADS, D).transpose(1, 2)  # noqa: E731
    out = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), attn_mask=allow)
    return out.transpose(1, 2).flatten(2)


def _gate():
    rows = torch.tensor([1, 2, 5, 6])
    keep = torch.tensor([i for i in range(N_REF) if i not in (2, 5, 6, 7)])
    return rows, keep


def test_isolated_path_matches_dense_mask():
    torch.manual_seed(0)
    L = N_IMG + N_TXT + N_REF
    q, k, v = (torch.randn(B, L, HEADS * D) for _ in range(3))
    s = N_IMG + N_TXT
    out = mm._attention(q, k, v, _rope(L), HEADS, ref_start=s, ref_isolate=True, ref_gate=_gate())  # fmt: skip
    allow = torch.ones(L, L, dtype=torch.bool)
    allow[s:, :s] = False
    rows, keep = _gate()
    cut = torch.ones(N_REF, dtype=torch.bool)
    cut[keep] = False
    allow[rows[:, None], s + cut.nonzero().flatten()[None]] = False
    torch.testing.assert_close(out, _dense(q, k, v, allow))


def test_cached_reference_path_matches_dense_mask():
    torch.manual_seed(1)
    L = N_IMG + N_TXT
    q, k, v = (torch.randn(B, L, HEADS * D) for _ in range(3))
    rk, rv = torch.randn(B, N_REF, HEADS * D), torch.randn(B, N_REF, HEADS * D)
    out = mm._attention(q, k, v, _rope(L), HEADS, ref_kv=(rk, rv, _rope(N_REF)), ref_gate=_gate())  # fmt: skip
    allow = torch.ones(L, L + N_REF, dtype=torch.bool)
    rows, keep = _gate()
    cut = torch.ones(N_REF, dtype=torch.bool)
    cut[keep] = False
    allow[rows[:, None], L + cut.nonzero().flatten()[None]] = False
    ref = _dense(q, torch.cat((k, rk), 1), torch.cat((v, rv), 1), allow)
    torch.testing.assert_close(out, ref)


def test_no_gate_is_the_stock_path():
    torch.manual_seed(2)
    L = N_IMG + N_TXT + N_REF
    q, k, v = (torch.randn(B, L, HEADS * D) for _ in range(3))
    a = mm._attention(q, k, v, _rope(L), HEADS, ref_start=N_IMG + N_TXT, ref_isolate=True)  # fmt: skip
    b = mm._attention(q, k, v, _rope(L), HEADS, ref_start=N_IMG + N_TXT, ref_isolate=True, ref_gate=None)  # fmt: skip
    assert torch.equal(a, b)


def test_cells_and_sampler():
    c = cells([[0.25, 0.0, 0.5, 0.5]], 4, 4)
    assert c.tolist() == [
        [False, True, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    hs = HoleSampler("", seed=0)
    for _ in range(50):
        g = hs.gate((20, 30), (20, 30), "cpu")
        assert g is not None
        rows, keep = g
        assert 2 <= rows.numel() <= 300 and keep.numel() >= 2
        # cut: a hole query's own reference cell is never kept
        assert not set(rows.tolist()) & set(keep.tolist())
