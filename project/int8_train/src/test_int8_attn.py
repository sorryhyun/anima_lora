"""Phase 5 tests for ``int8_attn`` on random shapes (``.venv/bin/python -m pytest project/int8_train/src/test_int8_attn.py``)."""

import math

import pytest
import torch
import torch.nn.functional as F

from int8_attn import LOG2E, PV_MODES, attn_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _qkv(lq, lk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (
        torch.randn(1, n, 16, 128, generator=g, device="cuda").bfloat16()
        for n in (lq, lk, lk)
    )
    return q, k, v


def _ref(q, k, v):
    qt, kt, vt = (x.float().transpose(1, 2) for x in (q, k, v))
    o = F.scaled_dot_product_attention(qt, kt, vt).transpose(1, 2)
    s = qt @ kt.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return o, torch.logsumexp(s, -1) * LOG2E


@pytest.mark.parametrize("lq,lk", [(1000, 1000), (1000, 512)])
@pytest.mark.parametrize("qk", ["bf16", "int8"])
@pytest.mark.parametrize("pv", list(PV_MODES))
@pytest.mark.parametrize(
    "gran,rot", [("block", False), ("token", False), ("token", True)]
)
def test_fwd_matches_fp32(lq, lk, qk, pv, gran, rot):
    if qk == "bf16" and (gran, rot) != ("block", False):
        pytest.skip("gran and rot apply to int8 QKᵀ only")
    q, k, v = _qkv(lq, lk)
    # sharpen the logits so the softmax is not near-uniform
    q = (q * 4).bfloat16()
    o, lse = attn_fwd(q, k, v, qk, pv, gran=gran, rot=rot)
    o_ref, _ = _ref(q, k, v)
    # int8 QKᵀ runs on K minus its token mean: same softmax, logsumexp shifted per row
    k_lse = (k.float() - k.float().mean(1, keepdim=True)) if qk == "int8" else k
    _, lse_ref = _ref(q, k_lse, v)
    err = float((o.float() - o_ref).norm() / o_ref.norm())
    assert err < (0.01 if qk == pv == "bf16" else 0.06), err
    d = lse - lse_ref
    if qk == "bf16":
        assert float(d.abs().max()) < 0.02
    else:  # int8 logit rounding: unbiased, a few hundredths per row
        assert abs(float(d.mean())) < 0.02 and float(d.std()) < 0.1


def test_k_mean_shift_invariant():
    """Removing K's token mean must not change the int8 result beyond rounding."""
    q, k, v = _qkv(512, 512, seed=1)
    o_ref, _ = _ref(q, k + 3.0, v)
    o, _ = attn_fwd(q, (k + 3.0).bfloat16(), v, "int8", "bf16")
    assert float((o.float() - o_ref).norm() / o_ref.norm()) < 0.06
