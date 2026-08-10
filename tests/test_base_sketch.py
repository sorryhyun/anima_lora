"""E30.1 base-sketch instrument invariants (CPU, synthetic).

The in-run accuracy gate is e30's gate 2 (sketched-vs-exact on the param
family); these tests pin the properties the instrument assumes before any
GPU is spent: hash determinism, sketch linearity (complement = full −
adaln is exact only because of it), inner-product preservation at the
JL-expected scale, and the hook path (per-draw sketch-and-free summing to
the sketch of the bin-summed gradient, exact adaln slot matching the true
slice sum, grads freed).
"""

import numpy as np
import pytest
import torch

from project.sigma_lowres.bench.sigma_probe.base_sketch import (
    BaseSketchAccumulator,
    _hash_constants,
)


def _sketch(acc, flat, consts):
    buf = torch.zeros(acc.k, dtype=torch.float32)
    acc._sketch_add(buf, flat, consts)
    return buf


@pytest.fixture()
def tiny():
    torch.manual_seed(0)
    anima = torch.nn.Module()
    anima.adaln_up_self_attn = torch.nn.Linear(8, 24, bias=False)
    anima.proj = torch.nn.Linear(8, 8, bias=False)
    network = torch.nn.Module()  # no shared params
    acc = BaseSketchAccumulator(
        anima, network, torch.device("cpu"), k=1 << 12, seed=3021
    )
    return anima, acc


def test_hash_deterministic_and_distinct():
    assert _hash_constants(3021, "x") == _hash_constants(3021, "x")
    assert _hash_constants(3021, "x") != _hash_constants(3021, "y")
    assert _hash_constants(3021, "x") != _hash_constants(3022, "x")


def test_sketch_linearity(tiny):
    _, acc = tiny
    consts = _hash_constants(3021, "t")
    a, b = torch.randn(5000), torch.randn(5000)
    lhs = _sketch(acc, a, consts) + _sketch(acc, b, consts)
    rhs = _sketch(acc, a + b, consts)
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_inner_product_preserved():
    anima = torch.nn.Module()
    anima.w = torch.nn.Parameter(torch.zeros(1))
    acc = BaseSketchAccumulator(
        anima, torch.nn.Module(), torch.device("cpu"), k=1 << 14, seed=3021
    )
    consts = _hash_constants(3021, "t")
    g = torch.Generator().manual_seed(1)
    for _ in range(3):
        u, v = torch.randn(100_000, generator=g), torch.randn(100_000, generator=g)
        su, sv = _sketch(acc, u, consts), _sketch(acc, v, consts)
        cos = float(torch.dot(su, sv) / (su.norm() * sv.norm()))
        cos_true = float(torch.dot(u, v) / (u.norm() * v.norm()))
        assert abs(cos - cos_true) < 0.05


def test_hook_path_sums_and_frees(tiny):
    anima, acc = tiny
    assert list(acc.adaln_names) == ["adaln_up_self_attn.weight"]
    assert acc.adaln_numel == 24 * 8

    x = torch.randn(4, 8)
    acc.begin_arm("a")
    for _ in range(2):  # two draws, grads freed by the hook each time
        loss = (anima.adaln_up_self_attn(x).sum() + anima.proj(x).sum()) ** 2
        loss.backward()
        assert anima.adaln_up_self_attn.weight.grad is None
        assert anima.proj.weight.grad is None
    # recompute the two per-draw grads without hooks interfering: redo with
    # autograd.grad on fresh graph
    g_sum = {}
    for _ in range(2):
        loss = (anima.adaln_up_self_attn(x).sum() + anima.proj(x).sum()) ** 2
        gs = torch.autograd.grad(loss, [p for _, p in sorted(anima.named_parameters())])
        for (n, _), g in zip(sorted(anima.named_parameters()), gs):
            g_sum[n] = g_sum.get(n, 0) + g

    lora_vec = torch.randn(64)
    acc.end_bin(0, lora_vec)

    # exact adaln slot == true summed slice
    ex = torch.from_numpy(acc.exact[("a", 0)].astype(np.float32))
    truth = g_sum["adaln_up_self_attn.weight"].reshape(-1)
    assert torch.allclose(ex, truth, rtol=2e-3, atol=1e-3)  # fp16 slot rounding

    # full-family sketch == sketch of the true summed grads (linearity
    # through the per-draw hook path)
    expect = torch.zeros(acc.k)
    for name, _, _, consts in acc.entries:
        acc._sketch_add(expect, g_sum[name].reshape(-1), consts)
    got = torch.from_numpy(acc.sk[("full", "a", 0)])
    assert torch.allclose(got, expect, rtol=1e-4, atol=1e-4)

    # complement-by-linearity: full − adaln == sketch of non-adaln tensors
    comp_expect = torch.zeros(acc.k)
    for name, _, off, consts in acc.entries:
        if off < 0:
            acc._sketch_add(comp_expect, g_sum[name].reshape(-1), consts)
    comp = got - torch.from_numpy(acc.sk[("adaln", "a", 0)])
    assert torch.allclose(comp, comp_expect, rtol=1e-4, atol=1e-4)

    # param family == sketch of the handed-over LoRA vector
    pexpect = torch.zeros(acc.k)
    acc._sketch_add(pexpect, lora_vec, acc.param_consts)
    assert torch.allclose(
        torch.from_numpy(acc.sk[("param", "a", 0)]), pexpect, rtol=1e-4, atol=1e-4
    )


def test_finalize_writes_store(tiny, tmp_path):
    anima, acc = tiny
    x = torch.randn(2, 8)
    for key in ("a", "768"):
        acc.begin_arm(key)
        loss = (anima.adaln_up_self_attn(x).sum() + anima.proj(x).sum()) ** 2
        loss.backward()
        acc.end_bin(0, torch.randn(16))
    meta = acc.finalize(tmp_path)
    out = tmp_path / "base_sketch"
    for fam in ("full", "adaln", "complement", "param"):
        assert (out / f"sketch_{fam}.npz").exists()
    ex = np.load(out / "adaln_exact.npz")
    assert ex["gram"].shape == (2, 2)
    assert list(ex["conditions"]) == ["768__b0", "a__b0"]
    assert meta["adaln_resolved"] == {"adaln_up_self_attn.weight": 192}
    assert not acc.exact  # exact slice vectors discarded (the e30 forfeit)
