"""Phase 2 tests for ``int8_linear``: numerics, LoRA grads, dtype casts, compile.

.venv/bin/python -m pytest project/int8_train/src/test_int8_linear.py
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from int8_linear import Int8TrainLinear, convert_, int8_mm  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

M, K, N, R = 256, 1024, 768, 16
DEV = "cuda"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def linear(seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(K, N, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.requires_grad_(False)
    return lin


@pytest.mark.parametrize("mode", ["fwd", "full"])
def test_forward_and_dx_match_bf16(mode):
    lin = linear()
    w = lin.weight.detach().clone()
    convert_(lin, mode)
    x = torch.randn(2, M, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    assert y.shape == (2, M, N) and y.dtype == torch.bfloat16
    assert rel(y, x.detach() @ w.T) < 0.03
    dy = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, dy)
    assert rel(dx, dy @ w) < 0.03


@pytest.mark.parametrize("mode", ["fwd", "full"])
def test_lora_grads_match_eager_reference(mode):
    lin = linear()
    w = lin.weight.detach().clone()
    convert_(lin, mode)
    org_forward = lin.forward  # captured after the swap, as LoRA's apply_to does
    torch.manual_seed(1)
    A = (torch.randn(R, K, device=DEV) / K**0.5).bfloat16().requires_grad_()
    B = (torch.randn(N, R, device=DEV) / R**0.5).bfloat16().requires_grad_()
    scale = 0.5
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)

    y = org_forward(x) + F.linear(F.linear(x, A), B) * scale
    dx, dA, dB = torch.autograd.grad(y, (x, A, B), g)

    with torch.no_grad():
        h = x @ A.T
        ref_dB = scale * g.T @ h
        dh = scale * g @ B
        ref_dA = dh.T @ x
        base_dx = int8_mm(g, lin.wq_t, lin.s_t) if mode == "full" else g @ w
        ref_dx = base_dx + dh @ A
    assert rel(dA, ref_dA) < 1e-2
    assert rel(dB, ref_dB) < 1e-2
    assert rel(dx, ref_dx) < 1e-2


def test_dtype_cast_keeps_fp32_scales():
    lin = linear()
    convert_(lin, "full")
    lin.to(dtype=torch.bfloat16)
    assert lin.wq.dtype == lin.wq_t.dtype == torch.int8
    assert lin.s.dtype == lin.s_t.dtype == torch.float32
    lin.to("cpu")
    assert lin.s.device.type == lin.s_t.device.type == lin.wq.device.type == "cpu"
    assert not hasattr(lin, "weight")
    assert isinstance(lin, Int8TrainLinear)


class MiniBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(K, elementwise_affine=False)
        self.layer1 = nn.Linear(K, 4 * K, bias=False)
        self.layer2 = nn.Linear(4 * K, K, bias=False)

    def forward(self, x):
        return x + self.layer2(F.gelu(self.layer1(self.norm(x)), approximate="tanh"))


@pytest.mark.parametrize("mode", ["fwd", "full"])
def test_compiled_block_has_no_graph_break(mode):
    torch.manual_seed(0)
    block = MiniBlock().to(DEV, torch.bfloat16).requires_grad_(False)
    x = torch.randn(1, M, K, device=DEV, dtype=torch.bfloat16)
    ref = block(x)
    for lin in (block.layer1, block.layer2):
        convert_(lin, mode)
    torch._dynamo.reset()
    explained = torch._dynamo.explain(block)(x)
    assert explained.graph_break_count == 0, explained.break_reasons

    compiled = torch.compile(block, fullgraph=True)
    xg = x.clone().requires_grad_()
    y = compiled(xg)
    assert rel(y, ref) < 0.03
    y.square().mean().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
