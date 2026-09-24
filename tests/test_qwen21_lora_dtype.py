"""Dtype policy of the Qwen-Image-2.1 LoRA adapter (``library/qwen21/lora.py``).

Same contract as ``tests/test_lora_dtype_policy.py`` pins for Anima: the
adapter's parameters are fp32 master weights, and the rank GEMMs run in the
model's dtype — an fp32 adapter must never lift the activation to fp32, and a
bf16 input must come back bf16 with no fp32 tensor on the path.

CPU only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from library.qwen21.lora import LoRAAdapter, LoRANetwork  # noqa: E402


def _adapter(dtype=torch.float32):
    torch.manual_seed(0)
    adapter = LoRAAdapter(32, 24, rank=4, alpha=4.0, dtype=dtype)
    with torch.no_grad():
        adapter.up.weight.normal_()
    return adapter


def test_default_master_weights_are_fp32():
    base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
    network = LoRANetwork(torch.nn.Sequential(base), targets=r"0", rank=4)
    assert {p.dtype for p in network.parameters()} == {torch.float32}


def test_fp32_adapter_computes_in_the_model_dtype():
    adapter = _adapter(torch.float32)
    seen: list[torch.dtype] = []
    original = torch.nn.functional.linear

    def spy(x, w, b=None):
        seen.append(x.dtype)
        seen.append(w.dtype)
        return original(x, w, b)

    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    base_out = torch.randn(2, 8, 24, dtype=torch.bfloat16)
    torch.nn.functional.linear = spy
    try:
        out = adapter(x, base_out)
    finally:
        torch.nn.functional.linear = original
    assert out.dtype == torch.bfloat16
    assert seen and set(seen) == {torch.bfloat16}, seen


def test_gradients_land_on_the_fp32_masters():
    adapter = _adapter(torch.float32)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    base_out = torch.zeros(2, 8, 24, dtype=torch.bfloat16)
    adapter(x, base_out).float().sum().backward()
    assert adapter.down.weight.grad is not None
    assert adapter.down.weight.grad.dtype == torch.float32
    assert adapter.up.weight.grad.dtype == torch.float32
    assert torch.isfinite(adapter.down.weight.grad).all()


def test_multiplier_zero_is_the_base_output():
    adapter = _adapter()
    adapter.multiplier = 0.0
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    base_out = torch.randn(2, 8, 24, dtype=torch.bfloat16)
    assert adapter(x, base_out) is base_out
