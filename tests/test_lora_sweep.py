"""``--lora_sweep``: kept-live adapters come off a loaded DiT cleanly, and the
sweep plan names one output dir per entry."""

from __future__ import annotations

import argparse

import torch

import inference
from library.inference.models import detach_adapters
from networks.lora_modules.lora import LoRAModule


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8)
        self.b = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.b(self.a(x))


def _attach(model: _Net, slot: str) -> torch.nn.Module:
    network = torch.nn.Module()
    for name in ("a", "b"):
        lora = LoRAModule(f"lora_{name}", getattr(model, name), lora_dim=2, alpha=2)
        torch.nn.init.normal_(lora.lora_up.weight)
        lora.apply_to()
        network.add_module(name, lora)
    network.eval()
    held = getattr(model, slot, None)
    setattr(model, slot, [*held, network] if isinstance(held, list) else network)
    return network


def test_detach_restores_class_forward():
    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(2, 8)
    ref = model(x)
    _attach(model, "_pgraft_network")
    assert not torch.equal(model(x), ref)

    detach_adapters(model)
    assert torch.equal(model(x), ref)
    assert "forward" not in model.a.__dict__ and "forward" not in model.b.__dict__
    assert not hasattr(model, "_pgraft_network")


def test_detach_unwinds_stacked_networks():
    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(2, 8)
    ref = model(x)
    model._pgraft_networks = []
    _attach(model, "_pgraft_networks")
    _attach(model, "_pgraft_networks")

    detach_adapters(model)
    assert torch.equal(model(x), ref)
    assert "forward" not in model.a.__dict__


def test_detach_keeps_a_forward_patched_before_the_adapter():
    """An int8 / wrapper forward installed before the LoRA stays in place."""
    torch.manual_seed(0)
    model = _Net().eval()
    inner = model.a.forward

    def doubled(x):
        return 2 * inner(x)

    model.a.forward = doubled
    x = torch.randn(2, 8)
    ref = model(x)
    _attach(model, "_pgraft_network")

    detach_adapters(model)
    assert model.a.__dict__["forward"] is doubled
    assert torch.equal(model(x), ref)


def test_sweep_plan_dirs():
    args = argparse.Namespace(
        lora_weight=["base.safetensors"],
        lora_sweep=["r/a.safetensors", "r/b.safetensors"],
        save_path="out",
    )
    assert inference._sweep_plan(args) == [
        (["base.safetensors", "r/a.safetensors"], "out/a"),
        (["base.safetensors", "r/b.safetensors"], "out/b"),
    ]
    args.lora_sweep = ["r1/a.safetensors", "r2/a.safetensors"]
    assert [d for _, d in inference._sweep_plan(args)] == ["out/00_a", "out/01_a"]
    args.lora_sweep = None
    assert inference._sweep_plan(args) == []
