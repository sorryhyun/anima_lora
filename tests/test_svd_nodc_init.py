"""Invariants for ``down_init="weight_svd_nodc"`` and ``channel_scaling_floor``.

* every nodc ``lora_down`` row is ⊥ the calibrated input mean x̄, at the same
  ``1/sqrt(3)`` row norm as ``weight_svd`` (an init arm, not a step-size arm);
* no input mean falls back to the plain ``weight_svd`` basis;
* the floor only lifts channels under ``floor · median(mean|x|)``.
"""

import math

import pytest
import torch

from networks.lora_anima import factory
from networks.lora_modules.lora import LoRAModule

IN, OUT, RANK = 48, 40, 4


def _org() -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(IN, OUT, bias=False)


def test_nodc_rows_are_orthogonal_to_the_input_mean():
    org = _org()
    xbar = torch.randn(IN) + 3.0
    m = LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd_nodc", svd_drop=xbar)
    down = m.lora_down.weight.data.float()
    u = xbar / xbar.norm()
    assert (down @ u).abs().max().item() < 1e-5
    assert torch.allclose(
        down.norm(dim=1), torch.full((RANK,), 1 / math.sqrt(3)), atol=1e-4
    )
    plain = LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd")
    assert (plain.lora_down.weight.data.float() @ u).abs().max().item() > 1e-3


def test_nodc_without_a_mean_is_weight_svd():
    org = _org()
    a = LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd_nodc", svd_drop=None)
    b = LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd")
    assert torch.equal(a.lora_down.weight, b.lora_down.weight)


def test_channel_scaling_floor_lifts_only_dead_channels(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    mean_abs = torch.ones(IN)
    mean_abs[:3] = 1e-5  # dead
    mean_abs[3] = 0.5  # quiet but live
    path = tmp_path / "channel_stats.safetensors"
    save_file({"m": mean_abs}, str(path))
    monkeypatch.setattr(factory, "_CHANNEL_STATS_PATH", path)

    off = factory._load_channel_scales({"channel_scaling_alpha": "0.5"})["m"]
    on = factory._load_channel_scales(
        {"channel_scaling_alpha": "0.5", "channel_scaling_floor": "0.01"}
    )["m"]
    # step boost of a dead channel over a median one: (median / floor)^alpha = 10
    assert (off[4] / off[:3]).min().item() > 100
    assert (on[4] / on[:3]).max().item() == pytest.approx(10.0, rel=1e-4)
    # live channels keep their ratios (only the mean-normalisation moves)
    assert (on[3] / on[4]).item() == pytest.approx((off[3] / off[4]).item(), rel=1e-5)
