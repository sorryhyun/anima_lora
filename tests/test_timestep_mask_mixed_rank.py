"""T-LoRA's shared mask must be built PER MODULE RANK, not per network.

Regression test for the mixed-rank broadcast crash: ``base.toml`` ships
``train_adaln = true`` + ``adaln_rank = 16``, so a ``network_dim = 32`` run has
its ``adaln_up_*`` Linears at rank 16 while everything else sits at 32. The old
``set_timestep_mask`` allocated one ``(1, cfg.lora_dim)`` buffer and rebound it
onto *every* module, so the ``lx * self._timestep_mask`` gate in
``lora_modules/base.py`` tried to broadcast ``(..., 16)`` against ``(1, 32)``
and blew up at the first adaln block (under compile: "RuntimeError when making
fake tensor call").

These tests poke ``set_timestep_mask`` / ``clear_timestep_mask`` directly on a
hand-built stand-in rather than a real ``LoRANetwork`` — the methods only touch
``cfg`` + the two module lists, and building a real network needs a DiT.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from networks.lora_anima.network import LoRANetwork  # noqa: E402


class _FakeCfg:
    def __init__(self, lora_dim=32, min_rank=1, alpha_rank_scale=1.0):
        self.use_timestep_mask = True
        self.lora_dim = lora_dim
        self.min_rank = min_rank
        self.alpha_rank_scale = alpha_rank_scale


class _FakeModule:
    """Minimal stand-in: the mask machinery reads ``lora_dim`` and writes
    ``_timestep_mask``."""

    def __init__(self, lora_dim):
        self.lora_dim = lora_dim
        self._timestep_mask = torch.ones(1, lora_dim)


def _net(ranks, **cfg_kwargs):
    net = LoRANetwork.__new__(LoRANetwork)
    net.cfg = _FakeCfg(**cfg_kwargs)
    net.text_encoder_loras = []
    net.unet_loras = [_FakeModule(r) for r in ranks]
    return net


def _t(value):
    return torch.tensor([value], dtype=torch.float32)


def test_modules_of_equal_rank_share_one_buffer():
    """One buffer per rank group, not one per module (the no-CPU-transfer
    property the original single-buffer design existed for)."""
    net = _net([32, 32, 16, 16])
    net.set_timestep_mask(_t(0.5))

    a, b, c, d = net.unet_loras
    assert a._timestep_mask is b._timestep_mask
    assert c._timestep_mask is d._timestep_mask
    assert a._timestep_mask is not c._timestep_mask
    assert set(net._shared_timestep_masks) == {32, 16}


@pytest.mark.core
@pytest.mark.parametrize("t_val", [0.25, 0.75])
def test_rank_groups_follow_the_same_schedule_fraction(t_val):
    """A rank-16 group must mask the same *fraction* of its rank as the rank-32
    bulk — not saturate to full rank early (which is what slicing a shared
    width-32 mask down to 16 would have done)."""
    net = _net([32, 16], min_rank=0)
    net.set_timestep_mask(_t(t_val))

    m32, m16 = (m._timestep_mask for m in net.unet_loras)
    frac32 = m32.sum().item() / 32
    frac16 = m16.sum().item() / 16
    assert frac32 == pytest.approx(frac16, abs=1.0 / 16)


def test_min_rank_clamped_into_small_groups():
    """``min_rank`` above a group's own rank must not disable the schedule for
    that group (it would pin the whole mask to ones at every t)."""
    net = _net([32, 8], min_rank=16)
    net.set_timestep_mask(_t(1.0))  # t=1 → frac=0 → everything at its floor

    m32, m8 = (m._timestep_mask for m in net.unet_loras)
    assert m32.sum().item() == 16  # floor = min_rank
    assert m8.sum().item() == 8  # floor clamped to the group's rank


@pytest.mark.core
def test_clear_restores_full_rank_on_every_group():
    net = _net([32, 16])
    net.set_timestep_mask(_t(1.0))
    assert net.unet_loras[0]._timestep_mask.sum().item() < 32

    net.clear_timestep_mask()
    for module in net.unet_loras:
        assert torch.equal(module._timestep_mask, torch.ones(1, module.lora_dim)), (
            f"rank-{module.lora_dim} group not cleared"
        )
