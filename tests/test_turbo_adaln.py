"""Invariant tests for turbo adaln targeting (train_adaln / fake_adaln / adaln_rank).

Tier-1.5 guards for the adaln.md training path:

  * train_adaln targets the adaln_up_{branch} Linears on the student (and, by
    default, the fake) — include_patterns beating the factory default exclude;
  * fake_adaln=False keeps the student's adaln modules but builds an adaln-less
    critic (the VRAM lever);
  * adaln_rank builds the adaln modules at their own rank via the factory's
    reg_dims override while attn/MLP keep network_dim — and per-module alpha
    stays the NETWORK alpha on the reg_dims path (documented hotter scale);
  * adaln_alpha gives them their own alpha (the reg_alphas override).

No GPU, no DiT: same one-Block tiny module as tests/test_turbo_tau_critic.py
(class name "Block" is what ANIMA_TARGET_REPLACE_MODULE matches).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from networks.methods.turbo_dmd import TurboDMDNetwork

IN, HEAD, ADALN_IN = 8, 6, 4
RANK, ADALN_RANK = 4, 2


class _SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(IN, 3 * HEAD, bias=False)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttn()
        self.mlp_layer1 = nn.Linear(IN, HEAD, bias=False)
        self.adaln_up_self_attn = nn.Linear(ADALN_IN, 3 * IN, bias=False)
        self.adaln_up_cross_attn = nn.Linear(ADALN_IN, 3 * IN, bias=False)
        self.adaln_up_mlp = nn.Linear(ADALN_IN, 3 * IN, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block()])


def _make(**kwargs) -> TurboDMDNetwork:
    torch.manual_seed(0)
    return TurboDMDNetwork(unet=_TinyDiT(), student_rank=RANK, fake_rank=RANK, **kwargs)


def _adaln_loras(net) -> list:
    return [m for m in net.unet_loras if "adaln_up_" in m.lora_name]


@pytest.mark.core
def test_train_adaln_targets_student_and_fake() -> None:
    turbo = _make(train_adaln=True)
    assert len(_adaln_loras(turbo.student)) == 3
    assert len(_adaln_loras(turbo.fake)) == 3


def test_fake_adaln_opt_out() -> None:
    turbo = _make(train_adaln=True, fake_adaln=False)
    assert len(_adaln_loras(turbo.student)) == 3
    assert not _adaln_loras(turbo.fake)


def test_adaln_rank_override() -> None:
    turbo = _make(train_adaln=True, adaln_rank=ADALN_RANK, student_alpha=8)
    for net in (turbo.student, turbo.fake):
        adaln = _adaln_loras(net)
        assert adaln and all(m.lora_dim == ADALN_RANK for m in adaln)
        others = [m for m in net.unet_loras if "adaln_up_" not in m.lora_name]
        assert others and all(m.lora_dim == RANK for m in others)
    # reg_dims path keeps the NETWORK alpha → hotter alpha/rank scale on adaln.
    adaln = _adaln_loras(turbo.student)[0]
    assert float(adaln.alpha) == 8.0
    assert adaln.scale == pytest.approx(8.0 / ADALN_RANK)


def test_adaln_alpha_override() -> None:
    # Scale-preserving: network 8/4 = 2.0 → adaln alpha 2.0 × ADALN_RANK = 4.
    turbo = _make(
        train_adaln=True,
        adaln_rank=ADALN_RANK,
        adaln_alpha=4.0,
        student_alpha=8,
        fake_alpha=8,
    )
    for net in (turbo.student, turbo.fake):
        adaln = _adaln_loras(net)
        assert adaln and all(float(m.alpha) == 4.0 for m in adaln)
        assert all(m.scale == pytest.approx(2.0) for m in adaln)
        others = [m for m in net.unet_loras if "adaln_up_" not in m.lora_name]
        assert others and all(float(m.alpha) == 8.0 for m in others)
