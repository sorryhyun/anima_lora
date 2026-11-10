"""Invariant tests for the transition-aware fake (network.fake_transition).

  * zero-init: the fake view's t-embedding is unchanged until the embed trains;
  * the delta reaches the fake view only, and depends on the source σ;
  * a fake-view forward without a source σ raises;
  * the embed's params are in fake_params() (so the fake optimizer trains them)
    and get a gradient through the t-embedding.

No GPU, no DiT: the one-Block tiny module of tests/test_turbo_adaln.py plus a
``t_embedding_norm`` and ``model_channels``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from networks.methods.turbo_dmd import TurboDMDNetwork

IN, HEAD, ADALN_IN, RANK = 8, 6, 4, 4


class _SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(IN, 3 * HEAD, bias=False)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttn()
        self.mlp_layer1 = nn.Linear(IN, HEAD, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_channels = IN
        self.blocks = nn.ModuleList([Block()])
        self.t_embedding_norm = nn.LayerNorm(IN)


def _make(**kwargs) -> TurboDMDNetwork:
    torch.manual_seed(0)
    return TurboDMDNetwork(unet=_TinyDiT(), student_rank=RANK, fake_rank=RANK, **kwargs)


def _emb() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 1, IN)


def test_off_by_default_no_hook():
    turbo = _make()
    assert turbo.fake_transition is None
    turbo.set_view("fake")
    emb = _emb()
    ref = nn.functional.layer_norm(emb, (IN,))
    torch.testing.assert_close(turbo.unet.t_embedding_norm(emb), ref)


def test_zero_init_is_identity():
    turbo = _make(fake_transition=True)
    turbo.set_view("fake")
    turbo.set_fake_source(0.5)
    emb = _emb()
    ref = nn.functional.layer_norm(emb, (IN,))
    torch.testing.assert_close(turbo.unet.t_embedding_norm(emb), ref)


def test_delta_fake_view_only_and_source_dependent():
    turbo = _make(fake_transition=True)
    nn.init.normal_(turbo.fake_transition.fc2.weight)
    emb = _emb()
    ref = nn.functional.layer_norm(emb, (IN,))
    norm = turbo.unet.t_embedding_norm

    for view in ("teacher", "student"):
        turbo.set_view(view)
        turbo.set_fake_source(0.5)
        torch.testing.assert_close(norm(emb), ref)

    turbo.set_view("fake")
    turbo.set_fake_source(0.25)
    a = norm(emb)
    turbo.set_fake_source(torch.tensor([0.75]))
    b = norm(emb)
    assert not torch.allclose(a, ref)
    assert not torch.allclose(a, b)
    # a per-sample tensor source broadcasts per row
    turbo.set_fake_source(torch.tensor([0.25, 0.75]))
    c = norm(emb)
    torch.testing.assert_close(c[0], a[0])
    torch.testing.assert_close(c[1], b[1])


def test_fake_view_without_source_raises():
    turbo = _make(fake_transition=True)
    turbo.set_view("fake")
    with pytest.raises(RuntimeError, match="no source"):
        turbo.unet.t_embedding_norm(_emb())


def test_params_in_fake_optimizer_and_get_grad():
    turbo = _make(fake_transition=True)
    ids = {id(p) for p in turbo.fake_params()}
    assert all(id(p) in ids for p in turbo.fake_transition.parameters())
    assert not any(
        id(p) in {id(q) for q in turbo.student_params()}
        for p in turbo.fake_transition.parameters()
    )
    turbo.set_view("fake")
    turbo.set_fake_source(0.5)
    turbo.unet.t_embedding_norm(_emb()).pow(2).sum().backward()
    assert turbo.fake_transition.fc2.weight.grad is not None
    assert turbo.fake_transition.fc2.weight.grad.abs().sum() > 0
