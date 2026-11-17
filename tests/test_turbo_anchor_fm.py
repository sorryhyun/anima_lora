"""Invariant tests for the anchor feature matching (``steps.anchor_fm_loss``).

* The student gradient is that of ``mse + fm_weight·fm`` taken in one graph, with
  fm the relative L2 between the teacher's features of ε − (1 − σ_k)·v_student and
  of ε − (1 − σ_k)·v_target at σ_k.
* The teacher's real-side forward runs under no_grad; the student side with grad,
  both in the teacher view at σ_k.
* config: off by default; > 0 needs anchor_only.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.steps import _rel_feat_mse, anchor_fm_loss

SIG_K = 0.75
FM_W = 0.3


def _resolve(cli: list[str] | None = None, cfg: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, cfg or {})


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def v(self, x):
        return self.w * torch.tanh(x) + 0.1


def _teacher_feat(x, sigma: float):
    torch.manual_seed(1)
    a = torch.randn(x.shape[1], 6)
    return torch.tanh(torch.einsum("bchw,cd->bhwd", x, a) * (1.0 + sigma))


def _ctx(net: _Student, calls: list):
    views = []

    def forward(
        view,
        x,
        sig,
        c,
        *,
        no_grad,
        return_block_features=None,
        return_features_early=False,
    ):
        calls.append((view, round(float(sig[0]), 4), no_grad))
        assert view == views[-1]
        with torch.no_grad() if no_grad else torch.enable_grad():
            if view == "student":
                return net.v(x)
            assert return_features_early and return_block_features == {2}
            return {2: _teacher_feat(x, float(sig[0]))}

    def fwd(view, *a, **k):
        views.append(view)
        return forward(view, *a, **k)

    blocks = [
        SimpleNamespace(
            gradient_checkpointing=False, unsloth_offload_checkpointing=False
        )
        for _ in range(4)
    ]
    return SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        forward=fwd,
        student_sigmas=[1.0, 0.9, 0.75, 0.5],
        teacher_anchor_sigmas=[1.0, SIG_K, 0.0],
        model=SimpleNamespace(blocks=blocks, num_blocks=4),
        turbo=SimpleNamespace(set_view=views.append),
    )


def test_grad_is_mse_plus_weighted_fm():
    torch.manual_seed(0)
    eps, v_t, c = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4), torch.randn(2, 3, 8)
    cfg = SimpleNamespace(k_anchor=1, fm_weight=FM_W, fm_block_idx=-1)
    net, calls = _Student(), []
    r = anchor_fm_loss(_ctx(net, calls), cfg, eps.clone(), v_t, c, 2)

    assert calls == [
        ("student", 1.0, False),
        ("teacher", SIG_K, True),
        ("teacher", SIG_K, False),
    ]

    ref = _Student()
    v = ref.v(eps)
    mse = nn.functional.mse_loss(v, v_t)
    f_s = _teacher_feat(eps - (1 - SIG_K) * v, SIG_K)
    f_t = _teacher_feat(eps - (1 - SIG_K) * v_t, SIG_K)
    fm = _rel_feat_mse(f_s, f_t)
    (g,) = torch.autograd.grad(mse + FM_W * fm, ref.w)
    assert torch.allclose(net.w.grad, g, rtol=1e-5)
    assert torch.allclose(r.mse, mse.detach()) and torch.allclose(r.fm, fm.detach())

    v = ref.v(eps).detach().requires_grad_()
    mse = nn.functional.mse_loss(v, v_t)
    fm = _rel_feat_mse(_teacher_feat(eps - (1 - SIG_K) * v, SIG_K), f_t.detach())
    g_mse = torch.autograd.grad(mse, v)[0]
    g_fm = torch.autograd.grad(FM_W * fm, v)[0]
    ratio = g_fm.pow(2).mean().sqrt() / g_mse.pow(2).mean().sqrt()
    assert torch.allclose(r.grad_ratio, ratio, rtol=1e-5)


def test_rel_feat_mse_is_scale_free():
    torch.manual_seed(0)
    a, b = torch.randn(2, 5, 7), torch.randn(2, 5, 7)
    assert torch.allclose(_rel_feat_mse(10 * a, 10 * b), _rel_feat_mse(a, b))
    assert _rel_feat_mse(a, a) == 0


def test_config():
    c = _resolve()
    assert c.fm_weight == 0.0 and c.fm_block_idx == -1
    c = _resolve(
        cfg={"dpdmd": {"anchor_only": True, "fm_weight": 0.5, "fm_block_idx": 7}}
    )
    assert c.fm_weight == 0.5 and c.fm_block_idx == 7
    with pytest.raises((ValueError, SystemExit)):
        _resolve(cfg={"dpdmd": {"fm_weight": 0.5}})
    assert dataclasses.is_dataclass(c)
