"""Invariant tests for Mean Flow Distillation in the turbo DM slot (arXiv:2606.11155).

* ``interval=to_zero`` with one Euler step must be the DMD surrogate exactly (same
  RNG draws, same signal), so MFD is a strict generalisation of the shipped DM term.
* K > 1 must be the uniform-step Euler mean velocity ``(x_a − x_b)/(σ_a − σ_b)``.
* config resolve: off by default, CLI/TOML plumbing, validation.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.steps import (
    _mean_velocity,
    dmd_surrogate,
    mfd_steps,
    mfd_surrogate,
)


def _resolve(cli: list[str] | None = None, cfg: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, cfg or {})


def _v_teacher(x, sig, c, c_null):
    s = sig.float().view(-1, 1, 1, 1)
    return torch.tanh(x.float()) * (1.0 + s) + 0.3 * c.mean() - 0.1 * c_null.mean()


def _ctx():
    def forward(view, x, sig, c, no_grad=True):
        assert view == "fake"
        s = sig.float().view(-1, 1, 1, 1)
        return torch.sin(x.float()) * (0.5 + s) + 0.2 * c.mean()

    return SimpleNamespace(
        turbo=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        forward=forward,
        teacher_cfg_velocity=_v_teacher,
    )


@pytest.mark.core
@pytest.mark.parametrize("x0_norm", [False, True])
def test_one_step_to_zero_equals_dmd(x0_norm):
    cfg = dataclasses.replace(
        _resolve(),
        dm_x0_norm=x0_norm,
        mfd_dtau=10.0,
        mfd_max_steps=1,
        mfd_interval="to_zero",
    )
    ctx = _ctx()
    x_pred = torch.randn(3, 16, 8, 8)
    c = torch.randn(3, 4, 32)
    c_null = torch.randn(3, 4, 32)

    torch.manual_seed(7)
    want = dmd_surrogate(ctx, cfg, x_pred, c, c_null, 3)
    torch.manual_seed(7)
    got = mfd_surrogate(ctx, cfg, x_pred, c, c_null, 3)

    assert torch.equal(got.grad_signal, want.grad_signal)
    assert torch.equal(got.tau_dm, want.tau_dm)
    assert torch.equal(got.eps_dm, want.eps_dm)
    assert torch.equal(got.tau_dm_e, want.tau_dm_e)
    assert torch.equal(got.delta_dm, want.delta_dm)


@pytest.mark.core
def test_multi_step_is_euler_mean_velocity():
    torch.manual_seed(0)
    x_a = torch.randn(2, 16, 4, 4)
    sa = torch.tensor([0.9, 0.4])
    sb = torch.tensor([0.0, 0.1])
    K = 4

    def v_fn(x, sig):
        return 0.7 * x + sig.view(-1, 1, 1, 1)

    u = _mean_velocity(v_fn, x_a, sa, sb, K)

    x = x_a.clone()
    for k in range(K):
        sig = sa + (sb - sa) * (k / K)
        x = x - ((sa - sb) / K).view(-1, 1, 1, 1) * v_fn(x, sig)
    u_ref = (x_a - x) / (sa - sb).view(-1, 1, 1, 1)
    assert torch.allclose(u, u_ref, atol=1e-5)


def test_uniform_interval_ends_inside_start():
    cfg = dataclasses.replace(
        _resolve(), mfd_dtau=0.25, mfd_max_steps=4, mfd_interval="uniform"
    )
    torch.manual_seed(3)
    out = mfd_surrogate(
        _ctx(),
        cfg,
        torch.randn(4, 16, 4, 4),
        torch.randn(4, 4, 8),
        torch.randn(4, 4, 8),
        4,
    )
    span = out.tau_dm_e.view(-1)
    assert torch.all(span >= 0) and torch.all(span <= out.tau_dm)
    assert torch.isfinite(out.grad_signal).all()


def test_step_count():
    assert mfd_steps(0.95, 0.1, 4) == 4
    assert mfd_steps(0.95, 0.5, 4) == 2
    assert mfd_steps(0.3, 0.1, 8) == 3
    assert mfd_steps(1e-4, 0.1, 4) == 1


def test_config_default_off_and_overrides():
    c = _resolve()
    assert c.mfd_dtau == 0.0 and c.mfd_max_steps == 4 and c.mfd_interval == "to_zero"
    c = _resolve(cfg={"mfd": {"dtau": 0.1, "interval": "uniform"}})
    assert c.mfd_dtau == 0.1 and c.mfd_interval == "uniform"
    c = _resolve(["--mfd_dtau", "0.05", "--mfd_max_steps", "6"])
    assert c.mfd_dtau == 0.05 and c.mfd_max_steps == 6
    with pytest.raises(ValueError, match="mfd.dtau"):
        _resolve(["--mfd_dtau", "-0.1"])
