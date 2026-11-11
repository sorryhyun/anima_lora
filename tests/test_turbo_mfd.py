"""Invariant tests for Mean Flow Distillation in the turbo DM slot (arXiv:2606.11155).

* ``interval=to_zero`` with one Euler step must be the DMD surrogate exactly (same
  RNG draws, same signal), so MFD is a strict generalisation of the shipped DM term.
* K > 1 must be the uniform-step Euler mean velocity ``(x_a − x_b)/(σ_a − σ_b)``.
* config resolve: off by default, CLI/TOML plumbing, validation.
* ``dm_slot``: ``dm_draws = 1`` is the surrogate call bit for bit; N draws average
  the signal of N sequential calls and keep the first draw's other fields.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.steps import (
    ClippedGradMean,
    _mean_velocity,
    clip_or_norm,
    dm_slot,
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


@pytest.mark.core
@pytest.mark.parametrize("dtau", [0.0, 0.25])
def test_dm_slot_draws(dtau):
    base = dataclasses.replace(_resolve(), dm_x0_norm=True, mfd_dtau=dtau)
    fn = mfd_surrogate if dtau > 0.0 else dmd_surrogate
    ctx = _ctx()
    x_pred = torch.randn(2, 16, 8, 8)
    c = torch.randn(2, 4, 32)
    c_null = torch.randn(2, 4, 32)

    torch.manual_seed(3)
    want = fn(ctx, base, x_pred, c, c_null, 2)
    torch.manual_seed(3)
    got = dm_slot(ctx, base, x_pred, c, c_null, 2)
    for f in dataclasses.fields(want):
        assert torch.equal(getattr(got, f.name), getattr(want, f.name)), f.name

    torch.manual_seed(3)
    draws = [fn(ctx, base, x_pred, c, c_null, 2) for _ in range(4)]
    torch.manual_seed(3)
    got = dm_slot(ctx, dataclasses.replace(base, dm_draws=4), x_pred, c, c_null, 2)
    mean = torch.stack([d.grad_signal for d in draws]).mean(0)
    torch.testing.assert_close(got.grad_signal, mean)
    assert not torch.allclose(got.grad_signal, draws[0].grad_signal)
    assert torch.equal(got.tau_dm, draws[0].tau_dm)
    assert torch.equal(got.eps_dm, draws[0].eps_dm)
    assert torch.equal(got.delta_dm, draws[0].delta_dm)


def test_dm_draws_config():
    assert _resolve().dm_draws == 1
    assert _resolve(cfg={"dmd": {"dm_draws": 4}}).dm_draws == 4
    assert _resolve(["--dm_draws", "3"], {"dmd": {"dm_draws": 4}}).dm_draws == 3
    with pytest.raises(ValueError, match="dm_draws"):
        _resolve(["--dm_draws", "0"])


def test_dm_tau_distribution():
    c = _resolve()
    assert c.dm_tau_distribution == "uniform" and c.dm_tau_scale == 1.0
    c = _resolve(
        cfg={"dmd": {"dm_tau_distribution": "logit_normal", "dm_tau_scale": 0.8}}
    )
    assert c.dm_tau_distribution == "logit_normal" and c.dm_tau_scale == 0.8
    with pytest.raises(ValueError, match="dm_tau_distribution"):
        _resolve(["--dm_tau_distribution", "beta"])
    ctx = _ctx()
    x_pred = torch.randn(4096, 1, 1, 1)
    c0 = torch.zeros(4096, 1, 4)
    ln = dataclasses.replace(c, dm_tau_scale=1.0)
    torch.manual_seed(0)
    tau = dmd_surrogate(ctx, ln, x_pred, c0, c0, 4096).tau_dm
    assert 0.0 < float(tau.min()) and float(tau.max()) < 1.0
    assert abs(float(tau.mean()) - 0.5) < 0.02
    assert float((tau > 0.9).float().mean()) < 0.03  # uniform would give 0.10


def test_grad_accum_is_mean_of_clipped_grads():
    assert _resolve().grad_accum == 1
    assert _resolve(cfg={"optim": {"grad_accum": 4}}).grad_accum == 4
    with pytest.raises(ValueError, match="grad_accum"):
        _resolve(["--grad_accum", "0"])

    w = torch.nn.Parameter(torch.zeros(3))
    unused = torch.nn.Parameter(torch.zeros(2))
    acc = ClippedGradMean([w, unused], 2)
    grads = [torch.tensor([30.0, 0.0, 0.0]), torch.tensor([0.0, 0.5, 0.0])]
    done = []
    for g in grads:
        w.grad = g.clone()
        clip_or_norm([w, unused], 1.0)
        done.append(acc.add())
        if not done[-1]:
            w.grad = None
    assert done == [False, True]
    torch.testing.assert_close(
        w.grad, torch.tensor([0.5, 0.25, 0.0]), atol=1e-5, rtol=0
    )
    assert float(unused.grad.abs().sum()) == 0.0
    assert acc.count == 0 and float(acc.sums[0].abs().sum()) == 0.0
