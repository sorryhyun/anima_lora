"""Invariant tests for the turbo student rollout (``steps.student_rollout``).

* g ≥ 1 with the anchor on is the DP-DMD step: the diversity term is backwarded
  inside, the prefix runs under no_grad, and x_pred's graph covers step g only.
* g = 0 (``dm_first_step``) makes step 0 the grad step: x_pred = ε − σ₀·v_first and
  the diversity term is left pending on the shared graph.
* ``v_target=None`` (``div_weight = 0``) drops the diversity term, and step 0 runs
  under no_grad unless it is the grad step.
* config resolve: off by default, CLI/TOML plumbing.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.steps import student_rollout

SIGMAS = [1.0, 0.75, 0.5, 0.0]
DIV_W = 0.1


def _resolve(cli: list[str] | None = None, cfg: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, cfg or {})


def _cfg(**over):
    return dataclasses.replace(
        _resolve(), div_weight=DIV_W, detach_after_first=True, **over
    )


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def v(self, x, sigma: float):
        return self.w * torch.tanh(x) + 0.1 * sigma


def _ctx(net: _Student, calls: list):
    def forward(view, x, sig, c, no_grad):
        assert view == "student"
        calls.append((round(float(sig[0]), 4), no_grad))
        with torch.no_grad() if no_grad else torch.enable_grad():
            return net.v(x, float(sig[0]))

    return SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32, forward=forward
    )


def _inputs():
    torch.manual_seed(0)
    return torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4), torch.randn(2, 3, 8)


def test_refinement_step_is_dpdmd():
    net, calls = _Student(), []
    eps, v_target, c = _inputs()
    roll = student_rollout(_ctx(net, calls), _cfg(), eps, v_target, SIGMAS, 2, c, 2)

    assert calls == [(1.0, False), (0.75, True), (0.5, False)]
    assert not roll.div_pending and roll.s_g == 0.5

    ref = _Student()
    e = eps.detach()
    v0 = ref.v(e, 1.0)
    div = nn.functional.mse_loss(v0, v_target)
    (g_div,) = torch.autograd.grad(DIV_W * div, ref.w)
    assert torch.allclose(net.w.grad, g_div)  # backwarded inside
    assert torch.allclose(roll.div_loss, div)

    with torch.no_grad():
        x1 = e - 0.25 * v0
        x2 = x1 - 0.25 * ref.v(x1, 0.75)
    x_pred = x2 - 0.5 * ref.v(x2, 0.5)
    assert torch.allclose(roll.x_pred, x_pred)
    assert torch.allclose(roll.x_g, x2)

    # The DM grad reaches the weight through step g only.
    net.w.grad = None
    roll.x_pred.sum().backward()
    (g_dm,) = torch.autograd.grad(x_pred.sum(), ref.w)
    assert torch.allclose(net.w.grad, g_dm)


def test_first_step_is_the_grad_step():
    net, calls = _Student(), []
    eps, v_target, c = _inputs()
    roll = student_rollout(_ctx(net, calls), _cfg(), eps, v_target, SIGMAS, 0, c, 2)

    assert calls == [(1.0, False)]
    assert roll.div_pending and net.w.grad is None  # nothing backwarded inside
    assert roll.s_g == 1.0 and roll.x_g is eps and roll.v_g is roll.v_first
    assert torch.equal(roll.x_pred, eps - roll.v_first)

    (roll.x_pred.sum() + DIV_W * roll.div_loss).backward()
    ref = _Student()
    v0 = ref.v(eps.detach(), 1.0)
    (g_ref,) = torch.autograd.grad(
        (eps.detach() - v0).sum() + DIV_W * nn.functional.mse_loss(v0, v_target), ref.w
    )
    assert torch.allclose(net.w.grad, g_ref)


@pytest.mark.parametrize("g", [0, 1])
def test_anchor_off(g):
    net, calls = _Student(), []
    eps, _, c = _inputs()
    roll = student_rollout(_ctx(net, calls), _cfg(), eps, None, SIGMAS, g, c, 2)

    assert not roll.div_pending and float(roll.div_loss) == 0.0
    assert net.w.grad is None
    assert calls[0] == (1.0, g != 0)  # step 0 keeps a graph only as the grad step
    roll.x_pred.sum().backward()
    assert net.w.grad is not None


def test_rollout_grad_reaches_step_zero():
    net, calls = _Student(), []
    eps, v_target, c = _inputs()
    sig = [1.0, 0.75, 0.0]
    roll = student_rollout(
        _ctx(net, calls), _cfg(rollout_grad=True), eps, v_target, sig, 1, c, 2
    )

    assert calls == [(1.0, False), (0.75, False)]
    assert roll.div_pending and net.w.grad is None  # nothing backwarded inside
    assert roll.x_g.grad_fn is not None  # step 0's graph, not a fresh leaf

    (roll.x_pred.sum() + DIV_W * roll.div_loss).backward()
    ref = _Student()
    e = eps.detach()
    v0 = ref.v(e, 1.0)
    x1 = e - 0.25 * v0  # grad kept through step 0
    x_pred = x1 - 0.75 * ref.v(x1, 0.75)
    (g_ref,) = torch.autograd.grad(
        x_pred.sum() + DIV_W * nn.functional.mse_loss(v0, v_target), ref.w
    )
    assert torch.allclose(net.w.grad, g_ref)


@pytest.mark.parametrize(
    "cli",
    [
        ["--student_steps", "4"],
        ["--student_steps", "2", "--dm_first_step"],
    ],
)
def test_rollout_grad_refuses_other_grids(cli):
    with pytest.raises(ValueError, match="rollout_grad"):
        _resolve(["--rollout_grad", *cli])


def test_config_default_off_and_overrides():
    assert _resolve().rollout_grad is False
    assert _resolve(["--rollout_grad", "--student_steps", "2"]).rollout_grad is True
    assert _resolve().dm_first_step is False
    assert _resolve(cfg={"dpdmd": {"dm_first_step": True}}).dm_first_step is True
    assert _resolve(["--dm_first_step"]).dm_first_step is True
    c = _resolve(["--no-dm_first_step"], {"dpdmd": {"dm_first_step": True}})
    assert c.dm_first_step is False
    c = _resolve(["--div_weight", "0", "--dm_first_step"])
    assert c.div_weight == 0.0 and c.dm_first_step is True
