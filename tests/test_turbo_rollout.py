"""Invariant tests for the turbo student rollout (``steps.student_rollout``).

* With the anchor on, the step-0 term is backwarded inside, the steps before the
  grad step run under no_grad, and x_pred's graph covers step g only.
* ``v_target=None`` (``div_weight = 0``) drops the anchor term and step 0 runs
  under no_grad.
* The config refuses the keys of the removed rollout variants.
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
    return dataclasses.replace(_resolve(), div_weight=DIV_W, **over)


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
    assert roll.s_g == 0.5

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

    # The DM grad reaches the weight through step g only.
    net.w.grad = None
    roll.x_pred.sum().backward()
    (g_dm,) = torch.autograd.grad(x_pred.sum(), ref.w)
    assert torch.allclose(net.w.grad, g_dm)


def test_anchor_off():
    net, calls = _Student(), []
    eps, _, c = _inputs()
    roll = student_rollout(_ctx(net, calls), _cfg(), eps, None, SIGMAS, 1, c, 2)

    assert float(roll.div_loss) == 0.0
    assert net.w.grad is None
    assert calls == [(1.0, True), (0.75, False)]  # step 0 keeps no graph
    roll.x_pred.sum().backward()
    assert net.w.grad is not None


@pytest.mark.parametrize(
    "cfg",
    [
        {"dpdmd": {"dm_first_step": True}},
        {"dpdmd": {"rollout_grad": True}},
        {"dpdmd": {"teacher_handoff": True}},
        {"dmd": {"dynamic_schedule": True}},
        {"gan": {"weight_gen": 0.03}},
        {"cdm": {"weight": 1.0}},
    ],
)
def test_removed_keys_are_refused(cfg):
    with pytest.raises(ValueError, match="removed from the turbo loop"):
        _resolve(cfg=cfg)


def test_removed_keys_at_their_only_value_load():
    c = _resolve(cfg={"dpdmd": {"dm_first_step": False}, "gan": {"weight_gen": 0.0}})
    assert c.k_anchor == 7 and c.teacher_anchor_steps == 28
