"""Invariant tests for the L_CDM off-trajectory loss (docs/anima/proposal/cdm.md Phase 1).

Two surfaces:

* ``cdm_extrapolate`` (scripts/distill_turbo/primitives.py) — the velocity-driven
  Euler extrapolation (CDM Eq. 7). It must be the rollout's own Euler form (so an
  on-grid stride reproduces a rollout step exactly) and a detached fresh launch
  point (no graph back into the rollout — the deliberate one-grad-forward
  deviation the proposal documents).
* config resolve — the --grad_ckpt guard (view × ckpt-recompute hazard class).
"""

from __future__ import annotations

import pytest
import torch

from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.primitives import cdm_extrapolate


def _resolve(cli: list[str] | None = None, cfg: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, cfg or {})


@pytest.mark.core
def test_extrapolate_on_grid_stride_matches_euler_step():
    # x(t') = x + (t' − s)·v at t' = s_next must equal the rollout's own
    # Euler update x − (s − s_next)·v.
    x = torch.randn(1, 16, 8, 8)
    v = torch.randn(1, 16, 8, 8)
    s, s_next = 0.6, 0.35
    got = cdm_extrapolate(x, v, s, torch.full((1,), s_next))
    want = x.float() - (s - s_next) * v.float()
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.core
def test_extrapolate_detached_fresh_leaf():
    # Grad-bearing inputs must NOT leak a graph into the launch point: one grad
    # forward at (x_off, t'), no second BPTT chain into the rollout.
    x = torch.randn(1, 16, 4, 4, requires_grad=True) * 2.0
    v = torch.randn(1, 16, 4, 4, requires_grad=True) * 2.0
    out = cdm_extrapolate(x, v, 0.5, torch.rand(1))
    assert not out.requires_grad
    assert out.grad_fn is None


def test_extrapolate_per_sample_stride_broadcast():
    x = torch.zeros(3, 16, 4, 4)
    v = torch.ones(3, 16, 4, 4)
    t_to = torch.tensor([0.1, 0.5, 0.9])
    out = cdm_extrapolate(x, v, 0.5, t_to)
    for b, expected in enumerate([-0.4, 0.0, 0.4]):
        assert torch.allclose(out[b], torch.full_like(out[b], expected), atol=1e-6)


def test_cdm_refuses_grad_ckpt():
    with pytest.raises(ValueError, match="grad_ckpt"):
        _resolve(["--cdm_weight", "0.1", "--grad_ckpt"])
