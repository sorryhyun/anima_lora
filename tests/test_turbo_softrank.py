"""Turbo soft-rank caption-auxiliary wiring tests (turbo_caption_ranking.md Phase 1).

Covers the load-bearing invariants:
  - The ranking-loss math (``caption_rank_loss``): the matched caption's soft rank
    falls toward 0 as it out-scores the negatives, and the gradient through the
    matched velocity reduces its distance to the diversity anchor (the negatives,
    being detached, carry no gradient).
"""

from __future__ import annotations

import pytest
import torch

from scripts.distill_turbo.softrank import (
    CaptionNegativePool,
    caption_rank_loss,
    soft_rank,
)


# ── caption negative pool ─────────────────────────────────────────────────────


def test_pool_fires_at_batch_size_one():
    """The cross-step pool yields negatives at B=1 (the bug the pool fixes) —
    capacity-bounded, drawn with the right [batch, seq, dim] shape."""
    pool = CaptionNegativePool(capacity=4)
    assert not pool.ready(2)  # empty
    for _ in range(3):  # three B=1 steps
        pool.add(torch.randn(1, 8, 5))
    assert pool.ready(2)
    negs = pool.draw(k=2, batch=1)
    assert len(negs) == 2
    assert all(n.shape == (1, 8, 5) for n in negs)


# ── ranking-loss math ─────────────────────────────────────────────────────────


def test_soft_rank_orders_by_score():
    # Higher score ⇒ lower (better) rank; a clear max gets ~rank 0.
    r = soft_rank(torch.tensor([3.0, 1.0, 0.0]), tau=0.05)
    assert r[0] < r[1] < r[2]
    assert r[0] == pytest.approx(0.0, abs=1e-2)


def test_caption_rank_loss_matched_loses_is_larger():
    """When a negative explains the anchor better, the matched rank is worse."""
    torch.manual_seed(0)
    v_target = torch.randn(2, 4, 3, 3)
    v_pos = v_target + 5.0  # matched is the FAR one
    v_negs = [v_target.clone(), v_target + 5.0]  # one negative is perfect
    loss = caption_rank_loss(v_pos, v_negs, v_target, tau=0.1)
    # At least one negative out-scores the matched caption → rank ≳ 1.
    assert loss.item() > 0.9


def test_caption_rank_loss_grad_only_through_matched():
    """Gradient flows through v_pos only; detached negatives stay grad-free, and
    the step reduces the matched caption's distance to the anchor."""
    torch.manual_seed(0)
    v_target = torch.randn(1, 4, 3, 3)
    v_pos = (v_target + 0.5).clone().requires_grad_()
    # Negatives are what the no_grad forwards produce — explicitly detached.
    v_negs = [(v_target + 0.2).detach(), (v_target - 0.3).detach()]
    loss = caption_rank_loss(v_pos, v_negs, v_target, tau=0.1)
    loss.backward()
    assert v_pos.grad is not None and v_pos.grad.abs().sum() > 0
    # A gradient-descent step on v_pos moves it toward the anchor (lower MSE).
    mse_before = ((v_pos - v_target) ** 2).mean().item()
    with torch.no_grad():
        v_stepped = v_pos - 1.0 * v_pos.grad
    mse_after = ((v_stepped - v_target) ** 2).mean().item()
    assert mse_after < mse_before
