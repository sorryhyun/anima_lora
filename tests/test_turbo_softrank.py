"""Turbo soft-rank caption-auxiliary wiring tests (turbo_caption_ranking.md Phase 1).

Covers the load-bearing invariants:
  - ``softrank.weight = 0`` default ⇒ the whole path is off (no extra forwards, no
    extra RNG draws) → byte-identical DP-DMD. The guarantee is structural: the
    distill loop gates every soft-rank effect on ``cfg.softrank_weight > 0.0``.
  - TOML / CLI precedence for the ``[softrank]`` keys.
  - Config validation: negative weight, k < 2, bad every_n, non-dpdmd base,
    block-swap conflict.
  - The ranking-loss math (``caption_rank_loss``): the matched caption's soft rank
    falls toward 0 as it out-scores the negatives, and the gradient through the
    matched velocity reduces its distance to the diversity anchor (the negatives,
    being detached, carry no gradient).
"""

from __future__ import annotations

import pytest
import torch

from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.softrank import (
    CaptionNegativePool,
    caption_rank_loss,
    soft_rank,
)


# ── config resolution ─────────────────────────────────────────────────────────


def _resolve(cli: list[str] | None = None, toml: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, toml or {})


def test_config_defaults_off():
    c = _resolve()
    assert c.softrank_weight == 0.0  # off ⇒ byte-identical DP-DMD
    assert c.softrank_k == 2
    assert c.softrank_every_n == 4
    assert c.softrank_softness == 0.1
    assert c.softrank_pool_size == 64
    assert c.softrank_warmup_ratio == 1.0


def test_config_toml_and_cli_precedence():
    toml = {
        "softrank": {
            "weight": 0.05,
            "k": 3,
            "every_n": 2,
            "softness": 0.2,
            "pool_size": 128,
            "warmup_ratio": 0.5,
        }
    }
    c = _resolve(toml=toml)
    assert c.softrank_weight == 0.05
    assert c.softrank_k == 3
    assert c.softrank_every_n == 2
    assert c.softrank_softness == 0.2
    assert c.softrank_pool_size == 128
    assert c.softrank_warmup_ratio == 0.5
    # CLI overrides win over TOML.
    c = _resolve(
        [
            "--softrank_weight",
            "0.1",
            "--softrank_k",
            "4",
            "--softrank_softness",
            "0.05",
        ],
        toml=toml,
    )
    assert c.softrank_weight == 0.1
    assert c.softrank_k == 4
    assert c.softrank_softness == 0.05
    assert c.softrank_every_n == 2  # untouched by CLI → keeps TOML


def test_config_validation():
    with pytest.raises(ValueError, match="softrank.weight"):
        _resolve(toml={"softrank": {"weight": -0.1}})
    with pytest.raises(ValueError, match="softrank.k"):
        _resolve(toml={"softrank": {"weight": 0.05, "k": 1}})
    with pytest.raises(ValueError, match="softrank.every_n"):
        _resolve(toml={"softrank": {"weight": 0.05, "every_n": 0}})
    # The term sites on the DP-DMD step-0 anchor — plain DMD2 has none.
    with pytest.raises(ValueError, match="base_loss='dpdmd'"):
        _resolve(toml={"base_loss": "dmd", "softrank": {"weight": 0.05}})
    # Extra caption-negative forwards are unaudited under block swap.
    with pytest.raises(ValueError, match="blocks_to_swap"):
        _resolve(["--blocks_to_swap", "2"], toml={"softrank": {"weight": 0.05}})
    # Pool must hold at least k captions or it never reaches `ready`.
    with pytest.raises(ValueError, match="pool_size"):
        _resolve(toml={"softrank": {"weight": 0.05, "k": 4, "pool_size": 2}})
    # warmup_ratio is a fraction.
    with pytest.raises(ValueError, match="warmup_ratio"):
        _resolve(toml={"softrank": {"weight": 0.05, "warmup_ratio": 1.5}})


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


def test_pool_capacity_is_bounded():
    pool = CaptionNegativePool(capacity=3)
    for i in range(10):
        pool.add(torch.full((1, 2, 2), float(i)))
    assert len(pool._buf) == 3
    # Ring keeps the most recent → the last add (9) is present.
    assert any(b[0, 0].item() == 9.0 for b in pool._buf)


# ── ranking-loss math ─────────────────────────────────────────────────────────


def test_soft_rank_orders_by_score():
    # Higher score ⇒ lower (better) rank; a clear max gets ~rank 0.
    r = soft_rank(torch.tensor([3.0, 1.0, 0.0]), tau=0.05)
    assert r[0] < r[1] < r[2]
    assert r[0] == pytest.approx(0.0, abs=1e-2)


def test_caption_rank_loss_matched_wins_is_near_zero():
    """Matched velocity == anchor (perfect), negatives far ⇒ soft rank → 0."""
    torch.manual_seed(0)
    v_target = torch.randn(2, 4, 3, 3)
    v_pos = v_target.clone()  # zero distance → highest reward
    v_negs = [v_target + 5.0, v_target - 5.0]  # large distance → low reward
    loss = caption_rank_loss(v_pos, v_negs, v_target, tau=0.1)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


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
