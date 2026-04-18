"""Smoke tests for the M1 loss registry / composer.

Asserts the set of active losses built from each method config matches the
intended pre-refactor behavior. No forward pass — these run with a fake
network object because the composer's activation logic only inspects
attribute presence.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from library.training import (
    LOSS_REGISTRY,
    LossComposer,
    build_loss_composer,
)


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        method="lora",
        loss_type="l2",
        masked_loss=False,
        multiscale_loss_weight=0.0,
        functional_loss_weight=0.0,
        apex_lambda_p=1.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _net(**attrs) -> SimpleNamespace:
    # Composer reads `_ortho_reg_weight` and `_balance_loss_weight`.
    defaults = {"_ortho_reg_weight": 0.0, "_balance_loss_weight": 0.0}
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def test_registry_contains_expected_keys():
    assert {
        "flow_match",
        "apex_mix",
        "apex_fake",
        "ortho_reg",
        "hydra_balance",
        "functional",
        "condition_shift",
        "multiscale",
    }.issubset(set(LOSS_REGISTRY.keys()))


def test_lora_composer_is_flow_match_only():
    args = _make_args(method="lora")
    composer = build_loss_composer(args, _net())
    assert isinstance(composer, LossComposer)
    assert composer.active_losses == ["flow_match"]
    assert composer.l_sup_scalar == 1.0


def test_hydra_balance_activates_when_network_requests_it():
    args = _make_args(method="lora")
    composer = build_loss_composer(args, _net(_balance_loss_weight=0.01))
    assert "hydra_balance" in composer.active_losses
    assert "flow_match" in composer.active_losses


def test_apex_activates_mix_fake_conditionshift_and_lambda_p():
    args = _make_args(method="apex", apex_lambda_p=0.7)
    composer = build_loss_composer(args, _net())
    for name in ("flow_match", "apex_mix", "apex_fake", "condition_shift"):
        assert name in composer.active_losses
    assert composer.l_sup_scalar == pytest.approx(0.7)


def test_postfix_func_activates_functional():
    args = _make_args(method="postfix_func", functional_loss_weight=1.0)
    composer = build_loss_composer(args, _net())
    assert "functional" in composer.active_losses


def test_prefix_activates_multiscale_when_weight_set():
    args = _make_args(method="prefix", multiscale_loss_weight=0.5)
    composer = build_loss_composer(args, _net())
    assert "multiscale" in composer.active_losses
