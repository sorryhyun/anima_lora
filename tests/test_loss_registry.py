"""Smoke tests for the M1 loss registry / composer.

Asserts the set of active losses built from each method config matches the
intended pre-refactor behavior. No forward pass — these run with a fake
network object because the composer's activation logic only inspects
attribute presence.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from library.training import (
    LOSS_REGISTRY,
    LossComposer,
    build_loss_composer,
)
from library.training.losses import LossContext


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        method="lora",
        loss_type="l2",
        masked_loss=False,
        multiscale_loss_weight=0.0,
        functional_loss_weight=0.0,
        apex_lambda_c=1.0,
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


def test_hydra_balance_activates_when_network_requests_it():
    args = _make_args(method="lora")
    composer = build_loss_composer(args, _net(_balance_loss_weight=0.01))
    assert "hydra_balance" in composer.active_losses
    assert "flow_match" in composer.active_losses


def test_apex_drops_flow_match_and_uses_apex_mix():
    """APEX subsumes flow_match into apex_mix (T_mix at lam=0 is pure FM)."""
    args = _make_args(method="apex")
    composer = build_loss_composer(args, _net())
    for name in ("apex_mix", "apex_fake", "condition_shift"):
        assert name in composer.active_losses
    assert "flow_match" not in composer.active_losses


def test_postfix_func_activates_functional():
    args = _make_args(method="postfix_func", functional_loss_weight=1.0)
    composer = build_loss_composer(args, _net())
    assert "functional" in composer.active_losses


def test_prefix_activates_multiscale_when_weight_set():
    args = _make_args(method="prefix", multiscale_loss_weight=0.5)
    composer = build_loss_composer(args, _net())
    assert "multiscale" in composer.active_losses


# -----------------------------------------------------------------------------
# Split-backward equivalence: compose_real_branch + compose_fake_branch must
# numerically match compose() for the APEX loss block when multiscale is off.
# (Multiscale blends with the real-branch scalar only — see LossComposer note.)
# -----------------------------------------------------------------------------


def _apex_loss_ctx(B: int = 2, C: int = 3, H: int = 4, W: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    args = _make_args(method="apex", apex_lambda_c=1.0)
    model_pred = torch.randn(B, C, H, W, generator=g)
    target = torch.randn(B, C, H, W, generator=g)
    timesteps = torch.rand(B, generator=g)
    weighting = torch.rand(B, 1, 1, 1, generator=g) + 0.1
    huber_c = None
    loss_weights = torch.ones(B)
    aux = {
        "apex": {
            "T_mix_v": torch.randn(B, C, H, W, generator=g),
            "F_fake_on_fake_xt": torch.randn(B, C, H, W, generator=g),
            "target_fake": torch.randn(B, C, H, W, generator=g),
            "weighting_fake": torch.rand(B, 1, 1, 1, generator=g) + 0.1,
            "t_fake": torch.rand(B, generator=g),
            "lam_inner_eff": 0.5,
            "lam_f_eff": 0.4,
        }
    }
    ctx = LossContext(
        args=args,
        batch={"loss_weights": loss_weights},
        model_pred=model_pred,
        target=target,
        timesteps=timesteps,
        weighting=weighting,
        huber_c=huber_c,
        loss_weights=loss_weights,
        network=_net(),
        aux=aux,
    )
    return args, ctx


def test_split_compose_matches_full_compose_for_apex():
    args, ctx = _apex_loss_ctx()
    composer = build_loss_composer(args, _net())
    full = composer.compose(ctx)
    real = composer.compose_real_branch(ctx)
    fake = composer.compose_fake_branch(ctx)
    assert torch.allclose(full, real + fake, atol=1e-6)


def test_compose_real_branch_excludes_apex_fake():
    """During the inline real-branch backward, F_fake_on_fake_xt is NOT yet in
    aux (forward 3 hasn't run). compose_real_branch must not require it."""
    args, ctx = _apex_loss_ctx()
    # Strip the fake-branch keys to mimic the mid-step state.
    ctx.aux["apex"].pop("F_fake_on_fake_xt")
    ctx.aux["apex"].pop("target_fake")
    ctx.aux["apex"].pop("weighting_fake")
    composer = build_loss_composer(args, _net())
    real = composer.compose_real_branch(ctx)
    assert torch.isfinite(real).all()


def test_compose_fake_branch_zero_when_lam_f_eff_zero():
    args, ctx = _apex_loss_ctx()
    ctx.aux["apex"]["lam_f_eff"] = 0.0
    composer = build_loss_composer(args, _net())
    fake = composer.compose_fake_branch(ctx)
    assert fake.item() == 0.0


def test_compose_real_branch_apex_mix_active_with_constant_lam_c():
    """L_mix outer weight is constant (apex_lambda_c) — no schedule gating
    on the outer weight. With T_mix_v in aux and lam_c=1.0 set in args, the
    real branch must produce a positive scalar."""
    args, ctx = _apex_loss_ctx()
    composer = build_loss_composer(args, _net())
    real = composer.compose_real_branch(ctx)
    assert real.item() > 0.0


def test_apex_mix_falls_back_to_flow_match_during_validation():
    """Validation runs ApexMethodAdapter.extra_forwards as a no-op (is_train=
    False), so loss_aux has no "apex" key. apex_mix must fall back to plain FM
    against ctx.target instead of returning zeros — otherwise the tracker logs
    a spurious 0 under loss/validation/*."""
    from library.training.losses import _apex_mix_loss, _flow_match_loss

    args, ctx = _apex_loss_ctx()
    ctx.aux = {}  # mimic the validation-time aux (extra_forwards skipped)
    fm_only = _flow_match_loss(ctx)
    apex_mix = _apex_mix_loss(ctx)
    assert torch.allclose(apex_mix, fm_only)
    assert apex_mix.mean().item() > 0.0


def test_method_adapter_default_split_backward_off():
    """Default MethodAdapter must not opt into split-backward; only APEX does."""
    from library.training.method_adapter import MethodAdapter
    from networks.methods.apex import ApexMethodAdapter

    assert MethodAdapter().wants_split_backward(is_train=True) is False
    assert ApexMethodAdapter().wants_split_backward(is_train=True) is True
    assert ApexMethodAdapter().wants_split_backward(is_train=False) is False
