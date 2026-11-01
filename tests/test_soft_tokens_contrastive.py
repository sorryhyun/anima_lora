"""Tests for the soft-tokens contrastive objective (Phase 1).

Covers the InfoNCE loss math, the warmup-gated weight contract, the cadence over
optimizer steps, and the dual-bank checkpoint layout. The dataset-sourcing and extra-forward wiring are integration-tested elsewhere;
here we exercise the pure-CPU network surface. See
docs/anima/proposal/soft_tokens_contrastive.md.
"""

from __future__ import annotations

import math

import pytest
import torch

from networks.methods.soft_tokens import SoftTokensNetwork


def _net(**kw):
    base = dict(
        num_tokens=4,
        embed_dim=16,
        n_layers=2,
        n_t_buckets=4,
        init_std=0.02,
    )
    base.update(kw)
    return SoftTokensNetwork(**base)


def test_contrastive_loss_perfect_positive():
    """v_pos == target, v_neg far → low loss, accuracy 1, positive gap."""
    net = _net(contrastive_weight=0.1, contrastive_tau=0.5)
    target = torch.zeros(1, 4, 8, 8)
    v_pos = torch.zeros(1, 4, 8, 8)  # pos_err = 0  → logit_pos = 0
    v_neg = torch.ones(1, 1, 4, 8, 8)  # neg_err = 1  → logit_neg = -1/0.5 = -2
    loss, diag = net.contrastive_loss(v_pos, v_neg, target)

    expected = -0.0 + math.log(math.exp(0.0) + math.exp(-2.0))
    assert loss.item() == pytest.approx(expected, abs=1e-5)
    assert diag["contrastive_acc"] == 1.0
    assert diag["contrastive_logit_gap"] == pytest.approx(2.0, abs=1e-5)


def test_contrastive_warmup_gate():
    net = _net(contrastive_weight=0.3, contrastive_warmup_ratio=0.1)
    assert net._contrastive_weight == 0.0  # held during warmup
    net.step_contrastive_warmup(global_step=5, max_train_steps=100)
    assert net._contrastive_weight == 0.0
    net.step_contrastive_warmup(global_step=10, max_train_steps=100)
    assert net._contrastive_weight == 0.3
    net.step_contrastive_warmup(global_step=50, max_train_steps=100)
    assert net._contrastive_weight == 0.3


def test_contrastive_every_n_uniform_within_accum_window():
    net = _net(
        contrastive_weight=0.3, contrastive_warmup_ratio=0.0, contrastive_every_n=2
    )
    # accum=2 → optimizer steps {0,0,1,1,2,2,3,3}; fire when opt_step even.
    fired = []
    for micro in range(8):
        net.step_contrastive_warmup(global_step=micro, max_train_steps=100, accum=2)
        fired.append(net._contrastive_fire_this_step)
    # Both micro-batches of each optimizer window agree.
    assert fired == [True, True, False, False, True, True, False, False]


def test_jaccard_penalty_lowers_loss():
    """Down-weighting a negative's logit (jaccard mode) makes the positive win
    more easily → strictly lower InfoNCE loss than the unpenalized case."""
    net = _net(contrastive_weight=0.1, contrastive_tau=0.5)
    target = torch.zeros(1, 4, 8, 8)
    v_pos = torch.full((1, 4, 8, 8), 0.5)  # some pos error
    v_neg = torch.full((1, 2, 4, 8, 8), 0.5)  # negatives equally close
    base, _ = net.contrastive_loss(v_pos, v_neg, target)
    penalty = torch.full((1, 2), 1.0)  # α·s on every negative
    penalized, _ = net.contrastive_loss(v_pos, v_neg, target, neg_penalty=penalty)
    assert penalized.item() < base.item()


# ── Dual bank ψ⁺/ψ⁻ ──────────────────────────────────────────────────────────


def test_single_bank_is_default_and_3d():
    """Default (no flag) stays single-bank with the unchanged 3D token shape +
    legacy on-disk t_offsets width — Phase-2 checkpoints keep loading."""
    net = _net()
    assert net.n_banks == 1 and net.dual_bank is False
    assert net.tokens.shape == (2, 4, 16)  # (n_layers, K, D)
    assert net.t_offsets.weight.shape == (4, 2 * 16)  # (n_t_buckets, n_layers·D)
    md = net.metadata_fields()
    assert md["ss_n_banks"] == "1" and md["ss_dual_bank"] == "false"


def test_dual_bank_branches_are_independent():
    """ψ⁺ (branch 0) and ψ⁻ (branch 1) produce different step tokens — they are
    separately initialized, so the splice picks the right region per branch."""
    net = _net(dual_bank=True)
    t = torch.full((1,), 0.3)
    net._set_step_tokens(t, None, branch=0)
    psi_plus = net._step_layer_tokens.clone()
    net._set_step_tokens(t, None, branch=1)
    psi_minus = net._step_layer_tokens.clone()
    assert psi_plus.shape == psi_minus.shape
    assert not torch.allclose(psi_plus, psi_minus)
    # branch 0 must equal the explicit ψ⁺ lookup (append_postfix's branch).
    expected = net._layer_tokens_from(net.tokens, net.t_offsets.weight, t, branch=0)
    assert torch.allclose(psi_plus, expected)


def test_dual_bank_append_postfix_uses_psi_plus():
    """append_postfix (anchor + inference path) always splices ψ⁺ (branch 0)."""
    net = _net(dual_bank=True)
    t = torch.full((1,), 0.5)
    net.append_postfix(torch.zeros(1, 8, 16), torch.tensor([8]), timesteps=t)
    got = net._step_layer_tokens
    expected = net._layer_tokens_from(net.tokens, net.t_offsets.weight, t, branch=0)
    assert torch.allclose(got, expected)


@pytest.mark.core
def test_dual_bank_file_roundtrip_inference(tmp_path):
    """End-to-end: save a dual checkpoint, build an inference net via
    create_network_from_weights (single ψ⁺), load_weights from the file — the
    inference bank equals the saved ψ⁺ branch."""
    from safetensors.torch import save_file

    from networks.methods.soft_tokens import create_network_from_weights

    dual = _net(dual_bank=True)
    with torch.no_grad():
        dual.tokens.copy_(torch.randn_like(dual.tokens))
        dual.t_offsets.weight.copy_(torch.randn_like(dual.t_offsets.weight))
    sd = dual.state_dict_for_save(torch.float32)
    path = tmp_path / "dual.safetensors"
    save_file(dict(sd), str(path), metadata=dual.metadata_fields())

    net, _ = create_network_from_weights(
        1.0, str(path), None, None, None, for_inference=True
    )
    assert net.n_banks == 1  # inference is ψ⁺-only
    net.load_weights(str(path))
    assert torch.allclose(net.tokens, dual.tokens[0])
    assert torch.allclose(net.t_offsets.weight, dual.t_offsets.weight[:, : 2 * 16])
