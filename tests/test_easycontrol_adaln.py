"""EasyControl target-stream adaln LoRA (opt-in) — build, gating, round-trip.

Covers the contract from docs/anima/methods/adaln.md §EasyControl:
- off by default even though base.toml forwards train_adaln=true (the EC method
  TOMLs pin it false; sizing kwargs are inert while off, never a raise),
- √r-law alpha default from the cond LoRA's rank/alpha,
- zero delta at step 0 (up zero-init) so step-0 baseline equivalence holds,
- delta applied per-branch on the target modulation only,
- checkpoint round-trip through create_network_from_weights (shape-derived
  rank/in-dim, metadata-derived alpha).
"""

import math

import pytest
import torch
import torch.nn as nn

from networks.methods.easycontrol import (
    _adaln_self_cross_mlp,
    _LoRAProj,
    create_network,
    create_network_from_weights,
)


def _make_network(**kwargs):
    return create_network(
        1.0,
        kwargs.pop("network_dim", 32),
        kwargs.pop("network_alpha", 32),
        None,
        [],
        None,
        **kwargs,
    )


def test_adaln_off_when_pinned_false_despite_sizing_kwargs():
    # base.toml always forwards adaln_rank/adaln_alpha; with the method pin
    # train_adaln=false they must be inert (no raise, no modules, no keys).
    net = _make_network(train_adaln="false", adaln_rank="16", adaln_alpha="90")
    assert net.train_adaln is False
    assert net.adaln_lora_self_attn is None
    assert not any(k.startswith("adaln_lora_") for k in net.state_dict())
    assert net.metadata_fields()["ss_train_adaln"] == "0"


def test_adaln_opt_in_builds_modules_with_sqrt_alpha():
    net = _make_network(train_adaln="true", adaln_rank="8")
    assert net.train_adaln is True
    assert len(net.adaln_lora_self_attn) == net.num_blocks
    assert len(net.adaln_lora_cross_attn) == net.num_blocks
    assert len(net.adaln_lora_mlp) == net.num_blocks
    m = net.adaln_lora_self_attn[0]
    assert m.lora_down.weight.shape == (8, net.adaln_in_dim)
    assert m.lora_up.weight.shape == (3 * net.hidden_size, 8)
    # √r law: alpha = cond_lora_alpha * sqrt(adaln_rank / cond_lora_dim).
    assert net.adaln_alpha == pytest.approx(32 * math.sqrt(8 / 32))
    # Step-0 delta is exactly zero (up zero-init).
    x = torch.randn(2, 1, net.adaln_in_dim)
    assert torch.equal(m(x), torch.zeros(2, 1, 3 * net.hidden_size))
    # Trainable params include the adaln modules.
    adaln_params = {
        id(p) for ml in (net.adaln_lora_self_attn,) for p in ml.parameters()
    }
    assert adaln_params <= {id(p) for p in net.get_trainable_params()}


class _FakeBlock(nn.Module):
    def __init__(self, hidden=8, bottleneck=4):
        super().__init__()
        self.use_adaln_lora = True
        self.adaln_fused_down = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, 3 * bottleneck, bias=False)
        )
        self.adaln_up_self_attn = nn.Linear(bottleneck, 3 * hidden, bias=False)
        self.adaln_up_cross_attn = nn.Linear(bottleneck, 3 * hidden, bias=False)
        self.adaln_up_mlp = nn.Linear(bottleneck, 3 * hidden, bias=False)


def test_adaln_delta_applied_per_branch_on_target_modulation():
    torch.manual_seed(0)
    hidden, bott = 8, 4
    block = _FakeBlock(hidden, bott)
    emb = torch.randn(2, 1, hidden)
    adaln_lora = torch.randn(2, 1, 3 * hidden)
    deltas = tuple(_LoRAProj(bott, 3 * hidden, 2, 2.0) for _ in range(3))

    base = _adaln_self_cross_mlp(block, emb, adaln_lora)
    zeroed = _adaln_self_cross_mlp(block, emb, adaln_lora, adaln_deltas=deltas)
    for b_p, z_p in zip(base, zeroed):
        for b_t, z_t in zip(b_p, z_p):
            assert torch.equal(b_t, z_t)  # zero-init delta = exact baseline

    # Bump only the self-attn branch delta: self triple moves, cross/mlp don't.
    nn.init.ones_(deltas[0].lora_up.weight)
    bumped = _adaln_self_cross_mlp(block, emb, adaln_lora, adaln_deltas=deltas)
    assert not torch.equal(bumped[0][0], base[0][0])
    for br in (1, 2):
        for b_t, u_t in zip(base[br], bumped[br]):
            assert torch.equal(b_t, u_t)

    # delta_scale=0 gates the delta off entirely (the multiplier hook).
    gated = _adaln_self_cross_mlp(
        block, emb, adaln_lora, adaln_deltas=deltas, delta_scale=0.0
    )
    for b_p, g_p in zip(base, gated):
        for b_t, g_t in zip(b_p, g_p):
            assert torch.equal(b_t, g_t)


def test_adaln_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import save_file

    net = _make_network(train_adaln="true", adaln_rank="8", adaln_alpha="16.0")
    path = tmp_path / "ec_adaln.safetensors"
    save_file(net.state_dict_for_save(torch.float32), str(path), net.metadata_fields())

    loaded, weights_sd = create_network_from_weights(1.0, str(path), None, [], None)
    assert loaded.train_adaln is True
    assert loaded.adaln_rank == 8
    assert loaded.adaln_alpha == 16.0
    assert loaded.adaln_in_dim == net.adaln_in_dim
    missing, unexpected = loaded.load_state_dict(weights_sd, strict=True)
    assert not missing and not unexpected
