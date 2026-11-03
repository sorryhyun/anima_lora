"""ref_adapters = kv (_archive/proposals/reference_adapters.md plan A): the reference
rows stay on Base, only the export deltas leave them."""

import pytest
import torch

from library.moss.lora import LoRANetwork, ref_settings
from library.moss.model import MOSSImageTransformer, RefCache


def _tiny():
    torch.manual_seed(0)
    model = MOSSImageTransformer(
        dim=96,
        ffn_dim=128,
        num_heads=4,
        num_layers=3,
        text_dim=32,
        latent_channels=8,
    ).eval()
    model.requires_grad_(False)
    model.ref_timestep, model.ref_isolate = 0.0, True
    return model


def _inputs():
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(1, 8, 4, 6, generator=g),
        torch.randn(1, 5, 32, generator=g),
        torch.randn(1, 8, 4, 6, generator=g),
    )


def _network(model, ref_rank=4):
    net = LoRANetwork(
        model,
        rank=4,
        rank_pattern={"shared_block_modulation": (2, 2.0)},
        ref_rank=ref_rank,
    )
    torch.manual_seed(2)
    for adapter in net.adapters.values():
        torch.nn.init.normal_(adapter.up.weight, std=0.05)
    net.apply_to()
    return net


def _pass(model, x, e, ref):
    return model(x, torch.tensor([600.0]), e, ref=ref).sample


def _reference_stream(model, x, e, ref):
    """Per-block reference ``x`` / ``k`` / ``v`` the DiT builds for ``ref``."""
    tokens, rope = model.embed_sample(x, e, ref)
    n_ref = ref.shape[-2] * ref.shape[-1]
    mod, _ = model.time_conditioning(600.0, 1)
    mod, n_ref = model.ref_modulation(mod, tuple(ref.shape[-2:]))
    tokens, rope = tokens[:, -n_ref:], rope[:, -n_ref:]
    out = []
    for layer in model.layers:
        tokens, k, v = layer(tokens, rope, mod[:, :, 1], kv_out=True)
        out.append((tokens, k, v))
    return out


@pytest.mark.core
def test_reference_rows_stay_on_base():
    x, e, ref = _inputs()
    model = _tiny()
    with torch.no_grad():
        base = _reference_stream(model, x, e, ref)
    net = _network(model)
    with torch.no_grad():
        ours = _reference_stream(model, x, e, ref)
        for adapter_name, adapter in net.adapters.items():
            if adapter_name.startswith("ref_"):
                adapter.multiplier = 0.0
        plain = _reference_stream(model, x, e, ref)
    for (bx, bk, bv), (ox, ok, ov), (px, pk, pv) in zip(base, ours, plain):
        assert torch.equal(bx, ox) and torch.equal(bx, px)
        assert torch.equal(bk, pk) and torch.equal(bv, pv)
        assert not torch.equal(bk, ok) and not torch.equal(bv, ov)


@pytest.mark.core
def test_gradient_leaves_the_reference_only_through_the_bank():
    x, e, ref = _inputs()
    model = _tiny()
    net = _network(model)
    for p in net.parameters():
        p.requires_grad_(True)
    _pass(model, x, e, ref).square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_zero_multiplier_is_the_joint_base_forward():
    x, e, ref = _inputs()
    model = _tiny()
    with torch.no_grad():
        joint = _pass(model, x, e, ref)
    net = _network(model)
    net.set_multiplier(0.0)
    with torch.no_grad():
        split = _pass(model, x, e, ref)
    assert torch.allclose(split, joint, atol=1e-5)


@pytest.mark.core
def test_ref_cache_matches_the_uncached_forward():
    x, e, ref = _inputs()
    model = _tiny()
    _network(model)
    with torch.no_grad():
        uncached = _pass(model, x, e, ref)
        model.ref_cache = RefCache()
        cached = _pass(model, x, e, ref)
        model.ref_cache = None
    assert torch.allclose(cached, uncached, atol=1e-5)


@pytest.mark.core
def test_stamp_round_trips(tmp_path):
    x, e, ref = _inputs()
    model = _tiny()
    net = _network(model)
    with torch.no_grad():
        want = _pass(model, x, e, ref)
    path = tmp_path / "kv.safetensors"
    net.save(
        path,
        "fp32",
        ref_timestep=0.0,
        ref_isolate="1",
        ref_adapters="kv",
        ref_rank=4,
    )
    net.restore()
    assert all(block.ref_bank is None for block in model.layers)
    loaded = LoRANetwork.for_inference(model, path)
    assert loaded.ref_rank == 4 and ref_settings(path) == (0.0, True)
    with torch.no_grad():
        assert torch.allclose(_pass(model, x, e, ref), want, atol=1e-5)
    assert "ref_layers_2_proj_v" in loaded.adapters
    assert len(loaded.target_names) == 3 * 6 + 1
