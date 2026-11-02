"""ref_adapters = all (docs/moss/proposal/reference_adapters.md plan B): the reference
rows run a bank of their own, image and text rows the main one."""

import pytest
import torch

from library.moss.lora import LoRANetwork, load_for_inference, ref_settings
from library.moss.model import MOSSImageTransformer, RefCache
from library.moss.training.data import bank_masks

N_IMG, N_TEXT, N_REF = 24, 5, 24
SIGMA_T = torch.tensor([600.0])


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
        torch.randn(1, N_TEXT, 32, generator=g),
        torch.randn(1, 8, 4, 6, generator=g),
    )


def _network(model):
    net = LoRANetwork(
        model,
        rank=4,
        rank_pattern={"shared_block_modulation": (2, 2.0)},
        ref_rank=4,
        ref_adapters="all",
    )
    torch.manual_seed(2)
    for adapter in net.adapters.values():
        torch.nn.init.normal_(adapter.up.weight, std=0.05)
    net.apply_to()
    return net


def _rows(net):
    net.set_row_mask(*bank_masks(N_IMG, N_TEXT, None, N_REF))


def _trace(model, x, e, ref):
    """The reference tokens after each block of a joint forward."""
    seen = []

    def hook(index, out, n_img):
        seen.append(out[:, n_img + N_TEXT :].clone())
        return out

    model.layer_hook = hook
    out = model(x, SIGMA_T, e, ref=ref).sample
    model.layer_hook = None
    return out, seen


def _bank_alone(net, reference: bool):
    for name, adapter in net.adapters.items():
        adapter.multiplier = float(name.startswith("ref_") == reference)
        if name.startswith("shared_block"):
            adapter.multiplier = float(not reference)


@pytest.mark.core
def test_each_row_group_runs_its_own_bank():
    x, e, ref = _inputs()
    model = _tiny()
    net = _network(model)
    assert not any("adapters" in name for name, _ in model.named_parameters())
    with torch.no_grad():
        _rows(net)
        split, split_ref = _trace(model, x, e, ref)

        # Reference bank alone: the same reference rows.
        _bank_alone(net, reference=True)
        _, alone_ref = _trace(model, x, e, ref)
        for a, b in zip(split_ref, alone_ref):
            assert torch.allclose(a, b, atol=1e-5)

        # Main bank alone, the reference rows fed by the reference bank's stream: what
        # the image reads is the only tie between them, so compare through RefCache.
        net.set_multiplier(1.0)
        net.set_row_mask(None)
        model.ref_cache = RefCache()
        cached = model(x, SIGMA_T, e, ref=ref).sample
        model.ref_cache = None
    assert torch.allclose(split, cached, atol=1e-5)


@pytest.mark.core
def test_main_bank_stays_off_the_reference_stream():
    x, e, ref = _inputs()
    model = _tiny()
    net = _network(model)
    with torch.no_grad():
        _rows(net)
        _, first = _trace(model, x, e, ref)
        for name, adapter in net.adapters.items():
            if not name.startswith("ref_"):
                adapter.up.weight.mul_(3.0)
        _, second = _trace(model, x, e, ref)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@pytest.mark.core
def test_gradient_reaches_both_banks():
    x, e, ref = _inputs()
    model = _tiny()
    net = _network(model)
    for p in net.parameters():
        p.requires_grad_(True)
    _rows(net)
    model(x, SIGMA_T, e, ref=ref).sample.square().mean().backward()
    # The last block's reference output reaches nothing: of its bank, only k / v are read.
    dead = ("ref_layers_2_proj_q", "ref_layers_2_proj_o", "ref_layers_2_proj_ffn")
    for name, p in net.named_parameters():
        moves = p.grad is not None and p.grad.abs().sum() > 0
        assert moves != name.removeprefix("adapters.").startswith(dead), name


def test_zero_multiplier_is_the_base_forward():
    x, e, ref = _inputs()
    model = _tiny()
    with torch.no_grad():
        base = model(x, SIGMA_T, e, ref=ref).sample
    net = _network(model)
    net.set_multiplier(0.0)
    _rows(net)
    with torch.no_grad():
        off = model(x, SIGMA_T, e, ref=ref).sample
    assert torch.allclose(off, base, atol=1e-5)


@pytest.mark.core
def test_stamp_round_trips(tmp_path):
    x, e, ref = _inputs()
    model = _tiny()
    net = _network(model)
    with torch.no_grad():
        _rows(net)
        want = model(x, SIGMA_T, e, ref=ref).sample
    path = tmp_path / "bank.safetensors"
    net.save(
        path,
        "fp32",
        ref_timestep=0.0,
        ref_isolate="1",
        ref_adapters="all",
        ref_rank=4,
    )
    net.restore()
    assert model.ref_scope is None
    fresh = _tiny()
    loaded = load_for_inference(fresh, path)
    assert ref_settings(path) == (0.0, True)
    assert loaded.ref_adapters == "all" and loaded.ref_rank == 4
    assert "ref_layers_2_proj_ffn_down" in loaded.adapters
    assert len(loaded.target_names) == 3 * 6 + 1
    assert "ref_shared_block_modulation" not in loaded.adapters
    fresh.ref_cache = RefCache()
    with torch.no_grad():
        got = fresh(x, SIGMA_T, e, ref=ref).sample
    fresh.ref_cache = None
    assert torch.allclose(got, want, atol=1e-5)


def test_bank_masks_and_delta_rows():
    main, ref = bank_masks(2, 3, None, 2)
    assert main.flatten().tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert ref.flatten().tolist() == [0, 0, 0, 0, 0, 1, 1]
    main, ref = bank_masks(2, 3, [1], 2)
    assert main.flatten().tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert ref.flatten().tolist() == [0, 0, 0, 0, 0, 1, 1]
