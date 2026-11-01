"""Save/load round-trip for AdapterNetworkBase subclasses.

Track 3 of refactoring_proposal.md: networks/methods/{easycontrol, soft_tokens}
all subclass AdapterNetworkBase. These tests pin:

  - save_weights produces a .safetensors file with the required ss_* metadata
    (ss_network_module, ss_network_spec, sshs_model_hash, sshs_legacy_hash)
    and method-specific ss_* fields
  - load_weights round-trips bit-equivalently on a fresh instance
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open


# Each entry: (label, factory, must_have_metadata_keys)
#  - factory() returns a freshly constructed adapter network.
#  - must_have_metadata_keys are method-specific ss_* stamps that should
#    appear in the saved file's metadata.
def _make_soft_tokens():
    from networks.methods.soft_tokens import SoftTokensNetwork

    return SoftTokensNetwork(
        num_tokens=2,
        embed_dim=16,
        n_layers=3,
        n_t_buckets=8,
    )


def _make_easycontrol():
    from networks.methods.easycontrol import EasyControlNetwork

    return EasyControlNetwork(
        num_blocks=2,
        hidden_size=16,
        num_heads=4,
        mlp_ratio=2.0,
        cond_lora_dim=4,
        cond_lora_alpha=4.0,
        b_cond_init=-10.0,
        cond_scale=1.0,
        apply_ffn_lora=True,
    )


CASES = [
    pytest.param(
        "soft_tokens",
        _make_soft_tokens,
        "networks.methods.soft_tokens",
        "soft_tokens",
        {"ss_num_tokens": "2", "ss_n_layers": "3", "ss_n_t_buckets": "8"},
        id="soft_tokens",
    ),
    pytest.param(
        "easycontrol",
        _make_easycontrol,
        "networks.methods.easycontrol",
        "easycontrol",
        {"ss_num_blocks": "2", "ss_cond_lora_dim": "4", "ss_apply_ffn_lora": "1"},
        id="easycontrol",
    ),
]


@pytest.mark.parametrize("label,factory,module,spec,extra_meta", CASES)
def test_save_metadata(label, factory, module, spec, extra_meta, tmp_path: Path):
    """Saved .safetensors carries the right ss_* metadata."""
    del label
    net = factory()
    file = tmp_path / "adapter.safetensors"
    net.save_weights(str(file), dtype=torch.float32, metadata={"ss_test_marker": "x"})

    assert file.exists()
    with safe_open(str(file), framework="pt") as f:
        meta = f.metadata() or {}

    assert meta.get("ss_network_module") == module
    assert meta.get("ss_network_spec") == spec
    assert meta.get("ss_test_marker") == "x"
    assert "sshs_model_hash" in meta
    assert "sshs_legacy_hash" in meta
    for k, v in extra_meta.items():
        assert meta.get(k) == v, f"{k}: expected {v!r}, got {meta.get(k)!r}"


@pytest.mark.core
@pytest.mark.parametrize("label,factory,module,spec,extra_meta", CASES)
def test_save_load_roundtrip(label, factory, module, spec, extra_meta, tmp_path: Path):
    """Saving and reloading yields the same trainable tensors."""
    del module, spec, extra_meta
    net = factory()
    file = tmp_path / f"{label}.safetensors"
    # Pin a non-default value into trainable params so we can detect failed loads.
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad or p.numel() < 32:
                p.add_(torch.randn_like(p) * 0.1)

    net.save_weights(str(file), dtype=torch.float32, metadata=None)

    fresh = factory()
    fresh.load_weights(str(file))

    # Compare every persistent tensor that's present in both state dicts.
    a = {k: v for k, v in net.state_dict().items()}
    b = {k: v for k, v in fresh.state_dict().items()}
    shared = set(a.keys()) & set(b.keys())
    assert shared, "no shared state_dict keys after round-trip"
    for k in shared:
        if k.startswith("_pe_inner."):
            continue
        torch.testing.assert_close(
            a[k].to(torch.float32),
            b[k].to(torch.float32),
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, k=k: f"mismatch at key {k!r}: {m}",
        )
