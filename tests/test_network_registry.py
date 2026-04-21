"""Tests for the M2 network registry and save pipeline.

Covers:

* ``resolve_network_spec`` precedence and mutual-exclusion rules.
* The ``networks.lora_save`` pipeline round-trips a synthetic state_dict
  for each save_variant, emitting the expected file(s) and preserving
  tensor shapes through the per-variant conversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from networks import (
    NETWORK_REGISTRY,
    SHARED_KWARG_FLAGS,
    NetworkSpec,
    all_network_kwargs,
    resolve_network_spec,
)
from networks import lora_save


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


EXPECTED_VARIANTS = {
    "lora",
    "ortho",
    "hydra",
    "ortho_hydra",
    "dora",
}


def test_registry_has_expected_variants():
    assert EXPECTED_VARIANTS.issubset(NETWORK_REGISTRY.keys())
    for name, spec in NETWORK_REGISTRY.items():
        assert isinstance(spec, NetworkSpec)
        assert spec.name == name


def test_all_network_kwargs_is_union_of_shared_and_specs():
    """`all_network_kwargs()` must cover every kwarg any variant declares.

    Guards against the drift mode that previously silently dropped
    `hydra_router_layers` (and its σ-router siblings): a kwarg declared on
    a NetworkSpec but missing from the forwarding list.
    """
    all_kw = set(all_network_kwargs())
    assert set(SHARED_KWARG_FLAGS).issubset(all_kw)
    for spec in NETWORK_REGISTRY.values():
        assert set(spec.kwarg_flags).issubset(all_kw), (
            f"{spec.name}.kwarg_flags has keys missing from all_network_kwargs(): "
            f"{set(spec.kwarg_flags) - all_kw}"
        )


def test_hydra_router_kwargs_registered():
    """Regression pin: the bug that motivated the M2 finish.

    `hydra_router_layers` + σ-conditional router kwargs must be registered
    on the hydra / ortho_hydra specs so they flow through argparse schema
    and into `create_network`. If these drop off the spec, the router
    silently defaults to uniform MoE over every target module.
    """
    must_have = {
        "hydra_router_layers",
        "use_sigma_router",
        "sigma_router_layers",
        "sigma_feature_dim",
        "sigma_hidden_dim",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "expert_init_std",
        "expert_warmup_ratio",
    }
    for variant in ("hydra", "ortho_hydra"):
        flags = set(NETWORK_REGISTRY[variant].kwarg_flags)
        missing = must_have - flags
        assert not missing, f"{variant} spec missing kwarg_flags: {missing}"
    # and the union exposes them
    assert must_have.issubset(set(all_network_kwargs()))


# ---------------------------------------------------------------------------
# resolve_network_spec precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "lora"),
        ({"use_ortho": "true"}, "ortho"),
        ({"use_hydra": "true"}, "hydra"),
        ({"use_hydra": "true", "use_ortho": "true"}, "ortho_hydra"),
        ({"use_dora": "true"}, "dora"),
        # bool values should resolve identically to the string form
        ({"use_hydra": True}, "hydra"),
        ({"use_hydra": False}, "lora"),
        # casing + whitespace tolerance
        ({"use_hydra": "True"}, "hydra"),
    ],
)
def test_resolve_precedence(kwargs, expected):
    spec = resolve_network_spec(kwargs)
    assert spec.name == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_dora": "true", "use_hydra": "true"},
        {"use_dora": "true", "use_ortho": "true"},
    ],
)
def test_resolve_ambiguous_raises(kwargs):
    with pytest.raises(ValueError):
        resolve_network_spec(kwargs)


# ---------------------------------------------------------------------------
# save_network_weights round-trips — synthetic state_dicts, one per variant
# ---------------------------------------------------------------------------


def _alpha(value: float) -> torch.Tensor:
    return torch.tensor(float(value))


def _make_std_lora_sd(prefix: str, r: int, in_dim: int, out_dim: int) -> dict:
    """Fake fused-qkv LoRA state_dict entry (runtime form).

    The runtime uses fused self_attn.qkv_proj; save defuses it into q/k/v.
    """
    return {
        f"{prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{prefix}.lora_up.weight": torch.randn(3 * out_dim, r),
        f"{prefix}.alpha": _alpha(r),
    }


def _save_and_reload(
    state_dict: dict,
    tmp_path: Path,
    save_variant: str,
    filename: str = "out.safetensors",
) -> dict[str, torch.Tensor]:
    out = tmp_path / filename
    lora_save.save_network_weights(
        dict(state_dict),  # copy — save mutates
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": save_variant},
        save_variant=save_variant,
    )
    # hydra writes *_moe.safetensors alongside (not the main file)
    if save_variant in ("hydra_moe", "ortho_hydra_to_hydra"):
        moe_path = tmp_path / (out.stem + "_moe.safetensors")
        assert moe_path.exists(), f"expected _moe file at {moe_path}"
        return load_file(str(moe_path))
    assert out.exists()
    return load_file(str(out))


def test_save_standard_lora_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)

    loaded = _save_and_reload(sd, tmp_path, save_variant="standard")

    # qkv_proj should be defused into q/k/v with matching shapes
    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert f"{base}_{suffix}.lora_down.weight" in loaded
        assert f"{base}_{suffix}.lora_up.weight" in loaded
        assert f"{base}_{suffix}.alpha" in loaded
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, r)
    # fused key must be gone
    assert f"{prefix}.lora_down.weight" not in loaded


def test_save_ortho_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoLoRA (PSOFT) runtime keys: Cayley params + frozen SVD bases
    sd = {
        f"{prefix}.S_p": torch.randn(r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_basis": torch.randn(3 * out_dim, r),
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_to_lora")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, r)
    for k in loaded:
        assert not k.endswith(".S_p") and not k.endswith(".S_q")
        assert not k.endswith(".P_basis") and not k.endswith(".Q_basis")


def test_save_hydra_moe_roundtrip(tmp_path: Path):
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{prefix}.lora_up_weight": torch.randn(E, 3 * out_dim, r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="hydra_moe")

    base = "lora_unet_blocks_0_self_attn"
    # per-expert ups expanded, qkv defused per-expert
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)
        assert loaded[f"{base}_{suffix}.router.weight"].shape == (E, in_dim)
        assert loaded[f"{base}_{suffix}.router.bias"].shape == (E,)
    # fused lora_up_weight must be gone (expanded into per-expert keys)
    for k in loaded:
        assert not k.endswith(".lora_up_weight")


def test_save_hydra_moe_mixed_with_plain_lora_qkv_defuses_up(tmp_path: Path):
    """Regression: when ``hydra_router_layers`` filters some fused-qkv modules
    out of MoE, the resulting plain-LoRA leg for those modules must also be
    q/k/v-defused by the hydra save pipeline. Previously only ``lora_down`` /
    ``alpha`` were split; ``lora_up.weight`` stayed fused, producing a
    mismatched checkpoint.
    """
    E, r, in_dim, out_dim = 4, 4, 8, 12

    # Hydra-routed module (cross_attn.kv — regex-matched target)
    hydra_prefix = "lora_unet_blocks_0_cross_attn_kv_proj"
    # Plain-LoRA module (self_attn.qkv — regex-excluded by hydra_router_layers)
    plain_prefix = "lora_unet_blocks_0_self_attn_qkv_proj"

    sd = {
        # hydra leg — stacked lora_up_weight
        f"{hydra_prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{hydra_prefix}.lora_up_weight": torch.randn(E, 2 * out_dim, r),
        f"{hydra_prefix}.router.weight": torch.randn(E, r),
        f"{hydra_prefix}.router.bias": torch.randn(E),
        f"{hydra_prefix}.alpha": _alpha(r),
        # plain LoRA leg — standard single lora_up.weight, no router
        f"{plain_prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{plain_prefix}.lora_up.weight": torch.randn(3 * out_dim, r),
        f"{plain_prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="hydra_moe")

    # Hydra leg: split into k/v with per-expert ups
    hydra_base = "lora_unet_blocks_0_cross_attn"
    for suffix in ("k_proj", "v_proj"):
        assert loaded[f"{hydra_base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{hydra_base}_{suffix}.lora_ups.{e}.weight"].shape == (
                out_dim,
                r,
            )

    # Plain leg: must also be defused — lora_up.weight split per q/k/v,
    # fused prefix fully gone.
    plain_base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{plain_base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{plain_base}_{suffix}.lora_up.weight"].shape == (out_dim, r), (
            f"plain-LoRA self_attn_{suffix} lora_up.weight missing or still fused — "
            "hydra save pipeline didn't defuse the plain leg"
        )
        assert f"{plain_base}_{suffix}.alpha" in loaded
        # plain leg must NOT have hydra-only keys
        assert f"{plain_base}_{suffix}.lora_ups.0.weight" not in loaded
        assert f"{plain_base}_{suffix}.router.weight" not in loaded
    # fused prefix must be entirely purged
    for k in loaded:
        assert not k.startswith(plain_prefix), (
            f"fused plain-LoRA key survived: {k}"
        )


def test_save_ortho_hydra_roundtrip(tmp_path: Path):
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoHydraLoRAExp runtime keys: S_p is 3-D (E, r, r); P_bases is (E, out, r)
    sd = {
        f"{prefix}.S_p": torch.randn(E, r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_bases": torch.randn(E, 3 * out_dim, r),
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_hydra_to_hydra")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)
    for k in loaded:
        assert not k.endswith(".S_p") and not k.endswith(".S_q")
        assert not k.endswith(".P_bases") and not k.endswith(".P_basis")


def test_save_ortho_hydra_legacy_P_basis_still_bakes(tmp_path: Path):
    """Legacy OrthoHydra checkpoints (pre-disjoint-bases) used a single
    (out, r) ``P_basis`` shared across experts. The save pipeline must still
    bake these into hydra moe form so old artifacts remain convertible.
    """
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.S_p": torch.randn(E, r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_basis": torch.randn(3 * out_dim, r),  # legacy 2-D
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
    }
    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_hydra_to_hydra")
    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)


def test_save_dora_roundtrip(tmp_path: Path):
    # DoRA is standard LoRA + a .magnitude key that renames to .dora_scale.
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)
    sd[f"{prefix}.magnitude"] = torch.randn(3 * out_dim)
    sd[f"{prefix}._org_weight_norm"] = torch.randn(3 * out_dim)  # should be dropped

    loaded = _save_and_reload(sd, tmp_path, save_variant="standard")

    base = "lora_unet_blocks_0_self_attn"
    # dora_scale should be split per-component
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.dora_scale"].shape == (out_dim,)
    # magnitude + _org_weight_norm buffers must be absent
    for k in loaded:
        assert not k.endswith(".magnitude")
        assert not k.endswith("._org_weight_norm")


# ---------------------------------------------------------------------------
# Metadata stamp
# ---------------------------------------------------------------------------


def _load_metadata(path: Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return f.metadata() or {}


def test_metadata_stamps_ss_network_spec(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)

    out = tmp_path / "out.safetensors"
    lora_save.save_network_weights(
        dict(sd),
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": "lora"},
        save_variant="standard",
    )
    meta = _load_metadata(out)
    assert meta.get("ss_network_spec") == "lora"
