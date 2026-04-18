"""Tests for the M2 network registry and save pipeline.

Covers:

* ``resolve_network_spec`` precedence and mutual-exclusion rules.
* Every ``configs/methods/*.toml`` either resolves to a ``NetworkSpec``
  (for LoRA-family methods) or uses a non-``lora_anima`` ``network_module``.
* The ``networks.lora_save`` pipeline round-trips a synthetic state_dict
  for each save_variant, emitting the expected file(s) and preserving
  tensor shapes through the per-variant conversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from library.train_util import load_method_preset
from networks import NETWORK_REGISTRY, NetworkSpec, resolve_network_spec
from networks import lora_save
from tests.conftest import iter_method_names


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
# Every method config resolves cleanly (or uses a non-lora network_module)
# ---------------------------------------------------------------------------


def _extract_network_kwargs(merged: dict) -> dict:
    """Pull the keys relevant for resolve_network_spec from a merged config."""
    kwargs: dict = {}
    for k in ("use_hydra", "use_dora", "use_ortho"):
        if k in merged:
            kwargs[k] = merged[k]
    # network_args in TOML comes through as a list like ["mode=postfix", ...]
    for raw in merged.get("network_args") or []:
        if "=" in raw:
            key, val = raw.split("=", 1)
            kwargs[key.strip()] = val.strip()
    return kwargs


METHOD_NAMES = list(iter_method_names())


EXPECTED_SPEC_BY_METHOD = {
    "lora": "ortho",                # use_ortho = true
    "graft": "lora",                # no ortho/hydra flags
    "apex": "lora",                 # no ortho/hydra flags (warm-start from ortho checkpoint)
}

NON_LORA_METHODS = {"postfix", "postfix_exp", "postfix_func", "prefix"}


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_method_config_resolves(method: str):
    merged = load_method_preset(method, preset="default")

    network_module = merged.get("network_module", "networks.lora_anima")
    if network_module != "networks.lora_anima":
        # Postfix/prefix — covered by their own module; no NetworkSpec required.
        assert method in NON_LORA_METHODS, (
            f"{method} uses network_module={network_module!r}; add it to NON_LORA_METHODS"
        )
        return

    kwargs = _extract_network_kwargs(merged)
    spec = resolve_network_spec(kwargs)
    assert spec.name == EXPECTED_SPEC_BY_METHOD[method], (
        f"{method}: expected {EXPECTED_SPEC_BY_METHOD[method]!r}, got {spec.name!r}"
    )


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


def test_save_ortho_hydra_roundtrip(tmp_path: Path):
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoHydraLoRAExp runtime keys: S_p is 3-D (E, r, r)
    sd = {
        f"{prefix}.S_p": torch.randn(E, r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_basis": torch.randn(3 * out_dim, r),
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
