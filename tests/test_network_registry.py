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
from safetensors import safe_open
from safetensors.torch import load_file

import networks as _networks_pkg

from networks import (
    NETWORK_KWARGS,
    NETWORK_REGISTRY,
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
    "hydra",
    "step_expert",
}


def test_registry_has_expected_variants():
    assert EXPECTED_VARIANTS.issubset(NETWORK_REGISTRY.keys())
    for name, spec in NETWORK_REGISTRY.items():
        assert isinstance(spec, NetworkSpec)
        assert spec.name == name


def test_all_network_kwargs_matches_allowlist():
    """`all_network_kwargs()` must be exactly the sorted ``NETWORK_KWARGS`` set.

    The two were unified (the per-variant ``kwarg_flags`` split was collapsed
    into one flat allowlist); this pins them together so the forwarding list
    and the schema-validation set never drift apart.
    """
    assert set(all_network_kwargs()) == set(NETWORK_KWARGS)
    assert list(all_network_kwargs()) == sorted(NETWORK_KWARGS)


def test_hydra_router_kwargs_registered():
    """Regression pin: the bug that motivated the M2 finish.

    `router_targets` + σ-conditional router kwargs must stay in the allowlist
    so they flow through the argparse schema and into `create_network`. If any
    drops off, the router silently defaults to uniform MoE over every target
    module.
    """
    must_have = {
        "router_targets",
        "sigma_feature_dim",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
    }
    missing = must_have - set(NETWORK_KWARGS)
    assert not missing, f"allowlist missing hydra router kwargs: {missing}"


def test_repa_kwargs_registered():
    """REPA v2 kwargs must stay in the allowlist (else use_repa is silently
    inert + the config test rejects the key). See library/training/repa.py."""
    must_have = {
        "use_repa",
        "repa_mode",
        "repa_weight",
        "repa_layer",
        "repa_encoder",
        "repa_lr_scale",
    }
    missing = must_have - set(NETWORK_KWARGS)
    assert not missing, f"allowlist missing repa kwargs: {missing}"


# ---------------------------------------------------------------------------
# Allowlist derivation — the H1 gotcha is now retired (single edit)
# ---------------------------------------------------------------------------
#
# ``NETWORK_KWARGS`` is *derived* by AST-scanning the LoRA-family consumers for
# ``kwargs.get("literal")`` reads (``networks._derive_network_kwargs``), so a new
# knob auto-registers from its read alone — no second frozenset edit. These tests
# pin the derivation's invariants; the ``must_have`` tests above pin the
# load-bearing keys as a backstop against a scan that breaks.


def test_derivation_is_sane_and_independent_of_import_order():
    """Re-deriving yields the same non-trivial set the module exposes."""
    rederived = _networks_pkg._derive_network_kwargs()
    assert rederived == NETWORK_KWARGS
    # A broken scan (wrong path, pattern miss) would collapse the set; the live
    # allowlist is ~75 keys, so anything tiny means the derivation regressed.
    assert len(NETWORK_KWARGS) > 50


def test_alias_fallbacks_excluded():
    """The back-compat alias default (router_hidden) must NOT forward — only
    its canonical name does."""
    assert "router_hidden" not in NETWORK_KWARGS
    assert "router_hidden_dim" in NETWORK_KWARGS


def test_factory_only_keys_are_derived():
    """Keys read in factory.py (not config.from_kwargs) must still be picked up —
    this is the case the audit's 'introspect from_kwargs' fix would have dropped.
    """
    for k in ("use_repa", "channel_scaling_alpha", "use_custom_down_autograd"):
        assert k in NETWORK_KWARGS, f"factory-read key {k!r} not derived"


# ---------------------------------------------------------------------------
# resolve_network_spec precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "lora"),
        ({"use_moe_style": "shared_A"}, "hydra"),
        ({"step_expert_K": "4"}, "step_expert"),
        ({"step_expert_K": "1"}, "lora"),
        # Falsey forms of use_moe_style resolve to plain LoRA.
        ({"use_moe_style": False}, "lora"),
        ({"use_moe_style": "false"}, "lora"),
        ({"use_moe_style": ""}, "lora"),
    ],
)
def test_resolve_precedence(kwargs, expected):
    spec = resolve_network_spec(kwargs)
    assert spec.name == expected


def test_independent_a_moe_style_rejected():
    """The independent-A (stacked experts) layout was removed; a stale config
    must raise rather than resolve to plain LoRA."""
    with pytest.raises(ValueError, match="expected False or 'shared_A'"):
        resolve_network_spec({"use_moe_style": "independent_A"})


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
    if save_variant == "hydra_moe":
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


def _make_adaln_lora_sd(prefix: str, r: int, in_dim: int, out_dim: int) -> dict:
    """Fake adaln_up_{branch} LoRA entry in the runtime layout."""
    return {
        f"{prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{prefix}.lora_up.weight": torch.randn(out_dim, r),
        f"{prefix}.alpha": _alpha(r),
    }


def test_save_relays_adaln_to_comfy_layout(tmp_path: Path):
    """Trained adaln keys must ship in the ComfyUI layout — ComfyUI's generic
    key map only knows ``adaln_modulation_{br}_2`` and silently drops the
    runtime ``adaln_up_{br}`` names (adaln.md §Key-naming contract)."""
    r, in_dim, out_dim = 4, 8, 12
    sd = _make_std_lora_sd("lora_unet_blocks_0_self_attn_qkv_proj", r, 8, out_dim)
    for branch in ("self_attn", "cross_attn", "mlp"):
        sd |= _make_adaln_lora_sd(
            f"lora_unet_blocks_0_adaln_up_{branch}", r, in_dim, 3 * out_dim
        )

    loaded = _save_and_reload(sd, tmp_path, save_variant="standard")

    for branch in ("self_attn", "cross_attn", "mlp"):
        comfy = f"lora_unet_blocks_0_adaln_modulation_{branch}_2"
        assert f"{comfy}.lora_down.weight" in loaded
        assert f"{comfy}.lora_up.weight" in loaded
        assert f"{comfy}.alpha" in loaded
    assert not any("adaln_up_" in k for k in loaded)
    # the non-adaln keys still take the normal defuse path
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight" in loaded

    with safe_open(str(tmp_path / "out.safetensors"), framework="pt") as f:
        assert f.metadata()["ss_adaln_layout"] == "comfy"


def test_save_adaln_relayout_inert_without_adaln(tmp_path: Path):
    """An adaln-less checkpoint is untouched — no stamp, no renames."""
    sd = _make_std_lora_sd("lora_unet_blocks_0_self_attn_qkv_proj", 4, 8, 12)

    _save_and_reload(sd, tmp_path, save_variant="standard")

    with safe_open(str(tmp_path / "out.safetensors"), framework="pt") as f:
        assert "ss_adaln_layout" not in f.metadata()


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
    """Regression: when ``router_targets`` filters some fused-qkv modules
    out of MoE, the resulting plain-LoRA leg for those modules must also be
    q/k/v-defused by the hydra save pipeline. Previously only ``lora_down`` /
    ``alpha`` were split; ``lora_up.weight`` stayed fused, producing a
    mismatched checkpoint.
    """
    E, r, in_dim, out_dim = 4, 4, 8, 12

    # Hydra-routed module (cross_attn.kv — regex-matched target)
    hydra_prefix = "lora_unet_blocks_0_cross_attn_kv_proj"
    # Plain-LoRA module (self_attn.qkv — regex-excluded by router_targets)
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
        assert not k.startswith(plain_prefix), f"fused plain-LoRA key survived: {k}"


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


# ---------------------------------------------------------------------------
# Removed adapter families — detected and refused at load
# ---------------------------------------------------------------------------


_PLAIN_KEYS = ("m.lora_down.weight", "m.lora_up.weight", "m.alpha")


@pytest.mark.parametrize(
    "extra_keys, file_metadata, expected",
    [
        ((), {}, None),
        ((), {"ss_use_moe_style": "shared_A"}, None),
        (("m.lora_ups.0.weight", "m.lora_up_weight"), {}, None),
        (("m.S_p",), {}, "undistilled OrthoLoRA"),
        (("m.S_q",), {}, "undistilled OrthoLoRA"),
        (("m.P_init",), {}, "undistilled OrthoLoRA"),
        (("m.Q_init",), {}, "undistilled OrthoLoRA"),
        (("m.lora_ups_c.0.weight",), {}, "ChimeraHydra"),
        (("m.lora_up_c_weight",), {}, "ChimeraHydra"),
        (("m.lora_downs.0.weight",), {}, "stacked-experts (FeRA)"),
        (("m.lora_down_weight",), {}, "stacked-experts (FeRA)"),
        (("register_tokens",), {}, "register-token"),
        ((), {"ss_use_chimera_hydra": "true"}, "ChimeraHydra"),
        ((), {"ss_use_moe_style": "independent_A"}, "stacked-experts (FeRA)"),
        ((), {"ss_ortho_centered_gate": "true"}, "centered-gate OrthoHydra"),
    ],
)
def test_detect_removed_variant(extra_keys, file_metadata, expected):
    from networks.lora_anima.factory import _detect_removed_variant

    sd = {k: torch.zeros(1) for k in (*_PLAIN_KEYS, *extra_keys)}
    assert _detect_removed_variant(sd, file_metadata) == expected


@pytest.mark.parametrize(
    "extra_keys, file_metadata",
    [
        (("register_tokens",), {}),
        (("m.S_p",), {}),
        ((), {"ss_use_chimera_hydra": "true"}),
    ],
)
def test_create_network_from_weights_refuses_removed_variant(extra_keys, file_metadata):
    """The refusal fires before the DiT is touched, so no model is needed."""
    from networks.lora_anima.factory import create_network_from_weights

    sd = {k: torch.zeros(1) for k in (*_PLAIN_KEYS, *extra_keys)}
    with pytest.raises(ValueError, match="no longer supported"):
        create_network_from_weights(
            1.0, None, None, None, None, weights_sd=sd, metadata=file_metadata
        )
