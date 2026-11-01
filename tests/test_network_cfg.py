"""Tests for ``LoRANetworkCfg.from_kwargs`` and ``from_weights``.

The kwarg parsing was previously inlined ~30x in factory.create_network as
``str.lower() == "true"`` / ``int(...)`` / ``float(...)`` blocks, with no
test coverage on that boilerplate. Pin the parsing here so future cfg-shape
changes can't silently regress on str→T casts or default fallbacks.
"""

from __future__ import annotations

import pytest

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_modules import HydraLoRAModule, LoRAModule


@pytest.mark.core
def test_string_bool_parsing_matches_old_factory_path():
    """Every bool kwarg comes in as a literal "true"/"false" string from
    train.py's net_kwargs. Make sure the canonical 'true' parses true,
    arbitrary other strings parse false, and bool/None still work. The new
    three-axis routing keys parse as expected.
    """
    kwargs = {
        "train_llm_adapter": "true",
        "use_timestep_mask": "TRUE",
        "use_moe_style": "shared_A",
        "route_per_layer": "true",
        "router_source": "sigma",
        "verbose": "false",
    }
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=8,
        network_alpha=4.0,
        neuron_dropout=0.1,
        module_class=LoRAModule,
    )
    assert cfg.train_llm_adapter is True
    assert cfg.use_timestep_mask is True
    assert cfg.use_moe_style == "shared_A"
    assert cfg.route_per_layer is True
    assert cfg.router_source == "sigma"
    assert cfg.verbose is False


def test_exclude_include_patterns_literal_eval():
    """``exclude_patterns`` arrives as a python-literal string list from TOML."""
    cfg = LoRANetworkCfg.from_kwargs(
        {"exclude_patterns": "['foo.*', 'bar.*']", "include_patterns": "['baz.*']"},
        network_dim=4,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert "foo.*" in cfg.exclude_patterns
    assert "bar.*" in cfg.exclude_patterns
    # default exclude is always appended
    assert any("_modulation" in p for p in cfg.exclude_patterns)
    assert cfg.include_patterns == ["baz.*"]


def test_train_adaln_desugars_to_include_pattern():
    """``train_adaln`` is an exclude-override, not a whitelist: it appends the
    adaln include and leaves the default attn+MLP target set alone."""
    cfg = LoRANetworkCfg.from_kwargs(
        {"train_adaln": "true"},
        network_dim=4,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert ".*adaln_up_.*" in cfg.include_patterns
    # the default exclude still lists adaln_up_ — the include is what beats it
    assert any("adaln_up_" in p for p in cfg.exclude_patterns)
    # rank inherits the network's unless overridden; alpha is always pinned via
    # reg_alphas, but at adaln_rank=0 the √r factor is 1 → the network alpha
    assert not cfg.reg_dims
    assert cfg.reg_alphas == {".*adaln_up_.*": 1.0}


def test_adaln_alpha_derives_from_network_rank_alpha():
    """Unset ``adaln_alpha`` follows the network's rank/alpha by the √r law
    rather than inheriting network_alpha at the smaller adaln rank (which would
    run the adaln modules network_dim/adaln_rank hotter in alpha/rank)."""
    cfg = LoRANetworkCfg.from_kwargs(
        {"train_adaln": "true", "adaln_rank": "16"},
        network_dim=32,
        network_alpha=128.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.reg_alphas[".*adaln_up_.*"] == pytest.approx(128.0 * (0.5**0.5))
    # and it tracks network_alpha rather than a hard-coded constant
    cfg = LoRANetworkCfg.from_kwargs(
        {"train_adaln": "true", "adaln_rank": "16"},
        network_dim=32,
        network_alpha=32.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.reg_alphas[".*adaln_up_.*"] == pytest.approx(32.0 * (0.5**0.5))


def test_adaln_rank_and_alpha_ride_the_reg_overrides():
    """``adaln_rank`` / ``adaln_alpha`` desugar onto the same pattern via
    reg_dims / reg_alphas, merging with any user-supplied entries."""
    cfg = LoRANetworkCfg.from_kwargs(
        {
            "train_adaln": "true",
            "adaln_rank": "32",
            "adaln_alpha": "128",
            "network_reg_dims": "blocks\\.0.*=8",
        },
        network_dim=4,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.reg_dims == {"blocks\\.0.*": 8, ".*adaln_up_.*": 32}
    assert cfg.reg_alphas == {".*adaln_up_.*": 128.0}


def _moe_stamps(router_source: str = "sigma") -> dict:
    """Three-axis stamps mimicking ``LoRANetwork.save_weights`` for a Hydra
    checkpoint. plan2 task #6 retired the legacy ``ss_use_hydra`` /
    ``ss_use_fei_router`` fallback; ``from_weights`` now requires these.
    """
    return {
        "new_use_moe_style": "shared_A",
        "new_route_per_layer": True,
        "new_router_source": router_source,
    }


def test_from_weights_moe_without_stamps_raises():
    """plan2 task #6: a Hydra checkpoint missing the three-axis stamps must
    refuse to load (pre-plan2 artifacts stop loading by design)."""
    with pytest.raises(RuntimeError, match="three-axis routing stamps"):
        LoRANetworkCfg.from_weights(
            modules_dim={"foo": 4},
            modules_alpha={"foo": 1.0},
            module_class=HydraLoRAModule,
            train_llm_adapter=False,
            is_hydra=True,
            hydra_num_experts=4,
            sigma_feature_dim_detected=16,
            sigma_router_names=["foo"],
            hydra_router_names=None,
            channel_scales_dict=None,
        )


def test_from_weights_sigma_band_partition_round_trip():
    cfg = LoRANetworkCfg.from_weights(
        modules_dim={"foo": 4},
        modules_alpha={"foo": 1.0},
        module_class=HydraLoRAModule,
        train_llm_adapter=False,
        is_hydra=True,
        hydra_num_experts=6,
        sigma_feature_dim_detected=16,
        sigma_router_names=["foo"],
        hydra_router_names=None,
        channel_scales_dict=None,
        specialize_experts_by_sigma_buckets=True,
        num_sigma_buckets=3,
        sigma_bucket_boundaries=[0.0, 0.5, 0.8, 1.0],
        **_moe_stamps("sigma"),
    )
    assert cfg.specialize_experts_by_sigma_buckets is True
    assert cfg.num_sigma_buckets == 3
    assert cfg.sigma_bucket_boundaries == [0.0, 0.5, 0.8, 1.0]
