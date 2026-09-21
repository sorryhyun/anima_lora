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


def _base_kwargs() -> dict:
    """Empty kwargs — all fields fall to their stringless defaults."""
    return {}


def test_defaults_when_all_kwargs_absent():
    cfg = LoRANetworkCfg.from_kwargs(
        _base_kwargs(),
        network_dim=None,
        network_alpha=None,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.lora_dim == 4
    assert cfg.alpha == 1.0
    assert cfg.module_class is LoRAModule
    assert cfg.train_llm_adapter is False
    assert cfg.use_timestep_mask is False
    assert cfg.use_moe_style is False
    assert cfg.route_per_layer is False
    assert cfg.router_source == "none"
    # exclude regex always appended
    assert any("_modulation" in p for p in cfg.exclude_patterns)
    assert cfg.include_patterns is None
    assert cfg.dropout is None
    assert cfg.rank_dropout is None
    assert cfg.module_dropout is None
    assert cfg.reg_dims is None
    assert cfg.reg_lrs is None


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


def test_crossattn_emb_router_source_parses():
    """``router_source="crossattn_emb"`` is a network-level cell — it requires
    ``route_per_layer=False`` and an MoE layout. The pooled cross-attention
    text feature routes one GlobalRouter for the whole pool.
    """
    cfg = LoRANetworkCfg.from_kwargs(
        {
            "use_moe_style": "shared_A",
            "route_per_layer": "false",
            "router_source": "crossattn_emb",
        },
        network_dim=8,
        network_alpha=4.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.use_moe_style == "shared_A"
    assert cfg.route_per_layer is False
    assert cfg.router_source == "crossattn_emb"


def test_crossattn_emb_router_source_rejects_per_layer():
    """There is no per-Linear crossattn signal — pairing the source with
    ``route_per_layer=True`` must raise rather than silently fall back.
    """
    with pytest.raises(ValueError, match="route_per_layer=False"):
        LoRANetworkCfg.from_kwargs(
            {
                "use_moe_style": "shared_A",
                "route_per_layer": "true",
                "router_source": "crossattn_emb",
            },
            network_dim=8,
            network_alpha=4.0,
            neuron_dropout=None,
            module_class=LoRAModule,
        )


def test_independent_a_moe_style_raises():
    """The independent-A (stacked experts) layout was removed — only ``False``
    and ``"shared_A"`` are legal, and a stale TOML must fail loudly."""
    with pytest.raises(ValueError, match="expected False or 'shared_A'"):
        LoRANetworkCfg.from_kwargs(
            {"use_moe_style": "independent_A", "route_per_layer": "false"},
            network_dim=8,
            network_alpha=4.0,
            neuron_dropout=None,
            module_class=LoRAModule,
        )


def test_legacy_router_kwargs_raise():
    """plan2 task #6 retired ``use_hydra`` / ``use_sigma_router`` /
    ``use_fei_router``. Surfacing them must raise so a stale TOML can't
    silently produce a no-MoE network.
    """
    for legacy_key in ("use_hydra", "use_sigma_router", "use_fei_router"):
        with pytest.raises(ValueError, match="Legacy router kwarg"):
            LoRANetworkCfg.from_kwargs(
                {legacy_key: True},
                network_dim=4,
                network_alpha=1.0,
                neuron_dropout=None,
                module_class=LoRAModule,
            )


def test_numeric_string_parsing():
    kwargs = {
        "min_rank": "2",
        "alpha_rank_scale": "0.75",
        "num_experts": "8",
        "network_router_lr_scale": "0.5",
        "sigma_feature_dim": "32",
        "per_bucket_balance_weight": "0.4",
        "num_sigma_buckets": "5",
        "rank_dropout": "0.05",
        "module_dropout": "0.1",
        "layer_start": "4",
        "layer_end": "28",
    }
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=32,
        network_alpha=16.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.min_rank == 2 and isinstance(cfg.min_rank, int)
    assert cfg.alpha_rank_scale == pytest.approx(0.75)
    assert cfg.num_experts == 8
    assert cfg.router_lr_scale == pytest.approx(0.5)
    assert cfg.sigma_feature_dim == 32
    assert cfg.per_bucket_balance_weight == pytest.approx(0.4)
    assert cfg.num_sigma_buckets == 5
    assert cfg.rank_dropout == pytest.approx(0.05)
    assert cfg.module_dropout == pytest.approx(0.1)
    assert cfg.layer_start == 4 and cfg.layer_end == 28


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


def test_reg_dims_and_reg_lrs_kv_pairs():
    cfg = LoRANetworkCfg.from_kwargs(
        {
            "network_reg_dims": "blocks\\.0.*=8, blocks\\.1.*=16",
            "network_reg_lrs": "blocks\\.0.*=1e-4, blocks\\.1.*=2e-4",
        },
        network_dim=4,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.reg_dims == {"blocks\\.0.*": 8, "blocks\\.1.*": 16}
    assert cfg.reg_lrs == {"blocks\\.0.*": 1e-4, "blocks\\.1.*": 2e-4}


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


def test_train_adaln_off_leaves_include_patterns_untouched():
    cfg = LoRANetworkCfg.from_kwargs(
        {"include_patterns": "['baz.*']"},
        network_dim=4,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.include_patterns == ["baz.*"]


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


@pytest.mark.parametrize("key, value", [("adaln_rank", "32"), ("adaln_alpha", "128")])
def test_adaln_rank_alpha_require_train_adaln(key, value):
    with pytest.raises(ValueError, match="train_adaln"):
        LoRANetworkCfg.from_kwargs(
            {key: value},
            network_dim=4,
            network_alpha=1.0,
            neuron_dropout=None,
            module_class=LoRAModule,
        )


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


def test_from_weights_warm_start_shape():
    cfg = LoRANetworkCfg.from_weights(
        modules_dim={"foo": 4, "bar": 8},
        modules_alpha={"foo": 1.0, "bar": 2.0},
        module_class=HydraLoRAModule,
        train_llm_adapter=True,
        is_hydra=True,
        hydra_num_experts=8,
        sigma_feature_dim_detected=16,
        sigma_router_names=["foo"],
        hydra_router_names=None,
        channel_scales_dict=None,
        **_moe_stamps("sigma"),
    )
    assert cfg.modules_dim == {"foo": 4, "bar": 8}
    assert cfg.modules_alpha == {"foo": 1.0, "bar": 2.0}
    assert cfg.module_class is HydraLoRAModule
    assert cfg.train_llm_adapter is True
    assert cfg.num_experts == 8
    assert cfg.use_moe_style == "shared_A"
    assert cfg.route_per_layer is True
    assert cfg.router_source == "sigma"
    assert cfg.sigma_feature_dim == 16
    assert cfg.sigma_router_names == ["foo"]


def test_from_weights_no_sigma():
    cfg = LoRANetworkCfg.from_weights(
        modules_dim={"foo": 4},
        modules_alpha={"foo": 1.0},
        module_class=LoRAModule,
        train_llm_adapter=False,
        is_hydra=False,
        hydra_num_experts=0,
        sigma_feature_dim_detected=None,
        sigma_router_names=None,
        hydra_router_names=None,
        channel_scales_dict=None,
    )
    assert cfg.num_experts == 4  # default fallback (not 0)
    assert cfg.use_moe_style is False
    assert cfg.router_source == "none"


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


def test_from_weights_sigma_band_partition_off_by_default():
    cfg = LoRANetworkCfg.from_weights(
        modules_dim={"foo": 4},
        modules_alpha={"foo": 1.0},
        module_class=HydraLoRAModule,
        train_llm_adapter=False,
        is_hydra=True,
        hydra_num_experts=12,
        sigma_feature_dim_detected=16,
        sigma_router_names=["foo"],
        hydra_router_names=None,
        channel_scales_dict=None,
        **_moe_stamps("sigma"),
    )
    assert cfg.specialize_experts_by_sigma_buckets is False


def test_from_weights_sigma_band_partition_round_trip():
    cfg = LoRANetworkCfg.from_weights(
        modules_dim={"foo": 4},
        modules_alpha={"foo": 1.0},
        module_class=HydraLoRAModule,
        train_llm_adapter=False,
        is_hydra=True,
        hydra_num_experts=12,
        sigma_feature_dim_detected=16,
        sigma_router_names=["foo"],
        hydra_router_names=None,
        channel_scales_dict=None,
        specialize_experts_by_sigma_buckets=True,
        num_sigma_buckets=4,
        **_moe_stamps("sigma"),
    )
    assert cfg.specialize_experts_by_sigma_buckets is True
    assert cfg.num_sigma_buckets == 4
    assert cfg.sigma_bucket_boundaries is None


def test_from_weights_sigma_band_partition_with_custom_boundaries():
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
    assert cfg.sigma_bucket_boundaries == [0.0, 0.5, 0.8, 1.0]


def test_sigma_bucket_boundaries_parsed_and_validated():
    """Custom σ-bucket boundary list parses from native list and from a
    python-literal string (CLI/TOML pathways), and rejects malformed input.
    """
    # native list (TOML path)
    cfg = LoRANetworkCfg.from_kwargs(
        {
            "num_experts": "6",
            "specialize_experts_by_sigma_buckets": "true",
            "num_sigma_buckets": "3",
            "sigma_bucket_boundaries": [0.0, 0.5, 0.8, 1.0],
        },
        network_dim=8,
        network_alpha=8.0,
        neuron_dropout=None,
        module_class=HydraLoRAModule,
    )
    assert cfg.sigma_bucket_boundaries == [0.0, 0.5, 0.8, 1.0]

    # stringified literal (CLI/legacy path)
    cfg = LoRANetworkCfg.from_kwargs(
        {
            "num_experts": "6",
            "specialize_experts_by_sigma_buckets": "true",
            "num_sigma_buckets": "3",
            "sigma_bucket_boundaries": "[0.0, 0.5, 0.8, 1.0]",
        },
        network_dim=8,
        network_alpha=8.0,
        neuron_dropout=None,
        module_class=HydraLoRAModule,
    )
    assert cfg.sigma_bucket_boundaries == [0.0, 0.5, 0.8, 1.0]

    # wrong length
    with pytest.raises(ValueError, match="length"):
        LoRANetworkCfg.from_kwargs(
            {
                "num_experts": "6",
                "specialize_experts_by_sigma_buckets": "true",
                "num_sigma_buckets": "3",
                "sigma_bucket_boundaries": [0.0, 0.5, 1.0],
            },
            network_dim=8,
            network_alpha=8.0,
            neuron_dropout=None,
            module_class=HydraLoRAModule,
        )

    # not strictly increasing
    with pytest.raises(ValueError, match="increasing"):
        LoRANetworkCfg.from_kwargs(
            {
                "num_experts": "6",
                "specialize_experts_by_sigma_buckets": "true",
                "num_sigma_buckets": "3",
                "sigma_bucket_boundaries": [0.0, 0.5, 0.5, 1.0],
            },
            network_dim=8,
            network_alpha=8.0,
            neuron_dropout=None,
            module_class=HydraLoRAModule,
        )

    # wrong start / end
    with pytest.raises(ValueError, match="0.0"):
        LoRANetworkCfg.from_kwargs(
            {
                "num_experts": "6",
                "specialize_experts_by_sigma_buckets": "true",
                "num_sigma_buckets": "3",
                "sigma_bucket_boundaries": [0.1, 0.5, 0.8, 1.0],
            },
            network_dim=8,
            network_alpha=8.0,
            neuron_dropout=None,
            module_class=HydraLoRAModule,
        )


def test_cfg_is_frozen():
    cfg = LoRANetworkCfg.from_kwargs(
        {},
        network_dim=4,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
        cfg.lora_dim = 999  # type: ignore[misc]
