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
    assert cfg.add_reft is False
    assert cfg.use_timestep_mask is False
    assert cfg.use_sigma_router is False
    # exclude regex always appended
    assert any("_modulation" in p for p in cfg.exclude_patterns)
    assert cfg.include_patterns is None
    assert cfg.dropout is None
    assert cfg.rank_dropout is None
    assert cfg.module_dropout is None
    assert cfg.reg_dims is None
    assert cfg.reg_lrs is None


def test_string_bool_parsing_matches_old_factory_path():
    """Every bool kwarg used to come in as a literal "true"/"false" string
    from train.py's net_kwargs. Make sure the canonical 'true' parses true,
    arbitrary other strings parse false, and bool/None still work.
    """
    kwargs = {
        "train_llm_adapter": "true",
        "add_reft": "True",  # case-insensitive
        "use_timestep_mask": "TRUE",
        "use_sigma_router": True,  # already a bool
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
    assert cfg.add_reft is True
    assert cfg.use_timestep_mask is True
    assert cfg.use_sigma_router is True
    assert cfg.verbose is False


def test_numeric_string_parsing():
    kwargs = {
        "min_rank": "2",
        "alpha_rank_scale": "0.75",
        "reft_dim": "16",
        "reft_alpha": "8.0",
        "num_experts": "8",
        "expert_warmup_ratio": "0.2",
        "expert_warmup_k": "3",
        "expert_best_warmup_ratio": "0.0",
        "network_router_lr_scale": "0.5",
        "sigma_feature_dim": "32",
        "sigma_hidden_dim": "256",
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
    assert cfg.reft_dim == 16
    assert cfg.reft_alpha == pytest.approx(8.0)
    assert cfg.num_experts == 8
    assert cfg.expert_warmup_ratio == pytest.approx(0.2)
    assert cfg.expert_warmup_k == 3
    assert cfg.router_lr_scale == pytest.approx(0.5)
    assert cfg.sigma_feature_dim == 32
    assert cfg.sigma_hidden_dim == 256
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


def test_reft_dim_falls_back_to_network_dim():
    """Old factory behavior: ``reft_dim`` defaults to ``network_dim`` when
    not specified, not to the dataclass default of 4."""
    cfg = LoRANetworkCfg.from_kwargs(
        {},  # no reft_dim
        network_dim=64,
        network_alpha=32.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.reft_dim == 64


def test_from_weights_warm_start_shape():
    cfg = LoRANetworkCfg.from_weights(
        modules_dim={"foo": 4, "bar": 8},
        modules_alpha={"foo": 1.0, "bar": 2.0},
        module_class=HydraLoRAModule,
        train_llm_adapter=True,
        has_reft=True,
        reft_dim=8,
        reft_block_indices=[0, 1, 2],
        is_hydra_or_ortho_hydra=True,
        hydra_num_experts=8,
        sigma_feature_dim_detected=16,
        sigma_router_names=["foo"],
        hydra_router_names=None,
        channel_scales_dict=None,
    )
    assert cfg.modules_dim == {"foo": 4, "bar": 8}
    assert cfg.modules_alpha == {"foo": 1.0, "bar": 2.0}
    assert cfg.module_class is HydraLoRAModule
    assert cfg.train_llm_adapter is True
    assert cfg.add_reft is True
    assert cfg.reft_dim == 8
    assert cfg.reft_layers == [0, 1, 2]
    assert cfg.num_experts == 8
    assert cfg.use_sigma_router is True
    assert cfg.sigma_feature_dim == 16
    assert cfg.sigma_router_names == ["foo"]
    # Training-time schedules off in warm-start
    assert cfg.expert_warmup_ratio == 0.0
    assert cfg.expert_warmup_k == 1
    assert cfg.expert_best_warmup_ratio == 0.0


def test_from_weights_no_reft_no_sigma():
    cfg = LoRANetworkCfg.from_weights(
        modules_dim={"foo": 4},
        modules_alpha={"foo": 1.0},
        module_class=LoRAModule,
        train_llm_adapter=False,
        has_reft=False,
        reft_dim=None,
        reft_block_indices=set(),
        is_hydra_or_ortho_hydra=False,
        hydra_num_experts=0,
        sigma_feature_dim_detected=None,
        sigma_router_names=None,
        hydra_router_names=None,
        channel_scales_dict=None,
    )
    assert cfg.add_reft is False
    assert cfg.reft_dim == 4  # default fallback
    assert cfg.reft_layers == "all"
    assert cfg.num_experts == 4  # default fallback (not 0)
    assert cfg.use_sigma_router is False


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
