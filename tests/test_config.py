"""Tests for the M3 config schema: validation, provenance, print-config.

Covers:

* schema population (known keys present, aliases resolved)
* typo detection (unknown key → warning with file:line; strict → raises)
* off-list ``choices`` rejection
* soft type coercion (TOML ``1`` → ``float`` when schema says float)
* every ``methods × presets`` combination round-trips without warnings
* ``_render_merged_toml`` output re-parses as valid TOML whose keys are
  a subset of the populated schema
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest
import toml

from library import train_util
from library.config import schema as config_schema
from tests.conftest import iter_method_names


# ---------------------------------------------------------------------------
# Schema population
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_parser():
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())
    return parser


def test_schema_has_known_keys(populated_parser):
    schema = config_schema.get_schema()
    # a handful of must-have keys that come from different argparse layers
    for k in (
        "network_dim",
        "network_alpha",
        "optimizer_type",
        "learning_rate",
        "max_train_epochs",
        "mixed_precision",
        "attn_mode",
        "base_config",  # manual extra
        "use_hydra",  # network-module allowlist
    ):
        assert k in schema, f"expected {k!r} in populated schema"


def test_choices_preserved(populated_parser):
    mp = config_schema.get_schema()["mixed_precision"]
    assert "bf16" in mp.choices
    assert "no" in mp.choices


# ---------------------------------------------------------------------------
# Typo / choice detection
# ---------------------------------------------------------------------------


def test_unknown_key_warns(populated_parser, tmp_path: Path, caplog):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with caplog.at_level(logging.WARNING):
        out = train_util._flatten_toml({"a": {"network_ditm": 16}}, source=str(bogus))
    assert out == {"network_ditm": 16}
    assert any("unknown key 'network_ditm'" in rec.getMessage() for rec in caplog.records)
    # line locator should include the line number
    assert any(":1:" in rec.getMessage() for rec in caplog.records)


def test_unknown_key_strict_raises(populated_parser, tmp_path: Path):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with pytest.raises(config_schema.ConfigSchemaError):
        train_util._flatten_toml(
            {"a": {"network_ditm": 16}}, source=str(bogus), strict=True
        )


def test_off_list_choice_warns(populated_parser, caplog):
    with caplog.at_level(logging.WARNING):
        train_util._flatten_toml({"a": {"mixed_precision": "fp4"}}, source="x.toml")
    assert any(
        "mixed_precision" in rec.getMessage() and "not in choices" in rec.getMessage()
        for rec in caplog.records
    )


def test_int_to_float_coerced(populated_parser):
    # schema says network_alpha is float; TOML ``1`` comes in as int.
    out = train_util._flatten_toml({"a": {"network_alpha": 64}}, source="x.toml")
    assert isinstance(out["network_alpha"], float)
    assert out["network_alpha"] == 64.0


# ---------------------------------------------------------------------------
# Round-trip: all methods × presets produce no warnings
# ---------------------------------------------------------------------------


METHODS = list(iter_method_names())


def _load_preset_names() -> list[str]:
    return list(toml.load("configs/presets.toml").keys())


@pytest.mark.parametrize("method", METHODS)
def test_method_configs_clean(populated_parser, method: str, caplog):
    presets = _load_preset_names()
    for preset in presets:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            train_util.load_method_preset(method, preset)
        offenders = [
            rec.getMessage()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING and rec.name.startswith("library.train_util")
        ]
        assert not offenders, f"{method} × {preset} warnings: {offenders}"


# ---------------------------------------------------------------------------
# Provenance + render
# ---------------------------------------------------------------------------


def test_provenance_returned():
    merged, provenance = train_util.load_method_preset(
        "lora", "default", return_provenance=True
    )
    # base key
    assert provenance["network_module"] == "configs/base.toml"
    # preset key
    assert provenance["blocks_to_swap"] == "configs/presets.toml[default]"
    # method key
    assert provenance["network_dim"] == "configs/methods/lora.toml"
    assert set(provenance) == set(merged)


def _reparse_without_comments(text: str) -> dict:
    # toml.loads ignores comments natively, but our output has `# --- from ... ---`
    # headers that are valid TOML comments, so it round-trips directly.
    return toml.loads(text)


def test_render_roundtrips_to_valid_toml(populated_parser):
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = train_util.load_method_preset(
        "lora", "default", return_provenance=True
    )
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(["--method", "lora", "--preset", "default"], namespace=ns)

    rendered = train_util._render_merged_toml(args, parser, provenance)
    parsed = _reparse_without_comments(rendered)

    schema = config_schema.get_schema()
    for key in parsed:
        assert key in schema, f"rendered key {key!r} not in schema"


def test_render_header_includes_method_and_preset(populated_parser):
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = train_util.load_method_preset(
        "hydralora", "low_vram", return_provenance=True
    )
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(
        ["--method", "hydralora", "--preset", "low_vram"], namespace=ns
    )
    rendered = train_util._render_merged_toml(args, parser, provenance)
    assert "Method: hydralora" in rendered
    assert "Preset: low_vram" in rendered
    # section ordering: base → preset → method
    base_idx = rendered.index("configs/base.toml")
    preset_idx = rendered.index("configs/presets.toml[low_vram]")
    method_idx = rendered.index("configs/methods/hydralora.toml")
    assert base_idx < preset_idx < method_idx
