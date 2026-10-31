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

from library.config import schema as config_schema
from library.config.io import (
    _flatten_toml,
    _render_merged_toml,
    load_method_preset,
    read_config_from_file,
)
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
        "attn_mode",
        "base_config",  # manual extra
        "use_moe_style",  # network-module allowlist (three-axis routing)
    ):
        assert k in schema, f"expected {k!r} in populated schema"


@pytest.mark.parametrize(
    "cli, steps, epochs",
    [
        (["--max_train_steps", "12"], 12, None),
        (["--max_train_steps", "1600"], 1600, None),  # the argparse default, typed
        (["--max_train_epochs", "2"], 1600, 2),
        (["--max_train_steps", "12", "--max_train_epochs", "2"], 12, 2),
        ([], 1600, 8),
    ],
)
def test_cli_steps_beat_config_epochs(populated_parser, cli, steps, epochs):
    argv = ["--method", "lora", "--preset", "default", *cli]
    args = read_config_from_file(
        populated_parser.parse_args(argv), populated_parser, argv
    )
    assert (args.max_train_steps, args.max_train_epochs) == (steps, epochs)


def test_choices_preserved(populated_parser):
    lw = config_schema.get_schema()["log_with"]
    assert "tensorboard" in lw.choices
    assert "wandb" in lw.choices


# ---------------------------------------------------------------------------
# Typo / choice detection
# ---------------------------------------------------------------------------


def test_unknown_key_warns(populated_parser, tmp_path: Path, caplog):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with caplog.at_level(logging.WARNING):
        out = _flatten_toml({"a": {"network_ditm": 16}}, source=str(bogus))
    assert out == {"network_ditm": 16}
    assert any(
        "unknown key 'network_ditm'" in rec.getMessage() for rec in caplog.records
    )
    # line locator should include the line number
    assert any(":1:" in rec.getMessage() for rec in caplog.records)


def test_unknown_key_strict_raises(populated_parser, tmp_path: Path):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with pytest.raises(config_schema.ConfigSchemaError):
        _flatten_toml({"a": {"network_ditm": 16}}, source=str(bogus), strict=True)


def test_off_list_choice_warns(populated_parser, caplog):
    with caplog.at_level(logging.WARNING):
        _flatten_toml({"a": {"log_with": "carrierpigeon"}}, source="x.toml")
    assert any(
        "log_with" in rec.getMessage() and "not in choices" in rec.getMessage()
        for rec in caplog.records
    )


def test_int_to_float_coerced(populated_parser):
    # schema says network_alpha is float; TOML ``1`` comes in as int.
    out = _flatten_toml({"a": {"network_alpha": 64}}, source="x.toml")
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
            load_method_preset(method, preset)
        offenders = [
            rec.getMessage()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
            and rec.name.startswith("library.train_util")
        ]
        assert not offenders, f"{method} × {preset} warnings: {offenders}"


# ---------------------------------------------------------------------------
# Provenance + render
# ---------------------------------------------------------------------------


def test_provenance_returned():
    merged, provenance = load_method_preset("lora", "default", return_provenance=True)
    # base key
    assert provenance["network_module"] == "configs/base.toml"
    # method key
    assert provenance["network_dim"] == "configs/methods/lora.toml"
    assert set(provenance) == set(merged)


def test_train_adaln_defaults_on_and_reaches_the_network():
    """``train_adaln`` is a base.toml default (2026-07-16) that must survive the
    merge AND be forwarded to the network. It rides the top-level allowlist, not
    ``network_args``, so a regression in either half silently un-trains adaln.
    See docs/anima/methods/adaln.md.
    """
    from networks import all_network_kwargs

    merged, provenance = load_method_preset("lora", "default", return_provenance=True)
    assert merged["train_adaln"] is True
    assert provenance["train_adaln"] == "configs/base.toml"
    # cli.resolve_network_kwargs (library/anima/trainer) only forwards allowlisted top-level keys.
    assert {"train_adaln", "adaln_rank", "adaln_alpha"} <= set(all_network_kwargs())


def test_preset_gui_metadata_stripped():
    """``[<preset>.gui]`` display metadata (label/group for the GUI Hardware
    picker) must never reach the flat argparse merge."""
    merged = load_method_preset("lora", "low_vram")
    assert "gui" not in merged
    for key in ("label", "group", "description", "order"):
        assert key not in merged
    # ...while the preset's real knobs still land.
    assert merged["gradient_checkpointing"] is True
    assert merged["unsloth_offload_checkpointing"] is True


def test_hardware_preset_composes_with_gui_variant():
    """The GUI's Hardware picker composes presets.toml[low_vram] with a clean
    gui-methods variant (replaces the retired per-variant ``-8gb`` copies).
    Regression guard: a variant file pinning the hardware keys would silently
    defeat the preset (method wins over preset in the merge)."""
    import toml as _toml

    gui_dir = Path(__file__).resolve().parent.parent / "configs" / "gui-methods"
    hw_keys = ("gradient_checkpointing", "unsloth_offload_checkpointing")
    for path in sorted(gui_dir.glob("*.toml")):
        data = _toml.loads(path.read_text(encoding="utf-8"))
        for key in hw_keys:
            assert key not in data, (
                f"{path.name} pins {key}, which would defeat the Hardware "
                "preset picker (method wins over preset in the merge)"
            )
    merged = load_method_preset("lora", "low_vram", methods_subdir="gui-methods")
    assert merged["gradient_checkpointing"] is True
    assert merged["unsloth_offload_checkpointing"] is True


def _reparse_without_comments(text: str) -> dict:
    # toml.loads ignores comments natively, but our output has `# --- from ... ---`
    # headers that are valid TOML comments, so it round-trips directly.
    return toml.loads(text)


def test_render_roundtrips_to_valid_toml(populated_parser):
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = load_method_preset("lora", "default", return_provenance=True)
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(["--method", "lora", "--preset", "default"], namespace=ns)

    rendered = _render_merged_toml(args, parser, provenance)
    parsed = _reparse_without_comments(rendered)

    schema = config_schema.get_schema()
    for key in parsed:
        assert key in schema, f"rendered key {key!r} not in schema"


def test_render_header_includes_method_and_preset(populated_parser):
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = load_method_preset("lora", "low_vram", return_provenance=True)
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(["--method", "lora", "--preset", "low_vram"], namespace=ns)
    rendered = _render_merged_toml(args, parser, provenance)
    assert "Method: lora" in rendered
    assert "Preset: low_vram" in rendered
    # section ordering: base → preset → method
    base_idx = rendered.index("configs/base.toml")
    preset_idx = rendered.index("configs/presets.toml[low_vram]")
    method_idx = rendered.index("configs/methods/lora.toml")
    assert base_idx < preset_idx < method_idx
