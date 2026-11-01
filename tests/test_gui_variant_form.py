"""gui.core.variant_form: Save writeback."""

from __future__ import annotations

import pytest

from gui.core import variant_form as vf


def _reader(values: dict):
    return lambda key, _baseline: values[key]


def _save(orig, values, *, base=None, preset=None, extras=None):
    return vf.variant_from_form(
        orig,
        values,
        _reader(values),
        base=base or {},
        preset_overlay=preset or {},
        extras=extras,
    )


@pytest.mark.core
def test_preset_value_is_not_baked_in():
    out = _save(
        {},
        {"blocks_to_swap": 8},
        base={"blocks_to_swap": 0},
        preset={"blocks_to_swap": 8},
    )
    assert "blocks_to_swap" not in out
    out = _save(
        {},
        {"blocks_to_swap": 4},
        base={"blocks_to_swap": 0},
        preset={"blocks_to_swap": 8},
    )
    assert out == {"blocks_to_swap": 4}


def test_key_already_in_file_is_kept_even_at_baseline():
    out = _save({"learning_rate": 1e-4, "keep": 1}, {"learning_rate": 1e-4})
    assert out == {"learning_rate": 1e-4, "keep": 1}


def test_path_scope_goes_to_variant_table():
    out = _save({"variant": {"family": "lora"}}, {"path_scope": " run1 "})
    assert out == {"variant": {"family": "lora", "path_scope": "run1"}}
    out = _save(
        {"variant": {"path_scope": "run1"}, "path_scope": "stale"}, {"path_scope": ""}
    )
    assert out == {}


def test_virtual_keys_and_extras():
    out = _save(
        {"learning_rate": 1e-4},
        {"use_valid": False, "validation_split_num": 0, "repeat_by_folder_name": True},
        base={"datasets": [{"validation_split_num": 0}]},
        extras={"learning_rate": 2e-4},
    )
    assert out["learning_rate"] == 2e-4
    assert out["datasets"] == [
        {
            "validation_split_num": 0,
            "validation_split": 0.0,
            "repeat_by_folder_name": True,
        }
    ]
    assert "use_valid" not in out and "repeat_by_folder_name" not in out
