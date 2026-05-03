"""Smoke tests for the per-LoRA DCW calibration path.

Covers the multi-anchor selector, safetensors round-trip of the
``ss_dcw_recipe`` metadata key, and the stacked-LoRA combination rule.
Does NOT exercise the trainer end-to-end — that needs a GPU and a real
val dataloader.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from library.inference.dcw_calibration import (
    DEFAULT_BAND_MASK,
    DEFAULT_LAMBDA_LL,
    DEFAULT_SCHEDULE_LL,
    combine_recipes,
    read_recipe_from_safetensors,
    resolve_dcw_args,
    select_best_anchor_LL,
)


def _sigmas(n: int) -> list[float]:
    return [(n - i) / n for i in range(n)]


def test_selector_picks_lowest_loss_anchor():
    # Constant baseline gap = 1.0; anchor λ=-0.015 cuts it to 0.5,
    # λ=-0.022 cuts it to 0.2 (best), λ=-0.030 overshoots to -0.4.
    n = 12
    sig = _sigmas(n)
    gap_b = [1.0] * n
    out = select_best_anchor_LL(
        gap_b,
        [(-0.015, [0.5] * n), (-0.022, [0.2] * n), (-0.030, [-0.4] * n)],
        sig,
        schedule="const",
    )
    assert math.isclose(out["lambda_LL"], -0.022)
    assert out["schedule_LL"] == "const"
    assert out["num_late_steps"] == 6
    assert not out["baseline_was_best"]
    # Candidates dict has baseline (λ=0) first, then anchors in order.
    assert [c["lambda"] for c in out["candidates"]] == [0.0, -0.015, -0.022, -0.030]


def test_selector_picks_baseline_when_all_anchors_worse():
    # Flat-style LoRA — DCW makes every anchor worse than no correction.
    n = 12
    sig = _sigmas(n)
    gap_b = [0.5] * n
    out = select_best_anchor_LL(
        gap_b,
        [(-0.015, [1.5] * n), (-0.022, [2.0] * n), (-0.030, [3.0] * n)],
        sig,
        schedule="const",
    )
    assert out["lambda_LL"] == 0.0
    assert out["baseline_was_best"]


def test_selector_one_minus_sigma_weights_late_steps():
    # Anchor A reduces gap on early-late steps but worsens it on the
    # very last step; anchor B does the opposite. Under one_minus_sigma
    # the very-last step gets the highest weight, so B should win.
    n = 12
    sig = _sigmas(n)
    gap_b = [2.0] * n
    gap_a = list(gap_b)
    gap_a[6:11] = [0.5] * 5  # A: huge improvement mid-late
    gap_a[11] = 4.0          # A: blows up at very last step
    gap_b_anchor = list(gap_b)
    gap_b_anchor[6:11] = [1.8] * 5  # B: marginal improvement mid-late
    gap_b_anchor[11] = 0.2          # B: nails the last step
    out = select_best_anchor_LL(
        gap_b,
        [(-0.015, gap_a), (-0.022, gap_b_anchor)],
        sig,
        schedule="one_minus_sigma",
    )
    assert math.isclose(out["lambda_LL"], -0.022)


def test_selector_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        select_best_anchor_LL(
            [1.0, 1.0], [(-0.015, [0.5])], [0.5, 0.4], schedule="const"
        )


def test_selector_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        select_best_anchor_LL([], [(-0.015, [])], [], schedule="const")


def test_selector_no_anchors_returns_baseline():
    n = 8
    sig = _sigmas(n)
    gap_b = [1.0] * n
    out = select_best_anchor_LL(gap_b, [], sig, schedule="const")
    assert out["lambda_LL"] == 0.0
    assert out["baseline_was_best"]
    assert len(out["candidates"]) == 1


def _write_lora_with_recipe(path: Path, recipe_json: str | None) -> None:
    tensors = {"lora_unet_dummy.lora_down.weight": torch.zeros(2, 2)}
    metadata = {"ss_network_dim": "8"}
    if recipe_json is not None:
        metadata["ss_dcw_recipe"] = recipe_json
    save_file(tensors, str(path), metadata=metadata)


def test_round_trip_recipe_metadata(tmp_path: Path):
    p = tmp_path / "lora.safetensors"
    _write_lora_with_recipe(
        p, '{"lambda_LL":-0.0123,"schedule_LL":"one_minus_sigma"}'
    )
    rec = read_recipe_from_safetensors(str(p))
    assert rec is not None
    assert math.isclose(rec["lambda_LL"], -0.0123)
    assert rec["schedule_LL"] == "one_minus_sigma"


def test_missing_recipe_returns_none(tmp_path: Path):
    p = tmp_path / "lora.safetensors"
    _write_lora_with_recipe(p, recipe_json=None)
    assert read_recipe_from_safetensors(str(p)) is None


def test_malformed_recipe_returns_none(tmp_path: Path):
    p = tmp_path / "lora.safetensors"
    _write_lora_with_recipe(p, "{not valid json")
    assert read_recipe_from_safetensors(str(p)) is None


def test_combine_recipes_single_passthrough():
    rec = {"lambda_LL": -0.02, "schedule_LL": "one_minus_sigma"}
    out = combine_recipes([rec], multipliers=1.0)
    assert math.isclose(out["lambda_LL"], -0.02)
    assert out["schedule_LL"] == "one_minus_sigma"
    assert out["_n_present"] == 1
    assert out["_n_total"] == 1


def test_combine_recipes_weighted_average():
    a = {"lambda_LL": -0.01, "schedule_LL": "one_minus_sigma"}
    b = {"lambda_LL": -0.03, "schedule_LL": "one_minus_sigma"}
    # 0.25·(-0.01) + 0.75·(-0.03) = -0.025
    out = combine_recipes([a, b], multipliers=[0.25, 0.75])
    assert math.isclose(out["lambda_LL"], -0.025, rel_tol=1e-9)
    assert out["_n_present"] == 2


def test_combine_recipes_skips_missing_recipes():
    a = {"lambda_LL": -0.02, "schedule_LL": "one_minus_sigma"}
    out = combine_recipes([a, None], multipliers=[0.5, 0.5])
    # LoRA without recipe contributes nothing — λ stays at a's value.
    assert math.isclose(out["lambda_LL"], -0.02)
    assert out["_n_present"] == 1
    assert out["_n_total"] == 2


def test_combine_recipes_modal_schedule():
    a = {"lambda_LL": -0.01, "schedule_LL": "one_minus_sigma"}
    b = {"lambda_LL": -0.01, "schedule_LL": "const"}
    c = {"lambda_LL": -0.01, "schedule_LL": "one_minus_sigma"}
    out = combine_recipes([a, b, c], multipliers=[1.0, 1.0, 1.0])
    assert out["schedule_LL"] == "one_minus_sigma"


def test_combine_recipes_all_none_returns_none():
    assert combine_recipes([None, None], multipliers=1.0) is None


def _make_args(**overrides):
    a = argparse.Namespace(
        dcw=False,
        dcw_lambda=None,
        dcw_schedule=None,
        dcw_band_mask=None,
        dcw_disable_per_lora_recipe=False,
        lora_weight=None,
        lora_multiplier=1.0,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def test_resolve_args_no_lora_uses_defaults():
    a = _make_args(dcw=True)
    resolve_dcw_args(a)
    assert math.isclose(a.dcw_lambda, DEFAULT_LAMBDA_LL)
    assert a.dcw_schedule == DEFAULT_SCHEDULE_LL
    assert a.dcw_band_mask == DEFAULT_BAND_MASK


def test_resolve_args_cli_lambda_wins(tmp_path: Path):
    p = tmp_path / "lora.safetensors"
    _write_lora_with_recipe(p, '{"lambda_LL":-0.099,"schedule_LL":"const"}')
    a = _make_args(
        dcw=True,
        dcw_lambda=-0.5,  # explicit CLI value
        lora_weight=[str(p)],
    )
    resolve_dcw_args(a)
    assert a.dcw_lambda == -0.5  # CLI wins, recipe ignored for lambda
    # Schedule + band_mask still resolved from recipe / default.
    assert a.dcw_schedule == "const"
    assert a.dcw_band_mask == DEFAULT_BAND_MASK


def test_resolve_args_recipe_loaded(tmp_path: Path):
    p = tmp_path / "lora.safetensors"
    _write_lora_with_recipe(p, '{"lambda_LL":-0.0085,"schedule_LL":"one_minus_sigma"}')
    a = _make_args(dcw=True, lora_weight=[str(p)])
    resolve_dcw_args(a)
    assert math.isclose(a.dcw_lambda, -0.0085)
    assert a.dcw_schedule == "one_minus_sigma"


def test_resolve_args_disable_flag_skips_recipe(tmp_path: Path):
    p = tmp_path / "lora.safetensors"
    _write_lora_with_recipe(p, '{"lambda_LL":-0.099,"schedule_LL":"const"}')
    a = _make_args(
        dcw=True,
        dcw_disable_per_lora_recipe=True,
        lora_weight=[str(p)],
    )
    resolve_dcw_args(a)
    assert math.isclose(a.dcw_lambda, DEFAULT_LAMBDA_LL)
    assert a.dcw_schedule == DEFAULT_SCHEDULE_LL


def test_resolve_args_dcw_off_skips_io(tmp_path: Path):
    # When --dcw is off, the resolver shouldn't fail even if lora_weight
    # points at a nonexistent path. It should still populate defaults so
    # downstream getattr() calls see floats, not None.
    a = _make_args(dcw=False, lora_weight=["/nonexistent/path.safetensors"])
    resolve_dcw_args(a)
    assert math.isclose(a.dcw_lambda, DEFAULT_LAMBDA_LL)
    assert a.dcw_schedule == DEFAULT_SCHEDULE_LL
    assert a.dcw_band_mask == DEFAULT_BAND_MASK
