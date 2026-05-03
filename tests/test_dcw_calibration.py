"""Smoke tests for the per-LoRA DCW calibration path.

Covers the closed-form solver math, safetensors round-trip of the
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
    solve_recipe_LL,
)


def _const_gap_inputs(num_steps: int = 12, gap_value: float = 1.0):
    """Build a synthetic baseline+probe pair with constant gap and
    perfect linear-response (s = 1 across all steps).

    With s=1, schedule weight w, and constant gap g:
        λ* = -Σ_late w·g·1 / Σ_late w·1²  =  -g
    """
    sigmas = [(num_steps - i) / num_steps for i in range(num_steps)]
    eps = 0.01
    gap_baseline = [gap_value] * num_steps
    # gap_probe = baseline + s * (-eps); s = 1 → gap_probe = baseline - eps
    gap_probe = [g - eps for g in gap_baseline]
    return gap_baseline, gap_probe, sigmas, eps


def test_solver_constant_gap_recovers_minus_gap():
    g, gp, sig, eps = _const_gap_inputs(num_steps=12, gap_value=2.5)
    out = solve_recipe_LL(g, gp, sig, eps=eps, schedule="const")
    # const schedule + s=1 + constant gap → λ* = -gap exactly.
    assert math.isclose(out["lambda_LL"], -2.5, rel_tol=1e-9, abs_tol=1e-9)
    assert out["schedule_LL"] == "const"
    assert out["num_late_steps"] == 6
    # Linear-response prediction: gap + λ*s = 0 everywhere → residual ≈ 0.
    assert out["residual_gap_LL_late"] < 1e-12
    assert out["baseline_gap_LL_late"] > 0.0


def test_solver_one_minus_sigma_uses_late_weighting():
    # Increasing baseline gap toward late steps; (1-σ) schedule weights
    # the late half more, so λ* should land closer to a late-half mean
    # than to a uniform mean.
    num_steps = 12
    gap_baseline = [float(i) for i in range(num_steps)]  # 0,1,...,11
    eps = 0.01
    gap_probe = [g - eps for g in gap_baseline]  # s = 1 everywhere
    sigmas = [(num_steps - i) / num_steps for i in range(num_steps)]

    out = solve_recipe_LL(
        gap_baseline, gap_probe, sigmas, eps=eps, schedule="one_minus_sigma"
    )
    # With s=1 everywhere, λ* = -Σ_late w·g / Σ_late w
    late = num_steps // 2
    num = sum((1.0 - sigmas[i]) * gap_baseline[i] for i in range(late, num_steps))
    den = sum((1.0 - sigmas[i]) for i in range(late, num_steps))
    expected = -num / den
    assert math.isclose(out["lambda_LL"], expected, rel_tol=1e-9, abs_tol=1e-9)


def test_solver_rejects_zero_eps():
    with pytest.raises(ValueError, match="eps must be > 0"):
        solve_recipe_LL([0.0], [0.0], [0.5], eps=0.0)


def test_solver_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        solve_recipe_LL([0.0, 1.0], [0.0], [0.5, 0.4], eps=0.01)


def test_solver_rejects_zero_response():
    # Probe identical to baseline → s=0 everywhere → division by zero.
    n = 8
    g = [1.0] * n
    sig = [(n - i) / n for i in range(n)]
    with pytest.raises(ValueError, match="zero"):
        solve_recipe_LL(g, list(g), sig, eps=0.01)


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
