"""Turbo × REPA Phase-1 wiring tests (docs/proposal/turbo_repa.md).

Covers the load-bearing invariants:
  - ``repa.weight = 0`` default ⇒ the whole path is off: config gate false,
    dataset keeps the legacy tuple, collate keeps the legacy shape (the
    byte-identical guarantee is structural — no PE loading, no extra RNG
    draws, no extra forward ever runs).
  - TOML / CLI precedence for the ``[repa]`` keys.
  - Config validation: negative weight, bad every_n, block-swap conflict.
  - ``CachedDataset.load_repa_pe`` appends the PE sidecar as the LAST tuple
    element (None when missing) without touching existing consumers.
  - Collate all-or-nothing: any missing sidecar (or a cross-batch token-count
    mismatch) collapses the batch's PE element to None instead of crashing.
  - Per-step-expert head routing: nearest student-grid σ for the sampled τ.

The alignment-loss math itself is covered by tests/test_repa.py (the distill
loop calls the same ``library/training/repa.py`` helpers).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from library.datasets.cache import CachedDataset, make_cached_collate
from scripts.distill_turbo.config import build_argparser, resolve_config


# ── config resolution ─────────────────────────────────────────────────────────


def _resolve(cli: list[str] | None = None, toml: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, toml or {})


def test_config_defaults_off():
    c = _resolve()
    assert c.repa_weight == 0.0  # off ⇒ byte-identical DP-DMD
    assert c.repa_layer == 8
    assert c.repa_encoder == "pe_spatial"
    assert c.repa_every_n == 4
    assert c.repa_spatial_norm is True


def test_config_toml_and_cli_precedence():
    toml = {
        "repa": {
            "weight": 0.05,
            "layer": 10,
            "every_n": 2,
            "encoder": "pe",
            "spatial_norm": False,
        }
    }
    c = _resolve(toml=toml)
    assert c.repa_weight == 0.05
    assert c.repa_layer == 10
    assert c.repa_every_n == 2
    assert c.repa_encoder == "pe"
    assert c.repa_spatial_norm is False
    # CLI sentinel overrides win over TOML.
    c = _resolve(
        ["--repa_weight", "0.1", "--repa_layer", "6", "--repa_spatial_norm"],
        toml=toml,
    )
    assert c.repa_weight == 0.1
    assert c.repa_layer == 6
    assert c.repa_spatial_norm is True
    # REPA-DoG band-pass off by default (spatial_norm path).
    assert c.repa_target_dog is False
    assert c.repa_dog_sigma1_div == 16.0
    assert c.repa_dog_sigma2_div == 0.0
    assert c.repa_dog_norm_std == 0.0


def test_config_dog_toml_and_cli_precedence():
    toml = {
        "repa": {
            "weight": 0.05,
            "target_dog": True,
            "dog_sigma1_div": 32.0,
            "dog_sigma2_div": 8.0,
            "dog_norm_std": 1.0,
        }
    }
    c = _resolve(toml=toml)
    assert c.repa_target_dog is True
    assert c.repa_dog_sigma1_div == 32.0
    assert c.repa_dog_sigma2_div == 8.0
    assert c.repa_dog_norm_std == 1.0
    # CLI overrides win over TOML (BooleanOptionalAction + sentinel floats).
    c = _resolve(
        ["--no-repa_target_dog", "--repa_dog_sigma1_div", "24"],
        toml=toml,
    )
    assert c.repa_target_dog is False
    assert c.repa_dog_sigma1_div == 24.0


def test_config_validation():
    with pytest.raises(ValueError, match="repa.weight"):
        _resolve(toml={"repa": {"weight": -0.1}})
    with pytest.raises(ValueError, match="repa.every_n"):
        _resolve(toml={"repa": {"weight": 0.05, "every_n": 0}})
    # Feature tap is unsupported under block swap — fail at config time.
    with pytest.raises(ValueError, match="blocks_to_swap"):
        _resolve(["--blocks_to_swap", "2"], toml={"repa": {"weight": 0.05}})
    # Head switch × deferred ckpt-recompute corruption — fail at config time.
    with pytest.raises(ValueError, match="per_step_expert"):
        _resolve(
            ["--grad_ckpt", "--per_step_expert"],
            toml={"repa": {"weight": 0.05}},
        )
    # DoG low-pass σ1 divisor must be positive when the band-pass is on.
    with pytest.raises(ValueError, match="dog_sigma1_div"):
        _resolve(
            toml={"repa": {"weight": 0.05, "target_dog": True, "dog_sigma1_div": 0}},
        )


# ── dataset PE gate ───────────────────────────────────────────────────────────


def _write_pair(d, stem: str, *, with_pe: bool, n_tok: int = 17):
    """Fabricate one cached (latent, TE[, PE]) trio under ``d``."""
    np.savez(
        d / f"{stem}_0128x0128_anima.npz",
        latents_16x16=np.random.randn(16, 16, 16).astype(np.float32),
    )
    save_file(
        {"crossattn_emb_v0": torch.randn(8, 32)},
        str(d / f"{stem}_anima_te.safetensors"),
    )
    if with_pe:
        save_file(
            {"image_features": torch.randn(n_tok, 24)},
            str(d / f"{stem}_anima_pe_spatial.safetensors"),
        )


def test_dataset_legacy_tuple_unchanged(tmp_path):
    _write_pair(tmp_path, "a", with_pe=True)
    ds = CachedDataset(str(tmp_path), batch_size=1)
    assert ds.load_repa_pe is False  # off by default
    assert len(ds[0]) == 4  # (idx, latents, crossattn, pooled)


def test_dataset_appends_pe_when_enabled(tmp_path):
    _write_pair(tmp_path, "a", with_pe=True)
    _write_pair(tmp_path, "b", with_pe=False)
    ds = CachedDataset(str(tmp_path), batch_size=1)
    ds.load_repa_pe = True
    items = {ds[i][0]: ds[i] for i in range(2)}
    # Stems sort a < b; samples order follows discovery, so key by content.
    pes = [it[-1] for it in items.values()]
    with_pe = [p for p in pes if p is not None]
    assert len(items[0]) == 5
    assert len(with_pe) == 1  # 'a' has a sidecar, 'b' yields None
    assert with_pe[0].shape == (17, 24)
    assert with_pe[0].dtype == torch.float32


# ── collate ───────────────────────────────────────────────────────────────────


def _sample(pe):
    return (0, torch.zeros(16, 4, 4), torch.zeros(8, 32), torch.zeros(32), pe)


def test_collate_legacy_shape():
    legacy = [s[:4] for s in (_sample(None), _sample(None))]
    out = make_cached_collate()(legacy)
    assert len(out) == 4


def test_collate_stacks_pe():
    out = make_cached_collate(load_repa_pe=True)(
        [_sample(torch.randn(17, 24)), _sample(torch.randn(17, 24))]
    )
    assert len(out) == 5
    assert out[-1].shape == (2, 17, 24)


def test_collate_all_or_nothing():
    # Any missing sidecar → None for the whole batch.
    out = make_cached_collate(load_repa_pe=True)(
        [_sample(torch.randn(17, 24)), _sample(None)]
    )
    assert out[-1] is None
    # Token-count mismatch across the batch (different encoder aspect buckets)
    # → None rather than a stack crash.
    out = make_cached_collate(load_repa_pe=True)(
        [_sample(torch.randn(17, 24)), _sample(torch.randn(13, 24))]
    )
    assert out[-1] is None


def test_collate_mask_and_pe_order():
    s = (*_sample(None)[:4], torch.ones(1, 4, 4), torch.randn(17, 24))
    out = make_cached_collate(use_masked_loss=True, load_repa_pe=True)([s, s])
    assert len(out) == 6
    assert out[4].shape == (2, 1, 4, 4)  # mask
    assert out[5].shape == (2, 17, 24)  # PE rides last


# ── unsloth-ckpt grad flow ────────────────────────────────────────────────────


def test_unsloth_ckpt_propagates_param_grads_via_input():
    """The REPA forward is wrapped in selective_block_grad_ckpt (unsloth path).

    The reentrant unsloth checkpoint silently ZEROES grads that reach only
    closed-over params ([[project_unsloth_reentrant_drops_grad]] — it broke BYG);
    a grad-requiring tensor *input* resurrects them (the EasyControl precedent).
    The distill loop relies on this by marking x_tau.requires_grad_() before the
    checkpointed REPA forward — this test pins the mechanism so a checkpoint
    refactor can't silently turn the REPA term into a no-op.
    """
    import torch.nn as nn

    from library.anima.models import unsloth_checkpoint

    lin = nn.Linear(4, 4)  # closed-over param, the student-LoRA stand-in
    x = torch.randn(2, 4, requires_grad=True)  # the x_tau stand-in
    unsloth_checkpoint(lin, x).sum().backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.abs().sum() > 0


# ── per-step-expert head routing ─────────────────────────────────────────────


def test_nearest_student_step():
    from scripts.distill_turbo.distill import nearest_student_step

    grid = [1.0, 0.9, 0.75, 0.5, 0.0]  # 4-step student (N+1 σ points)
    assert nearest_student_step(grid, 1.0, 4) == 0
    assert nearest_student_step(grid, 0.8, 4) == 2  # |0.75−0.8| < |0.9−0.8|
    assert nearest_student_step(grid, 0.55, 4) == 3
    assert nearest_student_step(grid, 0.1, 4) == 3  # σ=0 is the endpoint, not a head
