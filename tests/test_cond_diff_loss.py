"""Cond-diff loss weighting (``--cond_diff_loss``) invariants.

The weight map reallocates per-pixel FM gradient toward the cond↔target
edit region for paired ``cond_cache_dir`` tasks (sanitize / near-twins).
Invariants under test:

  - off by default → bit-identical loss.
  - per-image mean of the weight map ≈ 1 (pure reallocation, no LR shift).
  - edit region gets strictly more weight than the unchanged background,
    with the background:edit ratio bounded below by the floor.
  - cond == target degenerates to a uniform all-ones map.
  - 5D ``(B, C, 1, H, W)`` loss broadcasts via the dim-2 singleton.
"""

from __future__ import annotations

import argparse

import torch

from library.training.losses import (
    LossContext,
    _flow_match_loss,
    apply_cond_diff_loss,
    compute_cond_diff_weight,
)


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        loss_type="l2",
        masked_loss=False,
        cond_diff_loss=False,
        cond_diff_loss_floor=0.2,
        cond_diff_loss_blur=1.5,
        cond_diff_loss_quantile=0.9,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _paired_latents(b=2, c=16, h=24, w=32, blob=((6, 14), (8, 20))):
    """Target + cond identical except a strong rectangular blob."""
    g = torch.Generator().manual_seed(0)
    target = torch.randn(b, c, h, w, generator=g)
    cond = target.clone()
    (y0, y1), (x0, x1) = blob
    cond[:, :, y0:y1, x0:x1] += 3.0
    return target, cond, blob


def _ctx(args, batch, pred, target_v):
    return LossContext(
        args=args,
        batch=batch,
        model_pred=pred,
        target=target_v,
        timesteps=torch.rand(pred.shape[0]),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(pred.shape[0]),
        network=object(),
    )


def test_weight_map_mean_one_and_localized():
    target, cond, ((y0, y1), (x0, x1)) = _paired_latents()
    w = compute_cond_diff_weight(target, cond, floor=0.2, blur_sigma=1.5)
    assert w.shape == (2, 1, 24, 32)
    means = w.flatten(1).mean(1)
    assert torch.allclose(means, torch.ones_like(means), atol=1e-5)
    inside = w[:, :, y0:y1, x0:x1].mean()
    bg = w[:, :, : y0 - 3, : x0 - 3].mean()  # margin clear of the blur halo
    assert inside > 2.0 * bg
    # post-normalization the bg:edit ratio is still bounded by floor:1
    assert bg / inside >= 0.2 * 0.95


def test_zero_diff_degrades_to_uniform():
    target, _, _ = _paired_latents()
    w = compute_cond_diff_weight(target, target.clone())
    assert torch.allclose(w, torch.ones_like(w), atol=1e-4)


def test_5d_loss_broadcasts_on_dim2():
    target, cond, _ = _paired_latents()
    loss5d = torch.rand(2, 16, 1, 24, 32)
    ctx = _ctx(
        _make_args(cond_diff_loss=True),
        {"latents": target, "cond_latents": cond},
        loss5d,
        loss5d,
    )
    out = apply_cond_diff_loss(loss5d, ctx)
    assert out.shape == loss5d.shape


def test_flow_match_loss_end_to_end_gating():
    target, cond, _ = _paired_latents()
    pred = torch.randn(2, 16, 24, 32)
    target_v = torch.randn(2, 16, 24, 32)
    batch = {"latents": target, "cond_latents": cond}

    off = _flow_match_loss(_ctx(_make_args(), batch, pred, target_v))
    on = _flow_match_loss(_ctx(_make_args(cond_diff_loss=True), batch, pred, target_v))
    assert off.shape == (2,) and on.shape == (2,)
    assert torch.isfinite(on).all()
    assert not torch.allclose(off, on)  # weighting actually fired

    # off-path matches the plain masked-less FM reduction exactly
    manual = ((pred.float() - target_v.float()) ** 2).mean(dim=(1, 2, 3))
    assert torch.allclose(off, manual, atol=1e-6)
