"""Invariant tests for the online memorization Δ-gap tracker.

CONTRIBUTING Tier-1.5 companion to ``library/training/mem_reweight.py``
(_archive/proposals/memorization_lowsigma_reweight.md). Pure-python tracker
numerics — no GPU, no DiT.
"""

import argparse
import random

import torch

from library.training.mem_reweight import (
    MemGapTracker,
    measure_grid_delta,
)


def _args(**over):
    base = dict(
        mem_reweight_mode="reweight",
        mem_sigma_max=0.7,
        mem_measure_every=1,
        mem_ema_beta=0.3,
        mem_z_threshold=1.5,
        mem_weight_temp=0.5,
        mem_weight_floor=0.0,
        mem_warmup_updates=10,
        mem_delta=1e-3,
        mem_snapshot_every=0,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _feed(tracker, rounds=20, n_items=20, hot="k0", hot_delta=0.3, seed=0):
    rng = random.Random(seed)
    for _ in range(rounds):
        keys = [f"/d/k{i}.png" for i in range(n_items)]
        deltas = [
            hot_delta if f"k{i}" == hot else rng.gauss(0.0, 0.02)
            for i in range(n_items)
        ]
        tracker.update(keys, deltas)


def test_flags_the_memorized_item(tmp_path):
    t = MemGapTracker(_args(), tmp_path / "m.json")
    _feed(t)
    assert t.flagged_stems() == ["k0"]
    # weight suppressed on low-σ draws only; unmeasured keys untouched
    w = t.weights(
        ["/d/k0.png", "/d/k1.png", "/d/k0.png", "/x/new.png"], [True, True, False, True]
    )
    assert w[0] < 0.05  # flagged + low σ → suppressed
    assert w[1] > 0.9  # unflagged
    assert w[2] == 1.0  # flagged but high σ → untouched (Arm A contract)
    assert w[3] == 1.0  # never measured


def test_measure_mode_never_reweights(tmp_path):
    t = MemGapTracker(_args(mem_reweight_mode="measure"), tmp_path / "m.json")
    _feed(t)
    assert t.flagged_stems() == ["k0"]  # detection still works
    assert t.weights(["/d/k0.png"], [True]) == [1.0]


def test_nonfinite_deltas_dropped(tmp_path):
    t = MemGapTracker(_args(), tmp_path / "m.json")
    t.update(["/d/a.png", "/d/b.png"], [float("nan"), float("inf")])
    assert t._count == {}


class _Net:
    """Stub adapter — only the multiplier toggle the measurement touches."""

    def __init__(self):
        self.multiplier = 1.0

    def set_multiplier(self, m):
        self.multiplier = float(m)


def _grid_call(net, z0):
    """Stub DiT: perfect velocity when the adapter is live, zeros when zeroed.

    Recovers ε from (z_t, t) via ε = (z_t − (1−σ)·z0) / σ — exactly inverting
    the rectified-flow re-noise the measurement applies.
    """

    def call(z_t, t, emb, padding_mask=None, **kw):
        s = float(t[0])
        eps = (z_t.squeeze(2).float() - (1.0 - s) * z0) / s
        v = eps - z0
        if net.multiplier == 0.0:
            v = torch.zeros_like(v)
        return v.unsqueeze(2)

    return call


def _run_grid(z0, net, sigmas=(0.5, 0.7), n_noise=2, delta=1e-3, seed=7):
    return measure_grid_delta(
        anima_call=_grid_call(net, z0),
        network=net,
        latents=z0,
        crossattn_emb=torch.zeros(z0.shape[0], 4, 8),
        padding_mask=torch.zeros(z0.shape[0], 1, 8, 8),
        forward_kwargs={},
        model_dtype=torch.float32,
        sigmas=list(sigmas),
        n_noise=n_noise,
        delta=delta,
        generator=torch.Generator().manual_seed(seed),
    )


def test_grid_delta_math_and_restore():
    torch.manual_seed(0)
    z0 = torch.randn(2, 4, 8, 8)
    net = _Net()
    net.multiplier = 0.8  # non-default: restore must return exactly this
    d = _run_grid(z0, net)
    assert d.shape == (2,)
    assert not d.requires_grad
    # adapted is a perfect model (mse=0 → log δ); base predicts zeros, so
    # Δ = log(mean‖ε−z0‖²+δ) − log(δ) > 0 for every sample.
    assert bool((d > 0).all())
    assert net.multiplier == 0.8
