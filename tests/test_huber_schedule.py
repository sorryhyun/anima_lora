"""Regression test for the σ-scale Huber `exponential` schedule.

Anima feeds the DiT its time argument as σ∈[0,1] (runtime/noise.py), not the
[0,1000] scale the original sd-scripts Huber formula assumed. The exponential
schedule must therefore decay across σ∈[0,1] — not stay pinned flat at
huber_scale (the pre-fix bug, where the /num_train_timesteps divisor shrank the
exponent ~1000×). See library/training/losses.py::get_huber_threshold_if_needed.
"""

from __future__ import annotations

import argparse
import math

import torch

from library.training.losses import get_huber_threshold_if_needed


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        loss_type="huber",
        huber_schedule="exponential",
        huber_c=0.1,
        huber_scale=1.0,
        pseudo_huber_c=0.03,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_exponential_schedule_decays_across_sigma():
    # σ on the [0,1] scale, as Anima actually passes it.
    sigmas = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    result = get_huber_threshold_if_needed(_args(), sigmas, noise_scheduler=None)

    # σ=0 (clean) → huber_scale; σ=1 (noise) → huber_c·huber_scale.
    assert math.isclose(result[0].item(), 1.0, rel_tol=1e-5)
    assert math.isclose(result[-1].item(), 0.1, rel_tol=1e-5)

    # Strictly decreasing — the schedule must NOT be flat (the bug).
    diffs = result[1:] - result[:-1]
    assert torch.all(diffs < 0)
    assert (result[0] - result[-1]).item() > 0.5  # real spread, not ~0
