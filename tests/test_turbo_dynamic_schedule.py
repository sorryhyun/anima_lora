"""Invariant tests for the CDM dynamic rollout grid (scripts/distill_turbo/primitives.py).

``sample_dynamic_sigmas`` re-samples the student rollout grid every iteration
(arXiv:2605.06376 §3.2). The loop consumes it exactly like the static grid —
``sigmas[i] - sigmas[i+1]`` is the Euler dt for step i — so these tests pin the
layout properties the rollout math assumes:

  * (N+1) points spanning 1 → 0, non-increasing (dt >= 0 for every step);
  * t_1 = 1 pinned — the DP-DMD diversity anchor supervises v_first at t=1 and
    composes unchanged only because of this;
  * N covers exactly U{n_min..n_max} (n_min=2 under dpdmd: step 0 is the
    anchor, DMD needs >= 1 refinement step);
  * CPU RNG seeded by torch.manual_seed (resume determinism).
"""

from __future__ import annotations

import pytest
import torch

from scripts.distill_turbo.primitives import sample_dynamic_sigmas


@pytest.mark.core
def test_layout_endpoints_and_monotonic():
    torch.manual_seed(0)
    for _ in range(200):
        grid = sample_dynamic_sigmas(2, 4)
        assert grid[0] == 1.0
        assert grid[-1] == 0.0
        assert all(a >= b for a, b in zip(grid, grid[1:]))
        assert all(0.0 < s < 1.0 for s in grid[1:-1])


def test_length_range_matches_n_bounds():
    torch.manual_seed(0)
    seen = set()
    for _ in range(500):
        grid = sample_dynamic_sigmas(2, 4)
        n = len(grid) - 1
        assert 2 <= n <= 4
        seen.add(n)
    assert seen == {2, 3, 4}  # every rollout length gets drawn
