"""SEA-schedule invariant tests (spectrum_sea_schedule.md P1).

Covers the load-bearing invariants of the SeaCache SEA-distance trigger that
grafts onto Spectrum's scheduler:

  - SEA filter physical limits: σ→0 is identity (passes all detail); the gain
    is unit-mean (the relative-L1 distance stays scale-stable).
  - ``solve_delta_for_refresh_ratio`` lands the refresh count on the target
    fraction.
  - The auto-δ target reproduces the window schedule's own forward count.
"""

from __future__ import annotations


import pytest
import torch

from networks.spectrum_sea import (
    count_refreshes,
    sea_filter,
    sea_gain,
    solve_delta_for_refresh_ratio,
)


# ── SEA filter physics ─────────────────────────────────────────────────────────


def test_sea_filter_sigma_zero_is_identity():
    x = torch.randn(4, 16, 16)
    y = sea_filter(x, sigma=0.0)
    assert torch.allclose(y, x, atol=1e-4)


def test_sea_gain_is_unit_mean():
    for sigma in (0.1, 0.3, 0.5, 0.75, 0.9):
        g = sea_gain(32, 32, sigma, beta=2.0, device="cpu", dtype=torch.float32)
        assert g.mean().item() == pytest.approx(1.0, abs=1e-5)


# ── auto-δ refresh-ratio calibration ───────────────────────────────────────────


def test_solve_delta_hits_target_refresh_ratio():
    torch.manual_seed(0)
    dists = torch.rand(40).tolist()
    for ratio in (0.3, 0.5, 0.62, 0.8):
        delta = solve_delta_for_refresh_ratio(dists, ratio)
        target = max(1, round(ratio * len(dists)))
        # Step function in δ — landing within 1 of the target is exact convergence.
        assert abs(count_refreshes(dists, delta) - target) <= 1


def test_window_decision_fraction_matches_window_compute():
    """The auto-δ target must reproduce the window's own forward count.

    refresh_ratio·n_decision (rounded) == the window schedule's decision-region
    actual count under the *same* warmup/stop — so a SEA arm calibrated to it
    spends exactly as many forwards as the window (matched compute by
    construction). This is what the hard-coded 0.62 got wrong: 0.62 was the
    window's *total* fraction, applied to the decision region → over-compute.
    """
    import math

    from networks.spectrum import _window_decision_fraction

    def ref_window_decision_count(n, warmup, stop, ws, flex):
        curr, consec, dec_actual = ws, 0, 0
        for i in range(n):
            forced = i < warmup or i >= stop
            actual = forced or ((consec + 1) % max(1, math.floor(curr)) == 0)
            if not forced:
                dec_actual += int(actual)
            if actual:
                if i >= warmup:
                    curr = round(curr + flex, 3)
                consec = 0
            else:
                consec += 1
        return dec_actual

    for n, warmup, stop in [(28, 7, 27), (24, 6, 21), (50, 5, 47)]:
        n_dec = stop - warmup
        frac = _window_decision_fraction(n, warmup, stop, 2.0, 0.25)
        assert 0.0 < frac <= 1.0
        assert round(frac * n_dec) == ref_window_decision_count(
            n, warmup, stop, 2.0, 0.25
        )
