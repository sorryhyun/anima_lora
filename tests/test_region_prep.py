"""Invariant tests for the region EasyControl staging pipeline.

Covers the two load-bearing pieces of easycontrol_adapters/region/prep.py:

* ``_mask_gates`` — the dirty-mask filter (drop reasons, speck cleanup).
* ``_augment_mask`` — every augmentation level must keep the pose readable:
  a wide gap between limbs (the spread-legs case) must NOT be bridged into
  one fat blob, and the per-stem seeding must be deterministic.

Torch-free and CPU-only (numpy + cv2).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from easycontrol_adapters.region.prep import _augment_mask, _mask_gates

GATES = dict(
    min_coverage=0.03,
    max_coverage=0.85,
    min_dominance=0.75,
    max_hole_frac=0.05,
    min_comp_frac=0.002,
    max_raggedness=20.0,
)

H = W = 256


def _ellipse(cx=0.5, cy=0.5, rx=0.25, ry=0.35) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    m = ((xs / W - cx) / rx) ** 2 + ((ys / H - cy) / ry) ** 2 <= 1.0
    return m.astype(np.uint8)


def _two_legs(gap_frac=0.25) -> np.ndarray:
    """Torso + two vertical legs with a wide gap — the M-leg proxy."""
    m = np.zeros((H, W), np.uint8)
    m[20 : H // 2, W // 4 : 3 * W // 4] = 1  # torso
    leg_w = int(W * (0.5 - gap_frac) / 2)
    m[H // 2 : H - 10, W // 4 : W // 4 + leg_w] = 1
    m[H // 2 : H - 10, 3 * W // 4 - leg_w : 3 * W // 4] = 1
    return m


class TestMaskGates:
    def test_clean_silhouette_passes(self):
        clean, reason = _mask_gates(_ellipse(), **GATES)
        assert reason is None
        assert clean is not None and clean.any()

    def test_empty_mask(self):
        _, reason = _mask_gates(np.zeros((H, W), np.uint8), **GATES)
        assert reason == "empty"

    def test_specks_only(self):
        m = np.zeros((H, W), np.uint8)
        for i in range(10):  # each dot ~4px << min_comp_frac * area
            m[10 + i * 20 : 12 + i * 20, 10:12] = 1
        _, reason = _mask_gates(m, **GATES)
        assert reason == "specks_only"

    def test_too_small_and_too_large(self):
        _, reason = _mask_gates(_ellipse(rx=0.05, ry=0.05), **GATES)
        assert reason == "too_small"
        _, reason = _mask_gates(np.ones((H, W), np.uint8), **GATES)
        assert reason == "too_large"

    def test_fragmented(self):
        m = np.zeros((H, W), np.uint8)
        m[20:120, 20:120] = 1
        m[140:240, 140:240] = 1  # two equal comps → dominance 0.5 < 0.75
        _, reason = _mask_gates(m, **GATES)
        assert reason == "fragmented"

    def test_ragged_comb(self):
        # One connected comp made of thin stripes — huge perimeter per area,
        # the salt-noise signature whose background connects outward (so the
        # hole gate alone can't see it).
        m = np.zeros((H, W), np.uint8)
        for y in range(30, 220, 12):
            m[y : y + 2, 30:220] = 1
        m[30:222, 30:32] = 1  # spine connecting the stripes
        _, reason = _mask_gates(m, **GATES)
        assert reason == "ragged"

    def test_holey_donut(self):
        m = _ellipse(rx=0.35, ry=0.35)
        m[_ellipse(rx=0.18, ry=0.18) > 0] = 0  # enclosed hole
        _, reason = _mask_gates(m, **GATES)
        assert reason == "holey"

    def test_speck_cleanup_keeps_main_component(self):
        m = _ellipse()
        m[5:7, 5:7] = 1  # a far-away speck
        clean, reason = _mask_gates(m, **GATES)
        assert reason is None
        assert clean[5:7, 5:7].sum() == 0  # speck removed
        assert clean.sum() >= _ellipse().sum()  # main body intact (close may grow)


class TestAugmentMask:
    WEIGHTS = [0.25, 0.35, 0.30, 0.10]

    @pytest.mark.parametrize("seed", range(12))
    def test_limb_gap_survives(self, seed):
        """No augment level may bridge a wide limb gap into one blob
        (user feedback 2026-08-23: fat blobs over-simplified an M-leg pose)."""
        m = _two_legs(gap_frac=0.25)
        out = _augment_mask(m, random.Random(seed), self.WEIGHTS)
        gap_lo = int(W * 0.5 - W * 0.25 / 2)
        gap_hi = int(W * 0.5 + W * 0.25 / 2)
        gap_band = out[int(H * 0.65) : H - 10, gap_lo:gap_hi]
        assert gap_band.mean() < 0.35, (
            f"seed {seed}: limb gap {gap_band.mean():.0%} painted — over-simplified"
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_covers_most_of_source(self, seed):
        """Augmentation may round the mask but not relocate/shrink it away."""
        m = _ellipse()
        out = _augment_mask(m, random.Random(seed), self.WEIGHTS)
        assert (out & m).sum() / m.sum() > 0.75

    @pytest.mark.parametrize("seed", range(12))
    def test_strict_superset_of_source(self, seed):
        """No character pixel may escape the paint (any augment level).

        On a real-background cond a leaked fringe (hair wisp, foot) exposes
        pose/identity the model can cheat off — a channel that's empty at
        inference, where nothing sits under the paint."""
        for maker in (_ellipse, _two_legs):
            m = maker()
            out = _augment_mask(m, random.Random(seed), self.WEIGHTS)
            assert int((m & ~out).sum()) == 0

    def test_deterministic_per_seed(self):
        m = _two_legs()
        a = _augment_mask(m, random.Random(7), self.WEIGHTS)
        b = _augment_mask(m, random.Random(7), self.WEIGHTS)
        assert np.array_equal(a, b)

    def test_degenerate_never_empty(self):
        m = np.zeros((H, W), np.uint8)
        m[100:110, 100:110] = 1
        for seed in range(8):
            out = _augment_mask(m, random.Random(seed), self.WEIGHTS)
            assert out.any()
