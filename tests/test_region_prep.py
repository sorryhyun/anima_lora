"""Invariant tests for the region EasyControl staging pipeline.

Covers the two load-bearing pieces of i2i_tasks/region/prep.py:

* ``_mask_gates`` — the dirty-mask filter (drop reasons, speck cleanup).
* ``_augment_mask`` — every silhouette level must keep the pose readable
  (slack is a region by design and is only held to superset + bounds):
  a wide gap between limbs (the spread-legs case) must NOT be bridged into
  one fat blob, and the per-stem seeding must be deterministic.

Torch-free and CPU-only (numpy + cv2).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from i2i_tasks.region.prep import _augment_mask, _mask_gates

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
    TIGHT_LEVELS = (0, 1, 2, 3)  # exact / smooth / blob / wobble — silhouette levels
    SLACK = 4

    @pytest.mark.parametrize("seed", range(3))
    def test_limb_gap_survives(self, seed):
        """No silhouette level may bridge a wide limb gap into one blob
        (user feedback 2026-08-23: fat blobs over-simplified an M-leg pose).
        Slack is exempt by design — it is a *region*, not a silhouette."""
        m = _two_legs(gap_frac=0.25)
        gap_lo = int(W * 0.5 - W * 0.25 / 2)
        gap_hi = int(W * 0.5 + W * 0.25 / 2)
        for level in self.TIGHT_LEVELS:
            out = _augment_mask(m, random.Random(seed), level)
            gap_band = out[int(H * 0.65) : H - 10, gap_lo:gap_hi]
            assert gap_band.mean() < 0.35, (
                f"seed {seed} level {level}: limb gap {gap_band.mean():.0%} painted"
            )

    @pytest.mark.parametrize("seed", range(3))
    def test_strict_superset_of_source(self, seed):
        """No character pixel may escape the paint (any augment level).

        On a real-background cond a leaked fringe (hair wisp, foot) exposes
        pose/identity the model can cheat off — a channel that's empty at
        inference, where nothing sits under the paint."""
        for maker in (_ellipse, _two_legs):
            m = maker()
            for level in (*self.TIGHT_LEVELS, self.SLACK):
                out = _augment_mask(m, random.Random(seed), level)
                if out is None:
                    continue  # slack with no room — caller falls back
                assert int((m & ~out).sum()) == 0, f"level {level}"

    @pytest.mark.parametrize("seed", range(2))
    def test_slack_is_loose_and_bounded(self, seed):
        """Slack paints a region clearly larger than the character (the
        harmonize lever: background must be continued under the rest) but
        never the whole frame (no scene left to harmonize with)."""
        m = np.zeros((H, W), np.uint8)
        m[int(H * 0.3) : int(H * 0.7), int(W * 0.35) : int(W * 0.55)] = 1
        out = _augment_mask(
            m,
            random.Random(seed),
            self.SLACK,
            slack_grow=(1.5, 3.0),
            max_slack_coverage=0.7,
        )
        assert out is not None
        assert out.sum() >= 1.25 * m.sum()
        assert out.mean() <= 0.72
        assert int((m & ~out).sum()) == 0

    def test_slack_full_frame_falls_back(self):
        """A figure filling the frame leaves no room: slack returns None so the
        caller picks a tight level instead of painting the whole image."""
        m = np.zeros((H, W), np.uint8)
        m[4 : H - 4, 4 : W - 4] = 1
        assert _augment_mask(m, random.Random(0), self.SLACK) is None

    def test_deterministic_per_seed(self):
        m = _two_legs()
        for level in (*self.TIGHT_LEVELS, self.SLACK):
            a = _augment_mask(m, random.Random(7), level)
            b = _augment_mask(m, random.Random(7), level)
            assert (a is None and b is None) or np.array_equal(a, b)
