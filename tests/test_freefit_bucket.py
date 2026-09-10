"""Free-aspect token-band resize ("free-fit") solver invariants.

Locks the Phase 0 contract from
docs/proposal/free_aspect_token_band_resize.md:
  * every fitted grid lands inside the tier's token band,
  * the result respects the max_ratio clamp and the RoPE per-axis cap,
  * crop residual is sub-patch (< 16px) for in-range aspects on the 1024 tier,
  * the solver is deterministic,
  * the GUI/CLI preview (compute_resize_preview) agrees with the solver feeding
    process_image (select_resize_bucket).
"""

import pytest

from library.datasets.buckets import (
    ALLOWED_TARGET_RES,
    BucketManager,
    freefit_band_for_edge,
    freefit_bucket,
    token_count_range,
)
from library.preprocess.resize_preview import (
    compute_resize_preview,
    select_resize_bucket,
)

PATCH = 16
ROPE_CAP = 256

# A spread of native pixel sizes × aspect ratios. Aspects stay within the
# default 1:4 clamp so no crop should be forced.
_NATIVE_SIZES = [
    (1024, 1024),
    (1920, 1080),
    (1080, 1920),
    (1500, 1000),
    (1000, 1500),
    (2048, 768),
    (768, 2048),
    (1234, 987),
    (640, 2400),
    (3000, 2000),
]
_ASPECTS = [
    0.25,
    0.3,
    0.4,
    0.5,
    0.5625,
    0.667,
    0.75,
    0.8,
    1.0,
    1.25,
    1.5,
    1.78,
    2.0,
    3.0,
    4.0,
]


def _grids(edge):
    band = freefit_band_for_edge(edge)
    return band, [freefit_bucket(round(a * 1000), 1000, band) for a in _ASPECTS]


def test_band_derivation():
    # 1024 is frozen at its natural 2-family band (the top-5 aspect set keys off it).
    assert freefit_band_for_edge(1024) == (4032, 4200)
    # Every other tier is widened symmetrically by ±2.5% around its nominal so
    # free-fit has aspect freedom even on the single-family tiers — without it,
    # 768 (2160), 1280 (6300), 1536 (8640) and the 16-wide 512 leave no room and
    # free-fit crops like Snap (regression guarded by the sub-patch-crop test).
    assert freefit_band_for_edge(512) == (983, 1050)
    assert freefit_band_for_edge(768) == (2106, 2214)
    assert freefit_band_for_edge(896) == (2925, 3100)
    assert freefit_band_for_edge(1280) == (6142, 6457)
    assert freefit_band_for_edge(1536) == (8424, 8856)


def _crop_residual(sw, sh, bw, bh):
    scale = max(bw / sw, bh / sh)
    return max(sw * scale - bw, sh * scale - bh)


# Common aspect ratios (real photos / art) — never elongated enough to hit the
# low-token grid-granularity floor on the small tiers.
_COMMON_ASPECTS = [0.5, 0.667, 0.75, 0.8, 0.866, 1.0, 1.25, 1.33, 1.5, 2.0]


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_widened_band_never_crops_more(edge):
    """The widened band contains the tier's natural ``(min, max)`` band, so the
    solver can only find an equal-or-better aspect match → it never crops *more*
    than the pre-fix degenerate band. The strong invariant behind #53's free-fit
    Phase 0 crop bug (768 @ ar 0.866 went 63px → 5px)."""
    new = freefit_band_for_edge(edge)
    old = token_count_range((edge,))
    for a in _ASPECTS:
        sw, sh = round(a * 1000), 1000
        new_res = _crop_residual(sw, sh, *freefit_bucket(sw, sh, new))
        old_res = _crop_residual(sw, sh, *freefit_bucket(sw, sh, old))
        assert new_res <= old_res + 1e-6, (edge, a, new_res, old_res)


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_common_aspects_crop_subpatch_on_every_tier(edge):
    """The fix's payoff: common aspect ratios crop ≤ 1 patch on *every* tier,
    including the single-family 768/1280/1536 whose degenerate natural band used
    to force a Snap-sized crop. (Exactly one patch can remain on the smallest 512
    tier's ~32-patch grid; extreme aspects, e.g. 1:2.7, can exceed a patch there —
    inherent to coarse low-token grids, design §4. Pre-fix these were 48–96px.)"""
    band = freefit_band_for_edge(edge)
    for a in _COMMON_ASPECTS:
        sw, sh = round(a * 1000), 1000
        bw, bh = freefit_bucket(sw, sh, band)
        assert _crop_residual(sw, sh, bw, bh) <= PATCH, (edge, a, bw, bh)


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_grid_lands_in_band(edge):
    band, grids = _grids(edge)
    lo, hi = band
    for w, h in grids:
        assert w % PATCH == 0 and h % PATCH == 0, (edge, w, h)
        tokens = (w // PATCH) * (h // PATCH)
        assert lo <= tokens <= hi, (edge, w, h, tokens, band)


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_respects_rope_cap(edge):
    _, grids = _grids(edge)
    for w, h in grids:
        assert max(w // PATCH, h // PATCH) <= ROPE_CAP, (edge, w, h)


def test_max_ratio_clamp_on_degenerate_input():
    band = freefit_band_for_edge(1024)
    # 1:8 portrait — far outside the 1:4 clamp. The fitted aspect must not exceed
    # the clamp (within one patch of the integer grid).
    w, h = freefit_bucket(500, 4000, band, max_ratio=4.0)
    fitted_ar = w / h  # < 1 (portrait)
    assert fitted_ar >= 1.0 / 4.0 - (1.0 / (h // PATCH)), (w, h, fitted_ar)
    # And a custom tighter clamp is honored too.
    w2, h2 = freefit_bucket(4000, 500, band, max_ratio=2.0)
    assert (w2 / h2) <= 2.0 + (1.0 / (h2 // PATCH)), (w2, h2)


def test_crop_residual_subpatch_on_1024_tier():
    """In-range aspects crop < 1 patch (the <16px residual the proposal claims)."""
    band = freefit_band_for_edge(1024)
    for sw, sh in _NATIVE_SIZES:
        a = sw / sh
        if not (0.25 <= a <= 4.0):
            continue
        bw, bh = freefit_bucket(sw, sh, band)
        # Cover-resize native → bucket preserving aspect, then measure the crop.
        scale = max(bw / sw, bh / sh)
        covered_w, covered_h = sw * scale, sh * scale
        residual = max(covered_w - bw, covered_h - bh)
        assert residual < PATCH, (sw, sh, bw, bh, residual)


def test_deterministic():
    band = freefit_band_for_edge(1024)
    for sw, sh in _NATIVE_SIZES:
        first = freefit_bucket(sw, sh, band)
        for _ in range(3):
            assert freefit_bucket(sw, sh, band) == first


def test_near_square_is_exact_square():
    # 4096 = 64*64 ∈ [4032, 4200], so a square fits with zero aspect error.
    assert freefit_bucket(1000, 1000, freefit_band_for_edge(1024)) == (1024, 1024)


def test_aspect_error_is_minimal():
    """No neighboring in-band grid is closer to the (clamped) target aspect."""
    band = freefit_band_for_edge(1024)
    lo, hi = band
    for sw, sh in _NATIVE_SIZES:
        a = sw / sh
        bw, bh = freefit_bucket(sw, sh, band)
        wp, hp = bw // PATCH, bh // PATCH
        best_err = abs(wp / hp - a)
        for h2 in range(1, min(ROPE_CAP, hi) + 1):
            w_lo = max(1, -(-lo // h2))
            w_hi = min(ROPE_CAP, hi // h2)
            for w2 in range(w_lo, w_hi + 1):
                assert abs(w2 / h2 - a) >= best_err - 1e-12, (sw, sh, w2, h2)


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_preview_agrees_with_solver(edge):
    """compute_resize_preview == select_resize_bucket == freefit_bucket."""
    band = freefit_band_for_edge(edge)
    for sw, sh in _NATIVE_SIZES:
        solver = freefit_bucket(sw, sh, band)
        _, selected = select_resize_bucket(sw, sh, [edge])
        preview = compute_resize_preview(sw, sh, [edge])
        assert selected == solver, (edge, sw, sh)
        assert preview.bucket_size == solver, (edge, sw, sh)
        assert preview.target_edge == edge


def test_single_tier_choose_edge_is_noop():
    # Free-fit operates inside the choose_edge-picked tier; single tier is direct.
    _, b = select_resize_bucket(1500, 1000, [1024])
    assert b == freefit_bucket(1500, 1000, freefit_band_for_edge(1024))


# --- Phase 1: train-side bucket mode -----------------------------------------


def test_bucketmanager_freefit_keeps_native_shapes():
    """Free-fit bucket mode: each on-disk (W, H) exact-matches — never AR-snaps."""
    band = freefit_band_for_edge(1024)
    sizes = {
        freefit_bucket(1500, 1000, band),
        freefit_bucket(1000, 1500, band),
        (1024, 1024),
        freefit_bucket(2048, 700, band),
    }
    bm = BucketManager()
    bm.make_buckets(freefit_resos=sizes)
    assert set(bm.predefined_resos) == sizes
    for w, h in sizes:
        reso, resized, ar_error = bm.select_bucket(w, h)
        assert reso == (w, h), (w, h, reso)  # exact match, no snap
        assert resized == (w, h)
        assert ar_error == 0.0


def test_bucketmanager_freefit_token_counts_in_one_band():
    """All free-fit shapes for the 1024 tier share its [4032, 4200] band → one
    dynamic-seq graph (the compile-coupling invariant)."""
    lo, hi = freefit_band_for_edge(1024)
    sizes = {freefit_bucket(round(a * 1000), 1000, (lo, hi)) for a in _ASPECTS}
    counts = {(w // PATCH) * (h // PATCH) for (w, h) in sizes}
    assert all(lo <= c <= hi for c in counts)
