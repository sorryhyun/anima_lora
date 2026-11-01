"""Per-band dynamic-seq compilation (--compile_seq_bands) invariants.

Phase 0 of _archive/proposals/perband_dynamic_seq.md: band clustering is
data-driven over the actual token-count set (not EDGE_TOKEN_BANDS), band
dispatch picks the containing band or nothing, and register-token widening
must never silently merge adjacent bands. All torch-free (buckets helpers).
"""

import pytest

from library.datasets.buckets import (
    EDGE_TOKEN_BANDS,
    band_for_seq,
    cluster_token_bands,
    widen_bands,
)

# Realistic count sets, straight from the tier tables.
COUNTS_1024 = {4032, 4100, 4160, 4200}  # inside (4032, 4200)
COUNTS_896 = {3000, 3008, 3024}  # inside (3000, 3024)


def test_gap_sample_prompt_becomes_singleton_band():
    # A sample prompt landing between tiers (e.g. 768x1200 -> 3600 tokens)
    # must become its own band, not widen a tier band across the dead zone.
    bands = cluster_token_bands(COUNTS_896 | {3600} | COUNTS_1024)
    assert bands == [(3000, 3024), (3600, 3600), (4032, 4200)]


def test_tier_table_edges_cluster_one_band_per_tier_family():
    # The full tier table's band endpoints: adjacent-tier gaps all exceed the
    # 10% threshold, so each tier's endpoints stay in their own band.
    counts = {c for band in EDGE_TOKEN_BANDS.values() for c in band}
    bands = cluster_token_bands(counts)
    assert len(bands) == len(EDGE_TOKEN_BANDS)


def test_band_for_seq_membership():
    bands = [(3000, 3024), (4032, 4200)]
    assert band_for_seq(bands, 3000) == (3000, 3024)
    assert band_for_seq(bands, 3024) == (3000, 3024)
    assert band_for_seq(bands, 4100) == (4032, 4200)
    # gap, below, above, empty
    assert band_for_seq(bands, 3600) is None
    assert band_for_seq(bands, 2000) is None
    assert band_for_seq(bands, 9000) is None
    assert band_for_seq([], 4100) is None


def test_widen_bands_refuses_band_merge():
    # K >= the inter-band gap would make bands touch -> loud failure, never
    # a silent merge (proposal: assert K < smallest inter-band gap).
    bands = [(3000, 3024), (4032, 4200)]
    with pytest.raises(ValueError, match="inter-band gap"):
        widen_bands(bands, 4032 - 3024)  # touching counts as merged
    with pytest.raises(ValueError):
        widen_bands(bands, 2000)
    # A single band can widen arbitrarily (no neighbor to collide with).
    assert widen_bands([(4032, 4200)], 2000) == [(4032, 6200)]
