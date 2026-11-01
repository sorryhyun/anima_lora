"""Unit tests for ``--artists_shard k_N`` → path_pattern expansion.

Covers the deterministic round-robin partition (exhaustive, disjoint, balanced),
fnmatch-safe pattern emission for bracket/paren artist names, and the
config-mutation contract used by ``train.py``.
"""

from library.datasets.artist_shard import (
    apply_artist_shard,
    list_artists,
    shard_of,
)
from library.datasets.path_filter import filter_paths_by_glob


def test_round_robin_is_partition():
    artists = [f"a{i:02d}" for i in range(76)]
    n = 4
    shards = [shard_of(artists, k, n) for k in range(1, n + 1)]
    # exhaustive + disjoint
    flat = [a for s in shards for a in s]
    assert sorted(flat) == artists
    assert len(set(flat)) == len(artists)
    # balanced to within 1
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_apply_sets_path_pattern_and_filters(tmp_path):
    # artist tree with fnmatch-special chars in names
    names = ["abmayo", "asou_(asabu202)", "kat_[bu]", "sincos", "zzz"]
    for nm in names:
        (tmp_path / nm).mkdir()
    found = list_artists(str(tmp_path))
    assert found == sorted(names)

    uc = {"datasets": [{"subsets": [{"image_dir": str(tmp_path)}]}]}
    info = apply_artist_shard(uc, "1_2", resolve=None)
    pat = uc["datasets"][0]["subsets"][0]["path_pattern"]

    # the emitted pattern selects exactly shard 1's artists (abs paths, as
    # training passes), including the bracketed/paren names
    expected = shard_of(found, 1, 2)
    sample = [f"{tmp_path}/{a}/x.png" for a in found]
    keep = filter_paths_by_glob(sample, str(tmp_path), pat)
    kept = [a for a, k in zip(found, keep) if k]
    assert kept == expected
    assert info[str(tmp_path)]["n_artists"] == len(found)
    assert info[str(tmp_path)]["n_shard"] == len(expected)


def test_apply_skips_reg_and_missing(tmp_path):
    (tmp_path / "a").mkdir()
    uc = {
        "datasets": [
            {
                "subsets": [
                    {"image_dir": str(tmp_path), "is_reg": True},
                    {"image_dir": str(tmp_path / "does_not_exist")},
                ]
            }
        ]
    }
    info = apply_artist_shard(uc, "1_2", resolve=None)
    # reg subset untouched, missing dir leaves no pattern
    assert "path_pattern" not in uc["datasets"][0]["subsets"][0]
    assert info == {}
