"""PoolAccumulator invariants for the E3 memory fix (streaming + disk spill).

The pooled batch-SGD object must not depend on HOW it was accumulated:
per-arm streaming (``add_arm`` + ``add_native``), disk-backed memmaps
(``backing_dir`` + ``release``/``reset`` between strata), and the
``keep_norm=False`` slimming must all reproduce the original whole-image
in-RAM ``add_image`` path bit-for-bit (same per-key addition order).
"""

import numpy as np
import pytest
import torch

from project.sigma_lowres.bench.sigma_probe.stats import PoolAccumulator, pool_stats

BINS = 3
DIM = 64
ARM_KEYS = ["reenc", "896"]


_BASE = np.random.default_rng(7).standard_normal((BINS, DIM)).astype(np.float32)


def _fake_image(rng: np.random.Generator) -> dict[str, list[torch.Tensor]]:
    # shared per-bin direction + small noise → every pooled cosine is safely
    # positive, so the debiased-gap entries never hit the None (floor ≤ 0) leg
    keys = ["a", "b"] + [k for key in ARM_KEYS for k in (key, f"{key}__2")]
    return {
        k: [
            torch.from_numpy(
                _BASE[b] + 0.1 * rng.standard_normal(DIM).astype(np.float32)
            )
            for b in range(BINS)
        ]
        for k in keys
    }


def _stats_of(acc: PoolAccumulator) -> dict:
    return pool_stats(acc, ARM_KEYS)


@pytest.fixture()
def images():
    rng = np.random.default_rng(0)
    return [_fake_image(rng) for _ in range(4)]


def _stream_in(acc: PoolAccumulator, img: dict) -> None:
    for key, vecs in img.items():
        if key not in ("a", "b"):
            acc.add_arm(key, [v.clone() for v in vecs])
    acc.add_native([v.clone() for v in img["a"]], [v.clone() for v in img["b"]], 0.5)


def test_streaming_matches_add_image(images):
    whole = PoolAccumulator()
    streamed = PoolAccumulator()
    for img in images:
        whole.add_image({k: [v.clone() for v in vs] for k, vs in img.items()}, 0.5)
        _stream_in(streamed, img)
    assert _stats_of(whole) == _stats_of(streamed)


def test_backed_matches_ram(images, tmp_path):
    ram = PoolAccumulator()
    backed = PoolAccumulator(backing_dir=tmp_path / "spill")
    for img in images:
        ram.add_image({k: [v.clone() for v in vs] for k, vs in img.items()}, 0.5)
        _stream_in(backed, img)
        backed.release()  # the between-images handle drop of the run loop
    assert _stats_of(ram) == _stats_of(backed)


def test_reset_reuses_backing_files(images, tmp_path):
    """Two strata through one backed accumulator (reset between) must match
    two fresh accumulators — reset must not leak the previous stratum."""
    backed = PoolAccumulator(backing_dir=tmp_path / "spill")
    fresh0, fresh1 = PoolAccumulator(), PoolAccumulator()
    _stream_in(backed, images[0])
    _stream_in(backed, images[1])
    for img in images[:2]:
        fresh0.add_image({k: [v.clone() for v in vs] for k, vs in img.items()}, 0.5)
    s0 = _stats_of(backed)
    assert s0 == _stats_of(fresh0)
    backed.reset()
    _stream_in(backed, images[2])
    _stream_in(backed, images[3])
    for img in images[2:]:
        fresh1.add_image({k: [v.clone() for v in vs] for k, vs in img.items()}, 0.5)
    assert _stats_of(backed) == _stats_of(fresh1)


def test_merge_from_backed_stratum(images, tmp_path):
    """The aggregate path: backed cur_pool strata merged into a backed
    aggregate must equal one flat in-RAM accumulation of all images."""
    flat = PoolAccumulator()
    for img in images:
        flat.add_image({k: [v.clone() for v in vs] for k, vs in img.items()}, 0.5)
    agg = PoolAccumulator(backing_dir=tmp_path / "agg")
    cur = PoolAccumulator(backing_dir=tmp_path / "cur")
    for i, img in enumerate(images):
        _stream_in(cur, img)
        if cur.n == 2:
            agg.merge(cur)
            agg.release()
            cur.reset()
    got, want = _stats_of(agg), _stats_of(flat)
    assert got.keys() == want.keys()
    for k in want:
        if k == "n_images":
            assert got[k] == want[k]
            continue
        # merge adds stratum sums in a different order than flat per-image
        # adds — equal up to fp reassociation, not bit-equal
        np.testing.assert_allclose(
            np.array(got[k], dtype=float),
            np.array(want[k], dtype=float),
            atol=1e-4,
        )


def test_keep_norm_false_drops_only_norm_keys(images):
    full = PoolAccumulator()
    slim = PoolAccumulator(keep_norm=False)
    for img in images:
        full.add_image({k: [v.clone() for v in vs] for k, vs in img.items()}, 0.5)
        _stream_in(slim, img)
    fs, ss = _stats_of(full), _stats_of(slim)
    assert not any(k.startswith("norm_") for k in ss)
    assert {k for k in fs if not k.startswith("norm_")} == set(ss)
    for k in ss:
        assert fs[k] == ss[k]
