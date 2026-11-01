"""PE sidecar resolution fallback chain (issues.md P2.2).

``_try_load_repa_pe`` used to resolve the ``{stem}_anima_{encoder}``
sidecar only next to the TE cache. Colorize redirects only the TE cache
(``text_cache_dir``) — so the lookup landed in a directory with zero
sidecars while the latent-cache dir held them all, and a staged
colorize+REPA run would have trained as a silent baseline again. The
chain is now TE-cache dir → subset latent-cache dir (writer's nesting
rule) → image dir.

Tests drive the real ``BaseDataset`` methods on a stub ``self`` (the
methods only touch ``load_repa_pe`` / ``repa_pe_encoder`` /
``image_to_subset``), with real safetensors files in tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from library.datasets.base import BaseDataset

ENCODER = "pe_spatial"


class _StubDataset:
    """Minimal stand-in carrying only the state the real methods read,
    borrowing the real resolution/loading methods from BaseDataset."""

    _repa_pe_sidecar_candidates = BaseDataset._repa_pe_sidecar_candidates
    _try_load_repa_pe = BaseDataset._try_load_repa_pe
    count_repa_pe_sidecars = BaseDataset.count_repa_pe_sidecars

    def __init__(self, image_to_subset):
        self.load_repa_pe = True
        self.repa_pe_encoder = ENCODER
        self.image_to_subset = image_to_subset
        self.image_data = {}


def _write_sidecar(directory: Path, stem: str, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {"image_features": torch.full((4, 8), value)},
        str(directory / f"{stem}_anima_{ENCODER}.safetensors"),
    )


def _layout(tmp_path: Path, *, redirect_te: bool):
    """Standard subset layout: images/ + latent cache/ (+ te_cache/ when the
    TE cache is redirected, mirroring colorize's text_cache_dir)."""
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    te_dir = tmp_path / "te_cache" if redirect_te else cache_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    te_dir.mkdir(parents=True, exist_ok=True)

    subset = SimpleNamespace(cache_dir=str(cache_dir), image_dir=str(image_dir))
    info = SimpleNamespace(
        image_key="img1",
        absolute_path=str(image_dir / "img1.png"),
        text_encoder_outputs_npz=str(te_dir / "img1_anima_te.safetensors"),
    )
    ds = _StubDataset({"img1": subset})
    return ds, info, image_dir, cache_dir, te_dir


def _load(ds, info):
    return ds._try_load_repa_pe(info)


def test_te_dir_preferred_over_latent_cache_dir(tmp_path):
    # When both dirs hold a sidecar the TE-cache sibling still wins.
    ds, info, _, cache_dir, te_dir = _layout(tmp_path, redirect_te=True)
    _write_sidecar(te_dir, "img1", 1.0)
    _write_sidecar(cache_dir, "img1", 2.0)
    feats = _load(ds, info)
    assert feats is not None and feats[0, 0].item() == 1.0


@pytest.mark.core
def test_redirected_te_falls_back_to_latent_cache_dir(tmp_path):
    # The colorize blocker: text_cache_dir redirects the TE cache to a dir
    # with zero sidecars; the features must still load from the latent cache.
    ds, info, _, cache_dir, te_dir = _layout(tmp_path, redirect_te=True)
    _write_sidecar(cache_dir, "img1", 3.0)
    assert not (te_dir / f"img1_anima_{ENCODER}.safetensors").exists()
    feats = _load(ds, info)
    assert feats is not None and feats[0, 0].item() == 3.0
    assert feats.dtype == torch.float32


def test_latent_cache_fallback_mirrors_nested_subdirs(tmp_path):
    # resolve_cache_path nests caches under cache_dir for nested source
    # layouts (images/artistA/img1.png → cache/artistA/img1_…); the fallback
    # must replicate the writer's rule, not just join cache_dir/stem.
    ds, info, image_dir, cache_dir, _ = _layout(tmp_path, redirect_te=True)
    info.absolute_path = str(image_dir / "artistA" / "img1.png")
    _write_sidecar(cache_dir / "artistA", "img1", 4.0)
    feats = _load(ds, info)
    assert feats is not None and feats[0, 0].item() == 4.0


def test_falls_back_to_image_dir(tmp_path):
    # Legacy layout: no sidecar in either cache dir, but one next to the image.
    ds, info, image_dir, _, _ = _layout(tmp_path, redirect_te=True)
    _write_sidecar(image_dir, "img1", 5.0)
    feats = _load(ds, info)
    assert feats is not None and feats[0, 0].item() == 5.0


# ---------------------------------------------------------------------------
# count_repa_pe_sidecars — the coverage probe train.py uses to fail fast on a
# fully-absent PE cache (use_repa would otherwise be a silent no-op).
# ---------------------------------------------------------------------------


def _layout_multi(tmp_path: Path, stems: list[str]):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    subset = SimpleNamespace(cache_dir=str(cache_dir), image_dir=str(image_dir))
    ds = _StubDataset({})
    for stem in stems:
        ds.image_to_subset[stem] = subset
        ds.image_data[stem] = SimpleNamespace(
            image_key=stem,
            absolute_path=str(image_dir / f"{stem}.png"),
            text_encoder_outputs_npz=str(cache_dir / f"{stem}_anima_te.safetensors"),
        )
    return ds, cache_dir


def test_count_partial(tmp_path):
    ds, cache_dir = _layout_multi(tmp_path, ["a", "b", "c"])
    _write_sidecar(cache_dir, "a", 1.0)
    assert ds.count_repa_pe_sidecars() == (1, 3)
