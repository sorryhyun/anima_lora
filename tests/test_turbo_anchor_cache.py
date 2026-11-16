"""dpdmd.anchor_cache: pair keys, ε regeneration, the manifest check and the draw cycle."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts.distill_turbo.anchor_cache import (
    FORMAT,
    MANIFEST,
    AnchorDataset,
    anchor_collate,
    pair_eps,
    pair_path,
    pair_seed,
    sample_key,
    signature,
)


def _cfg(tmp_path, **over):
    base = dict(
        dit_path="models/diffusion_models/anima_merged15_int8_gptq.safetensors",
        dit_nonquant_path="models/diffusion_models/anima_merged15.safetensors",
        int8="stream",
        int8_file=None,
        stream="fp32",
        teacher_cfg=4.0,
        flow_shift=3.0,
        k_anchor=14,
        teacher_anchor_steps=28,
        seed=42,
        data_dir=str(tmp_path / "data"),
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Base(torch.utils.data.Dataset):
    """Stand-in for CachedDataset: ``samples`` plus items in its shape."""

    def __init__(self, cfg, n=3, shape=(16, 4, 6)):
        self.samples = [
            ("x.npz", f"{cfg.data_dir}/artist/{i}_anima_te.safetensors")
            for i in range(n)
        ]
        self.shape = shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "latents": torch.zeros(self.shape),
            "crossattn_emb": torch.zeros(5, 8),
        }


def _write_cache(cache_dir, cfg, samples, draws, shape=(16, 4, 6), skip=()):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / MANIFEST).write_text(
        json.dumps({"format": FORMAT, "draws": draws, "signature": signature(cfg)})
    )
    for _, te in samples:
        key = sample_key(te, cfg.data_dir)
        for k in range(draws):
            if (key, k) in skip:
                continue
            v = torch.full(shape, float(k)) + hash(key) % 7
            save_file({"v_target": v}, str(pair_path(cache_dir, key, k)))


def test_sample_key_keeps_subdir(tmp_path):
    cfg = _cfg(tmp_path)
    te = f"{cfg.data_dir}/artist/123_anima_te.safetensors"
    assert sample_key(te, cfg.data_dir) == "artist__123"


def test_pair_eps_is_reproducible_and_distinct():
    s0, s1 = pair_seed(42, "a__1", 0), pair_seed(42, "a__1", 1)
    assert s0 != s1 and s0 == pair_seed(42, "a__1", 0)
    a = pair_eps((1, 16, 4, 6), s0, device="cpu", dtype=torch.float32)
    b = pair_eps((1, 16, 4, 6), s0, device="cpu", dtype=torch.float32)
    c = pair_eps((1, 16, 4, 6), s1, device="cpu", dtype=torch.float32)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_items_carry_the_epoch_draw(tmp_path):
    cfg = _cfg(tmp_path)
    base = _Base(cfg)
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, cfg, base.samples, draws=2)
    ds = AnchorDataset(base, str(cache_dir), cfg, epoch=3)
    assert ds.draw == 1
    ds.next_epoch()
    assert (ds.epoch, ds.draw) == (4, 0)
    ds.next_epoch()

    batch = anchor_collate([ds[1], ds[2]])
    assert batch["idx"] == [1, 2]
    assert batch["anchor_eps"].shape == batch["anchor_v"].shape == (2, 16, 4, 6)
    key = sample_key(base.samples[1][1], cfg.data_dir)
    want = pair_eps(
        (16, 4, 6), pair_seed(42, key, 1), device="cpu", dtype=torch.float32
    )
    assert torch.equal(batch["anchor_eps"][0], want)
    assert torch.allclose(
        batch["anchor_v"][0], torch.full((16, 4, 6), 1.0) + hash(key) % 7
    )


def test_cache_refuses_other_signature(tmp_path):
    cfg = _cfg(tmp_path)
    base = _Base(cfg)
    _write_cache(tmp_path / "cache", cfg, base.samples, draws=1)
    other = _cfg(tmp_path, k_anchor=6, teacher_anchor_steps=12)
    with pytest.raises(ValueError, match="k_anchor"):
        AnchorDataset(base, str(tmp_path / "cache"), other)


def test_cache_refuses_missing_pairs(tmp_path):
    cfg = _cfg(tmp_path)
    base = _Base(cfg)
    key = sample_key(base.samples[0][1], cfg.data_dir)
    _write_cache(tmp_path / "cache", cfg, base.samples, draws=2, skip={(key, 1)})
    with pytest.raises(FileNotFoundError, match="1 of 6 pairs missing"):
        AnchorDataset(base, str(tmp_path / "cache"), cfg)


def test_stale_shape_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    base = _Base(cfg, shape=(16, 8, 6))
    _write_cache(tmp_path / "cache", cfg, base.samples, draws=1)  # (16, 4, 6) pairs
    with pytest.raises(ValueError, match="stale"):
        AnchorDataset(base, str(tmp_path / "cache"), cfg)[0]
