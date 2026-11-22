"""Alpha head (library/anima/pixel_alpha*.py): a changed input re-encodes, the read
images never train, and training runs from the cache alone."""

import numpy as np
import pytest
import torch

from library.anima import pixel_alpha_cache as C
from library.anima import pixel_alpha_data as D
from library.anima.pixel_alpha import AlphaHead, load_alpha_head
from library.anima.pixel_head import LAT_PATCH_DIM, PixelHead
from library.anima.pixel_head_train_x1 import save_x1_head

D_FEAT = 32
PALETTE = [
    {"coef": [[0.9, 0.9, 0.9], [0.03, 0.03, 0.03], [0, 0, 0]], "res": 0.004},
    {"coef": [[0.95, 0.85, 0.85], [0, 0, 0], [0, 0, 0]], "res": 0.002},
]


def _cut(key: str, group: str = "") -> D.Cutout:
    a = np.zeros((96, 128), np.uint8)
    a[16:80, 24:100] = 255
    a[30:50, 40:70] = 128
    fg = np.full((96, 128, 3), 90, np.uint8)
    cap = "1girl, transparent background"
    return D.Cutout(key, fg, a, cap, f"sha-{key}", {}, group)


def _build(root, recipe, cuts, fit, read):
    rows = C.plan_rows(cuts, fit, read, 1, recipe)
    by_key = {c.key: c for c in cuts}
    for row in rows:
        (root / row.set).mkdir(parents=True, exist_ok=True)
        comp, bg, sigma = C.compose(row, by_key[row.key], PALETTE, recipe)
        f = torch.randn(6, 8, D_FEAT).to(torch.bfloat16)
        lat = torch.randn(6, 8, LAT_PATCH_DIM)
        C.write_row(root, row, by_key[row.key], f, lat, sigma, comp, bg)
    C.write_manifest(root, recipe, rows)
    return rows


@pytest.fixture
def cache(tmp_path):
    recipe = {"sigma": [0.0577, 0.1364], "seed": 0, "stream": "fp32", "dit": "a"}
    fit, read = [_cut("f", "thin")], [_cut("r", "thin")]
    cuts = [*fit, *read, _cut("t")]
    root = tmp_path / "cache"
    return root, recipe, cuts, fit, read, _build(root, recipe, cuts, fit, read)


def test_changed_input_is_reencoded(cache):
    root, recipe, cuts, fit, read, rows = cache
    assert all(C.is_current(root, r) for r in rows)
    other_dit = C.plan_rows(cuts, fit, read, 1, {**recipe, "dit": "b"})
    assert not any(C.is_current(root, r) for r in other_dit)
    cuts[2].caption = "1boy, transparent background"
    stale = {
        r.key
        for r in C.plan_rows(cuts, fit, read, 1, recipe)
        if not C.is_current(root, r)
    }
    assert stale == {"t"}


def test_read_images_never_train(cache):
    *_, rows = cache
    assert "r" not in {r.key for r in rows if r.set == "train"}
    assert {r.key for r in rows if r.set == "other"} == {"r"}


def test_alpha_output_has_the_rgb_layout():
    torch.manual_seed(0)
    base = PixelHead(D_FEAT, LAT_PATCH_DIM, 16, 2, use_skip=False, up=1)
    head = AlphaHead.from_base(base)
    with torch.no_grad():
        base.out.weight[0::3] = head.out.weight
        base.out.bias[0::3] = head.out.bias
        f, lat = torch.randn(6, 8, D_FEAT), torch.randn(6, 8, LAT_PATCH_DIM)
        assert torch.allclose(base(f, lat)[0], head(f, lat)[0], atol=1e-5)


def test_training_needs_only_the_cache(cache, tmp_path, monkeypatch):
    from library.models import qwen_vae
    from library.runtime import harness
    from scripts.pixel_head.train_alpha import TrainAlphaRequest, run

    def refuse(*a, **k):
        raise AssertionError("training loaded a model")

    monkeypatch.setattr(qwen_vae, "load_vae", refuse)
    monkeypatch.setattr(harness, "build_anima", refuse)
    monkeypatch.setenv("ANIMA_PROGRESS", "0")
    base = PixelHead(D_FEAT, LAT_PATCH_DIM, 16, 2, use_skip=False, up=1)
    save_x1_head(base, tmp_path / "base.safetensors", {"sigma": "0.0577-0.1364"})
    req = TrainAlphaRequest(
        output_name="t",
        output_dir=str(tmp_path / "out"),
        cache_dir=str(cache[0]),
        init=str(tmp_path / "base.safetensors"),
        rollouts=3,
        opt_steps=2,
        replay=2,
        warmup=1,
        lambda_lpips=0.0,
        eval_every=3,
        compile=False,
        device="cpu",
    )
    path = run(req)
    head = load_alpha_head(path)
    assert head(torch.randn(6, 8, D_FEAT), torch.randn(6, 8, LAT_PATCH_DIM)).shape == (
        1,
        96,
        128,
    )
    assert head.sigma_ranges == ((0.0577, 0.1364),)
