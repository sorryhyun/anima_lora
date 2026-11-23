"""RGBA head (library/anima/pixel_rgba.py, scripts/pixel_head/{train_rgba,*alpha*}.py): a
changed input re-encodes, the read images never train, the init keeps the RGB head's output,
and training needs no VAE or TE."""

import numpy as np
import pytest
import torch

from scripts.pixel_head import _alpha_cache as C
from scripts.pixel_head import _alpha_data as D
from library.anima.pixel_head import LAT_PATCH_DIM, PixelHead, load_pixel_head
from library.anima.pixel_rgba import rgba_from_rgb, straight_rgba
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
        colour = row.key in {c.key for c in [*fit, *read]}
        C.write_cutout(root, by_key[row.key], colour)
        cross = torch.zeros(C.TEXT_TOKENS, 1024, dtype=torch.bfloat16)
        cross[:5] = 1
        cross[2] = 0  # a zero inside the text stays
        C.write_text(root, row, cross)
        C.write_latent(root, row, torch.randn(16, 12, 16).to(torch.bfloat16))
    C.write_manifest(root, recipe, rows)
    return rows


@pytest.fixture
def cache(tmp_path):
    recipe = {"seed": 0, "dit": "a", "palette": PALETTE}
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


def test_rgba_init_keeps_the_rgb():
    torch.manual_seed(0)
    base = PixelHead(D_FEAT, LAT_PATCH_DIM, 16, 2, up=1)
    head = rgba_from_rgb(base)
    f, lat = torch.randn(6, 8, D_FEAT), torch.randn(6, 8, LAT_PATCH_DIM)
    with torch.no_grad():
        x = head(f, lat)
        assert x.shape == (4, 96, 128)
        assert torch.equal(x[:3], base(f, lat))
        assert x[3].std() > 0  # the α rows are not zero


def test_straight_rgba_takes_out_the_background():
    a = np.zeros((64, 64), np.float32)
    a[16:48, 16:48] = 1
    a[16:48, 44:48] = 0.5  # a soft rim
    bg, fg = np.array([0.9, 0.95, 1.0]), np.array([0.2, 0.4, 0.1])
    c = (a[..., None] * fg + (1 - a[..., None]) * bg).astype(np.float32)
    x = straight_rgba(c, a)
    assert np.allclose(x[..., 3], a)
    assert np.allclose(x[16:48, 44:48, :3], fg, atol=0.02)


def test_cached_inputs_are_max_padded(cache):
    root, *_, rows = cache
    latent, cross = C.read_inputs(root, rows[0])
    assert latent.shape == (1, 16, 12, 16) and cross.shape == (1, C.TEXT_TOKENS, 1024)
    assert [i for i in range(512) if cross[0, i].any()] == [0, 1, 3, 4]
    for row in rows:
        assert np.array_equal(C.read_alpha(root, row.key), _cut(row.key).a)


class _FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer = torch.nn.Module()
        self.final_layer.linear = torch.nn.Linear(D_FEAT, 64)

    def forward(self, x, sigma, cross, padding_mask=None):
        assert cross.shape == (1, 512, 1024)
        h, w = x.shape[-2:]
        self.final_layer.linear(torch.randn(1, 1, h // 2, w // 2, D_FEAT))
        return torch.zeros_like(x)


def test_training_loads_no_vae_or_te(cache, tmp_path, monkeypatch):
    from library.inference import models
    from library.models import qwen_vae
    from scripts.pixel_head import _setup
    from scripts.pixel_head.train_rgba import TrainRGBARequest, run

    def refuse(*a, **k):
        raise AssertionError("training loaded a VAE or a TE")

    monkeypatch.setattr(qwen_vae, "load_vae", refuse)
    monkeypatch.setattr(models, "load_text_encoder", refuse)
    monkeypatch.setattr(_setup, "load_dit", lambda *a, **k: (_FakeDiT(), None))
    monkeypatch.setenv("ANIMA_PROGRESS", "0")
    base = PixelHead(D_FEAT, LAT_PATCH_DIM, 16, 2, up=1)
    save_x1_head(base, tmp_path / "base.safetensors", {"sigma": "0.0577-0.1364"})
    req = TrainRGBARequest(
        output_name="t",
        output_dir=str(tmp_path / "out"),
        cache_dir=str(cache[0]),
        init=str(tmp_path / "base.safetensors"),
        rollouts=3,
        opt_steps=2,
        replay=2,
        warmup=1,
        lambda_dino=0.0,
        eval_every=3,
        compile=False,
        device="cpu",
    )
    path = run(req)
    head = load_pixel_head(path)
    assert head.channels == 4 and head.meta["init"] == req.init
    assert head(torch.randn(6, 8, D_FEAT), torch.randn(6, 8, LAT_PATCH_DIM)).shape == (
        4,
        96,
        128,
    )
    assert head.sigma_ranges == ((0.0577, 0.1364),)
