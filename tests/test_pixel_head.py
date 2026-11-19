"""Pixel head (library/anima/pixel_head.py): load from metadata, output shape per token
grid, the tap's CFG combine and step bookkeeping, the trained-σ check. CPU only."""

import torch
import torch.nn as nn
from safetensors.torch import save_file

from library.anima.pixel_head import (
    DEFAULT_SIGMA_RANGES,
    PATCH,
    PixelHead,
    PixelTap,
    lat_patches,
    load_pixel_head,
    out_of_range,
    parse_sigma_ranges,
)

D_FEAT = 32


def _head(up: int) -> PixelHead:
    torch.manual_seed(0)
    return PixelHead(D_FEAT, 64, width=16, depth=3, ls_init=1.0, use_skip=False, up=up)


def test_load_from_metadata(tmp_path):
    head = _head(up=1)
    path = tmp_path / "head.safetensors"
    meta = {
        "width": "16",
        "depth": "3",
        "up": "1",
        "skip": "0",
        "sigma": "0.06-0.13,0.5",
    }
    save_file(
        {k: v.contiguous() for k, v in head.state_dict().items()}, str(path), meta
    )
    loaded = load_pixel_head(path)
    assert loaded.up == 1 and not loaded.use_skip
    assert loaded.sigma_ranges == ((0.06, 0.13), (0.5, 0.5))
    f, lat = torch.randn(3, 5, D_FEAT), torch.randn(3, 5, 64)
    with torch.no_grad():
        assert torch.equal(head.eval()(f, lat), loaded(f, lat))


def test_output_shape_per_token_grid():
    for up in (0, 1):
        head = _head(up).eval()
        for gh, gw in ((4, 4), (3, 7)):
            with torch.no_grad():
                img = head(torch.randn(gh, gw, D_FEAT), torch.randn(gh, gw, 64))
            assert img.shape == (3, gh * PATCH, gw * PATCH)


def test_lat_patches_layout():
    lat = torch.arange(16 * 4 * 6, dtype=torch.float32).view(16, 4, 6)
    p = lat_patches(lat)
    assert p.shape == (2, 3, 64)
    # token (0, 1) holds latent rows 0-1, cols 2-3 of every channel
    assert torch.equal(p[0, 1].view(16, 2, 2), lat[:, 0:2, 2:4])
    assert torch.equal(lat_patches(lat[None]), p)


class _ToyDiT(nn.Module):
    """Anima's call shape: forward(x, timesteps) → final_layer.linear(feature)."""

    def __init__(self):
        super().__init__()
        self.final_layer = nn.Module()
        self.final_layer.linear = nn.Linear(D_FEAT, 4, bias=False)

    def forward(self, x, timesteps, feature=None):
        return self.final_layer.linear(feature)


def test_tap_cfg_combine_and_reset():
    dit = _ToyDiT()
    tap = PixelTap(dit)
    x = torch.randn(1, 16, 1, 8, 8)
    f_old = torch.randn(1, 1, 4, 4, D_FEAT)
    f_c, f_u = torch.randn(1, 1, 4, 4, D_FEAT), torch.randn(1, 1, 4, 4, D_FEAT)
    dit(x, torch.tensor([0.3]), feature=f_old)  # an earlier step, cleared below
    dit(x, torch.tensor([0.1]), feature=f_c)
    dit(x, torch.tensor([0.1]), feature=f_u)
    f, x_t = tap.feature(4.0)
    assert tap.sigma == torch.tensor(0.1).item() and len(tap.calls) == 2
    assert torch.allclose(f, f_u + 4.0 * (f_c - f_u))
    assert x_t is not None and not tap.tail_cached
    f1, _ = tap.feature(1.0)
    assert torch.allclose(f1, f_c)  # g = 1 → the cond feature

    # a final_layer call with no forward of its own (Spectrum cached step)
    dit.final_layer.linear(f_c)
    assert tap.tail_cached
    tap.remove()
    dit(x, torch.tensor([0.05]), feature=f_c)
    assert tap.sigma == torch.tensor(0.1).item()  # hooks gone


def test_tap_decode_shape():
    dit = _ToyDiT()
    tap = PixelTap(dit)
    head = _head(up=1).eval()
    x = torch.randn(2, 16, 1, 8, 6)  # B=2, 4×3 tokens
    dit(x, torch.tensor([0.1, 0.1]), feature=torch.randn(2, 1, 4, 3, D_FEAT))
    pixels = tap.decode(head, 1.0)
    assert pixels.shape == (2, 3, 64, 48)
    assert pixels.abs().max() <= 1.0


def test_sigma_ranges():
    assert parse_sigma_ranges("") == DEFAULT_SIGMA_RANGES
    assert not out_of_range(0.1)  # base 28 steps
    assert not out_of_range(0.5)  # turbo
    assert out_of_range(0.25)
    assert out_of_range(0.03)


def test_catalog_row():
    from library import downloads

    asset = downloads.by_id()["anima_pixel_head"]
    assert asset.files == (downloads.ANIMA_PIXEL_HEAD_FILE,)
    assert (
        downloads.default_pixel_head_path()
        == asset.dest / downloads.ANIMA_PIXEL_HEAD_FILE
    )
