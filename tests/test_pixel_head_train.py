"""×2 pixel-head training pieces (library/anima/pixel_head_train.py): the window and its
margins, the loss terms, the schedule, the GAN term's start, the saved file. CPU only."""

import json
import random

import torch

from library.anima.pixel_head import PATCH, PixelHead, PixelHeadX2, load_pixel_head
from library.anima.pixel_head_train import (
    Window,
    X2TrainConfig,
    X2Trainer,
    cosine_lr,
    crop_margin,
    dc_loss,
    load_x2flex,
    pick_window,
    rec_loss,
    save_x2_head,
)

D_FEAT = 32


def _head() -> PixelHeadX2:
    torch.manual_seed(0)
    base = PixelHead(D_FEAT, 64, width=16, depth=3, ls_init=1.0, use_skip=False, up=1)
    return PixelHeadX2(base, depth2=1)


class _StubDisc:
    """Counts calls; the generator term is a plain function of the crop."""

    def __init__(self):
        self.gen_calls = self.steps = 0

    def gen_term(self, fake, real):
        self.gen_calls += 1
        assert fake.shape == real.shape and fake.shape[-1] == 64
        return (fake - real).abs().mean()

    def step(self):
        self.steps += 1
        return {"d_margin": 0.0}


def test_window_margin_only_inside_the_image():
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        w = pick_window(20, 10, window=6, margin=2, rng=rng)
        assert (w.h, w.w) == (10, 10)
        assert w.mar[0] == (0 if w.y0 == 0 else 2)
        assert w.mar[1] == (0 if w.y0 + w.h == 20 else 2)
        assert w.mar[2:] == (0, 0)  # the window spans the grid's width: both are edges
        seen.add(w.mar[:2])
    assert seen == {(0, 2), (2, 2), (2, 0)}
    # a grid smaller than the window: the whole grid, nothing left out
    assert pick_window(5, 4, 6, 2, rng) == Window(0, 0, 5, 4, (0, 0, 0, 0))


def test_window_slices_match_the_head_output():
    w = Window(y0=2, x0=1, h=6, w=5, mar=(3, 0, 0, 3))
    grid = torch.zeros(10, 8, D_FEAT)
    img2 = torch.zeros(3, 10 * 2 * PATCH, 8 * 2 * PATCH)
    assert w.tokens(grid).shape == (6, 5, D_FEAT)
    win2 = w.pixels(img2, 2 * PATCH)
    assert win2.shape == (3, 6 * 2 * PATCH, 5 * 2 * PATCH)
    assert crop_margin(win2, w.mar, 2 * PATCH).shape == (3, 3 * 32, 2 * 32)


def test_loss_terms():
    torch.manual_seed(0)
    a = torch.rand(3, 64, 64) * 2 - 1
    loss, mse = rec_loss(a, a)
    assert loss.item() == 0 and mse.item() == 0
    b = a + 0.1
    loss, mse = rec_loss(b, a)  # a constant offset: no gradient-difference part
    assert torch.isclose(mse, torch.tensor(0.01), atol=1e-6)
    assert torch.isclose(loss, torch.tensor(0.5 * 0.1 + 5 * 0.01), atol=1e-5)
    # dc_loss reads block means: a zero-mean pattern inside each block is invisible
    checker = torch.ones(3, 64, 64)
    checker[:, ::2, ::2] = -1
    checker[:, 1::2, 1::2] = -1
    assert dc_loss(a + 0.2 * checker, a, 32).item() < 1e-6
    assert torch.isclose(dc_loss(b, a, 32), torch.tensor(0.1), atol=1e-6)


def test_cosine_lr():
    assert cosine_lr(0, 1000, 200) == 1 / 200
    assert cosine_lr(199, 1000, 200) > 0.9
    assert abs(cosine_lr(500, 1000, 200) - 0.5) < 1e-9
    assert cosine_lr(999, 1000, 200) < 1e-4


def test_trainer_gan_start_and_untouched_x1_layer():
    head = _head()
    cfg = X2TrainConfig(
        rollouts=3,
        opt_steps=2,
        replay=4,
        window=4,
        margin=1,
        warmup=1,
        lambda_dino=0.0,
        pegan_ratio=2.0,
        pegan_start=1,
        pegan_lambda_max=1.0,
        pegan_crop=64,
    )
    disc = _StubDisc()
    tr = X2Trainer(head, cfg, disc=disc)
    x1_out = head.base.out.weight.detach().clone()
    x2_out = head.out2.weight.detach().clone()
    rng = random.Random(0)
    gh = gw = 8
    for r in range(cfg.rollouts):
        f, lat = torch.randn(gh, gw, D_FEAT), torch.randn(gh, gw, 64)
        target2 = torch.rand(3, gh * 2 * PATCH, gw * 2 * PATCH) * 2 - 1
        tr.push(f, lat, target2, pick_window(gh, gw, cfg.window, cfg.margin, rng))
        log = tr.update(r)
        if r == 0:  # before the start: no term, no discriminator update
            assert "pegan" not in log and disc.gen_calls == 0 and disc.steps == 0
        else:  # one eager pass sets the weight, then one term + one update per step
            assert disc.gen_calls == r * (cfg.opt_steps + 1)
            assert disc.steps == r * cfg.opt_steps
            assert 0 < log["lam_pegan"] <= cfg.pegan_lambda_max
    assert tr.step == cfg.rollouts * cfg.opt_steps
    assert not torch.equal(head.out2.weight, x2_out)
    # scale=2 training: the ×1 output layer takes no gradient and no weight decay
    assert torch.equal(head.base.out.weight, x1_out)


def test_save_load_round_trip(tmp_path):
    head = _head().eval()
    path = tmp_path / "head_x2.safetensors"
    save_x2_head(head, path, {"sigma": "0.0577-0.1364", "pegan_ratio": 2.0})
    back = load_pixel_head(path)
    assert isinstance(back, PixelHeadX2)
    assert back.sigma_ranges == ((0.0577, 0.1364),)
    assert back.meta["pegan_ratio"] == "2.0" and back.meta["x2_depth"] == "1"
    f, lat = torch.randn(3, 4, D_FEAT), torch.randn(3, 4, 64)
    with torch.no_grad():
        assert torch.equal(back(f, lat, 2), head(f, lat, 2))


def test_save_without_x1_output(tmp_path):
    from library.anima.pixel_head import drop_x1_output, head_scales

    head = drop_x1_output(_head().eval())
    path = tmp_path / "head_x2.safetensors"
    save_x2_head(head, path, {"sigma": "0.0577-0.1364"})
    back = load_pixel_head(path)
    assert back.meta["scales"] == "2" and head_scales(back) == (2,)
    assert not any(
        k.startswith(("base.out.", "base.norm_out.")) for k in back.state_dict()
    )
    f, lat = torch.randn(3, 4, D_FEAT), torch.randn(3, 4, 64)
    with torch.no_grad():
        assert torch.equal(back(f, lat, 2), head(f, lat, 2))


def test_load_x2flex(tmp_path):
    npz = tmp_path / "a_1024x768_anima.npz"
    npz.touch()
    row = {
        "stem": "a",
        "x1": str(tmp_path / "x1" / "a.png"),
        "x2": str(tmp_path / "x2" / "a.png"),
        "npz": str(npz),
        "te": str(tmp_path / "a_anima_te.safetensors"),
        "kind": "down",
    }
    rows = [{**row, "split": "train"}, {**row, "split": "text"}, {**row, "split": "x"}]
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps(rows))
    data = load_x2flex(man)
    assert [len(data[k]) for k in ("train", "test", "text")] == [1, 0, 1]
    assert data["train"][0].tokens == (1024 // PATCH) * (768 // PATCH)


def test_noise_inputs(tmp_path):
    """Zero-init noise inputs leave the output as it was; trained ones make two draws
    differ, ``noise=False`` leaves the draw out, and the file carries them."""
    head = _head().eval()
    f, lat = torch.randn(3, 4, D_FEAT), torch.randn(3, 4, 64)
    with torch.no_grad():
        plain = head(f, lat, 2)
        head.add_noise(8)
        assert torch.equal(head(f, lat, 2), plain)
        for m in head.noise_in:
            torch.nn.init.normal_(m.weight, std=0.5)
        a, b = head(f, lat, 2), head(f, lat, 2)
        assert not torch.equal(a, b)
        assert torch.equal(head(f, lat, 2, noise=False), plain)
    path = tmp_path / "head_x2.safetensors"
    save_x2_head(head, path, {"sigma": "0.0577-0.1364"})
    back = load_pixel_head(path)
    assert back.noise == 8 and back.meta["x2_noise"] == "8"
    assert torch.equal(back.noise_in[0].weight, head.noise_in[0].weight)


def test_overlap_output(tmp_path):
    """The overlapping output: the window's shifts add to 1 up to the image rim, a
    converted head whose patches are flat reproduces the old output, and the file loads
    back with the 16×16 layer."""
    head = _head().eval()
    f, lat = torch.randn(3, 4, D_FEAT), torch.randn(3, 4, 64)
    with torch.no_grad():
        # a flat colour per position and channel: old and converted outputs agree
        # wherever the four blended positions do, so compare on a uniform image
        head.out2.weight.zero_()
        head.out2.bias.copy_(torch.tensor([0.3, -0.2, 0.1]).repeat(64))
        plain = head(f, lat, 2)
        head.to_overlap()
        over = head(f, lat, 2)
    assert head.out2.out_features == 16 * 16 * 3 and over.shape == plain.shape
    assert torch.allclose(over, plain, atol=1e-5)
    path = tmp_path / "head_x2.safetensors"
    save_x2_head(head, path, {"sigma": "0.0577-0.1364"})
    back = load_pixel_head(path)
    assert back.overlap and back.meta["x2_overlap"] == "1"
    with torch.no_grad():
        assert torch.equal(back(f, lat, 2), over)


def test_overlap_has_no_patch_step():
    """Patches that are flat but differ per position: the 8×8 layer steps by the full
    difference at a patch border, the overlapping one spreads it over the cell."""
    torch.manual_seed(0)
    head = _head().eval()
    f, lat = torch.randn(4, 4, D_FEAT), torch.randn(4, 4, 64)

    def max_step(img):
        return (img[:, :, 1:] - img[:, :, :-1]).abs().max().item()

    with torch.no_grad():
        colour = torch.randn(3, head.out2.in_features) * 0.05  # one row per channel
        head.out2.weight.copy_(colour.repeat(64, 1))
        head.out2.bias.zero_()
        steps = max_step(head(f, lat, 2))
        head.to_overlap()
        smooth = max_step(head(f, lat, 2))
    # the sin² window moves at most π/16 of a difference per pixel
    assert steps > 0 and smooth < 0.25 * steps, (steps, smooth)
