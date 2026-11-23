"""The pixel-DINO aux LoRA term (library/training/pixel_dino.py): the surrogate's value and
gradient, the gradient-ratio rule, the window picker, the composer wiring. CPU, with a
linear stand-in for the VAE decode and MSE for the DINO distance."""

import argparse
import random

import torch
import torch.nn.functional as F

from library.training.losses import LossContext, build_loss_composer
from library.training.pixel_dino import (
    fm_grad_norm,
    pick_window,
    surrogate,
    window_term,
)


def _decode(lat):  # (C, h, w) -> (3, 8h, 8w), linear in the latent
    return F.interpolate(lat[None, :3], scale_factor=8, mode="bilinear")[0]


def _mse(a, b):
    return ((a - b) ** 2).mean()


def test_surrogate_carries_value_and_gradient():
    torch.manual_seed(0)
    lat, noise = torch.randn(16, 12, 12), torch.randn(16, 12, 12)
    sigma = 0.3
    x_t = (1 - sigma) * lat + sigma * noise
    v = (noise - lat + 0.1 * torch.randn(16, 12, 12)).requires_grad_(True)
    mar = (16, 0, 0, 16)
    value, g = window_term(v, x_t, lat, sigma, _decode, _mse, mar)

    def direct(vv):
        img, tgt = _decode(x_t - sigma * vv), _decode(lat)
        return _mse(img[:, 16:, :-16], tgt[:, 16:, :-16])

    (g_ref,) = torch.autograd.grad(direct(v), v)
    assert torch.allclose(g, g_ref, atol=1e-7)
    assert torch.allclose(value, direct(v).detach())

    lam = 0.5 * fm_grad_norm(v.detach(), noise, lat) / g.norm().item()
    term = surrogate(v, value, g, lam)
    assert torch.allclose(term, lam * value)
    (g_term,) = torch.autograd.grad(term, v)
    assert torch.allclose(g_term, lam * g)
    ratio = g_term.norm().item() / fm_grad_norm(v.detach(), noise, lat)
    assert abs(ratio - 0.5) < 1e-5


def test_pick_window_margins():
    rng = random.Random(0)
    for _ in range(50):
        y0, x0, h, w, mar = pick_window(40, 28, 24, 4, rng)
        assert (h, w) == (32, 28) and x0 == 0
        assert mar[2] == mar[3] == 0  # the whole width: both sides are image edges
        assert mar[0] == (0 if y0 == 0 else 4) and mar[1] == (0 if y0 + h == 40 else 4)


def test_composer_adds_the_term():
    args = argparse.Namespace(pixel_dino_ratio=0.5, loss_type="l2", masked_loss=False)
    composer = build_loss_composer(args, network=object())
    assert "pixel_dino" in composer.active_losses
    pred, target = torch.zeros(2, 4, 8, 8), torch.ones(2, 4, 8, 8)

    def ctx(aux):
        return LossContext(
            args=args,
            batch={},
            model_pred=pred,
            target=target,
            timesteps=torch.tensor([0.2, 0.9]),
            weighting=None,
            huber_c=None,
            loss_weights=torch.ones(2),
            network=object(),
            aux=aux,
        )

    base = composer.compose(ctx({}))
    with_term = composer.compose(ctx({"pixel_dino": torch.tensor(0.25)}))
    assert torch.allclose(with_term - base, torch.tensor(0.25))
    off = build_loss_composer(argparse.Namespace(), network=object())
    assert "pixel_dino" not in off.active_losses
