"""The RGBA head: its init from an RGB head, its trainer, and the decode's fringe step.

One ×1 head writes RGB and 2α − 1 (``PixelHead(channels=4)``,
``project/pixel_decoder/rgba_plan.md``). It starts from an RGB head: trunk and the RGB rows
of the output layer copied, the α rows fresh. Training holds RGB to the frozen init head's
RGB on the same features and fits α to the cutout's.

The head's RGB is the composite's colour: in a pixel with 0 < α < 1 it still holds the
background. :func:`straight_rgba` takes it out at decode (numpy, float32 in [0, 1]):
:func:`estimate_bg`, :func:`unmix`, then P / α. The data, the cache and the eval are the
training scripts': ``scripts/pixel_head/{cache_alpha,train_rgba}.py`` with
``_alpha_{data,cache,read}.py`` beside them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from library.anima.pixel_head import PixelHead
from library.anima.pixel_head_train import rec_loss
from library.anima.pixel_head_train_x1 import X1TrainConfig, X1Trainer, save_x1_head


def rgba_from_rgb(base: PixelHead, seed: int = 0) -> PixelHead:
    """A four-channel head on ``base``'s weights: its RGB output is ``base``'s, the α
    rows of the output layer fresh from ``seed`` (``nn.Linear``'s own init)."""
    if not base.up or base.channels != 3:
        raise ValueError("init: an RGB head with an upsample stage")
    head = PixelHead(
        base.proj.in_features,
        base.lat_embed.in_features,
        base.proj.out_features,
        len(base.blocks) + len(base.up_blocks),
        up=base.up,
        channels=4,
    ).to(base.proj.weight.device)
    kept = {k: v for k, v in base.state_dict().items() if not k.startswith("out.")}
    missing, unexpected = head.load_state_dict(kept, strict=False)
    assert set(missing) == {"out.weight", "out.bias"} and not unexpected
    torch.manual_seed(seed)
    fresh = nn.Linear(base.out.in_features, base.patch**2).to(head.out.weight.device)
    with torch.no_grad():  # out rows are (py, px, c), c fastest
        w = head.out.weight.view(base.patch**2, 4, -1)
        b = head.out.bias.view(base.patch**2, 4)
        w[:, :3] = base.out.weight.view(base.patch**2, 3, -1)
        b[:, :3] = base.out.bias.view(base.patch**2, 3)
        w[:, 3] = fresh.weight
        b[:, 3] = fresh.bias
    return head.requires_grad_(True).train()


class RGBATrainer(X1Trainer):
    """``X1Trainer`` on (4, H, W) targets: RGB = the init head's RGB on the same features,
    the fourth channel 2α − 1. Loss: ``lambda_rgb`` · rec(RGB) + rec(α) + λ_dino · DINO
    on α read as three channels. ``rec_log`` keeps the last step's RGB and α terms."""

    def __init__(
        self,
        head: PixelHead,
        cfg: X1TrainConfig,
        lambda_rgb: float = 1.0,
        dino=None,
        compile: bool = False,
    ):
        assert head.channels == 4, "RGBATrainer: a four-channel head"
        assert cfg.pegan_ratio == 0, "RGBATrainer: no GAN term"
        super().__init__(head, cfg, dino=dino, compile=compile)
        self.lambda_rgb = lambda_rgb
        self.rec_log: dict[str, float] = {}

    def _rec(self, pred, tgt):
        rgb, rgb_mse = rec_loss(pred[:3], tgt[:3])
        a, a_mse = rec_loss(pred[3:], tgt[3:])
        self.rec_log = {
            "rgb": rgb.item(),
            "rgb_mse": rgb_mse.item(),
            "alpha": a.item(),
        }
        return self.lambda_rgb * rgb + a, a_mse

    def _dino_term(self, pred, tgt, rng, compiled: bool = True):
        a, b = (x[3:].repeat(3, 1, 1) for x in (pred, tgt))
        return super()._dino_term(a, b, rng, compiled)

    def update(self) -> dict[str, float]:
        out = super().update()
        return {**out, **self.rec_log}


def save_rgba_head(head: PixelHead, path: str | Path, stamp: dict) -> None:
    save_x1_head(head, path, {**stamp, "channels": 4, "target": "rgba"})


# ---------------------------------------------------------------- alpha → colour


def affine_grid(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([np.ones_like(x), x / (w - 1) - 0.5, y / (h - 1) - 0.5], -1)


def fit_field(c: np.ndarray, mask: np.ndarray):
    """Robust affine colour fit on ``mask`` → (coef, inlier share, residual std)."""
    X, C = affine_grid(*mask.shape)[mask], c[mask]
    if len(C) > 40000:
        pick = np.linspace(0, len(C) - 1, 40000).astype(int)
        X, C = X[pick], C[pick]
    keep = np.abs(C - np.median(C, 0)).max(1) < 40 / 255
    if keep.mean() < 0.2:
        return None
    for th in (30, 18, 12, 8):
        k = np.linalg.lstsq(X[keep], C[keep], rcond=None)[0]
        new = np.abs(C - X @ k).max(1) < th / 255
        if new.mean() < 0.2:
            break
        keep = new
    res = float((C[keep] - X[keep] @ k).std())
    return k.astype(np.float32), float(keep.mean()), res


def border_mask(h: int, w: int) -> np.ndarray:
    b = max(4, round(0.025 * max(h, w)))
    m = np.ones((h, w), bool)
    m[b:-b, b:-b] = False
    return m


def estimate_bg(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """An affine B from the pixels the alpha calls background."""
    cs, sub = c[2::4, 2::4], a[2::4, 2::4]  # an affine field needs no full grid
    mask = sub <= 0.02
    if mask.sum() < 256:
        mask = border_mask(*sub.shape)
    fit = fit_field(cs, mask)
    if fit is None:
        return np.broadcast_to(np.median(cs[mask], 0), c.shape).astype(np.float32)
    return (affine_grid(*a.shape) @ fit[0]).clip(0, 1).astype(np.float32)


def unmix(c: np.ndarray, a: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """P = clamp(C − (1 − α)·B, 0, α), the premultiplied foreground."""
    a = a[..., None]
    return np.clip(c - (1 - a) * bg, 0, a).astype(np.float32)


def straight_rgba(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The head's colour ``c`` (H, W, 3) and α (H, W) → straight RGBA (H, W, 4), the
    background taken out of the partly transparent pixels."""
    p = unmix(c, a, estimate_bg(c, a))
    f = np.divide(p, a[..., None], out=np.zeros_like(p), where=a[..., None] > 0)
    return np.dstack([f.clip(0, 1), a]).astype(np.float32)
