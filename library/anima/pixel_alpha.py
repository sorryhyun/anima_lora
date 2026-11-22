"""The alpha head: ``PixelHead``'s form with one output channel, its trainer and α → colour.

A file of its own beside the RGB head (``project/pixel_decoder/rgba_plan.md``): the RGB
base's weights with a fresh 8 × 8 × 1 output layer, the whole head trained on 2α − 1.
``library/anima/pixel_head.py`` is vendored into the ComfyUI node as it is, so the
one-channel form lives here until the alpha head ships.

Decode (numpy, float32 in [0, 1]): :func:`estimate_bg`, then :func:`unmix`. The data,
the cache and the eval are the training scripts' own: ``scripts/pixel_head/cache_alpha.py``
and ``train_alpha.py``, with ``_alpha_{data,cache,read}.py`` beside them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from library.anima.pixel_head import LAT_PATCH_DIM, PixelHead, parse_sigma_ranges
from library.anima.pixel_head_train_x1 import X1Trainer, save_x1_head


class AlphaHead(PixelHead):
    """The upsampled ``PixelHead`` → (1, H, W) in [-1, 1]."""

    def __init__(self, d_feat: int, d_lat: int, width: int, depth: int, up: int = 1):
        if not up:
            raise ValueError("the alpha head is the upsampled form (up >= 1)")
        super().__init__(d_feat, d_lat, width, depth, use_skip=False, up=up)
        del self.linear
        self.out = nn.Linear(self.out.in_features, self.patch * self.patch)

    @classmethod
    def from_base(cls, base: PixelHead, seed: int = 0) -> "AlphaHead":
        """The RGB head's weights, the output layer fresh from ``seed``."""
        if not base.up or base.use_skip:
            raise ValueError("init: a head with an upsample stage and no skip")
        head = cls(
            base.proj.in_features,
            base.lat_embed.in_features,
            base.proj.out_features,
            len(base.blocks) + len(base.up_blocks),
            up=base.up,
        )
        kept = {
            k: v
            for k, v in base.state_dict().items()
            if not k.startswith(("out.", "linear."))
        }
        missing, unexpected = head.load_state_dict(kept, strict=False)
        assert set(missing) == {"out.weight", "out.bias"} and not unexpected
        torch.manual_seed(seed)
        head.out.reset_parameters()
        return head.to(base.proj.weight.device).requires_grad_(True).train()

    def forward(self, f, lat):  # f (gh, gw, D), lat (gh, gw, 64) → (1, H, W)
        h = (self.proj(f) + self.lat_embed(lat)).permute(2, 0, 1)[None]
        for b in self.blocks:
            h = b(h)
        u = self.up_proj(self.up_norm(h.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        h = F.pixel_shuffle(u, 2)
        for b in self.up_blocks:
            h = b(h)
        h = self.norm_out(h[0].permute(1, 2, 0))
        with torch.autocast(h.device.type, enabled=False):
            tok = self.out(h.float()).permute(2, 0, 1)
            return F.pixel_shuffle(tok[None], self.patch)[0]


class AlphaTrainer(X1Trainer):
    """``X1Trainer`` on a one-channel target: LPIPS reads it as three channels."""

    def _lpips_term(self, pred, tgt, rng, compiled: bool = True):
        a, b = (x.repeat(3, 1, 1) for x in (pred, tgt))
        return super()._lpips_term(a, b, rng, compiled)


def save_alpha_head(head: AlphaHead, path: str | Path, stamp: dict) -> None:
    save_x1_head(head, path, {**stamp, "channels": 1, "target": "alpha"})


def load_alpha_head(path: str | Path, device="cpu") -> AlphaHead:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), "pt") as fh:
        meta = fh.metadata() or {}
        if meta.get("channels") != "1":
            raise ValueError(f"{path}: not an alpha head (no channels = 1 stamp)")
        d_feat = int(fh.get_slice("proj.weight").get_shape()[1])
    head = AlphaHead(
        d_feat,
        LAT_PATCH_DIM,
        int(meta["width"]),
        int(meta["depth"]),
        up=int(meta["up"]),
    )
    head.load_state_dict(load_file(str(path)))
    head.sigma_ranges = parse_sigma_ranges(meta.get("sigma"))
    head.meta = meta
    return head.to(device).eval().requires_grad_(False)


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
