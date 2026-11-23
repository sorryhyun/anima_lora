"""The RGBA head's training: its init from an RGB head, its trainer and its file.

One ×1 head writes RGB and 2α − 1 (``PixelHead(channels=4)``,
``project/pixel_decoder/rgba_plan.md``). It starts from an RGB head: trunk and the RGB rows
of the output layer copied, the α rows fresh. The target is the RGBA image itself: its own
colour where α > 0, the composite (the background it was laid on) where α = 0, and its α
(:meth:`scripts.pixel_head.cache._alpha_data.Cutout.target`). The head's RGB is straight colour,
saved with its α as it is.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn as nn

from library.anima.pixel_head import PixelHead
from scripts.pixel_head.train._common import rec_loss
from scripts.pixel_head.train._x1_trainer import X1TrainConfig, X1Trainer, save_x1_head


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
    """``X1Trainer`` on (4, H, W) targets (RGB, 2α − 1). Loss: ``lambda_rgb`` · rec(RGB)
    + rec(α) + ``lambda_dino_rgb`` · DINO(RGB) + ``lambda_dino_alpha`` · DINO(α read as
    three channels). ``rec_log`` keeps the last step's terms."""

    def __init__(
        self,
        head: PixelHead,
        cfg: X1TrainConfig,
        lambda_rgb: float = 1.0,
        lambda_dino_rgb: float = 331.6,
        lambda_dino_alpha: float = 331.6,
        dino=None,
        compile: bool = False,
    ):
        assert head.channels == 4, "RGBATrainer: a four-channel head"
        assert cfg.pegan_ratio == 0, "RGBATrainer: no GAN term"
        on = lambda_dino_rgb > 0 or lambda_dino_alpha > 0
        # the two DINO weights are applied in _dino_term; the base loop only gates on it
        cfg = dataclasses.replace(cfg, lambda_dino=1.0 if on else 0.0)
        super().__init__(head, cfg, dino=dino, compile=compile)
        self.lambda_rgb = lambda_rgb
        self.lambda_dino_rgb, self.lambda_dino_alpha = (
            lambda_dino_rgb,
            lambda_dino_alpha,
        )
        self.rec_log: dict[str, float] = {}

    def _rec(self, pred, tgt):
        rgb, rgb_mse = rec_loss(pred[:3], tgt[:3])
        a, a_mse = rec_loss(pred[3:], tgt[3:])
        self.rec_log.update(rgb=rgb.item(), rgb_mse=rgb_mse.item(), alpha=a.item())
        return self.lambda_rgb * rgb + a, a_mse

    def _dino_term(self, pred, tgt, rng, compiled: bool = True):
        out = pred.new_zeros(())
        if self.lambda_dino_rgb > 0:
            d = super()._dino_term(pred[:3], tgt[:3], rng, compiled)
            self.rec_log["dino_rgb"] = d.item()
            out = out + self.lambda_dino_rgb * d
        if self.lambda_dino_alpha > 0:
            a, b = (x[3:].repeat(3, 1, 1) for x in (pred, tgt))
            d = super()._dino_term(a, b, rng, compiled)
            self.rec_log["dino_alpha"] = d.item()
            out = out + self.lambda_dino_alpha * d
        return out

    def update(self) -> dict[str, float]:
        out = super().update()
        return {**out, **self.rec_log}


def save_rgba_head(head: PixelHead, path: str | Path, stamp: dict) -> None:
    save_x1_head(head, path, {**stamp, "channels": 4, "target": "rgba"})
