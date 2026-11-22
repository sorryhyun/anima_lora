"""The alpha head: ``PixelHead``'s form with one output channel, and its trainer.

A file of its own beside the RGB head (``project/pixel_decoder/rgba_plan.md``): the RGB
base's weights with a fresh 8 × 8 × 1 output layer, the whole head trained on 2α − 1.
``library/anima/pixel_head.py`` is vendored into the ComfyUI node as it is, so the
one-channel form lives here until the alpha head ships.

Cache: ``pixel_alpha_cache.py``. Entry points: ``scripts/pixel_head/cache_alpha.py``
(DiT, VAE, TE → features on disk), ``scripts/pixel_head/train_alpha.py`` (head alone).
"""

from __future__ import annotations

from pathlib import Path

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
