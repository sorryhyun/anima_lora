"""PE-Spatial discriminator for the ×2 head (x2_plan.md, arm B+GAN).

FastFrames' form (``plan.md``, "pixel losses on the head"): a frozen ViT, small trainable
heads on its tokens at three depths, a relativistic loss and an input-smoothness penalty.
Here the ViT is the local PE-Spatial-B16-512 (12 layers; taps after layers 4 / 8 / 12 by
default), the inputs are paired ``--pegan_crop`` crops of the ×2 output and of the real 2×
image at the same place, and each head writes one logit per PE token.

Loss (paired relativistic, per token, averaged over heads):
``L_D = softplus(−(D(real) − D(fake)))``, ``L_G = softplus(−(D(fake) − D(real)))``.
The smoothness penalty is the noise-perturbation form of R1 / R2 (no double backward):
``λ_r · mean((D(x + σε) − D(x))²)`` on the real crop (R1) and on the fake crop (R2).

The generator's gradient reaches the head through the frozen PE tower (input grad only).
Images enter in [-1, 1], which is PE's own normalisation of [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenHead(nn.Module):
    """(B, N, C) PE tokens on a (gh, gw) grid → (B, gh·gw) logits."""

    def __init__(self, in_ch: int = 768, width: int = 256):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv2d(in_ch, width, 1),
            nn.GroupNorm(32, width),
            nn.LeakyReLU(0.2),
        )
        self.res = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(32, width),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, tok: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        x = tok.transpose(1, 2).reshape(tok.shape[0], -1, gh, gw).float()
        x = self.inp(x)
        x = x + self.res(x)
        return self.out(x).flatten(1)


class PEDisc:
    """Frozen PE-Spatial-B16 + trainable token heads.

    ``gen_term(fake, real)`` returns the generator's relativistic loss on one paired crop
    and keeps both crops detached; ``step()`` then makes one discriminator update on them.
    """

    def __init__(
        self,
        device,
        layers: tuple[int, ...] = (3, 7, 11),
        width: int = 256,
        lr: float = 1e-4,
        r_weight: float = 0.1,
        r_sigma: float = 0.01,
    ):
        from library.models.pe import load_pe_spatial

        self.device = device
        self.pe = load_pe_spatial(device, dtype=torch.bfloat16)
        self.pe.eval().requires_grad_(False)
        self.layers = tuple(layers)
        self.r_weight, self.r_sigma = r_weight, r_sigma
        self.heads = nn.ModuleList(
            TokenHead(self.pe.width, width) for _ in self.layers
        ).to(device)
        self.opt = torch.optim.AdamW(
            self.heads.parameters(), lr=lr, betas=(0.0, 0.99), weight_decay=0.0
        )
        self._taps: dict[int, torch.Tensor] = {}
        for i in self.layers:
            self.pe.transformer.resblocks[i].register_forward_hook(
                lambda _m, _inp, out, i=i: self._taps.__setitem__(i, out)
            )
        self._last = max(self.layers)
        self._stash: tuple[torch.Tensor, torch.Tensor] | None = None

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, c, c) in [-1, 1] → (B, n_heads · tokens) logits."""
        p = self.pe.patch_size
        gh, gw = x.shape[-2] // p, x.shape[-1] // p
        self._taps.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.pe.forward_features(x, layer_idx=self._last - self.pe.transformer.layers)
        outs = []
        for i, head in zip(self.layers, self.heads):
            tok = self._taps[i][:, 1:] if self.pe.use_cls_token else self._taps[i]
            outs.append(head(tok, gh, gw))
        return torch.cat(outs, 1)

    def gen_term(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        self.heads.requires_grad_(False)
        lf = self.logits(fake)
        with torch.no_grad():
            lr_ = self.logits(real)
        self.heads.requires_grad_(True)
        self._stash = (fake.detach(), real.detach())
        return F.softplus(-(lf - lr_)).mean()

    def step(self) -> dict[str, float]:
        assert self._stash is not None, "gen_term before step"
        fake, real = self._stash
        self.opt.zero_grad(set_to_none=True)
        lr_, lf = self.logits(real), self.logits(fake)
        d = F.softplus(-(lr_ - lf)).mean()
        loss = d
        r1 = r2 = torch.zeros((), device=fake.device)
        if self.r_weight > 0:
            r1 = (self.logits(real + self.r_sigma * torch.randn_like(real)) - lr_).pow(2).mean()
            r2 = (self.logits(fake + self.r_sigma * torch.randn_like(fake)) - lf).pow(2).mean()
            loss = d + self.r_weight * (r1 + r2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.heads.parameters(), 1.0)
        self.opt.step()
        return {
            "d_loss": d.item(),
            "d_margin": (lr_ - lf).mean().item(),
            "d_r1": r1.item(),
            "d_r2": r2.item(),
        }
