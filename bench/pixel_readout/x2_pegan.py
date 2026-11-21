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

Cost per update: the PE tower runs on the fake crop (with grad) and the real crop in
``gen_term``; ``step`` reuses those tokens for the discriminator's loss and runs the tower
once more, on the two noised crops as one batch (``pegan_speed.py`` times it against the
four separate forwards it replaced). ``compile=True`` compiles the tower's forward.
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
        compile: bool = False,
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
        # (fake, real[, fake taps, real taps]): the crops of the last gen_term
        self._stash: tuple | None = None
        self._taps_c = torch.compile(self._taps_raw) if compile else None

    def _taps_raw(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``forward_features`` up to the last tap, written out so it compiles as one
        graph; the grid's RoPE table is set by the caller (``taps``)."""
        pe = self.pe
        b = x.shape[0]
        gh, gw = x.shape[-2] // pe.patch_size, x.shape[-1] // pe.patch_size
        x = pe.conv1(x).permute(0, 2, 3, 1).reshape(b, -1, pe.width)
        if pe.use_cls_token:
            x = torch.cat([pe.class_embedding.view(1, 1, -1).expand(b, -1, -1), x], 1)
        if pe.use_abs_posemb:
            x = x + pe._sample_abs_posemb(gh, gw)
        x = pe.ln_pre(x)
        outs = []
        for i, blk in enumerate(pe.transformer.resblocks):
            x = blk(x)
            if i in self.layers:
                outs.append(x)
            if i == self._last:
                break
        return outs

    def taps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B, 3, c, c) in [-1, 1] → the PE tokens after each tapped block."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self._taps_c is not None:
                p = self.pe.patch_size
                if self.pe.use_rope2d:
                    self.pe.rope.update_grid(
                        x.device, x.shape[-2] // p, x.shape[-1] // p
                    )
                return self._taps_c(x)
            self._taps.clear()
            self.pe.forward_features(
                x, layer_idx=self._last - self.pe.transformer.layers
            )
            return [self._taps[i] for i in self.layers]

    def heads_on(self, taps: list[torch.Tensor], gh: int, gw: int) -> torch.Tensor:
        outs = []
        for tok, head in zip(taps, self.heads):
            outs.append(head(tok[:, 1:] if self.pe.use_cls_token else tok, gh, gw))
        return torch.cat(outs, 1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, c, c) in [-1, 1] → (B, n_heads · tokens) logits."""
        p = self.pe.patch_size
        return self.heads_on(self.taps(x), x.shape[-2] // p, x.shape[-1] // p)

    def gen_term(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        p = self.pe.patch_size
        gh, gw = fake.shape[-2] // p, fake.shape[-1] // p
        self.heads.requires_grad_(False)
        tf = self.taps(fake)
        lf = self.heads_on(tf, gh, gw)
        with torch.no_grad():
            tr = self.taps(real)
            lr_ = self.heads_on(tr, gh, gw)
        self.heads.requires_grad_(True)
        self._stash = (fake.detach(), real.detach(), [t.detach() for t in tf], tr)
        return F.softplus(-(lf - lr_)).mean()

    def step(self) -> dict[str, float]:
        assert self._stash is not None, "gen_term before step"
        fake, real, *cached = self._stash
        p = self.pe.patch_size
        gh, gw = fake.shape[-2] // p, fake.shape[-1] // p
        self.opt.zero_grad(set_to_none=True)
        if cached:  # the tower is frozen: gen_term's tokens are this step's
            tf, tr = cached
            lr_, lf = self.heads_on(tr, gh, gw), self.heads_on(tf, gh, gw)
        else:
            lr_, lf = self.logits(real), self.logits(fake)
        d = F.softplus(-(lr_ - lf)).mean()
        loss = d
        r1 = r2 = torch.zeros((), device=fake.device)
        if self.r_weight > 0:
            xn = torch.cat(
                [
                    real + self.r_sigma * torch.randn_like(real),
                    fake + self.r_sigma * torch.randn_like(fake),
                ]
            )
            ln = self.logits(xn)
            nb = real.shape[0]
            r1 = (ln[:nb] - lr_).pow(2).mean()
            r2 = (ln[nb:] - lf).pow(2).mean()
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
