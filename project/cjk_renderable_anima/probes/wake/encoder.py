"""W2d: amortized glyph encoder ``g(glyph render) → Δ_row`` (arm ``encoder``)."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def row_texts(tok, pack, rows):
    """ext row → the piece text it stands for (Qwen piece, char row or symbol
    row); rows the pack cannot name are left out (zero delta)."""
    inv_q = {int(v): int(k) for k, v in pack.mapping["qwen"].items()}
    inv_c = {int(v): k for k, v in pack.mapping.get("char", {}).items()}
    inv_s = {int(v): k for k, v in pack.mapping.get("sym_char", {}).items()}
    qtok = tok.qwen3_tokenizer
    out = {}
    for r in rows:
        r = int(r)
        if r in inv_q:
            t = qtok.decode([inv_q[r]]).strip()
        elif r in inv_c:
            t = inv_c[r]
        elif r in inv_s:
            t = inv_s[r]
        else:
            continue
        if t:
            out[r] = t
    return out


def glyph_bank(texts, fonts, size: int):
    """uint8 (rows, fonts, size, size) grayscale renders, ink 0 on 255: the
    encoder's input, one render per font so the font is drawn per step."""
    from PIL import Image, ImageDraw, ImageFont

    bank = np.full((len(texts), len(fonts), size, size), 255, dtype=np.uint8)
    for fi, fp in enumerate(fonts):
        cache: dict = {}
        for ri, t in enumerate(texts):
            n = max(1, len(t))
            fs = int(size * 0.78 / n) if n > 1 else int(size * 0.78)
            font = cache.get(fs)
            if font is None:
                font = cache[fs] = ImageFont.truetype(fp, fs, index=0)
            im = Image.new("L", (size, size), 255)
            d = ImageDraw.Draw(im)
            left, top, right, bottom = d.textbbox((0, 0), t, font=font)
            d.text(
                ((size - (right - left)) / 2 - left, (size - (bottom - top)) / 2 - top),
                t,
                fill=0,
                font=font,
            )
            bank[ri, fi] = np.array(im)
    return torch.from_numpy(bank)


def glyph_batch(
    bank, device, rng: random.Random, shift: int = 6, fonts=None, font_mean=False
):
    """One render per row: a random font (or ``fonts`` per row), or with
    ``font_mean`` the mean render over every font (a font-free glyph
    descriptor — attempt 7 showed the output tracking font 8× more than
    glyph), plus one random shift for the batch, as ink in [0, 1].

    ``rng`` is consumed only by a random font draw and a nonzero shift."""
    R, F = bank.shape[:2]
    if font_mean:
        x = bank.to(device).float().mean(dim=1).div_(255.0)
    else:
        f = (
            torch.tensor(fonts)
            if fonts is not None
            else torch.tensor([rng.randrange(F) for _ in range(R)])
        )
        x = bank[torch.arange(R), f].to(device).float().div_(255.0)
    x = 1.0 - x  # ink 1, paper 0
    if shift:
        dx, dy = rng.randint(-shift, shift), rng.randint(-shift, shift)
        x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    return x.unsqueeze(1)


def reference_batch(bank, device, font_mean: bool):
    """The fixed reference render — font 0 (or the font mean), no shift."""
    return glyph_batch(
        bank,
        device,
        random.Random(0),
        shift=0,
        fonts=[0] * bank.shape[0],
        font_mean=font_mean,
    )


class GlyphEncoder(nn.Module):
    """``g(glyph render) → Δ_row`` in row-norm units. Small CNN + MLP, last
    layer zero-init (step 0 = pack rows), output split into an identity part
    and one layout vector:

        Δ_r = (d_r − mean_rows d) + c

    Three attempts taught the shape. (1) With a zero-init last layer every
    one of the ``hidden`` weights feeding an output coordinate steps by
    ``lr`` in the same direction, so the output moves ``hidden × lr`` per
    step → ``out_scale`` 1/64. (2) Any component every row shares gets the
    *summed* gradient of every ext token in the batch — a direction
    consistent enough that Adam marches at full lr forever (a shared bias
    reached 36× row norm, every row identical). (3) A per-row output-norm
    cap does not help: at the cap the output is ``d / ‖d‖``, the internal
    ``d`` keeps growing along the common direction and the per-glyph part is
    divided by it (spread 0.000 for 750 steps).

    So the common mode is *projected out* of the CNN's output — centring
    across the full row table every step removes the common-mode gradient
    from the shared weights, leaving the per-glyph part with the same
    inconsistent gradients free rows had (they saturated at ~1× on every
    rows arm) — and the layout mode ("big glyph on a blank canvas", the
    direction every rows arm converged to) lives in one free vector ``c`` at
    the rows lr, bounded on the **parameter** after each optimizer step
    (``clamp_common``), never on the output."""

    def __init__(
        self,
        dim: int,
        width: int = 32,
        out_scale: float = 1.0 / 64,
        common_cap: float = 0.75,
        pool: str = "spatial",
        glyph_size: int = 96,
        head_init: str = "zero",
    ):
        super().__init__()
        ch = [1, width, width * 2, width * 4, width * 8]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1), nn.GELU()]
        grid = (glyph_size + 15) // 16
        feat = ch[-1]
        self.conv = nn.Sequential(*layers)
        self.pool = pool
        # ``mean``: global mean pool — channel statistics only, which carry
        # ink mass / stroke weight (font) and not arrangement (glyph);
        # ``spatial``: flatten the final grid and project, so the arrangement
        # survives
        self.proj = nn.Linear(feat * grid * grid, feat) if pool == "spatial" else None
        self.norm = nn.LayerNorm(feat)
        self.head = nn.Sequential(nn.Linear(feat, 512), nn.GELU(), nn.Linear(512, dim))
        # rank lever (2026-09-14): attempt 10's zero-init last layer grew as
        # one outer product (PR 2.8, table PR 1.0) — with Adam a consistent
        # gradient direction on shared weights marches while the per-glyph
        # tail random-walks. ``random`` keeps the default full-rank init; the
        # train stage rescales it to --init_spread row norms on the reference
        # render
        if head_init == "zero":
            nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.common = nn.Parameter(torch.zeros(dim))
        self.out_scale = out_scale
        self.common_cap = common_cap

    def features(self, x):
        """pooled conv features, pre-LayerNorm (the collapse diagnostic reads
        their spread across rows)"""
        h = self.conv(x)
        if self.pool == "spatial":
            return self.proj(h.flatten(1))
        return h.mean(dim=(2, 3))

    def identity(self, x):
        """centred per-glyph part only (mean over the rows passed in — call
        with the full table)"""
        h = self.norm(self.features(x))
        d = self.head(h) * self.out_scale
        return d - d.mean(dim=0, keepdim=True)

    def forward(self, x):
        return self.identity(x) + self.common

    @torch.no_grad()
    def clamp_common(self):
        n = self.common.norm()
        if n > self.common_cap:
            self.common.mul_(self.common_cap / n)

    def enc_params(self):
        return [p for n, p in self.named_parameters() if n != "common"]
