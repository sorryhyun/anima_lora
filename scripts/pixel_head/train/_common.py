"""Pixel-head training pieces every recipe shares (×1, ×2, RGBA) and the bench reads.

- :class:`DiTRollout`, :func:`attach_adapter`, :func:`compile_dit` — the frozen DiT's
  forward at a decode point and what the head reads.
- :func:`rec_loss`, :func:`dc_loss`, :class:`DinoFeat`, :class:`PEDisc` — the loss terms.
- :func:`cosine_lr` — the schedule.
- :func:`psnr`, :func:`tone_stats`, :func:`densest_window`, :func:`zoom_sheet`,
  :func:`pair_sheet` — the read.

The recipes: ``_x1_trainer.py``, ``_x2_trainer.py``, ``_rgba_trainer.py`` beside this file;
the algorithm: ``project/pixel_decoder/README.md``.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from library.anima.pixel_head import PATCH, lat_patches

if TYPE_CHECKING:
    from scripts.pixel_head.train._x2_trainer import X2Item


def npz_tokens(path: str) -> int:
    """DiT tokens of a cached latent, from the ``WxH`` in its file name."""
    w, h = (int(v) for v in path.rsplit("_", 2)[-2].split("x"))
    return (w // PATCH) * (h // PATCH)


def load_image(path: str | Path, device="cpu") -> torch.Tensor:
    """An RGB image file → (3, H, W) float in [-1, 1]."""
    from PIL import Image

    px = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(px).permute(2, 0, 1).to(device).float() / 127.5 - 1


# --------------------------------------------------------------------------- rollout


class DiTRollout:
    """One DiT forward at a decode point → what the head reads and VAE(x̂0).

    ``anima`` is built, frozen and (if at all) compiled by the caller; the hook taps the
    input of ``final_layer.linear``. The noise is a function of (``seed``, ``idx``), so a
    held-out item reads the same x_t on every eval."""

    def __init__(self, anima, vae, device, seed: int = 0):
        self.anima, self.vae, self.device, self.seed = anima, vae, device, seed
        self._tap: dict[str, torch.Tensor] = {}
        anima.final_layer.linear.register_forward_pre_hook(
            lambda _m, inp: self._tap.__setitem__("x", inp[0].detach())
        )

    def load(self, item: X2Item) -> tuple[torch.Tensor, torch.Tensor]:
        from library.io.cache import load_cached_latents, load_cached_text_features

        lat = load_cached_latents(item.npz)[0].unsqueeze(0).to(self.device)
        cross, _ = load_cached_text_features(item.te, variant=0)
        return lat, cross.unsqueeze(0).to(self.device, torch.bfloat16)

    @torch.no_grad()
    def forward(self, lat: torch.Tensor, cross: torch.Tensor, idx: int, sigma: float):
        """→ (features (gh, gw, D), x_t patches (gh, gw, 64), x̂0 (1, C, h, w))."""
        dev = self.device
        g = torch.Generator(device=dev).manual_seed(self.seed * 100003 + idx)
        noise = torch.randn(lat.shape, generator=g, device=dev, dtype=lat.dtype)
        x_t = (1 - sigma) * lat + sigma * noise
        sig = torch.tensor([sigma], device=dev)
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = self.anima(
                x_t.unsqueeze(2).to(torch.bfloat16), sig, cross, padding_mask=pad
            )
        f = self._tap.pop("x")[0, 0]
        x0_hat = x_t - sigma * v.float().squeeze(2)
        return f.float(), lat_patches(x_t).float(), x0_hat

    @torch.no_grad()
    def __call__(self, item: X2Item, idx: int, sigma: float):
        """→ (features, x_t patches, VAE(x̂0) (3, H, W))."""
        f, lat_p, x0_hat = self.forward(*self.load(item), idx, sigma)
        img = self.vae.decode_to_pixels(x0_hat)[0].float().clamp(-1, 1)
        return f, lat_p, img


def attach_adapter(anima, path: str, device):
    """A LoRA-family adapter as hooks on a built DiT (``set_enabled`` switches it).

    Call after ``stream_fp32_`` and before :func:`compile_dit`: each hook captures its
    module's forward when it is applied, so the reverse order would keep the bf16
    linears under the adapter. ``inference.py`` merges a plain LoRA before the stream
    change instead; same function."""
    from library.env import resolve_under_home
    from networks.lora_anima.factory import create_network_from_weights

    path = str(resolve_under_home(path))
    network, _ = create_network_from_weights(
        1.0, path, None, None, anima, for_inference=True
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    print(f"adapter {path}: {network.load_weights(path)}", flush=True)
    network.to(device=device, dtype=torch.bfloat16).requires_grad_(False)
    return network


def compile_dit(anima, seq_bands: list | None = None, tag: str = "") -> None:
    """``compile_blocks`` under a cache dir of its own signature. The persistent inductor
    cache keys on the graph, not the ``mark_dynamic`` range, so a graph compiled for
    another script's token bands would replay its guards here
    (``ConstraintViolationError`` on the first trace). Call after ``stream_fp32_`` and
    any adapter, before the first forward. ``tag`` names a base the signature cannot
    see (``int8``)."""
    from library.runtime.dynamo import compile_signature, isolate_compile_cache

    isolate_compile_cache(
        compile_signature(
            n_token_families=None,
            seq_range=None,
            dynamic_seq=True,
            mode="",
            seq_bands=seq_bands,
        )
        + f";stream={getattr(anima, 'stream', 'bf16')}"
        + (f";{tag}" if tag else "")
    )
    anima.compile_blocks(dynamic_seq=True, seq_bands=seq_bands)


# --------------------------------------------------------------------------- losses


def rec_loss(pred: torch.Tensor, tgt: torch.Tensor):
    """The recipe loss 0.5 L1 + 5 MSE + 0.05 L_grad → (loss, MSE)."""
    l1 = (pred - tgt).abs().mean()
    mse = ((pred - tgt) ** 2).mean()
    gh = (pred.diff(dim=-2) - tgt.diff(dim=-2)).abs().mean()
    gw = (pred.diff(dim=-1) - tgt.diff(dim=-1)).abs().mean()
    return 0.5 * l1 + 5.0 * mse + 0.05 * (gh + gw), mse


def dc_loss(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """L1 on ``k``-px block means (the tone term); both (3, H, W) in [-1, 1]."""
    return F.l1_loss(F.avg_pool2d(a[None], k), F.avg_pool2d(b[None], k))


class DinoFeat:
    """Frozen DINOv3 feature distance on 224-px tiles, the heads' perceptual term: that of
    ViTok-v2 (arXiv:2605.05331 §2.4, App. E), which takes the place of LPIPS there.

    Tiles from the same place of two (3, H, W) images in [-1, 1] go through the tower as
    one batch; the features of ``layers`` are L2-normalised per position and compared by
    MSE, averaged over the layers. A tile's place is drawn per pixel, so the tower's grid
    falls anywhere on the image's.

    A ViT (the paper's ViT-S/16) gives the patch tokens after each block of ``layers``
    (default: blocks 3 / 6 / 9 / 12). A ConvNeXt gives its stage maps (default: all four,
    4 / 8 / 16 / 32 px per position); MSE over elements weighs a stage by 1 / its width,
    so the narrow early stages count most."""

    TILE = 224

    def __init__(
        self,
        device,
        repo: str = "",
        layers: tuple[int, ...] | None = None,
        compile: bool = False,
        net: nn.Module | None = None,
    ):
        if net is None:
            from transformers import AutoModel

            net = AutoModel.from_pretrained(repo)
        self.net = net.to(device).eval().requires_grad_(False)
        cfg = self.net.config
        self.convnext = cfg.model_type == "dinov3_convnext"
        if self.convnext:
            depth, self.patch, self.skip = len(cfg.hidden_sizes), 32, 0
            layers = layers or tuple(range(depth))
        else:
            depth, self.patch = cfg.num_hidden_layers, cfg.patch_size
            self.skip = (
                1 + cfg.num_register_tokens
            )  # CLS and registers lead the sequence
            layers = layers or (2, 5, 8, 11)
        assert max(layers) < depth, (layers, depth)
        self.layers = tuple(layers)
        # ImageNet statistics, the LVD-1689M files' preprocessing
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self._train = torch.compile(self._dist) if compile else self._dist

    def _dist(self, x: torch.Tensor) -> torch.Tensor:
        """(2n, 3, t, t) in [-1, 1], the first n one image's tiles and the last n the
        other's at the same places → the distance."""
        x = ((x + 1) / 2 - self.mean) / self.std
        hs = self.net(pixel_values=x, output_hidden_states=True).hidden_states
        n = x.shape[0] // 2
        tot = x.new_zeros((), dtype=torch.float32)
        for i in self.layers:  # hs[0] is the input side, hs[i + 1] layer i's output
            if self.convnext:  # a stage map (2n, C, h, w)
                t = F.normalize(hs[i + 1].float(), dim=1)
            else:  # (2n, tokens, C)
                t = F.normalize(hs[i + 1][:, self.skip :].float(), dim=-1)
            tot = tot + F.mse_loss(t[:n], t[n:])
        return tot / len(self.layers)

    def _run(self, x: torch.Tensor, compiled: bool = False) -> torch.Tensor:
        cuda = x.device.type == "cuda"
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=cuda):
            return (self._train if compiled else self._dist)(x).float()

    def _side(self, h: int, w: int) -> int:
        return min(
            self.TILE, h // self.patch * self.patch, w // self.patch * self.patch
        )

    def term(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        rng: random.Random,
        n: int = 4,
        compiled: bool = True,
    ) -> torch.Tensor:
        """The training term on one (3, H, W) pair: ``n`` tiles at places drawn from
        ``rng``. The gradient reaches ``a`` only when ``b`` carries none."""
        h, w = a.shape[-2:]
        t = self._side(h, w)
        at = [(rng.randint(0, h - t), rng.randint(0, w - t)) for _ in range(n)]
        x = torch.stack(
            [im[:, y : y + t, x0 : x0 + t].float() for im in (a, b) for y, x0 in at]
        )
        return self._run(x, compiled)

    @torch.no_grad()
    def image(self, a: torch.Tensor, b: torch.Tensor, batch: int = 8) -> float:
        """Whole-image distance: the mean over a grid of tiles covering the image, the
        last row and column moved inward to end at the border."""
        h, w = a.shape[-2:]
        t = self._side(h, w)
        ys = sorted({*range(0, h - t + 1, t), h - t})
        xs = sorted({*range(0, w - t + 1, t), w - t})
        at = [(y, x0) for y in ys for x0 in xs]
        tot = 0.0
        for i in range(0, len(at), batch):
            part = at[i : i + batch]
            x = torch.stack(
                [
                    im[:, y : y + t, x0 : x0 + t].float()
                    for im in (a, b)
                    for y, x0 in part
                ]
            )
            tot += self._run(x).item() * len(part)
        return tot / len(at)


# --------------------------------------------------------------------------- PE-GAN


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
    """Frozen PE-Spatial-B16 + trainable token heads (FastFrames' discriminator form):
    one logit per PE token from each tapped depth, on paired crops of the head's output
    and the real image (×2: the real 2× image; ×1: the original, or the whole image) at
    the same place, both in [-1, 1].

    Paired relativistic loss per token, ``L_D = softplus(−(D(real) − D(fake)))`` and
    ``L_G = softplus(−(D(fake) − D(real)))``, with the noise-perturbation form of R1 / R2
    (no double backward): ``r_weight · mean((D(x + σε) − D(x))²)`` on each crop.

    ``gen_term(fake, real)`` returns L_G on one paired crop and keeps both crops and their
    tokens detached; ``step()`` then makes one discriminator update on them, running the
    tower once more on the two noised crops as one batch."""

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


# --------------------------------------------------------------------------- schedule


def cosine_lr(step: int, total: int, warmup: int) -> float:
    """LR multiplier: linear warmup over ``warmup`` steps on one cosine cycle."""
    return min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))


def _grad_norm(grads) -> torch.Tensor:
    return torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None))


# --------------------------------------------------------------------------- read


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.float(), b.float()).item()  # range [-1, 1] → peak-to-peak 2
    return 10 * math.log10(4.0 / max(mse, 1e-12))


def luma8(img: torch.Tensor) -> torch.Tensor:
    """(3, H, W) in [-1, 1] → Rec.601 luma in 8-bit levels."""
    w = img.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return ((img.float() + 1) * 127.5 * w).sum(0)


def tone_stats(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[float, float | None]:
    """(whole-image luma, luma over the pixels ``tgt`` has under 64), ``pred`` −
    ``tgt`` in 8-bit levels: the dark-pixel lift."""
    lp, lt = luma8(pred), luma8(tgt)
    d = lp - lt
    dark = lt < 64
    return d.mean().item(), (d[dark].mean().item() if dark.any() else None)


def densest_window(img: torch.Tensor, crop: int) -> tuple[int, int]:
    """(x, y) of the ``crop``-px window with the most edges, on a ``crop``/4 grid."""
    g = img.float().mean(0)
    e = (g.diff(dim=0).abs()[:, :-1] + g.diff(dim=1).abs()[:-1, :])[None, None]
    s = F.avg_pool2d(e, crop, stride=crop // 4)[0, 0]
    iy, ix = divmod(int(s.argmax()), s.shape[1])
    return ix * (crop // 4), iy * (crop // 4)


def to_pil(img: torch.Tensor):
    """(3, H, W) in [-1, 1] → an RGB PIL image."""
    from PIL import Image

    arr = ((img.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


def zoom_sheet(
    cols, path: Path, crop: int = 384, at: tuple[int, int] | None = None
) -> None:
    """Labelled columns over their ``crop``-px window at ``at`` = (x, y) (default: the
    centre), doubled nearest; the window is boxed on the full images."""
    from PIL import Image, ImageDraw

    pils = [to_pil(t) for _, t in cols]
    w, h = pils[0].size
    cx, cy = at if at is not None else ((w - crop) // 2, (h - crop) // 2)
    crops = [
        p.crop((cx, cy, cx + crop, cy + crop)).resize(
            (2 * crop, 2 * crop), Image.NEAREST
        )
        for p in pils
    ]
    cw = max(w, 2 * crop)
    canvas = Image.new("RGB", (cw * len(cols), 24 + h + 2 * crop), "white")
    d = ImageDraw.Draw(canvas)
    for i, ((name, _), p, c) in enumerate(zip(cols, pils, crops)):
        d.text((i * cw + 6, 4), name, fill="black")
        canvas.paste(p, (i * cw, 24))
        d.rectangle(
            (i * cw + cx, 24 + cy, i * cw + cx + crop, 24 + cy + crop), outline="red"
        )
        canvas.paste(c, (i * cw, 24 + h))
    canvas.save(path)


def pair_sheet(cols, path: Path, at: tuple[int, int, int] | None = None) -> None:
    """Labelled columns ``[(name, (3, H, W) in [-1, 1]), …]`` side by side at native
    pixels; ``at`` = (x, y, size) crops every column."""
    from PIL import Image, ImageDraw

    pils = [to_pil(t) for _, t in cols]
    if at is not None:
        x, y, s = at
        pils = [p.crop((x, y, x + s, y + s)) for p in pils]
    w, h = pils[0].size
    canvas = Image.new("RGB", (w * len(cols), 24 + h), "white")
    d = ImageDraw.Draw(canvas)
    for i, ((name, _), p) in enumerate(zip(cols, pils)):
        d.text((i * w + 6, 4), name, fill="black")
        canvas.paste(p, (i * w, 24))
    canvas.save(path)
