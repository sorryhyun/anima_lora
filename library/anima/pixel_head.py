"""Pixel head: decode Anima's last DiT features straight to RGB, in place of the VAE.

A small ConvNeXt head (``project/pixel_decoder``, FastFrames-style) reads the input of the
DiT's ``final_layer.linear`` at the last sampling step (LayerNorm + adaLN-modulated,
``hidden_size`` per token) and x_t's 2×2 latent patch (64 values), and writes a 16×16 RGB
patch per token in [-1, 1]. Under CFG the feature is combined like the velocity,
f_u + g·(f_c − f_u): the tap sits right before a linear map, so the combined feature maps
to v_cfg exactly.

Pieces:

- :class:`PixelHead` / :func:`load_pixel_head` — the module and a loader that reads the
  shape from the file's metadata (``width``, ``depth``, ``up``, ``channels``).
  ``channels = 4``: the RGBA file (``project/pixel_decoder/rgba_plan.md``), RGB and 2α − 1.
- :class:`PixelHeadX2` — a ``PixelHead`` plus a second 2× stage (files stamped
  ``x2_depth``, 32×32 output pixels per token). The file decides its scales: stamped
  ``scales = "2"`` it holds no ×1 output layer and decodes at ×2 only; unstamped (the
  preview file) it decodes at ×1 and ×2. :func:`default_scale` is the file's own.
- :class:`PixelTap` — forward hooks on a built DiT that keep the last actual forward's
  (σ, x_t, feature) per CFG branch; :meth:`PixelTap.decode` runs the head on them.
- :func:`out_of_range` — whether a decode point lies outside the σ ranges the head was
  trained at (``sigma`` in the metadata); callers warn, never refuse.

Training: ``scripts/pixel_head/`` (``base_x1.py``, ``train_x1.py``, ``train_x2.py``) on
``library/anima/pixel_head_train_x1.py`` and ``pixel_head_train.py``. Evals and probes:
``bench/pixel_readout/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PATCH = 16  # pixels per DiT token side (VAE /8 × DiT patch 2)
LAT_PATCH_DIM = 64  # 16 latent channels × 2 × 2

# The released head's training points when a file carries no ``sigma`` stamp:
# base DiT σ 0.0577 … 0.1364 (shift-3 Euler at 50 … 20 steps), turbo σ 0.5.
DEFAULT_SIGMA_RANGES: tuple[tuple[float, float], ...] = ((0.0577, 0.1364), (0.5, 0.5))
SIGMA_TOL = 2e-3


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, ls_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), ls_init))

    def forward(self, x):  # (B, C, H, W)
        h = self.dw(x).permute(0, 2, 3, 1)
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        return x + (self.gamma * h).permute(0, 3, 1, 2)


def from_patches(
    tok: torch.Tensor, gh: int, gw: int, patch: int = PATCH, channels: int = 3
) -> torch.Tensor:
    """(gh·gw, patch·patch·C) token patches → (C, gh·patch, gw·patch)."""
    return (
        tok.view(gh, gw, patch, patch, channels)
        .permute(4, 0, 2, 1, 3)
        .reshape(channels, gh * patch, gw * patch)
    )


def lat_patches(lat: torch.Tensor) -> torch.Tensor:
    """(C, h, w) or (1, C, h, w) latent → (h/2, w/2, 4C)."""
    if lat.ndim == 4:
        lat = lat[0]
    c, h, w = lat.shape
    return (
        lat.view(c, h // 2, 2, w // 2, 2)
        .permute(1, 3, 0, 2, 4)
        .reshape(h // 2, w // 2, 4 * c)
    )


class PixelHead(nn.Module):
    def __init__(
        self,
        d_feat: int,
        d_lat: int,
        width: int,
        depth: int,
        ls_init: float = 1e-6,
        up: int = 0,
        channels: int = 3,
    ):
        """``up`` > 0: the last ``up`` of the ``depth`` blocks run after one 2× upsample
        (LayerNorm + per-token linear to 4 × width/2, pixel shuffle) at width/2, and
        ``out`` writes an 8×8 patch per upsampled position instead of 16×16 per token.
        ``channels``: 3 (RGB) or 4 (RGB + 2α − 1)."""
        super().__init__()
        if channels not in (3, 4):
            raise ValueError(f"PixelHead: channels {channels} (3 or 4)")
        self.channels = channels
        self.up = up
        self.proj = nn.Linear(d_feat, width)
        self.lat_embed = nn.Linear(d_lat, width)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(width, ls_init) for _ in range(depth - up)
        )
        w_out = width
        if up:
            w_out = width // 2
            self.up_norm = nn.LayerNorm(width, eps=1e-6)
            self.up_proj = nn.Linear(width, 4 * w_out)
            self.up_blocks = nn.ModuleList(
                ConvNeXtBlock(w_out, ls_init) for _ in range(up)
            )
        self.patch = PATCH // 2 if up else PATCH
        self.norm_out = nn.LayerNorm(w_out, eps=1e-6)
        self.out = nn.Linear(w_out, self.patch * self.patch * channels)

    def forward(self, f, lat):  # f (gh, gw, D), lat (gh, gw, 64) → (C, H, W)
        gh, gw = f.shape[:2]
        h = (self.proj(f) + self.lat_embed(lat)).permute(2, 0, 1)[None]
        for b in self.blocks:
            h = b(h)
        if self.up:
            u = self.up_proj(self.up_norm(h.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
            h = F.pixel_shuffle(u, 2)  # (1, width/2, 2gh, 2gw)
            for b in self.up_blocks:
                h = b(h)
        h = self.norm_out(h[0].permute(1, 2, 0))
        with torch.autocast(h.device.type, enabled=False):  # output layer in fp32
            tok = self.out(h.float())
        if self.up:
            n, m, p, c = h.shape[0], h.shape[1], self.patch, self.channels
            img = tok.view(n, m, p, p, c).permute(4, 0, 2, 1, 3)
            return img.reshape(c, n * p, m * p)
        return from_patches(tok.reshape(gh * gw, -1), gh, gw, channels=self.channels)


class PixelHeadX2(nn.Module):
    """A ``PixelHead`` (``up`` = 1) plus a second 2× stage → (×1 image, ×2 image).

    The second stage: LayerNorm + per-position linear ``w`` → 4 × ``width2``, pixel
    shuffle, ``depth2`` ConvNeXt blocks at 4-px spacing, and a linear writing an 8×8 RGB
    patch per position, i.e. 32×32 output pixels per DiT token. The ×1 output is the base
    head's. ``scales=(2,)`` drops it (``base.norm_out`` / ``base.out``): a file trained
    with the ×1 loss off carries no usable ×1 output. ``forward(…, scale=1 | 2)`` runs only
    the stages that scale needs; ``scale=None`` returns both (training, two-scale heads).

    ``noise`` > 0 (files stamped ``x2_noise``): before each second-stage block, ``noise``
    Gaussian channels per position, a fresh draw per forward, enter through a linear map
    that starts at zero (StyleGAN's noise inputs). 64 = one value per output pixel of the
    position's 8×8 patch. ``forward(…, noise=False)`` leaves the draw out.

    ``overlap`` (files whose ``out2`` writes 16×16 patches, stamped ``x2_overlap``): each
    position writes a patch twice its 8-px cell, centred on it; the patches are summed
    under a sin² window whose shifts by one cell add to 1, so every output pixel blends
    its four nearest positions and no patch or token border carries a step."""

    def __init__(
        self,
        base: PixelHead,
        depth2: int = 2,
        ls_init: float = 1.0,
        width2: int | None = None,
        scales: tuple[int, ...] = (1, 2),
        noise: int = 0,
        overlap: bool = False,
    ):
        super().__init__()
        assert base.up, "base head: one upsample stage"
        self.noise = 0
        self.overlap = overlap
        self.base = base
        w = base.up_proj.out_features // 4
        w2 = width2 or w // 2
        self.patch2 = base.patch  # 8 output px per 4-px position = 2× the base
        self.up2_norm = nn.LayerNorm(w, eps=1e-6)
        self.up2_proj = nn.Linear(w, 4 * w2)
        self.up2_blocks = nn.ModuleList(
            ConvNeXtBlock(w2, ls_init) for _ in range(depth2)
        )
        self.norm2 = nn.LayerNorm(w2, eps=1e-6)
        k2 = self.patch2 * (2 if overlap else 1)  # the side of the patch out2 writes
        self.out2 = nn.Linear(w2, k2 * k2 * 3)
        k = 2 * self.patch2
        win = torch.sin(torch.pi * (torch.arange(k) + 0.5) / k) ** 2
        self.register_buffer("win2", win[:, None] * win[None, :], persistent=False)
        self.scales = (1, 2)
        if tuple(scales) == (2,):
            drop_x1_output(self)
        elif tuple(scales) != (1, 2):
            raise ValueError(f"PixelHeadX2: scales {scales}; (1, 2) or (2,)")
        if noise:
            self.add_noise(noise)

    def add_noise(self, channels: int = 64) -> "PixelHeadX2":
        """Give the second stage its noise inputs (zero-init: the output is unchanged
        until they are trained), on the head's device."""
        assert not self.noise, "PixelHeadX2: noise inputs already present"
        w2 = self.norm2.normalized_shape[0]
        self.noise_in = nn.ModuleList(
            nn.Linear(channels, w2, bias=False) for _ in self.up2_blocks
        ).to(self.out2.weight.device)
        for m in self.noise_in:
            nn.init.zeros_(m.weight)
        self.noise = channels
        return self

    def to_overlap(self) -> "PixelHeadX2":
        """Turn an 8×8 ``out2`` into the overlapping 16×16 one in place. Each new patch
        starts as the old one with its edge pixels repeated outwards, so the output
        starts as the old output blended across the patch borders."""
        assert not self.overlap, "PixelHeadX2: out2 already overlaps"
        p, old = self.patch2, self.out2
        src = (torch.arange(2 * p) - p // 2).clamp(0, p - 1)
        new = nn.Linear(old.in_features, 4 * p * p * 3).to(old.weight.device)
        with torch.no_grad():
            w = old.weight.view(p, p, 3, -1)[src][:, src]
            new.weight.copy_(w.reshape(4 * p * p * 3, -1))
            new.bias.copy_(old.bias.view(p, p, 3)[src][:, src].reshape(-1))
        self.out2, self.overlap = new, True
        return self

    def _overlap_add(self, tok: torch.Tensor) -> torch.Tensor:
        """(n, m, 16·16·3) patches at stride 8 → (3, 8n, 8m), window-weighted and
        divided by the window's own sum (under 1 only on the image's 4-px rim)."""
        n, m = tok.shape[:2]
        p, k = self.patch2, 2 * self.patch2
        win = self.win2.to(tok.dtype)
        t = tok.view(n * m, k, k, 3) * win[None, :, :, None]
        t = t.permute(3, 1, 2, 0).reshape(1, 3 * k * k, n * m)
        size = dict(output_size=(n * p, m * p), kernel_size=k, stride=p, padding=p // 2)
        img = F.fold(t, **size)
        norm = F.fold(win.reshape(1, k * k, 1).expand(1, k * k, n * m), **size)
        return (img / norm)[0]

    @staticmethod
    def _unpatch(tok: torch.Tensor, p: int) -> torch.Tensor:
        n, m = tok.shape[:2]
        return tok.view(n, m, p, p, 3).permute(4, 0, 2, 1, 3).reshape(3, n * p, m * p)

    def forward(self, f, lat, scale: int | None = None, noise: bool = True):
        """f (gh, gw, D), lat (gh, gw, 64) → (×1, ×2) images, or one of them."""
        b = self.base
        h = (b.proj(f) + b.lat_embed(lat)).permute(2, 0, 1)[None]
        for blk in b.blocks:
            h = blk(h)
        u = b.up_proj(b.up_norm(h.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        h = F.pixel_shuffle(u, 2)  # (1, w, 2gh, 2gw)
        for blk in b.up_blocks:
            h = blk(h)
        x1 = x2 = None
        if scale != 2:
            if 1 not in self.scales:
                raise ValueError("pixel head: this ×2 file has no ×1 output (scales 2)")
            h1 = b.norm_out(h[0].permute(1, 2, 0))
            with torch.autocast(h.device.type, enabled=False):  # output layer in fp32
                x1 = self._unpatch(b.out(h1.float()), b.patch)
            if scale == 1:
                return x1
        u = self.up2_proj(self.up2_norm(h.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        g = F.pixel_shuffle(u, 2)  # (1, w/2, 4gh, 4gw)
        for i, blk in enumerate(self.up2_blocks):
            if self.noise and noise:
                z = torch.randn(
                    g.shape[2], g.shape[3], self.noise, device=g.device, dtype=g.dtype
                )
                g = g + self.noise_in[i](z).permute(2, 0, 1)[None]
            g = blk(g)
        h2 = self.norm2(g[0].permute(1, 2, 0))
        with torch.autocast(h.device.type, enabled=False):
            tok = self.out2(h2.float())
            if self.overlap:
                x2 = self._overlap_add(tok)
            else:
                x2 = self._unpatch(tok, self.patch2)
        return x2 if scale == 2 else (x1, x2)


def drop_x1_output(head: PixelHeadX2) -> PixelHeadX2:
    """Remove the ×1 output layer (``base.norm_out`` / ``base.out``) in place: the head
    decodes at ×2 only and its file is stamped ``scales = "2"``."""
    for name in ("norm_out", "out"):
        if hasattr(head.base, name):
            delattr(head.base, name)
    head.scales = (2,)
    return head


def head_scales(head: nn.Module) -> tuple[int, ...]:
    """The output scales a loaded head can decode: (1,), (1, 2) or (2,)."""
    if isinstance(head, PixelHeadX2):
        return tuple(getattr(head, "scales", (1, 2)))
    return (1,)


def default_scale(head: nn.Module) -> int:
    """The scale a head decodes at when none is asked for: its only one, 1 for a
    two-scale file."""
    return min(head_scales(head))


def check_scale(head: nn.Module, scale: int | None, name: str = "pixel head") -> int:
    """``scale`` (``None`` → the file's default) if the head holds it, else ValueError."""
    have = head_scales(head)
    if scale is None:
        return default_scale(head)
    if scale not in have:
        if scale == 2:
            why = "needs a ×2 head file (stamped x2_depth); this file has the ×1 output only"
        else:
            why = "this ×2 file has no ×1 output (stamped scales = 2); use the ×1 file"
        raise ValueError(f"{name}: scale {scale} {why}")
    return scale


def parse_sigma_ranges(stamp: str | None) -> tuple[tuple[float, float], ...]:
    """``"0.0577-0.1364,0.5"`` → ((0.0577, 0.1364), (0.5, 0.5)); empty → the defaults."""
    if not stamp:
        return DEFAULT_SIGMA_RANGES
    out = []
    for part in stamp.split(","):
        lo, _, hi = part.strip().partition("-")
        out.append((float(lo), float(hi or lo)))
    return tuple(out)


def load_pixel_head(
    path: str | Path, device="cpu", d_feat: int | None = None
) -> PixelHead | PixelHeadX2:
    """A trained head from a ``.safetensors`` file, shape from its metadata.

    A file stamped ``x2_depth`` (keys ``base.*``, ``up2_*``, ``out2.*``) loads as a
    :class:`PixelHeadX2`: at ×2 only when also stamped ``scales = "2"`` (no ×1 output
    layer in the file), else at ×1 and ×2, with noise inputs when stamped ``x2_noise``
    and the overlapping output when ``out2`` holds 16×16 patches; any other file as a
    :class:`PixelHead`.
    ``d_feat`` defaults to the stored ``proj.weight``'s input width (2048 on Anima). The
    output channels come from the ``channels`` stamp (3 when unstamped; 4 = RGBA). The
    trained σ ranges ride along as ``head.sigma_ranges``, the rest of the stamp as
    ``head.meta``. A file stamped ``skip = 1`` (or unstamped: the P1 readout heads) needs
    the skip linear this module no longer has and does not load; the ``linear.*`` weights
    older files carry unused are dropped."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    x2 = False
    with safe_open(str(path), "pt") as fh:
        meta = fh.metadata() or {}
        if meta.get("skip", "1") != "0":
            raise ValueError(
                f"{path}: a skip-linear head (P1 readout); no longer loads"
            )
        x2 = "x2_depth" in meta
        pre = "base." if x2 else ""
        if d_feat is None:
            d_feat = int(fh.get_slice(f"{pre}proj.weight").get_shape()[1])
        if x2:
            width2 = int(fh.get_slice("up2_proj.weight").get_shape()[0]) // 4
            # an overlapping out2 writes 16×16 patches (4× the 8×8 layer's rows)
            overlap = int(fh.get_slice("out2.weight").get_shape()[0]) == 3 * PATCH**2
    head = PixelHead(
        d_feat,
        LAT_PATCH_DIM,
        int(meta.get("width", "1024")),
        int(meta.get("depth", "3")),
        up=int(meta.get("up", "0")),
        channels=int(meta.get("channels", "3")),
    )
    if x2:
        scales = tuple(int(v) for v in meta.get("scales", "1,2").split(","))
        head = PixelHeadX2(
            head,
            int(meta["x2_depth"]),
            width2=width2,
            scales=scales,
            noise=int(meta.get("x2_noise", "0")),
            overlap=overlap,
        )
    sd = load_file(str(path))
    head.load_state_dict(
        {k: v for k, v in sd.items() if not k.startswith(("linear.", "base.linear."))}
    )
    head.sigma_ranges = parse_sigma_ranges(meta.get("sigma"))
    head.meta = dict(meta)
    return head.to(device).eval()


def default_pixel_head_path() -> Path:
    """The catalog row's file (``library/downloads.py``, id ``anima_pixel_head``)."""
    from library import downloads

    return downloads.default_pixel_head_path()


def resolve_pixel_head_path(path: str | None, scale: int | None = None) -> Path:
    """An explicit path, or the catalog file for ``scale`` (``anima_pixel_head_x2`` at
    2, else ``anima_pixel_head``; fetched from the Hub on first use)."""
    from library import downloads
    from library.env import resolve_under_home

    if path:
        return Path(resolve_under_home(path))
    asset = downloads.by_id()[
        "anima_pixel_head_x2" if scale == 2 else "anima_pixel_head"
    ]
    if not asset.installed:
        logger.info(f"pixel head: fetching {asset.repo} -> {asset.dest}")
        downloads.fetch(asset, log=logger.info)
    return asset.dest / asset.files[0]


def out_of_range(
    sigma: float, ranges: tuple[tuple[float, float], ...] = DEFAULT_SIGMA_RANGES
) -> bool:
    return not any(lo - SIGMA_TOL <= sigma <= hi + SIGMA_TOL for lo, hi in ranges)


class PixelTap:
    """Hooks on a built Anima that keep the last actual forward's inputs for the head.

    Every ``anima(x, timesteps, …)`` call opens a record (σ, x_t); the next
    ``final_layer.linear`` call fills its feature. A new σ clears the records, so after
    sampling they hold the last step's forwards in call order: one at CFG 1, cond then
    uncond under CFG (``generate_body``'s order). A ``final_layer`` call with no forward of
    its own (a Spectrum cached step) sets :attr:`tail_cached`: the last step was not an
    actual forward.

    Attach after the network and the stream change, before ``compile_blocks`` or after —
    the hooks sit on the top module and ``final_layer.linear``, outside the compiled
    blocks."""

    def __init__(self, anima: nn.Module):
        self.calls: list[dict] = []
        self.tail_cached = False
        self._handles = [
            anima.register_forward_pre_hook(self._on_forward, with_kwargs=True),
            anima.final_layer.linear.register_forward_pre_hook(self._on_linear),
        ]

    def _on_forward(self, _m, args, kwargs):
        x = args[0] if args else kwargs["x"]
        t = args[1] if len(args) > 1 else kwargs["timesteps"]
        sigma = float(t.flatten()[0])
        if self.calls and self.calls[-1]["sigma"] != sigma:
            self.calls.clear()
        self.calls.append({"sigma": sigma, "x": x.detach(), "f": None})
        self.tail_cached = False

    def _on_linear(self, _m, inp):
        if self.calls and self.calls[-1]["f"] is None:
            self.calls[-1]["f"] = inp[0].detach()
        else:
            self.tail_cached = True

    def reset(self) -> None:
        self.calls.clear()
        self.tail_cached = False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def sigma(self) -> float | None:
        return self.calls[-1]["sigma"] if self.calls else None

    def feature(self, guidance_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(f, x_t) of the last step: f (B, T, gh, gw, D) fp32, CFG-combined when the
        step ran two forwards; x_t (B, C, T, h, w)."""
        calls = [c for c in self.calls if c["f"] is not None]
        if not calls:
            raise RuntimeError("PixelTap: no forward recorded (attach before sampling)")
        if len(calls) == 1:
            return calls[0]["f"].float(), calls[0]["x"]
        if len(calls) != 2:
            raise RuntimeError(
                f"PixelTap: {len(calls)} forwards at the last step; expected 1 or 2"
            )
        f_c, f_u = calls[0]["f"].float(), calls[1]["f"].float()
        return f_u + guidance_scale * (f_c - f_u), calls[0]["x"]

    @torch.no_grad()
    def decode(
        self,
        head: PixelHead | PixelHeadX2,
        guidance_scale: float,
        scale: int | None = None,
    ) -> torch.Tensor:
        """Pixels (B, C, scale·H, scale·W) fp32 on the head's device, clamped to
        [-1, 1]; C is the file's 3 or 4 (RGB, 2α − 1).
        ``scale=None`` decodes at the file's own (:func:`default_scale`); a scale the
        file does not hold raises."""
        scale = check_scale(head, scale)
        if isinstance(head, PixelHeadX2):
            fwd = lambda f, lat: head(f, lat, scale=scale)  # noqa: E731
        else:
            fwd = head
        f, x_t = self.feature(guidance_scale)
        dev = next(head.parameters()).device
        f = f.to(dev)
        x_t = x_t.to(dev).float()
        if x_t.ndim == 5:
            x_t = x_t[:, :, 0]
        if f.ndim == 5:
            f = f[:, 0]
        out = []
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            for b in range(f.shape[0]):
                out.append(fwd(f[b], lat_patches(x_t[b])).float())
        return torch.stack(out).clamp(-1, 1)
