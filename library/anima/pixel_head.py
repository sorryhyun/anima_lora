"""Pixel head: decode Anima's last DiT features straight to RGB, in place of the VAE.

A small ConvNeXt head (``project/pixel_decoder``, FastFrames-style) reads the input of the
DiT's ``final_layer.linear`` at the last sampling step (LayerNorm + adaLN-modulated,
``hidden_size`` per token) and x_t's 2×2 latent patch (64 values), and writes a 16×16 RGB
patch per token in [-1, 1]. Under CFG the feature is combined like the velocity,
f_u + g·(f_c − f_u): the tap sits right before a linear map, so the combined feature maps
to v_cfg exactly.

Pieces:

- :class:`PixelHead` / :func:`load_pixel_head` — the module and a loader that reads the
  shape from the file's metadata (``width``, ``depth``, ``up``, ``skip``).
- :class:`PixelHeadX2` — a ``PixelHead`` plus a second 2× stage (files stamped
  ``x2_depth``): one file decodes at ×1 and ×2 (32×32 output pixels per token).
- :class:`PixelTap` — forward hooks on a built DiT that keep the last actual forward's
  (σ, x_t, feature) per CFG branch; :meth:`PixelTap.decode` runs the head on them.
- :func:`out_of_range` — whether a decode point lies outside the σ ranges the head was
  trained at (``sigma`` in the metadata); callers warn, never refuse.

Training, evals and probes stay in ``bench/pixel_readout/``.
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
    tok: torch.Tensor, gh: int, gw: int, patch: int = PATCH
) -> torch.Tensor:
    """(gh·gw, patch·patch·3) token patches → (3, gh·patch, gw·patch)."""
    return (
        tok.view(gh, gw, patch, patch, 3)
        .permute(4, 0, 2, 1, 3)
        .reshape(3, gh * patch, gw * patch)
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
        use_skip: bool = True,
        up: int = 0,
    ):
        """``up`` > 0: the last ``up`` of the ``depth`` blocks run after one 2× upsample
        (LayerNorm + per-token linear to 4 × width/2, pixel shuffle) at width/2, and
        ``out`` writes an 8×8 patch per upsampled position instead of 16×16 per token.
        ``use_skip`` adds a per-token linear from the feature straight to the patch
        (the P1 readout); the released head runs without it."""
        super().__init__()
        out = PATCH * PATCH * 3
        self.use_skip = use_skip
        self.up = up
        self.linear = nn.Linear(d_feat, out)  # skip (unused when use_skip is off)
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
        self.out = nn.Linear(w_out, self.patch * self.patch * 3)
        if use_skip:  # step 0 reproduces the skip alone
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, f, lat):  # f (gh, gw, D), lat (gh, gw, 64) → (3, H, W)
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
        # the skip loses ~dB under bf16 weights; the output layer stays in fp32
        with torch.autocast(h.device.type, enabled=False):
            tok = self.out(h).float()
            if self.up:
                n, m, p = h.shape[0], h.shape[1], self.patch
                img = tok.view(n, m, p, p, 3).permute(4, 0, 2, 1, 3)
                img = img.reshape(3, n * p, m * p)
                if self.use_skip:
                    img = img + from_patches(
                        self.linear(f.float()).reshape(gh * gw, -1), gh, gw
                    )
                return img
            if self.use_skip:
                tok = tok + self.linear(f.float())
        return from_patches(tok.reshape(gh * gw, -1).float(), gh, gw)


class PixelHeadX2(nn.Module):
    """A ``PixelHead`` (``up`` = 1, no skip) plus a second 2× stage → (×1 image, ×2 image).

    The second stage: LayerNorm + per-position linear ``w`` → 4 × ``width2``, pixel
    shuffle, ``depth2`` ConvNeXt blocks at 4-px spacing, and a linear writing an 8×8 RGB
    patch per position, i.e. 32×32 output pixels per DiT token. The ×1 output is the base
    head's, so one file decodes at either scale. ``forward(…, scale=1 | 2)`` runs only the
    stages that scale needs; ``scale=None`` returns both (training)."""

    def __init__(
        self,
        base: PixelHead,
        depth2: int = 2,
        ls_init: float = 1.0,
        width2: int | None = None,
    ):
        super().__init__()
        assert base.up and not base.use_skip, "base head: one upsample stage, no skip"
        self.base = base
        w = base.out.in_features
        w2 = width2 or w // 2
        self.patch2 = base.patch  # 8 output px per 4-px position = 2× the base
        self.up2_norm = nn.LayerNorm(w, eps=1e-6)
        self.up2_proj = nn.Linear(w, 4 * w2)
        self.up2_blocks = nn.ModuleList(
            ConvNeXtBlock(w2, ls_init) for _ in range(depth2)
        )
        self.norm2 = nn.LayerNorm(w2, eps=1e-6)
        self.out2 = nn.Linear(w2, self.patch2 * self.patch2 * 3)

    @staticmethod
    def _unpatch(tok: torch.Tensor, p: int) -> torch.Tensor:
        n, m = tok.shape[:2]
        return tok.view(n, m, p, p, 3).permute(4, 0, 2, 1, 3).reshape(3, n * p, m * p)

    def forward(self, f, lat, scale: int | None = None):
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
            h1 = b.norm_out(h[0].permute(1, 2, 0))
            with torch.autocast(h.device.type, enabled=False):  # output layer in fp32
                x1 = self._unpatch(b.out(h1.float()), b.patch)
            if scale == 1:
                return x1
        u = self.up2_proj(self.up2_norm(h.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        g = F.pixel_shuffle(u, 2)  # (1, w/2, 4gh, 4gw)
        for blk in self.up2_blocks:
            g = blk(g)
        h2 = self.norm2(g[0].permute(1, 2, 0))
        with torch.autocast(h.device.type, enabled=False):
            x2 = self._unpatch(self.out2(h2.float()), self.patch2)
        return x2 if scale == 2 else (x1, x2)


def head_scales(head: nn.Module) -> tuple[int, ...]:
    """The output scales a loaded head can decode: (1,) or (1, 2)."""
    return (1, 2) if isinstance(head, PixelHeadX2) else (1,)


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
    :class:`PixelHeadX2`, which decodes at ×1 and ×2; any other as a :class:`PixelHead`.
    ``d_feat`` defaults to the stored ``proj.weight``'s input width (2048 on Anima). The
    trained σ ranges ride along as ``head.sigma_ranges``, the rest of the stamp as
    ``head.meta``."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    x2 = False
    with safe_open(str(path), "pt") as fh:
        meta = fh.metadata() or {}
        x2 = "x2_depth" in meta
        pre = "base." if x2 else ""
        if d_feat is None:
            d_feat = int(fh.get_slice(f"{pre}proj.weight").get_shape()[1])
        if x2:
            width2 = int(fh.get_slice("up2_proj.weight").get_shape()[0]) // 4
    head = PixelHead(
        d_feat,
        LAT_PATCH_DIM,
        int(meta.get("width", "1024")),
        int(meta.get("depth", "3")),
        use_skip=meta.get("skip", "1") == "1",
        up=int(meta.get("up", "0")),
    )
    if x2:
        head = PixelHeadX2(head, int(meta["x2_depth"]), width2=width2)
    head.load_state_dict(load_file(str(path)))
    head.sigma_ranges = parse_sigma_ranges(meta.get("sigma"))
    head.meta = dict(meta)
    return head.to(device).eval()


def default_pixel_head_path() -> Path:
    """The catalog row's file (``library/downloads.py``, id ``anima_pixel_head``)."""
    from library import downloads

    return downloads.default_pixel_head_path()


def resolve_pixel_head_path(path: str | None) -> Path:
    """An explicit path, or the catalog default (fetched from the Hub on first use)."""
    from library import downloads
    from library.env import resolve_under_home

    if path:
        return Path(resolve_under_home(path))
    asset = downloads.by_id()["anima_pixel_head"]
    if not asset.installed:
        logger.info(f"pixel head: fetching {asset.repo} -> {asset.dest}")
        downloads.fetch(asset, log=logger.info)
    return downloads.default_pixel_head_path()


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
        self, head: PixelHead | PixelHeadX2, guidance_scale: float, scale: int = 1
    ) -> torch.Tensor:
        """Pixels (B, 3, scale·H, scale·W) fp32 on the head's device, clamped to
        [-1, 1]. ``scale=2`` needs a ×2 head file (:class:`PixelHeadX2`)."""
        if scale not in head_scales(head):
            raise ValueError(
                f"pixel head: scale {scale} needs a ×2 head file (stamped x2_depth); "
                "this file has the ×1 output only"
            )
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
