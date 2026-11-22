"""Pixel-head training: the pieces both scales share, and the ×2 recipe's own.

Shared with ×1 (``library/anima/pixel_head_train_x1.py``) and the bench:
:class:`DiTRollout`, :func:`attach_adapter`, :func:`compile_dit`, :func:`rec_loss`,
:class:`Lpips`, :class:`PEDisc`, :func:`cosine_lr` and the read helpers.

``scripts/pixel_head/train_x2.py`` composes the ×2 recipe. The recipe (``project/pixel_decoder/README.md``): a frozen Anima DiT is run once per
rollout on a noised cached latent at a decode point σ; the head reads the feature before
``final_layer.linear`` and x_t's latent patches and writes the 2× image, trained against
the real 2× image the latent's own image was made from (the ``x2flex`` set). One random
window of ``window`` + 2 × ``margin`` tokens per rollout goes into a replay buffer, and the
loss leaves ``margin`` tokens out on every window side that is not an image edge.

Pieces, in the order a rollout uses them:

- :func:`load_x2flex` / :class:`X2Item` — the manifest rows, by split.
- :class:`DiTRollout` — (item, σ) → (features, x_t patches, VAE(x̂0)).
- :func:`pick_window` / :class:`Window` — the loss window and its margins.
- :func:`rec_loss`, :func:`dc_loss`, :class:`Lpips`, :class:`PEDisc` — the loss terms.
- :class:`X2Trainer` — replay buffer, schedule, the PE-GAN weight rule and its start.
- :func:`psnr`, :func:`tone_stats`, :func:`densest_window`, :func:`pair_sheet` — the read.
- :func:`save_x2_head` — the file :func:`library.anima.pixel_head.load_pixel_head` reads.

The earlier arms (teacher target, ×1 loss in a ×2 file) and the whole-image reads with the
teacher columns are in ``bench/pixel_readout/experiments/x2_head.py``.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from library.anima.pixel_head import PATCH, PixelHeadX2, head_scales, lat_patches

# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class X2Item:
    """One ``x2flex`` manifest row: a cached latent, its ×1 image and the real 2× image."""

    stem: str
    x1: str
    x2: str
    npz: str
    te: str
    kind: str  # "down" (×1 is the 2× image halved) | "exact" (2× is the upscale source)

    @property
    def tokens(self) -> int:
        return npz_tokens(self.npz)


def npz_tokens(path: str) -> int:
    """DiT tokens of a cached latent, from the ``WxH`` in its file name."""
    w, h = (int(v) for v in path.rsplit("_", 2)[-2].split("x"))
    return (w // PATCH) * (h // PATCH)


def load_x2flex(manifest: str | Path) -> dict[str, list[X2Item]]:
    """``post_image_dataset/x2flex/manifest.json`` → {"train" | "test" | "text": rows}."""
    from library.env import resolve_under_home

    rows = json.loads(Path(resolve_under_home(str(manifest))).read_text())
    out: dict[str, list[X2Item]] = {"train": [], "test": [], "text": []}
    for row in rows:
        if row["split"] not in out:
            continue
        item = X2Item(
            stem=row["stem"],
            x1=str(resolve_under_home(row["x1"])),
            x2=str(resolve_under_home(row["x2"])),
            npz=str(resolve_under_home(row["npz"])),
            te=str(resolve_under_home(row["te"])),
            kind=row["kind"],
        )
        assert Path(item.npz).exists(), f"no latent {item.npz} (encode x1/)"
        out[row["split"]].append(item)
    return out


def load_image(path: str | Path, device="cpu") -> torch.Tensor:
    """An RGB image file → (3, H, W) float in [-1, 1]."""
    from PIL import Image

    px = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(px).permute(2, 0, 1).to(device).float() / 127.5 - 1


def real_x2(item: X2Item, like: torch.Tensor) -> torch.Tensor:
    """The item's real 2× image on ``like``'s device; ``like`` is its ×1 image (3, H, W)."""
    t = load_image(item.x2, like.device)
    want = (2 * like.shape[-2], 2 * like.shape[-1])
    assert tuple(t.shape[-2:]) == want, (item.stem, tuple(t.shape), tuple(like.shape))
    return t


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


# --------------------------------------------------------------------------- window


@dataclass(frozen=True)
class Window:
    """A token window (y0, x0, h, w) and the margin (top, bottom, left, right), in
    tokens, that its loss leaves out: ``margin`` on a side inside the image, 0 on a side
    that is an image edge."""

    y0: int
    x0: int
    h: int
    w: int
    mar: tuple[int, int, int, int]

    def tokens(self, t: torch.Tensor) -> torch.Tensor:
        """Slice a (gh, gw, …) token grid."""
        return t[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]

    def pixels(self, img: torch.Tensor, px: int) -> torch.Tensor:
        """Slice a (3, H, W) image at ``px`` pixels per token, margins included."""
        ys = slice(self.y0 * px, (self.y0 + self.h) * px)
        xs = slice(self.x0 * px, (self.x0 + self.w) * px)
        return img[:, ys, xs]


def pick_window(
    gh: int, gw: int, window: int, margin: int, rng: random.Random
) -> Window:
    """A random window of ``window`` + 2 × ``margin`` tokens a side (the whole grid on an
    axis shorter than that)."""
    wt = window + 2 * margin
    wh, ww = min(gh, wt), min(gw, wt)
    y0, x0 = rng.randint(0, gh - wh), rng.randint(0, gw - ww)
    mar = (
        0 if y0 == 0 else margin,
        0 if y0 + wh == gh else margin,
        0 if x0 == 0 else margin,
        0 if x0 + ww == gw else margin,
    )
    return Window(y0, x0, wh, ww, mar)


def crop_margin(img: torch.Tensor, mar: tuple[int, int, int, int], px: int):
    """Drop ``mar`` = (top, bottom, left, right) tokens, ``px`` pixels per token."""
    t, b, lf, r = (k * px for k in mar)
    h, w = img.shape[-2:]
    return img[..., t : h - b, lf : w - r]


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


class Lpips:
    """LPIPS-VGG, frozen, channels_last; ``compile`` halves a 768² training step."""

    def __init__(self, device, compile: bool = False):
        import lpips

        self.net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        self.net.requires_grad_(False)
        self.net.to(memory_format=torch.channels_last)
        self._train = torch.compile(self.net) if compile else self.net

    def term(self, a: torch.Tensor, b: torch.Tensor, compiled: bool = True):
        """The training term on one (3, H, W) pair."""
        cl = torch.channels_last
        net = self._train if compiled else self.net
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(
                a[None].contiguous(memory_format=cl),
                b[None].contiguous(memory_format=cl),
            )
        return out.mean().float()

    @torch.no_grad()
    def image(self, a: torch.Tensor, b: torch.Tensor, tile: int = 768) -> float:
        """Whole-image LPIPS as the area-weighted mean over ``tile``-px tiles (one VGG
        pass over a 2560 × 1664 image does not fit beside training)."""
        h, w = a.shape[-2:]
        tot = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    ta = a[None, :, y : y + tile, x : x + tile].float()
                    tb = b[None, :, y : y + tile, x : x + tile].float()
                    tot += self.net(ta, tb).mean().item() * ta.shape[-2] * ta.shape[-1]
        return tot / (h * w)


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


# --------------------------------------------------------------------------- trainer


@dataclass
class X2TrainConfig:
    """The loop's knobs; defaults are the shipped ×2 file's (``x2bgan2_1121_1340``)."""

    rollouts: int = 4000
    opt_steps: int = 16  # head updates per rollout, each on one replay window
    replay: int = 64  # windows kept
    window: int = 24  # loss window, tokens
    margin: int = 3  # context strip, tokens (the head's receptive field)
    lr: float = 2e-4
    warmup: int = 200  # opt steps
    lambda_lpips: float = 0.1
    lambda_dc: float = 1.0
    dc_block: int = 32  # ×2 px
    # PE-GAN: once per rollout the term's weight is set so its gradient norm over the
    # head is `pegan_ratio` times that of rec2 + λ_lpips·LPIPS + λ_dc·dc. 0 = no term.
    pegan_ratio: float = 2.0
    pegan_start: int = 1000  # first rollout with the term and the discriminator
    pegan_lambda_max: float = 1.0
    pegan_crop: int = 512  # ×2 px, a multiple of 16
    seed: int = 0


def cosine_lr(step: int, total: int, warmup: int) -> float:
    """LR multiplier: linear warmup over ``warmup`` steps on one cosine cycle."""
    return min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))


def _grad_norm(grads) -> torch.Tensor:
    return torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None))


class X2Trainer:
    """Replay buffer + the head's updates for one rollout at a time.

    ``push`` adds a rollout's window; ``update(r)`` runs rollout ``r``'s ``opt_steps``
    updates, each on one random replay window, and returns the last step's terms. The
    head runs at ``scale=2`` only: a file of this recipe carries no trained ×1 output."""

    def __init__(
        self,
        head: PixelHeadX2,
        cfg: X2TrainConfig,
        lpips: Lpips | None = None,
        disc: PEDisc | None = None,
        compile: bool = False,
    ):
        assert cfg.lambda_lpips == 0 or lpips is not None, "lambda_lpips needs Lpips"
        assert cfg.pegan_ratio == 0 or disc is not None, "pegan_ratio needs a PEDisc"
        assert cfg.pegan_crop % 16 == 0, "pegan_crop: a multiple of 16"
        self.head, self.cfg, self.lpips, self.disc = head, cfg, lpips, disc
        self.head_c = torch.compile(head, dynamic=True) if compile else head
        self.opt = torch.optim.AdamW(
            head.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.01
        )
        self.total_steps = cfg.rollouts * cfg.opt_steps
        self.step = 0
        self.replay: deque = deque(maxlen=cfg.replay)
        self._replay_rng = random.Random(cfg.seed + 3)
        self._crop_rng = random.Random(cfg.seed + 9)
        self.lam_pegan = 0.0
        self.gan_live = False

    def push(self, f, lat, target2, win: Window) -> None:
        """A rollout's features, x_t patches (whole grids) and the real 2× image."""
        t2 = win.pixels(target2, 2 * PATCH)
        self.replay.append(
            (
                win.tokens(f).to(torch.bfloat16).clone(),
                win.tokens(lat).clone(),
                crop_margin(t2, win.mar, 2 * PATCH).half(),
                win.mar,
            )
        )

    def terms(self, sample, compiled: bool = True) -> dict[str, torch.Tensor]:
        """Unweighted loss terms on one replay window."""
        bf, blat, b2, mar = sample
        cfg = self.cfg
        with torch.autocast(bf.device.type, dtype=torch.bfloat16):
            x2 = (self.head_c if compiled else self.head)(bf, blat, 2)
        p2, t2 = crop_margin(x2, mar, 2 * PATCH), b2.float()
        rec, mse = rec_loss(p2, t2)
        out = {"rec2": rec, "mse2": mse.detach()}
        if cfg.lambda_lpips > 0:
            out["lpips"] = self.lpips.term(p2, t2, compiled)
        if cfg.lambda_dc > 0:
            out["dc"] = dc_loss(p2, t2, cfg.dc_block)
        if self.gan_live:  # a paired crop: the ×2 output and the real 2× image
            c = cfg.pegan_crop
            h, w = p2.shape[-2:]
            assert min(h, w) >= c, (h, w, c)
            y, x = self._crop_rng.randint(0, h - c), self._crop_rng.randint(0, w - c)
            sl = (..., slice(y, y + c), slice(x, x + c))
            out["pegan"] = self.disc.gen_term(p2[None].float()[sl], t2[None][sl])
        return out

    def regression(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        loss = t["rec2"]
        if "lpips" in t:
            loss = loss + cfg.lambda_lpips * t["lpips"]
        if "dc" in t:
            loss = loss + cfg.lambda_dc * t["dc"]
        return loss

    def _set_gan_weight(self) -> None:
        """λ = ratio · ‖∇ regression‖ / ‖∇ pegan‖ over the head, eager, one window."""
        params = [q for q in self.head.parameters() if q.requires_grad]
        t = self.terms(self._replay_rng.choice(self.replay), compiled=False)
        g_reg = torch.autograd.grad(
            self.regression(t), params, retain_graph=True, allow_unused=True
        )
        g_gan = torch.autograd.grad(t["pegan"], params, allow_unused=True)
        lam = self.cfg.pegan_ratio * _grad_norm(g_reg) / (_grad_norm(g_gan) + 1e-12)
        self.lam_pegan = min(lam.item(), self.cfg.pegan_lambda_max)

    def update(self, r: int) -> dict[str, float]:
        cfg = self.cfg
        if self.disc is not None and cfg.pegan_ratio > 0 and r >= cfg.pegan_start:
            self.gan_live = True
        if self.gan_live:
            self._set_gan_weight()
        log: dict[str, float] = {}
        for _ in range(cfg.opt_steps):
            s = cosine_lr(self.step, self.total_steps, cfg.warmup)
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr * s
            self.opt.zero_grad(set_to_none=True)
            t = self.terms(self._replay_rng.choice(self.replay))
            loss = self.regression(t)
            if self.gan_live:
                loss = loss + self.lam_pegan * t["pegan"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.opt.step()
            self.step += 1
            log = {k: v.item() for k, v in t.items()}
            if self.gan_live:
                log["lam_pegan"] = self.lam_pegan
                log.update(self.disc.step())  # on this step's paired crop, detached
        return log


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


def save_x2_head(head: PixelHeadX2, path: str | Path, stamp: dict[str, str]) -> None:
    """Write the head with the shape metadata ``load_pixel_head`` reads, plus ``stamp``
    (``sigma`` = the trained decode range, and whatever names the run). A head without
    its ×1 output layer (:func:`~library.anima.pixel_head.drop_x1_output`) is stamped
    ``scales = "2"``."""
    from safetensors.torch import save_file

    b = head.base
    meta = {
        "width": str(b.proj.out_features),
        "depth": str(len(b.blocks) + len(b.up_blocks)),
        "up": str(b.up),
        "skip": "0",
        "x2_depth": str(len(head.up2_blocks)),
        "scales": ",".join(str(v) for v in head_scales(head)),
        **{k: str(v) for k, v in stamp.items()},
    }
    save_file(
        {k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()},
        str(path),
        metadata=meta,
    )
