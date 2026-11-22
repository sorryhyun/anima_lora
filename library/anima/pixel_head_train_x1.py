"""×1 pixel-head training: what ``scripts/pixel_head/{base_x1,train_x1}.py`` compose.

The recipe (``project/pixel_decoder/plan.md``): one DiT forward per rollout at a decode
point, the points taken in turn. A ``:real`` point trains the head against the image the
latent was encoded from (with LPIPS, and the PE-GAN term in a finish), any other point
against ``target`` (the VAE's decode of x̂0 or x0). A share of the rollouts are manga
pages. Whole images go into a replay buffer, and each update draws one of them.

Pieces, in the order a run uses them:

- :func:`held_out_split` / :func:`pick_samples` / :func:`pick_text` — the line's held-out
  images; every other cached pair trains.
- :func:`parse_points` — ``"0.0577-0.1364:real 0.5:adapter"`` → the trained and read points.
- :class:`X1Rollout` — (item, point) → (features, x_t patches, the targets asked for).
- :class:`X1Trainer` — replay buffer, schedule, LPIPS and the PE-GAN weight rule.
- :func:`save_x1_head` — the file :func:`library.anima.pixel_head.load_pixel_head` reads.

The shared pieces are in :mod:`library.anima.pixel_head_train`. The arms the recipe left
behind (a P1 linear skip, gradient accumulation) are in
``bench/pixel_readout/experiments/convnext_head.py``.
"""

from __future__ import annotations

import math
import random
import re
from collections import deque
from dataclasses import dataclass
from glob import escape as glob_escape
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from library.anima.pixel_head import PATCH, PixelHead
from library.anima.pixel_head_train import (
    DiTRollout,
    Lpips,
    PEDisc,
    cosine_lr,
    npz_tokens,
    rec_loss,
)
from library.env import resolve_under_home
from library.io.cache import CachedImage, discover_cached_pairs

# --------------------------------------------------------------------------- data

# P1's held-out artists (readout_1118_2219): what pick_samples returns at its defaults
def _local_split(key: str) -> tuple[str, ...]:
    """An artist list of the local split file (``output/pixel_head/x1_split.json``, kept
    out of the repo); () when it is absent."""
    import json

    from library.env import resolve_under_home

    path = resolve_under_home("output/pixel_head/x1_split.json")
    return tuple(json.loads(path.read_text())[key]) if path.exists() else ()


TEST_ARTISTS = _local_split("test_artists")
TEXT_ARTISTS = _local_split("text_artists")  # most "Japanese text" captions in post_image_dataset/resized

# a cached pair, or a page of bench/pixel_readout/build_pages.py
Item = CachedImage | Path


def pick_samples(
    cache_root: Path,
    n_train: int = 256,
    n_test_artists: int = 8,
    per_artist: int = 4,
    max_tokens: int = 4608,
    seed: int = 0,
):
    """P1's picker → (its train images, test images, test artists)."""
    rng = random.Random(seed)
    artists = sorted(d.name for d in cache_root.iterdir() if d.is_dir())
    rng.shuffle(artists)
    test_artists, train_artists = artists[:n_test_artists], artists[n_test_artists:]

    def ok(ci):
        return npz_tokens(ci.npz_path) <= max_tokens

    test = []
    for a in test_artists:
        pairs = [p for p in discover_cached_pairs(str(cache_root / a)) if ok(p)]
        rng.shuffle(pairs)
        test += pairs[:per_artist]
    pool = []
    for a in train_artists:
        pool += [p for p in discover_cached_pairs(str(cache_root / a)) if ok(p)]
    rng.shuffle(pool)
    return pool[:n_train], test, test_artists


def caption_path(ci: CachedImage) -> Path:
    artist = Path(ci.npz_path).parent.name
    return Path(
        resolve_under_home(f"post_image_dataset/resized/{artist}/{ci.stem}.txt")
    )


def pick_text(cache_root: Path, exclude: set[str]) -> list[CachedImage]:
    """Per text artist, the image whose quoted caption text is longest."""
    out = []
    for a in TEXT_ARTISTS:
        best, best_len = None, 0
        for ci in discover_cached_pairs(str(cache_root / a)):
            if ci.npz_path in exclude:
                continue
            p = caption_path(ci)
            if not p.exists():
                continue
            cap = p.read_text(encoding="utf-8")
            if "japanese text" not in cap.lower():
                continue
            n = sum(len(s) for s in re.findall(r'"([^"]*)"', cap))
            if n > best_len:
                best, best_len = ci, n
        if best is not None:
            out.append(best)
    return out


def held_out_split(cache_root: Path) -> dict[str, list[CachedImage]]:
    """{"train" | "test" | "text": cached pairs}: P1's test images, one text image per
    text artist, and every pair outside those images and the test artists."""
    p1_train, test, artists = pick_samples(cache_root)
    assert tuple(artists) == TEST_ARTISTS, (
        f"held-out set did not reproduce: the picker shuffles {cache_root}'s artist "
        f"dirs, and they gave {artists}"
    )
    text = pick_text(cache_root, {ci.npz_path for ci in p1_train})
    held = {ci.npz_path for ci in text}
    train = [
        ci
        for a in sorted(d.name for d in cache_root.iterdir() if d.is_dir())
        if a not in TEST_ARTISTS
        for ci in discover_cached_pairs(str(cache_root / a))
        if ci.npz_path not in held
    ]
    return {"train": train, "test": test, "text": text}


def page_tokens(path: Path) -> int:
    """Tokens of a page, from its stored ``size`` (WxH pixels)."""
    from safetensors import safe_open

    with safe_open(str(path), "pt") as fh:
        w, h = (int(v) for v in fh.metadata()["size"].split("x"))
    return (w // PATCH) * (h // PATCH)


def item_tokens(item: Item) -> int:
    return page_tokens(item) if isinstance(item, Path) else npz_tokens(item.npz_path)


def real_pixels(item: Item) -> torch.Tensor:
    """The image an item's latent was encoded from, (3, H, W) uint8: a cached pair's
    resized image, or a page (stored pixels, else its source resized as the builder did)."""
    from PIL import Image
    from safetensors import safe_open

    if isinstance(item, Path):
        with safe_open(str(item), "pt") as fh:
            if "pixels" in fh.keys():
                return fh.get_tensor("pixels")
            meta = fh.metadata()
        w, h = (int(v) for v in meta["size"].split("x"))
        im = Image.open(meta["source"]).convert("RGB").resize((w, h), Image.LANCZOS)
        return torch.from_numpy(np.array(im)).permute(2, 0, 1)
    src = item.image_path  # x2_flex.py's rows name their ×1 image
    if not src:
        cap = caption_path(item)
        hits = [
            q
            for q in cap.parent.glob(f"{glob_escape(cap.stem)}.*")
            if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ]
        assert hits, f"no resized image for {item.npz_path}"
        src = hits[0]
    im = Image.open(src).convert("RGB")
    return torch.from_numpy(np.array(im)).permute(2, 0, 1)


# --------------------------------------------------------------------------- points


class TrainPoint(NamedTuple):
    lo: float  # σ is drawn uniformly in [lo, hi] per rollout
    hi: float
    adapter: bool
    target: str  # "real" | "xhat0" | "x0"


class ReadPoint(NamedTuple):
    name: str
    sigma: float
    adapter: bool


def parse_points(
    points: str, eval_points: str = "", target: str = "xhat0"
) -> tuple[list[TrainPoint], list[ReadPoint]]:
    """Space-separated ``SIGMA[-SIGMA][:adapter][:real]`` → (trained, read) points.

    A range is read at its two ends and the middle. ``eval_points``
    (``SIGMA[:adapter]``) are read and not trained on; one already read is skipped."""

    def split(spec: str, allowed: set[str]) -> tuple[str, list[str]]:
        sigma, *tags = spec.split(":")
        if not set(tags) <= allowed:
            raise ValueError(f"point {spec}: tags are {sorted(allowed)}")
        return sigma, tags

    def named(sigma: float, adapter: bool) -> ReadPoint:
        return ReadPoint(
            f"{'lora' if adapter else 'base'}_s{sigma:.4g}", sigma, adapter
        )

    train: list[TrainPoint] = []
    read: list[ReadPoint] = []
    for spec in points.split():
        sigma, tags = split(spec, {"adapter", "real"})
        lo_s, _, hi_s = sigma.partition("-")
        lo, hi = float(lo_s), float(hi_s or lo_s)
        if lo > hi:
            raise ValueError(f"point {spec}: LO-HI")
        adapter = "adapter" in tags
        train.append(TrainPoint(lo, hi, adapter, "real" if "real" in tags else target))
        for sg in dict.fromkeys((lo, (lo + hi) / 2, hi)):
            read.append(named(sg, adapter))
    if not train:
        raise ValueError("no training point")
    if len({p.name for p in read}) != len(read):
        raise ValueError(f"duplicate points in {points!r}")
    for spec in eval_points.split():
        sigma, tags = split(spec, {"adapter"})
        point = named(float(sigma), "adapter" in tags)
        if point.name not in {p.name for p in read}:
            read.append(point)
    return train, read


# --------------------------------------------------------------------------- rollout


class X1Rollout(DiTRollout):
    """One DiT forward at a decode point, with the adapter switched as the point says,
    → what the head reads and the targets asked for."""

    def __init__(
        self, anima, vae, device, seed: int = 0, network=None, page_cross=None
    ):
        super().__init__(anima, vae, device, seed)
        self.network, self.page_cross = network, page_cross

    def load(self, item: Item) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(item, Path):
            from library.io.cache import load_cached_latents, load_cached_text_features

            lat = load_cached_latents(item.npz_path)[0].unsqueeze(0).to(self.device)
            cross, _ = load_cached_text_features(item.te_path, variant=0)
            return lat, cross.unsqueeze(0).to(self.device, torch.bfloat16)
        from safetensors.torch import load_file

        lat = load_file(str(item))["latent"].float()[None].to(self.device)
        return lat, self.page_cross

    def real(self, item: Item, hw: tuple[int, int]) -> torch.Tensor:
        """The original image, (3, H, W) in [-1, 1] on the rollout's device."""
        px = real_pixels(item)
        assert tuple(px.shape[-2:]) == hw, (item, tuple(px.shape), hw)
        return px.to(self.device).float() / 127.5 - 1

    @torch.no_grad()
    def __call__(
        self,
        item: Item,
        idx: int,
        sigma: float,
        adapter: bool = False,
        targets: tuple[str, ...] = ("xhat0",),
    ):
        """→ (features, x_t patches, {"real" | "xhat0" | "x0": (3, H, W)})."""
        if self.network is not None:
            # off = each module returns its org_forward: the base DiT's own output
            self.network.set_enabled(adapter)
        lat, cross = self.load(item)
        f, lat_p, x0_hat = self.forward(lat, cross, idx, sigma)
        out = {}
        if "x0" in targets:
            out["x0"] = self.vae.decode_to_pixels(lat)[0].float()
        if "xhat0" in targets:
            out["xhat0"] = self.vae.decode_to_pixels(x0_hat)[0].float()
        if "real" in targets:
            out["real"] = self.real(item, (lat.shape[-2] * 8, lat.shape[-1] * 8))
        return f, lat_p, out


# --------------------------------------------------------------------------- trainer


@dataclass
class X1TrainConfig:
    """The loop's knobs; defaults are the base recipe (``plan.md``)."""

    rollouts: int = 12000
    opt_steps: int = 16  # head updates per rollout, each on one replay sample
    replay: int = 64  # rollouts kept
    lr: float = 5e-5
    warmup: int = 200  # opt steps
    lambda_lpips: float = 0.05  # on the :real samples
    lpips_crop: int = 512  # 0 = the whole image
    # PE-GAN, the finish's term, on the :real samples: once per rollout its weight is
    # set so its gradient norm over the head is `pegan_ratio` times that of
    # rec + λ_lpips·LPIPS. 0 = no term.
    pegan_ratio: float = 0.0
    pegan_crop: int = 0  # a multiple of 16; 0 = the whole image
    pegan_lambda_max: float = 1.0
    seed: int = 0


class X1Trainer:
    """Replay buffer + the head's updates for one rollout at a time.

    ``push`` adds a rollout; ``update()`` runs its ``opt_steps`` updates and returns the
    last step's loss and MSE. ``lpips_ema`` and ``gan_log`` (the last weight and the last
    discriminator update's meters) are kept on the trainer."""

    def __init__(
        self,
        head: PixelHead,
        cfg: X1TrainConfig,
        lpips: Lpips | None = None,
        disc: PEDisc | None = None,
        compile: bool = False,
    ):
        assert cfg.lambda_lpips == 0 or lpips is not None, "lambda_lpips needs Lpips"
        assert cfg.pegan_ratio == 0 or disc is not None, "pegan_ratio needs a PEDisc"
        assert cfg.pegan_crop % 16 == 0, "pegan_crop: a multiple of 16"
        self.head, self.cfg, self.lpips = head, cfg, lpips
        self.disc = disc if cfg.pegan_ratio > 0 else None
        self.head_c = torch.compile(head, dynamic=True) if compile else head
        # without a skip the linear never runs: no updates and no weight decay on it
        params = [
            p
            for n, p in head.named_parameters()
            if head.use_skip or not n.startswith("linear.")
        ]
        self.opt = torch.optim.AdamW(
            params, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.01
        )
        self.total_steps = cfg.rollouts * cfg.opt_steps
        self.step = 0
        self.gan_steps = 0
        self.replay: deque = deque(maxlen=cfg.replay)
        self.lam_gan = 0.0
        self.lpips_ema: float | None = None
        self.gan_log: dict[str, float] = {}
        # the GAN term has its own streams: a run at ratio 0 draws the same samples
        self._replay_rng = random.Random(cfg.seed + 3)
        self._lpips_rng = random.Random(cfg.seed + 5)
        self._gan_rng = random.Random(cfg.seed + 9)  # the crop of an update
        self._lam_rng = random.Random(cfg.seed + 10)  # the weight rule's sample, crops

    def push(self, f, lat, target, real: bool) -> None:
        """A rollout's features, x_t patches and target image."""
        # bf16 features lose nothing: the head's first linear runs under bf16 autocast
        self.replay.append((f.to(torch.bfloat16), lat, target.half(), real))

    def _predict(self, f, lat, compiled: bool = True) -> torch.Tensor:
        with torch.autocast(f.device.type, dtype=torch.bfloat16):
            return (self.head_c if compiled else self.head)(f, lat)

    def _lpips_term(self, pred, tgt, rng: random.Random, compiled: bool = True):
        a, b = pred.float(), tgt.float()
        c = self.cfg.lpips_crop
        if c and min(a.shape[-2:]) > c:
            y = rng.randint(0, a.shape[-2] - c)
            x = rng.randint(0, a.shape[-1] - c)
            a, b = a[:, y : y + c, x : x + c], b[:, y : y + c, x : x + c]
        return self.lpips.term(a, b, compiled)

    def _gan_term(self, pred, tgt, rng: random.Random) -> torch.Tensor:
        """L_G on a paired image, whole or one square crop."""
        a, b = pred[None].float(), tgt[None].float()
        if self.cfg.pegan_crop:
            h, w = pred.shape[-2:]
            c = min(self.cfg.pegan_crop, h, w)
            y, x = rng.randint(0, h - c), rng.randint(0, w - c)
            box = (..., slice(y, y + c), slice(x, x + c))
            a, b = a[box], b[box]
        return self.disc.gen_term(a, b)

    def _set_gan_weight(self, sample) -> None:
        """λ = ratio · ‖∇ regression‖ / ‖∇ GAN‖ over the head, eager, one :real sample."""
        f, lat, tgt, _ = sample
        params = [q for q in self.head.parameters() if q.requires_grad]
        pred = self._predict(f, lat, compiled=False)
        reg = rec_loss(pred, tgt.float())[0]
        if self.cfg.lambda_lpips > 0:
            reg = reg + self.cfg.lambda_lpips * self._lpips_term(
                pred, tgt, self._lam_rng, compiled=False
            )
        gan = self._gan_term(pred, tgt, self._lam_rng)
        norms = []
        for term in (reg, gan):
            gs = torch.autograd.grad(
                term, params, retain_graph=term is reg, allow_unused=True
            )
            sq = sum((g.float() ** 2).sum() for g in gs if g is not None)
            norms.append(math.sqrt(float(sq)))
        lam = self.cfg.pegan_ratio * norms[0] / (norms[1] + 1e-12)
        self.lam_gan = min(lam, self.cfg.pegan_lambda_max)
        self.gan_log["lam"] = self.lam_gan

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        if self.disc is not None:
            reals = [s for s in self.replay if s[3]]
            if reals:
                self._set_gan_weight(self._lam_rng.choice(reals))
        for _ in range(cfg.opt_steps):
            s = cosine_lr(self.step, self.total_steps, cfg.warmup)
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr * s
            self.opt.zero_grad(set_to_none=True)
            f, lat, tgt, real = self._replay_rng.choice(self.replay)
            pred = self._predict(f, lat)
            loss, mse = rec_loss(pred, tgt.float())
            if real and cfg.lambda_lpips > 0:
                lp = self._lpips_term(pred, tgt, self._lpips_rng)
                loss = loss + cfg.lambda_lpips * lp
                v, ema = lp.item(), self.lpips_ema
                self.lpips_ema = v if ema is None else 0.98 * ema + 0.02 * v
            gan = real and self.disc is not None
            if gan:
                g_loss = self._gan_term(pred, tgt, self._gan_rng)
                loss = loss + self.lam_gan * g_loss
                self.gan_log["g_loss"] = g_loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.opt.step()
            self.step += 1
            if gan:  # one discriminator update, on this step's pair, detached
                self.gan_log.update(self.disc.step())
                self.gan_steps += 1
        return {"loss": loss.item(), "mse": mse.item()}


def save_x1_head(head: PixelHead, path: str | Path, stamp: dict[str, str]) -> None:
    """Write the head with the shape metadata ``load_pixel_head`` reads, plus ``stamp``
    (``sigma`` = the trained decode ranges, and whatever names the run)."""
    from safetensors.torch import save_file

    n_up = len(head.up_blocks) if head.up else 0
    meta = {
        "width": str(head.proj.out_features),
        "depth": str(len(head.blocks) + n_up),
        "up": str(head.up),
        "skip": str(int(head.use_skip)),
        **{k: str(v) for k, v in stamp.items()},
    }
    save_file(
        {k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()},
        str(path),
        metadata=meta,
    )
