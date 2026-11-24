"""×2 pixel-head training: what ``scripts/pixel_head/train_x2.py`` composes.

The recipe (``project/pixel_decoder/README.md``): a frozen Anima DiT is run once per
rollout on a noised cached latent at a decode point σ; the head reads the feature before
``final_layer.linear`` and x_t's latent patches and writes the 2× image, trained against
the real 2× image the latent's own image was made from (the ``x2flex`` set). One random
window of ``window`` + 2 × ``margin`` tokens per rollout goes into a replay buffer, and the
loss leaves ``margin`` tokens out on every window side that is not an image edge.

- :func:`load_x2flex` / :class:`X2Item` — the manifest rows, by split.
- :func:`pick_window` / :class:`Window` — the loss window and its margins.
- :class:`X2Trainer` — replay buffer, schedule, the PE-GAN weight rule.
- :func:`save_x2_head` — the file :func:`library.anima.pixel_head.load_pixel_head` reads.

The shared pieces (rollout, losses, read) are in ``_train.py``.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from library.anima.pixel_head import (
    PATCH,
    PixelHeadX2,
    head_scales,
)
from scripts.pixel_head.train._common import (
    DinoFeat,
    PEDisc,
    _grad_norm,
    cosine_lr,
    dc_loss,
    flat_cells,
    npz_tokens,
    rec_loss,
)

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


# --------------------------------------------------------------------------- trainer


@dataclass
class X2TrainConfig:
    """The loop's knobs; defaults are the ×2 recipe's (``configs/pixel_head/x2.toml``)."""

    rollouts: int = 4000
    opt_steps: int = 16  # head updates per rollout, each on one replay window
    replay: int = 64  # windows kept
    window: int = 24  # loss window, tokens
    margin: int = 3  # context strip, tokens (the head's receptive field)
    lr: float = 2e-4
    warmup: int = 200  # opt steps
    # DINO feature term (DinoFeat) on the loss window. 245.2: its gradient norm over a
    # trained head equals that of the 0.1 × LPIPS it replaced (x2dv3sw_1123_0718)
    lambda_dino: float = 245.2
    dino_tiles: int = 4  # 224-px tiles per update
    lambda_dc: float = 1.0
    dc_block: int = 32  # ×2 px
    # PE-GAN: once per rollout the term's weight is set so its gradient norm over the
    # head is `pegan_ratio` times that of rec2 + λ_dino·DINO + λ_dc·dc. 0 = no term.
    pegan_ratio: float = 2.0
    pegan_start: int = 1000  # first rollout with the term and the discriminator
    pegan_lambda_max: float = 1.0
    pegan_crop: int = 512  # ×2 px, a multiple of 16
    # the discriminator's real crop comes from another replay window, at its own place:
    # no logit can rest on comparing the two crops pixel by pixel
    pegan_unpaired: bool = False
    # > 0: the GAN term sends no gradient into the 32-px cells whose target is flat
    # (flat_cells at this many levels). Its weight is still set on the whole crop, so
    # the other cells take the push they take without the mask.
    pegan_flat_level: float = 0.0
    seed: int = 0


class X2Trainer:
    """Replay buffer + the head's updates for one rollout at a time.

    ``push`` adds a rollout's window; ``update(r)`` runs rollout ``r``'s ``opt_steps``
    updates, each on one random replay window, and returns the last step's terms. The
    head runs at ``scale=2`` only: a file of this recipe carries no trained ×1 output."""

    def __init__(
        self,
        head: PixelHeadX2,
        cfg: X2TrainConfig,
        dino: DinoFeat | None = None,
        disc: PEDisc | None = None,
        compile: bool = False,
    ):
        assert cfg.lambda_dino == 0 or dino is not None, "lambda_dino needs a DinoFeat"
        assert cfg.pegan_ratio == 0 or disc is not None, "pegan_ratio needs a PEDisc"
        assert cfg.pegan_crop % 16 == 0, "pegan_crop: a multiple of 16"
        self.head, self.cfg, self.dino, self.disc = head, cfg, dino, disc
        self.head_c = torch.compile(head, dynamic=True) if compile else head
        # every value of a plain head; the factors alone on one with a LoRA (_lora.py)
        self.params = [q for q in head.parameters() if q.requires_grad]
        self.opt = torch.optim.AdamW(
            self.params, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.01
        )
        self.total_steps = cfg.rollouts * cfg.opt_steps
        self.step = 0
        self.replay: deque = deque(maxlen=cfg.replay)
        self._replay_rng = random.Random(cfg.seed + 3)
        self._crop_rng = random.Random(cfg.seed + 9)
        self.lam_pegan = 0.0
        self.gan_live = False
        self._dino_rng = random.Random(cfg.seed + 11)  # the tiles of an update
        self._pair_rng = random.Random(cfg.seed + 13)  # the unpaired real crop

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

    def terms(
        self, sample, compiled: bool = True, flat_off: bool = True
    ) -> dict[str, torch.Tensor]:
        """Unweighted loss terms on one replay window. ``flat_off`` = apply
        ``pegan_flat_level`` (off where the term's weight is set)."""
        bf, blat, b2, mar = sample
        cfg = self.cfg
        with torch.autocast(bf.device.type, dtype=torch.bfloat16):
            x2 = (self.head_c if compiled else self.head)(bf, blat, 2)
        p2, t2 = crop_margin(x2, mar, 2 * PATCH), b2.float()
        rec, mse = rec_loss(p2, t2)
        out = {"rec2": rec, "mse2": mse.detach()}
        if cfg.lambda_dc > 0:
            out["dc"] = dc_loss(p2, t2, cfg.dc_block)
        if cfg.lambda_dino > 0:
            out["dino"] = self.dino.term(
                p2, t2, self._dino_rng, cfg.dino_tiles, compiled
            )
        if self.gan_live:  # the ×2 output and the real 2× image at the same place
            c = cfg.pegan_crop
            h, w = p2.shape[-2:]
            assert min(h, w) >= c, (h, w, c)
            y, x = self._crop_rng.randint(0, h - c), self._crop_rng.randint(0, w - c)
            sl = (..., slice(y, y + c), slice(x, x + c))
            fake, real = p2[None].float()[sl], t2[None][sl]
            if cfg.pegan_flat_level > 0 and flat_off:
                flat = flat_cells(real[0], cfg.pegan_flat_level)
                fake = self.cut_flat(fake, flat)
                out["flat_share"] = flat.float().mean()
            if cfg.pegan_unpaired:
                real = self._other_real(sample, c)
            out["pegan"] = self.disc.gen_term(fake, real)
        return out

    @staticmethod
    def cut_flat(fake: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        """``fake`` with its values kept and no gradient through the ``flat`` pixels."""
        return torch.where(flat, fake.detach(), fake)

    def _other_real(self, sample, c: int) -> torch.Tensor:
        """A ``c``-px crop of the real 2× image of another replay window (of this one
        while the buffer holds no other), at a place of its own."""
        others = [o for o in self.replay if o is not sample] or [sample]
        t2 = self._pair_rng.choice(others)[2].float()
        h, w = t2.shape[-2:]
        assert min(h, w) >= c, (h, w, c)
        y, x = self._pair_rng.randint(0, h - c), self._pair_rng.randint(0, w - c)
        return t2[None, :, y : y + c, x : x + c]

    def regression(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        loss = t["rec2"]
        if "dc" in t:
            loss = loss + cfg.lambda_dc * t["dc"]
        if "dino" in t:
            loss = loss + cfg.lambda_dino * t["dino"]
        return loss

    def _set_gan_weight(self) -> None:
        """λ = ratio · ‖∇ regression‖ / ‖∇ pegan‖ over the head, eager, one window."""
        params = self.params
        t = self.terms(
            self._replay_rng.choice(self.replay), compiled=False, flat_off=False
        )
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
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.step += 1
            log = {k: v.item() for k, v in t.items()}
            if self.head.noise:  # 0 = the noise inputs are unused
                log["noise_w"] = (
                    sum(
                        m.weight.detach().float().pow(2).sum()
                        for m in self.head.noise_in
                    )
                    .sqrt()
                    .item()
                )
            if self.gan_live:
                log["lam_pegan"] = self.lam_pegan
                log.update(self.disc.step())  # on this step's paired crop, detached
        return log


def save_x2_head(head: PixelHeadX2, path: str | Path, stamp: dict[str, str]) -> None:
    """Write the head with the shape metadata ``load_pixel_head`` reads, plus ``stamp``
    (``sigma`` = the trained decode range, and whatever names the run). A head without
    its ×1 output layer (:func:`~library.anima.pixel_head.drop_x1_output`) is stamped
    ``scales = "2"``. A head with a LoRA (``_lora.py``) is written with it folded in:
    the file is a plain head's."""
    from safetensors.torch import save_file

    from scripts.pixel_head.train._lora import has_lora, merged_state

    sd = merged_state(head) if has_lora(head) else head.state_dict()
    b = head.base
    meta = {
        "width": str(b.proj.out_features),
        "depth": str(len(b.blocks) + len(b.up_blocks)),
        "up": str(b.up),
        "x2_depth": str(len(head.up2_blocks)),
        "scales": ",".join(str(v) for v in head_scales(head)),
        **({"x2_noise": str(head.noise)} if head.noise else {}),
        **({"x2_overlap": "1"} if head.overlap else {}),
        **{k: str(v) for k, v in stamp.items()},
    }
    save_file(
        {k: v.detach().contiguous().cpu() for k, v in sd.items()},
        str(path),
        metadata=meta,
    )
