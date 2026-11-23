"""DINOv3 distance on the decoded x̂0, through a VAE window, as an auxiliary LoRA loss
(experimental).

The LPIPS version of this term (``project/pixel_decoder/reports/lpips_lora_2026_11_21.md``,
removed in 9d1240ee) with LPIPS-VGG replaced by the pixel heads' DINOv3 tile distance
(``library.anima.pixel_head_train.DinoFeat``, ViTok-v2's term).

Per train sample whose σ ≤ ``pixel_dino_sigma_max``: one random window of
``pixel_dino_window`` + 2 × ``pixel_dino_margin`` tokens; x̂0 = x_t − σ·v on it is decoded
by the frozen bf16 VAE and compared with the same decode of the clean latent, margins left
out, on ``pixel_dino_tiles`` 224-px tiles at the same places in both. Only the VAE and the
tower are back-propagated here: the gradient g = ∂D/∂v is taken at once and the term handed
to the composer is

    λ · (D + ⟨g, v − sg(v)⟩)

whose value is λ·D and whose gradient into the DiT is λ·g. λ is either fixed
(``pixel_dino_weight``) or set per sample so that ‖λ·g‖ = ``pixel_dino_ratio`` ·
‖∂L_FM/∂v‖, the FM gradient taken as 2·(v − (ε − x0)) / N (uniform weighting, plain MSE).

Memory: the VAE backward (2.5 GB at the default window) runs while the DiT's graph is alive,
so at the default preset the term needs ``gradient_checkpointing`` on a 16 GB card.

Switched on by ``--pixel_dino_ratio`` or ``--pixel_dino_weight`` > 0
(``resolve_adapters``); the composer's ``pixel_dino`` entry adds the term as it comes.
``--pixel_dino_measure`` alone logs the same distance on the same windows without a term
(the control arm of an A/B: under ``--paired_step_rng`` both arms read the same draws).
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

import torch

from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
)

logger = logging.getLogger(__name__)

TOKEN_PX = 16  # pixels per DiT token side (VAE /8 × patch 2)
DINO_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"


def pick_window(gh: int, gw: int, window: int, margin: int, rng: random.Random):
    """A random token window (y0, x0, h, w) of ``window`` + 2 × ``margin`` a side (the
    whole grid on a shorter axis) and its margins (top, bottom, left, right) in tokens:
    ``margin`` on a side inside the image, 0 on an image edge."""
    wt = window + 2 * margin
    h, w = min(gh, wt), min(gw, wt)
    y0, x0 = rng.randint(0, gh - h), rng.randint(0, gw - w)
    mar = (
        0 if y0 == 0 else margin,
        0 if y0 + h == gh else margin,
        0 if x0 == 0 else margin,
        0 if x0 + w == gw else margin,
    )
    return y0, x0, h, w, mar


def window_term(
    v_c: torch.Tensor,
    xt_c: torch.Tensor,
    lat_c: torch.Tensor,
    sigma: float,
    decode: Callable[[torch.Tensor], torch.Tensor],
    dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mar_px: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``dist_fn`` of decode(x_t − σ·v) against decode(x0) on one latent window, and its
    gradient into v.

    ``v_c`` / ``xt_c`` / ``lat_c`` are (C, h, w) fp32 crops; ``decode`` maps a (C, h, w)
    latent to a (3, H, W) image in [-1, 1]; ``mar_px`` = (top, bottom, left, right) pixels
    left out of the comparison. Returns (distance, g = ∂D/∂v), both detached; no graph is
    kept."""
    t, b, lf, r = mar_px

    def inner(img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        return img[:, t : h - b, lf : w - r]

    with torch.no_grad():
        tgt = inner(decode(lat_c))
    leaf = v_c.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = dist_fn(inner(decode(xt_c - sigma * leaf)), tgt)
        (g,) = torch.autograd.grad(loss, leaf)
    return loss.detach(), g.detach()


def surrogate(
    v_c: torch.Tensor, value: torch.Tensor, g: torch.Tensor, lam: float
) -> torch.Tensor:
    """λ·(value + ⟨g, v − sg(v)⟩): value λ·D, gradient λ·g into ``v_c``."""
    return lam * (value + (g * (v_c - v_c.detach())).sum())


def fm_grad_norm(v: torch.Tensor, noise: torch.Tensor, lat: torch.Tensor) -> float:
    """‖∂ mean((v − (ε − x0))²) / ∂v‖ of one sample."""
    return (2.0 * (v.float() - (noise - lat).float()) / v.numel()).norm().item()


class PixelDinoMethodAdapter(MethodAdapter):
    name = "pixel_dino"
    sigma_demote_safe = False  # a demoted step's latent is not the cached image's

    def __init__(self) -> None:
        self._metrics: dict[str, float] = {}
        self._decode: Optional[Callable] = None
        self._dist: Optional[Callable] = None

    def on_network_built(self, ctx: SetupCtx) -> None:
        args = ctx.args
        self.ratio = float(getattr(args, "pixel_dino_ratio", 0.0) or 0.0)
        self.weight = float(getattr(args, "pixel_dino_weight", 0.0) or 0.0)
        self.measure_only = self.ratio <= 0 and self.weight <= 0
        if self.ratio > 0 and self.weight > 0:
            raise ValueError("pixel_dino: set pixel_dino_ratio or pixel_dino_weight")
        self.sigma_max = float(getattr(args, "pixel_dino_sigma_max", 0.7))
        self.window = int(getattr(args, "pixel_dino_window", 24))
        self.margin = int(getattr(args, "pixel_dino_margin", 4))
        self.tiles = int(getattr(args, "pixel_dino_tiles", 4))
        if getattr(args, "sigma_lowres", False):
            raise ValueError("pixel_dino: not built for sigma_lowres steps")
        if getattr(args, "weighting_scheme", None) in (
            "sigma_sqrt",
            "cosmap",
            "min_snr",
        ):
            logger.warning(
                "pixel_dino: the ratio rule takes the FM gradient at uniform weighting"
            )
        seed = int(getattr(args, "seed", 0) or 0)
        self._rng = random.Random(seed + 9173)  # the windows
        self._tile_rng = random.Random(seed + 9181)  # the tiles inside them
        self._load(args, ctx.accelerator.device)
        logger.info(
            f"pixel_dino: σ ≤ {self.sigma_max}, window {self.window} + 2 × "
            f"{self.margin} tokens, {self.tiles} tiles, "
            + (
                "measured only"
                if self.measure_only
                else f"ratio {self.ratio}"
                if self.ratio > 0
                else f"weight {self.weight}"
            )
        )

    def _load(self, args, device) -> None:
        from library.anima.pixel_head_train import DinoFeat
        from library.env import resolve_under_home
        from library.models import qwen_vae

        vae = qwen_vae.load_vae(
            str(resolve_under_home(args.vae)),
            device="cpu",
            dtype=torch.bfloat16,
            eval=True,
            vae_2d=True,
        )
        vae.to(device).requires_grad_(False)
        dino = DinoFeat(device, getattr(args, "pixel_dino_repo", None) or DINO_REPO)
        self._decode = lambda lat: vae.decode_to_pixels(lat[None])[0].float()
        # the tiles sit at the same places in both images; the clean decode carries no grad
        self._dist = lambda a, b: dino.term(
            a, b, self._tile_rng, self.tiles, compiled=False
        )

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        if not primary.is_train:
            return None
        v = primary.model_pred[:, :, 0]  # (B, C, H, W), carries the DiT graph
        x_t = primary.noisy_model_input[:, :, 0]
        lat, noise = primary.latents, primary.noise
        sig = primary.timesteps.reshape(-1).float()
        bsz = v.shape[0]
        terms, values, lams, sigs = [], [], [], []
        with torch.autocast(v.device.type, enabled=False):
            for b in range(bsz):
                sigma = float(sig[b])
                if sigma > self.sigma_max:
                    continue
                gh, gw = v.shape[-2] // 2, v.shape[-1] // 2
                y0, x0, h, w, mar = pick_window(
                    gh, gw, self.window, self.margin, self._rng
                )
                ys, xs = slice(2 * y0, 2 * (y0 + h)), slice(2 * x0, 2 * (x0 + w))
                v_c = v[b, :, ys, xs].float()
                value, g = window_term(
                    v_c,
                    x_t[b, :, ys, xs].float(),
                    lat[b, :, ys, xs].float(),
                    sigma,
                    self._decode,
                    self._dist,
                    tuple(k * TOKEN_PX for k in mar),
                )
                values.append(value.item())
                sigs.append(sigma)
                if self.measure_only:
                    continue
                if self.ratio > 0:
                    lam = (
                        self.ratio
                        * fm_grad_norm(v[b].detach(), noise[b], lat[b])
                        / (g.norm().item() + 1e-20)
                    )
                else:
                    lam = self.weight
                terms.append(surrogate(v_c, value, g, lam))
                lams.append(lam)
        self._metrics["pixel_dino/active"] = len(values) / bsz
        if not values:
            return None
        self._metrics["pixel_dino/dist"] = sum(values) / len(values)
        self._metrics["pixel_dino/sigma"] = sum(sigs) / len(sigs)
        if not terms:
            return None
        self._metrics["pixel_dino/lambda"] = sum(lams) / len(lams)
        return {"pixel_dino": torch.stack(terms).sum() / bsz}

    def metrics(self, ctx) -> dict[str, float]:
        del ctx
        return dict(self._metrics)
