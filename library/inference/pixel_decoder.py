"""``--decoder pixel_head``: decode with the pixel head instead of the VAE.

The head and its tap ride on the DiT as a plain attribute (``anima._pixel_decoder``, not
a submodule, so state dicts, ``.to()`` and LoRA application never see it). ``generate()``
and the batch path call :func:`ensure_pixel_decoder` before sampling and
:func:`pixel_decode` after; the result replaces the VAE decode.

The head is read only where it was trained (its ``sigma`` stamp: base DiT at shift-3
Euler 20 … 50 steps, the 4-step turbo LoRA at σ 0.5). Elsewhere it still decodes and a
warning goes to the log.
"""

from __future__ import annotations

import argparse
import logging

import torch

from library.anima.pixel_head import (
    PixelTap,
    load_pixel_head,
    out_of_range,
    resolve_pixel_head_path,
)

logger = logging.getLogger(__name__)


class PixelDecoder:
    def __init__(self, anima, path: str | None, device):
        resolved = resolve_pixel_head_path(path)
        self.path = str(resolved)
        self.head = load_pixel_head(resolved, device=device)
        self.tap = PixelTap(anima)
        logger.info(f"pixel head: {self.path} (trained σ {self.head.sigma_ranges})")


def wants_pixel_head(args: argparse.Namespace) -> bool:
    return getattr(args, "decoder", "vae") == "pixel_head"


def check_pixel_head_args(args: argparse.Namespace) -> None:
    """Settings the tap cannot read at all (not range warnings: those come at decode)."""
    if not wants_pixel_head(args):
        return
    if getattr(args, "tiled_diffusion", False):
        raise SystemExit(
            "--decoder pixel_head reads one forward per step; --tiled_diffusion runs "
            "one per tile. Use --decoder vae."
        )


def ensure_pixel_decoder(
    args: argparse.Namespace, anima, device
) -> PixelDecoder | None:
    if not wants_pixel_head(args):
        return None
    dec = getattr(anima, "_pixel_decoder", None)
    if dec is None:
        dec = PixelDecoder(anima, getattr(args, "pixel_head", None), device)
        anima._pixel_decoder = dec
    dec.tap.reset()
    return dec


def _warn_settings(args: argparse.Namespace, dec: PixelDecoder) -> None:
    sigma = dec.tap.sigma
    if dec.tap.tail_cached:
        logger.warning(
            "pixel head: the last step was a Spectrum cached step; the head reads the "
            f"last actual forward (σ {sigma:.4f}), not the final latent. Keep "
            "--spectrum_stop_caching_step before the last step."
        )
    if sigma is not None and out_of_range(sigma, dec.head.sigma_ranges):
        logger.warning(
            f"pixel head: last-step σ {sigma:.4f} is outside the trained "
            f"{dec.head.sigma_ranges} (base: shift-3 Euler 20-50 steps; turbo: 4 steps, "
            "σ 0.5); decoding anyway."
        )
    if args.guidance_scale != 1.0 and (
        getattr(args, "cfgpp", False) or getattr(args, "smc_cfg", False)
    ):
        logger.warning(
            "pixel head: the feature is combined as plain CFG; --cfgpp / --smc_cfg "
            "combine the velocity differently at the last step."
        )


def pixel_decode(args: argparse.Namespace, anima) -> torch.Tensor:
    """Pixels (B, 3, H, W), fp32 on the CPU, in [-1, 1], from the last sampling."""
    dec: PixelDecoder | None = getattr(anima, "_pixel_decoder", None)
    if dec is None:
        raise RuntimeError("pixel_decode: ensure_pixel_decoder was not called")
    _warn_settings(args, dec)
    pixels = dec.tap.decode(dec.head, args.guidance_scale).cpu()
    dec.tap.reset()
    return pixels
