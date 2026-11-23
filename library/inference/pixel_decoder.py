"""``--decoder pixel_head``: decode with the pixel head instead of the VAE.

The head and its tap ride on the DiT as a plain attribute (``anima._pixel_decoder``, not
a submodule, so state dicts, ``.to()`` and LoRA application never see it). ``generate()``
and the batch path call :func:`ensure_pixel_decoder` before sampling and
:func:`pixel_decode` after; the result replaces the VAE decode.

The head is read only where it was trained (its ``sigma`` stamp: base DiT at shift-3
Euler 20 … 50 steps, the 4-step turbo LoRA at σ 0.5). Elsewhere it still decodes and a
warning goes to the log. The file decides the scale: ``--pixel_scale`` unset decodes at
the file's own (2 for the ×2 file, whose saved image is twice ``--image_size``); an
explicit scale the file does not hold refuses. With no ``--pixel_head``, ``--pixel_scale 2``
takes the catalog's ×2 file. An RGBA file (``channels = 4``) writes straight RGBA (RGB, 2α − 1)
and saves as a transparent PNG.
"""

from __future__ import annotations

import argparse
import logging

import torch

from library.anima.pixel_head import (
    PixelTap,
    check_scale,
    head_scales,
    load_pixel_head,
    out_of_range,
    resolve_pixel_head_path,
)

logger = logging.getLogger(__name__)


def _check(head, scale: int | None, path: str) -> int:
    try:
        return check_scale(head, scale, name=f"--pixel_scale ({path})")
    except ValueError as e:
        raise SystemExit(str(e)) from None


class PixelDecoder:
    def __init__(self, anima, path: str | None, device, scale: int | None = None):
        resolved = resolve_pixel_head_path(path, scale)
        self.path = str(resolved)
        self.head = load_pixel_head(resolved, device=device)
        self.scale = _check(self.head, scale, self.path)
        self.tap = PixelTap(anima)
        adapter = self.head.meta.get("adapter")
        logger.info(
            f"pixel head: {self.path} (scales {head_scales(self.head)}, decoding "
            f"×{self.scale}; "
            f"trained σ {self.head.sigma_ranges}"
            + (f"; turbo point on {adapter}" if adapter else "")
            + ")"
        )


def wants_pixel_head(args: argparse.Namespace) -> bool:
    return getattr(args, "decoder", "vae") == "pixel_head"


def check_pixel_head_args(args: argparse.Namespace) -> None:
    """Settings the tap cannot read at all (not range warnings: those come at decode)."""
    if not wants_pixel_head(args):
        if getattr(args, "pixel_scale", None) is not None:
            logger.warning(
                "--pixel_scale reads the pixel head's output; with --decoder vae it does "
                "nothing."
            )
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
    scale = getattr(args, "pixel_scale", None)
    if dec is None:
        dec = PixelDecoder(anima, getattr(args, "pixel_head", None), device, scale)
        anima._pixel_decoder = dec
    else:
        dec.scale = _check(dec.head, scale, dec.path)
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
    """Pixels (B, C, s·H, s·W), fp32 on the CPU, in [-1, 1], from the last sampling
    (s = ``--pixel_scale``): C = 3, or 4 (straight RGBA) for an RGBA file."""
    dec: PixelDecoder | None = getattr(anima, "_pixel_decoder", None)
    if dec is None:
        raise RuntimeError("pixel_decode: ensure_pixel_decoder was not called")
    _warn_settings(args, dec)
    pixels = dec.tap.decode(dec.head, args.guidance_scale, dec.scale).cpu()
    dec.tap.reset()
    return pixels
