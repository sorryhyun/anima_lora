#!/usr/bin/env python3
"""Tone of a head against the VAE decode, read off sheets (CPU, 8-bit PNG levels).

Reads the native-resolution part of ``p3_generated.py`` / ``convnext_head.py`` sheets
(three columns; below the 24 px label strip, above the 768 px crop row); the red crop
outline and white padding are masked. Luma is Rec. 709 on the code values for both
columns, so the two tones below are comparable on colour and grey images alike.

Per 32 px block:

- ``HF``: rms of luma minus its 9 px box blur, in 8-bit levels (texture: screentone,
  hatching, line art).
- code shift: block mean luma, head − VAE.
- linear-light shift: mean of the linearised luma mapped back to an sRGB code, head − VAE.
  This is the tone an eye (or a downscale in linear light) averages to; smoothing a
  pattern in code values keeps the code mean and lowers this one.
- full-flatten: what the linear-light shift would be if the VAE's block were replaced by
  its code mean, the bound for a head that smooths everything.

Usage::

    python bench/pixel_readout/tone_probe.py <run_dir> [--glob 'sheet_*.png'] \
        [--vae_col 0] [--head_col 2] [--mono]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

MONO = ("manga_action", "manga_page", "sketch")  # P3 prompt names rendered in greyscale
BLOCK = 32
REC709 = np.array([0.2126, 0.7152, 0.0722])
HF_BINS = ((0, 2), (2, 8), (8, 16), (16, 32), (32, 64), (64, float("inf")))
TONE_BINS = (("dark <96", 0, 96), ("mid 96–200", 96, 200), ("light >200", 200, 256))


def to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_code(v: np.ndarray) -> np.ndarray:
    hi = 1.055 * np.maximum(v, 1e-12) ** (1 / 2.4) - 0.055
    return np.where(v <= 0.0031308, v * 12.92, hi)


def block_mean(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked means over BLOCK × BLOCK cells; NaN where over 10 % of a cell is masked."""
    h, w = (s // BLOCK * BLOCK for s in a.shape)
    a, m = a[:h, :w], mask[:h, :w].astype(float)
    shape = (h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    n = m.reshape(shape).sum((1, 3))
    out = (a * m).reshape(shape).sum((1, 3)) / np.maximum(n, 1)
    out[n < 0.9 * BLOCK * BLOCK] = np.nan
    return out.ravel()


def is_red(x: np.ndarray) -> np.ndarray:
    return (x[..., 0] == 255) & (x[..., 1] == 0) & (x[..., 2] == 0)


def read_sheet(path: Path, vae_col: int, head_col: int):
    """→ (whole-image code shift, linear-light shift, luma std ratio, per-block table).

    Block table columns: VAE HF, head HF, VAE code mean, head code mean, VAE
    linear-light tone, head linear-light tone (8-bit levels)."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    height, width, _ = im.shape
    cw, h = width // 3, height - 24 - 768
    vae = im[24 : 24 + h, vae_col * cw : (vae_col + 1) * cw]
    head = im[24 : 24 + h, head_col * cw : (head_col + 1) * cw]
    pad = (vae.min(-1) == 255).all(0) & (head.min(-1) == 255).all(0)
    mask = ~(is_red(vae) | is_red(head)) & ~pad[None, :]
    yv, yd = (vae / 255) @ REC709, (head / 255) @ REC709
    lv, ld = to_linear(yv), to_linear(yd)

    def hf(y):
        return np.sqrt(np.maximum(block_mean((y - uniform_filter(y, 9)) ** 2, mask), 0))

    table = 255 * np.stack(
        [
            hf(yv),
            hf(yd),
            block_mean(yv, mask),
            block_mean(yd, mask),
            to_code(block_mean(lv, mask)),
            to_code(block_mean(ld, mask)),
        ],
        1,
    )
    code = 255 * (yd[mask].mean() - yv[mask].mean())
    linear = 255 * (to_code(ld[mask].mean()) - to_code(lv[mask].mean()))
    return code, linear, yd[mask].std() / yv[mask].std(), table[~np.isnan(table).any(1)]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("--glob", default="sheet_*.png")
    p.add_argument("--vae_col", type=int, default=0)
    p.add_argument("--head_col", type=int, default=2)
    p.add_argument(
        "--mono", action="store_true", help="every sheet is a greyscale page"
    )
    args = p.parse_args()

    whole: dict[str, list] = {}
    blocks: dict[str, list] = {}
    sheets = sorted(Path(args.run_dir).glob(args.glob))
    for path in sheets:
        name = path.stem.split("_", 3)[-1].rsplit("_s", 1)[0]
        grp = "mono" if args.mono or name in MONO else "colour"
        code, linear, ratio, table = read_sheet(path, args.vae_col, args.head_col)
        whole.setdefault(grp, []).append((code, linear, ratio))
        blocks.setdefault(grp, []).append(table)
    print(f"== {args.run_dir}  {args.glob}  {len(sheets)} sheets")
    for grp, rows in whole.items():
        code, linear, ratio = (np.array(c) for c in zip(*rows))
        print(
            f" {grp} ({len(rows)}): whole image, head − VAE: code {code.mean():+.2f} "
            f"({code.min():+.2f}…{code.max():+.2f})  linear-light {linear.mean():+.2f} "
            f"({linear.min():+.2f}…{linear.max():+.2f})  luma std ratio {ratio.mean():.3f}"
        )
        a = np.concatenate(blocks[grp])
        d_code, d_lin, flat = a[:, 3] - a[:, 2], a[:, 5] - a[:, 4], a[:, 2] - a[:, 4]
        print(
            "   VAE HF bin |      n | HF head/VAE | shift code | shift linear-light "
            "| full-flatten"
        )
        for lo, hi in HF_BINS:
            s = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if s.sum() < 20:
                continue
            label = f"{lo:g}–{'∞' if hi == float('inf') else f'{hi:g}'}"
            print(
                f"   {label:>10} | {s.sum():6d} | {a[s, 1].mean() / a[s, 0].mean():11.3f} "
                f"| {d_code[s].mean():+10.2f} | {d_lin[s].mean():+18.2f} "
                f"| {-flat[s].mean():+.2f}"
            )
        print("   textured blocks (VAE HF ≥ 16) by VAE block tone:")
        for label, lo, hi in TONE_BINS:
            s = (a[:, 0] >= 16) & (a[:, 2] >= lo) & (a[:, 2] < hi)
            if s.sum() < 20:
                continue
            print(
                f"     {label:<11} n {s.sum():5d}  HF head/VAE "
                f"{a[s, 1].mean() / a[s, 0].mean():.3f}  shift code {d_code[s].mean():+.2f}"
                f"  linear-light {d_lin[s].mean():+.2f}  blocks darker in linear light "
                f"{(d_lin[s] < 0).mean() * 100:.0f} %"
            )


if __name__ == "__main__":
    main()
