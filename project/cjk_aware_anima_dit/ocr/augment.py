#!/usr/bin/env python3
"""Train-time crop augmentation for the SFX reader (``plan_ocr.md`` O1, decided here).

The O1 crops are cut once at 12 % pad, orientation preserved, and stay fixed
(the eval reads the same files). Everything that varies is applied here, in
the training loader, on the BGR ``uint8`` crop:

    from augment import Augment
    aug = Augment(seed=0)
    crop = aug(crop)                      # one random draw per call

Draws (each an independent Bernoulli, probabilities in :class:`Augment`):

* **pad jitter** 5–25 % — the crop already carries 12 %; below that we cut
  inward, above it we extend with the crop's border colour (the page context
  is not in the file). Also shifts the glyphs off centre.
* **rotation** ±8° (border replicate) — COO polygons are hand-drawn and the
  ``minAreaRect`` deskew leaves a residual tilt; sincos SFX are often diagonal.
* **scale jitter** — downscale to 0.5–1.0 and back (area / cubic): sincos min
  side p10 is 33 px, COO's is 25 px; the reader must not depend on stroke width.
* **JPEG** quality 30–95 — doujin pages come through image boards as JPEG.
* **contrast / brightness** — gamma 0.6–1.5 and a ±30 level shift; screentone
  vs flat tint changes the local contrast of the strokes.
* **invert** — white-on-dark lettering (outlined SFX on dark panels).
* **colour tint** — the domain-gap draw: the grey crop's darkness becomes an
  alpha over a *background* tint (skin / pastel / flat pink) with the strokes in
  a *text* tint (pink / red / white / dark). sincos SFX are pink on skin tones;
  COO is grey on white. Roles flip with a small probability.

``python augment.py --demo`` writes a contact sheet of 8 manifest crops × 6
draws to ``$MANGA109S_ROOT/../derived/aug_demo.png`` for a visual check.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# background tints (BGR): skin tones, pastels, flat panel colours
BG_TINTS = [
    (214, 228, 255),  # light skin
    (190, 210, 250),  # skin
    (170, 190, 240),  # tan
    (250, 240, 255),  # pale pink
    (230, 215, 255),  # pink
    (255, 245, 235),  # pale blue
    (235, 255, 240),  # pale green
    (250, 250, 250),  # near white
]
# text tints (BGR)
TEXT_TINTS = [
    (150, 70, 255),  # pink
    (120, 40, 230),  # magenta-red
    (60, 40, 220),  # red
    (30, 30, 30),  # near black
    (90, 60, 120),  # plum
    (255, 255, 255),  # white (with a dark background draw below)
]
DARK_BG = [(60, 40, 90), (40, 40, 40), (90, 60, 130), (120, 60, 160)]


def _border_color(img: np.ndarray) -> tuple[int, int, int]:
    b = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], axis=0)
    return tuple(int(v) for v in np.median(b, axis=0))


def pad_jitter(img: np.ndarray, rng: random.Random, lo=0.05, hi=0.25, base=0.12):
    """Re-pad from ``base`` to a random fraction of the max side, per side."""
    h, w = img.shape[:2]
    m = max(h, w)
    # the glyph extent is the crop minus the base pad on each side
    glyph = m / (1 + 2 * base)
    target = rng.uniform(lo, hi)
    color = _border_color(img)
    out = img
    # per-side offsets: negative = cut inward, positive = extend outward
    deltas = [
        int(round((target - base) * glyph * rng.uniform(0.6, 1.4))) for _ in range(4)
    ]
    t, b, lf, rt = deltas
    # cut inward first (never past a third of the side)
    ct, cb, cl, cr = (max(0, -d) for d in (t, b, lf, rt))
    ct, cb = min(ct, h // 3), min(cb, h // 3)
    cl, cr = min(cl, w // 3), min(cr, w // 3)
    out = out[ct : h - cb, cl : w - cr]
    et, eb, el, er = (max(0, d) for d in (t, b, lf, rt))
    if et or eb or el or er:
        out = cv2.copyMakeBorder(out, et, eb, el, er, cv2.BORDER_CONSTANT, value=color)
    return out


def rotate(img: np.ndarray, rng: random.Random, deg=8.0):
    h, w = img.shape[:2]
    a = rng.uniform(-deg, deg)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(
        img, rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def scale_jitter(img: np.ndarray, rng: random.Random, lo=0.5):
    h, w = img.shape[:2]
    s = rng.uniform(lo, 1.0)
    nh, nw = max(8, int(h * s)), max(8, int(w * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def jpeg(img: np.ndarray, rng: random.Random, lo=30, hi=95):
    q = rng.randint(lo, hi)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def contrast(img: np.ndarray, rng: random.Random):
    gamma = rng.uniform(0.6, 1.5)
    shift = rng.uniform(-30, 30)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0 + shift, 0, 255).astype(
        np.uint8
    )
    return cv2.LUT(img, lut)


def invert(img: np.ndarray, rng: random.Random):
    return 255 - img


def tint(img: np.ndarray, rng: random.Random):
    """Grey darkness → alpha; strokes in a text tint over a background tint."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # stretch so the paper is ~1 and the ink ~0 even on screentone
    lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
    alpha = 1.0 - np.clip((gray - lo) / max(hi - lo, 1e-3), 0, 1)
    if rng.random() < 0.15:  # white / light lettering on a dark panel
        bg = rng.choice(DARK_BG)
        fg = rng.choice([(255, 255, 255), (230, 200, 255), (200, 230, 255)])
    else:
        bg = rng.choice(BG_TINTS)
        fg = rng.choice(TEXT_TINTS[:-1])
    bg = np.asarray(bg, np.float32)
    fg = np.asarray(fg, np.float32)
    # keep some of the original luminance texture inside the strokes
    tex = (0.7 + 0.3 * gray)[..., None]
    out = (
        bg[None, None] * (1 - alpha[..., None])
        + fg[None, None] * tex * alpha[..., None]
    )
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class Augment:
    seed: int = 0
    p_pad: float = 0.8
    p_rotate: float = 0.5
    p_scale: float = 0.4
    p_jpeg: float = 0.5
    p_contrast: float = 0.6
    p_invert: float = 0.1
    p_tint: float = 0.35
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        r = self.rng
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if r.random() < self.p_pad:
            img = pad_jitter(img, r)
        if r.random() < self.p_rotate:
            img = rotate(img, r)
        if r.random() < self.p_tint:
            img = tint(img, r)
        if r.random() < self.p_contrast:
            img = contrast(img, r)
        if r.random() < self.p_invert:
            img = invert(img, r)
        if r.random() < self.p_scale:
            img = scale_jitter(img, r)
        if r.random() < self.p_jpeg:
            img = jpeg(img, r)
        return img


# --------------------------------------------------------------------------- demo


def _demo(n_crops: int, n_draws: int, cell: int, seed: int):
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import manga109 as m109

    derived = m109.derived_root()
    df = pd.read_parquet(derived / "manifest.parquet")
    rng = random.Random(seed)
    sfx = df[df.kind == "sfx"].sample(n_crops - n_crops // 4, random_state=seed)
    sp = df[df.kind == "speech"].sample(n_crops // 4, random_state=seed)
    picks = pd.concat([sfx, sp])
    aug = Augment(seed=rng.randint(0, 1 << 30))

    def fit(im):
        h, w = im.shape[:2]
        s = min(cell / h, cell / w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.full((cell, cell, 3), 128, np.uint8)
        h, w = im.shape[:2]
        canvas[
            (cell - h) // 2 : (cell - h) // 2 + h, (cell - w) // 2 : (cell - w) // 2 + w
        ] = im
        return canvas

    rows = []
    for _, row in picks.iterrows():
        img = cv2.imread(str(derived / row.path))
        tiles = [fit(img)] + [fit(aug(img.copy())) for _ in range(n_draws)]
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    out = derived / "aug_demo.png"
    cv2.imwrite(str(out), sheet)
    print(out, [r.text for _, r in picks.iterrows()])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n_crops", type=int, default=8)
    ap.add_argument("--n_draws", type=int, default=6)
    ap.add_argument("--cell", type=int, default=160)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    a = ap.parse_args()
    if a.demo:
        _demo(a.n_crops, a.n_draws, a.cell, a.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
