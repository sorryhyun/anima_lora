#!/usr/bin/env python3
"""Free-form inpaint mask generator for the EasyControl ``inpaint`` task.

A LaMa-style synthetic hole generator: a few thick random brush strokes
(random-walk polylines with random width + round joins) plus 1–2 axis-aligned
rectangles, covering a target fraction of the frame. The hole is painted with
flat **mid-gray** (the SD-inpaint convention — ``128`` in ``[0, 255]`` maps to
``0`` in the VAE's ``[-1, 1]`` input space), which VAE-encodes to a flat latent
region the per-block ``b_cond`` gate + cond LoRA learn to read as "regenerate
here."

Public API:
    generate_mask(h, w, seed, ...) -> np.ndarray[bool]   # True = hole
    mask_array(img_rgb, seed, ...) -> np.ndarray[uint8]  # gray-filled copy
"""

from __future__ import annotations

import cv2
import numpy as np

# Mid-gray fill in uint8 [0, 255] image space. The VAE input transform is
# x/127.5 - 1, so 128 → ~0.004 ≈ 0 in [-1, 1] (SD-inpaint convention). Black
# would collide with legitimately-black art.
GRAY = 128


def _draw_strokes(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Paint a few thick, *compact* brush dabs onto ``mask`` (255 = hole).

    Each stroke is a short random walk (1–3 short segments) drawn thick, so it
    reads as a localized blob rather than a thin snake meandering across the whole
    page. Both the thickness and the per-segment step scale with the *short* edge
    so the dabs stay compact and consistent across the multi-resolution tier set.
    """
    h, w = mask.shape
    short = min(h, w)
    n_strokes = int(rng.integers(1, 4))
    for _ in range(n_strokes):
        # Thick: a substantial fraction of the short edge (≈7–14%), so the dab
        # has real area even with few/short segments.
        thickness = int(rng.integers(max(14, short // 14), max(28, short // 7)))
        n_verts = int(rng.integers(1, 4))  # short: 1–3 segments, not 4–11
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        cv2.circle(mask, (x, y), thickness // 2, 255, -1)  # round cap at the start
        for _ in range(n_verts):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            length = rng.uniform(0.03, 0.10) * short  # short steps → stays compact
            nx = int(np.clip(x + length * np.cos(angle), 0, w - 1))
            ny = int(np.clip(y + length * np.sin(angle), 0, h - 1))
            cv2.line(mask, (x, y), (nx, ny), 255, thickness)
            cv2.circle(mask, (nx, ny), thickness // 2, 255, -1)  # round the joins
            x, y = nx, ny


def _draw_rects(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Paint 1–2 random axis-aligned filled rectangles onto ``mask``."""
    h, w = mask.shape
    n_rect = int(rng.integers(1, 3))
    for _ in range(n_rect):
        rw = max(1, int(rng.uniform(0.1, 0.4) * w))
        rh = max(1, int(rng.uniform(0.1, 0.4) * h))
        x0 = int(rng.integers(0, max(1, w - rw)))
        y0 = int(rng.integers(0, max(1, h - rh)))
        cv2.rectangle(mask, (x0, y0), (x0 + rw, y0 + rh), 255, -1)


def generate_mask(
    h: int,
    w: int,
    seed: int,
    *,
    min_frac: float = 0.10,
    max_frac: float = 0.40,
    max_tries: int = 8,
) -> np.ndarray:
    """Deterministic free-form hole mask for a ``(h, w)`` frame.

    Returns a boolean array (``True`` = hole). The same ``seed`` always yields
    the same mask (per-stem stability — feed ``_stable_seed(stem)``). Strokes +
    rectangles are re-rolled (deterministically, by perturbing the seed) until
    the covered fraction lands in ``[min_frac, max_frac]``; if no attempt does,
    the attempt closest to the band midpoint is returned, so a mask is always
    produced.
    """
    target = 0.5 * (min_frac + max_frac)
    best: np.ndarray | None = None
    best_gap = float("inf")
    for t in range(max_tries):
        rng = np.random.default_rng((seed + t * 7919) & 0xFFFFFFFF)
        mask = np.zeros((h, w), np.uint8)
        _draw_strokes(mask, rng)
        _draw_rects(mask, rng)
        frac = float(mask.mean()) / 255.0
        if min_frac <= frac <= max_frac:
            return mask.astype(bool)
        gap = abs(frac - target)
        if gap < best_gap:
            best, best_gap = mask, gap
    assert best is not None
    return best.astype(bool)


def mask_array(
    img_rgb: np.ndarray,
    seed: int,
    *,
    min_frac: float = 0.10,
    max_frac: float = 0.40,
    fill: int = GRAY,
) -> np.ndarray:
    """Return a copy of ``img_rgb`` (H, W, 3 uint8) with a seeded hole gray-filled."""
    h, w = img_rgb.shape[:2]
    hole = generate_mask(h, w, seed, min_frac=min_frac, max_frac=max_frac)
    out = img_rgb.copy()
    out[hole] = fill
    return out
