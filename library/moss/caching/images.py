"""Fitting an image to what MOSS encodes: sizing, pixels, background loading."""

from __future__ import annotations

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STRIDE = 16
AREA_TOLERANCE = 1.10


def target_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """``(W, H)``: native aspect, multiples of 16, area ≤ ~``resolution²``, never
    larger than the source on either side."""
    cap = resolution * resolution
    if (
        width % STRIDE == 0
        and height % STRIDE == 0
        and width * height <= cap * AREA_TOLERANCE
    ):
        return width, height
    area = min(cap, width * height)
    aspect = width / height
    out_w = round(math.sqrt(area * aspect) / STRIDE) * STRIDE
    out_h = round(math.sqrt(area / aspect) / STRIDE) * STRIDE
    out_w = max(STRIDE, min(out_w, width // STRIDE * STRIDE))
    out_h = max(STRIDE, min(out_h, height // STRIDE * STRIDE))
    return out_w, out_h


def load_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize to cover ``(width, height)`` and centre-crop; identity at that size."""
    if image.size == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    rw = max(width, math.ceil(image.width * scale))
    rh = max(height, math.ceil(image.height * scale))
    image = image.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    return image.crop((left, top, left + width, top + height))


def to_pixels(image: Image.Image) -> torch.Tensor:
    """RGB PIL → ``(1, 3, H, W)`` fp32 in ``[-1, 1]``."""
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)[None] / 127.5 - 1.0


def prefetch(items, load, ahead: int = 8, workers: int = 4):
    """Yield ``(item, load(item))`` in order while the next ``ahead`` loads run on
    ``workers`` threads, so PNG decode and resize overlap the VAE encode."""
    items = list(items)
    with ThreadPoolExecutor(workers) as ex:
        pending = deque(ex.submit(load, x) for x in items[:ahead])
        for i, x in enumerate(items):
            if i + ahead < len(items):
                pending.append(ex.submit(load, items[i + ahead]))
            yield x, pending.popleft().result()


def ref_size(width: int, height: int, ref_scale: int) -> tuple[int, int]:
    """Reference ``(w, h)`` for a ``width × height`` target: 1/``ref_scale`` of
    each side, multiples of 16."""
    return (
        max(STRIDE, round(width / ref_scale / STRIDE) * STRIDE),
        max(STRIDE, round(height / ref_scale / STRIDE) * STRIDE),
    )
