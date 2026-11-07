"""Random holes for ``TrainRequest.ref_gate``: per step, a set of boxes (fractions of the
page) rasterised to the image and reference grids, and the ``(rows, keep)`` index pair
``model._attention`` reads: the image queries in the hole, the reference keys outside it.

A hole is a text layout from ``gate_holes`` (another page's OCR boxes, flipped at random
and shifted by up to ``SHIFT``), with ``BOX_RATE`` of the draws adding 1–3 random boxes;
with no layouts, random boxes alone. A draw whose hole covers under 2 cells, or more than
``MAX_COVER`` of the page, is drawn again (``TRIES`` times, then the step has no hole).
"""

from __future__ import annotations

import json
import math
import random

import torch

from library.env import resolve_under_home

SHIFT = 0.05
BOX_RATE = 0.5
BOX_SIDE = (0.05, 0.3)
MAX_COVER = 0.5
TRIES = 10


def cells(boxes, h: int, w: int) -> torch.Tensor:
    """Bool ``(h, w)``: a cell is in the hole when a box overlaps it."""
    out = torch.zeros(h, w, dtype=torch.bool)
    for x0, y0, x1, y1 in boxes:
        c0, c1 = max(0, math.floor(x0 * w)), min(w, math.ceil(x1 * w))
        r0, r1 = max(0, math.floor(y0 * h)), min(h, math.ceil(y1 * h))
        if c1 > c0 and r1 > r0:
            out[r0:r1, c0:c1] = True
    return out


class HoleSampler:
    def __init__(self, path: str, seed: int):
        self.layouts = (
            [ly for ly in json.loads(resolve_under_home(path).read_text()) if ly]
            if path
            else []
        )
        self.rng = random.Random(f"holes:{seed}")
        self.cover: list[float] = []

    def boxes(self) -> list[list[float]]:
        rng, out = self.rng, []
        if self.layouts:
            flip = rng.random() < 0.5
            dx, dy = rng.uniform(-SHIFT, SHIFT), rng.uniform(-SHIFT, SHIFT)
            for x0, y0, x1, y1 in rng.choice(self.layouts):
                if flip:
                    x0, x1 = 1 - x1, 1 - x0
                out.append([x0 + dx, y0 + dy, x1 + dx, y1 + dy])
        if not out or rng.random() < BOX_RATE:
            for _ in range(rng.randint(1, 3)):
                bw, bh = rng.uniform(*BOX_SIDE), rng.uniform(*BOX_SIDE)
                x0, y0 = rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh)
                out.append([x0, y0, x0 + bw, y0 + bh])
        return out

    def gate(self, grid, ref_grid, device):
        """``(rows, keep)`` on ``device`` for an image grid ``(h, w)`` and a reference
        grid ``(rh, rw)``, or None when no draw fits."""
        h, w = grid
        for _ in range(TRIES):
            boxes = self.boxes()
            hole = cells(boxes, h, w).flatten()
            ref_hole = cells(boxes, *ref_grid).flatten()
            n, n_keep = int(hole.sum()), int((~ref_hole).sum())
            if 2 <= n <= MAX_COVER * hole.numel() and n_keep >= 2:
                self.cover.append(n / hole.numel())
                rows = hole.nonzero().flatten()
                keep = (~ref_hole).nonzero().flatten()
                return rows.to(device), keep.to(device)
        return None
