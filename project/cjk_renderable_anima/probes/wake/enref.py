"""EN-reference ruler (2026-09-15, user): a trained ext row should behave like
an EN word token — render the same scene prompt with ``English text reads
as "<word>"`` at the same seed and ask how little the image changes when the
word is swapped for the glyph.

Three numbers per trained render, all against ``enref_p<pi>_s<seed>.png``:

- ``en_cos``      PE-Spatial pooled cos to the EN reference (whole image)
- ``en_cos_out``  the same over the patch tokens *outside* the union of the
                  two text boxes (the glyph's box and the EN word's box) —
                  "is the scene the same once the text is excused"
- ``box_iou``     IoU of the glyph's box and the EN word's box — "did the
                  glyph land where the word would"

The references are arm-independent (no ext id in the caption, so the delta
is inert) and shared under ``output/wake_probe/native_enref/<size>_<steps>_<cfg>/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .common import OUT


def enref_dir(a) -> Path:
    return OUT / "native_enref" / f"{a.eval_size}_{a.steps}_{a.cfg:g}"


def enref_caption(prompt: str, word: str) -> str:
    # mirrors NATIVE_CLAUSES["en"] with the language flipped
    return f'{prompt}, english text. English text reads as "{word}".'


def enref_file(d: Path, pi: int, seed: int) -> Path:
    return d / f"enref_p{pi:02d}_s{seed}.png"


def render_enref(a, prompts, seeds: int, args, gen, shared, vae, device) -> Path:
    """Render the missing references (``generate_to`` skips existing files)."""
    from .models import generate_to

    d = enref_dir(a)
    (d).mkdir(parents=True, exist_ok=True)
    shared["conds_cache"].clear()
    n = 0
    for pi, p in enumerate(prompts):
        for seed in range(seeds):
            fn = enref_file(d, pi, seed)
            if not fn.exists():
                n += 1
            generate_to(
                fn, args, gen, shared, vae, device, enref_caption(p, a.en_word), seed
            )
    (d / "prompts.json").write_text(
        json.dumps(
            {"word": a.en_word, "prompts": prompts}, ensure_ascii=False, indent=1
        )
    )
    print(
        f"enref: {len(prompts)} prompts × {seeds} seeds in {d} ({n} rendered)",
        flush=True,
    )
    return d


def enref_boxes(d: Path, rd=None, device="cuda") -> dict:
    """``{file name: largest detector box or None}`` for every reference,
    cached in ``enref_reads.json`` (the reads are kept too, as a sanity check
    that the word rendered)."""
    from .readers import Readers, load_bgr

    cache = d / "enref_reads.json"
    reads = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [f for f in sorted(d.glob("enref_p*_s*.png")) if f.name not in reads]
    if todo:
        rd = rd or Readers(device)
        for f in todo:
            reads[f.name] = rd.read_image(load_bgr(f), whole=True)
        cache.write_text(json.dumps(reads, ensure_ascii=False, indent=1))
    return {k: largest_box(v) for k, v in reads.items()}


def largest_box(reads: list) -> list | None:
    bs = [r["box"] for r in reads if not r.get("whole")]
    return max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None


def glyph_box(m: dict) -> list | None:
    """The box that read the target (either reader), else the largest box."""
    from .common import norm

    for r in m.get("reads") or []:
        if r.get("whole"):
            continue
        if norm(r.get("sfx") or "") == norm(m["text"]) or norm(
            r.get("vl") or ""
        ) == norm(m["text"]):
            return r["box"]
    return largest_box(m.get("reads") or [])


def box_iou(a: list | None, b: list | None) -> float:
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class EnRef:
    """PE-Spatial token features per image (cached); cos whole / cos outside
    the box union / box IoU against the EN reference."""

    def __init__(self, device, d: Path, boxes: dict):
        import torch

        from library.vision.encoder import (
            encode_pe_from_imageminus1to1,
            load_pe_encoder,
        )

        self.device = torch.device(device)
        self.bundle = load_pe_encoder(self.device, name="pe_spatial")
        self._enc = encode_pe_from_imageminus1to1
        self.d, self.boxes = d, boxes
        self.cache: dict = {}

    def tokens(self, path: Path):
        """``([T-1, D] patch tokens (CLS dropped), (H, W) of the image)``."""
        import numpy as np
        import torch
        from PIL import Image

        key = str(path)
        if key not in self.cache:
            im = Image.open(path).convert("RGB")
            t = torch.from_numpy(np.asarray(im))
            t = (t.permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0)
            with torch.no_grad():
                f = self._enc(self.bundle, t.to(self.device))[0].float()
            self.cache[key] = (f[1:].cpu(), im.size[::-1])
        return self.cache[key]

    @staticmethod
    def _grid(n: int) -> tuple[int, int]:
        g = int(round(math.sqrt(n)))
        assert g * g == n, f"enref: non-square token grid ({n} tokens)"
        return g, g

    def _outside_mask(self, n: int, hw, boxes):
        import torch

        gh, gw = self._grid(n)
        H, W = hw
        keep = torch.ones(gh, gw, dtype=torch.bool)
        for b in boxes:
            if not b:
                continue
            x0, y0 = int(b[0] / W * gw), int(b[1] / H * gh)
            x1, y1 = int(math.ceil(b[2] / W * gw)), int(math.ceil(b[3] / H * gh))
            keep[max(0, y0) : min(gh, y1), max(0, x0) : min(gw, x1)] = False
        return keep.flatten()

    def score(self, m: dict) -> tuple[float, float, float] | None:
        import torch.nn.functional as F

        ref = enref_file(self.d, m["pi"], m["seed"])
        if not ref.exists():
            return None
        fi, hw = self.tokens(Path(m["file"]))
        fr, _ = self.tokens(ref)
        bi, br = glyph_box(m), self.boxes.get(ref.name)
        cos = float(F.cosine_similarity(fi.mean(0), fr.mean(0), dim=0))
        keep = self._outside_mask(fi.shape[0], hw, [bi, br])
        if keep.sum() < 4 or fi.shape[0] != fr.shape[0]:
            cos_out = cos
        else:
            cos_out = float(
                F.cosine_similarity(fi[keep].mean(0), fr[keep].mean(0), dim=0)
            )
        return cos, cos_out, box_iou(bi, br)
