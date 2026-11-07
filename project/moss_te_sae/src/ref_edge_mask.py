"""Inpainting through the reference edges: with ``ref_isolate`` the reference reaches
the main rows only through main-query → reference-key attention, so a logit bias on
those edges for the image queries inside a mask gates the copy in every block and
head at once, with no head selection and no parameters.

``copy_transplant.py`` (20261107-1539) showed the copy is not a transplantable head:
b39 h19 alone, its q / k alone, the ten top copy heads, blocks 36-41, and the full
LoRA on the reference stream with only h19 on the main rows all leave the copy at
Base level or below. The reference encoding and the main rows' reading of it change
together across the network. This run controls the copy at the edges instead.

LoRA: chinoll/Moss-Image-Edit (``copy_head_ext.py``), 1024-edge area, Euler 50,
CFG 4, reference on RoPE frame 0. Per input a box mask and an instruction about the
box content; arms:

- ``lora``: no mask;
- ``hard``: image queries in the box cannot attend to any reference key
  (cond and uncond, every block, head and step);
- ``soft4`` / ``soft2``: those edges get -4 / -2 on the logit instead.

Reads: luma L1 against the reference outside the box (a one-cell ring left out) and
inside it, and the sheet.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import copy_head_ext as ext  # noqa: E402
import library.moss.model as mm  # noqa: E402
from copy_head import log, luma  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

CASES = [
    dict(
        ref="image_dataset/sushispin/10043045.webp",
        prompt="Replace the flower hair ornament with a blue ribbon.",
        box=(48, 48, 288, 256),  # x0, y0, x1, y1 at the 576x1024 output
    ),
    dict(
        ref="output/moss_i2i_eval/sanitize/inputs/usual_case1.png",
        prompt="Replace her horns with white cat ears.",
        box=(288, 80, 688, 224),  # at the 816x720 output
    ),
]
ARMS = {"lora": None, "hard": float("-inf"), "soft4": -4.0, "soft2": -2.0}


class EdgeBias:
    """Adds ``bias`` to main image queries in ``cells`` (bool over the latent grid,
    row-major) against every reference key."""

    def __init__(self, cells, bias):
        self.cells = cells
        self.bias = bias
        self.orig = mm._attention

    def __enter__(self):
        if self.bias is None:
            return self
        orig, cells, bias = self.orig, self.cells, self.bias

        def attention(q, k, v, rope, heads, key_mask=None, ref_start=None,
                      ref_isolate=False, ref_kv=None):  # fmt: skip
            if ref_kv is None or ref_start is not None:
                return orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv)  # fmt: skip
            shape = (*q.shape[:2], heads, q.shape[-1] // heads)
            qr = mm.rotate(q.reshape(shape), rope).to(v.dtype)
            kr = mm.rotate(k.reshape(shape), rope).to(v.dtype)
            vr = v.reshape(shape)
            ref_k, ref_v, ref_rope = ref_kv
            ref_shape = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
            ref_k = mm.rotate(ref_k.reshape(ref_shape), ref_rope).to(v.dtype)
            kr = torch.cat((kr, ref_k), 1)
            vr = torch.cat((vr, ref_v.reshape(ref_shape)), 1)
            B, L, n_ref = q.shape[0], q.shape[1], ref_k.shape[1]
            m = torch.zeros(B, 1, L, L + n_ref, device=q.device, dtype=v.dtype)
            if key_mask is not None:
                km = torch.cat((key_mask, key_mask.new_ones(B, n_ref)), 1)
                m.masked_fill_(~km[:, None, None, :], float("-inf"))
            rows = torch.zeros(L, dtype=torch.bool, device=q.device)
            rows[: cells.numel()] = cells.to(q.device)
            m[:, :, rows, L:] += bias
            return mm._sdpa(qr, kr, vr, m).flatten(2)

        mm._attention = torch.compiler.disable(attention)
        return self

    def __exit__(self, *exc):
        mm._attention = self.orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cases", default="0,1")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    ext.SAMPLER.extend(["--width", "576", "--height", "1024"])
    out = args.out or resolve_under_home(
        f"output/tests/ref_edge_mask/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    te_cache = out / "_te_cache"
    summary = {}
    for n in map(int, args.cases.split(",")):
        case = CASES[n]
        ref = ext.find_image(case["ref"])
        rows = {}
        for arm in args.arms.split(","):
            # the output size is known only after a render; the box is in output px
            size = rows[next(iter(rows))]["size"] if rows else None
            if size is None and ARMS[arm] is not None:
                raise SystemExit("run the unmasked arm first")
            cells = None
            if size is not None:
                W, H = size
                x0, y0, x1, y1 = (c // 16 for c in case["box"])
                grid = torch.zeros(H // 16, W // 16, dtype=torch.bool)
                grid[y0:y1, x0:x1] = True
                cells = grid.flatten()
            t0 = time.time()
            with EdgeBias(cells, ARMS[arm]):
                e = ext.render(
                    ref, case["prompt"], out / f"case{n}" / arm, mult=1, cut=None,
                    capture=False, frame0=True, te_cache=te_cache,
                )  # fmt: skip
            e["size"] = (e["width"], e["height"])
            rows[arm] = e
            log(f"case{n} {arm}: {time.time() - t0:.0f}s")
        W, H = rows["lora"]["size"]
        R = luma(Image.open(ref).resize((W, H)))
        x0, y0, x1, y1 = case["box"]
        inside = np.zeros((H, W), bool)
        inside[y0:y1, x0:x1] = True
        ring = np.zeros((H, W), bool)
        ring[max(y0 - 16, 0) : y1 + 16, max(x0 - 16, 0) : x1 + 16] = True
        outside = ~ring
        tiles = [Image.open(ref).convert("RGB").resize((W, H))]
        for arm, e in rows.items():
            Li = luma(Image.open(e["image"]))
            d = np.abs(Li - R)
            e["luma_outside"] = float(d[outside].mean())
            e["luma_inside"] = float(d[inside].mean())
            im = Image.open(e["image"]).convert("RGB")
            ImageDraw.Draw(im).rectangle(case["box"], outline=(0, 255, 0), width=2)
            tiles.append(im)
            log(
                f"  case{n} {arm:6} L/ref outside {e['luma_outside']:.1f}"
                f" inside {e['luma_inside']:.1f}"
            )
        summary[f"case{n}"] = dict(
            ref=ref, prompt=case["prompt"], box=case["box"], arms=rows
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        tw = 360
        th = int(H * tw / W)
        sheet = Image.new("RGB", (tw * len(tiles), th), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((tw, th)), (i * tw, 0))
        sheet.save(out / f"sheet_case{n}.png")
        log(f"case{n} sheet: ref, {', '.join(rows)}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
