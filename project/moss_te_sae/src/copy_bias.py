"""A logit bias on the reference keys of colorize's copy heads: does weaker copy give
more colour, and at what cost to line alignment.

The copy heads are colorize v5's top eight by own-cell reference mass in
``copy_head.py`` (run 20261105-run1, both wild2 inputs, all steps). For image queries
of those heads, every reference key's logit gets ``+bias`` (cond and uncond branch, all
steps). Two ``wild2`` pages, the shipped colorize generate config, seed 0.

Reads, as ``bench/moss_i2i_eval/colorize_wild.py``: luma r against the page (128 px)
and chroma (mean CbCr distance from grey at 1/8 size).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

import library.moss.generate as gen
import library.moss.model as mm
from bench.moss_i2i_eval.colorize_wild import chroma, luma_r
from copy_head import EVAL, Probe, log
from library.env import resolve_under_home
from library.moss.requests import GenerateRequest

LORA = "output/ckpt_moss_i2i/moss_colorize_v5.safetensors"
CONFIG = "configs/moss/i2i/colorize/generate.toml"
HEADS = {35: [3, 13], 34: [6, 9], 39: [19], 37: [20], 40: [20], 33: [20]}


class Bias(Probe):
    def __init__(self, te_cache, bias):
        super().__init__(te_cache)
        self.bias = bias

    def _attention(self, q, k, v, rope, heads, key_mask=None, ref_start=None,
                   ref_isolate=False, ref_kv=None):  # fmt: skip
        orig = self._orig[0]
        out = orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv)
        hs = HEADS.get(self.block, [])
        if ref_kv is None or ref_start is not None or not hs or self.bias == 0:
            return out
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        qr = mm.rotate(q.reshape(shape), rope).to(v.dtype)
        kr = mm.rotate(k.reshape(shape), rope).to(v.dtype)
        vr = v.reshape(shape)
        ref_k, ref_v, ref_rope = ref_kv
        n_ref = ref_k.shape[1]
        ref_shape = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
        ref_k = mm.rotate(ref_k.reshape(ref_shape), ref_rope).to(v.dtype)
        kr = torch.cat((kr, ref_k), 1)
        vr = torch.cat((vr, ref_v.reshape(ref_shape)), 1)
        L = q.shape[1]
        m = torch.zeros(L, L + n_ref, dtype=vr.dtype, device=q.device)
        m[:n_ref, L:] = self.bias  # image queries (ref_scale 1: n_img = n_ref)
        m = m[None, None]
        if key_mask is not None:
            km = torch.cat((key_mask, key_mask.new_ones(key_mask.shape[0], n_ref)), 1)
            m = m.masked_fill(~km[:, None, None, :], float("-inf"))
        h_t = torch.tensor(hs, device=q.device)
        sub = mm._sdpa(qr[:, :, h_t], kr[:, :, h_t], vr[:, :, h_t], m)
        out = out.unflatten(-1, (heads, -1)).clone()
        out[:, :, h_t] = sub.to(out.dtype)
        return out.flatten(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biases", default="-4,-2,-1,1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or resolve_under_home(
        f"output/tests/copy_bias/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    pages = sorted(resolve_under_home(f"{EVAL}/colorize/inputs/wild2").iterdir())[:2]
    biases = [float(b) for b in args.biases.split(",")]
    res = {}
    for b in [0.0] + biases:
        for n, page in enumerate(pages):
            d = out / f"bias{b:+g}" / f"in{n}"
            req = GenerateRequest.from_argv([
                "--config", CONFIG, "--lora", LORA, "--ref_image", str(page),
                "--seed", "0", "--out_dir", str(d),
            ])  # fmt: skip
            t0 = time.time()
            with Bias(out / "_te_cache", b):
                gen.run_generate(req)
            rec = json.loads((d / "manifest.json").read_text())["images"][0]
            im = Image.open(rec["image"]).convert("RGB")
            pg = Image.open(page).convert("RGB")
            r = dict(image=rec["image"], luma_r=round(luma_r(pg, im), 4),
                     chroma=round(chroma(im), 2))  # fmt: skip
            res.setdefault(f"{b:+g}", []).append(r)
            log(
                f"bias {b:+g} in{n}: luma r {r['luma_r']:.4f} chroma {r['chroma']:.2f} ({time.time() - t0:.0f}s)"
            )
            (out / "summary.json").write_text(json.dumps(res, indent=1))
    keys = sorted(res, key=float)
    tw = 420
    rows = []
    for n, page in enumerate(pages):
        row = [Image.open(page).convert("RGB")] + [
            Image.open(res[k][n]["image"]) for k in keys
        ]
        rows.append(row)
    th = max(int(r[0].height * tw / r[0].width) for r in rows)
    sheet = Image.new("RGB", (tw * len(rows[0]), th * len(rows)), "white")
    for y, row in enumerate(rows):
        for x, im in enumerate(row):
            im = im.convert("RGB").copy()
            im.thumbnail((tw, th))
            sheet.paste(im, (x * tw, y * th))
    sheet.save(out / "sheet.png")
    log(f"sheet columns: page, {', '.join('bias ' + k for k in keys)}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
