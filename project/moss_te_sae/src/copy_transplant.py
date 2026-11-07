"""Which part of an edit LoRA carries the reference copy: is b39 h19 self-contained,
or does it need the reference stream the LoRA's earlier blocks shaped?

LoRA: chinoll/Moss-Image-Edit (``copy_head_ext.py``), whose ΔW puts 0.56 / 0.81 of
block 39's v / o energy on head 19 and whose cut of that head moves a render 21.5 luma
L1 at a 1024 edge. Each arm applies the LoRA through per-adapter masks, separately for
the main rows (image + text queries) and for the isolated reference stream (RefCache,
reference at t = 0):

- ``base``: no LoRA;
- ``full``: the LoRA everywhere;
- ``h19``: only b39 h19 (q / k / v output rows, o input columns), both streams;
- ``h19_qk``: only b39 h19's q / k (the route; Base's value path);
- ``copyheads``: the LoRA's top copy heads by own-cell reference mass (step 0 and all
  steps, ``copy_head_ext`` run 20261107-1532) plus b39 h19, q / k / v / o, both streams;
- ``late``: every adapter in blocks 36-41, both streams;
- ``ref_h19``: the full LoRA on the reference stream, only b39 h19 on the main rows;
- ``ref_only``: the full LoRA on the reference stream, nothing on the main rows.

Eager (``--no-compile``): the stream flag is Python state the compiled blocks would
guard on. Reads as ``copy_head_ext``: b39 h19 copy mass by step bucket, luma L1
against the reference and against ``full``, and the copy map against the edit region.
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
from PIL import Image  # noqa: E402

import copy_head  # noqa: E402
import copy_head_ext as ext  # noqa: E402
import library.moss.lora as mlora  # noqa: E402
import library.moss.model as mm  # noqa: E402
from copy_head import log, luma  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.lines.adapters import LoRAAdapter  # noqa: E402

HD = 128
COPY_HEADS = [
    (37, 20), (35, 3), (31, 17), (34, 6), (33, 20), (34, 9), (36, 21), (38, 14),
    (41, 7), (39, 19),
]  # fmt: skip
PROJ = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def head_spec(heads, projs=("q", "k", "v", "o")):
    """{adapter name: list of heads} for per-head slices."""
    spec = {}
    for b, h in heads:
        for p in projs:
            spec.setdefault(f"layers_{b}_proj_{p}", []).append(h)
    return spec


def block_spec(blocks):
    return {f"layers_{b}_proj_{p}": None for b in blocks for p in PROJ}


ALL = block_spec(range(42))
ARMS = {
    "base": None,
    "full": (ALL, ALL),
    "h19": (head_spec([(39, 19)]),) * 2,
    "h19_qk": (head_spec([(39, 19)], ("q", "k")),) * 2,
    "copyheads": (head_spec(COPY_HEADS),) * 2,
    "late": (block_spec(range(36, 42)),) * 2,
    "ref_h19": (head_spec([(39, 19)]), ALL),
    "ref_only": ({}, ALL),
}


class Streams:
    """Masks every LoRAAdapter by stream. ``main`` / ``ref``: {adapter name: None
    (whole adapter) | [heads]}; an adapter absent from a stream's spec is off there."""

    def __init__(self, main, ref):
        self.spec = {"main": main, "ref": ref}
        self.in_ref = False
        self.names = {}
        self.masks = {}
        self._orig = (LoRAAdapter.forward, mm.RefCache.lookup, mlora.load_for_inference)

    def __enter__(self):
        streams = self
        fwd, lookup, load = self._orig

        def load_patch(transformer, path, *a, **kw):
            net = load(transformer, path, *a, **kw)
            for name, ad in net.adapters.items():
                streams.names[id(ad)] = name
            return net

        def lookup_patch(cache, *a, **kw):
            streams.in_ref = True
            try:
                return lookup(cache, *a, **kw)
            finally:
                streams.in_ref = False

        def forward(ad, x, base_out):
            if ad.multiplier == 0.0:
                return base_out
            name = streams.names.get(id(ad))
            spec = streams.spec["ref" if streams.in_ref else "main"]
            if name is None or name not in spec:
                return base_out
            heads = spec[name]
            if heads is None:
                return fwd(ad, x, base_out)
            proj = name.rsplit("_", 1)[-1]
            m = streams._mask(ad, name, heads, x.device)
            if proj == "o":  # heads index the input columns
                return fwd(ad, x * m.to(x.dtype), base_out)
            delta = fwd(ad, x, base_out) - base_out
            return base_out + delta * m.to(delta.dtype)

        LoRAAdapter.forward = forward
        mm.RefCache.lookup = lookup_patch
        mlora.load_for_inference = load_patch
        return self

    def _mask(self, ad, name, heads, device):
        key = (name, tuple(heads))
        if key not in self.masks:
            n = ad.down.in_features if name.endswith("_o") else ad.up.out_features
            m = torch.zeros(n, device=device)
            for h in heads:
                m[h * HD : (h + 1) * HD] = 1
            self.masks[key] = m
        return self.masks[key]

    def __exit__(self, *exc):
        LoRAAdapter.forward, mm.RefCache.lookup, mlora.load_for_inference = self._orig


def main():
    copy_head.HEAD_CHUNK = 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--inputs", default="0,2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--size", default="576x1024")
    args = ap.parse_args()
    w, h = args.size.split("x")
    ext.SAMPLER.extend(["--width", w, "--height", h, "--no-compile"])
    out = args.out or resolve_under_home(
        f"output/tests/copy_transplant/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    te_cache = out / "_te_cache"
    summary = {}
    for n in map(int, args.inputs.split(",")):
        ref, prompt = ext.INPUTS[n]
        ref = ext.find_image(ref)
        rows = {}
        for arm in args.arms.split(","):
            spec = ARMS[arm]
            t0 = time.time()
            with Streams(*(spec or ({}, {}))):
                rows[arm] = ext.render(
                    ref, prompt, out / f"in{n}" / arm,
                    mult=0 if spec is None else 1, cut=None, capture=True,
                    frame0=True, te_cache=te_cache,
                )  # fmt: skip
            log(f"in{n} {arm}: {time.time() - t0:.0f}s")
        full = rows.get("full")
        W, H = rows[next(iter(rows))]["width"], rows[next(iter(rows))]["height"]
        R = luma(Image.open(ref).resize((W, H)))
        L_full = luma(Image.open(full["image"])) if full else None
        for arm, e in rows.items():
            Li = luma(Image.open(e["image"]))
            e["luma_vs_ref"] = float(np.abs(Li - R).mean())
            if L_full is not None:
                e["luma_vs_full"] = float(np.abs(Li - L_full).mean())
            if full:
                e["gate"] = ext.gate_vs_edit(
                    out / f"in{n}" / arm, ref, full["image"], W, H
                )
        summary[f"in{n}"] = dict(ref=ref, prompt=prompt, arms=rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        tiles = [Image.open(ref).convert("RGB").resize((W, H))]
        tiles += [Image.open(rows[a]["image"]).convert("RGB") for a in rows]
        tw = 320
        th = int(H * tw / W)
        sheet = Image.new("RGB", (tw * len(tiles), th), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((tw, th)), (i * tw, 0))
        sheet.save(out / f"sheet_in{n}.png")
        log(f"in{n} sheet: ref, {', '.join(rows)}")
        for arm, e in rows.items():
            c = e.get("b39h19", {}).get("copy")
            g = e.get("gate", {}).get("h19")
            log(
                f"  {arm:10} L/ref {e['luma_vs_ref']:.1f}"
                + (f" L/full {e['luma_vs_full']:.1f}" if "luma_vs_full" in e else "")
                + (
                    f" h19 copy s0 {c[0]:.3f} mid {c[2]:.3f} late {c[3]:.3f}"
                    if c
                    else ""
                )
                + (
                    f" gate in/out {g['inside']:.2f}/{g['outside']:.2f}"
                    if g and g["inside"] is not None
                    else ""
                )
            )
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
