#!/usr/bin/env python3
"""Weight-space mixes of two ×2 heads that share a parent (CPU, no training).

``head(α) = a + α · (b − a)`` on every tensor, or on one stage's tensors only. ``a`` is
the parent and ``b`` a finetune of it (a GAN finish), so ``b − a`` is small and the mix
is read as a dial between the two: α 0 = ``a``, 1 = ``b``, above 1 past ``b``.

Stages of a ``PixelHeadX2`` file, by key prefix:

    trunk   base.proj, base.lat_embed, base.blocks         (16-px spacing)
    up1     base.up_norm, base.up_proj, base.up_blocks     (8 px)
    up2     up2_norm, up2_proj, up2_blocks, noise_in       (4 px)
    out     norm2, out2                                    (the ×2 output layer)

``--alphas`` writes ``<out_dir>/a<α·100>/head_x2.safetensors`` (every stage at α);
``--stages`` writes ``<out_dir>/g_<stage>/head_x2.safetensors`` (that stage at
``--stage_alpha``, the rest at 0) and ``--stages_out`` ``<out_dir>/x_<stage>/…`` (that
stage at 0, the rest at ``--stage_alpha``). Each file carries ``a``'s stamp plus
``merge_a`` / ``merge_b`` / ``merge_alpha`` (per stage). A folder name is the column
name ``p3_generated.py --x2`` gives the file.

``--ranks k …`` splits ``b − a`` by singular directions: per linear weight with more
than ``k`` of them, the ``k`` largest go to ``r<k>/`` and the rest to ``c<k>/``; every
other tensor (biases, norms, layer scales, depthwise kernels) carries its whole change
in both files, so the two differ in the matrices' split alone.

Usage::

    .venv/bin/python -m bench.pixel_readout.experiments.head_merge \
        --a output/pixel_head/<parent>/head_x2.safetensors \
        --b output/pixel_head/<finish>/head_x2.safetensors \
        --out_dir output/pixel_head/x2mg_mmdd_hhmm \
        --alphas 0.25 0.5 0.75 1.25 --stages trunk up1 up2 out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home

STAGES = {
    "trunk": ("base.proj.", "base.lat_embed.", "base.blocks."),
    "up1": ("base.up_norm.", "base.up_proj.", "base.up_blocks."),
    "up2": ("up2_norm.", "up2_proj.", "up2_blocks.", "noise_in."),
    "out": ("norm2.", "out2."),
}


def stage_of(key: str) -> str:
    for name, prefixes in STAGES.items():
        if key.startswith(prefixes):
            return name
    raise KeyError(f"head_merge: {key} belongs to no stage")


def mix(a: dict, b: dict, alpha: dict[str, float]) -> dict:
    """a + α[stage] · (b − a) per tensor, in fp32, stored in ``a``'s dtype."""
    out = {}
    for k, w in a.items():
        d = b[k].float() - w.float()
        out[k] = (w.float() + alpha[stage_of(k)] * d).to(w.dtype)
    return out


def split_rank(a: dict, b: dict, k: int) -> tuple[dict, dict, dict]:
    """→ (a + top-``k`` of b − a, a + the rest of it, share of |b − a|² in the top
    ``k`` per stage, over the matrices that were split)."""
    top, rest = {}, {}
    kept = {s: [0.0, 0.0] for s in STAGES}
    for key, w in a.items():
        d = b[key].float() - w.float()
        if d.ndim == 2 and min(d.shape) > k:
            u, sv, vh = torch.linalg.svd(d, full_matrices=False)
            t = (u[:, :k] * sv[:k]) @ vh[:k]
            e = kept[stage_of(key)]
            e[0] += sv[:k].pow(2).sum().item()
            e[1] += sv.pow(2).sum().item()
        else:
            t = d
        top[key] = (w.float() + t).to(w.dtype)
        rest[key] = (w.float() + (d if t is d else d - t)).to(w.dtype)
    return top, rest, {s: (e[0] / e[1] if e[1] else None) for s, e in kept.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="the parent head (α 0)")
    p.add_argument("--b", required=True, help="the finetune of it (α 1)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--alphas", type=float, nargs="*", default=[])
    p.add_argument("--stages", nargs="*", default=[], choices=list(STAGES))
    p.add_argument("--stages_out", nargs="*", default=[], choices=list(STAGES))
    p.add_argument("--stage_alpha", type=float, default=1.0)
    p.add_argument("--ranks", type=int, nargs="*", default=[])
    args = p.parse_args()

    pa, pb = (Path(resolve_under_home(x)) for x in (args.a, args.b))
    a, b = load_file(str(pa)), load_file(str(pb))
    with safe_open(str(pa), "pt") as fh:
        meta = dict(fh.metadata() or {})
    if a.keys() != b.keys() or any(a[k].shape != b[k].shape for k in a):
        raise SystemExit("head_merge: the two files differ in keys or shapes")

    # the size of b − a per stage, relative to a
    norms = {}
    for name in STAGES:
        ks = [k for k in a if stage_of(k) == name]
        da = sum((b[k].float() - a[k].float()).pow(2).sum().item() for k in ks)
        na = sum(a[k].float().pow(2).sum().item() for k in ks)
        norms[name] = {
            "params": sum(a[k].numel() for k in ks),
            "rel_delta": (da / na) ** 0.5 if na else 0.0,
        }
        print(
            f"{name:6s} {norms[name]['params']:>10d}  |b-a|/|a| {norms[name]['rel_delta']:.4f}"
        )

    arms: dict[str, dict[str, float]] = {}
    for al in args.alphas:
        arms[f"a{round(al * 100):03d}"] = {s: al for s in STAGES}
    for s in args.stages:
        arms[f"g_{s}"] = {t: args.stage_alpha if t == s else 0.0 for t in STAGES}
    for s in args.stages_out:
        arms[f"x_{s}"] = {t: 0.0 if t == s else args.stage_alpha for t in STAGES}

    out_dir = Path(resolve_under_home(args.out_dir))
    for name, alpha in arms.items():
        dst = out_dir / name / "head_x2.safetensors"
        dst.parent.mkdir(parents=True, exist_ok=True)
        stamp = dict(
            meta,
            run=f"{out_dir.name}/{name}",
            merge_a=args.a,
            merge_b=args.b,
            merge_alpha=json.dumps(alpha),
        )
        save_file(mix(a, b, alpha), str(dst), metadata=stamp)
        print(f"wrote {dst}")
    ranks = {}
    for k in args.ranks:
        top, rest, share = split_rank(a, b, k)
        ranks[k] = share
        print(f"rank {k}: share of |b-a|^2 in the top {k}, per stage: {share}")
        for name, sd in ((f"r{k:03d}", top), (f"c{k:03d}", rest)):
            dst = out_dir / name / "head_x2.safetensors"
            dst.parent.mkdir(parents=True, exist_ok=True)
            stamp = dict(
                meta,
                run=f"{out_dir.name}/{name}",
                merge_a=args.a,
                merge_b=args.b,
                merge_rank=f"{'top' if name[0] == 'r' else 'rest'} {k}",
            )
            save_file(sd, str(dst), metadata=stamp)
            print(f"wrote {dst}")
    (out_dir / "merge.json").write_text(
        json.dumps(
            {"a": args.a, "b": args.b, "stages": norms, "arms": arms, "ranks": ranks},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
