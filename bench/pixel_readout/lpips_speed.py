#!/usr/bin/env python3
"""VGG-LPIPS step time (forward on pred + target, backward through pred) for the ×2 head's
training loss: eager vs channels_last vs one batch-2 forward vs torch.compile vs a random
sub-crop. Inputs are random images at the ×2 window side (24 tokens × 32 px = 768).

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/lpips_speed.py"
"""

from __future__ import annotations

import argparse
import time

import lpips
import torch


def lpips_cat(lp: lpips.LPIPS, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """LPIPS with pred and target through VGG as one batch of 2 (lpips.LPIPS.forward's
    math: scaling layer, unit-normalised features, 1×1 lin heads, spatial mean)."""
    x = lp.scaling_layer(torch.cat([a, b]))
    feats = lp.net(x)
    val = 0
    for k, f in enumerate(feats):
        f = lpips.normalize_tensor(f)
        d = (f[:1] - f[1:]) ** 2
        val = val + lp.lins[k](d).mean([2, 3], keepdim=True)
    return val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", type=int, default=768)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    for q in lp.parameters():
        q.requires_grad_(False)

    s = args.side
    tgt = torch.rand(1, 3, s, s, device=dev) * 2 - 1
    src = torch.rand(1, 3, s, s, device=dev) * 2 - 1

    def crop(t, c, y, x):
        return t[..., y : y + c, x : x + c]

    def make(fn, cl=False, c=None):
        def step():
            a = src.clone().requires_grad_(True)
            aa, bb = a, tgt
            if c:
                y, x = torch.randint(0, s - c + 1, (2,)).tolist()
                aa, bb = crop(aa, c, y, x), crop(bb, c, y, x)
            if cl:
                aa = aa.contiguous(memory_format=torch.channels_last)
                bb = bb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fn(aa, bb).mean().float()
            loss.backward()
            return loss.detach(), a.grad

        return step

    def ref_fn(a, b):
        return lp(a, b)

    def cat_fn(a, b):
        return lpips_cat(lp, a, b)

    # the batch-2 form must equal the library's forward
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        d = (ref_fn(src, tgt) - cat_fn(src, tgt)).abs().max().item()
    print(f"batch-2 form vs lpips.forward: max |Δ| {d:.2e}", flush=True)

    results = {}

    def bench(name, step):
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        gb = torch.cuda.max_memory_allocated() / 1e9
        results[name] = ms
        print(f"{name:42s} {ms:7.1f} ms / step  peak {gb:.2f} GB", flush=True)

    torch.backends.cudnn.benchmark = False
    bench("eager (current)", make(ref_fn))
    torch.backends.cudnn.benchmark = True
    bench("eager + cudnn.benchmark", make(ref_fn))
    bench("channels_last", make(ref_fn, cl=True))
    lp.to(memory_format=torch.channels_last)
    bench("channels_last (weights too)", make(ref_fn, cl=True))
    bench("channels_last + batch-2", make(cat_fn, cl=True))
    comp = torch.compile(cat_fn)
    bench("channels_last + batch-2 + compile", make(comp, cl=True))
    comp_ref = torch.compile(ref_fn)
    bench("channels_last + compile (two passes)", make(comp_ref, cl=True))
    for c in (512, 384):
        bench(f"channels_last + batch-2, crop {c}", make(cat_fn, cl=True, c=c))
        bench(f"channels_last + batch-2 + compile, crop {c}", make(comp, cl=True, c=c))
        bench(
            f"channels_last + compile (two passes), crop {c}",
            make(comp_ref, cl=True, c=c),
        )


if __name__ == "__main__":
    main()
