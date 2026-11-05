"""Anima fp16 headroom: fp32 ranges per module, and per-block error of fp16 arms.

Arms: ref (true fp32), bf16s (fp32 stream + bf16 GEMM, the default), f16s (fp32 stream +
fp16 GEMM), f16acc (f16s with ``allow_fp16_accumulation``), f16all (everything fp16),
f16all_s8 / _s16 (the stream scaled by 1/8, 1/16), f16accall_s16. Backward: the gradient
each block linear's output gets on bf16s, mean loss, no loss scale.
Finding: ``docs/anima/findings/fp16_headroom.md``.

    make daemon-run ARGS="--stall-timeout 600 bench/anima_fp16/probe.py --out <file.json>"
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench.anima_stream.run_bench import (  # noqa: E402
    fp32_reference,
    load_cell,
    pick_images,
    set_arm,
    unstream,
)
from library.anima.models import Block  # noqa: E402
from library.anima.stream import Fp32Linear, Fp32RMSNorm, fp32_linears  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

F16_MAX = 65504.0
ARMS = ("bf16s", "f16s", "f16acc", "f16all", "f16all_s8", "f16all_s16", "f16accall_s16")
F16_TINY = 6.1035e-5  # smallest normal
F16_ZERO = 2.98e-8  # rounds to 0 below this


def key_of(name: str) -> str:
    m = re.match(r"blocks\.(\d+)\.(.*)", name)
    return (m.group(2), int(m.group(1))) if m else (name, -1)


class Recorder:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.peak = defaultdict(float)  # (kind, name) -> absmax
        self.blocks = []
        self.nonfinite = []
        self.on = False

    def attach(self, ranges: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, Block):
                self.hooks.append(m.register_forward_hook(self._block(name)))
            if ranges and isinstance(m, (nn.Linear, nn.LayerNorm, nn.GELU)) or (
                ranges and type(m).__name__ in ("RMSNorm", "Fp32RMSNorm")
            ):
                self.hooks.append(m.register_forward_hook(self._range(name)))
            if not ranges and isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_hook(self._nf(name)))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _block(self, name):
        def hook(_m, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            self.blocks.append(o.detach().float())
            if not torch.isfinite(o).all():
                self.nonfinite.append(name)

        return hook

    def _range(self, name):
        def hook(_m, args, out):
            x = args[0] if args else None
            if x is not None and torch.is_tensor(x):
                k = ("in", name)
                self.peak[k] = max(self.peak[k], float(x.detach().abs().amax()))
            k = ("out", name)
            self.peak[k] = max(self.peak[k], float(out.detach().abs().amax()))

        return hook

    def _nf(self, name):
        def hook(_m, _a, out):
            if not torch.isfinite(out).all():
                self.nonfinite.append(name)

        return hook


def forward(anima, cell, sigma, dtype, autocast_dtype=None):
    latents, crossattn, noise = cell
    x = ((1.0 - sigma) * latents + sigma * noise).unsqueeze(2).to(dtype)
    t = torch.tensor([sigma], device=x.device, dtype=dtype)
    pad = torch.zeros(1, 1, *latents.shape[-2:], dtype=dtype, device=x.device)
    ctx = crossattn.to(dtype if dtype != torch.float32 else torch.bfloat16)
    if dtype == torch.float32:
        ctx = crossattn.float()
    with torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None):
        return anima(x, t, ctx, padding_mask=pad).squeeze(2).float()


def stream_fp32_f16gemm(dit):
    """stream_fp32_ with fp16 autocast instead of bf16."""
    for m in fp32_linears(dit):
        m.__class__ = Fp32Linear
    for block in dit.blocks:
        for attn in (block.self_attn, block.cross_attn):
            attn.q_norm.__class__ = Fp32RMSNorm
            attn.k_norm.__class__ = Fp32RMSNorm
    fwd = type(dit).forward_mini_train_dit.__get__(dit)

    def under(x, *a, **k):
        with torch.autocast(x.device.type, dtype=torch.float16):
            return fwd(x, *a, **k)

    dit.forward_mini_train_dit = under
    dit.stream = "fp32"


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@torch.no_grad()
def run_forward(args, anima, device, out):
    images = pick_images(Path(resolve_under_home(args.cache_root)), args.n_images,
                         args.min_tokens, args.max_tokens, args.seed)
    print("images:", [Path(c.npz_path).stem for c in images], flush=True)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    named = lambda: [*anima.named_parameters(), *anima.named_buffers()]  # noqa: E731
    saved = {k: v.detach().cpu().clone() for k, v in named()}

    def restore():
        unstream(anima)
        for k, t in named():
            t.data = saved[k].to(t.device)

    # weights in fp16
    wstat = {}
    for name, p in anima.named_parameters():
        if p.dim() == 2:
            a = p.detach().float().abs()
            nz = a[a > 0]
            wstat[name] = dict(max=float(a.max()), sub=float((nz < F16_TINY).float().mean()),
                               zero=float((nz < F16_ZERO).float().mean()))
    out["weights"] = {
        "max": max(v["max"] for v in wstat.values()),
        "worst_sub": sorted(((v["sub"], k) for k, v in wstat.items()), reverse=True)[:8],
        "worst_zero": sorted(((v["zero"], k) for k, v in wstat.items()), reverse=True)[:8],
    }
    print("weights", json.dumps(out["weights"]), flush=True)

    rec = Recorder(anima)
    peaks = defaultdict(float)
    err = defaultdict(lambda: defaultdict(list))  # arm -> block -> [rel]
    verr = defaultdict(lambda: defaultdict(list))  # arm -> sigma -> [rel]
    nonfinite = defaultdict(list)
    stream_peak = defaultdict(lambda: [0.0] * 64)

    for j, ci in enumerate(images):
        cell = load_cell(ci, j, args.seed, device)
        for s in sigmas:
            # ref
            rec.blocks, rec.peak = [], defaultdict(float)
            with fp32_reference(anima):
                rec.attach(ranges=True)
                v_ref = forward(anima, cell, s, torch.float32)
                rec.detach()
            ref_blocks = rec.blocks
            for k, v in rec.peak.items():
                peaks[k] = max(peaks[k], v)
            for b, t in enumerate(ref_blocks):
                stream_peak[s][b] = max(stream_peak[s][b], float(t.abs().amax()))
            for arm in ARMS:
                if arm == "bf16s":
                    set_arm(anima, "fp32")
                    dtype, ac = torch.float32, None  # stream_fp32_ enters bf16 autocast itself
                elif arm in ("f16s", "f16acc"):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = arm == "f16acc"
                    anima.half()
                    stream_fp32_f16gemm(anima)
                    dtype, ac = torch.float32, None
                else:
                    torch.backends.cuda.matmul.allow_fp16_accumulation = arm.startswith("f16acc")
                    anima.half()
                    dtype, ac = torch.float16, None
                c = int(arm.split("_s")[1]) if "_s" in arm else 1
                shooks = []
                if c > 1:
                    shooks.append(anima.blocks[0].register_forward_pre_hook(
                        lambda _m, a, c=c: (a[0] / c, *a[1:])))
                    for blk in anima.blocks:
                        for sub in (blk.self_attn, blk.cross_attn, blk.mlp):
                            shooks.append(sub.register_forward_hook(lambda _m, _a, o, c=c: o / c))
                rec.blocks, rec.nonfinite = [], []
                rec.attach(ranges=False)
                v = forward(anima, cell, s, dtype, ac)
                rec.detach()
                for h in shooks:
                    h.remove()
                rec.blocks = [t * c for t in rec.blocks]
                if not torch.isfinite(v).all():
                    rec.nonfinite.append("OUTPUT")
                for b, t in enumerate(rec.blocks):
                    err[arm][b].append(rel(t, ref_blocks[b]))
                verr[arm][s].append(rel(v, v_ref))
                if rec.nonfinite:
                    nonfinite[arm].append(dict(img=j, sigma=s, first=rec.nonfinite[:5], n=len(rec.nonfinite)))
                torch.backends.cuda.matmul.allow_fp16_accumulation = False
                restore()
            print(f"img {j} σ {s}: " + ", ".join(
                f"{a} {verr[a][s][-1]*100:.3f}%" for a in ARMS), flush=True)
            del ref_blocks
            torch.cuda.empty_cache()

    # aggregate peaks by module suffix across blocks
    by_suffix = defaultdict(lambda: [0.0, -1])
    for (kind, name), v in peaks.items():
        suf, b = key_of(name)
        k = f"{kind}:{suf}"
        if v > by_suffix[k][0]:
            by_suffix[k] = [v, b]
    out["peaks_by_suffix"] = dict(sorted(by_suffix.items(), key=lambda kv: -kv[1][0]))
    out["stream_peak_by_block"] = {str(s): v[: len(anima.blocks)] for s, v in stream_peak.items()}
    out["block_err_mean_pct"] = {a: [sum(v) / len(v) * 100 for _, v in sorted(d.items())] for a, d in err.items()}
    out["v_err_mean_pct"] = {a: {str(s): sum(v) / len(v) * 100 for s, v in d.items()} for a, d in verr.items()}
    out["nonfinite"] = nonfinite


def run_backward(args, anima, device, out):
    if args.bw_images <= 0:
        return
    images = pick_images(Path(resolve_under_home(args.cache_root)), args.bw_images,
                         args.bw_min_tokens, args.bw_max_tokens, args.seed + 1)
    set_arm(anima, "fp32")
    anima.train()
    stats = defaultdict(lambda: [0.0, 1.0, 0.0, 0.0])  # absmax, min nonzero absmax?, sub frac, zero frac
    hooks = []

    def mk(name):
        def hook(_m, _gi, go):
            g = go[0]
            if g is None:
                return
            a = g.detach().float().abs()
            st = stats[name]
            st[0] = max(st[0], float(a.max()))
            st[2] = max(st[2], float((a < F16_TINY).float().mean()))
            st[3] = max(st[3], float((a < F16_ZERO).float().mean()))
            st[1] = min(st[1], float(a.median()))
        return hook

    def fwd_hook(name):
        def h(_m, _a, o):
            if o.requires_grad:
                o.register_hook(lambda g: mk(name)(None, None, (g,)))
        return h

    for name, m in anima.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fwd_hook(name)))
    anima.enable_gradient_checkpointing()
    for p in anima.parameters():
        p.requires_grad_(False)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    for j, ci in enumerate(images):
        latents, crossattn, noise = load_cell(ci, j, args.seed, device)
        for s in sigmas:
            x = ((1.0 - s) * latents + s * noise).unsqueeze(2).float().requires_grad_(True)
            t = torch.tensor([s], device=device)
            pad = torch.zeros(1, 1, *latents.shape[-2:], device=device)
            v = anima(x, t, crossattn, padding_mask=pad).squeeze(2).float()
            loss = ((v - (noise - latents)) ** 2).mean()
            loss.backward()
            g = x.grad.abs()
            stats["INPUT x"] = [float(g.max()), float(g.median()), float((g < F16_TINY).float().mean()), float((g < F16_ZERO).float().mean())]
            print(f"bw img {j} σ {s} loss {loss.item():.4f}", flush=True)
    for h in hooks:
        h.remove()
    agg = defaultdict(lambda: [0.0, 1.0, 0.0, 0.0, -1])
    for name, st in stats.items():
        suf, b = key_of(name)
        a = agg[suf]
        if st[0] > a[0]:
            a[0], a[4] = st[0], b
        a[1] = min(a[1], st[1])
        a[2] = max(a[2], st[2])
        a[3] = max(a[3], st[3])
    out["grad_out_by_suffix"] = {k: dict(absmax=v[0], at_block=v[4], min_median=v[1], max_sub_frac=v[2],
                                         max_zero_frac=v[3]) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap, vae=False, text_encoder=False)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=4)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--bw_images", type=int, default=2)
    ap.add_argument("--bw_min_tokens", type=int, default=3500)
    ap.add_argument("--bw_max_tokens", type=int, default=4608)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_forward", action="store_true")
    args = ap.parse_args()
    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    print("attn_mode", anima.attn_mode, "blocks", len(anima.blocks), flush=True)
    out = {}
    if not args.skip_forward:
        run_forward(args, anima, device, out)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    torch.cuda.empty_cache()
    with torch.enable_grad():
        run_backward(args, anima, device, out)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
