"""Anima: is the mean LoRA gradient biased against fp32, or only noisy? Over D paired
draws of ε at one σ, e_d = S_arm - S_fp32 per target (S as ``anima_stream_attrib.py``),
and per block group

    bias² = ‖ē‖² - s² / D,   s² = Σ ‖e_d - ē‖² / (D - 1)

which is unbiased for ‖E[e]‖² whatever the spread of e_d. ``fp32_jit`` is its control.

    make daemon-run ARGS="--stall-timeout 600 project/int8_train/src/anima_grad_bias.py"

reports/anima_stream_attrib.md.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import anima_stream_attrib as asa  # noqa: E402
from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import start_heartbeat  # noqa: E402
from bench.grad_init.probe_subspace import enumerate_targets  # noqa: E402
from library.anima.int8 import convert_dit_, forward_file  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.int8.linear import HeldCopy, quantize  # noqa: E402

ARMS = "bf16,bf16+s,bf16+wide,int8,int8+wide"


class Paired:
    """Running ``Σ e_d`` and ``Σ ‖e_d‖²`` per block group."""

    def __init__(self, group_of):
        self.group_of = group_of
        self.sum = {}
        self.sq = defaultdict(float)
        self.d = 0

    def add(self, S, ref):
        self.d += 1
        for ln, g in self.group_of.items():
            if ln not in ref:
                continue
            e = S[ln] - ref[ln]
            self.sum[ln] = self.sum[ln] + e if ln in self.sum else e
            self.sq[g] += float(e.pow(2).sum())

    def read(self, mean_norm2):
        out = {}
        d = self.d
        for g, n2 in mean_norm2.items():
            m2 = sum(
                float(v.pow(2).sum()) / d**2
                for ln, v in self.sum.items()
                if self.group_of[ln] == g
            )
            s2 = (self.sq[g] - d * m2) / (d - 1)
            bias2 = m2 - s2 / d
            out[g] = {
                "single": (self.sq[g] / d / n2) ** 0.5,
                "mean": (m2 / n2) ** 0.5,
                "bias": (max(bias2, 0.0) / n2) ** 0.5,
                "bias2_raw": bias2 / n2,
                "se2": s2 / d / n2,
            }
        return out


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap, vae=False, text_encoder=False)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--n_images", type=int, default=2)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--sketch_q", type=int, default=64)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--arms", default=ARMS)
    ap.add_argument("--out", default="project/int8_train/assets/anima_grad_bias.json")
    args = ap.parse_args()
    args.gradient_checkpointing = True
    arms = args.arms.split(",")
    start_heartbeat(label="grad_bias")
    for _ in range(90):
        if torch.cuda.mem_get_info()[0] > 14e9:
            break
        time.sleep(1)

    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    targets = enumerate_targets(anima)
    sk = asa.Sketcher(targets, args.sketch_q, device, args.seed)
    np_ = asa.NormProbe(anima)
    images = asa.pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    group_of = {
        ln: int(o.split(".")[1]) // 7 for ln, o, _ in targets if o.startswith("blocks.")
    }
    sigma = torch.tensor([args.sigma], device=device)
    t0 = time.perf_counter()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtypes = {
        n: t.dtype for n, t in [*anima.named_parameters(), *anima.named_buffers()]
    }
    attn_mode = anima.attn_mode
    anima.float()
    anima.attn_mode = "torch"
    refs = {}
    paired = {
        j: {a: Paired(group_of) for a in ["fp32_jit", *arms]}
        for j in range(len(images))
    }
    for j, ci in enumerate(images):
        for d in range(args.draws):
            batch = asa.load_batch(ci, j, args.seed + 1 + d, device)
            ref, _ = asa.one_pass(anima, sk, np_, batch, sigma, fp32=True)
            refs[j, d] = ref["S"]
            asa.JIT[0] = torch.Generator(device=device).manual_seed(d + 1)
            rec, _ = asa.one_pass(anima, sk, np_, batch, sigma, fp32=True)
            asa.JIT[0] = None
            paired[j]["fp32_jit"].add(rec["S"], ref["S"])
            print(
                f"fp32 {j + 1}/{len(images)} draw {d + 1}/{args.draws} "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    for n, t in [*anima.named_parameters(), *anima.named_buffers()]:
        t.data = t.data.to(dtypes[n])
    anima.attn_mode = attn_mode
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    names = convert_dit_(anima, "fwd", forward_file(None, args.dit) or None)
    mods = [anima.get_submodule(n) for n in names]
    for m in mods:
        wq_t, s_t = quantize(m.weight.detach().t().contiguous())
        m.held_wq_t = HeldCopy(wq_t)
        m.register_buffer("s_t", s_t, persistent=False)
        m.int8_mode = "full"
    mod_linears = [
        *(m for m in anima.t_embedder.modules() if isinstance(m, nn.Linear)),
        *(
            m
            for m in anima.final_layer.adaln_modulation.modules()
            if isinstance(m, nn.Linear)
        ),
    ]
    for block in anima.blocks:
        mod_linears.append(block.adaln_fused_down[1])
        for mod in (
            block.adaln_up_self_attn,
            block.adaln_up_cross_attn,
            block.adaln_up_mlp,
        ):
            mod_linears.append(mod)
            mod.register_forward_hook(asa.to_fp32)
    groups = {
        "mod": mod_linears,
        "head": [anima.final_layer.linear],
        "embed": [anima.x_embedder.proj[1]],
    }

    out = {"sigma": args.sigma, "draws": args.draws, "images": {}}
    for j, ci in enumerate(images):
        for d in range(args.draws):
            batch = asa.load_batch(ci, j, args.seed + 1 + d, device)
            for arm in arms:
                asa.set_arm(anima, mods, groups, attn_mode, arm)
                rec, _ = asa.one_pass(anima, sk, np_, batch, sigma)
                paired[j][arm].add(rec["S"], refs[j, d])
            asa.FLAGS.clear()
            print(
                f"{j + 1}/{len(images)} draw {d + 1}/{args.draws} "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
        mean = defaultdict(float)
        for ln in group_of:
            if (j, 0) in refs and ln in refs[j, 0]:
                m = sum(refs[j, d][ln] for d in range(args.draws)) / args.draws
                mean[group_of[ln]] += float(m.pow(2).sum())
        row = {a: p.read(mean) for a, p in paired[j].items()}
        out["images"][Path(ci.npz_path).stem] = row
        print(f"\nRESULT image {j}: % of ‖mean S_fp32‖, groups 0..3")
        for a, r in row.items():
            for key in ("single", "mean", "bias"):
                print(
                    f"RESULT {a} {key}: "
                    + " | ".join(f"{r[g][key] * 100:.1f}" for g in sorted(r))
                )
        Path(resolve_under_home(args.out)).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
