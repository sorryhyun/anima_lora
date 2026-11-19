#!/usr/bin/env python3
"""Can the P2 head fit 16 images? (pixel readout, head-only training on cached features)

``convnext_1119_0835``'s train loss did not move in 14,400 steps, and
``ceiling_1119_0920`` put the VAE decode of the head's own x_t input 4.2 dB above the
head. This probe takes the data size and the DiT out of the loop: features, x_t and VAE
targets of 16 train images are computed once (fixed noise), then each arm trains a
fresh head on those 16 pairs with the parent's optimizer. Arms differ in how the
residual path starts:

    parent   P1 skip (LR 2e-5), out zero-init, layer scale γ 1e-6   (= 0835)
    open     P1 skip (LR 2e-5), out zero-init, γ 1
    noskip   no skip, out default init, γ 1                          (FastFrames' form)

Read: PSNR vs VAE(x0) on the 16 fit images and on P2's 30 test images, per arm, beside
P1 and the two VAE references (VAE(x_t), VAE(x̂0)) on the same 16 images.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/overfit_probe.py --label overfit_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from safetensors.torch import load_file

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.convnext_head import (
    P1_RUN,
    PixelHead,
    lat_patches,
    pick_text,
    rec_loss,
    token_count,
)
from bench.pixel_readout.linear_probe import from_patches, pick_samples, psnr
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from library.models import qwen_vae

ARMS = {  # name: (use_skip, zero_out, ls_init)
    "parent": (True, True, 1e-6),
    "open": (True, True, 1.0),
    "noskip": (False, False, 1.0),
}


class Head(PixelHead):
    use_skip = True

    def forward(self, f, lat, parts: bool = False):
        gh, gw = f.shape[:2]
        h = (self.proj(f) + self.lat_embed(lat)).permute(2, 0, 1)[None]
        for b in self.blocks:
            h = b(h)
        h = self.norm_out(h[0].permute(1, 2, 0))
        with torch.autocast("cuda", enabled=False):
            res = self.out(h).float()
            skip = self.linear(f.float()) if self.use_skip else torch.zeros_like(res)
        if parts:
            return skip, res
        return from_patches((skip + res).reshape(gh * gw, -1), gh, gw)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--p1_run", default=P1_RUN)
    p.add_argument("--arms", default="parent,open,noskip")
    p.add_argument("--n_fit", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_linear", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    args = p.parse_args()
    start_heartbeat(label="overfit")
    device = torch.device(args.device)

    # --- data: convnext_head.py's split; 16 fit images drawn from its train list ---
    cache_root = Path(resolve_under_home(args.cache_root))
    p1_res = json.loads(
        (Path(resolve_under_home(args.p1_run)) / "result.json").read_text()
    )
    a = p1_res["args"]
    p1_train, test, test_artists = pick_samples(
        cache_root,
        a["n_train"],
        a["n_test_artists"],
        a["per_artist"],
        a["max_tokens"],
        a["seed"],
    )
    assert test_artists == p1_res["metrics"]["test_artists"]
    held = set(test_artists)
    held_paths = {
        ci.npz_path for ci in pick_text(cache_root, {c.npz_path for c in p1_train})
    }
    train = [
        ci
        for art in sorted(d.name for d in cache_root.iterdir() if d.is_dir())
        if art not in held
        for ci in discover_cached_pairs(str(cache_root / art))
        if ci.npz_path not in held_paths and token_count(ci) <= a["max_tokens"]
    ]
    fit = random.Random(args.seed + 7).sample(train, args.n_fit)
    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])

    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    args.compile = False  # 46 single forwards
    anima = build_anima(args, adapter=None, train_mode=False).anima
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )
    D = anima.final_layer.hidden_size

    @torch.no_grad()
    def rollout(ci, idx: int, refs: bool):
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        tgt = vae.decode_to_pixels(lat)[0].float()
        cross, _ = load_cached_text_features(ci.te_path, variant=0)
        cross = cross.unsqueeze(0).to(device, torch.bfloat16)
        g = torch.Generator(device=device).manual_seed(args.seed * 100003 + idx)
        noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
        x_t = (1 - sigma_last) * lat + sigma_last * noise
        sig = torch.tensor([sigma_last], device=device)
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = anima(x_t.unsqueeze(2).to(torch.bfloat16), sig, cross, padding_mask=pad)
        f = tap.pop("x")[0, 0].float()
        ref = {}
        if refs:
            x0_hat = (x_t.float() - sigma_last * v.float().squeeze(2)).to(lat.dtype)
            ref = {
                "vae_xt": psnr(vae.decode_to_pixels(x_t)[0].clamp(-1, 1), tgt),
                "vae_x0hat": psnr(vae.decode_to_pixels(x0_hat)[0].clamp(-1, 1), tgt),
            }
        return f, lat_patches(x_t).float(), tgt, ref

    fit_data, refs = [], []
    for i, ci in enumerate(fit):
        f, lat, tgt, ref = rollout(ci, 30_000 + i, refs=True)
        fit_data.append((f, lat, tgt))
        refs.append(ref)
    test_data = [
        tuple(t.to("cpu", torch.float16) for t in rollout(ci, 10_000 + j, False)[:3])
        for j, ci in enumerate(test)
    ]
    tap.clear()
    anima = vae = None  # free the card for the head arms
    torch.cuda.empty_cache()

    w1 = load_file(
        str(Path(resolve_under_home(args.p1_run)) / "readout_linear.safetensors")
    )["weight"]

    @torch.no_grad()
    def score(head, data):
        head.eval()
        out = []
        for f, lat, tgt in data:
            f, lat, tgt = (t.to(device).float() for t in (f, lat, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(f, lat)
            out.append(psnr(pred.float().clamp(-1, 1), tgt))
        head.train()
        return float(np.mean(out))

    results = {
        "fit_refs": {k: float(np.mean([r[k] for r in refs])) for k in refs[0]},
    }
    p1_head = Head(D, 64, args.width, args.depth).to(device)  # out zero-init → P1
    with torch.no_grad():
        p1_head.linear.weight.copy_(w1[:, :D])
        p1_head.linear.bias.copy_(w1[:, D])
    results["p1_linear"] = {
        "fit": score(p1_head, fit_data),
        "test": score(p1_head, test_data),
    }
    del p1_head
    print(f"refs on fit: {results['fit_refs']}  P1: {results['p1_linear']}", flush=True)
    for name in args.arms.split(","):
        use_skip, zero_out, ls_init = ARMS[name]
        torch.manual_seed(args.seed)
        head = Head(D, 64, args.width, args.depth, ls_init).to(device)
        head.use_skip = use_skip
        with torch.no_grad():
            head.linear.weight.copy_(w1[:, :D])
            head.linear.bias.copy_(w1[:, D])
        if not zero_out:
            head.out.reset_parameters()
        rest = [p for n, p in head.named_parameters() if not n.startswith("linear.")]
        groups = [{"params": rest, "lr": args.lr, "weight_decay": 0.01}]
        if use_skip:
            groups.append(
                {
                    "params": list(head.linear.parameters()),
                    "lr": args.lr_linear,
                    "weight_decay": 0.0,
                }
            )
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
        base_lrs = [g["lr"] for g in opt.param_groups]
        rng = random.Random(args.seed + 1)
        order: list[int] = []
        hist = [
            {"step": 0, "fit": score(head, fit_data), "test": score(head, test_data)}
        ]
        clipped = 0
        t0 = time.perf_counter()
        for step in range(args.steps):
            if not order:
                order = list(range(len(fit_data)))
                rng.shuffle(order)
            f, lat, tgt = fit_data[order.pop()]
            s = (
                min(1.0, (step + 1) / args.warmup)
                * 0.5
                * (1 + math.cos(math.pi * step / args.steps))
            )
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * s
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(f, lat)
            loss, _ = rec_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            clipped += int(gn > 1.0)
            opt.step()
            if (step + 1) % args.eval_every == 0:
                hist.append(
                    {
                        "step": step + 1,
                        "fit": score(head, fit_data),
                        "test": score(head, test_data),
                    }
                )
                print(
                    f"  {name} @ {step + 1}: fit {hist[-1]['fit']:.2f}  "
                    f"test {hist[-1]['test']:.2f}  "
                    f"{(time.perf_counter() - t0) / (step + 1) * 1e3:.0f} ms/step",
                    flush=True,
                )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            skip, res = head(fit_data[0][0], fit_data[0][1], parts=True)
        results[name] = {
            "use_skip": use_skip,
            "zero_out": zero_out,
            "ls_init": ls_init,
            "history": hist,
            "clipped_steps": clipped,
            "gamma_abs_mean": [float(b.gamma.abs().mean()) for b in head.blocks],
            "out_weight_abs_mean": float(head.out.weight.abs().mean()),
            "rms_residual": float(res.pow(2).mean().sqrt()),
            "rms_skip": float(skip.pow(2).mean().sqrt()),
            "ms_per_step": (time.perf_counter() - t0) / args.steps * 1e3,
        }
        del head, opt
        torch.cuda.empty_cache()

    run_dir = make_run_dir("pixel_readout", label=args.label)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "sigma_last": sigma_last,
            "fit_images": [f"{Path(c.npz_path).parent.name}/{c.stem}" for c in fit],
            **results,
        },
        label=args.label,
        artifacts=[],
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
