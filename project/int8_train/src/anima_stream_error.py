"""Anima int8 base: prediction and LoRA-gradient error against fp32, by stream dtype and
kernel input. S = (dY Ω)ᵀ X per target, as ``grad_error.py``. Arm ``*_s32`` casts ``x``
at each block and the ``adaln_up_*`` outputs to fp32; the rest stays under autocast.

    make daemon-run ARGS="--stall-timeout 600 project/int8_train/src/anima_stream_error.py"

reports/anima_fp32_stream.md.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import library.anima.models as am  # noqa: E402
from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import start_heartbeat  # noqa: E402
from bench.grad_init.probe_subspace import enumerate_targets, latent_tokens  # noqa: E402
from library.anima.int8 import convert_dit_, forward_file  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.int8.linear import (  # noqa: E402
    HeldCopy,
    Int8TrainLinear,
    linear_full,
    quantize,
)
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)

ARMS = ("bf16", "bf16_rep", "bf16_s32", "int8", "int8_in16", "int8_s32")
STREAM = [False]


class In16(Int8TrainLinear):
    """The kernel on a bf16 input, as before the fp32 cast."""

    def forward(self, x):
        return linear_full(x.to(torch.bfloat16), self.wq, self.s, self.wq_t, self.s_t)


orig_forward = am.Block.forward


def block_forward(self, x, *a, **k):
    return orig_forward(self, x.float() if STREAM[0] else x, *a, **k)


am.Block.forward = block_forward


def to_fp32(_mod, _inputs, y):
    return y.float() if STREAM[0] else y


def set_arm(mods, arm):
    STREAM[0] = arm.endswith("_s32")
    for m in mods:
        if arm.startswith("bf16"):
            m.__class__ = nn.Linear
        elif arm == "int8_in16":
            m.__class__ = In16
        else:
            m.__class__ = Int8TrainLinear


class Sketcher:
    def __init__(self, targets, q, device, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.omega = {
            ln: torch.randn(mod.out_features, q, generator=gen).to(device)
            for ln, _o, mod in targets
        }
        self.out = {}
        self.capturing = False
        for ln, _o, mod in targets:
            mod.register_forward_hook(self._hook(ln))

    def _hook(self, ln):
        def hook(_mod, inputs, y):
            if not self.capturing or not y.requires_grad:
                return
            x = inputs[0].detach().to("cpu")

            def on_grad(dy):
                d = dy.reshape(-1, dy.shape[-1]).float() @ self.omega[ln]
                xx = x.to(dy.device).reshape(-1, x.shape[-1]).float()
                self.out[ln] = d.T @ xx

            y.register_hook(on_grad)

        return hook

    def take(self):
        out, self.out = self.out, {}
        return out


def pick_images(root, n, lo, hi, seed):
    pairs = discover_cached_pairs(str(root))
    gen = torch.Generator().manual_seed(seed)
    picked = []
    for i in torch.randperm(len(pairs), generator=gen).tolist():
        lat = load_cached_latents(pairs[i].npz_path)[0]
        if lo <= latent_tokens(lat) <= hi:
            picked.append(pairs[i])
        if len(picked) == n:
            break
    return picked


def load_batch(ci, j, seed, device):
    latents = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
    crossattn, _ = load_cached_text_features(ci.te_path, variant=0)
    crossattn = crossattn.unsqueeze(0).to(device, torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed * 1000 + j)
    noise = torch.randn(latents.shape, generator=gen, device=device)
    return latents, crossattn, noise


def one_pass(anima, sk, latents, crossattn, noise, sigma, fp32=False):
    dtype = torch.float32 if fp32 else torch.bfloat16
    s4 = sigma.view(-1, 1, 1, 1)
    noisy = ((1.0 - s4) * latents + s4 * noise).unsqueeze(2).to(dtype)
    noisy.requires_grad_(True)
    pad = torch.zeros(1, 1, *latents.shape[-2:], dtype=dtype, device=latents.device)
    sk.capturing = True
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
        pred = anima(noisy, sigma, crossattn.to(dtype), padding_mask=pad)
    sk.capturing = False
    pred = pred.squeeze(2).float()
    loss = torch.nn.functional.mse_loss(pred, noise - latents)
    loss.backward()
    return sk.take(), float(loss), pred.detach()


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap, vae=False, text_encoder=False)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--sketch_q", type=int, default=64)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument(
        "--out", default="project/int8_train/assets/anima_stream_error.json"
    )
    args = ap.parse_args()
    args.gradient_checkpointing = True
    start_heartbeat(label="stream_error")
    for _ in range(90):
        if torch.cuda.mem_get_info()[0] > 14e9:
            break
        time.sleep(1)
    print(f"free VRAM {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)

    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    targets = enumerate_targets(anima)
    sk = Sketcher(targets, args.sketch_q, device, args.seed)
    images = pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    sigmas = [float(s) for s in args.sigmas.split(",")]
    t0 = time.perf_counter()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtypes = {
        n: t.dtype for n, t in [*anima.named_parameters(), *anima.named_buffers()]
    }
    attn_mode = anima.attn_mode
    anima.float()
    anima.attn_mode = "torch"
    ref = {}
    for sigma_v in sigmas:
        sigma = torch.tensor([sigma_v], device=device)
        for j, ci in enumerate(images):
            latents, crossattn, noise = load_batch(ci, j, args.seed, device)
            S, loss, pred = one_pass(
                anima, sk, latents, crossattn, noise, sigma, fp32=True
            )
            ref[sigma_v, j] = ({k: v.cpu() for k, v in S.items()}, pred.cpu(), loss)
            print(
                f"fp32 σ={sigma_v} {j + 1}/{len(images)} loss={loss:.4f} "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    for n, t in [*anima.named_parameters(), *anima.named_buffers()]:
        t.data = t.data.to(dtypes[n])
    anima.attn_mode = attn_mode
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    path = forward_file(None, args.dit)
    print(f"forward copies: {path}", flush=True)
    names = convert_dit_(anima, "fwd", path or None)
    mods = [anima.get_submodule(n) for n in names]
    for m in mods:
        wq_t, s_t = quantize(m.weight.detach().t().contiguous())
        m.held_wq_t = HeldCopy(wq_t)
        m.register_buffer("s_t", s_t, persistent=False)
        m.int8_mode = "full"
    for block in anima.blocks:
        for mod in (
            block.adaln_up_self_attn,
            block.adaln_up_cross_attn,
            block.adaln_up_mlp,
        ):
            mod.register_forward_hook(to_fp32)

    pred_err = {s: defaultdict(list) for s in sigmas}
    losses = {s: defaultdict(list) for s in sigmas}
    grad = {s: defaultdict(lambda: defaultdict(list)) for s in sigmas}
    block_of = {
        ln: int(re.search(r"blocks\.(\d+)\.", o)[1])
        for ln, o, _ in targets
        if re.search(r"blocks\.(\d+)\.", o)
    }
    for sigma_v in sigmas:
        sigma = torch.tensor([sigma_v], device=device)
        for j, ci in enumerate(images):
            latents, crossattn, noise = load_batch(ci, j, args.seed, device)
            S32, pred32, loss32 = ref[sigma_v, j]
            pred32 = pred32.to(device)
            losses[sigma_v]["fp32"].append(loss32)
            S = {}
            for arm in ARMS:
                set_arm(mods, arm)
                S[arm], loss, pred = one_pass(
                    anima, sk, latents, crossattn, noise, sigma
                )
                losses[sigma_v][arm].append(loss)
                pred_err[sigma_v][arm].append(
                    float((pred - pred32).norm() / pred32.norm())
                )
            dist = defaultdict(lambda: defaultdict(float))
            for ln, block in block_of.items():
                if ln not in S32:
                    continue
                r = S32[ln].to(device)
                for arm in ARMS:
                    dist[arm][block // 7] += float((S[arm][ln] - r).pow(2).sum())
            for arm in ARMS:
                for g, d in dist[arm].items():
                    grad[sigma_v][arm][g].append((d / dist["bf16"][g]) ** 0.5)
            print(
                f"σ={sigma_v} {j + 1}/{len(images)} tok={latent_tokens(latents)} "
                + " ".join(f"{a}={pred_err[sigma_v][a][-1] * 100:.3f}" for a in ARMS)
                + f" {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
            del S

    def p90(v):
        return sorted(v)[max(0, int(round(0.9 * (len(v) - 1))))]

    report = {"images": [c.npz_path for c in images], "forward_copies": path}
    print("\nRESULT prediction error, % of ‖v_fp32‖ (mean over images)")
    for s in sigmas:
        row = {a: statistics.mean(pred_err[s][a]) * 100 for a in ARMS}
        print(f"RESULT σ={s} " + " ".join(f"{a}={v:.3f}" for a, v in row.items()))
        report[f"pred_{s}"] = pred_err[s]
    print("RESULT loss (mean)")
    for s in sigmas:
        print(
            f"RESULT σ={s} "
            + " ".join(f"{a}={statistics.mean(v):.5f}" for a, v in losses[s].items())
        )
    print("RESULT gradient distance over bf16's, median [p90], blocks 0-6..21-27")
    for s in sigmas:
        for a in ARMS[1:]:
            cells = [
                f"{statistics.median(grad[s][a][g]):.2f} [{p90(grad[s][a][g]):.2f}]"
                for g in sorted(grad[s][a])
            ]
            print(f"RESULT σ={s} {a}: " + " | ".join(cells))
        report[f"grad_{s}"] = {a: dict(grad[s][a]) for a in ARMS}
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
