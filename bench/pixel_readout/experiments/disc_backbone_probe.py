#!/usr/bin/env python3
"""Which frozen tower sees the ×1 head's errors best: a linear probe per block.

The ×1 GAN finish's discriminator is frozen PE-Spatial-B16 with token heads on blocks
3 / 7 / 11 (``PEDisc``). This probe asks, before any DINO discriminator is written,
whether a DINOv3 tower separates the head's output from the original image better than PE
does (the selection rule of Vision-aided GAN, Kumari et al., CVPR 2022: linear-probe
accuracy of real against generated on the backbone's features).

Per image and σ: one DiT forward at the :real point, the head's output, and the original
image, both whole images in [-1, 1] as the GAN term sees them. Each tower runs on both;
``tokens`` random token places per image (the same places in both) keep every block's
patch tokens. Two probes per block and tower, fit on half the images and scored on the
other half (image-level split, alternating within each image group):

- ``unpaired``: logistic regression, real token = 1, head token = 0; test accuracy.
- ``paired``: the linear form of the paired relativistic D. A weight w with
  w · (φ(real) − φ(head)) > 0 at the same place; test accuracy = the share of test places
  where it holds.

Towers: PE-Spatial-B16 (768-d), DINOv3 ViT-S/16 (384-d, the DINO term's tower) and DINOv3
ConvNeXt-T, whose four stage maps (4 / 8 / 16 /
32 px per position, no 16-px patch grid) count as its "blocks". Inputs: PE takes [-1, 1] as
``PEDisc`` does, DINO the ImageNet-normalised image as ``DinoFeat`` does. The places are
drawn per block on that block's own grid.

    make daemon-run ARGS="--queue --stall-timeout 900 \
        bench/pixel_readout/experiments/disc_backbone_probe.py --label discprobe_1123_HHMM"

Writes ``bench/pixel_readout/results/<stamp>-<label>/result.json`` (per tower, σ, block,
probe: train / test accuracy).
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from library.lines.requests import request_field as _f
from scripts.pixel_head._x1 import X1Request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HEAD = "output/pixel_head/x1dll_1123_0821/head.safetensors"
TOWERS = (
    "pe",
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
)


@dataclass
class ProbeRequest(X1Request):
    """Linear probe of real against the head's output on frozen towers."""

    init: str = _f(HEAD, "the ×1 head whose output is probed")
    sigmas: str = _f(
        "0.0577 0.1364", "the :real points read (the trained range's ends)"
    )
    n_train_items: int = _f(64, "training-set images added to the held-out ones")
    tokens: int = _f(384, "token places kept per image")
    fit_steps: int = _f(400, "full-batch Adam steps per probe")
    weight_decay: float = _f(1e-3, "L2 on the probe's weight")
    towers: str = _f(",".join(TOWERS), "comma list: pe and / or DINOv3 repos")
    label: str = _f("discprobe", "results folder label")


def main() -> None:
    import torch
    import torch.nn.functional as F

    from bench._common import make_run_dir, start_heartbeat, write_result
    from library.anima.pixel_head import load_pixel_head
    from library.anima.pixel_head_train_x1 import (
        X1Rollout,
        held_out_split,
        item_tokens,
    )
    from library.datasets.buckets import cluster_token_bands
    from library.env import resolve_under_home
    from library.inference.text import prepare_text_inputs
    from scripts.pixel_head._setup import load_models

    req = ProbeRequest.from_argv()
    start_heartbeat()
    device = torch.device(req.device)
    torch.manual_seed(req.seed)
    run_dir = make_run_dir("pixel_readout", req.label)
    sigmas = [float(s) for s in req.sigmas.split()]
    towers = [t.strip() for t in req.towers.split(",") if t.strip()]

    # --- images: training sample + held-out, each group split alternately ---
    data = held_out_split(Path(resolve_under_home(req.cache_root)))
    rng = random.Random(req.seed)
    groups = {
        "train": rng.sample(data["train"], req.n_train_items),
        "test": data["test"],
        "text": data["text"],
    }
    if req.pages:
        root = Path(resolve_under_home(req.pages))
        groups["manga"] = sorted((root / "eval").glob("*.safetensors"))
    items = [(it, g, i % 2) for g, its in groups.items() for i, it in enumerate(its)]
    seq_bands = cluster_token_bands([item_tokens(it) for it, _, _ in items])
    print(
        f"images {len(items)} ({', '.join(f'{g} {len(v)}' for g, v in groups.items())})"
        f"  σ {sigmas}  towers {towers}",
        flush=True,
    )

    # --- models ---
    vae, anima, network = load_models(req, device, seq_bands, adapter="")
    del vae
    page_cross = None
    if req.pages:
        arg_c, _ = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=req.page_caption,
            text_encoder_path=str(resolve_under_home(req.text_encoder)),
        )
        page_cross = arg_c["embed"][0].to(device, torch.bfloat16)
    rollout = X1Rollout(anima, None, device, req.seed, network, page_cross)
    head = load_pixel_head(
        Path(resolve_under_home(req.init)), device, anima.final_layer.hidden_size
    ).eval()

    # --- towers: (B, 3, H, W) in [-1, 1] → [per block (B, tokens, C)] ---
    def pe_tower():
        from library.models.pe import load_pe_spatial

        pe = load_pe_spatial(device, dtype=torch.bfloat16).eval().requires_grad_(False)
        n = len(pe.transformer.resblocks)
        taps: dict[int, torch.Tensor] = {}
        for i in range(n):
            pe.transformer.resblocks[i].register_forward_hook(
                lambda _m, _a, o, i=i: taps.__setitem__(i, o)
            )
        skip = 1 if pe.use_cls_token else 0

        def run(x):
            taps.clear()
            pe.forward_features(x, layer_idx=-1)
            return [taps[i][:, skip:] for i in range(n)]

        return run, n

    def dino_tower(repo: str):
        from transformers import AutoModel

        net = AutoModel.from_pretrained(repo).to(device).eval().requires_grad_(False)
        cfg = net.config
        convnext = cfg.model_type == "dinov3_convnext"
        skip = 0 if convnext else 1 + cfg.num_register_tokens
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        def run(x):
            x = ((x + 1) / 2 - mean) / std
            hs = net(pixel_values=x, output_hidden_states=True).hidden_states
            if convnext:  # stage maps (B, C, h, w) → (B, h·w, C)
                return [h.flatten(2).transpose(1, 2) for h in hs[1:]]
            return [h[:, skip:] for h in hs[1:]]

        return run, len(cfg.hidden_sizes) if convnext else cfg.num_hidden_layers

    def build(name):
        return pe_tower() if name == "pe" else dino_tower(name)

    # --- features: [tower][σ] → (real, fake) lists of (tokens, C) per block, split ids ---
    feats = {t: {s: [] for s in sigmas} for t in towers}
    split = []
    t0 = time.perf_counter()
    pairs = []  # (j, σ, real, fake) on the CPU, so the towers load one at a time
    for j, (item, _g, half) in enumerate(items):
        for s in sigmas:
            with torch.no_grad():
                f, lat, _ = rollout(item, 10_000 + j, s, False, ())
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    fake = head(f.float(), lat).float().clamp(-1, 1)
            hw = tuple(fake.shape[-2:])
            real = rollout.real(item, hw)
            pairs.append(
                (j, s, real.to("cpu", torch.float16), fake.to("cpu", torch.float16))
            )
        split.append(half)
        if (j + 1) % 16 == 0:
            print(
                f"rollouts {j + 1}/{len(items)}  {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    del anima, network, rollout, head
    torch.cuda.empty_cache()

    n_blocks = {}
    for t in towers:
        run, n_blocks[t] = build(t)
        tt = time.perf_counter()
        for j, s, real, fake in pairs:
            x = torch.stack([real, fake]).to(device).float()
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                out = run(x)
            kept = []
            for k, b in enumerate(out):  # places drawn on the block's own grid
                g = torch.Generator().manual_seed(30_000 + 100 * j + k)
                pl = torch.randperm(b.shape[1], generator=g)[: req.tokens].to(device)
                kept.append(
                    (
                        b[0, pl].to("cpu", torch.float16),
                        b[1, pl].to("cpu", torch.float16),
                    )
                )
            feats[t][s].append(kept)
        print(
            f"tower {t}: {n_blocks[t]} blocks, {time.perf_counter() - tt:.0f}s",
            flush=True,
        )
        del run
        torch.cuda.empty_cache()

    # --- probes ---
    split_t = torch.tensor(split)

    def stack(rows, blk, half):
        r = [rows[i][blk][0] for i in range(len(rows)) if split[i] == half]
        k = [rows[i][blk][1] for i in range(len(rows)) if split[i] == half]
        return torch.cat(r).to(device).float(), torch.cat(k).to(device).float()

    def fit(x, y, bias: bool):
        w = torch.zeros(x.shape[1], device=device, requires_grad=True)
        b = torch.zeros((), device=device, requires_grad=bias)
        opt = torch.optim.Adam([w, b] if bias else [w], lr=1e-2)
        for _ in range(req.fit_steps):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(x @ w + b, y)
            loss = loss + req.weight_decay * w.square().sum()
            loss.backward()
            opt.step()
        return w.detach(), b.detach()

    rows = []
    for t in towers:
        for s in sigmas:
            for blk in range(n_blocks[t]):
                rtr, ftr = stack(feats[t][s], blk, 0)
                rte, fte = stack(feats[t][s], blk, 1)
                mu = torch.cat([rtr, ftr]).mean(0)
                sd = torch.cat([rtr, ftr]).std(0) + 1e-6
                z = lambda a: (a - mu) / sd  # noqa: E731
                # unpaired: real 1, head 0
                x = torch.cat([z(rtr), z(ftr)])
                y = torch.cat([torch.ones(len(rtr)), torch.zeros(len(ftr))]).to(device)
                w, b = fit(x, y, bias=True)

                def acc(r, f, w=w, b=b):
                    real_ok = ((z(r) @ w + b) > 0).float().mean()
                    fake_ok = ((z(f) @ w + b) <= 0).float().mean()
                    return 0.5 * (real_ok + fake_ok).item()

                un_tr, un_te = acc(rtr, ftr), acc(rte, fte)
                # paired: w · (φ(real) − φ(head)) > 0 at the same place
                d = (rtr - ftr) / sd
                x = torch.cat([d, -d])
                y = torch.cat([torch.ones(len(d)), torch.zeros(len(d))]).to(device)
                w, _ = fit(x, y, bias=False)
                pa_tr = ((d @ w) > 0).float().mean().item()
                pa_te = ((((rte - fte) / sd) @ w) > 0).float().mean().item()
                rows.append(
                    {
                        "tower": t,
                        "sigma": s,
                        "block": blk,
                        "unpaired_train": un_tr,
                        "unpaired_test": un_te,
                        "paired_train": pa_tr,
                        "paired_test": pa_te,
                    }
                )
                print(
                    f"{t.split('/')[-1]:>36}  σ {s:<7} block {blk:>2}  "
                    f"unpaired {un_tr:.3f} / {un_te:.3f}  paired {pa_tr:.3f} / {pa_te:.3f}",
                    flush=True,
                )

    best = {
        f"{t}@{s}": max(
            (r for r in rows if r["tower"] == t and r["sigma"] == s),
            key=lambda r: r["paired_test"],
        )
        for t in towers
        for s in sigmas
    }
    write_result(
        run_dir,
        script=__file__,
        args=req.__dict__,
        label=req.label,
        metrics={"best_paired_test": best},
        device=device,
        extra={
            "rows": rows,
            "n_images": len(items),
            "n_test_images": int(split_t.sum()),
            "groups": {g: len(v) for g, v in groups.items()},
        },
    )
    print(f"result {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
