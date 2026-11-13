#!/usr/bin/env python3
"""Is there a foreground mask in MOSS-Image's massive activations?

Question 1 of ``docs/moss/proposal/massive_activation_semantics.md``: the spatial
analysis of arXiv 2605.13974 §3.2 on MOSS. The shipped sampler (Euler, CFG, uncond
skip 22, reference per-sample forward) on Base with the fp32 stream. On the probe
steps, each block's output on the cond branch is restricted to a channel set, the
image tokens are clustered with K-means (K = 2), and the cluster with the higher
mean score ``Σ_c x̂[n, c]`` (``x̂``: the set's maps min-max normalised per channel)
is the foreground, as in the paper. The masks are scored against BiRefNet on the
final image, pooled to the token grid.

Channel sets (per sample and block, by ``|mean over image tokens|``):

  top12 | top2 | bottom12 | rand12 (one fixed draw per block) |
  fixed18 (``library/moss/calibration/massive_channels.npz``) | rest (all but top12)

and two readings of the paper's K-means input: ``raw`` (the restricted
activations) and ``norm`` (their min-max normalised maps).

Per (render, step, block, reading, set): mIoU (mean of foreground and background
IoU) under the paper's polarity rule, the same with the better polarity (oracle),
foreground IoU, and the mIoU of a random mask of the same area.

Gate (fixed before the run): at the last step, on the block picked on the first
seed's renders, top12 scores mIoU ≥ 0.5 on the other seeds' renders and is ≥ 0.1
above bottom12 and rand12 there, in either reading.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_ma_mask/probe_masks.py --label base"

Writes ``masks.npz`` (packed masks, BiRefNet masks, top12 heatmaps and channels),
``scores.npz``, ``images/``, ``probe.png``, ``sheet.png`` and ``result.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bench._common import make_run_dir, start_heartbeat, write_result

ARMS = ("top12", "top2", "bottom12", "rand12", "fixed18", "rest")
READINGS = ("raw", "norm")
K = 12

PROMPTS = {
    "sunset": "An anime illustration of a girl with long silver hair and red eyes, wearing a black "
    "school uniform, sitting on a classroom windowsill at sunset, cherry blossom petals drifting "
    "outside the window, detailed cel shading, soft rim light.",
    "knight": "An anime illustration of a knight in silver armor standing in a wheat field under a "
    "cloudy sky, holding a long sword, wind blowing her red cape.",
    "two_girls": "An anime illustration of two girls walking side by side on a snowy street at night, "
    "one in a red coat and one in a blue coat, street lamps glowing behind them.",
    "bicycle": "An anime illustration of a boy riding a bicycle along a seaside road, summer clouds "
    "over the ocean, his white shirt fluttering.",
    "witch": "An anime illustration of a witch with a large pointed hat sitting on a stack of books in "
    "a cluttered library, a black cat beside her.",
    "portrait": "An anime-style portrait of a woman with short black hair and glasses in front of a "
    "city skyline at dusk, upper body, looking at the viewer.",
    "raincoat": "An anime illustration of a small girl in a yellow raincoat holding a transparent "
    "umbrella, standing in a field of blue hydrangeas in the rain.",
    "chibi": "A chibi character with pink twin tails jumping in front of a plain pastel green "
    "background, full body.",
    "dog": "A photo of a golden retriever sitting on a wooden pier by a lake, mountains in the "
    "background.",
    "car": "A photo of a red vintage car parked on a cobblestone street in front of an old stone "
    "building.",
    "teapot": "A photo of a white ceramic teapot on a wooden table, a blurred kitchen behind it.",
    "subway": "A photo of a man in a gray suit standing in the middle of an empty subway platform.",
    "lighthouse": "An oil painting of a lighthouse on a rocky cliff above a stormy sea.",
    "fox": "A watercolor painting of a fox curled up asleep under a maple tree in autumn.",
    "robot": "A 3D render of a small orange robot standing on a white desk next to a potted cactus.",
    "manga_bow": "A black-and-white manga panel of a girl with a ponytail drawing a bow, speed lines "
    "behind her.",
}


def kmeans2(x, score, iters=12):
    """K-means (K = 2) on ``x (N, k)`` from two deterministic inits (a median split
    on ``score``; the point farthest from the mean and the point farthest from
    that), lower inertia kept. Returns bool ``(N,)``: True on the cluster whose
    mean ``score`` is higher."""
    x = x - x.mean(0)
    hi = score > score.median()
    if hi.all() or not hi.any():
        hi = torch.arange(x.shape[0], device=x.device) % 2 == 0
    far = x[x.square().sum(-1).argmax()]
    far2 = x[(x - far).square().sum(-1).argmax()]
    c0 = torch.stack([x[~hi].mean(0), far])  # (2 inits, k)
    c1 = torch.stack([x[hi].mean(0), far2])
    for _ in range(iters):
        d0 = (x[None] - c0[:, None]).square().sum(-1)  # (2, N)
        d1 = (x[None] - c1[:, None]).square().sum(-1)
        w = (d1 < d0).to(x.dtype)
        n1 = w.sum(1, keepdim=True)
        n0 = x.shape[0] - n1
        c1 = torch.where(n1 > 0, (w @ x) / n1.clamp_min(1), c1)
        c0 = torch.where(n0 > 0, ((1 - w) @ x) / n0.clamp_min(1), c0)
    d0 = (x[None] - c0[:, None]).square().sum(-1)
    d1 = (x[None] - c1[:, None]).square().sum(-1)
    best = torch.minimum(d0, d1).sum(1).argmin()
    lab = d1[best] < d0[best]
    if lab.all() or not lab.any():
        return lab
    return lab if score[lab].mean() >= score[~lab].mean() else ~lab


class Probe:
    """``layer_hook`` for the cond forward of the probe steps; one render's masks
    stay on the card until ``take``."""

    def __init__(self, num_blocks, probe_steps, n_img, dim, fixed, device):
        self.L, self.n_img, self.device = num_blocks, n_img, device
        self.slot = {s: j for j, s in enumerate(probe_steps)}
        self.last = len(probe_steps) - 1
        self.fixed = fixed
        gen = torch.Generator("cpu").manual_seed(0)
        self.rand = torch.stack(
            [torch.randperm(dim, generator=gen)[:K] for _ in range(num_blocks)]
        ).to(device)
        self.fi = -1
        self.new_render()

    def new_render(self):
        s, d = len(self.slot), self.device
        self.fi = -1
        self.labels = torch.zeros(
            s, self.L, len(READINGS), len(ARMS), self.n_img, dtype=torch.bool, device=d
        )
        self.heat = torch.zeros(self.L, self.n_img, device=d)
        self.top = torch.zeros(s, self.L, K, dtype=torch.int16, device=d)
        self.dc = torch.zeros(s, self.L, device=d)

    def __call__(self, index, x, n_img):
        # per-sample forwards: cond is the even one of each step's pair
        if self.fi % 2 or self.fi // 2 not in self.slot:
            return x
        j = self.slot[self.fi // 2]
        h = x[0, :n_img].float()
        mu = h.mean(0)
        order = mu.abs().argsort(descending=True)
        top = order[:K]
        rest = order[K:]
        sets = dict(
            top12=top,
            top2=top[:2],
            bottom12=order[-K:],
            rand12=self.rand[index],
            fixed18=self.fixed,
            rest=rest,
        )
        self.top[j, index] = top.to(torch.int16)
        self.dc[j, index] = mu[top].square().sum() / h[:, top].square().mean(0).sum()
        for a, name in enumerate(ARMS):
            xa = h[:, sets[name]]
            lo, hi = xa.min(0).values, xa.max(0).values
            xn = (xa - lo) / (hi - lo).clamp_min(1e-12)
            score = xn.sum(-1)
            self.labels[j, index, 0, a] = kmeans2(xa, score)
            self.labels[j, index, 1, a] = kmeans2(xn, score)
            if name == "top12" and j == self.last:
                self.heat[index] = score
        return x

    def take(self):
        return (
            np.packbits(self.labels.cpu().numpy(), axis=-1),
            self.heat.cpu().numpy().astype(np.float16),
            self.top.cpu().numpy(),
            self.dc.cpu().numpy(),
        )


def birefnet_masks(path, images, grid, device):
    """BiRefNet foreground probability of each PIL image, area-pooled to the token
    grid → ``(R, grid, grid)`` fp32."""
    import torch.nn.functional as F
    from transformers import AutoModelForImageSegmentation

    net = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True)
    net = net.eval().float().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for im in images:
        px = torch.from_numpy(np.asarray(im.convert("RGB"))).to(device)
        px = px.permute(2, 0, 1)[None].float() / 255
        px = F.interpolate(px, size=(1024, 1024), mode="bilinear", antialias=True)
        with torch.no_grad():
            prob = net((px - mean) / std)[-1].sigmoid()
        out.append(F.adaptive_avg_pool2d(prob, grid)[0, 0].cpu().numpy())
    del net
    torch.cuda.empty_cache()
    return np.stack(out)


def iou_scores(lab, gt):
    """``lab (..., N)`` bool against ``gt (N,)`` bool → rule mIoU, flipped-polarity
    mIoU, foreground IoU, same-area random-mask mIoU, predicted foreground share."""

    def iou(a, b):
        inter = (a & b).sum(-1)
        union = (a | b).sum(-1)
        return np.where(union > 0, inter / np.maximum(union, 1), 1.0)

    fg = iou(lab, gt)
    miou = (fg + iou(~lab, ~gt)) / 2
    flip = (iou(~lab, gt) + iou(lab, ~gt)) / 2
    p, f = lab.mean(-1), gt.mean()
    chance = (
        p * f / np.maximum(p + f - p * f, 1e-9)
        + (1 - p) * (1 - f) / np.maximum(1 - p * f, 1e-9)
    ) / 2
    return miou, flip, fg, chance, p


def plot_curves(path, miou, oracle, chance, steps):
    """``miou`` / ``oracle`` / ``chance`` ``(S, L, V, A)`` render means."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    blocks = np.arange(miou.shape[1])
    for v, reading in enumerate(READINGS):
        for ax, m, title in (
            (axes[v, 0], miou, "mIoU, paper polarity"),
            (axes[v, 1], oracle, "mIoU, better polarity"),
        ):
            for a, name in enumerate(ARMS):
                ax.plot(blocks, m[-1, :, v, a], label=name, lw=2 if a == 0 else 1)
            ax.plot(
                blocks, chance[-1, :, v, 0], "k:", lw=1, label="chance (top12 area)"
            )
            ax.set_title(f"{reading}: {title}, step {steps[-1]}")
            ax.set_xlabel("block")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
        axes[v, 0].legend(fontsize=8)
        im = axes[v, 2].imshow(
            miou[:, :, v, 0],
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        axes[v, 2].set_yticks(range(len(steps)), steps)
        axes[v, 2].set_title(f"{reading}: top12 mIoU (paper polarity), step × block")
        axes[v, 2].set_xlabel("block")
        axes[v, 2].set_ylabel("step")
        fig.colorbar(im, ax=axes[v, 2])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_sheet(path, rows, grid, block, reading):
    """``rows``: ``(name, PIL image, gt (N,), heat (N,), {arm: mask (N,)})``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3 + len(ARMS)
    fig, axes = plt.subplots(len(rows), cols, figsize=(2.2 * cols, 2.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, image, gt, heat, masks) in enumerate(rows):
        cells = [
            (image, "image", None),
            (gt.reshape(grid, grid), "BiRefNet", "gray"),
            (heat.reshape(grid, grid).astype(np.float32), "top12 score", "magma"),
        ] + [(masks[a].reshape(grid, grid), a, "gray") for a in ARMS]
        for c, (cell, title, cmap) in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(cell, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle(f"block {block}, last step, K-means on {reading}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    from library.moss.requests import DEFAULT_TEXT_ENCODER

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="base")
    ap.add_argument("--prompts", default=",".join(PROMPTS), help="keys of PROMPTS")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--probe_steps", default="0,3,7,11,15,19,23,29")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--stream", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    ap.add_argument("--prompt_tail", default="sure")
    ap.add_argument(
        "--massive", default="library/moss/calibration/massive_channels.npz"
    )
    ap.add_argument(
        "--birefnet", default="/media/sorryhyun/data/moss_image/bench/birefnet"
    )
    ap.add_argument("--model_dir", default="")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if args.cfg == 1.0:
        raise SystemExit("the probe reads the cond forward of a CFG pair")
    start_heartbeat(label="moss_ma_mask")

    from diffusers.image_processor import VaeImageProcessor

    from library.env import resolve_under_home
    from library.moss.generate import cached_embeds, denoise, initial_noise
    from library.moss.loader import (
        compile_blocks,
        load_transformer,
        load_vae,
        pad_embeds,
        stream_fp32_,
    )
    from library.moss.model import decode, snapped_sigmas
    from library.moss.requests import DEFAULT_UNCOND_SKIP, resolve_model_dir

    keys = [k for k in args.prompts.split(",") if k]
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    run_dir = make_run_dir("moss_ma_mask", args.label)
    (run_dir / "images").mkdir(exist_ok=True)
    fixed = torch.from_numpy(np.load(resolve_under_home(args.massive))["channels"]).to(
        device
    )

    embeds = cached_embeds(
        model_dir,
        [PROMPTS[k] for k in keys] + [""],
        Path(args.te_cache),
        args.text_encoder,
        "off",
        args.prompt_tail,
    )
    uncond, uncond_mask = pad_embeds([embeds[""]], device=device)
    transformer = load_transformer(model_dir, args.model, device=device)
    if args.stream == "fp32":
        stream_fp32_(transformer)
    if args.compile:
        compile_blocks(transformer.layers)
    num_blocks, dim = len(transformer.layers), transformer.config.dim
    sigmas = snapped_sigmas(args.steps)
    probe_steps = [int(s) for s in args.probe_steps.split(",") if s]
    probe_steps = [s for s in probe_steps if s < len(sigmas)]
    grid = args.size // 16
    n_img = grid * grid

    probe = Probe(num_blocks, probe_steps, n_img, dim, fixed, device)

    def on_forward(_m, _a):
        probe.fi += 1

    handle = transformer.register_forward_pre_hook(on_forward)
    transformer.layer_hook = probe
    renders, latents, packed, heats, tops, dcs = [], [], [], [], [], []
    try:
        for s in range(args.seeds):
            for key in keys:
                cond, cond_mask = pad_embeds([embeds[PROMPTS[key]]], device=device)
                probe.new_render()
                latents.append(
                    denoise(
                        transformer,
                        initial_noise(args.seed0 + s, args.size, args.size).to(device),
                        cond,
                        cond_mask,
                        uncond,
                        uncond_mask,
                        sigmas,
                        cfg=args.cfg,
                        uncond_skip=DEFAULT_UNCOND_SKIP,
                    ).cpu()
                )
                p, h, t, d = probe.take()
                packed.append(p)
                heats.append(h)
                tops.append(t)
                dcs.append(d)
                renders.append((key, s))
                print(f"[probe] {key} seed {s} ({len(renders)})", flush=True)
    finally:
        transformer.layer_hook = None
        handle.remove()
    transformer.to("cpu")
    del transformer
    probe.labels = probe.heat = None
    torch.cuda.empty_cache()

    vae = load_vae(model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    images = []
    for (key, s), z in zip(renders, latents):
        image = processor.postprocess(decode(z.to(device), vae), output_type="pil")[0]
        image.save(run_dir / "images" / f"{key}_s{s}.png")
        images.append(image)
    del vae
    torch.cuda.empty_cache()
    prob = birefnet_masks(args.birefnet, images, grid, device)
    gt = prob.reshape(len(renders), n_img) > 0.5

    # --- scoring -------------------------------------------------------------
    stats = []
    for r in range(len(renders)):
        lab = np.unpackbits(packed[r], axis=-1, count=n_img).astype(bool)
        stats.append(np.stack(iou_scores(lab, gt[r])))  # (5, S, L, V, A)
    stats = np.stack(stats)  # (R, 5, S, L, V, A)
    miou, flip, fg, chance, share = (stats[:, i] for i in range(5))
    oracle = np.maximum(miou, flip)
    seed_of = np.array([s for _, s in renders])
    pick = seed_of == 0
    held = seed_of != 0 if args.seeds > 1 else pick
    a_top, a_bot, a_rand = (ARMS.index(n) for n in ("top12", "bottom12", "rand12"))

    gate, per_reading = False, {}
    for v, reading in enumerate(READINGS):
        block = int(miou[pick, -1, :, v, a_top].mean(0).argmax())
        top, bot, rnd = (
            float(miou[held, -1, block, v, a].mean()) for a in (a_top, a_bot, a_rand)
        )
        passed = bool(top >= 0.5 and top - bot >= 0.1 and top - rnd >= 0.1)
        gate = gate or passed
        per_reading[reading] = {
            "block": block,
            "top12": top,
            "bottom12": bot,
            "rand12": rnd,
            "passed": passed,
            "by_arm_last_step": {
                name: {
                    "miou": miou[:, -1, :, v, a].mean(0).round(4).tolist(),
                    "miou_oracle": oracle[:, -1, :, v, a].mean(0).round(4).tolist(),
                    "fg_iou": fg[:, -1, :, v, a].mean(0).round(4).tolist(),
                    "chance": chance[:, -1, :, v, a].mean(0).round(4).tolist(),
                    "fg_share": share[:, -1, :, v, a].mean(0).round(4).tolist(),
                }
                for a, name in enumerate(ARMS)
            },
            "top12_miou_step_block": miou[..., v, a_top].mean(0).round(4).tolist(),
            "top12_oracle_step_block": oracle[..., v, a_top].mean(0).round(4).tolist(),
        }

    tops = np.stack(tops)  # (R, S, L, K)
    fixed_np = fixed.cpu().numpy()
    in_fixed = np.isin(tops, fixed_np).mean((0, 1, 3))  # (L,)
    np.savez_compressed(
        run_dir / "masks.npz",
        labels=np.stack(packed),
        gt=gt,
        gt_prob=prob.astype(np.float16),
        heat=np.stack(heats),
        top=tops,
        dc=np.stack(dcs),
        renders=np.array([f"{k}_s{s}" for k, s in renders]),
        probe_steps=np.array(probe_steps),
        arms=np.array(ARMS),
        readings=np.array(READINGS),
    )
    np.savez_compressed(
        run_dir / "scores.npz",
        miou=miou.astype(np.float32),
        flip=flip.astype(np.float32),
        fg=fg.astype(np.float32),
        chance=chance.astype(np.float32),
        share=share.astype(np.float32),
    )
    plot_curves(
        run_dir / "probe.png", miou.mean(0), oracle.mean(0), chance.mean(0), probe_steps
    )
    best_v = max(range(len(READINGS)), key=lambda v: per_reading[READINGS[v]]["top12"])
    block = per_reading[READINGS[best_v]]["block"]
    rows = []
    for r in np.flatnonzero(pick):
        lab = np.unpackbits(packed[r][-1, block, best_v], axis=-1, count=n_img)
        rows.append(
            (
                renders[r][0],
                images[r],
                gt[r],
                heats[r][block],
                {a: lab[i] for i, a in enumerate(ARMS)},
            )
        )
    plot_sheet(run_dir / "sheet.png", rows, grid, block, READINGS[best_v])

    metrics = {
        "gate_passed": gate,
        "renders": len(renders),
        "gt_fg_share": {
            f"{k}_s{s}": round(float(g.mean()), 4) for (k, s), g in zip(renders, gt)
        },
        "top12_in_fixed18_by_block": in_fixed.round(4).tolist(),
        "top12_dc_energy_by_block": np.stack(dcs).mean((0, 1)).round(5).tolist(),
        "readings": per_reading,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["masks.npz", "scores.npz", "probe.png", "sheet.png"],
    )
    for reading, m in per_reading.items():
        print(
            f"[gate] {reading}: block {m['block']} top12 {m['top12']:.3f} "
            f"bottom12 {m['bottom12']:.3f} rand12 {m['rand12']:.3f} "
            f"{'PASS' if m['passed'] else 'fail'}",
            flush=True,
        )
    print(f"[done] gate {'PASS' if gate else 'fail'} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
