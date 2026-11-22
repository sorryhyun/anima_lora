#!/usr/bin/env python3
"""LPIPS gradient into v: the pixel head against the VAE decode (forward + backward, no step).

``project/pixel_decoder/idea.md`` step 1. The question: can a pixel head stand in for the
VAE as the differentiable decode of a PixelGen-style LPIPS term on a LoRA? A head reads
the 2048-d input f of ``final_layer.linear`` (v = W f, W 64 × 2048), so its gradient
reaches v only through f's row-space part. Per (image, σ, noise draw), on one token window
of a teacher-forced base-DiT forward, against the target VAE(x0):

- **a** ∂ LPIPS(VAE(x_t − σ·v)) / ∂v: the reference, VAE in fp32.
- **a16** the same with the VAE in bf16: the reference's own rounding twin (ceiling).
- **b** the head, read in v space: g_v = (W Wᵀ)⁻¹ W · ∂L/∂f. This is the gradient a loss on
  ``f.detach() + P_W(f − f.detach())`` sends into v.
- **c** the head's raw ∂L/∂f, against a's pull-back Wᵀ g_a in f space.

Read per head and σ: cos(b, a) in v space (the first-order share of a's LPIPS decrease a
step along b buys) and in f space, the norm ratio, the row-space share of the raw
gradient, cos(c, Wᵀ g_a). Beside them: the floor cos(b on draw k, a on the other draw) and
cos(a, a) across the two draws (what a perfect decoder would read on that floor), the
head's PSNR against VAE(x̂0), cosines against the FM gradient v − (ε − x0), and time and
peak memory of each backward. Both decodes are clamped to [-1, 1] before LPIPS.

Each gradient is also checked by a step: v − η·ĝ along its unit direction, η set so that a's
own first-order LPIPS decrease is a share ``--fd_fracs`` of the loss, the loss re-read through
the fp32 VAE. Per share F: ``a.fdF_ratio`` = actual / predicted decrease for a (1 = the linear
regime); ``<arm>.fdF_share`` = an arm's actual decrease over a's (equals cos(arm, a) in that
regime); ``<head>.fdF_null`` = the same for the head's gradient from the other noise draw.

Images: P1's held-out test set (``--split train`` for images the heads trained on). The
DiT runs 128 forwards at mixed shapes (~20 s), so ``--compile`` is off by default.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/experiments/lpips_grad_probe.py --label lgrad_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout._shared import P1_RUN, resolve_run
from library.anima.pixel_head import PATCH, lat_patches, load_pixel_head
from library.anima.pixel_head_train import (
    Lpips,
    compile_dit,
    crop_margin,
    pair_sheet,
    pick_window,
    psnr,
)
from library.anima.pixel_head_train_x1 import item_tokens, pick_samples
from library.datasets.buckets import cluster_token_bands
from library.env import resolve_under_home
from library.io.cache import load_cached_latents, load_cached_text_features
from library.models import qwen_vae


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item()


def cos_tok_median(a: torch.Tensor, b: torch.Tensor) -> float:
    """Median over tokens of the per-token cosine; a, b (h, w, K)."""
    c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-30)
    return c.flatten().median().item()


def norm_map(g: torch.Tensor) -> torch.Tensor:
    """(h, w, K) gradient → its per-token norm as a (3, 16h, 16w) image in [-1, 1]."""
    n = g.norm(dim=-1)
    n = (n / n.max().clamp_min(1e-30)).float()
    n = n.repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)
    return (2 * n - 1)[None].expand(3, -1, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--p1_run", default=P1_RUN)
    p.add_argument(
        "--heads",
        nargs="+",
        default=["mixr=mixr_1119_1455", "realr=realr_1119_2200"],
        metavar="NAME=RUN",
        help="heads to read: a results dir or a run label holding head.safetensors",
    )
    p.add_argument("--sigmas", default="0.0577,0.1,0.1364,0.3,0.5,0.7")
    p.add_argument(
        "--fd_fracs",
        default="0.001,0.01,0.1",
        help="finite-difference steps: a's predicted LPIPS decrease as a share of the loss",
    )
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--n", type=int, default=16, help="images")
    p.add_argument("--draws", type=int, default=2, help="noise draws per (image, σ)")
    p.add_argument("--window", type=int, default=24, help="LPIPS window, tokens")
    p.add_argument("--margin", type=int, default=4, help="context strip, tokens")
    p.add_argument("--n_sheets", type=int, default=6)
    args = p.parse_args()
    assert args.draws >= 2, "the floor pairs each draw with the next one"
    start_heartbeat(label="lgrad")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    fracs = [float(s) for s in args.fd_fracs.split(",")]

    # --- data: P1's split ---
    cache_root = Path(resolve_under_home(args.cache_root))
    a1 = json.loads((Path(resolve_under_home(args.p1_run)) / "result.json").read_text())
    a1 = a1["args"]
    train, test, _ = pick_samples(
        cache_root,
        a1["n_train"],
        a1["n_test_artists"],
        a1["per_artist"],
        a1["max_tokens"],
        a1["seed"],
    )
    pool = list(test if args.split == "test" else train)
    random.Random(args.seed + 7).shuffle(pool)
    items = pool[: args.n]
    win_rng = random.Random(args.seed + 11)

    # --- models ---
    def load_vae(dtype):
        vae = qwen_vae.load_vae(
            args.vae, device="cpu", dtype=dtype, eval=True, vae_2d=True
        )
        return vae.to(device).requires_grad_(False)

    vaes = {"a": load_vae(torch.float32), "a16": load_vae(torch.bfloat16)}
    compile_blocks, args.compile = args.compile, False
    anima = build_anima(args, adapter=None, train_mode=False).anima
    args.compile = compile_blocks
    from library.anima.stream import stream_fp32_

    stream_fp32_(anima)
    if compile_blocks:
        compile_dit(anima, cluster_token_bands([item_tokens(ci) for ci in items]))
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )
    D = anima.final_layer.hidden_size
    W = anima.final_layer.linear.weight.detach().double()  # (64, D)
    gram = W @ W.T
    gram_inv = torch.linalg.inv(gram)
    heads = {}
    for spec in args.heads:
        name, _, run = spec.partition("=")
        heads[name] = load_pixel_head(
            resolve_run(run) / "head.safetensors", device, D
        ).requires_grad_(False)
    lp = Lpips(device)
    mem_base = torch.cuda.memory_allocated() / 2**30
    print(
        f"{args.split} images {len(items)}  sigmas {sigmas}  draws {args.draws}  "
        f"heads {list(heads)}  cond(W Wt) {torch.linalg.cond(gram).item():.1f}  "
        f"models on card {mem_base:.2f} GB",
        flush=True,
    )

    def unpatch(v_tok: torch.Tensor) -> torch.Tensor:
        """(h, w, 64) velocity tokens → (16, 2h, 2w), the DiT's own layout."""
        return anima.unpatchify(v_tok[None, None])[0, :, 0]

    def lpips_term(img: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return lp.net(img[None].float(), tgt[None]).mean()

    def timed(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return out, dt, (torch.cuda.max_memory_allocated() - base) / 2**30

    def vae_grad(vae, v_tok, xt_c, sigma, tgt, mar):
        leaf = v_tok.detach().clone().requires_grad_(True)
        img = vae.decode_to_pixels((xt_c - sigma * unpatch(leaf))[None])[0].float()
        loss = lpips_term(crop_margin(img, mar, PATCH), tgt)
        (g,) = torch.autograd.grad(loss, leaf)
        return g.double(), loss.item(), img.detach()

    @torch.no_grad()
    def vae_loss(v_tok, xt_c, sigma, tgt, mar) -> float:
        img = vaes["a"].decode_to_pixels((xt_c - sigma * unpatch(v_tok))[None])[0]
        return lpips_term(crop_margin(img.float(), mar, PATCH), tgt).item()

    def head_grad(head, f_w, lat_p, tgt, mar):
        leaf = f_w.detach().clone().requires_grad_(True)
        img = head(leaf, lat_p).clamp(-1, 1)
        loss = lpips_term(crop_margin(img, mar, PATCH), tgt)
        (g,) = torch.autograd.grad(loss, leaf)
        return g.double(), loss.item(), img.detach()

    run_dir = make_run_dir("pixel_readout", args.label)
    rows: list[dict] = []
    n_sheets = 0
    for i, ci in enumerate(items):
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device).float()
        cross, _ = load_cached_text_features(ci.te_path, variant=0)
        cross = cross.unsqueeze(0).to(device, torch.bfloat16)
        gh, gw = lat.shape[-2] // 2, lat.shape[-1] // 2
        win = pick_window(gh, gw, args.window, args.margin, win_rng)
        ys = slice(2 * win.y0, 2 * (win.y0 + win.h))
        xs = slice(2 * win.x0, 2 * (win.x0 + win.w))
        lat_c = lat[0, :, ys, xs]
        with torch.no_grad():
            tgt = crop_margin(
                vaes["a"].decode_to_pixels(lat_c[None])[0].float(), win.mar, PATCH
            )
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        for sigma in sigmas:
            draws = []
            for k in range(args.draws):
                g = torch.Generator(device=device).manual_seed(
                    args.seed * 100003 + i * 16 + k
                )
                noise = torch.randn(
                    lat.shape, generator=g, device=device, dtype=lat.dtype
                )
                x_t = (1 - sigma) * lat + sigma * noise
                sig = torch.tensor([sigma], device=device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    v = anima(
                        x_t.unsqueeze(2).to(torch.bfloat16),
                        sig,
                        cross,
                        padding_mask=pad,
                    )
                f_w = win.tokens(tap.pop("x")[0, 0].float())
                lat_p = win.tokens(lat_patches(x_t).float())
                xt_c = x_t[0, :, ys, xs]
                v_tok = (f_w.double() @ W.T).float()
                v_dit = v.float()[0, :, 0, ys, xs]
                v_rel = ((unpatch(v_tok) - v_dit).norm() / v_dit.norm()).item()

                rec = {
                    "image": ci.stem,
                    "sigma": sigma,
                    "draw": k,
                    "v_recon_rel": v_rel,
                }
                g_v = {}
                for name, vae in vaes.items():
                    (gv, loss, img), dt, mem = timed(
                        lambda: vae_grad(vae, v_tok, xt_c, sigma, tgt, win.mar)
                    )
                    g_v[name] = gv
                    rec[f"{name}.lpips"] = loss
                    rec[f"{name}.sec"] = dt
                    rec[f"{name}.peak_gb"] = mem
                    if name == "a":
                        img_a = img
                        rec["a.psnr_vs_tgt"] = psnr(
                            crop_margin(img, win.mar, PATCH), tgt
                        )
                # the FM gradient on the same tokens: v − (ε − x0)
                leaf = v_tok.detach().clone().requires_grad_(True)
                fm = 0.5 * ((unpatch(leaf) - (noise[0, :, ys, xs] - lat_c)) ** 2).sum()
                g_fm = torch.autograd.grad(fm, leaf)[0].double()
                a_f = g_v["a"] @ W
                rec["a16.cos_v"] = cos(g_v["a16"], g_v["a"])
                rec["a.cos_fm"] = cos(g_v["a"], g_fm)
                rec["a.norm_over_fm"] = (g_v["a"].norm() / g_fm.norm()).item()
                imgs = {}
                for name, head in heads.items():
                    (gf, loss, img), dt, mem = timed(
                        lambda: head_grad(head, f_w, lat_p, tgt, win.mar)
                    )
                    gb = gf @ W.T @ gram_inv  # (h, w, 64): the gradient v receives
                    pg = gb @ W  # the row-space part of gf
                    g_v[name] = gb
                    imgs[name] = img
                    inner_h = crop_margin(img, win.mar, PATCH)
                    inner_a = crop_margin(img_a, win.mar, PATCH)
                    rec.update(
                        {
                            f"{name}.cos_v": cos(gb, g_v["a"]),
                            f"{name}.cos_f": cos(pg, a_f),
                            f"{name}.cos_f_raw": cos(gf, a_f),
                            f"{name}.cos_tok_med": cos_tok_median(gb, g_v["a"]),
                            f"{name}.row_share": ((pg.norm() / gf.norm()) ** 2).item(),
                            f"{name}.norm_ratio": (gb.norm() / g_v["a"].norm()).item(),
                            f"{name}.cos_fm": cos(gb, g_fm),
                            f"{name}.lpips": loss,
                            f"{name}.psnr_vs_vae": psnr(inner_h, inner_a),
                            f"{name}.sec": dt,
                            f"{name}.peak_gb": mem,
                        }
                    )
                draws.append((rec, g_v, v_tok, xt_c))
                if k == 0 and sigma == sigmas[min(1, len(sigmas) - 1)]:
                    if n_sheets < args.n_sheets:
                        cols = [("VAE(x̂0)", img_a), ("|g| VAE", norm_map(g_v["a"]))]
                        for name in heads:
                            cols += [
                                (name, imgs[name]),
                                (f"|g| {name}", norm_map(g_v[name])),
                            ]
                        pair_sheet(cols, run_dir / f"grad_{i:02d}_s{sigma:.4g}.png")
                        n_sheets += 1
            # the floor: a head's gradient against the VAE's on another noise draw
            for k, (rec, g_v, v_tok, xt_c) in enumerate(draws):
                other = draws[(k + 1) % len(draws)][1]
                rec["a.cos_v_cross"] = cos(g_v["a"], other["a"])
                for name in heads:
                    rec[f"{name}.cos_v_null"] = cos(g_v[name], other["a"])
                # each gradient checked by a step, read through the fp32 VAE
                na = g_v["a"].norm()
                loss0 = vae_loss(v_tok, xt_c, sigma, tgt, win.mar)
                for frac in fracs:
                    eta = frac * loss0 / na

                    def drop(gv):
                        step = (v_tok.double() - eta * gv / gv.norm()).float()
                        return loss0 - vae_loss(step, xt_c, sigma, tgt, win.mar)

                    da = drop(g_v["a"])
                    rec[f"a.fd{frac:g}_ratio"] = da / (eta * na).item()
                    for name, gv in g_v.items():
                        if name != "a":
                            rec[f"{name}.fd{frac:g}_share"] = drop(gv) / da
                    for name in heads:
                        rec[f"{name}.fd{frac:g}_null"] = drop(other[name]) / da
                rows.append(rec)
        r = rows[-1]
        print(
            f"[{i + 1}/{len(items)}] {ci.stem} window {win.h}x{win.w}  "
            + "  ".join(f"{n} cos_v {r[f'{n}.cos_v']:.3f}" for n in ("a16", *heads)),
            flush=True,
        )

    # --- summary: mean over images and draws, per σ ---
    keys = [k for k in rows[0] if k not in ("image", "sigma", "draw")]
    summary: dict[str, dict[str, float]] = {}
    for sigma in sigmas:
        sel = [r for r in rows if r["sigma"] == sigma]
        summary[f"s{sigma:.4g}"] = {
            k: (
                sorted(r[k] for r in sel)[
                    len(sel) // 2
                ]  # median: the first call warms up
                if k.endswith(".sec")
                else sum(r[k] for r in sel) / len(sel)
            )
            for k in keys
        }
    print(f"\nmean over {len(rows) // len(sigmas)} (image, draw) rows per sigma")
    fd = f"fd{fracs[len(fracs) // 2]:g}"
    print(
        f"sigma   a.lpips a.{fd}_ratio a16.cos_v a16.{fd}  a.cross  a.cos_fm | "
        f"head: cos_v {fd}_share {fd}_null cos_f cos_null raw_f row_share norm psnr"
    )
    for sk, s in summary.items():
        line = (
            f"{sk:7s} {s['a.lpips']:.4f} {s[f'a.{fd}_ratio']:10.3f} "
            f"{s['a16.cos_v']:9.3f} {s[f'a16.{fd}_share']:6.3f} "
            f"{s['a.cos_v_cross']:8.3f} {s['a.cos_fm']:9.3f} |"
        )
        for n in heads:
            line += (
                f" {n}: {s[f'{n}.cos_v']:.3f} {s[f'{n}.{fd}_share']:.3f} "
                f"{s[f'{n}.{fd}_null']:.3f} {s[f'{n}.cos_f']:.3f} "
                f"{s[f'{n}.cos_v_null']:.3f} {s[f'{n}.cos_f_raw']:.3f} "
                f"{s[f'{n}.row_share']:.4f} {s[f'{n}.norm_ratio']:.2f} "
                f"{s[f'{n}.psnr_vs_vae']:.2f} |"
            )
        print(line)
    s0 = summary[next(iter(summary))]
    print(
        "backward sec / peak GB: "
        + "  ".join(
            f"{n} {s0[f'{n}.sec']:.3f} / {s0[f'{n}.peak_gb']:.2f}"
            for n in (*vaes, *heads)
        ),
        flush=True,
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"mem_base_gb": mem_base, "summary": summary, "rows": rows},
        label=args.label,
        device=device,
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
