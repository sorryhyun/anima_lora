"""Phase 0 of docs/moss/proposal/tone_xt_trust.md: how much of x_t's tone x̂₀ keeps.

Per sample and σ, one batched forward over x_t and copies of it with the whole page
shifted along a tone direction, x_t + (1−σ)·c·d (d from
``library/moss/calibration/tone_dirs.safetensors``, per +1 Lab unit; c in Lab units).
The page tone the state implies, (tone of x_t)/(1−σ), moves by exactly c. The shift
of x̂₀'s mean over the rest tokens, fitted on the L / a / b directions (least squares,
Lab units), divided by c is the trust k:

- k = 1: x̂₀ carries x_t's tone whole (no pull back to the reference);
- k = 0: x̂₀ ignores it and keeps the reference's;
- k = 1−σ: v̂ does not respond to the shift at all (x̂₀ = x_t − σ·v̂ moves with
  x_t alone), printed as ``1−σ`` beside the measurement.

The plain row also gives δ(σ), the single-step tone error on a correct x_t, from the
same fit. Same samples and noise as ``tone_probe.py`` (24 identity-tree pairs + the five
held-out pages); page rest = pixels the v3 render at seed 0 left alone. Runs on
the render base (int8 GPTQ forward copies, fp32 stream).

    make daemon-run ARGS="--stall-timeout 900 project/finished/moss_lora/src/tone_trust_probe.py"
    .venv/bin/python project/finished/moss_lora/src/tone_trust_probe.py --analyze
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from scipy import ndimage
from tone_probe import PAGES, pick_pairs

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_transformer, load_vae
from library.moss.model import encode
from library.moss.train import NUM_TRAIN_TIMESTEPS, edit_region

CKPT = "output/ckpt_moss_i2i"
V3 = "moss_i2i_sanitize_idt_w0003_late_lr5e5_e8"
LORAS = {
    "base": None,
    "v3": f"{CKPT}/{V3}.safetensors",
    "v3_e4": f"{CKPT}/{V3}/{V3}_e4.safetensors",
    "v3_raw": f"{CKPT}/{V3}/{V3}_raw.safetensors",
}
MASK_RENDERS = ["output/tests/moss_sanitize_probe/000_{page}_base_s0.png"]
OUT = "project/finished/moss_lora/space/tone_trust_probe.json"


def page_mask(page: str, size: tuple[int, int], src_dir: str) -> np.ndarray:
    from bench.moss_i2i_eval.tone_map import delta_e, lab

    src = moss_cache.fit(
        moss_cache.load_rgb(resolve_under_home(f"{src_dir}/{page}.png")), *size
    )
    src_lab, _ = lab(src)
    changed = np.zeros(src_lab.shape[:2], bool)
    for pattern in MASK_RENDERS:
        out_lab, _ = lab(Image.open(resolve_under_home(pattern.format(page=page))))
        changed |= delta_e(out_lab, src_lab) > 8
    return ~ndimage.binary_dilation(changed, iterations=20)


def lab_fit(dc: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``dc (..., 128)`` → Lab coefficients ``(..., 3)`` on ``D (128, 3)``."""
    pinv = np.linalg.pinv(D)
    return dc @ pinv.T


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="post_image_dataset/moss/i2i_sanitize_idt")
    p.add_argument("--pages", default="output/moss_i2i_eval/sanitize/inputs")
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--sigmas", default="0.1,0.3,0.45,0.6,0.8")
    p.add_argument("--shifts", default="L:-2,L:1,L:2,b:-2,b:1,b:2")
    p.add_argument("--loras", default=",".join(LORAS))
    p.add_argument("--chunk", type=int, default=2, help="forwards per batched call")
    p.add_argument("--tone_dilate", type=int, default=2)
    p.add_argument("--out", default=OUT)
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()
    if args.analyze:
        summarize(json.loads(resolve_under_home(args.out).read_text()))
        return
    device = torch.device("cuda")
    sigmas = [float(s) for s in args.sigmas.split(",")]
    shifts = [(t, float(c)) for t, c in (s.split(":") for s in args.shifts.split(","))]
    dirs = load_file(
        resolve_under_home("library/moss/calibration/tone_dirs.safetensors")
    )
    D = torch.stack([dirs[t].float() for t in ("L", "a", "b")], 1).to(
        device
    )  # (128, 3)
    to_lab = torch.linalg.pinv(D).T  # (128, 3): channel-mean shift → Lab units

    items = pick_pairs(resolve_under_home(args.cache), args.pairs)
    samples = []
    for it in items:
        x0, emb = moss_cache.load_item(it)
        ref = moss_cache.load_ref(it)
        region = edit_region(x0, ref, 0.3, 5, 1 + args.tone_dilate)
        samples.append(
            dict(name=it.stem, x0=x0[None], ref=ref[None], emb=emb[None], rest=~region)
        )
    fixed_emb = samples[0]["emb"]
    vae = load_vae(None, device=device)
    for page in PAGES:
        src = moss_cache.load_rgb(resolve_under_home(f"{args.pages}/{page}.png"))
        w, h = moss_cache.target_size(*src.size, 1024)
        ref = encode(moss_cache.to_pixels(moss_cache.fit(src, w, h)), vae).float().cpu()
        px = page_mask(page, (w, h), args.pages)
        tok = px.reshape(h // 16, 16, w // 16, 16).all((1, 3))
        samples.append(
            dict(
                name=page,
                x0=ref,
                ref=ref,
                emb=fixed_emb,
                rest=torch.from_numpy(tok),
                page=True,
            )
        )
        print(f"[page] {page} {w}x{h} rest {tok.mean():.0%}", flush=True)
    del vae
    empty_cache()
    print(
        f"{len(samples)} samples, rest share median "
        f"{np.median([s['rest'].float().mean().item() for s in samples]):.0%}",
        flush=True,
    )

    noise = {}
    for i, s in enumerate(samples):
        g = torch.Generator().manual_seed(1000 + i)
        noise[i] = torch.randn(s["x0"].shape, generator=g)

    # The render base: int8 forward copies (GPTQ file) + fp32 stream.
    from library.moss.dit_int8.base import generate_base
    from library.moss.loader import stream_fp32_
    from library.moss.lora import load_for_inference
    from library.moss.requests import GenerateRequest

    transformer = load_transformer(None, "base", device=device)
    generate_base(GenerateRequest())(transformer)
    stream_fp32_(transformer)

    rows = []
    for name in args.loras.split(","):
        net = None
        transformer.ref_timestep, transformer.ref_isolate = None, False
        if LORAS[name]:
            net = load_for_inference(transformer, resolve_under_home(LORAS[name]))
            net.to(device)
        else:
            transformer.ref_timestep, transformer.ref_isolate = 0.0, True
        t0 = time.time()
        for i, s in enumerate(samples):
            x0, eps = s["x0"].to(device), noise[i].to(device)
            rest = s["rest"].to(device)
            n_rest = rest.sum().clamp(min=1)
            for sigma in sigmas:
                x_t = (1 - sigma) * x0 + sigma * eps
                variants = [x_t] + [
                    x_t + (1 - sigma) * c * dirs[t].to(device)[None, :, None, None]
                    for t, c in shifts
                ]
                x_hats = []
                for j in range(0, len(variants), args.chunk):
                    xb = torch.cat(variants[j : j + args.chunk])
                    b = xb.shape[0]
                    v = transformer.forward_batched(
                        xb,
                        torch.full((b,), NUM_TRAIN_TIMESTEPS * sigma, device=device),
                        s["emb"].to(device).expand(b, -1, -1),
                        ref=s["ref"].to(device).expand(b, -1, -1, -1),
                    ).float()
                    x_hats.append(xb - sigma * v)
                x_hat = torch.cat(x_hats)
                # Mean over the rest tokens, per channel: (V, 128).
                dc_rest = (x_hat * rest).sum((2, 3)) / n_rest
                dc_edit = (
                    (x_hat * ~rest).sum((2, 3)) / (~rest).sum()
                    if (~rest).any()
                    else None
                )
                err_plain = dc_rest[0] - (x0 * rest).sum((2, 3))[0] / n_rest
                d_rest = dc_rest[1:] - dc_rest[:1]
                fit = d_rest @ to_lab
                rows.append(
                    dict(
                        lora=name,
                        sample=s["name"],
                        page=bool(s.get("page")),
                        sigma=sigma,
                        delta=(err_plain @ to_lab).cpu().tolist(),
                        shift_rest=fit.cpu().tolist(),
                        shift_edit=None
                        if dc_edit is None
                        else ((dc_edit[1:] - dc_edit[:1]) @ to_lab).cpu().tolist(),
                        resid=float(
                            (
                                (d_rest - fit @ D.T).norm(dim=1)
                                / d_rest.norm(dim=1).clamp(min=1e-9)
                            ).mean()
                        ),
                    )
                )
        print(
            f"[{name}] {len(samples)} samples x {len(sigmas)} σ x {1 + len(shifts)} "
            f"in {time.time() - t0:.0f}s",
            flush=True,
        )
        if net is not None:
            net.restore()
            del net
    out = resolve_under_home(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dict(args=vars(args), shifts=shifts, rows=rows)
    out.write_text(json.dumps(data))
    summarize(data)
    print(f"-> {out}", flush=True)


def summarize(data: dict) -> None:
    shifts = [tuple(s) for s in data["shifts"]]
    idx = {"L": 0, "a": 1, "b": 2}
    groups = collections.defaultdict(list)
    for r in data["rows"]:
        groups[(r["page"], r["lora"], r["sigma"])].append(r)
    for page in (False, True):
        print(
            f"\n=== {'held-out pages' if page else 'training pairs'}: k on the rest tokens ==="
        )
        head = " ".join(f"{t}{c:+g}".rjust(6) for t, c in shifts)
        print(
            f"{'lora':>6} {'σ':>4} {'1−σ':>4} | {head} | {'cross':>5} {'resid':>5} | δL     δb   | k_edit L/b"
        )
        for (pg, lora, sigma), rs in sorted(groups.items()):
            if pg != page:
                continue
            sh = np.array([r["shift_rest"] for r in rs])  # (n, V, 3)
            ks = [sh[:, v, idx[t]].mean() / c for v, (t, c) in enumerate(shifts)]
            # Cross-talk: the other tone's response to an L / b shift, per unit c.
            cross = np.mean(
                [
                    abs(sh[:, v, 2 - idx[t]].mean() / c)
                    for v, (t, c) in enumerate(shifts)
                ]
            )
            resid = np.mean([r["resid"] for r in rs])
            dl = np.array([r["delta"] for r in rs]).mean(0)
            ke = [r["shift_edit"] for r in rs if r["shift_edit"] is not None]
            if ke:
                ke = np.array(ke)
                kel = np.mean(
                    [
                        ke[:, v, 0].mean() / c
                        for v, (t, c) in enumerate(shifts)
                        if t == "L"
                    ]
                )
                keb = np.mean(
                    [
                        ke[:, v, 2].mean() / c
                        for v, (t, c) in enumerate(shifts)
                        if t == "b"
                    ]
                )
                ked = f"{kel:.2f} / {keb:.2f}"
            else:
                ked = "-"
            print(
                f"{lora:>6} {sigma:>4} {1 - sigma:4.2f} | {' '.join(f'{k:6.2f}' for k in ks)} | "
                f"{cross:5.2f} {resid:5.2f} | {dl[0]:+.2f} {dl[2]:+.2f} | {ked}"
            )


if __name__ == "__main__":
    main()
