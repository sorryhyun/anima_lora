#!/usr/bin/env python3
"""Gate for t2i self-distillation (``docs/moss/proposal/self_distill_lora.md``).

On a training cache, Base, no LoRA, cond branch: each latent noised at a σ grid
(``--eps`` draws per σ), the one-step velocity under four conditionings:

- ``cap``: the cached caption embeds (what the student trains on);
- ``img_cap``: the image through the vision tower, then the caption, tail off;
- ``img_cap_sure``: the same with the ``sure`` tail;
- ``img``: the image alone, tail off.

Per (image, σ, ε, conditioning): the flow loss ``‖v − (ε − x₀)‖²`` (= ``‖x̂₀ − x₀‖² / σ²``),
the distance to ``cap``'s velocity, and the x̂₀ spread across ε draws. The σ band where
``cap − img_cap`` is positive is where the image-conditioned teacher knows more.

Then ``--renders`` full renders (50 steps, CFG 3) per conditioning at the image's size,
and decoded x̂₀ sheets for the first ``--decode`` images.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_self_distill/xhat_probe.py \
        --cache 'post_image_dataset/moss/channel_(caststation)_layout_sure'"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from bench._common import make_run_dir, write_result
from bench.moss_ab._render import grid, render
from library.env import resolve_under_home
from library.moss.caching.images import fit, load_rgb
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    load_tokenizer,
    load_transformer,
    load_vae,
    pad_embeds,
    stream_fp32_,
)
from library.moss.model import decode, snapped_sigmas
from library.moss.prompt_tail import TAILS
from library.moss.requests import GenerateRequest
from library.moss.training.data import NUM_TRAIN_TIMESTEPS, samples_from_cache

CONDS = ("cap", "img_cap", "img_cap_sure", "img")


def to_png(z, vae, device):
    from PIL import Image

    x = decode(z.to(device), vae)[0].float().add(1).mul(127.5).round().clamp(0, 255)
    return Image.fromarray(x.permute(1, 2, 0).byte().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sigmas", default="0.1,0.25,0.4,0.6,0.8,0.95")
    ap.add_argument("--eps", type=int, default=2)
    ap.add_argument("--renders", type=int, default=3)
    ap.add_argument("--decode", type=int, default=3)
    ap.add_argument("--decode_sigmas", default="0.4,0.6,0.8,0.95")
    ap.add_argument("--te", default="int8")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    dec_sigmas = [float(s) for s in args.decode_sigmas.split(",")]
    device = torch.device("cuda")
    run_dir = make_run_dir("moss_self_distill", label=args.label or "xhat")

    samples = samples_from_cache(resolve_under_home(args.cache))
    samples.sort(key=lambda s: s.stem)
    metas = []
    for s in samples:
        with safe_open(str(s.te_path), "pt") as f:
            metas.append(f.metadata())
    print(f"{len(samples)} samples", flush=True)

    # --- TE: every conditioning, then free it ---
    from library.moss import vision_te

    tok = load_tokenizer("")
    te = vision_te.load_te(tok, device, precision=args.te)
    embeds = []
    sanity = []
    for s, meta in zip(samples, metas):
        h, w = (n * 16 for n in s.latent_hw)
        page = fit(load_rgb(Path(meta["source"])), w, h)
        cap = s.load()[1]
        e = {"cap": cap}
        e["img_cap"], n_img = vision_te.encode(
            tok, te, page, meta["caption"], w * h, device
        )
        e["img_cap_sure"], _ = vision_te.encode(
            tok, te, page, meta["caption"], w * h, device, tail=TAILS["sure"]
        )
        e["img"], _ = vision_te.encode(tok, te, page, "", w * h, device)
        txt, _ = vision_te.encode(
            tok, te, None, meta["caption"], w * h, device, tail=TAILS["sure"]
        )
        if len(sanity) < 3:
            same = txt.shape == cap.shape
            diff = (txt.float().cpu() - cap.float()).abs().max().item() if same else -1
            sanity.append({"stem": s.stem, "same_len": same, "max_abs": diff})
        embeds.append({k: v.to("cpu", torch.bfloat16) for k, v in e.items()})
        print(
            f"[te] {s.stem} {w}x{h}: cap {cap.shape[0]}, img_cap "
            f"{e['img_cap'].shape[0]} ({n_img} image), img {e['img'].shape[0]}",
            flush=True,
        )
    empty = {
        t: vision_te.encode(tok, te, None, "", 0, device, tail=TAILS[t])[0].to(
            "cpu", torch.bfloat16
        )
        for t in ("off", "sure")
    }
    print(f"[te] text-only re-encode vs cache: {sanity}", flush=True)
    del te
    empty_cache()

    # --- DiT: int8 Base, fp32 stream, as the training plan ---
    from library.moss.dit_int8.base import generate_base

    convert = generate_base(GenerateRequest(int8="on"))
    tr = load_transformer("", "base", device=device)
    convert(tr)
    stream_fp32_(tr)
    compile_blocks(tr.layers)

    gen = torch.Generator().manual_seed(args.seed)
    rows = []  # one per (image, σ, ε, cond)
    keep = {}  # (image idx, σ, cond) -> x̂₀ of draw 0
    with torch.inference_mode():
        for i, (s, e) in enumerate(zip(samples, embeds)):
            x0 = s.load()[0].unsqueeze(0).to(device, torch.float32)
            conds = {k: v.unsqueeze(0).to(device) for k, v in e.items()}
            for sigma in sigmas:
                t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
                xh = defaultdict(list)
                for k in range(args.eps):
                    eps = torch.randn(x0.shape, generator=gen).to(device)
                    xt = (1 - sigma) * x0 + sigma * eps
                    target = eps - x0
                    vs = {}
                    for c in CONDS:
                        v = tr(xt, t, conds[c], return_dict=False)[0].float()
                        vs[c] = v
                        xh[c].append(xt - sigma * v)
                        if k == 0 and i < args.decode and sigma in dec_sigmas:
                            keep[(i, sigma, c)] = xh[c][-1].cpu()
                    for c in CONDS:
                        rows.append(
                            dict(
                                stem=s.stem,
                                sigma=sigma,
                                eps=k,
                                cond=c,
                                fm=F.mse_loss(vs[c], target).item(),
                                to_cap=F.mse_loss(vs[c], vs["cap"]).item(),
                                to_img_cap=F.mse_loss(vs[c], vs["img_cap"]).item(),
                            )
                        )
                if args.eps > 1:
                    for c in CONDS:
                        spread = torch.stack(xh[c]).var(0, unbiased=True).mean().item()
                        rows.append(
                            dict(stem=s.stem, sigma=sigma, cond=c, xhat_var=spread)
                        )
            print(f"[x̂₀] {i + 1}/{len(samples)} {s.stem}", flush=True)

        # --- full renders: caption only vs image + caption ---
        steps = snapped_sigmas(50, GenerateRequest().shift)
        unconds = {t: pad_embeds([v], device) for t, v in empty.items()}
        renders = {}
        for i, (s, e) in enumerate(zip(samples[: args.renders], embeds)):
            h, w = (n * 16 for n in s.latent_hw)
            for c, tail in (
                ("cap", "sure"),
                ("img_cap", "off"),
                ("img_cap_sure", "sure"),
            ):
                renders[(i, c)] = render(
                    tr, e[c], *unconds[tail], steps, seed=1, height=h, width=w, cfg=3.0
                ).cpu()
                print(f"[render] {s.stem} {c}", flush=True)
    del tr
    empty_cache()

    # --- summary ---
    agg = defaultdict(list)
    for r in rows:
        for m in ("fm", "to_cap", "to_img_cap", "xhat_var"):
            if m in r:
                agg[(r["cond"], r["sigma"], m)].append(r[m])
    table = {
        c: {
            str(sg): {
                m: sum(agg[(c, sg, m)]) / len(agg[(c, sg, m)])
                for m in ("fm", "to_cap", "to_img_cap", "xhat_var")
                if agg[(c, sg, m)]
            }
            for sg in sigmas
        }
        for c in CONDS
    }
    # paired gap per image: fm(cap) − fm(c), mean over ε
    per = defaultdict(list)
    for r in rows:
        if "fm" in r:
            per[(r["stem"], r["sigma"], r["cond"])].append(r["fm"])
    gaps = {}
    for c in CONDS[1:]:
        gaps[c] = {}
        for sg in sigmas:
            g = [
                sum(per[(s.stem, sg, "cap")]) / args.eps
                - sum(per[(s.stem, sg, c)]) / args.eps
                for s in samples
            ]
            gaps[c][str(sg)] = {
                "mean": sum(g) / len(g),
                "positive": sum(x > 0 for x in g) / len(g),
            }
    print("\nσ      " + "  ".join(f"{c:>13}" for c in CONDS), flush=True)
    for sg in sigmas:
        print(
            f"{sg:<6} " + "  ".join(f"{table[c][str(sg)]['fm']:13.4f}" for c in CONDS),
            flush=True,
        )
    print("\ngap fm(cap) − fm(c), mean [share of images > 0]", flush=True)
    for sg in sigmas:
        print(
            f"{sg:<6} "
            + "  ".join(
                f"{c}: {gaps[c][str(sg)]['mean']:+.4f} [{gaps[c][str(sg)]['positive']:.0%}]"
                for c in CONDS[1:]
            ),
            flush=True,
        )
    (run_dir / "rows.json").write_text(json.dumps(rows))

    # --- decode ---
    vae = load_vae("", device=device)
    artifacts = ["rows.json"]
    with torch.inference_mode():
        for i in range(min(args.decode, len(samples))):
            s = samples[i]
            src = load_rgb(Path(metas[i]["source"]))
            sheet = [
                (
                    f"σ {sg}",
                    [("x0", src)]
                    + [(c, to_png(keep[(i, sg, c)], vae, device)) for c in CONDS],
                )
                for sg in dec_sigmas
            ]
            name = f"xhat_{s.stem}.png"
            grid(sheet, cell_h=384).save(run_dir / name)
            artifacts.append(name)
        for i in range(min(args.renders, len(samples))):
            s = samples[i]
            src = load_rgb(Path(metas[i]["source"]))
            cells = [("source", src)] + [
                (c, to_png(renders[(i, c)], vae, device))
                for c in ("cap", "img_cap", "img_cap_sure")
            ]
            name = f"render_{s.stem}.png"
            grid([(s.stem, cells)], cell_h=640).save(run_dir / name)
            artifacts.append(name)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"fm_by_sigma": table, "gap_vs_cap": gaps, "te_sanity": sanity},
        artifacts=artifacts,
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
