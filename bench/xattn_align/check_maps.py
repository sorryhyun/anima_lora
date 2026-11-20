#!/usr/bin/env python
"""Precondition 1 of the xattn_align A/B (docs/anima/proposal/xattn_align_loss.md § 5):
the training adapter's maps (``block_cross_probs``, rebuilt in a block pre-hook)
equal the probe's (``bench/xattn_loc/probe.py``'s ``Reader``, read inside the
cross-attn call) on the same item and σ.

Both readers sit on one eager forward of the base DiT (no LoRA) over the
caststation pages with a matched string; per (page, σ, block) the max abs
difference of the head-mean maps at the string columns and each side's own-box
lift are reported.

    make daemon-run ARGS="--queue --stall-timeout 300 bench/xattn_align/check_maps.py"
"""

from __future__ import annotations

import argparse
import json

import numpy as np

BLOCKS = (8, 13, 14, 15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigmas", default="0.9,0.6")
    p.add_argument("--pages", type=int, default=6)
    a = p.parse_args()

    import torch
    from safetensors import safe_open

    from anima_lora import GenerationRequest
    from bench._anima import DEFAULT_DIT
    from bench._common import make_run_dir, write_result
    from bench.xattn_loc.probe import Reader, discover, load_encoder
    from library.anima.stream import latent_dtype
    from library.inference.models import load_dit_model
    from library.training.xattn_align import block_cross_probs, region

    encoder, _ = load_encoder()
    items, _ = discover(encoder)
    items = [m for m in items if m["set"] == "channel_(caststation)"][: a.pages]
    dev = torch.device("cuda")
    g = GenerationRequest(prompt="x", dit=DEFAULT_DIT, device="cuda", seed=0).to_args()
    g.compile_blocks = False
    anima = load_dit_model(g, dev, torch.bfloat16)
    anima.eval()
    rd = Reader(anima)
    mine: dict = {}
    cur = {}

    def hook_for(b):
        def hook(block, args, kwargs):
            if cur.get("cols") is None:
                return
            x, emb, ctx_emb, ap = args[:4]
            with torch.no_grad():
                mine[b] = block_cross_probs(
                    block, x, emb, ctx_emb, ap, kwargs.get("rope_cos_sin"),
                    kwargs.get("adaln_lora_B_T_3D"), cur["cols"],
                )[0]  # fmt: skip

        return hook

    for b in BLOCKS:
        anima.blocks[b].register_forward_pre_hook(hook_for(b), with_kwargs=True)

    rows = []
    for n, m in enumerate(items):
        z = np.load(m["npz"])
        x0 = torch.from_numpy(
            z[next(k for k in z.files if k.startswith("latents_"))]
        ).float()
        h, w = x0.shape[-2:]
        gh, gw = h // 2, w // 2
        with safe_open(m["te"], "pt") as f:
            ctx = f.get_tensor("crossattn_emb_v0")
        cols = sorted({c for s in m["strings"] for c in s["cols"]})
        at = {c: i for i, c in enumerate(cols)}
        rd.cols = torch.tensor(cols, device=dev)
        cur["cols"] = rd.cols
        eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(n))
        for sg in (float(s) for s in a.sigmas.split(",")):
            xt = ((1 - sg) * x0 + sg * eps)[None, :, None].to(dev, latent_dtype(anima))
            t = torch.tensor([sg], device=dev, dtype=xt.dtype)
            pm = torch.zeros(1, 1, h, w, dtype=torch.bfloat16, device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                anima(xt, t, ctx[None].to(dev, torch.bfloat16), padding_mask=pm)
            for b in BLOCKS:
                P, Q = rd.maps[b].float(), mine[b].float()  # (Nq, n)
                for i, s in enumerate(m["strings"]):
                    r = region(s["box"], gh, gw).to(dev)
                    ci = [at[c] for c in s["cols"]]
                    pp, qq = P[:, ci].sum(-1), Q[:, ci].sum(-1)

                    def lift(v):
                        return float(v[r].sum() / v.sum() / r.float().mean())

                    rows.append(
                        {"stem": m["stem"], "str": i, "sigma": sg, "block": b,
                         "maxdiff": float((P[:, ci] - Q[:, ci]).abs().max()),
                         "maxval": float(P[:, ci].abs().max()),
                         "lift_probe": lift(pp), "lift_adapter": lift(qq)}
                    )  # fmt: skip
    out = make_run_dir("xattn_align", "check-maps")
    d = np.array([[r["maxdiff"] / max(r["maxval"], 1e-20)] for r in rows])
    lp = np.array([r["lift_probe"] for r in rows])
    la = np.array([r["lift_adapter"] for r in rows])
    metrics = {
        "rows": len(rows),
        "rel_maxdiff_max": float(d.max()),
        "rel_maxdiff_mean": float(d.mean()),
        "lift_absdiff_max": float(np.abs(lp - la).max()),
        "lift_probe_mean": float(lp.mean()),
        "lift_adapter_mean": float(la.mean()),
    }
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    write_result(out, script=__file__, args=a, metrics=metrics, label="check-maps")
    print(json.dumps(metrics, indent=1), flush=True)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
