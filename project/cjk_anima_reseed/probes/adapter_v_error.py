"""adapter_v_error — the prediction error the bf16 ``llm_adapter`` leaves on the fp32 stream.

Forward only. ``hira_cold_hp``'s own items, its rows at ``s4050``, the adapter run in the
forward from the cached Qwen states and T5 ids (the Anima LoRA TE cache holds the adapter's
bf16 *output*, so ``bench/anima_stream`` never ran it). Per σ, ``‖v − v_ref‖ / ‖v_ref‖``
in % over the whole latent and inside the glyph boxes (``loss.box_mask``, grid boxes
on), and the adapter output's distance from ref on the valid T5 tokens:

- ``ref``: the whole model in fp32, SDPA, no TF32, no autocast;
- ``bf16``: bf16 stream, bf16 adapter (the reseed trainer of record);
- ``stream``: ``stream_fp32_``, bf16 adapter (generation's default today);
- ``stream_adapter``: ``stream`` + ``library.anima.stream.adapter_fp32_`` (bf16 weights).

    make daemon-run ARGS="--stall-timeout 600 project/cjk_anima_reseed/probes/adapter_v_error.py"

→ ``results/<ts>-adapter_v_error[-label]/result.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, bootstrap  # noqa: E402

bootstrap()

import torch  # noqa: E402

ARMS = ("bf16", "stream", "stream_adapter")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="hira_cold_hp")
    ap.add_argument("--rows", default="s4050")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--batches", default="scene,scene,grid,grid")
    ap.add_argument("--seeds", type=int, default=2, help="ε draws a batch and σ")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    from bench._common import start_heartbeat
    from common.models import checkpoints, gen_args
    from library.anima.stream import adapter_fp32_, stream_fp32_
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from reseed import trainer as T
    from reseed.config import load
    from reseed.loss import box_mask
    from reseed.rows import Rows
    from train.stage import Batcher, LatentStore, _encode_text

    start_heartbeat(label="adapter_v_error")
    out = (
        HOME
        / "results"
        / (
            time.strftime("%Y%m%d-%H%M")
            + "-adapter_v_error"
            + (f"-{a.label}" if a.label else "")
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]

    run = load(a.run)
    run.use_pack()
    data = run.data
    recs, ev, vocabs = T.load_items(data)
    build = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if build.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    gargs = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    gargs.stream = "bf16"
    gargs.compile_blocks = False
    device = get_generation_settings(gargs).device
    cache, touched, _ = _encode_text(recs, ev, device, out, te_cache=data / "te_cache")
    ctx = run.seed_rows()
    p = T.plan(run, data, recs, vocabs, touched, ctx, run.steps_per_row, None)
    ns = SimpleNamespace(seed=T.SEED, batch=T.BATCH, train_size=512)
    lat = LatentStore(ns, data, recs, list(range(len(recs))), device)
    anima = load_dit_model(gargs, device, torch.bfloat16)
    anima.requires_grad_(False)
    anima.eval()
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    rows_pt = run.dir / a.rows / "trained.pt" if a.rows else run.dir / "trained.pt"
    Rows(
        anima,
        device,
        p.idx,
        pack,
        warm=rows_pt,
        init_anchor=0.0,
        free_residual=0.0,
        lr=T.LR,
        touched=p.touched,
        frozen=p.frozen,
        context=ctx,
    )

    batcher = Batcher(ns, recs, lat)
    want = a.batches.split(",")
    batches, step = [], 0
    while len(batches) < len(want):
        step += 1
        assert step < 10_000
        idx = batcher.next(step)
        if recs[idx[0]]["src"] == want[len(batches)]:
            batches.append(idx)

    adapter_out = {}
    anima.llm_adapter.register_forward_hook(
        lambda m, i, o: adapter_out.__setitem__("x", o.detach().float())
    )

    @torch.no_grad()
    def predict(bi, si, seed, *, dtype, autocast):
        idx = batches[bi]
        caps = [recs[i]["caption"] for i in idx]

        def stacked(k):
            return torch.stack([cache[c][k] for c in caps]).to(device)

        latents = lat[idx].to(device).float()
        g = torch.Generator(device=device).manual_seed(10_000 * bi + 100 * si + seed)
        noise = torch.randn(latents.shape, generator=g, device=device)
        s = sigmas[si]
        x = ((1.0 - s) * latents + s * noise).to(dtype)
        t = torch.full((len(idx),), s, device=device, dtype=dtype)
        c = stacked(0)
        if not autocast:
            c = c.float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            v = (
                anima(
                    x.unsqueeze(2),
                    t,
                    c,
                    padding_mask=torch.zeros(
                        len(idx), 1, *x.shape[-2:], dtype=dtype, device=device
                    ),
                    target_input_ids=stacked(2),
                    target_attention_mask=stacked(3),
                    source_attention_mask=stacked(1),
                )
                .squeeze(2)
                .float()
            )
        return v, adapter_out["x"], stacked(3).bool()

    cells = [
        (bi, si, sd)
        for bi in range(len(batches))
        for si in range(len(sigmas))
        for sd in range(a.seeds)
    ]
    t0 = time.perf_counter()

    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtypes = {
        n: t.dtype for n, t in [*anima.named_parameters(), *anima.named_buffers()]
    }
    attn_mode = anima.attn_mode
    anima.float()
    anima.attn_mode = "torch"
    refs = {c: predict(*c, dtype=torch.float32, autocast=False) for c in cells}
    for n, t in [*anima.named_parameters(), *anima.named_buffers()]:
        t.data = t.data.to(dtypes[n])
    anima.attn_mode = attn_mode
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    print(f"ref: {len(cells)} forwards, {time.perf_counter() - t0:.0f}s", flush=True)

    rows_out: dict = {}
    for arm in ARMS:
        if arm == "stream":
            stream_fp32_(anima)
        if arm == "stream_adapter":
            adapter_fp32_(anima)
        dtype = torch.bfloat16 if arm == "bf16" else torch.float32
        for c in cells:
            bi, si, _ = c
            v, ao, valid = predict(*c, dtype=dtype, autocast=True)
            vr, aor, _ = refs[c]
            brecs = [recs[i] for i in batches[bi]]
            m = box_mask(vr.shape, brecs, device, grid_box=True).expand_as(vr).bool()
            e = v - vr
            r = rows_out.setdefault(
                (arm, sigmas[si], recs[batches[bi][0]]["src"]),
                {"all": [], "box": [], "adapter": []},
            )
            for k in range(vr.shape[0]):
                r["all"].append(float(e[k].norm() / vr[k].norm() * 100))
                if bool(m[k].any()):
                    r["box"].append(float(e[k][m[k]].norm() / vr[k][m[k]].norm() * 100))
                vm = valid[k]
                r["adapter"].append(
                    float((ao[k][vm] - aor[k][vm]).norm() / aor[k][vm].norm() * 100)
                )
        print(f"{arm}: {time.perf_counter() - t0:.0f}s", flush=True)

    res = {
        "run": a.run,
        "rows": str(rows_pt),
        "sigmas": sigmas,
        "seeds": a.seeds,
        "cells": {},
    }
    print(
        "\nRESULT % from the fp32 ref (mean over items × seeds): v whole | v in glyph boxes | adapter output"
    )
    for src in sorted({k[2] for k in rows_out}):
        for s in sigmas:
            line = []
            for arm in ARMS:
                r = rows_out[arm, s, src]
                mean = {k: statistics.fmean(v) if v else None for k, v in r.items()}
                res["cells"][f"{arm}|{s}|{src}"] = {**mean, "n": len(r["all"])}
                box = f"{mean['box']:.2f}" if mean["box"] is not None else "-"
                line.append(f"{arm} {mean['all']:.2f} | {box} | {mean['adapter']:.2f}")
            print(f"RESULT {src} σ {s}: " + "   ".join(line))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(
        f"→ {out / 'result.json'} ({(time.perf_counter() - t0) / 60:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
