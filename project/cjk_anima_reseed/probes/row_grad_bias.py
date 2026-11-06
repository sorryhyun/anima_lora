"""row_grad_bias — is the vocab rows' training gradient biased on the bf16 stream?

The reseed trainer's box-share FM loss on a run's own batches, its rows at a save
(default ``hira_cold_hp`` at ``s4050``), the gradient taken on ``rows.delta.raw``
(the run's touched rows). Arms, each on the same batches and the same ε per draw:

- ``bf16``: the trainer of record (bf16 noisy and σ, bf16 autocast, FA2, compiled
  blocks, bf16 stream);
- ``stream``: ``library.anima.stream.stream_fp32_`` (fp32 noisy and σ), compiled;
- ``stream_jit``: ``stream`` with relative noise 2^-9 on ``x`` at each block entry,
  the control: a rounding-sized perturbation with no bias;
- ``stream_adapter``: ``stream`` with ``llm_adapter`` in fp32 (weights cast, autocast
  off inside it); its output still rounds to bf16 at the cross-attn k / v GEMMs;
- ``ref`` (first ``--ref_draws`` only): the whole model in fp32, eager, SDPA, no TF32.

Over D paired draws at a fixed σ, ``e_d = g_arm − g_stream`` and

    bias² = ‖ē‖² − s² / D,   s² = Σ ‖e_d − ē‖² / (D − 1)

(``project/finished/int8_train/src/anima_grad_bias.py``'s estimator), unbiased for
‖E[e]‖² whatever the spread of e_d; in % of ‖mean g_stream‖. ``bias(bf16)`` is what a
bf16-trained row drifts by against an fp32-stream one in expectation. The ref draws give
each arm's single-draw distance from fp32. σ values are exact in bf16 (0.3125, 0.625).

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/row_grad_bias.py"

→ ``results/<ts>-row_grad_bias[-label]/result.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, bootstrap  # noqa: E402

bootstrap()

from library.runtime.allocator import default_expandable_segments  # noqa: E402

default_expandable_segments()

import torch  # noqa: E402

PAIRED = ("bf16", "stream_jit", "stream_adapter")
JIT = [None]


def forward(
    anima, noisy, ts, cache, caps, device, *, ctx_dtype=None, pad_dtype=torch.bfloat16
):
    """``common.models.dit_forward`` with the context and padding mask dtypes set."""

    def stacked(k):
        return torch.stack([cache[c][k] for c in caps]).to(device)

    ctx = stacked(0)
    if ctx_dtype is not None:
        ctx = ctx.to(ctx_dtype)
    b = len(caps)
    return anima(
        noisy.unsqueeze(2),
        ts,
        ctx,
        padding_mask=torch.zeros(
            b, 1, *noisy.shape[-2:], dtype=pad_dtype, device=device
        ),
        target_input_ids=stacked(2),
        target_attention_mask=stacked(3),
        source_attention_mask=stacked(1),
    ).squeeze(2)


def pair_stats(xs: torch.Tensor, base: torch.Tensor, base_m2: float) -> dict:
    """``xs``, ``base``: (D, rows, dim) on the same draws."""
    e = xs - base
    d = e.shape[0]
    mean = e.mean(0)
    m2 = float(mean.pow(2).sum())
    s2 = float((e - mean).pow(2).sum()) / (d - 1)
    bias2 = m2 - s2 / d
    am = xs.mean(0)
    bm = base.mean(0)
    return {
        "single": (float(e.pow(2).sum()) / d / base_m2) ** 0.5,
        "bias": (max(bias2, 0.0) / base_m2) ** 0.5,
        "bias2_raw": bias2 / base_m2,
        "se": (s2 / d / base_m2) ** 0.5,
        "cos_mean": float((am * bm).sum() / (am.norm() * bm.norm())),
        "_m2": m2,
        "_s2d": s2 / d,
        "_sq": float(e.pow(2).sum()) / d,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="hira_cold_hp")
    ap.add_argument(
        "--rows", default="s4050", help="a save under the run dir, '' = its end"
    )
    ap.add_argument("--sigmas", default="0.3125,0.625")
    ap.add_argument("--draws", type=int, default=128)
    ap.add_argument("--ref_draws", type=int, default=16)
    ap.add_argument(
        "--srcs", default="scene,grid", help="one batch per entry, by source"
    )
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    assert 2 <= a.ref_draws <= a.draws

    from bench._common import start_heartbeat
    from common.models import checkpoints, gen_args
    from library.anima import models as am
    from library.anima.stream import stream_fp32_
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.harness import compile_blocks_for_training
    from reseed import trainer as T
    from reseed.config import load
    from reseed.loss import box_share_fm_loss
    from reseed.rows import Rows
    from train.stage import Batcher, LatentStore, _encode_text

    start_heartbeat(label="row_grad_bias")
    out = (
        HOME
        / "results"
        / (
            time.strftime("%Y%m%d-%H%M")
            + "-row_grad_bias"
            + (f"-{a.label}" if a.label else "")
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    for s in sigmas:
        assert float(torch.tensor(s, dtype=torch.bfloat16)) == s, (
            f"σ {s} is not exact in bf16"
        )

    run = load(a.run)
    run.use_pack()
    data = run.data
    recs, ev, vocabs = T.load_items(data)
    build = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if build.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    if build.get("glyph_route_ko"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE_KO"] = "1"
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
    assert getattr(anima, "stream", "bf16") == "bf16"
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    rows_pt = run.dir / a.rows / "trained.pt" if a.rows else run.dir / "trained.pt"
    rows = Rows(
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
    live = (rows.touched_mask & ~rows.frozen_mask).nonzero().squeeze(1)
    anima.train()

    batcher = Batcher(ns, recs, lat)
    want = a.srcs.split(",")
    batches, step = [], 0
    while len(batches) < len(want):
        step += 1
        assert step < 10_000, f"no {want[len(batches)]} batch in {a.run}'s data"
        idx = batcher.next(step)
        if recs[idx[0]]["src"] == want[len(batches)]:
            batches.append(idx)
    print(
        "batches: "
        + "; ".join(
            f"{recs[b[0]]['src']} {recs[b[0]]['shape']} "
            + " | ".join(recs[i].get("text", "")[:24] for i in b)
            for b in batches
        ),
        flush=True,
    )
    cells = [(bi, si) for bi in range(len(batches)) for si in range(len(sigmas))]

    def one(
        bi, si, d, *, noisy_dtype, autocast, ctx_dtype=None, pad_dtype=torch.bfloat16
    ):
        idx = batches[bi]
        brecs = [recs[i] for i in idx]
        caps = [r["caption"] for r in brecs]
        latents = lat[idx].to(device).float()
        g = torch.Generator(device=device).manual_seed(100_000 * bi + 1_000 * si + d)
        noise = torch.randn(latents.shape, generator=g, device=device)
        s = sigmas[si]
        noisy = ((1.0 - s) * latents + s * noise).to(noisy_dtype)
        ts = torch.full((len(idx),), s, device=device, dtype=noisy_dtype)
        target = noise - latents
        src = brecs[0]["src"]
        bs = T.BOX_SHARE if src == "scene" or (T.GRID_BOX and src == "grid") else 0.0
        rows.delta.raw.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            pred = forward(
                anima,
                noisy,
                ts,
                cache,
                caps,
                device,
                ctx_dtype=ctx_dtype,
                pad_dtype=pad_dtype,
            )
        loss = box_share_fm_loss(
            pred, target, brecs, bs, T.BOX_SHARE_CAP, T.BOX_SHARE_GLYPHS, T.GRID_BOX
        )
        loss.backward()
        return rows.delta.raw.grad[live].detach().float().cpu(), float(loss)

    t0 = time.perf_counter()
    grads: dict = {}
    losses: dict = {}

    def collect(arm, n, **kw):
        for bi, si in cells:
            gs, ls = [], []
            for d in range(n):
                g, ls_ = one(bi, si, d, **kw)
                gs.append(g)
                ls.append(ls_)
            grads[arm, bi, si] = torch.stack(gs)
            losses[arm, bi, si] = sum(ls) / n
        print(
            f"{arm}: {n} draws × {len(cells)} cells, {time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    # -- ref: the whole model in fp32, eager ------------------------------------
    anima.enable_gradient_checkpointing()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtypes = {
        n: t.dtype for n, t in [*anima.named_parameters(), *anima.named_buffers()]
    }
    attn_mode = anima.attn_mode
    anima.float()
    anima.attn_mode = "torch"
    collect(
        "ref",
        a.ref_draws,
        noisy_dtype=torch.float32,
        autocast=False,
        ctx_dtype=torch.float32,
        pad_dtype=torch.float32,
    )
    for n, t in [*anima.named_parameters(), *anima.named_buffers()]:
        t.data = t.data.to(dtypes[n])
    anima.attn_mode = attn_mode
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    anima.disable_gradient_checkpointing()

    def compile_(suffix=""):
        if a.no_compile:
            return
        for block in anima.blocks:
            block.__dict__.pop("_forward", None)
        torch._dynamo.reset()
        compile_blocks_for_training(
            anima,
            None,
            backend="inductor",
            n_token_families=lat.n_families,
            signature_suffix=suffix,
        )

    # -- the arms, in the order the conversions stack ---------------------------
    compile_()
    collect("bf16", a.draws, noisy_dtype=torch.bfloat16, autocast=True)
    stream_fp32_(anima)
    compile_(";stream=fp32")
    collect("stream", a.draws, noisy_dtype=torch.float32, autocast=True)

    orig_block = am.Block.forward

    def block_forward(self, x, *args, **kwargs):
        # additive on a detached x: the forward is x · (1 + 2^-9 n), and nothing extra
        # is saved for the backward (the multiplicative form held a stream-sized
        # tensor a block and ran out of memory)
        with torch.no_grad():
            n = torch.randn(x.shape, generator=JIT[0], device=x.device, dtype=x.dtype)
            n.mul_(x).mul_(2.0**-9)
        return orig_block(self, x + n, *args, **kwargs)

    am.Block.forward = block_forward
    JIT[0] = torch.Generator(device=device).manual_seed(1)
    collect("stream_jit", a.draws, noisy_dtype=torch.float32, autocast=True)
    am.Block.forward = orig_block
    JIT[0] = None

    adapter = anima.llm_adapter
    adapter.float()
    adapter_forward = adapter.forward

    def adapter_fp32(*args, **kwargs):
        def up(v):
            return v.float() if torch.is_tensor(v) and v.is_floating_point() else v

        with torch.autocast("cuda", enabled=False):
            return adapter_forward(
                *(up(v) for v in args), **{k: up(v) for k, v in kwargs.items()}
            )

    adapter.forward = adapter_fp32
    # an fp32 context does not fit beside the fp32 stream at batch 4 on 16 GB;
    # recompute leaves the numerics as they are
    anima.enable_gradient_checkpointing()
    torch.cuda.empty_cache()
    collect("stream_adapter", a.draws, noisy_dtype=torch.float32, autocast=True)

    # -- reads ------------------------------------------------------------------
    res = {
        "run": a.run,
        "rows": str(rows_pt),
        "n_live_rows": int(live.numel()),
        "sigmas": sigmas,
        "draws": a.draws,
        "ref_draws": a.ref_draws,
        "compiled": not a.no_compile,
        "batches": [
            {
                "src": recs[b[0]]["src"],
                "shape": recs[b[0]]["shape"],
                "items": [recs[i].get("text", "") for i in b],
            }
            for b in batches
        ],
        "cells": {},
        "pooled": {},
    }
    R = a.ref_draws
    pool = {
        s: {arm: {"m2": 0.0, "s2d": 0.0, "sq": 0.0} for arm in PAIRED}
        | {"_base": 0.0, "_spread": 0.0}
        for s in sigmas
    }
    for bi, si in cells:
        s = sigmas[si]
        base = grads["stream", bi, si]
        bmean = base.mean(0)
        base_m2 = float(bmean.pow(2).sum())
        spread2 = float((base - bmean).pow(2).sum()) / (base.shape[0] - 1)
        cell = {
            "stream_norm": base_m2**0.5,
            "eps_spread": (spread2 / base_m2) ** 0.5,
            "loss": {
                arm: losses[arm, b, i] for arm, b, i in losses if (b, i) == (bi, si)
            },
            "vs_ref_single": {},
        }
        ref = grads["ref", bi, si]
        rm2 = float(ref.mean(0).pow(2).sum())
        for arm in ("bf16", "stream", "stream_jit", "stream_adapter"):
            e = grads[arm, bi, si][:R] - ref
            cell["vs_ref_single"][arm] = (float(e.pow(2).sum()) / R / rm2) ** 0.5
        pool[s]["_base"] += base_m2
        pool[s]["_spread"] += spread2
        for arm in PAIRED:
            st = pair_stats(grads[arm, bi, si], base, base_m2)
            pool[s][arm]["m2"] += st["_m2"]
            pool[s][arm]["s2d"] += st["_s2d"]
            pool[s][arm]["sq"] += st["_sq"]
            cell[arm] = {k: v for k, v in st.items() if not k.startswith("_")}
        res["cells"][f"b{bi}_{recs[batches[bi][0]]['src']}_s{s}"] = cell
    for s, pp in pool.items():
        bm2 = pp["_base"]
        res["pooled"][str(s)] = {"eps_spread": (pp["_spread"] / bm2) ** 0.5} | {
            arm: {
                "bias": (max(pp[arm]["m2"] - pp[arm]["s2d"], 0.0) / bm2) ** 0.5,
                "bias2_raw": (pp[arm]["m2"] - pp[arm]["s2d"]) / bm2,
                "se": (pp[arm]["s2d"] / bm2) ** 0.5,
                "single": (pp[arm]["sq"] / bm2) ** 0.5,
            }
            for arm in PAIRED
        }
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print("\nRESULT % of ‖mean g_stream‖; pairs against stream on the same draws")
    for s, r in res["pooled"].items():
        print(
            f"RESULT σ {s} (pooled): ε spread of one draw {r['eps_spread'] * 100:.0f}"
        )
        for arm in PAIRED:
            x = r[arm]
            print(
                f"RESULT σ {s} {arm:15s} bias {x['bias'] * 100:6.2f}  (√raw² "
                f"{x['bias2_raw'] * 1e4:+8.2f} %², se {x['se'] * 100:5.2f})  "
                f"single {x['single'] * 100:6.2f}"
            )
    for c, cell in res["cells"].items():
        print(
            f"CELL {c}: ε spread {cell['eps_spread'] * 100:.0f} | "
            + " | ".join(
                f"{arm} bias {cell[arm]['bias'] * 100:.2f}±{cell[arm]['se'] * 100:.2f} "
                f"cos {cell[arm]['cos_mean']:.4f}"
                for arm in PAIRED
            )
            + " | vs ref single: "
            + ", ".join(f"{k} {v * 100:.1f}" for k, v in cell["vs_ref_single"].items())
        )
    print(
        f"→ {out / 'result.json'} ({(time.perf_counter() - t0) / 60:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
