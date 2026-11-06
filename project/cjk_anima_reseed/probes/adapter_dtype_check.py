"""adapter_dtype_check — does ``row_grad_bias``'s ``stream_adapter`` arm run the adapter in
fp32? Forward only, one scene batch of ``hira_cold_hp`` at ``s4050``, the probe's wrapper:

- dtypes at the embed output (after the pack and ExtDelta hooks), at every adapter
  linear's input and output, at the adapter's output, and at a DiT block's cross-attn
  ``k_proj`` input;
- a row nudge below bf16's resolution (``raw`` + 2^-10 relative on one row in the batch):
  does the adapter output move? It should on fp32 and not on bf16;
- adapter output, fp32 against bf16: relative distance.

    make daemon-run ARGS="--stall-timeout 300 project/cjk_anima_reseed/probes/adapter_dtype_check.py"
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import bootstrap  # noqa: E402

bootstrap()

import torch  # noqa: E402


def main():
    from common.models import checkpoints, gen_args
    from library.anima.stream import stream_fp32_
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from reseed import trainer as T
    from reseed.config import load
    from reseed.rows import Rows
    from train.stage import Batcher, LatentStore, _encode_text

    run = load("hira_cold_hp")
    run.use_pack()
    data = run.data
    recs, ev, vocabs = T.load_items(data)
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    out = Path("/tmp") / "adapter_dtype_check"
    out.mkdir(exist_ok=True)
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
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    print("pack table dtype:", pack.table.dtype, tuple(pack.table.shape), flush=True)
    rows = Rows(
        anima,
        device,
        p.idx,
        pack,
        warm=run.dir / "s4050" / "trained.pt",
        init_anchor=0.0,
        free_residual=0.0,
        lr=T.LR,
        touched=p.touched,
        frozen=p.frozen,
        context=ctx,
    )
    stream_fp32_(anima)
    anima.eval()

    batcher = Batcher(ns, recs, lat)
    step = 0
    while True:
        step += 1
        idx = batcher.next(step)
        if recs[idx[0]]["src"] == "scene":
            break
    caps = [recs[i]["caption"] for i in idx]

    def stacked(k):
        return torch.stack([cache[c][k] for c in caps]).to(device)

    adapter = anima.llm_adapter
    seen: Counter = Counter()
    captured = {}

    def tag(name):
        def hook(mod, inputs, output):
            x = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            seen[
                f"{name}: in {x.dtype if x is not None else '-'} → out {output.dtype}"
            ] += 1

        return hook

    hs = [adapter.embed.register_forward_hook(tag("embed"))]
    for n, m in adapter.named_modules():
        if isinstance(m, torch.nn.Linear):
            hs.append(m.register_forward_hook(tag("adapter linear")))
    hs.append(
        adapter.register_forward_hook(
            lambda m, i, o: captured.__setitem__("out", o.detach().float().clone())
        )
    )
    hs.append(
        anima.blocks[0].cross_attn.kv_proj.register_forward_hook(
            tag("block0 cross_attn.kv_proj")
        )
    )
    hs.append(
        anima.blocks[0].cross_attn.kv_proj.register_forward_pre_hook(
            lambda m, a: seen.__setitem__(
                f"block0 kv_proj input (pre-autocast) {a[0].dtype}", 1
            )
        )
    )

    lat_ = lat[idx].to(device).float()
    noisy = 0.375 * lat_ + 0.625 * torch.randn_like(lat_)
    ts = torch.full((len(idx),), 0.625, device=device)

    def fwd():
        seen.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            anima(
                noisy.unsqueeze(2),
                ts,
                stacked(0),
                padding_mask=torch.zeros(
                    len(idx), 1, *noisy.shape[-2:], dtype=torch.bfloat16, device=device
                ),
                target_input_ids=stacked(2),
                target_attention_mask=stacked(3),
                source_attention_mask=stacked(1),
            )
        return captured["out"], dict(seen)

    ids = stacked(2)
    ext_in_batch = sorted({int(e) - 32128 for e in ids[ids >= 32128].tolist()})
    live = [
        i
        for i, e in enumerate(rows.delta.ext_ids)
        if e in ext_in_batch and not bool(rows.frozen_mask[i])
    ]
    k = live[0]
    valid = stacked(3).bool()
    tok = ids == 32128 + rows.delta.ext_ids[k]
    row_eff = rows.pack_rows[k] + rows.delta.raw.detach()[k] * rows.row_scale
    nudge = 2.0**-10 * row_eff.norm() / rows.row_scale / row_eff.numel() ** 0.5

    res = {}
    for mode in ("bf16", "fp32"):
        if mode == "fp32":
            adapter.float()
            orig = adapter.forward

            def adapter_fp32(*args, **kwargs):
                def up(v):
                    return (
                        v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                    )

                with torch.autocast("cuda", enabled=False):
                    return orig(
                        *(up(v) for v in args),
                        **{kk: up(v) for kk, v in kwargs.items()},
                    )

            adapter.forward = adapter_fp32
        base, dt = fwd()
        again, _ = fwd()
        raw0 = rows.delta.raw.detach().clone()
        sign = torch.sign(
            torch.randn(raw0[k].shape, generator=torch.Generator().manual_seed(0))
        ).to(device)
        with torch.no_grad():
            rows.delta.raw[k] += nudge * sign
        moved, _ = fwd()
        with torch.no_grad():
            rows.delta.raw.copy_(raw0)

        def rel(x, m):
            return float((x - base)[m].norm() / base[m].norm())

        res[mode] = {
            "dtypes": dt,
            "out": base,
            "nudge_moved": rel(moved, valid),
            "nudge_moved_tok": rel(moved, tok),
            "nudge_moved_other": rel(moved, valid & ~tok),
            "rerun": rel(again, valid),
            "pad_norm_share": float(base[~valid].norm() / base.norm()),
        }
        print(f"\n== {mode}", flush=True)
        for kk, v in dt.items():
            print(f"  {kk}  ×{v}")
        r = res[mode]
        print(
            f"  valid tokens: rerun with no nudge moved {r['rerun']:.3e}; nudge 2^-10 on ext row "
            f"{rows.delta.ext_ids[k]} moved {r['nudge_moved']:.3e} (its {int(tok.sum())} token(s) "
            f"{r['nudge_moved_tok']:.3e}, the rest {r['nudge_moved_other']:.3e}); padding holds "
            f"{r['pad_norm_share'] * 100:.1f} % of the output norm",
            flush=True,
        )
    f32, b16 = res["fp32"]["out"], res["bf16"]["out"]
    d = float((f32 - b16)[valid].norm() / f32[valid].norm())
    dp = float((f32 - b16).norm() / f32.norm())
    print(
        f"\nadapter output fp32 vs bf16: valid tokens {d * 100:.3f} %, all positions {dp * 100:.3f} %",
        flush=True,
    )
    print(
        json.dumps(
            {
                m: {kk: v for kk, v in r.items() if kk not in ("out", "dtypes")}
                for m, r in res.items()
            }
        )
    )


if __name__ == "__main__":
    main()
