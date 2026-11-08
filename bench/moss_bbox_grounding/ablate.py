#!/usr/bin/env python3
"""Causal check for ``probe.py``: which blocks read the bbox, and through which tokens.

Same pages, noise and σ as the probe (default σ 0.9, where the box cuts the in-box
loss 14 %). Each condition masks a key set for every query (``key_mask``) in a block
set, then measures the flow-matching MSE in the true box ``T`` and the moved box ``M``:

  keys   ``coord`` (the bbox list) · ``tail`` (the caption after it: the item's
         closers and the top-level bbox keys) · ``suffix`` (the chat template's
         7 closing tokens, the attention sink). The TE is causal, so every token
         after the box already carries it; ``tail`` / ``suffix`` test that route.
  blocks a range of the 42 DiT blocks; ``+r`` also masks the keys in the text
         refiner (``text_embedder``), where the text tokens mix before the DiT.

Read-outs, per condition (mean over pages, bootstrap 95 % CI):

  recovery  (T(abl) − T(bbox)) / (T(ctrl) − T(bbox)) on the ``bbox`` caption:
            1 = the box no longer helps. ``ctrl`` is ``none`` with the same keys
            masked where it has them (its tail / suffix), else plain ``none``.
  moved     (M(abl) − M(ctrl)) / (M(bbox_moved) − M(ctrl)) on ``bbox_moved``:
            1 = lettering is still pushed into the moved box, 0 = no longer.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_bbox_grounding/ablate.py"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench._common import make_run_dir, start_heartbeat, write_result
from bench.moss_bbox_grounding.probe import box_cells, load_samples, variants

RANGES = {
    "0-4": range(0, 5),
    "5-8": range(5, 9),
    "9-13": range(9, 14),
    "14-20": range(14, 21),
    "21-25": range(21, 26),
    "26-35": range(26, 36),
    "36-41": range(36, 42),
    "0-8": range(0, 9),
    "0-13": range(0, 14),
    "0-20": range(0, 21),
    "0-25": range(0, 26),
    "9-41": range(9, 42),
    "21-41": range(21, 42),
    "all": range(0, 42),
}


def conditions():
    """``[(name, keys, block range name, refiner)]``."""
    out = [("coord@all+r", ("coord",), "all", True)]
    out += [(f"coord@{r}", ("coord",), r, False) for r in RANGES if r != "all"]
    out += [
        ("coord@all", ("coord",), "all", False),
        ("coord@r", ("coord",), None, True),
        ("coord+tail@all+r", ("coord", "tail"), "all", True),
        ("coord+tail+suffix@all+r", ("coord", "tail", "suffix"), "all", True),
        ("coord+tail@0-8+r", ("coord", "tail"), "0-8", True),
    ]
    return out


def segments(tok, caption):
    """Text-relative token indices: ``coord`` (bbox list), ``tail`` (rest of the
    caption after the coord, or after the item text when there is no box) and
    ``suffix`` (template after the caption)."""
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": caption}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    base = formatted.index(caption)
    end = base + len(caption)
    items = caption.index('"items"')
    if '"bbox":[' in caption:
        lo = caption.index('"bbox":[', items) + len('"bbox":[')
        coord = (base + lo, base + caption.index("]", lo))
        after = coord[1]
    else:
        coord = None
        text = json.loads(caption)["items"][0]["text"]
        lo = caption.index('"text":"', items) + len('"text":"')
        after = base + lo + len(json.dumps(text, ensure_ascii=False)) - 2
    offs = tok(formatted, return_offsets_mapping=True)["offset_mapping"]
    seg = {"coord": [], "tail": [], "suffix": []}
    for j, (s, e) in enumerate(offs):
        if coord and s < coord[1] and e > coord[0]:
            seg["coord"].append(j)
        elif s >= after and e <= end and e > s:
            seg["tail"].append(j)
        elif s >= end:
            seg["suffix"].append(j)
    seg["n"] = len(offs)
    return seg


def boot(values, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache_dir", default="post_image_dataset/moss/i2i_swap_text_v3_all"
    )
    ap.add_argument("--n", type=int, default=0, help="samples (0 = all one-item)")
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    start_heartbeat(label="moss_bbox_ablate")

    import library.moss.model as moss_model
    from library.env import resolve_under_home
    from library.moss.generate import cached_embeds
    from library.moss.loader import load_tokenizer, load_transformer
    from library.moss.requests import resolve_model_dir

    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    tok = load_tokenizer(model_dir)
    jobs = []
    for s in load_samples(resolve_under_home(args.cache_dir), args.n):
        caps, moved = variants(s["caption"], s["height"], s["width"])
        if caps is None:
            continue
        caps = {v: caps[v] for v in ("bbox", "bbox_moved", "none")}
        segs = {v: segments(tok, c) for v, c in caps.items()}
        box = json.loads(s["caption"])["items"][0]["bbox"]
        jobs.append(dict(s, caps=caps, segs=segs, box=box, moved=moved))
    del tok
    embeds = cached_embeds(
        model_dir,
        [c for j in jobs for c in j["caps"].values()],
        Path(args.te_cache),
        precision=args.text_encoder,
    )
    for j in jobs:
        for v, c in j["caps"].items():
            assert embeds[c].shape[0] == j["segs"][v]["n"], (j["stem"], v)
    conds = conditions()
    print(f"[ablate] {len(jobs)} samples, {len(conds)} conditions", flush=True)

    transformer = load_transformer(model_dir, args.model, device=device)
    original = moss_model._attention
    state = {"block": None, "blocks": frozenset(), "refiner": False, "drop": None}

    def wrapped(q, k, v, rope, heads, key_mask=None, *rest, **kw):
        drop, b = state["drop"], state["block"]
        if drop is not None:
            n = q.shape[1]
            hit = (b is None and state["refiner"]) or (b in state["blocks"])
            if hit:
                off = n - state["n_text"]  # 0 in the refiner, n_img in the DiT
                mask = torch.ones(1, n, dtype=torch.bool, device=q.device)
                mask[0, off + drop] = False
                key_mask = mask if key_mask is None else key_mask & mask
        return original(q, k, v, rope, heads, key_mask, *rest, **kw)

    def set_block(index):
        def hook(_m, _a):
            state["block"] = index

        return hook

    handles = [transformer.text_embedder.register_forward_pre_hook(set_block(None))]
    for i, layer in enumerate(transformer.layers):
        handles.append(layer.register_forward_pre_hook(set_block(i)))
    moss_model._attention = wrapped

    # loss[(variant, condition)] = (S, 2) T / M
    names = ["plain"] + [c[0] for c in conds]
    loss = {(v, c): np.full((len(jobs), 2), np.nan) for v in ("bbox", "bbox_moved", "none") for c in names}  # fmt: skip
    sigma = args.sigma
    try:
        with torch.inference_mode():
            for si, job in enumerate(jobs):
                height, width = job["height"], job["width"]
                t_m = box_cells(job["box"], height, width)
                m_m = box_cells(job["moved"], height, width)
                regw = torch.stack([t_m, m_m], 1).float()
                regw = (regw / regw.sum(0, keepdim=True)).to(device)
                latent = job["latent"][None].to(device, torch.float32)
                # the probe's noise for its first σ draw when --sigma matches it
                gen = torch.Generator(device="cpu").manual_seed(args.seed + si)
                noise = torch.randn(latent.shape, generator=gen).to(device)
                noisy = (1.0 - sigma) * latent + sigma * noise
                target = noise - latent
                timestep = torch.tensor([1000.0 * sigma], device=device)

                def run(v, keys, blocks, refiner):
                    seg = job["segs"][v]
                    drop = [i for k in keys for i in seg[k]]
                    if keys and not drop:
                        return None  # nothing of this key set in the caption
                    state.update(
                        drop=torch.tensor(drop, device=device) if drop else None,
                        blocks=frozenset(RANGES[blocks]) if blocks else frozenset(),
                        refiner=refiner,
                        n_text=seg["n"],
                    )
                    pred = transformer(
                        noisy,
                        timestep,
                        embeds[job["caps"][v]][None].to(device, torch.bfloat16),
                        return_dict=False,
                    )[0]
                    state["drop"] = None
                    err = (pred.float() - target).pow(2).mean(1)[0].flatten()
                    return (err @ regw).cpu().numpy()

                for v in ("bbox", "bbox_moved", "none"):
                    loss[(v, "plain")][si] = run(v, (), None, False)
                for name, keys, blocks, refiner in conds:
                    for v in ("bbox", "bbox_moved"):
                        loss[(v, name)][si] = run(v, keys, blocks, refiner)
                    ctrl_keys = tuple(k for k in keys if k != "coord")
                    if ctrl_keys:
                        out = run("none", ctrl_keys, blocks, refiner)
                        if out is not None:
                            loss[("none", name)][si] = out
                print(f"[ablate] {si + 1}/{len(jobs)} {job['stem']}", flush=True)
    finally:
        moss_model._attention = original
        for h in handles:
            h.remove()

    def ctrl(name):
        c = loss[("none", name)]
        return c if not np.isnan(c).all() else loss[("none", "plain")]

    b_t = loss[("bbox", "plain")][:, 0]
    mv_m = loss[("bbox_moved", "plain")][:, 1]
    rows = {}
    for name in names:
        c = ctrl(name)
        a_t = loss[("bbox", name)][:, 0]
        a_m = loss[("bbox_moved", name)][:, 1]
        gap_t = np.nanmean(c[:, 0] - b_t)
        gap_m = np.nanmean(mv_m - c[:, 1])
        # per-sample numerators over the mean gap: stable when one page's gap is ~0
        rec = (a_t - b_t) / gap_t
        mov = (a_m - c[:, 1]) / gap_m
        rows[name] = dict(
            T_bbox=round(float(np.nanmean(a_t)), 5),
            M_moved=round(float(np.nanmean(a_m)), 5),
            T_ctrl=round(float(np.nanmean(c[:, 0])), 5),
            recovery=round(float(np.nanmean(rec)), 3),
            recovery_ci=[round(x, 3) for x in boot(rec)],
            moved=round(float(np.nanmean(mov)), 3),
            moved_ci=[round(x, 3) for x in boot(mov)],
        )
        print(f"[ablate] {name:26s} recovery {rows[name]['recovery']:+.2f} "
              f"{rows[name]['recovery_ci']}  moved {rows[name]['moved']:+.2f} "
              f"{rows[name]['moved_ci']}", flush=True)  # fmt: skip

    run_dir = make_run_dir("moss_bbox_grounding", args.label or "ablate")
    np.savez_compressed(
        run_dir / "loss.npz",
        **{f"{v}|{c}": a for (v, c), a in loss.items()},
        stems=np.array([j["stem"] for j in jobs]),
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=dict(n_samples=len(jobs), sigma=sigma, conditions=rows),
        label=args.label,
        artifacts=["loss.npz"],
        device=device,
    )
    print(f"[ablate] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
