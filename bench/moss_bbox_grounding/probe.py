#!/usr/bin/env python3
"""Where MOSS-Image Base reads a layout item's location: bbox vs position word.

Teacher-forced (no sampling): cached swap_text targets with one ``text`` item (the
item's box holds the pasted lettering), noised at fixed σ with a fixed noise per
sample, one cond forward per prompt variant. The variants differ only in the
item's location:

  bbox       the staged caption (``abs_centralized`` ``yxyx`` box)
  bbox_moved the same box moved off itself (point reflection, else a flip)
  none       no ``bbox`` key (and no top-level bbox keys)
  pos        ``"position"`` word of the true box (3×3 grid: "top left" … "center")
  pos_moved  the moved box's position word

Image-token regions: ``T`` (cells whose centre is in the true box), ``M`` (the
moved box), ``R`` (the rest). Per (sample, σ, variant), every joint block's
attention is recomputed with an explicit softmax for the image queries:

  * ``content``: mean mass a region's queries put on the item's ``"text"`` value
    tokens, per (block, head, region)
  * ``coord``: the same on the location tokens (the bbox list or position word)
  * ``hdiff``: per block, ‖h_variant − h_bbox‖ over image tokens (massive channels
    out), region means over the image mean of ‖h_bbox‖
  * ``loss``: flow-matching MSE per region

The summary's localisation is ``loc = log content_T − log content_M``; a head that
reads the coordinates has ``loc(bbox) > 0`` and ``loc(bbox_moved) < 0``, one that
follows the visible lettering keeps ``loc > 0`` under both. ``follow = loc(bbox) −
loc(bbox_moved)`` (and the ``pos`` pair) ranks the heads.

    make daemon-run ARGS="--stall-timeout 600 bench/moss_bbox_grounding/probe.py"
    make daemon-run ARGS="bench/moss_bbox_grounding/probe.py --n 2 --label smoke"
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from bench._common import make_run_dir, start_heartbeat, write_result

VARIANTS = ("bbox", "bbox_moved", "none", "pos", "pos_moved")
REGIONS = ("T", "M", "R")
EPS = 1e-8


def dumps(layout):
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def position_word(box, height, width):
    """Centre of a centralised ``yxyx`` box → a 3×3 grid word."""
    fy = ((box[0] + box[2]) / 2 + height / 2) / height
    fx = ((box[1] + box[3]) / 2 + width / 2) / width
    row = "top" if fy < 1 / 3 else "bottom" if fy > 2 / 3 else ""
    col = "left" if fx < 1 / 3 else "right" if fx > 2 / 3 else ""
    return " ".join(w for w in (row, col) if w) or "center"


def box_cells(box, height, width):
    """Bool ``(h/16 · w/16)`` of the cells whose centre is inside the box (the
    cell holding the box centre when none is)."""
    h16, w16 = height // 16, width // 16
    y0, x0, y1, x1 = (box[0] + height / 2, box[1] + width / 2,
                      box[2] + height / 2, box[3] + width / 2)  # fmt: skip
    cy = (torch.arange(h16) + 0.5) * 16
    cx = (torch.arange(w16) + 0.5) * 16
    m = ((cy >= y0) & (cy <= y1))[:, None] & ((cx >= x0) & (cx <= x1))[None]
    if not m.any():
        r = min(h16 - 1, int((y0 + y1) / 2 // 16))
        c = min(w16 - 1, int((x0 + x1) / 2 // 16))
        m[r, c] = True
    return m.flatten()


def moved_box(box, height, width):
    """The box moved to a spot sharing no cell and no position word with it."""
    y0, x0, y1, x1 = box
    true = box_cells(box, height, width)
    word = position_word(box, height, width)
    for cand in ([-y1, -x1, -y0, -x0], [y0, -x1, y1, -x0], [-y1, x0, -y0, x1]):
        if (box_cells(cand, height, width) & true).any():
            continue
        if position_word(cand, height, width) == word:
            continue
        return cand
    return None


def variants(caption, height, width):
    """``{variant: caption}``, the moved box, or None when no move qualifies."""
    layout = json.loads(caption)
    (item,) = layout["items"]
    box = item["bbox"]
    moved = moved_box(box, height, width)
    if moved is None:
        return None, None

    def without_box(extra=None):
        out = copy.deepcopy(layout)
        out.pop("bbox_coord_space", None)
        out.pop("bbox_format", None)
        it = out["items"][0]
        it.pop("bbox")
        if extra:
            it["position"] = extra
        return out

    shifted = copy.deepcopy(layout)
    shifted["items"][0]["bbox"] = moved
    caps = {
        "bbox": caption,
        "bbox_moved": dumps(shifted),
        "none": dumps(without_box()),
        "pos": dumps(without_box(position_word(box, height, width))),
        "pos_moved": dumps(without_box(position_word(moved, height, width))),
    }
    return caps, moved


def token_spans(tok, caption):
    """Joint-text token indices of the item's text value and of its location
    (bbox list or position word) under the chat template."""
    from library.moss.loader import prompt_ids

    formatted = tok.apply_chat_template(
        [{"role": "user", "content": caption}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    base = formatted.index(caption)
    items = caption.index('"items"')
    key = '"text":"'
    lo = caption.index(key, items) + len(key)
    text = json.loads(caption)["items"][0]["text"]
    content = (lo, lo + len(json.dumps(text, ensure_ascii=False)) - 2)
    if '"bbox":[' in caption:
        lo = caption.index('"bbox":[', items) + len('"bbox":[')
        coord = (lo, caption.index("]", lo))
    elif '"position":"' in caption:
        lo = caption.index('"position":"', items) + len('"position":"')
        coord = (lo, caption.index('"', lo))
    else:
        coord = None
    enc = tok(formatted, return_offsets_mapping=True)
    n = prompt_ids(tok, caption).input_ids.shape[1]
    assert len(enc["input_ids"]) == n, (len(enc["input_ids"]), n)

    def pick(span):
        if span is None:
            return []
        a, b = base + span[0], base + span[1]
        return [j for j, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a]

    return pick(content), pick(coord), n


class Probe:
    """Explicit-softmax ``_attention`` wrapper + block-output hook for one forward."""

    def __init__(self, num_blocks, heads, massive, chunk):
        self.L, self.H, self.chunk = num_blocks, heads, chunk
        self.massive = massive
        self.block = None
        self.active = False

    def start(self, n_img, n_text, content, coord, regw, ref_h, keep_h):
        self.n_img, self.N = n_img, n_img + n_text
        self.content = torch.tensor([n_img + j for j in content], device=regw.device)
        self.coord = torch.tensor([n_img + j for j in coord], device=regw.device)
        self.regw = regw  # (Q, 3), columns sum to 1
        self.ref_h, self.keep_h = ref_h, keep_h
        self.content_out = np.full((self.L, self.H, 3), np.nan, np.float32)
        self.coord_out = np.full_like(self.content_out, np.nan)
        self.hdiff_out = np.full((self.L, 3), np.nan, np.float32)
        self.hmap = np.zeros((self.L, n_img), np.float16)
        self.cmap = np.zeros((self.L, n_img), np.float16)
        self.active = True

    def attention(self, q, k, v, rope, heads):
        from library.moss.model import rotate

        if not self.active or self.block is None or q.shape[1] != self.N:
            return
        b, n_img = self.block, self.n_img
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        q = rotate(q.reshape(shape), rope).transpose(1, 2)[0]
        k = rotate(k.reshape(shape), rope).transpose(1, 2)[0]
        scale = 1.0 / math.sqrt(shape[-1])
        cm_sum = torch.zeros(n_img, device=q.device)
        for h0 in range(0, heads, self.chunk):
            hs = slice(h0, min(h0 + self.chunk, heads))
            p = torch.softmax(
                (q[hs, :n_img].float() @ k[hs].float().transpose(-1, -2)) * scale, -1
            )
            cm = p[..., self.content].sum(-1)  # (h, Q)
            self.content_out[b, hs] = (cm @ self.regw).cpu().numpy()
            cm_sum += cm.sum(0)
            if self.coord.numel():
                cd = p[..., self.coord].sum(-1)
                self.coord_out[b, hs] = (cd @ self.regw).cpu().numpy()
            del p
        self.cmap[b] = (cm_sum / heads).cpu().numpy()

    def __call__(self, index, x, n_img):
        if not self.active:
            return x
        if self.ref_h is None:
            return x
        h = x[0, :n_img].float() * self.keep_h
        ref = self.ref_h[index].to(h.device, non_blocking=True).float()
        d = (h - ref).norm(dim=-1) / ref.norm(dim=-1).mean()
        self.hdiff_out[index] = (d @ self.regw).cpu().numpy()
        self.hmap[index] = d.cpu().numpy()
        return x


def load_samples(cache_dir, n):
    out = []
    for te in sorted(glob.glob(f"{cache_dir}/*.moss_te.safetensors")):
        with safe_open(te, "pt") as s:
            caption = s.metadata()["caption"]
        if len(json.loads(caption)["items"]) != 1:
            continue
        stem = Path(te).name[: -len(".moss_te.safetensors")]
        (lat,) = glob.glob(f"{cache_dir}/{stem}_*x*.moss_latent.safetensors")
        with safe_open(lat, "pt") as s:
            meta = s.metadata()
            latent = s.get_tensor("latent")
        out.append(
            dict(
                stem=stem,
                caption=caption,
                latent=latent,
                height=int(meta["height"]),
                width=int(meta["width"]),
            )  # fmt: skip
        )
    return out[:n] if n else out


def plot(path, follow, hd, blocks_follow):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, key in zip(axes[:2], ("bbox", "pos")):
        m = follow[key].mean(0)  # σ mean, (L, H)
        lim = max(1e-3, float(np.nanmax(np.abs(m))))
        im = ax.imshow(m.T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-lim, vmax=lim)  # fmt: skip
        ax.set_title(f"follow ({key}): loc(true) − loc(moved), σ mean")
        ax.set_xlabel("block")
        ax.set_ylabel("head")
        fig.colorbar(im, ax=ax)
    for key, style in (("bbox", "-"), ("pos", "--")):
        for j, row in enumerate(blocks_follow[key]):
            axes[2].plot(row, style, label=f"{key} σ{j}")
    axes[2].set_title("max over heads of follow, per block")
    axes[2].set_xlabel("block")
    axes[2].legend(fontsize=7)
    for v, (tm, r) in hd.items():
        axes[3].plot(tm / np.maximum(r, EPS), label=v)
    axes[3].axhline(1, c="grey", lw=0.6)
    axes[3].set_title("hdiff vs bbox: mean(T,M) / R, per block (σ mean)")
    axes[3].set_xlabel("block")
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache_dir", default="post_image_dataset/moss/i2i_swap_text_v3_all"
    )
    ap.add_argument("--n", type=int, default=40, help="samples (0 = all one-item)")
    ap.add_argument("--sigmas", default="0.9,0.6,0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps", type=int, default=4, help="samples whose maps are kept")
    ap.add_argument("--model", default="base")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument(
        "--massive", default="library/moss/calibration/massive_channels.npz"
    )
    ap.add_argument("--head_chunk", type=int, default=6)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    start_heartbeat(label="moss_bbox_grounding")

    import library.moss.model as moss_model
    from library.env import resolve_under_home
    from library.moss.generate import cached_embeds
    from library.moss.loader import load_tokenizer, load_transformer
    from library.moss.requests import resolve_model_dir

    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    samples = load_samples(resolve_under_home(args.cache_dir), args.n)
    tok = load_tokenizer(model_dir)
    jobs = []
    for s in samples:
        caps, moved = variants(s["caption"], s["height"], s["width"])
        if caps is None:
            print(f"[skip] {s['stem']}: no qualifying move", flush=True)
            continue
        spans = {v: token_spans(tok, c) for v, c in caps.items()}
        box = json.loads(s["caption"])["items"][0]["bbox"]
        jobs.append(dict(s, caps=caps, box=box, moved=moved, spans=spans))
    del tok
    print(f"[probe] {len(jobs)} samples, σ {sigmas}", flush=True)
    embeds = cached_embeds(
        model_dir,
        [c for j in jobs for c in j["caps"].values()],
        Path(args.te_cache),
        precision=args.text_encoder,
    )
    for j in jobs:
        for v, c in j["caps"].items():
            assert embeds[c].shape[0] == j["spans"][v][2], (j["stem"], v)

    transformer = load_transformer(model_dir, args.model, device=device)
    num_blocks, heads = len(transformer.layers), transformer.config.num_heads
    massive = np.load(resolve_under_home(args.massive))["channels"]
    keep = torch.ones(transformer.config.dim, device=device)
    keep[torch.from_numpy(massive).to(device)] = 0

    probe = Probe(num_blocks, heads, massive, args.head_chunk)
    original = moss_model._attention

    def wrapped(q, k, v, rope, heads_, *rest, **kw):
        probe.attention(q, k, v, rope, heads_)
        return original(q, k, v, rope, heads_, *rest, **kw)

    def set_block(index):
        def hook(_m, _a):
            probe.block = index

        return hook

    store = {"h": None}

    def capture(index, x, n_img):
        if store["h"] is not None:
            store["h"][index] = (x[0, :n_img].float() * keep).to(torch.float16).cpu()
        return probe(index, x, n_img)

    handles = [transformer.text_embedder.register_forward_pre_hook(set_block(None))]
    for i, layer in enumerate(transformer.layers):
        handles.append(layer.register_forward_pre_hook(set_block(i)))
    moss_model._attention = wrapped
    transformer.layer_hook = capture

    S, Z, V = len(jobs), len(sigmas), len(VARIANTS)
    content = np.full((S, Z, V, num_blocks, heads, 3), np.nan, np.float32)
    coord = np.full_like(content, np.nan)
    hdiff = np.full((S, Z, V, num_blocks, 3), np.nan, np.float32)
    loss = np.full((S, Z, V, 3), np.nan, np.float32)
    maps = {}
    try:
        with torch.inference_mode():
            for si, job in enumerate(jobs):
                height, width = job["height"], job["width"]
                n_img = (height // 16) * (width // 16)
                t_m = box_cells(job["box"], height, width)
                m_m = box_cells(job["moved"], height, width)
                r_m = ~(t_m | m_m)
                regw = torch.stack([t_m, m_m, r_m], 1).float()
                regw = (regw / regw.sum(0, keepdim=True)).to(device)
                latent = job["latent"][None].to(device, torch.float32)
                gen = torch.Generator(device="cpu").manual_seed(args.seed + si)
                for zi, sigma in enumerate(sigmas):
                    noise = torch.randn(latent.shape, generator=gen).to(device)
                    noisy = (1.0 - sigma) * latent + sigma * noise
                    target = noise - latent
                    timestep = torch.tensor([1000.0 * sigma], device=device)
                    ref_h = None
                    for vi, v in enumerate(VARIANTS):
                        cap = job["caps"][v]
                        c_idx, l_idx, n_text = job["spans"][v]
                        first = vi == 0
                        store["h"] = [None] * num_blocks if first else None
                        probe.start(
                            n_img, n_text, c_idx, l_idx, regw,
                            ref_h, keep,
                        )  # fmt: skip
                        pred = transformer(
                            noisy,
                            timestep,
                            embeds[cap][None].to(device, torch.bfloat16),
                            return_dict=False,
                        )[0]
                        probe.active = False
                        if first:
                            ref_h = store["h"]
                            store["h"] = None
                        err = (pred.float() - target).pow(2).mean(1)[0].flatten()
                        loss[si, zi, vi] = (err @ regw).cpu().numpy()
                        content[si, zi, vi] = probe.content_out
                        coord[si, zi, vi] = probe.coord_out
                        hdiff[si, zi, vi] = probe.hdiff_out
                        if si < args.maps:
                            maps[f"{job['stem']}/{sigma}/{v}/cmap"] = probe.cmap
                            if not first:
                                maps[f"{job['stem']}/{sigma}/{v}/hmap"] = probe.hmap
                    del ref_h
                print(f"[probe] {si + 1}/{S} {job['stem']}", flush=True)
    finally:
        moss_model._attention = original
        transformer.layer_hook = None
        for h in handles:
            h.remove()

    # --- summary -----------------------------------------------------------------
    loc = np.log(content[..., 0] + EPS) - np.log(content[..., 1] + EPS)  # (S,Z,V,L,H)
    loc_mean = np.nanmean(loc, 0)  # (Z, V, L, H)
    vi = {v: i for i, v in enumerate(VARIANTS)}
    follow = {
        "bbox": loc_mean[:, vi["bbox"]] - loc_mean[:, vi["bbox_moved"]],
        "pos": loc_mean[:, vi["pos"]] - loc_mean[:, vi["pos_moved"]],
    }
    # a head "goes with the moved location" on a sample when loc flips sign
    flips = {
        "bbox": np.nanmean(
            (loc[:, :, vi["bbox"]] > 0) & (loc[:, :, vi["bbox_moved"]] < 0), 0
        ),
        "pos": np.nanmean(
            (loc[:, :, vi["pos"]] > 0) & (loc[:, :, vi["pos_moved"]] < 0), 0
        ),
    }

    def top_heads(key):
        f = follow[key].mean(0)
        order = np.argsort(f, axis=None)[::-1][: args.top]
        rows = []
        for flat in order:
            b, h = divmod(int(flat), heads)
            rows.append(
                dict(
                    block=b,
                    head=h,
                    follow=round(float(f[b, h]), 3),
                    follow_per_sigma=[round(float(x), 3) for x in follow[key][:, b, h]],
                    flip_frac=[round(float(x), 3) for x in flips[key][:, b, h]],
                    loc={
                        v: round(float(loc_mean[:, vi[v], b, h].mean()), 3)
                        for v in VARIANTS
                    },
                    content_T_bbox=float(content[:, :, vi["bbox"], b, h, 0].mean()),
                )
            )
        return rows

    coord_ratio = {
        k: np.log(np.nanmean(coord[:, :, vi[k], ..., 0], (0, 1)) + EPS)
        - np.log(np.nanmean(coord[:, :, vi[k], ..., 2], (0, 1)) + EPS)
        for k in ("bbox", "pos")
    }  # (L, H): log mass on location tokens from T queries over R queries
    coord_mass = {
        k: np.nanmean(coord[:, :, vi[k]] @ np.array([0, 0, 1.0]), (0, 1))
        for k in ("bbox", "pos")
    }  # (L, H): R-query mass on location tokens (the "everyone reads it" heads)

    def top_coord(k):
        r = coord_ratio[k]
        order = np.argsort(r, axis=None)[::-1][: args.top]
        return [
            dict(
                block=int(f // heads),
                head=int(f % heads),
                log_T_over_R=round(float(r.flat[f]), 3),
                mass_R=round(float(coord_mass[k].flat[f]), 5),
            )  # fmt: skip
            for f in order
        ]

    hd = {}
    for v in VARIANTS[1:]:
        m = np.nanmean(hdiff[:, :, vi[v]], (0, 1))  # (L, 3)
        hd[v] = ((m[:, 0] + m[:, 1]) / 2, m[:, 2])
    hdiff_table = {
        v: [round(float(x), 4) for x in tm / np.maximum(r, EPS)]
        for v, (tm, r) in hd.items()
    }
    hdiff_abs = {
        v: {
            reg: [
                round(float(x), 4) for x in np.nanmean(hdiff[:, :, vi[v], :, i], (0, 1))
            ]
            for i, reg in enumerate(REGIONS)
        }  # fmt: skip
        for v in VARIANTS[1:]
    }
    loss_table = {
        f"sigma_{s}": {
            v: {
                reg: round(float(np.nanmean(loss[:, zi, vi[v], i])), 5)
                for i, reg in enumerate(REGIONS)
            }  # fmt: skip
            for v in VARIANTS
        }
        for zi, s in enumerate(sigmas)
    }
    blocks_follow = {k: follow[k].max(-1) for k in follow}  # (Z, L)

    run_dir = make_run_dir("moss_bbox_grounding", args.label)
    np.savez_compressed(
        run_dir / "per_sample.npz",
        content=content, coord=coord, hdiff=hdiff, loss=loss,
        stems=np.array([j["stem"] for j in jobs]), sigmas=np.array(sigmas),
    )  # fmt: skip
    np.savez_compressed(run_dir / "maps.npz", **maps)
    (run_dir / "samples.json").write_text(
        json.dumps(
            [
                dict(
                    stem=j["stem"],
                    height=j["height"],
                    width=j["width"],
                    moved=j["moved"],
                    caps=j["caps"],
                    spans={v: s[:2] for v, s in j["spans"].items()},
                )  # fmt: skip
                for j in jobs
            ],
            ensure_ascii=False,
            indent=1,
        )
    )
    plot(run_dir / "probe.png", follow, hd, blocks_follow)
    metrics = dict(
        n_samples=S,
        sigmas=sigmas,
        top_follow_bbox=top_heads("bbox"),
        top_follow_pos=top_heads("pos"),
        block_max_follow={
            k: [[round(float(x), 3) for x in row] for row in blocks_follow[k]]
            for k in blocks_follow
        },
        top_coord_readers_bbox=top_coord("bbox"),
        top_coord_readers_pos=top_coord("pos"),
        hdiff_TM_over_R=hdiff_table,
        hdiff_abs=hdiff_abs,
        loss=loss_table,
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["per_sample.npz", "maps.npz", "samples.json", "probe.png"],
        device=device,
    )
    print(f"[probe] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
