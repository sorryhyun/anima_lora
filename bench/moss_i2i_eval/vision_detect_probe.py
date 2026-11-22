"""Do Qwen3.5's image rows mark the cells sanitize has to edit?

TE-only probe for `ref_vision` on sanitize (no DiT). Each staged pair's texted page goes
through the vision tower; a linear probe on the image rows predicts, per 32 px cell,
whether the clean twin differs there. Folds are grouped by artist, so a recurring
watermark is never on both sides.

Arms (image rows only; rows per page are logged, they are the cost):
  img         image alone at the page's pixel count (= the rows of the shipped image → caption order)
  cap_img     the sanitize caption, then the image
  neu_img     a neutral caption, then the image
  img_lo      image alone at a quarter of the pixels
  cap_img_lo  the sanitize caption, then the image, at a quarter of the pixels
  vae         the cached reference latent, 2×2 cells concatenated (what the DiT already holds,
              read linearly: a floor, the DiT reads it through 42 blocks)

Read, fixed before the run:
  - worth a `ref_vision` micro arm: the rows fire on the cells v3 left behind on the five eval
    pages (heat maps beside v3's renders, by eye) and beat `vae` on held-out AP and faint recall;
  - order matters at a fixed caption: `cap_img` − `img` larger than the fold spread;
  - the caption's content matters: `cap_img` − `neu_img` larger than the fold spread;
  - a quarter of the rows is enough: `img_lo` within the fold spread of `img`.

Labels: mean |ΔRGB| per pixel at the cached page size; a pixel changed above 8/255; a cell
positive at ≥ 5 % changed pixels, ignored between 0.5 % and 5 %. A positive cell is faint
when its changed pixels average under 24/255. Recall is at 1 % false positives.

    make daemon-run ARGS="--queue --stall-timeout 600 bench/moss_i2i_eval/vision_detect_probe.py"
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from library.env import resolve_under_home

CAPTION = "no sfx, no dialogue, no watermark"
NEUTRAL = "an illustration, a picture, an image"
PIXEL_T, POS_FRAC, IGN_FRAC, FAINT_T = 8.0, 0.05, 0.005, 24.0
HI_ARMS = ("img", "cap_img", "neu_img")
LO_ARMS = ("img_lo", "cap_img_lo")


def te_inputs(tok, proc, image, pre, device):
    """The pipeline's chat call with ``pre`` (may be empty) ahead of the image item."""
    content = ([{"type": "text", "text": pre}] if pre else []) + [{"type": "image"}]
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return proc(text=[formatted], images=[image], return_tensors="pt").to(
        device
    ), formatted


@torch.no_grad()
def image_rows(tok, te, proc, image, pre, device):
    """``(rows (gh, gw, 4096) fp16 cpu, total row count)``."""
    inputs, _ = te_inputs(tok, proc, image, pre, device)
    hidden = te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
    is_image = inputs["mm_token_type_ids"][0] == 1
    _, gh, gw = inputs["image_grid_thw"][0].tolist()
    rows = hidden[is_image].reshape(gh // 2, gw // 2, -1)
    return rows.to("cpu", torch.float16), hidden.shape[0]


def cell_labels(diff: torch.Tensor, gh: int, gw: int):
    """``diff (H, W)`` → per cell ``label`` (1 / 0 / -1 ignored) and ``faint`` flag."""
    changed = (diff > PIXEL_T).float()[None, None]
    frac = F.adaptive_avg_pool2d(changed, (gh, gw))[0, 0]
    mass = F.adaptive_avg_pool2d(diff[None, None] * changed, (gh, gw))[0, 0]
    strength = mass / frac.clamp_min(1e-6)
    label = torch.full_like(frac, -1.0)
    label[frac >= POS_FRAC] = 1.0
    label[frac < IGN_FRAC] = 0.0
    return label, (label == 1) & (strength < FAINT_T)


def pick_pairs(root: Path, per_artist: int, n_identity: int):
    rows = [json.loads(line) for line in (root / "pairs.jsonl").open()]
    by_artist = defaultdict(list)
    for r in rows:
        by_artist[r["cond"].split("/")[0]].append(r)
    picked = []
    for artist, rs in sorted(by_artist.items()):
        cap = n_identity if artist == "_identical" else per_artist
        picked += [dict(r, artist=artist) for r in rs[:cap]]
    return picked


def fit_probe(x, y, device, epochs=20, seed=0):
    """Standardised logistic regression; returns ``score(x)``."""
    torch.manual_seed(seed)
    x = x.to(device)
    mean = x.float().mean(0)
    std = x.float().std(0).clamp_min(1e-4)
    w = torch.zeros(x.shape[1], device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=1e-3, weight_decay=1e-3)
    y = y.to(device)
    n = x.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n, device=device).split(8192):
            xb = (x[idx].float() - mean) / std
            loss = F.binary_cross_entropy_with_logits(xb @ w + b, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    w, b = w.detach(), b.detach()

    def score(z):
        out = []
        for chunk in z.split(65536):
            out.append((((chunk.to(device).float() - mean) / std) @ w + b).cpu())
        return torch.cat(out)

    return score


def metrics(score, label, faint):
    """AP, AUROC and recall at 1 % FPR (all / faint / strong positives)."""
    keep = label >= 0
    s, y, f = score[keep].numpy(), label[keep].numpy() > 0.5, faint[keep].numpy()
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    ap = float((tp / np.arange(1, len(ys) + 1))[ys].mean())
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    auroc = float(
        (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    )
    thr = np.quantile(s[~y], 0.99)
    out = {"ap": ap, "auroc": auroc, "recall": float((s[y] > thr).mean())}
    if (y & f).any():
        out["recall_faint"] = float((s[y & f] > thr).mean())
    if (y & ~f).any():
        out["recall_strong"] = float((s[y & ~f] > thr).mean())
    return out


def to_grid(score_map: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    if score_map.shape == (gh, gw):
        return score_map
    return F.interpolate(score_map[None, None].float(), (gh, gw), mode="nearest")[0, 0]


def overlay(page, heat: torch.Tensor, path: Path):
    from PIL import Image

    h = torch.sigmoid(heat.float()).numpy()
    h = (
        np.asarray(
            Image.fromarray((h * 255).astype(np.uint8)).resize(
                page.size, Image.NEAREST
            ),
            dtype=np.float32,
        )[..., None]
        / 255.0
    )
    base = np.asarray(page.convert("L").convert("RGB"), dtype=np.float32)
    red = np.array([255.0, 30.0, 30.0])
    out = base * (1 - 0.65 * h) + red * (0.65 * h)
    Image.fromarray(out.astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staging", default="post_image_dataset/i2i/sanitize/staging")
    ap.add_argument("--cache", default="post_image_dataset/moss/i2i_sanitize_idt")
    ap.add_argument("--eval", default="output/moss_i2i_eval/sanitize/inputs")
    ap.add_argument("--out", default="output/tests/vision_detect_probe")
    ap.add_argument("--per_artist", type=int, default=25)
    ap.add_argument("--n_identity", type=int, default=15)
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from PIL import Image

    from library.moss.caching.images import fit, load_rgb, target_size
    from library.moss.caching.items import list_items, load_ref
    from library.moss.loader import empty_cache, load_tokenizer
    from library.moss.vision_te import load_te, processor

    Image.MAX_IMAGE_PIXELS = None
    staging = resolve_under_home(args.staging)
    out = resolve_under_home(args.out) / time.strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    items = {it.stem: it for it in list_items(resolve_under_home(args.cache))}
    pairs = [
        p
        for p in pick_pairs(staging, args.per_artist, args.n_identity)
        if p["stem"] in items
        and items[p["stem"]].ref_path is not None
        and (staging / p["cond"]).is_file()
        and (staging / p["target"]).is_file()
    ]
    print(f"[probe] {len(pairs)} pairs, {len({p['artist'] for p in pairs})} artists")

    tok = load_tokenizer(None)
    enc = load_te(tok, device, precision=args.text_encoder)
    procs: dict[int, object] = {}

    def proc(max_pixels):
        if max_pixels not in procs:
            procs[max_pixels] = processor(tok, max_pixels)
        return procs[max_pixels]

    def encode_arms(page, arms):
        w, h = page.size
        feats, counts = {}, {}
        for arm in arms:
            pre = {"img": "", "cap": CAPTION, "neu": NEUTRAL}[arm.split("_")[0]]
            px = w * h // 4 if arm.endswith("_lo") else w * h
            feats[arm], counts[arm] = image_rows(tok, enc, proc(px), page, pre, device)
        return feats, counts

    feats = defaultdict(list)  # arm → [(n_cells, D)]
    labels = defaultdict(list)  # arm → [(n_cells,)] at the arm's own grid
    grids = defaultdict(list)  # arm → [(gh, gw)]
    hi_labels, hi_faint, groups, n_rows = [], [], [], defaultdict(list)
    t0 = time.time()
    for i, p in enumerate(pairs):
        it = items[p["stem"]]
        w, h = it.width, it.height
        page = fit(load_rgb(staging / p["cond"]), w, h)
        clean = fit(load_rgb(staging / p["target"]), w, h)
        diff = torch.from_numpy(
            np.abs(
                np.asarray(page, dtype=np.float32) - np.asarray(clean, dtype=np.float32)
            ).mean(-1)
        )
        f, counts = encode_arms(page, HI_ARMS + LO_ARMS)
        if i == 0:
            print(
                "[probe] cap_img prompt:",
                repr(te_inputs(tok, proc(w * h), page, CAPTION, device)[1][:160]),
            )
        gh, gw = f["img"].shape[:2]
        lab, faint = cell_labels(diff, gh, gw)
        hi_labels.append(lab.flatten())
        hi_faint.append(faint.flatten())
        groups.append(p["artist"])
        ref = load_ref(it)  # (128, h/16, w/16)
        rh, rw = ref.shape[1] // 2 * 2, ref.shape[2] // 2 * 2
        vae = F.pixel_unshuffle(ref[None, :, :rh, :rw], 2)
        vae = F.interpolate(vae, (gh, gw), mode="nearest")[0].permute(1, 2, 0)
        f["vae"] = vae.to(torch.float16)
        for arm, x in f.items():
            g = x.shape[:2]
            feats[arm].append(x.reshape(-1, x.shape[-1]))
            labels[arm].append(cell_labels(diff, *g)[0].flatten())
            grids[arm].append(g)
            if arm in counts:
                n_rows[arm].append((g[0] * g[1], counts[arm]))
        if i % 20 == 0:
            print(
                f"[probe] {i}/{len(pairs)} {p['stem']} {w}x{h} grid {gh}x{gw} "
                f"pos {int((lab == 1).sum())} faint {int(faint.sum())} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    eval_pages = {}
    for path in sorted(resolve_under_home(args.eval).glob("*.png")):
        img = load_rgb(path)
        page = fit(img, *target_size(*img.size, 1024))
        f, _ = encode_arms(page, HI_ARMS + LO_ARMS)
        eval_pages[path.stem] = (page, f)
        print(f"[probe] eval {path.stem} {page.size}", flush=True)
    enc = None  # frees the TE; encode_arms closes over it
    empty_cache()

    artists = sorted(set(groups), key=lambda a: -groups.count(a))
    fold_of = {a: k % args.folds for k, a in enumerate(artists)}
    page_fold = torch.tensor([fold_of[a] for a in groups])
    result = {
        "pairs": len(pairs),
        "cells": int(sum(len(x) for x in hi_labels)),
        "positive_cells": int(sum((x == 1).sum() for x in hi_labels)),
        "faint_cells": int(sum(x.sum() for x in hi_faint)),
        "rows": {
            arm: {
                "image": float(np.mean([a for a, _ in v])),
                "total": float(np.mean([b for _, b in v])),
            }
            for arm, v in n_rows.items()
        },
        "arms": {},
    }
    hi_grids = grids["img"]
    for arm in (*HI_ARMS, *LO_ARMS, "vae"):
        x_pages, y_pages = feats[arm], labels[arm]
        per_fold = []
        for k in range(args.folds):
            tr = [j for j in range(len(pairs)) if page_fold[j] != k]
            te_idx = [j for j in range(len(pairs)) if page_fold[j] == k]
            x = torch.cat([x_pages[j] for j in tr])
            y = torch.cat([y_pages[j] for j in tr])
            score = fit_probe(x[y >= 0], y[y >= 0], device)
            s = torch.cat(
                [
                    to_grid(
                        score(x_pages[j]).reshape(grids[arm][j]), *hi_grids[j]
                    ).flatten()
                    for j in te_idx
                ]
            )
            m = metrics(
                s,
                torch.cat([hi_labels[j] for j in te_idx]),
                torch.cat([hi_faint[j] for j in te_idx]),
            )
            if m:
                per_fold.append(m)
            del x, y
            torch.cuda.empty_cache()
        keys = sorted({k for m in per_fold for k in m})
        summary = {
            k: [
                float(np.mean([m[k] for m in per_fold if k in m])),
                float(np.std([m[k] for m in per_fold if k in m])),
            ]
            for k in keys
        }
        result["arms"][arm] = {"folds": per_fold, "mean_std": summary}
        print(
            f"[probe] {arm:11s} "
            + "  ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in summary.items()),
            flush=True,
        )
        if arm == "vae":
            continue
        x = torch.cat(x_pages)
        y = torch.cat(y_pages)
        score = fit_probe(x[y >= 0], y[y >= 0], device)
        for name, (page, f) in eval_pages.items():
            g = f[arm].shape[:2]
            heat = score(f[arm].reshape(-1, f[arm].shape[-1])).reshape(g)
            torch.save(heat, out / f"{name}_{arm}.pt")
            overlay(page, heat, out / f"{name}_{arm}.png")
        del x, y
        torch.cuda.empty_cache()
    for name, (page, _) in eval_pages.items():
        page.save(out / f"{name}_page.png")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
