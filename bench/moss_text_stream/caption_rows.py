#!/usr/bin/env python3
"""Does the image read back what a real colorize caption's rows picked up from the page?

``probe.py`` asked this of the empty caption's template rows. Here the caption is a manga
page's control caption (``comic``, a box-only text item per OCR line, a box per character
with its hair / eye / clothing colours): the longest of the cache's draws that carries a
colour. Teacher-forced, colorize v5, isolated reference, ``RefCache``, generation's DiT
(int8 block linears, fp32 stream).

Per page A, σ and noise draw, the text rows fed into the blocks of A's forward are replaced:

  own          A's own rows (control: no change)
  jitter       own × (1 + 1e-3·ε): the floor under the arms below
  swap_seed    A's caption on A under another noise draw
  swap_page    A's caption run on another page B (same tokens, the page-derived part
               from B): all rows, the caption rows alone, the template rows alone, and
               all rows in blocks 1–13 / 14–25 / 26–41
  jitter_norm  own + an isotropic draw with swap_page's row norms
  bare         the bare caption in place of A's (scale: what the caption's content does)

``dv``: mean squared difference of the velocity from A's live forward, over the page and
over the caption's character boxes. Also per block: the share of the rows' energy that
differs between A-on-A and A-on-B, by row class.

Read, fixed before the run: the page-derived part of the caption rows is read back when
swap_page is above both jitter and jitter_norm; a swap at the floor, or under an isotropic
draw of its norm, says the rows are the caption's content on a per-block trajectory, as
the template rows were.

    make daemon-run ARGS="--queue --stall-timeout 600 bench/moss_text_stream/caption_rows.py"
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from safetensors import safe_open

from bench._common import make_run_dir, start_heartbeat, write_result
from bench.moss_text_stream.probe import RANGES, Stream, load_dit

ARMS = (
    "own",
    "jitter",
    "swap_seed",
    "swap_page",
    "swap_page_caption",
    "swap_page_template",
    *(f"swap_page_{r}" for r in RANGES),
    "jitter_norm",
    "bare",
)


def longest_colour_draw(te_path):
    """``(key, caption)`` of the draw with the most rows among those whose caption
    names a colour item, or ``None``."""
    best = None
    with safe_open(te_path, "pt") as f:
        meta = f.metadata()
        for key in f.keys():
            if key == "embeds_plain" or not key.startswith("embeds"):
                continue
            cap = meta.get("caption" + key[len("embeds") :], "")
            if '"label":"character"' not in cap:
                continue
            rows = f.get_slice(key).get_shape()[0]
            if best is None or rows > best[0]:
                best = (rows, key, cap)
    return None if best is None else best[1:]


def character_boxes(caption, height, width):
    """Latent-cell mask ``(height·width,)`` of the caption's character boxes
    (``abs_centralized`` ``yxyx`` pixels, origin at the page centre)."""
    mask = torch.zeros(height, width, dtype=torch.bool)
    for item in json.loads(caption).get("items", []):
        if item.get("label") != "character":
            continue
        y0, x0, y1, x1 = item["bbox"]
        r0, r1 = (int(y + 8 * height) // 16 for y in (y0, y1))
        c0, c1 = (int(x + 8 * width) // 16 for x in (x0, x1))
        mask[max(r0, 0) : r1 + 1, max(c0, 0) : c1 + 1] = True
    return mask.flatten()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--lora", default="output/ckpt_moss_i2i/moss_colorize_v5.safetensors"
    )
    ap.add_argument(
        "--cache", default="post_image_dataset/moss/i2i_colorize_probe_manga"
    )
    ap.add_argument("--pages", type=int, default=24)
    ap.add_argument(
        "--base_t2i",
        action="store_true",
        help="Base without the LoRA and without the reference, on the same pages "
        "and captions: is the read-back Base's own",
    )
    ap.add_argument("--sigmas", default="0.95,0.6,0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--base", default="int8", choices=("int8", "bf16"))
    ap.add_argument("--stream", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--model_dir", default="")
    ap.add_argument(
        "--massive", default="library/moss/calibration/massive_channels.npz"
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    start_heartbeat(label="moss_text_stream")

    import library.moss.model as moss_model
    from bench.moss_sinks.probe_attention import template_split
    from library.env import resolve_under_home
    from library.moss.caching.items import list_items, load_ref
    from library.moss.loader import load_tokenizer
    from library.moss.requests import resolve_model_dir
    from safetensors.torch import load_file

    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    pages = []
    for it in list_items(resolve_under_home(args.cache)):
        draw = longest_colour_draw(it.te_path) if it.ref_path else None
        if draw is None:
            continue
        te = load_file(it.te_path)
        latent = load_file(it.latent_path)["latent"].float()
        ref = load_ref(it)
        if ref.shape != latent.shape:
            continue
        boxes = character_boxes(draw[1], *latent.shape[-2:])
        if not boxes.any():
            continue
        pages.append(
            dict(
                stem=it.stem,
                latent=latent[None],
                ref=ref[None],
                embeds=te[draw[0]],
                plain=te["embeds_plain"],
                caption=draw[1],
                boxes=boxes,
            )
        )
        if len(pages) >= args.pages:
            break
    tok = load_tokenizer(model_dir)
    pre, _, suf = template_split(tok, "comic")
    del tok
    rows_n = [p["embeds"].shape[0] for p in pages]
    print(
        f"[caption] {len(pages)} pages, rows {min(rows_n)}–{max(rows_n)} "
        f"(median {int(np.median(rows_n))}), template {pre}+{suf}; e.g. "
        f"{pages[0]['caption'][:300]}",
        flush=True,
    )

    transformer = load_dit(
        model_dir,
        args.model,
        None if args.base_t2i else resolve_under_home(args.lora),
        device,
        args.base,
        args.stream,
    )
    transformer.ref_cache = moss_model.RefCache()
    L, heads = len(transformer.layers), transformer.config.num_heads
    keep = torch.ones(transformer.config.dim)
    keep[torch.from_numpy(np.load(resolve_under_home(args.massive))["channels"])] = 0
    stream = Stream(L, heads, keep)
    handle = transformer.layers[0].register_forward_pre_hook(
        lambda _m, a: stream.first_input(a[0])
    )
    transformer.layer_hook = stream

    P, Z = len(pages), len(sigmas)
    dv = np.full((P, Z, len(ARMS), 2), np.nan, np.float32)  # page, boxes
    mse = np.full((P, Z, len(ARMS) + 1, 2), np.nan, np.float32)  # live, arms…
    share = np.full((P, Z, 2, 3, L), np.nan, np.float32)  # page|seed × all|cap|tmpl

    def noisy(page, pi, zi, offset):
        gen = torch.Generator(device="cpu").manual_seed(
            args.seed + offset + 1000 * pi + zi
        )
        latent = page["latent"].to(device)
        noise = torch.randn(latent.shape, generator=gen).to(device)
        return (1.0 - sigmas[zi]) * latent + sigmas[zi] * noise, noise - latent

    def run(x_t, sigma, embeds, ref):
        t = torch.tensor([1000.0 * sigma], device=device)
        emb = embeds[None].to(device, torch.bfloat16)
        return transformer(x_t, t, emb, return_dict=False, ref=ref)[0].float()

    def ref_of(page):
        return None if args.base_t2i else page["ref"].to(device)

    def rows_of(page, pi, zi, offset, embeds):
        x_t, target = noisy(page, pi, zi, offset)
        n_img = page["latent"].shape[-2] * page["latent"].shape[-1]
        stream.start(n_img, embeds.shape[0])
        v = run(x_t, sigmas[zi], embeds, ref_of(page))
        stream.active = False
        return torch.stack(stream.rows), v, target, x_t

    def two(err, boxes):
        e = err.mean(1)[0].flatten()
        return np.array([e.mean().item(), e[boxes].mean().item()])

    try:
        with torch.inference_mode():
            for pi, page in enumerate(pages):
                emb, S = page["embeds"], page["embeds"].shape[0]
                other = pages[(pi + 1) % P]
                boxes = page["boxes"].to(device)
                n_img = page["latent"].shape[-2] * page["latent"].shape[-1]
                ref = ref_of(page)
                cls = torch.zeros(S, dtype=torch.bool)
                cls[pre : S - suf] = True  # caption rows
                for zi, sigma in enumerate(sigmas):
                    own, v_live, target, x_t = rows_of(page, pi, zi, 0, emb)
                    seed, *_ = rows_of(page, pi, zi, 500_000, emb)
                    donor, *_ = rows_of(other, (pi + 1) % P, zi, 0, emb)
                    mse[pi, zi, 0] = two((v_live - target).pow(2), boxes)
                    k = keep
                    for di, d in enumerate((donor, seed)):
                        diff = ((own - d) * k).pow(2).sum(-1)[1:]  # (L, S)
                        tot = (own * k).pow(2).sum(-1)[1:]
                        for ci, sel in enumerate((slice(None), cls, ~cls)):
                            share[pi, zi, di, ci] = (
                                diff[:, sel].sum(-1) / tot[:, sel].sum(-1)
                            ).numpy()
                    gen = torch.Generator().manual_seed(args.seed + 7 * pi + zi)
                    eps = torch.randn(own.shape, generator=gen)
                    iso = torch.randn(own.shape, generator=gen)
                    iso = (
                        iso
                        * (donor - own).norm(dim=-1, keepdim=True)
                        / iso.norm(dim=-1, keepdim=True)
                    )
                    mixed_cap = torch.where(cls[None, :, None], donor, own)
                    mixed_tmpl = torch.where(cls[None, :, None], own, donor)
                    feeds = {
                        "own": own,
                        "jitter": own * (1 + 1e-3 * eps),
                        "swap_seed": seed,
                        "swap_page": donor,
                        "swap_page_caption": mixed_cap,
                        "swap_page_template": mixed_tmpl,
                        "jitter_norm": own + iso,
                    }
                    for name, blocks in RANGES.items():
                        feeds[f"swap_page_{name}"] = torch.stack(
                            [donor[j] if j in blocks else own[j] for j in range(L + 1)]
                        )
                    for ai, arm in enumerate(ARMS):
                        if arm == "bare":
                            stream.active = False
                            v = run(x_t, sigma, page["plain"], ref)
                        else:
                            feed = [None, *feeds[arm][1:L]]
                            stream.start(n_img, S, feed=feed, record=False)
                            v = run(x_t, sigma, emb, ref)
                            stream.active = False
                        dv[pi, zi, ai] = two((v - v_live).pow(2), boxes)
                        mse[pi, zi, 1 + ai] = two((v - target).pow(2), boxes)
                print(
                    f"[caption] {pi + 1}/{P} {page['stem']} rows {S} "
                    f"dv σ{sigmas[0]} {dv[pi, 0, :, 0].round(5)}",
                    flush=True,
                )
    finally:
        transformer.layer_hook = None
        transformer.ref_cache = None
        handle.remove()

    def r(a, n=5):
        return round(float(a), n)

    summary = {"pages": P, "rows_median": int(np.median(rows_n))}
    for zi, sigma in enumerate(sigmas):
        row = {
            "flow_mse_live": {"page": r(mse[:, zi, 0, 0].mean(), 4),
                              "boxes": r(mse[:, zi, 0, 1].mean(), 4)},
        }  # fmt: skip
        for ai, arm in enumerate(ARMS):
            row[arm] = {
                "dv_page": r(dv[:, zi, ai, 0].mean()),
                "dv_boxes": r(dv[:, zi, ai, 1].mean()),
                "dv_page_median": r(np.median(dv[:, zi, ai, 0])),
                "flow_mse_page": r(mse[:, zi, 1 + ai, 0].mean(), 4),
            }
        a = {n: dv[:, zi, ARMS.index(n), 0] for n in ARMS}
        row["pages_swap_page_above_jitter_norm"] = int(
            (a["swap_page"] > a["jitter_norm"]).sum()
        )
        row["pages_swap_page_above_swap_seed"] = int(
            (a["swap_page"] > a["swap_seed"]).sum()
        )
        sel = [0, 6, 13, 20, 25, 30, 35, 40]
        row["blocks"] = sel
        for di, dname in enumerate(("page", "seed")):
            for ci, cname in enumerate(("all", "caption", "template")):
                row[f"{dname}_share_{cname}"] = [
                    r(np.nanmean(share[:, zi, di, ci, b]), 4) for b in sel
                ]
        summary[f"sigma_{sigma}"] = row
        print(f"[caption] σ {sigma} " + json.dumps(row), flush=True)

    default_label = "caption_rows_base_t2i" if args.base_t2i else "caption_rows"
    run_dir = make_run_dir("moss_text_stream", args.label or default_label)
    np.savez_compressed(
        run_dir / "per_page.npz", dv=dv, mse=mse, share=share,
        sigmas=np.array(sigmas), arms=np.array(ARMS),
        stems=np.array([p["stem"] for p in pages]), rows=np.array(rows_n),
    )  # fmt: skip
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=summary,
        label=args.label,
        artifacts=["per_page.npz"],
        device=device,
    )
    print(f"[caption] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
