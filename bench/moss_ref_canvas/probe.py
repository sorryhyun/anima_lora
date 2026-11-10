#!/usr/bin/env python3
"""Does MOSS-Image Base read a reference panel beside the image?

Teacher-forced, no sampling, no LoRA: the premise of
``docs/moss/proposal/ref_canvas.md``. character_ref pairs whose reference has the
target's grid (the two panels then make one 2W-wide picture), fixed noise per pair,
flow-matching MSE over the image panel. Each reference condition runs with the
pair's own reference and with another artist's (``_shuf``, same grid); the read is
own − shuf, negative = the matching reference lowers the loss.

  none             the target alone, Base coordinates
  none_capshuf     the other pair's caption: the scale a wrong caption moves the loss by
  none_canvas      the target alone at its left-panel coordinates (the re-centre only)
  none_canvas_box  the same under the panel-box caption
  joint            reference at r = 0 beside the image, noised at the step's σ, its rows
                   at the step's timestep, attention open both ways: one wide picture
  joint_box        the same under the panel-box caption
  open             the reference clean at ``ref_timestep 0``, attention open
  iso              clean and isolated (``ref_isolate``): the i2i recipe on canvas coordinates
  iso_box          the same under the panel-box caption
  offset           clean and isolated on r = 1, ``ref_pos offset``: what character_ref ran
  iso_s3/6/9       isolated on canvas coordinates, the reference at a fixed σ 0.3 / 0.6 /
                   0.9 with ``ref_timestep`` to match: what ``RefCache`` can hold. Where
                   the fixed σ is the step's, ``joint`` with the isolation alone added

The panel-box caption adds two ``panel`` items to the cached layout, each with its box
on the canvas (``abs_centralized`` ``yxyx``): the target's caption on the left, a fixed
caption on the right.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_ref_canvas/probe.py"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from bench._common import make_run_dir, start_heartbeat, write_result

REF_CAPTION = "reference image of the character"

# name: (coordinates, caption, reference σ, ref_timestep, ref_isolate)
#   coordinates   base | shift | canvas | offset
#   caption       plain | other | box
#   reference σ   None = no reference | "step" = the step's σ | a fixed σ (0 = clean)
#   ref_timestep  None = the step's | 1000·σ of the reference rows
CONDITIONS = {
    "none": ("base", "plain", None, None, False),
    "none_capshuf": ("base", "other", None, None, False),
    "none_canvas": ("shift", "plain", None, None, False),
    "none_canvas_box": ("shift", "box", None, None, False),
    "joint": ("canvas", "plain", "step", None, False),
    "joint_box": ("canvas", "box", "step", None, False),
    "open": ("canvas", "plain", 0.0, 0.0, False),
    "iso": ("canvas", "plain", 0.0, 0.0, True),
    "iso_box": ("canvas", "box", 0.0, 0.0, True),
    "offset": ("offset", "plain", 0.0, 0.0, True),
    "iso_s3": ("canvas", "plain", 0.3, 300.0, True),
    "iso_s6": ("canvas", "plain", 0.6, 600.0, True),
    "iso_s9": ("canvas", "plain", 0.9, 900.0, True),
}


def dumps(layout):
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def panel_caption(caption, target_px, ref_px):
    """The cached layout with one ``panel`` item per panel, boxes on the canvas."""
    from anime_tools.captions.layout_json import centered_box

    (w, h), (rw, rh) = target_px, ref_px
    size = (w + rw, max(h, rh))
    top, rtop = (size[1] - h) / 2, (size[1] - rh) / 2
    layout = json.loads(caption)
    layout["items"] = [
        dict(
            label="panel",
            caption=layout["summary"]["caption"],
            bbox=centered_box((0, top, w, top + h), size),
        ),
        dict(
            label="panel",
            caption=REF_CAPTION,
            bbox=centered_box((w, rtop, w + rw, rtop + rh), size),
        ),
    ]
    layout["bbox_coord_space"] = "abs_centralized"
    layout["bbox_format"] = "yxyx"
    return dumps(layout)


def canvas_rope(bands, height, width, text_length, panel, with_ref, device):
    """``_RoPE.forward`` with every token at r = 0 on a two-panel canvas: the image on
    the left, the ``panel = (rh, rw)`` reference grid at unit spacing on its right,
    both centred on the canvas centre (where the text sits). ``with_ref`` False leaves
    the reference rows out: the image at its left-panel coordinates alone."""
    rh, rw = panel
    centre = (width + rw - 1) / 2

    def grid(gh, gw, left):
        h, w = torch.meshgrid(
            torch.arange(gh, device=device, dtype=torch.float32) - (gh - 1) / 2,
            torch.arange(gw, device=device, dtype=torch.float32) + left - centre,
            indexing="ij",
        )
        return torch.stack(
            (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
        ).flatten(0, 1)

    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    parts = [grid(height, width, 0), text]
    if with_ref:
        parts.append(grid(rh, rw, width))
    phase = torch.cat(parts, 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def load_pairs(cache_dir, n):
    """Up to ``n`` pairs whose reference has the target's grid and whose caption has
    no items, one per artist per round; each with ``other``, a same-grid pair by
    another artist."""
    from library.moss.caching.items import list_items

    pool = []
    for it in list_items(cache_dir):
        if it.ref_path is None:
            continue
        with safe_open(it.latent_path, "pt") as f:
            grid = tuple(f.get_slice("latent").get_shape()[1:])
        with safe_open(it.ref_path, "pt") as f:
            ref_grid = tuple(f.get_slice("latent").get_shape()[1:])
        with safe_open(it.te_path, "pt") as f:
            caption = f.metadata()["caption"]
        if grid != ref_grid or json.loads(caption).get("items"):
            continue
        pool.append(
            dict(item=it, grid=grid, caption=caption, artist=it.stem.split("__")[0])
        )
    groups = defaultdict(list)
    for p in pool:
        groups[p["grid"]].append(p)
    for g in groups.values():
        for i, p in enumerate(g):
            ring = (g[(i + k) % len(g)] for k in range(1, len(g)))
            p["other"] = next((q for q in ring if q["artist"] != p["artist"]), None)
    by_artist = defaultdict(list)
    for p in pool:
        if p["other"] is not None:
            by_artist[p["artist"]].append(p)
    picked, depth = [], 0
    while len(picked) < n and any(len(v) > depth for v in by_artist.values()):
        picked += [v[depth] for _, v in sorted(by_artist.items()) if len(v) > depth]
        depth += 1
    return picked[:n], len(pool)


def boot(values, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache_dir", default="post_image_dataset/moss/i2i_character_ref")
    ap.add_argument("--n", type=int, default=24, help="pairs")
    ap.add_argument("--sigmas", default="0.9,0.6,0.3")
    ap.add_argument(
        "--conditions", default="", help="comma-separated subset (default: all)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument(
        "--int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the generation default: int8 block linears, fp32 stream; --no-int8 = bf16",
    )
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    start_heartbeat(label="moss_ref_canvas")

    from library.env import resolve_under_home
    from library.moss.caching.items import load_item, load_ref
    from library.moss.generate import cached_embeds
    from library.moss.loader import compile_blocks, load_transformer, stream_fp32_
    from library.moss.requests import GenerateRequest, resolve_model_dir

    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    sigmas = [float(s) for s in args.sigmas.split(",")]
    pairs, pool = load_pairs(resolve_under_home(args.cache_dir), args.n)
    for p in pairs:
        it = p["item"]
        p["latent"], p["embeds"] = load_item(it)
        p["ref"] = load_ref(it)
        p["ref_other"] = load_ref(p["other"]["item"])
        p["embeds_other"] = load_item(p["other"]["item"])[1]
        size = (it.width, it.height)
        p["box_caption"] = panel_caption(p["caption"], size, size)
    box = cached_embeds(
        model_dir,
        [p["box_caption"] for p in pairs],
        Path(args.te_cache),
        precision=args.text_encoder,
    )
    chosen = [c.strip() for c in args.conditions.split(",") if c.strip()]
    conditions = {n: CONDITIONS[n] for n in chosen or CONDITIONS}
    ref_conditions = [n for n, c in conditions.items() if c[2] is not None]
    names = list(conditions) + [f"{n}_shuf" for n in ref_conditions]
    print(f"[canvas] {len(pairs)} of {pool} same-grid pairs, "
          f"{len({p['artist'] for p in pairs})} artists, {len(names)} conditions, "
          f"σ {sigmas}", flush=True)  # fmt: skip

    transformer = load_transformer(model_dir, args.model, device=device)
    if args.int8:
        from library.moss.dit_int8.base import generate_base

        generate_base(GenerateRequest(model=args.model))(transformer)
        stream_fp32_(transformer)
    if args.compile:
        compile_blocks(transformer.layers)
    rope = transformer.rope
    stock = rope.forward
    state = {"coords": "base", "panel": None}

    def rope_forward(height, width, text_length, device, ref_hw=None, ref_pos="aligned"):  # fmt: skip
        if state["coords"] in ("base", "offset"):
            return stock(height, width, text_length, device, ref_hw, ref_pos)
        return canvas_rope(
            rope._default_bands(device),
            height,
            width,
            text_length,
            state["panel"],
            ref_hw is not None,
            device,
        )

    rope.forward = rope_forward
    loss = {n: np.full((len(sigmas), len(pairs)), np.nan) for n in names}
    with torch.no_grad():
        for pi, p in enumerate(pairs):
            latent = p["latent"][None].to(device)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + pi)
            noise = torch.randn(latent.shape, generator=gen).to(device)
            ref_noise = torch.randn(latent.shape, generator=gen).to(device)
            refs = {
                "": p["ref"][None].to(device),
                "_shuf": p["ref_other"][None].to(device),
            }
            embeds = {
                "plain": p["embeds"],
                "other": p["embeds_other"],
                "box": box[p["box_caption"]],
            }
            embeds = {k: v[None].to(device, torch.bfloat16) for k, v in embeds.items()}
            state["panel"] = tuple(latent.shape[-2:])
            for si, sigma in enumerate(sigmas):
                noisy = (1.0 - sigma) * latent + sigma * noise
                target = noise - latent
                timestep = torch.tensor([1000.0 * sigma], device=device)
                for name, (coords, cap, ref_s, ref_t, iso) in conditions.items():
                    state["coords"] = coords
                    transformer.ref_timestep = ref_t
                    transformer.ref_isolate = iso
                    transformer.ref_pos = "offset" if coords == "offset" else "aligned"
                    ref_s = sigma if ref_s == "step" else ref_s
                    for tag, ref in refs.items() if ref_s is not None else (("", None),):  # fmt: skip
                        if ref_s:
                            ref = (1.0 - ref_s) * ref + ref_s * ref_noise
                        pred = transformer(
                            noisy, timestep, embeds[cap], return_dict=False, ref=ref
                        )[0]
                        loss[name + tag][si, pi] = float(
                            (pred.float() - target).pow(2).mean()
                        )
            print(f"[canvas] {pi + 1}/{len(pairs)} {p['item'].stem}", flush=True)
    rope.forward = stock

    def paired(a, b):
        d = loss[a] - loss[b]
        return [
            dict(mean=round(float(r.mean()), 5), ci=[round(x, 5) for x in boot(r)])
            for r in d
        ]

    metrics = dict(
        n_pairs=len(pairs),
        pool=pool,
        sigmas=sigmas,
        loss={n: [round(float(x), 5) for x in loss[n].mean(1)] for n in names},
        own_minus_shuf={n: paired(n, f"{n}_shuf") for n in ref_conditions},
        minus_none=(
            {n: paired(n, "none") for n in names if n != "none"}
            if "none" in conditions
            else {}
        ),
    )
    head = "".join(f"  σ {s:<28}" for s in sigmas)
    print(f"[canvas] {'loss':16s}{head}", flush=True)
    for n in names:
        row = "".join(f"  {x:<30.4f}" for x in metrics["loss"][n])
        print(f"[canvas] {n:16s}{row}", flush=True)
    for title, table in (
        ("own − shuf", metrics["own_minus_shuf"]),
        ("− none", metrics["minus_none"]),
    ):
        print(f"[canvas] {title:16s}{head}", flush=True)
        for n, cells in table.items():
            row = "".join(
                f"  {c['mean']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]".ljust(32)
                for c in cells
            )
            print(f"[canvas] {n:16s}{row}", flush=True)

    run_dir = make_run_dir("moss_ref_canvas", args.label or "probe")
    np.savez_compressed(
        run_dir / "loss.npz",
        **loss,
        sigmas=np.array(sigmas),
        stems=np.array([p["item"].stem for p in pairs]),
        others=np.array([p["other"]["item"].stem for p in pairs]),
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["loss.npz"],
        device=device,
    )
    print(f"[canvas] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
