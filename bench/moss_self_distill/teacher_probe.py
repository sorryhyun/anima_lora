#!/usr/bin/env python3
"""How much of the self-distill teacher's shift is signal, by caption format.

Forward-only, on the cached student embeds and the cached teacher embeds
(``{stem}.moss_vis.safetensors``) of two caches of one image set: tag captions and
layout JSON. Each latent is noised at high σ; per conditioning the one-step velocity
``v``, its flow loss ``fm = ‖v − (ε − x₀)‖²`` and the distance between any two.

For a student ``s`` and a teacher ``t`` on the same ``x_t``:

    eff = (fm_s − fm_t) / ‖v_t − v_s‖²

1 = the teacher's whole shift is toward the image, 0 = the shift buys no flow loss,
below 0 = it costs. Summed over images before dividing, bootstrap over images.

Conditionings (``--groups``): ``tag`` the tag caption, ``v1`` its first shuffled /
tag-dropped variant, ``lay`` the layout JSON; ``.cap`` the student's embeds, ``.img``
the teacher's (image, then the caption), ``.cut`` the teacher's with the image rows
and their two markers dropped before the DiT (the caption rows saw the image in the
TE; the sequence is the student's shape). With ``--lora`` every one runs under the
LoRA too (multiplier 0 and 1 on one DiT).

``--renders`` full renders (CFG 3, 50 steps) of the teacher conditionings.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_self_distill/teacher_probe.py \
        --lora output/ckpt_moss/sincostag_1113_0717.safetensors --label sincos_tag"
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from bench._common import make_run_dir, write_result
from bench.moss_ab._render import grid, render, uncond_embeds
from library.env import resolve_under_home
from library.moss.caching.images import load_rgb
from library.moss.distill import teacher_path
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    load_transformer,
    load_vae,
    stream_fp32_,
)
from library.moss.model import decode, snapped_sigmas, unpatchify
from library.moss.requests import GenerateRequest
from library.moss.training.data import NUM_TRAIN_TIMESTEPS, samples_from_cache

HAND_WORDS = ("hand", "holding", "finger", "waving", "v,", "peace sign", "reaching")
# (student, teacher) pairs summarised; "w:" = that weight state, else the row's own
PAIRS = (
    ("tag.cap", "tag.img"),
    ("v1.cap", "v1.img"),
    ("lay.cap", "lay.img"),
    ("tag.cap", "lay.img"),
    ("v1.cap", "lay.img"),
    ("tag.cap", "tag.cut"),
    ("lay.cap", "lay.cut"),
)
# Upper edges of the radial bands on the 32 ch grid, as fractions of its Nyquist; the
# first band is DC alone, the last everything above 0.5.
BAND_EDGES = (0.125, 0.25, 0.5)


def band_masks(h: int, w: int, device) -> list[torch.Tensor]:
    fy = torch.fft.fftfreq(h, device=device)[:, None] * 2
    fx = torch.fft.fftfreq(w, device=device)[None, :] * 2
    r = (fy**2 + fx**2).sqrt()
    masks, lo = [r == 0], 0.0
    for hi in (*BAND_EDGES, float("inf")):
        masks.append((r > lo) & (r <= hi))
        lo = hi
    return masks


def band_mse(x: torch.Tensor, masks) -> list[float]:
    """``mean(x²)`` of a packed latent split by radial band; the parts sum to it."""
    f = torch.fft.fft2(unpatchify(x.float()), norm="ortho").abs().pow(2).sum((0, 1))
    return [(f[m].sum() / x.numel()).item() for m in masks]


LEAD = 3  # <|im_start|> user \n, then <|vision_start|>, the image rows, <|vision_end|>


def cut_image(path: Path) -> torch.Tensor:
    """``embeds_full`` of a teacher file without the image rows and their markers."""
    with safe_open(str(path), "pt") as f:
        n_img = int(f.metadata()["image_rows"])
        full = f.get_tensor("embeds_full")
    return torch.cat([full[:LEAD], full[LEAD + 1 + n_img + 1 :]])


def to_png(z, vae, device):
    from PIL import Image

    x = decode(z.to(device), vae)[0].float().add(1).mul(127.5).round().clamp(0, 255)
    return Image.fromarray(x.permute(1, 2, 0).byte().cpu().numpy())


def tensor_of(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), "pt") as f:
        return f.get_tensor(key)


def ratio_ci(num, den, n_boot=2000, seed=0):
    """Σnum / Σden over images with a 95 % bootstrap interval."""
    rng = random.Random(seed)
    n = len(num)
    point = sum(num) / sum(den)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(num[i] for i in idx) / sum(den[i] for i in idx))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_cache", default="post_image_dataset/moss/sincos_tags_var")
    ap.add_argument(
        "--layout_cache", default="post_image_dataset/moss/sincos_layout_sure"
    )
    ap.add_argument("--lora", default="")
    ap.add_argument("--groups", default="tag,v1,lay")
    ap.add_argument("--no_base", action="store_true", help="LoRA rows only")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--sigmas", default="0.9,0.95,0.98")
    ap.add_argument("--eps", type=int, default=2)
    ap.add_argument("--renders", type=int, default=4)
    ap.add_argument(
        "--bands",
        action="store_true",
        help="also split every flow loss and distance by radial frequency band",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    groups = args.groups.split(",")
    device = torch.device("cuda")
    run_dir = make_run_dir("moss_self_distill", label=args.label or "teacher")

    tags = {s.stem: s for s in samples_from_cache(resolve_under_home(args.tag_cache))}
    lays = {
        s.stem: s for s in samples_from_cache(resolve_under_home(args.layout_cache))
    }
    stems = sorted(set(tags) & set(lays))
    stems = stems[:: max(1, len(stems) // args.limit)][: args.limit]
    print(f"{len(stems)} images, groups {groups}, σ {sigmas}", flush=True)

    def conds_of(stem):
        t, la = tags[stem], lays[stem]
        out = {}
        if "tag" in groups:
            out["tag.cap"] = t.load()[1]
            out["tag.img"] = tensor_of(teacher_path(t.te_path), "embeds_full")
            out["tag.cut"] = cut_image(teacher_path(t.te_path))
        if "v1" in groups:
            out["v1.cap"] = t.load_variant(
                1
            )  # stored as the rows after a shared prefix
            out["v1.img"] = tensor_of(teacher_path(t.te_path), "embeds_v1")
        if "lay" in groups:
            out["lay.cap"] = la.load()[1]
            out["lay.img"] = tensor_of(teacher_path(la.te_path), "embeds_full")
            out["lay.cut"] = cut_image(teacher_path(la.te_path))
        return out

    print(
        "rows, first image: "
        + ", ".join(f"{k} {v.shape[0]}" for k, v in conds_of(stems[0]).items()),
        flush=True,
    )
    captions = {}
    for stem in stems:
        with safe_open(str(tags[stem].te_path), "pt") as f:
            captions[stem] = f.metadata()
    # Renders: images whose tags name a hand, non-explicit first (fingers get counted).
    order = sorted(
        stems,
        key=lambda s: (
            not any(w in captions[s]["caption"] for w in HAND_WORDS),
            captions[s]["caption"].startswith("explicit"),
        ),
    )
    render_stems = order[: args.renders]

    # The teacher's tail is off: uncond is the empty prompt without a tail.
    _, uncond, uncond_mask = (
        uncond_embeds(device, precision="int8") if render_stems else (None, None, None)
    )

    from library.moss.dit_int8.base import generate_base

    convert = generate_base(GenerateRequest(int8="on"))
    tr = load_transformer("", "base", device=device)
    net = None
    if args.lora:
        from library.moss.lora import load_for_inference

        net = load_for_inference(
            tr, resolve_under_home(args.lora), 1.0, before_apply=convert
        )
        net.to(device)
    else:
        convert(tr)
    stream_fp32_(tr)
    compile_blocks(tr.layers)
    states = ["base", "lora"] if net is not None else ["base"]
    if args.no_base:
        states = [w for w in states if w != "base"]
    print(f"weight states {states}", flush=True)

    def set_state(w):
        if net is not None:
            net.set_multiplier(1.0 if w == "lora" else 0.0)

    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    with torch.inference_mode():
        for i, stem in enumerate(stems):
            x0 = tags[stem].load()[0].unsqueeze(0).to(device, torch.float32)
            assert x0.shape[1:] == lays[stem].load()[0].shape, stem
            conds = {k: v.unsqueeze(0).to(device) for k, v in conds_of(stem).items()}
            for sigma in sigmas:
                t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
                for k in range(args.eps):
                    eps = torch.randn(x0.shape, generator=gen).to(device)
                    xt = (1 - sigma) * x0 + sigma * eps
                    target = eps - x0
                    vs = {}
                    for w in states:
                        set_state(w)
                        for c, e in conds.items():
                            vs[f"{w}:{c}"] = tr(xt, t, e, return_dict=False)[0].float()
                    keys = list(vs)
                    row = dict(
                        stem=stem,
                        sigma=sigma,
                        eps=k,
                        fm={a: F.mse_loss(vs[a], target).item() for a in keys},
                        d={
                            f"{a}|{b}": F.mse_loss(vs[a], vs[b]).item()
                            for n, a in enumerate(keys)
                            for b in keys[n + 1 :]
                        },
                    )
                    if args.bands:
                        masks = band_masks(2 * x0.shape[-2], 2 * x0.shape[-1], device)
                        row["fm_b"] = {a: band_mse(vs[a] - target, masks) for a in keys}
                        row["d_b"] = {
                            f"{a}|{b}": band_mse(vs[a] - vs[b], masks)
                            for n, a in enumerate(keys)
                            for b in keys[n + 1 :]
                        }
                    rows.append(row)
            print(f"[v] {i + 1}/{len(stems)} {stem}", flush=True)

        steps = snapped_sigmas(50, GenerateRequest().shift)
        arms = [
            ("base", "tag.img"),
            ("base", "tag.cut"),
            ("base", "lay.img"),
            ("lora", "tag.img"),
            ("lora", "tag.cut"),
        ]
        arms = [(w, c) for w, c in arms if w in states and c.split(".")[0] in groups]
        renders = {}
        for stem in render_stems:
            h, w_px = (n * 16 for n in tags[stem].latent_hw)
            conds = conds_of(stem)
            for w, c in arms:
                set_state(w)
                renders[(stem, w, c)] = render(
                    tr,
                    conds[c],
                    uncond,
                    uncond_mask,
                    steps,
                    seed=1,
                    height=h,
                    width=w_px,
                    cfg=3.0,
                ).cpu()
                print(f"[render] {stem} {w}:{c}", flush=True)
    del tr
    empty_cache()

    # --- summary: per image means over ε, then Σ over images ---
    def dist(r, a, b):
        return r["d"].get(f"{a}|{b}", r["d"].get(f"{b}|{a}"))

    summary = {}
    lines = []
    combos = [(w, w) for w in states]
    if len(states) == 2:
        combos.append(("lora", "base"))  # LoRA student, Base teacher
    for ws, wt in combos:
        for s_c, t_c in PAIRS:
            a, b = f"{ws}:{s_c}", f"{wt}:{t_c}"
            if a not in rows[0]["fm"] or b not in rows[0]["fm"]:
                continue
            for sigma in sigmas:
                per = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
                for r in rows:
                    if r["sigma"] != sigma:
                        continue
                    p = per[r["stem"]]
                    p[0] += r["fm"][a] / args.eps
                    p[1] += r["fm"][b] / args.eps
                    p[2] += dist(r, a, b) / args.eps
                vals = list(per.values())
                gain = [p[0] - p[1] for p in vals]
                gap = [p[2] for p in vals]
                eff, lo, hi = ratio_ci(gain, gap)
                n = len(vals)
                key = f"{a} -> {b} @ {sigma}"
                summary[key] = dict(
                    fm_student=sum(p[0] for p in vals) / n,
                    fm_teacher=sum(p[1] for p in vals) / n,
                    gap=sum(gap) / n,
                    eff=eff,
                    eff_lo=lo,
                    eff_hi=hi,
                    teacher_lower=sum(g > 0 for g in gain) / n,
                )
                lines.append(
                    f"{a:>14} -> {b:<14} σ {sigma:<5} fm {summary[key]['fm_student']:.4f}"
                    f" -> {summary[key]['fm_teacher']:.4f}  gap {summary[key]['gap']:.5f}"
                    f"  eff {eff:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                    f"  teacher lower on {summary[key]['teacher_lower']:.0%}"
                )
    # how far the caption format moves the teacher, and the student
    for w in states:
        for a, b in (
            ("tag.img", "lay.img"),
            ("tag.cap", "lay.cap"),
            ("tag.img", "v1.img"),
            ("tag.img", "tag.cut"),
            ("lay.img", "lay.cut"),
        ):
            ka, kb = f"{w}:{a}", f"{w}:{b}"
            if ka not in rows[0]["fm"] or kb not in rows[0]["fm"]:
                continue
            for sigma in sigmas:
                d = [dist(r, ka, kb) for r in rows if r["sigma"] == sigma]
                summary[f"{ka} vs {kb} @ {sigma}"] = sum(d) / len(d)
                lines.append(
                    f"{ka:>14} vs {kb:<14} σ {sigma:<5} dist {sum(d) / len(d):.5f}"
                )
    print("\n" + "\n".join(lines), flush=True)
    (run_dir / "rows.json").write_text(json.dumps(rows))

    artifacts = ["rows.json"]
    if renders:
        vae = load_vae("", device=device)
        with torch.inference_mode():
            sheet = []
            for stem in render_stems:
                cells = [("source", load_rgb(Path(captions[stem]["source"])))]
                cells += [
                    (f"{w} {c}", to_png(renders[(stem, w, c)], vae, device))
                    for w, c in arms
                ]
                for label, im in cells[1:]:
                    name = f"render_{stem}_{label.replace(' ', '_')}.png"
                    im.save(run_dir / name)
                    artifacts.append(name)
                sheet.append((stem, cells))
            grid(sheet, cell_h=640).save(run_dir / "teacher_renders.png")
            artifacts.append("teacher_renders.png")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"pairs": summary, "stems": stems, "render_stems": render_stems},
        artifacts=artifacts,
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
