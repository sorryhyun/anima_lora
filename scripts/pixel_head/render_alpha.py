"""Anima renders -> RGBA heads -> transparent PNGs, a 4 x 2 sheet on a checker per head, block
counts, and each head's RGB against its init head's RGB (dB).

argv: <out dir> <RGBA head path> [<RGBA head path> ...]. Every head reads the same renders;
each writes ``<out dir>/<head's run folder>[_<file stem>]/``. Blocks: 8 x 8 tiles whose alpha mean is in (0.1, 0.9) with
std < 0.05 (a flat mid tile); interior mid: 0.02 < a < 0.98 more than 2 px from the 0.5 contour."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from PIL import Image

from bench._anima import DEFAULT_NEG
from bench._common import start_heartbeat
from library.anima.pixel_head import PATCH, PixelTap, load_pixel_head
from library.anima.pixel_rgba import straight_rgba
from library.anima.vocab_pack import default_vocab_pack
from library.anima.weights import _attach_vocab_pack_if_set
from library.env import default_checkpoints, resolve_under_home
from library.inference.args import build_default_args
from library.inference.generation import generate_body
from library.inference.models import load_text_encoder
from library.inference.text import prepare_text_inputs
from scripts.pixel_head._alpha_read import on_checker, rgb_db
from scripts.pixel_head._setup import load_models

Q = "masterpiece, best quality, score_7, safe. "
TALL = (1216, 832)
PROMPTS = [
    (
        "maid",
        "1girl, maid, frills, lace, detailed clothes, holding tray, full body, white background.",
        TALL,
    ),
    (
        "oldman",
        "1boy, old man, long white hair, long beard, white robe, full body, white background.",
        TALL,
    ),
    (
        "glass",
        "no humans, wine glass, glass bottle, transparent, white background.",
        TALL,
    ),
    ("bicycle", "no humans, bicycle, side view, white background.", TALL),
]
SEEDS, STEPS, CFG = (0, 1), 28, 4.0

out = Path(sys.argv[1])
start_heartbeat(label="render_alpha")
device = torch.device("cuda")
ck = default_checkpoints()
req = SimpleNamespace(
    dit=ck.dit,
    vae=ck.vae,
    device="cuda",
    stream="fp32",
    attn_mode="flash",
    compile=True,
)
toks = sorted({(h // PATCH) * (w // PATCH) for *_, (h, w) in PROMPTS})
vae, anima, _ = load_models(req, device, [(toks[0], toks[-1])])
del vae  # the head writes the colour
_attach_vocab_pack_if_set(anima, default_vocab_pack())
te_path = str(resolve_under_home(ck.text_encoder))
te = load_text_encoder(text_encoder=te_path, dtype=torch.bfloat16, device=device)
shared = {"text_encoder": te.eval()}
ctxs = [
    prepare_text_inputs(
        device=device,
        anima=anima,
        shared_models=shared,
        prompt=Q + p,
        negative_prompt=DEFAULT_NEG,
        text_encoder_path=te_path,
    )
    for _, p, _ in PROMPTS
]
shared.clear()
del te
torch.cuda.empty_cache()


def _tag(path: Path) -> str:
    return path.parent.name + (
        "" if path.stem == "rgba_head" else "_" + path.stem.split("_")[-1]
    )


heads, inits = {}, {}
for a in sys.argv[2:]:
    path = Path(resolve_under_home(a))
    head = load_pixel_head(path, device)
    assert head.channels == 4, f"{path}: not an RGBA head (channels {head.channels})"
    heads[_tag(path)] = head
    init = head.meta["init"]
    if init not in inits:
        inits[init] = load_pixel_head(resolve_under_home(init), device)
for t in heads:
    (out / t).mkdir(parents=True, exist_ok=True)


def blocks(a):
    from scipy import ndimage as nd

    H, W = a.shape
    t = (
        a[: H // 8 * 8, : W // 8 * 8]
        .reshape(H // 8, 8, W // 8, 8)
        .transpose(0, 2, 1, 3)
    )
    t = t.reshape(H // 8, W // 8, 64)
    flat = int(((t.mean(-1) > 0.1) & (t.mean(-1) < 0.9) & (t.std(-1) < 0.05)).sum())
    hard = a >= 0.5
    edge = hard ^ nd.binary_erosion(hard)
    d = nd.distance_transform_edt(~(edge | (hard ^ nd.binary_dilation(hard))))
    mid = (a > 0.02) & (a < 0.98)
    return (
        flat,
        float((mid & (d > 2)).mean() * 100),
        float((mid & (d <= 2)).mean() * 100),
    )


stats = {t: {} for t in heads}
tap = PixelTap(anima)
cells = {t: [] for t in heads}
for (name, _, (h, w)), (ctx, null) in zip(PROMPTS, ctxs):
    for seed in SEEDS:
        args = build_default_args(
            [
                "--prompt",
                "x",
                "--negative_prompt",
                DEFAULT_NEG,
                "--dit",
                str(ck.dit),
                "--vae",
                str(ck.vae),
                "--text_encoder",
                str(ck.text_encoder),
                "--save_path",
                "unused",
                "--infer_steps",
                str(STEPS),
                "--flow_shift",
                "3.0",
                "--guidance_scale",
                str(CFG),
                "--sampler",
                "euler",
                "--image_size",
                str(h),
                str(w),
                "--seed",
                str(seed),
            ]
        )
        tap.reset()
        generate_body(args, anima, ctx, null, device, seed)

        def decode(head):
            x = (tap.decode(head, CFG)[0] + 1) / 2
            return x.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        ref = {k: decode(v) for k, v in inits.items()}
        for t, head in heads.items():
            x = decode(head)
            c, a = x[..., :3], x[..., 3]
            rgba = straight_rgba(c, a)
            Image.fromarray(np.round(rgba * 255).astype(np.uint8)).save(
                out / t / f"{name}_s{seed}.png"
            )
            p = rgba[..., :3] * a[..., None]
            cells[t].append(
                Image.fromarray(
                    np.round(on_checker(p, a).clip(0, 1) * 255).astype(np.uint8)
                )
            )
            fl, far, near = blocks(a)
            db = rgb_db(c, ref[head.meta["init"]])
            stats[t][f"{name}_s{seed}"] = {
                "flat_mid_tiles": fl,
                "interior_mid_pct": far,
                "edge_mid_pct": near,
                "rgb_db_vs_init": db,
            }
            print(
                f"{t} {name} s{seed} sigma {tap.sigma:.4f} alpha mean {a.mean():.3f}  flat mid "
                f"tiles {fl}  interior mid {far:.2f}%  edge mid {near:.2f}%  RGB vs init "
                f"{db:.1f} dB",
                flush=True,
            )

for t in heads:
    w, h, gap = cells[t][0].width, cells[t][0].height, 12
    sheet = Image.new("RGB", (4 * (w + gap) - gap, 2 * (h + gap) - gap), "white")
    for i, im in enumerate(cells[t]):
        sheet.paste(im, ((i // 2) * (w + gap), (i % 2) * (h + gap)))
    sheet.save(out / t / "renders_4x2.png")
    (out / t / "blocks.json").write_text(json.dumps(stats[t], indent=1))
    print(
        t, "flat mid tiles, total", sum(v["flat_mid_tiles"] for v in stats[t].values())
    )
