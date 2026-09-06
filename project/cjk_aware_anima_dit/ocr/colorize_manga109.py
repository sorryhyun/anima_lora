#!/usr/bin/env python3
"""O3 colorized COO (``plan_ocr.md`` decision 4): repaint Manga109-s spreads with the
EasyControl colorize LoRA, lettering and layout untouched, so the COO polygons
(and the speech boxes) can be re-cropped from a doujin-surfaced copy.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 0 \\
        project/cjk_aware_anima_dit/ocr/colorize_manga109.py --pages 20"        # the pilot
    … --pages 2000                                                             # the 2k subset
    … --all                                                                    # every train page with a COO polygon
    … --tier 1280                                                              # generation tier (default 1024)
    … --no_halves --prompt ""                                                  # the first pilot's whole-spread, caption-free form

A Manga109 "page" is a two-page **spread** (1654×1170). Default ``--halves``
splits it at the middle, colorizes each half as its own portrait page (the
1024 tier lands a half at ≈ 864×1216 — native resolution, where the whole
spread had to shrink to 1216×864 and lost the small kanji: pilot 1 in
``findings.md``) and stitches the two back; polygons stay valid.

Pages = the **train**-split pages that carry at least one COO polygon (6,034),
in one seeded permutation; ``--pages N`` takes its first N, so the pilot is a
prefix of the 2k subset, which is a prefix of ``--all`` (the mixes nest).

The ``make test-easycontrol`` / ``EASYADAPTER=colorize`` recipe
(``scripts/tasks/inference.py``) with a ``comic`` prompt (the tag the adapter's
captions kept — ``text_keep_comic``; ``--prompt ""`` = caption-free), the shared
negative prompt, cfg 4, 28 euler steps at flow_shift 3, seed 42, ref free-fit
into the chosen tier's band (``--easycontrol_image_match_size`` uses 1024) —
but the DiT / VAE / text context / adapter are loaded **once** and only the
cond latent + KV cache are re-primed per page (``set_cond`` +
``precompute_cond_kv``). The output is resized back to the page's native size
(1654×1170) with LANCZOS so the O1 crop builder can reuse the same polygons:
``derived/colorized/<name>/<Book>/<page>.png`` + ``pages.jsonl`` (one row per
page: book, page, gen size, seconds); ``<name>`` defaults to
``<tier>[_half][_<prompt>]``. Never in-tree (Manga109-s derivative).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

sys.path.insert(0, str(m109.REPO))

DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
TE = "models/text_encoders/qwen_3_06b_base.safetensors"
VAE = "models/vae/qwen_image_vae.safetensors"
NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
NATIVE = (1654, 1170)  # (w, h) of every Manga109 spread


def latest_colorize_weight() -> Path:
    cands = sorted(
        (m109.REPO / "output/ckpt").glob("anima_colorize*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise SystemExit("no anima_colorize*.safetensors under output/ckpt/")
    return cands[0]


def page_draw(seed: int) -> list[tuple[str, int]]:
    """Every train page with ≥ 1 COO polygon, one seeded permutation."""
    split = m109.load_split()
    pages: list[tuple[str, int]] = []
    for book in split["train"]:
        pages += sorted((book, p) for p in {ln.page for ln in m109.iter_coo(book)})
    random.Random(seed).shuffle(pages)
    return pages


def out_root(name: str) -> Path:
    d = m109.derived_root() / "colorized" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pages", type=int, help="first N pages of the seeded draw")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--tier", type=int, default=1024, choices=[896, 1024, 1280, 1536])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42, help="noise seed (every page)")
    ap.add_argument("--draw_seed", type=int, default=0, help="page permutation seed")
    ap.add_argument("--ec_scale", type=float, help="--easycontrol_scale override")
    ap.add_argument("--weight", help="adapter (default newest anima_colorize*)")
    ap.add_argument("--prompt", default="comic", help='"" = caption-free')
    ap.add_argument(
        "--no_halves",
        action="store_true",
        help="colorize the whole spread in one pass (pilot 1's form)",
    )
    ap.add_argument("--name", help="output subdir (default <tier>[_half][_<prompt>])")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    halves = not a.no_halves
    name = a.name or (
        f"{a.tier}"
        + ("_half" if halves else "")
        + (f"_{a.prompt.replace(' ', '-')}" if a.prompt else "")
    )

    import torch
    from PIL import Image
    from torchvision import transforms

    from library.datasets.buckets import (
        choose_edge,
        freefit_band_for_edge,
        freefit_bucket,
    )
    from library.inference.generation import generate_body
    from library.inference.models import load_dit_model, load_text_encoder
    from library.inference.output import pixels_to_pil
    from library.inference.request import GenerationRequest
    from library.inference.text import prepare_text_inputs
    from library.models import qwen_vae
    from library.runtime.device import clean_memory_on_device
    from networks.methods.easycontrol import create_network_from_weights

    weight = Path(a.weight) if a.weight else latest_colorize_weight()
    pages = page_draw(a.draw_seed)
    if not a.all:
        pages = pages[: a.pages]
    out = out_root(name)
    todo = [
        (b, p)
        for b, p in pages
        if a.overwrite or not (out / b / f"{p:03d}.png").exists()
    ]
    print(
        f"colorize {len(pages)} pages ({len(todo)} to do) at tier {a.tier} "
        f"{'halves' if halves else 'whole'} prompt {a.prompt!r} with {weight.name} "
        f"→ {out}",
        flush=True,
    )
    if not todo:
        return

    device = torch.device("cuda")
    extra = ["--easycontrol_image_match_size"]
    if a.ec_scale is not None:
        extra += ["--easycontrol_scale", str(a.ec_scale)]
    req = GenerationRequest(
        dit=str(m109.REPO / DIT),
        vae=str(m109.REPO / VAE),
        text_encoder=str(m109.REPO / TE),
        prompt=a.prompt,
        negative_prompt=NEG,
        infer_steps=a.steps,
        guidance_scale=a.cfg,
        flow_shift=a.flow_shift,
        sampler="euler",
        seed=a.seed,
        attn_mode="flash",
        vae_chunk_size=64,
        vae_disable_cache=True,
        easycontrol_weight=str(weight),
        easycontrol_image=str(m109.page_path(*todo[0])),
        save_path=str(out),
        extra_argv=extra,
    )
    args = req.to_args()
    args.device = device

    # -- resident models: DiT, text context (empty prompt — constant), VAE, adapter
    anima = load_dit_model(args, device, torch.bfloat16)
    te = load_text_encoder(args, dtype=torch.bfloat16, device=torch.device("cpu"))
    te.to(device)
    context, context_null = prepare_text_inputs(
        args, device, anima, {"text_encoder": te, "conds_cache": {}}
    )
    del te
    clean_memory_on_device(device)
    vae = qwen_vae.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        vae_2d=args.vae_2d,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()
    kw = {}
    if a.ec_scale is not None:
        kw["cond_scale"] = float(a.ec_scale)
    network, _ = create_network_from_weights(
        multiplier=1.0, file=str(weight), ae=None, text_encoders=None, unet=anima, **kw
    )
    network.load_weights(str(weight))
    network.to(device, dtype=torch.bfloat16)
    network.apply_to(text_encoders=None, unet=anima)
    anima._easycontrol_network = network
    print(
        f"adapter r={network.cond_lora_dim} scale={network.get_effective_scale():.3f}; "
        f"steps {a.steps} cfg {a.cfg} seed {a.seed}",
        flush=True,
    )
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    def paint(ref: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
        w, h = ref.size
        edge = choose_edge(w, h, [a.tier])
        bw, bh = freefit_bucket(w, h, freefit_band_for_edge(edge))
        args.image_size = [bh, bw]
        args.seed = a.seed
        img_t = (
            tfm(ref.resize((bw, bh), Image.LANCZOS))
            .unsqueeze(0)
            .to(device, torch.bfloat16)
        )
        with torch.no_grad():
            cond = vae.encode_pixels_to_latents(img_t).squeeze(2)  # (1, C, H', W')
            network.set_cond(cond.to(device, torch.bfloat16))
            network.precompute_cond_kv()
            latent = generate_body(args, anima, context, context_null, device, a.seed)
            pix = vae.decode_to_pixels(latent.to(device, vae.dtype))
        if pix.ndim == 5:
            pix = pix.squeeze(2)
        im = pixels_to_pil(pix[0].float().cpu())
        if im.size != (w, h):
            im = im.resize((w, h), Image.LANCZOS)
        return im, (bw, bh)

    log = (out / "pages.jsonl").open("a", encoding="utf-8")
    t_all = time.time()
    for i, (book, page) in enumerate(todo, 1):
        t0 = time.time()
        with Image.open(m109.page_path(book, page)) as src:
            ref = src.convert("RGB")
        w, h = ref.size
        if halves:
            mid = w // 2
            left, (bw, bh) = paint(ref.crop((0, 0, mid, h)))
            right, _ = paint(ref.crop((mid, 0, w, h)))
            im = Image.new("RGB", (w, h))
            im.paste(left, (0, 0))
            im.paste(right, (mid, 0))
        else:
            im, (bw, bh) = paint(ref)
        dst = out / book / f"{page:03d}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst)
        dt = time.time() - t0
        log.write(
            json.dumps(
                dict(
                    book=book,
                    page=page,
                    gen_w=bw,
                    gen_h=bh,
                    halves=halves,
                    prompt=a.prompt,
                    steps=a.steps,
                    tier=a.tier,
                    seconds=round(dt, 2),
                )
            )
            + "\n"
        )
        log.flush()
        if i <= 3 or i % 25 == 0 or i == len(todo):
            el = time.time() - t_all
            print(
                f"[{i}/{len(todo)}] {book}/{page:03d} gen {bw}x{bh} {dt:.1f}s "
                f"avg {el / i:.1f}s/page eta {(len(todo) - i) * el / i / 60:.0f} min "
                f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                flush=True,
            )
    log.close()
    print(f"done {len(todo)} pages in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
