#!/usr/bin/env python3
"""Cache the alpha head's features: cutouts on light backgrounds through the frozen DiT.

Composite (white | flat light colour | light gradient with grain) → VAE encode → one DiT
forward at a decode σ → feature, x_t patches and the alpha on disk
(``library/anima/pixel_alpha_cache.py``). Training reads only these files
(``scripts/pixel_head/train_alpha.py``). A rerun encodes only what changed, and loads no
model when nothing did.

Every flag is a :class:`CacheAlphaRequest` field; ``--config <file.toml>`` takes a flat
TOML of field names (field default → ``--config`` → CLI).

    make daemon-run ARGS="--queue --stall-timeout 900 scripts/pixel_head/cache_alpha.py \
        --config project/pixel_decoder/configs/<cache run>.toml"
"""

from __future__ import annotations

import gc
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from library.env import default_checkpoints, resolve_under_home
from library.lines.requests import Request
from library.lines.requests import request_field as _f

_CKPTS = default_checkpoints()


@dataclass
class CacheAlphaRequest(Request):
    """Cache the alpha head's features."""

    cutouts: str = _f("", "RGBA cutouts: <artist>/<id>.png + .txt")
    bg_source: str = _f("image_dataset", "whose simple backgrounds give the colours")
    cache_dir: str = _f("post_image_dataset/pixel_alpha", "the cache")
    draws: int = _f(1, "train draws per cutout and background kind")
    n_read: int = _f(16, "images that never train, and as many train images also read")
    limit: int = _f(0, "first N cutouts (smoke); 0 = all")
    sigma_lo: float = _f(0.0577, "decode σ range, low end")
    sigma_hi: float = _f(0.1364, "decode σ range, high end")
    seed: int = _f(0, "noise, train backgrounds and σ")
    dit: str = _f(_CKPTS.dit, "frozen DiT")
    vae: str = _f(_CKPTS.vae, "VAE of the composites")
    text_encoder: str = _f(_CKPTS.text_encoder, "encodes the captions")
    stream: str = _f("fp32", "DiT residual stream", choices=("fp32", "bf16"))
    attn_mode: str = _f("flash", "DiT attention")
    compile: bool = _f(True, "compile the DiT's blocks")
    device: str = _f("cuda", "torch device")


def encode_captions(req: CacheAlphaRequest, anima, device, captions: list[str]):
    """→ {caption: crossattn embed, bf16 on the CPU}; the TE is freed after."""
    import torch

    from library.inference.models import load_text_encoder
    from library.inference.text import prepare_text_inputs

    path = str(resolve_under_home(req.text_encoder))
    te = load_text_encoder(text_encoder=path, dtype=torch.bfloat16, device=device)
    shared = {"text_encoder": te.eval(), "conds_cache": {}}
    out = {}
    # prepare_text_inputs collects garbage per prompt: not over the whole heap
    gc.collect()
    gc.freeze()
    try:
        for cap in captions:
            cond, _ = prepare_text_inputs(
                device=device,
                anima=anima,
                shared_models=shared,
                prompt=cap,
                text_encoder_path=path,
            )
            out[cap] = cond["embed"][0].to("cpu", torch.bfloat16)
    finally:
        gc.unfreeze()
    shared.clear()
    del te
    torch.cuda.empty_cache()
    return out


def run(req: CacheAlphaRequest) -> Path:
    import numpy as np
    import torch

    from library.anima import pixel_alpha_cache as C
    from library.anima import pixel_alpha_data as D
    from library.anima.pixel_head_train import DiTRollout
    from library.anima.vocab_pack import default_vocab_pack
    from library.anima.weights import _attach_vocab_pack_if_set
    from library.datasets.buckets import cluster_token_bands
    from library.datasets.image_utils import IMAGE_TRANSFORMS
    from scripts.pixel_head._setup import load_models

    if not req.cutouts:
        raise ValueError("--cutouts: the folder of RGBA cutouts")
    t0 = time.perf_counter()
    root = Path(resolve_under_home(req.cache_dir))
    pal = D.fit_palette(Path(resolve_under_home(req.bg_source)))
    palette = pal["light"]
    cuts = D.load_cutouts(
        Path(resolve_under_home(Path(req.cutouts).expanduser())), limit=req.limit
    )
    fit, read = D.pick_stress(cuts, req.n_read)
    pack = default_vocab_pack()
    recipe = {
        "dit": C.file_id(resolve_under_home(req.dit)),
        "vae": C.file_id(resolve_under_home(req.vae)),
        "text_encoder": C.file_id(resolve_under_home(req.text_encoder)),
        "vocab_pack": C.file_id(resolve_under_home(pack) if pack else None),
        "stream": req.stream,
        "sigma": [req.sigma_lo, req.sigma_hi],
        "seed": req.seed,
        "palette": palette,
    }
    rows = C.plan_rows(cuts, fit, read, req.draws, recipe)
    todo = [r for r in rows if not C.is_current(root, r)]
    need = sum(C.row_bytes(r) for r in todo)
    print(
        f"cutouts {len(cuts)}, never train {len(read)}; backgrounds {len(palette)} light "
        f"of {pal['found']}; rows {len(rows)}, to encode {len(todo)} ({need / 1e9:.1f} GB)"
        f"  ({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    if not todo:
        C.write_manifest(root, recipe, rows)
        print(f"cache current → {root}", flush=True)
        return root

    root.mkdir(parents=True, exist_ok=True)
    C.drop_manifest(root)  # no half-rebuilt cache behind a finished manifest
    for sub in {r.set for r in todo}:
        (root / sub).mkdir(exist_ok=True)

    device = torch.device(req.device)
    bands = cluster_token_bands([r.tokens for r in rows])
    vae, anima, _ = load_models(req, device, bands)
    _attach_vocab_pack_if_set(anima, pack)
    t1 = time.perf_counter()
    cross = encode_captions(
        req, anima, device, list(dict.fromkeys(r.caption for r in todo))
    )
    print(f"captions {len(cross)} in {time.perf_counter() - t1:.0f}s", flush=True)
    rollout = DiTRollout(anima, vae, device, req.seed)
    by_key = {c.key: c for c in cuts}

    def prepare(row):  # a worker thread: composite → the VAE's input
        comp, bg, sigma = C.compose(row, by_key[row.key], palette, recipe)
        px = IMAGE_TRANSFORMS(np.round(comp * 255).astype(np.uint8))[None]
        return comp, bg, sigma, px

    ahead, writing = deque(), deque()
    queue = iter(todo)
    t1 = time.perf_counter()
    with ThreadPoolExecutor(4) as cpu, ThreadPoolExecutor(4) as io, torch.no_grad():

        def refill() -> None:
            while len(ahead) < 8 and (row := next(queue, None)) is not None:
                ahead.append((row, cpu.submit(prepare, row)))

        refill()
        done = 0
        while ahead:
            row, fut = ahead.popleft()
            refill()
            comp, bg, sigma, px = fut.result()
            lat = vae.encode_pixels_to_latents(px.to(device, torch.bfloat16))
            f, lat_p, _ = rollout.forward(
                lat.squeeze(2).float(),
                cross[row.caption].to(device),
                row.noise,
                sigma,
            )
            f, lat_p = f.to("cpu", torch.bfloat16), lat_p.cpu()
            writing.append(
                io.submit(
                    C.write_row, root, row, by_key[row.key], f, lat_p, sigma, comp, bg
                )
            )
            while len(writing) > 8:
                writing.popleft().result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = (time.perf_counter() - t1) / done
                print(
                    f"  cache {done}/{len(todo)}  {el:.3f}s/row  "
                    f"eta {(len(todo) - done) * el / 60:.1f}m",
                    flush=True,
                )
        for fut in writing:
            fut.result()
    C.write_manifest(root, recipe, rows)
    print(f"cache done in {time.perf_counter() - t0:.0f}s → {root}", flush=True)
    return root


def main() -> None:
    req = CacheAlphaRequest.from_argv()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(req)


if __name__ == "__main__":
    main()
