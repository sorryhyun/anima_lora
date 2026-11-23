#!/usr/bin/env python3
"""Cache the alpha head's inputs: cutouts on light backgrounds, up to the DiT's door.

Composite (white | flat light colour | light gradient with grain) → VAE latent, caption →
TE + llm_adapter embed, and the alpha (``scripts/pixel_head/_alpha_cache.py``). No DiT
forward and no feature on disk: training runs the DiT (``scripts/pixel_head/train_alpha.py``).
A rerun encodes only what changed, and loads no model when nothing did.

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
    """Cache the alpha head's inputs."""

    cutouts: str = _f("", "RGBA cutouts: <artist>/<id>.png + .txt")
    extra_cutouts: str = _f(
        "",
        "comma-separated folders (<artist>/<id>.png or flat <id>.png) that only train",
    )
    thin: str = _f(
        "", "<artist>:<frac>, keep a stable share of that artist's train-only cutouts"
    )
    bg_source: str = _f("image_dataset", "whose simple backgrounds give the colours")
    cache_dir: str = _f("post_image_dataset/pixel_alpha", "the cache")
    draws: int = _f(1, "train draws per cutout and background kind")
    n_read: int = _f(16, "images that never train, and as many train images also read")
    limit: int = _f(0, "first N cutouts (smoke); 0 = all")
    seed: int = _f(0, "train backgrounds, and the read rows' noise")
    dit: str = _f(_CKPTS.dit, "whose llm_adapter the captions go through")
    vae: str = _f(_CKPTS.vae, "VAE of the composites")
    text_encoder: str = _f(_CKPTS.text_encoder, "encodes the captions")
    device: str = _f("cuda", "torch device")


def encode_captions(req: CacheAlphaRequest, root: Path, pack, rows: list, device):
    """One embed file per row of ``rows``; the TE and the llm_adapter are freed after."""
    import torch

    from scripts.pixel_head import _alpha_cache as C
    from library.anima.models import Anima
    from library.anima.weights import load_llm_adapter
    from library.inference.models import load_text_encoder
    from library.inference.text import prepare_text_inputs

    class Bridge:
        """What prepare_text_inputs takes from the DiT: the adapter alone."""

        _preprocess_text_embeds = Anima._preprocess_text_embeds

        def __init__(self, adapter):
            self.llm_adapter, self.device = adapter, device

    adapter = load_llm_adapter(
        str(resolve_under_home(req.dit)), device=device, vocab_pack=pack
    )
    bridge = Bridge(adapter)
    path = str(resolve_under_home(req.text_encoder))
    te = load_text_encoder(text_encoder=path, dtype=torch.bfloat16, device=device)
    shared = {"text_encoder": te.eval(), "conds_cache": {}}
    # prepare_text_inputs collects garbage per prompt: not over the whole heap
    gc.collect()
    gc.freeze()
    try:
        for row in rows:
            cond, _ = prepare_text_inputs(
                device=device,
                anima=bridge,
                shared_models=shared,
                prompt=row.caption,
                text_encoder_path=path,
            )
            C.write_text(root, row, cond["embed"][0][0].to("cpu", torch.bfloat16))
            shared["conds_cache"].pop(cond["prompt"], None)
    finally:
        gc.unfreeze()
    shared.clear()
    del te, adapter, bridge
    torch.cuda.empty_cache()


def run(req: CacheAlphaRequest) -> Path:
    import numpy as np
    import torch

    from scripts.pixel_head import _alpha_cache as C
    from scripts.pixel_head import _alpha_data as D
    from library.anima.vocab_pack import default_vocab_pack
    from library.datasets.image_utils import IMAGE_TRANSFORMS
    from library.models import qwen_vae

    if not req.cutouts:
        raise ValueError("--cutouts: the folder of RGBA cutouts")
    t0 = time.perf_counter()
    root = Path(resolve_under_home(req.cache_dir))
    pal = D.fit_palette(Path(resolve_under_home(req.bg_source)))
    palette = pal["light"]
    cuts = D.load_cutouts(
        Path(resolve_under_home(Path(req.cutouts).expanduser())), limit=req.limit
    )
    fit, read = D.pick_stress(cuts, req.n_read)  # from `cutouts` alone: the read stays
    if req.thin:
        artist, frac = req.thin.rsplit(":", 1)
        picked = {c.key for c in fit + read}
        cuts = [
            c
            for c in cuts
            if c.key in picked or D.thin_keep(c.key, artist, float(frac))
        ]
    for extra in filter(None, (e.strip() for e in req.extra_cutouts.split(","))):
        cuts = D.dedup(
            cuts + D.load_cutouts(Path(resolve_under_home(Path(extra).expanduser())))
        )
    pack = default_vocab_pack()
    recipe = {
        "dit": C.file_id(resolve_under_home(req.dit)),
        "vae": C.file_id(resolve_under_home(req.vae)),
        "text_encoder": C.file_id(resolve_under_home(req.text_encoder)),
        "vocab_pack": C.file_id(resolve_under_home(pack) if pack else None),
        "seed": req.seed,
        "palette": palette,
    }
    rows = C.plan_rows(cuts, fit, read, req.draws, recipe)
    todo = [r for r in rows if not C.is_current(root, r)]
    texts = [
        r
        for r in {r.text: r for r in todo}.values()
        if not C.text_path(root, r).exists()
    ]
    print(
        f"cutouts {len(cuts)}, never train {len(read)}; backgrounds {len(palette)} light "
        f"of {pal['found']}; rows {len(rows)}, to encode {len(todo)}, captions "
        f"{len(texts)}  ({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    if not todo:
        C.write_manifest(root, recipe, rows)
        print(f"cache current → {root}", flush=True)
        return root

    root.mkdir(parents=True, exist_ok=True)
    C.drop_manifest(root)  # no half-rebuilt cache behind a finished manifest
    by_key = {c.key: c for c in cuts}
    stale = {r.key for r in todo}
    coloured = {r.key for r in rows if r.set in C.READ_SETS}  # the read needs fg
    with ThreadPoolExecutor(4) as io:
        list(io.map(lambda k: C.write_cutout(root, by_key[k], k in coloured), stale))

    device = torch.device(req.device)
    if texts:
        t1 = time.perf_counter()
        encode_captions(req, root, pack, texts, device)
        print(f"captions {len(texts)} in {time.perf_counter() - t1:.0f}s", flush=True)

    vae = qwen_vae.load_vae(
        req.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)

    def prepare(row):  # a worker thread: composite → the VAE's input
        comp, _, _ = C.compose(row, by_key[row.key], palette, req.seed)
        return IMAGE_TRANSFORMS(np.round(comp * 255).astype(np.uint8))[None]

    ahead = deque()
    queue = iter(todo)
    t1 = time.perf_counter()
    with ThreadPoolExecutor(4) as cpu, torch.no_grad():

        def refill() -> None:
            while len(ahead) < 8 and (row := next(queue, None)) is not None:
                ahead.append((row, cpu.submit(prepare, row)))

        refill()
        done = 0
        while ahead:
            row, fut = ahead.popleft()
            refill()
            px = fut.result().to(device, torch.bfloat16)
            C.write_latent(root, row, vae.encode_pixels_to_latents(px)[0].cpu())
            done += 1
            if done % 100 == 0 or done == len(todo):
                el = (time.perf_counter() - t1) / done
                print(
                    f"  latent {done}/{len(todo)}  {el:.3f}s/row  "
                    f"eta {(len(todo) - done) * el / 60:.1f}m",
                    flush=True,
                )
    C.write_manifest(root, recipe, rows)
    print(f"cache done in {time.perf_counter() - t0:.0f}s → {root}", flush=True)
    return root


def main() -> None:
    req = CacheAlphaRequest.from_argv()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(req)


if __name__ == "__main__":
    main()
