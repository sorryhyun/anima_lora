#!/usr/bin/env python3
"""Train the alpha head: a frozen DiT forward on cached inputs each rollout.

The latents, caption embeds and alphas come from ``scripts/pixel_head/cache_alpha.py``;
no VAE or TE is loaded here and no image folder is read. Each rollout draws σ and noise,
runs the DiT once on a cached latent and pushes the feature into the ×1 trainer's replay
buffer, ``opt_steps`` updates each; no feature is written to disk. The whole head trains
from the ×1 base (its output layer fresh at one channel).

Every flag is a :class:`TrainAlphaRequest` field; ``--config <file.toml>`` takes a flat
TOML of field names (field default → ``--config`` → CLI).

    make daemon-run ARGS="--queue --stall-timeout 900 scripts/pixel_head/train_alpha.py \
        --config project/pixel_decoder/configs/<train run>.toml"

Writes ``<output_dir>/<output_name>/``: ``alpha_head.safetensors``, ``result.json`` and
``GT | alpha head`` sheets of the read sets.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from library.env import default_checkpoints, resolve_under_home
from library.lines.requests import Request
from library.lines.requests import request_field as _f
from scripts.pixel_head._setup import DINO_REPO

_CKPTS = default_checkpoints()


@dataclass
class TrainAlphaRequest(Request):
    """Train the alpha head on cached inputs."""

    output_name: str = _f("alpha", "the run's name")
    output_dir: str = _f("output/pixel_head", "→ <output_dir>/<output_name>/")
    cache_dir: str = _f("post_image_dataset/pixel_alpha", "cache_alpha.py's output")
    init: str = _f(
        "output/pixel_head/x1ll_1122_1000/head.safetensors", "the ×1 base head"
    )
    rollouts: int = _f(1000, "DiT forwards, opt_steps updates each")
    opt_steps: int = _f(16, "head updates per rollout")
    sigma_lo: float = _f(0.0577, "decode σ range, low end")
    sigma_hi: float = _f(0.1364, "decode σ range, high end")
    replay: int = _f(64, "samples kept for the updates")
    lr: float = _f(5e-5, "LR, one cosine cycle")
    warmup: int = _f(200, "opt steps")
    dino: str = _f(DINO_REPO, "DINOv3 repo of the DINO term (gated on the Hub)")
    lambda_dino: float = _f(
        331.6, "DINO term on the alpha as three channels (the ×1 weight); 0 = off"
    )
    dino_tiles: int = _f(4, "224-px tiles per update")
    seed: int = _f(0, "output layer, sample order, σ, noise, replay draws")
    eval_every: int = _f(250, "rollouts")
    n_sheets: int = _f(8, "sheets at the end")
    compile: bool = _f(True, "compile the DiT, the head and the DINO term")
    dit: str = _f(_CKPTS.dit, "frozen DiT")
    stream: str = _f("fp32", "DiT residual stream", choices=("fp32", "bf16"))
    attn_mode: str = _f("flash", "DiT attention")
    device: str = _f("cuda", "torch device")


def run(req: TrainAlphaRequest) -> Path:
    import torch

    from library.anima.pixel_alpha import AlphaHead, AlphaTrainer, save_alpha_head
    from scripts.pixel_head import _alpha_cache as C
    from library.anima.pixel_head import load_pixel_head
    from library.anima.pixel_head_train import DinoFeat, DiTRollout
    from library.anima.pixel_head_train_x1 import X1TrainConfig
    from library.datasets.buckets import cluster_token_bands
    from library.runtime.progress import run_scope
    from scripts.pixel_head import _alpha_read as read
    from scripts.pixel_head._setup import load_dit, open_sink

    t0 = time.perf_counter()
    root = Path(resolve_under_home(req.cache_dir))
    recipe, sets = C.read_manifest(root)
    rows = sets["train"]
    if not rows:
        raise ValueError(f"{root}: no train rows")
    out_dir = Path(resolve_under_home(req.output_dir)) / req.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = open_sink(req, out_dir, "pixel_head_alpha")
    device = torch.device(req.device)

    read_rows = [r for s in C.READ_SETS for r in sets[s]]
    bands = cluster_token_bands([r.tokens for r in rows + read_rows])
    anima, _ = load_dit(req, device, bands)
    rollout = DiTRollout(anima, None, device, recipe["seed"])

    def forward(latent, cross, noise: int, sigma: float):
        return rollout.forward(
            latent.to(device).float(), cross.to(device), noise, sigma
        )[:2]

    # --- the read sets: features once, kept on the CPU ---
    samples = []
    for row in read_rows:
        cut = C.read_cutout(root, row)
        comp, bg, bg_rng = C.compose(row, cut, recipe["palette"], recipe["seed"])
        sigma = float(bg_rng.uniform(req.sigma_lo, req.sigma_hi))
        f, lat = forward(*C.read_inputs(root, row), row.noise, sigma)
        f, lat = f.to("cpu", torch.bfloat16), lat.cpu()
        samples.append(read.to_sample(row, cut, comp, bg, sigma, f, lat))
    base = load_pixel_head(resolve_under_home(req.init), device)
    for s in samples:
        s["rgb"] = read.predict(base, s, device)  # the colour the decode unmixes

    head = AlphaHead.from_base(base, req.seed)
    del base
    cfg = X1TrainConfig(
        rollouts=req.rollouts,
        opt_steps=req.opt_steps,
        replay=req.replay,
        lr=req.lr,
        warmup=req.warmup,
        lambda_dino=req.lambda_dino,
        dino_tiles=req.dino_tiles,
        seed=req.seed,
    )
    dn = (
        DinoFeat(device, req.dino, compile=req.compile) if req.lambda_dino > 0 else None
    )
    trainer = AlphaTrainer(head, cfg, dino=dn, compile=req.compile)

    rng, order = random.Random(req.seed + 1), []
    sigma_rng = random.Random(req.seed + 4)
    noise0 = 1_000_000 * (req.seed + 1)  # above the read rows' noise indices
    while len(order) < req.rollouts:
        epoch = rows[:]
        rng.shuffle(epoch)
        order += epoch
    order = order[: req.rollouts]
    n_params = sum(p.numel() for p in head.parameters())
    print(
        f"train rows {len(rows)} ({len({r.key for r in rows})} cutouts), read "
        f"{len(samples)}; whole head {n_params / 1e6:.1f}M; LR {req.lr:g}, "
        f"{req.rollouts} rollouts x {req.opt_steps}, DINO {req.lambda_dino:g}  "
        f"(ready in {time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

    # --- train ---
    def load(row):
        a = torch.from_numpy(C.read_alpha(root, row.key).copy())
        return *C.read_inputs(root, row), a.float()[None] / 127.5 - 1

    history: list[dict] = []
    ema: dict[str, float] = {}
    done = 0
    waited = 0.0  # on the disk
    t1 = time.perf_counter()
    path = out_dir / "alpha_head.safetensors"
    stamp = {
        "ls_init": "1.0",
        "sigma": f"{req.sigma_lo}-{req.sigma_hi}",
        "stream": req.stream,
        "lambda_dino": req.lambda_dino,
        "lr": req.lr,
        "rollouts": req.rollouts,
        "init": req.init,
        "run": req.output_name,
    }
    with ThreadPoolExecutor(3) as ex, run_scope(sink, final_step=lambda: done):
        ahead = deque(ex.submit(load, row) for row in order[:8])
        for r in range(req.rollouts):
            t_wait = time.perf_counter()
            latent, cross, tgt = ahead.popleft().result()
            waited += time.perf_counter() - t_wait
            if r + 8 < req.rollouts:
                ahead.append(ex.submit(load, order[r + 8]))
            sigma = sigma_rng.uniform(req.sigma_lo, req.sigma_hi)
            f, lat = forward(latent, cross, noise0 + r, sigma)
            trainer.push(f, lat, tgt.to(device), real=True)
            for k, v in trainer.update().items():
                ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
            done = r + 1
            meters = dict(ema)
            if trainer.dino_ema is not None:
                meters["dino"] = trainer.dino_ema
            meters["disk_wait"] = waited / done
            sink.step(meters, global_step=done)
            if done % req.eval_every == 0 or done == req.rollouts:
                if samples:
                    preds = [read.predict(head, s, device) for s in samples]
                    ev = read.quick_l1(samples, preds)
                    history.append({"rollout": done, **ev})
                    sink.val(ev, global_step=done)
                save_alpha_head(head, path, stamp | {"rollouts_done": done})
                sink.ckpt(global_step=done, path=str(path))
    train_sec = time.perf_counter() - t1

    # --- the scores (the file is written at the last read) ---
    scores, sheets = [], []
    if samples:
        preds = [read.predict(head, s, device) for s in samples]
        with ThreadPoolExecutor(6) as ex:
            scores = [r for rs in ex.map(read.score, samples, preds) for r in rs]
        sheets = read.write_sheets(out_dir, samples, preds, req.n_sheets)
    result = {
        "request": dataclasses.asdict(req),
        "cache_recipe": {k: v for k, v in recipe.items() if k != "palette"},
        "train_rows": len(rows),
        "train_sec": train_sec,
        "disk_wait_sec": waited,
        "history": history,
        "summary": read.summarize(scores, ("name", "set")),
        "summary_bg": read.summarize(scores, ("name", "set", "bg")),
        "per_image": scores,
        "sheets": sheets,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    for g, d in result["summary"].items():
        print(g, "  ".join(f"{k} {v['mean']:.4f}" for k, v in d.items()), flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s → {out_dir}", flush=True)
    return path


def main() -> None:
    req = TrainAlphaRequest.from_argv()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(req)


if __name__ == "__main__":
    main()
