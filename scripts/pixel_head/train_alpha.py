#!/usr/bin/env python3
"""Train the alpha head on cached features: the head and LPIPS on the card, nothing else.

The features come from ``scripts/pixel_head/cache_alpha.py``; no DiT, VAE or TE is
loaded here and no image folder is read. The whole head trains from the ×1 base (its
output layer fresh at one channel) on the ×1 trainer's loop: one cached sample per
rollout goes into the replay buffer, ``opt_steps`` updates each.

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

from library.env import resolve_under_home
from library.lines.requests import Request
from library.lines.requests import request_field as _f


@dataclass
class TrainAlphaRequest(Request):
    """Train the alpha head on cached features."""

    output_name: str = _f("alpha", "the run's name")
    output_dir: str = _f("output/pixel_head", "→ <output_dir>/<output_name>/")
    cache_dir: str = _f("post_image_dataset/pixel_alpha", "cache_alpha.py's output")
    init: str = _f(
        "output/pixel_head/x1ll_1122_1000/head.safetensors", "the ×1 base head"
    )
    rollouts: int = _f(1000, "cached samples pushed, opt_steps updates each")
    opt_steps: int = _f(16, "head updates per rollout")
    replay: int = _f(64, "samples kept for the updates")
    lr: float = _f(5e-5, "LR, one cosine cycle")
    warmup: int = _f(200, "opt steps")
    lambda_lpips: float = _f(0.05, "LPIPS on the alpha as three channels; 0 = off")
    lpips_crop: int = _f(512, "LPIPS window, px; 0 = the whole image")
    seed: int = _f(0, "output layer, sample order, replay draws")
    eval_every: int = _f(250, "rollouts")
    n_sheets: int = _f(8, "sheets at the end")
    log_every: int = _f(25, "rollouts")
    compile: bool = _f(True, "compile the head and LPIPS")
    device: str = _f("cuda", "torch device")


def run(req: TrainAlphaRequest) -> Path:
    import torch

    from library.anima.pixel_alpha import AlphaHead, AlphaTrainer, save_alpha_head
    from library.anima.pixel_alpha_cache import READ_SETS, read_manifest, read_row
    from library.anima.pixel_head import load_pixel_head
    from library.anima.pixel_head_train import Lpips
    from library.anima.pixel_head_train_x1 import X1TrainConfig
    from library.runtime.progress import run_scope
    from scripts.pixel_head import _alpha_read as read
    from scripts.pixel_head._setup import open_sink

    t0 = time.perf_counter()
    root = Path(resolve_under_home(req.cache_dir))
    recipe, sets = read_manifest(root)
    rows = sets["train"]
    if not rows:
        raise ValueError(f"{root}: no train rows")
    out_dir = Path(resolve_under_home(req.output_dir)) / req.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(req.device)

    # --- the read sets, on the CPU ---
    read_rows = [r for s in READ_SETS for r in sets[s]]
    with ThreadPoolExecutor(6) as ex:
        samples = list(
            ex.map(lambda r: read.to_sample(r, *read_row(root, r)), read_rows)
        )
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
        lambda_lpips=req.lambda_lpips,
        lpips_crop=req.lpips_crop,
        seed=req.seed,
    )
    lp = Lpips(device, compile=req.compile) if req.lambda_lpips > 0 else None
    trainer = AlphaTrainer(head, cfg, lpips=lp, compile=req.compile)

    rng, order = random.Random(req.seed + 1), []
    while len(order) < req.rollouts:
        epoch = rows[:]
        rng.shuffle(epoch)
        order += epoch
    order = order[: req.rollouts]
    n_params = sum(p.numel() for p in head.parameters())
    print(
        f"train rows {len(rows)} ({len({r.key for r in rows})} cutouts), read "
        f"{len(samples)}; whole head {n_params / 1e6:.1f}M; LR {req.lr:g}, "
        f"{req.rollouts} rollouts x {req.opt_steps}, LPIPS {req.lambda_lpips:g}  "
        f"(ready in {time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

    # --- train ---
    def load(row):
        tensors, _ = read_row(root, row)
        tgt = tensors["alpha"].float()[None] / 127.5 - 1
        return tensors["f"], tensors["lat"], tgt

    sink = open_sink(req, out_dir, "pixel_head_alpha")
    history: list[dict] = []
    ema: dict[str, float] = {}
    done = 0
    waited = 0.0  # on the disk
    t1 = time.perf_counter()
    t_log, n_log = t1, 0
    with ThreadPoolExecutor(3) as ex, run_scope(sink, final_step=lambda: done):
        ahead = deque(ex.submit(load, row) for row in order[:8])
        for r in range(req.rollouts):
            t_wait = time.perf_counter()
            f, lat, tgt = ahead.popleft().result()
            waited += time.perf_counter() - t_wait
            if r + 8 < req.rollouts:
                ahead.append(ex.submit(load, order[r + 8]))
            trainer.push(f.to(device), lat.to(device), tgt.to(device), real=True)
            for k, v in trainer.update().items():
                ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
            done = r + 1
            if done % req.log_every == 0 or done == req.rollouts:
                now = time.perf_counter()
                pace = (now - t_log) / (done - n_log)
                t_log, n_log = now, done
                meters = dict(ema)
                if trainer.lpips_ema is not None:
                    meters["lpips"] = trainer.lpips_ema
                print(
                    f"  rollout {done}/{req.rollouts}  "
                    + "".join(f"{k} {v:.4f}  " for k, v in meters.items())
                    + f"{pace:.2f}s/rollout  disk wait {waited / done:.3f}s  "
                    f"eta {(req.rollouts - done) * pace / 60:.1f}m",
                    flush=True,
                )
                if sink is not None:
                    logged = {f"loss/{k}": v for k, v in meters.items()}
                    sink.log(logged, global_step=done, epoch=0)
            if samples and (done % req.eval_every == 0 or done == req.rollouts):
                preds = [read.predict(head, s, device) for s in samples]
                ev = read.quick_l1(samples, preds)
                history.append({"rollout": done, **ev})
                shown = "  ".join(f"{k} {v:.4f}" for k, v in ev.items())
                print(f"  eval @ {done}: {shown}", flush=True)
                t_log = time.perf_counter()  # the pace leaves the read out
    train_sec = time.perf_counter() - t1

    # --- the file first, then the scores ---
    path = out_dir / "alpha_head.safetensors"
    lo, hi = recipe["sigma"]
    stamp = {
        "ls_init": "1.0",
        "sigma": f"{lo}-{hi}",
        "stream": recipe["stream"],
        "lambda_lpips": req.lambda_lpips,
        "lr": req.lr,
        "rollouts": req.rollouts,
        "init": req.init,
        "run": req.output_name,
    }
    save_alpha_head(head, path, stamp)
    print(f"saved {path}", flush=True)

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
