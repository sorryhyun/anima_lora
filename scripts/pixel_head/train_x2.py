#!/usr/bin/env python3
"""Train a ×2 pixel head on the ``x2flex`` set (pieces: ``library.anima.pixel_head_train``).

A frozen Anima DiT gives the features; the head (a released ×1 head plus a second 2×
stage, or a ×2 file to continue from) is trained against the real 2× image with the
recipe loss + LPIPS + the tone term, and from ``pegan_start`` the PE-Spatial GAN term.
Recipe and its records: ``project/pixel_decoder/x2_plan.md``.

Every flag is an :class:`X2TrainRequest` field; ``--config <file.toml>`` takes a flat TOML
of field names (field default → ``--config`` → CLI). Shipped defaults:
``configs/pixel_head/x2.toml``. The run's name is ``output_name`` in the config.

    make daemon-run ARGS="--queue --stall-timeout 900 scripts/pixel_head/train_x2.py \
        --config configs/pixel_head/x2.toml"

Writes ``<output_dir>/<output_name>/``: ``head_x2.safetensors``, ``result.json`` (the
request, the eval history, per-image rows) and ``original 2x | x2 head`` sheets.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from library.env import default_checkpoints, resolve_under_home
from library.lines.requests import Request
from scripts.pixel_head._setup import load_models, open_sink

_CKPTS = default_checkpoints()


@dataclass
class X2TrainRequest(Request):
    """Train a ×2 pixel head on the x2flex set."""

    output_name: str = "x2"
    output_dir: str = "output/pixel_head"
    x2flex: str = "post_image_dataset/x2flex/manifest.json"
    init: str = ""  # a ×1 head file for a new second stage; "" = the catalog's head
    resume: str = ""  # a ×2 head file to continue from (replaces init / depth2)
    depth2: int = 2  # ConvNeXt blocks of the second stage
    sigma: str = "0.0577-0.1364"  # decode-point range, drawn per rollout
    # the loop (library.anima.pixel_head_train.X2TrainConfig)
    rollouts: int = 4000
    opt_steps: int = 16
    replay: int = 64
    window: int = 24
    margin: int = 3
    lr: float = 2e-4
    warmup: int = 200
    lambda_lpips: float = 0.1
    lambda_dc: float = 1.0
    dc_block: int = 32
    pegan_ratio: float = 2.0  # 0 = no GAN term
    pegan_start: int = 1000
    pegan_lambda_max: float = 1.0
    pegan_crop: int = 512
    seed: int = 0
    # the discriminator
    pegan_layers: str = "3,7,11"  # PE blocks tapped, 0-based
    pegan_lr: float = 1e-4
    pegan_r: float = 0.1
    pegan_r_sigma: float = 0.01
    # the read
    eval_every: int = 500  # rollouts
    n_test: int = 12  # test images read
    n_sheets: int = 4  # sheets per set (every text image gets one)
    crop: int = 768  # zoom sheet side, ×2 px
    # runtime
    dit: str = _CKPTS.dit
    vae: str = _CKPTS.vae
    stream: str = "fp32"  # fp32 | bf16
    attn_mode: str = "flash"
    compile: bool = True  # DiT blocks, head, LPIPS and the discriminator's tower
    device: str = "cuda"


def run(req: X2TrainRequest) -> Path:
    import numpy as np
    import torch

    from library.anima.pixel_head import (
        PATCH,
        PixelHeadX2,
        drop_x1_output,
        load_pixel_head,
        resolve_pixel_head_path,
    )
    from library.anima.pixel_head_train import (
        DiTRollout,
        Lpips,
        PEDisc,
        X2TrainConfig,
        X2Trainer,
        densest_window,
        load_x2flex,
        pair_sheet,
        pick_window,
        psnr,
        real_x2,
        save_x2_head,
        tone_stats,
    )
    from library.datasets.buckets import cluster_token_bands
    from library.runtime.progress import run_scope

    cfg = X2TrainConfig(
        **{f.name: getattr(req, f.name) for f in dataclasses.fields(X2TrainConfig)}
    )
    device = torch.device(req.device)
    torch.manual_seed(req.seed)
    lo, _, hi = req.sigma.partition("-")
    sig_lo, sig_hi = float(lo), float(hi or lo)
    sig_eval = (sig_lo + sig_hi) / 2
    out_dir = Path(resolve_under_home(req.output_dir)) / req.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    data = load_x2flex(req.x2flex)
    train, test, text = data["train"], data["test"][: req.n_test], data["text"]
    seq_bands = cluster_token_bands([it.tokens for it in train + test + text])
    print(
        f"train {len(train)}  test {len(test)}  text {len(text)}  bands {seq_bands}  "
        f"sigma {sig_lo}-{sig_hi} (eval {sig_eval:.4g})",
        flush=True,
    )

    # --- models: VAE + frozen DiT ---
    vae, anima, _ = load_models(req, device, seq_bands)
    rollout = DiTRollout(anima, vae, device, seed=req.seed)

    # --- head ---
    if req.resume:
        src = Path(resolve_under_home(req.resume))
        head = load_pixel_head(src, device, anima.final_layer.hidden_size)
        assert isinstance(head, PixelHeadX2), f"--resume {src}: not a ×2 head file"
    else:
        src = resolve_pixel_head_path(req.init or None)
        base = load_pixel_head(src, device, anima.final_layer.hidden_size)
        assert not isinstance(base, PixelHeadX2), f"--init {src}: a ×1 head file"
        head = PixelHeadX2(base, req.depth2).to(device)
    # the ×1 loss is off in this recipe: the file carries no ×1 output (scales = 2)
    drop_x1_output(head)
    head.train()
    print(
        f"head {sum(q.numel() for q in head.parameters()) / 1e6:.1f}M params from {src}",
        flush=True,
    )
    lp = Lpips(device, compile=req.compile)
    disc = None
    if cfg.pegan_ratio > 0:
        disc = PEDisc(
            device,
            layers=tuple(int(v) for v in req.pegan_layers.split(",")),
            lr=req.pegan_lr,
            r_weight=req.pegan_r,
            r_sigma=req.pegan_r_sigma,
            compile=req.compile,
        )
    trainer = X2Trainer(head, cfg, lpips=lp, disc=disc, compile=req.compile)

    # --- held-out: whole-image features and the real 2× image, kept on the CPU ---
    held = []  # (item, set, f, x_t patches, real 2×)
    for j, (item, key) in enumerate(
        [(it, "test") for it in test] + [(it, "text") for it in text]
    ):
        f, lat, img = rollout(item, 10_000 + j, sig_eval)
        ref = real_x2(item, img)
        held.append(
            (item, key, f.to("cpu", torch.bfloat16), lat.cpu(), ref.to("cpu").half())
        )
    sets = list(dict.fromkeys(h[1] for h in held))

    @torch.no_grad()
    def predict(f, lat) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x2 = head(f.to(device).float(), lat.to(device), 2)
        return x2.float().clamp(-1, 1)

    def read() -> list[dict]:
        """Per held-out image: ×2 output against the real 2× image."""
        head.eval()
        rows = []
        for item, key, f, lat, ref in held:
            x2, ref = predict(f, lat), ref.to(device).float()
            luma, dark = tone_stats(x2, ref)
            rows.append(
                {
                    "stem": item.stem,
                    "set": key,
                    "kind": item.kind,
                    "psnr": psnr(x2, ref),
                    "lpips": lp.image(x2, ref),
                    "luma": luma,
                    "dark": dark,
                }
            )
        head.train()
        return rows

    def by_set(rows: list[dict]) -> dict[str, float | None]:
        out = {}
        for st in sets:
            for col in ("psnr", "lpips", "luma", "dark"):
                v = [r[col] for r in rows if r["set"] == st and r[col] is not None]
                out[f"{st}_{col}"] = float(np.mean(v)) if v else None
        return out

    def fmt(ev) -> str:
        return "  ".join(
            f"{st} {ev[f'{st}_psnr']:.2f} dB lpips {ev[f'{st}_lpips']:.4f} "
            f"dark {ev[f'{st}_dark'] or 0:+.2f}"
            for st in sets
        )

    # --- train ---
    sink = open_sink(req, out_dir, "pixel_head_x2")
    order: list = []
    order_rng = random.Random(req.seed + 1)
    sig_rng = random.Random(req.seed + 4)
    win_rng = random.Random(req.seed + 5)
    ema: dict[str, float] = {}
    history: list[dict] = []
    done = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with run_scope(sink, final_step=lambda: done):
        for r in range(cfg.rollouts):
            if not order:
                order = train[:]
                order_rng.shuffle(order)
            item = order.pop()
            sg = sig_lo if sig_hi == sig_lo else sig_rng.uniform(sig_lo, sig_hi)
            f, lat, img = rollout(item, r, sg)
            win = pick_window(*f.shape[:2], cfg.window, cfg.margin, win_rng)
            trainer.push(f, lat, real_x2(item, img), win)
            if r == cfg.pegan_start and disc is not None:
                print(f"  PE-GAN term on from rollout {r}", flush=True)
            for k, v in trainer.update(r).items():
                ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
            done = r + 1
            if done % 50 == 0:
                el = time.perf_counter() - t0
                print(
                    f"  rollout {done}/{cfg.rollouts}  "
                    + " ".join(f"{k} {v:.4g}" for k, v in ema.items())
                    + f"  {el / done:.2f}s/rollout  "
                    f"eta {(cfg.rollouts - done) * el / done / 60:.0f}m",
                    flush=True,
                )
                if sink is not None:
                    sink.log(
                        {f"loss/{k}": v for k, v in ema.items()},
                        global_step=done,
                        epoch=0,
                    )
            if done % req.eval_every == 0 or done == cfg.rollouts:
                ev = by_set(read())
                history.append({"rollout": done, "step": trainer.step, **ev})
                print(f"  eval @ {done}: {fmt(ev)}", flush=True)
    sec_per_rollout = (time.perf_counter() - t0) / max(1, cfg.rollouts)

    # --- read: rows, sheets, the file ---
    rows = read()
    head.eval()
    counters: dict[str, int] = {}
    sheets = []
    for item, key, f, lat, ref in held:
        n = counters.get(key, 0)
        counters[key] = n + 1
        if n >= req.n_sheets and key != "text":
            continue
        x2, ref = predict(f, lat), ref.to(device).float()
        cols = [("original 2x", ref), ("x2 head", x2)]
        stem = f"{key}_{n:02d}_{item.stem}"
        pair_sheet(cols, out_dir / f"{stem}_full.png")
        cx, cy = densest_window(ref, req.crop)
        pair_sheet(cols, out_dir / f"{stem}_crop.png", at=(cx, cy, req.crop))
        sheets += [f"{stem}_full.png", f"{stem}_crop.png"]
    head_path = out_dir / "head_x2.safetensors"
    save_x2_head(
        head,
        head_path,
        {
            "sigma": req.sigma,
            "stream": req.stream,
            "source": Path(src).name,
            "x2_target": "real",
            "pegan_ratio": req.pegan_ratio,
            "pegan_start": req.pegan_start,
            "rollouts": req.rollouts,
            "run": req.output_name,
        },
    )
    result = {
        "request": dataclasses.asdict(req),
        "n_train": len(train),
        "patch": PATCH,
        "sec_per_rollout": sec_per_rollout,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "final": by_set(rows),
        "eval_history": history,
        "per_image": rows,
        "sheets": sheets,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"final: {fmt(result['final'])}", flush=True)
    print(
        f"{sec_per_rollout:.2f} s / rollout, peak {result['peak_vram_gb']:.1f} GB; "
        f"head {head_path}",
        flush=True,
    )
    return head_path


def main() -> None:
    req = X2TrainRequest.from_argv()
    # read when the CUDA allocator initializes: before torch is imported in run()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(req)


if __name__ == "__main__":
    main()
