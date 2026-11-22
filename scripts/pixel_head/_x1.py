"""The ×1 run behind ``base_x1.py`` and ``train_x1.py``: request, loop and read.

The pieces are in ``library.anima.pixel_head_train_x1``; the two entry points differ in
the request's defaults and the finish's GAN fields.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from library.env import default_checkpoints, resolve_under_home
from library.lines.requests import Request
from library.lines.requests import request_field as _f
from scripts.pixel_head._setup import load_models, open_sink

_CKPTS = default_checkpoints()
TURBO = "output/ckpt/release/anima_turbo_4step_v4_longer.safetensors"


@dataclass
class X1Request(Request):
    """The ×1 recipe's fields."""

    output_name: str = _f("x1", "the run's name")
    output_dir: str = _f("output/pixel_head", "→ <output_dir>/<output_name>/")
    cache_root: str = _f("post_image_dataset/lora", "cached latent + TE pairs")
    pages: str = _f(
        "post_image_dataset/pixel_pages",
        "build_pages.py's output (train/, eval/); empty = no pages",
    )
    page_share: float = _f(0.33, "P(rollout is a page)")
    page_caption: str = _f(
        "monochrome, greyscale, comic, speech bubble, japanese text",
        "the one caption every page is encoded with",
    )
    init: str = _f("", "head file to start from; empty = a fresh head")
    width: int = _f(1024, "fresh head: channels at token spacing")
    depth: int = _f(3, "fresh head: ConvNeXt blocks, the upsampled ones included")
    up: int = _f(1, "fresh head: blocks after the one 2x upsample")
    points: str = _f(
        "0.0577-0.1364:real 0.5:adapter",
        "decode points, in turn per rollout: SIGMA[-SIGMA][:adapter][:real]",
    )
    target: str = _f(
        "xhat0",
        "target of the points without :real: the VAE's decode of x0-hat or of x0",
        choices=("xhat0", "x0"),
    )
    adapter: str = _f(TURBO, "LoRA switched on at the :adapter points")
    eval_points: str = _f("0.03 0.25", "read, not trained on: SIGMA[:adapter]")
    # the loop (library.anima.pixel_head_train_x1.X1TrainConfig)
    rollouts: int = _f(12000, "DiT forwards")
    opt_steps: int = _f(16, "head updates per rollout")
    replay: int = _f(64, "rollouts kept for the updates")
    lr: float = _f(5e-5, "head LR, one cosine cycle")
    warmup: int = _f(200, "opt steps")
    lambda_lpips: float = _f(0.05, "LPIPS weight on the :real points")
    lpips_crop: int = _f(512, "LPIPS window, px; 0 = the whole image")
    seed: int = _f(0, "also moves the held-out noise")
    # the read
    eval_every: int = _f(1000, "rollouts")
    start_eval: bool = _f(False, "read the starting head before training")
    n_sheets: int = _f(4, "test sheets per point (text and manga: all)")
    # runtime
    dit: str = _f(_CKPTS.dit, "frozen DiT")
    vae: str = _f(_CKPTS.vae, "VAE of the targets")
    text_encoder: str = _f(_CKPTS.text_encoder, "encodes page_caption")
    stream: str = _f("fp32", "DiT residual stream", choices=("fp32", "bf16"))
    attn_mode: str = _f("flash", "DiT attention")
    compile: bool = _f(True, "DiT blocks, head and LPIPS (the PE tower runs eager)")
    device: str = _f("cuda", "torch device")


def run(req: X1Request) -> Path:
    import numpy as np
    import torch

    from library.anima.pixel_head import (
        LAT_PATCH_DIM,
        PixelHead,
        PixelHeadX2,
        load_pixel_head,
    )
    from library.anima.pixel_head_train import (
        Lpips,
        PEDisc,
        densest_window,
        psnr,
        zoom_sheet,
    )
    from library.anima.pixel_head_train_x1 import (
        X1Rollout,
        X1TrainConfig,
        X1Trainer,
        held_out_split,
        item_tokens,
        parse_points,
        save_x1_head,
    )
    from library.datasets.buckets import cluster_token_bands
    from library.inference.text import prepare_text_inputs
    from library.runtime.progress import run_scope

    # base_x1's request has no pegan_* fields: the term's default is off
    fields = {f.name for f in dataclasses.fields(req)}
    cfg = X1TrainConfig(
        **{
            f.name: getattr(req, f.name)
            for f in dataclasses.fields(X1TrainConfig)
            if f.name in fields
        }
    )
    train_points, read_points = parse_points(req.points, req.eval_points, req.target)
    assert req.adapter or not any(p.adapter for p in (*train_points, *read_points)), (
        "an :adapter point needs --adapter"
    )
    assert cfg.pegan_ratio == 0 or any(p.target == "real" for p in train_points), (
        "pegan_ratio needs a :real point: the original image is the term's real"
    )
    device = torch.device(req.device)
    torch.manual_seed(req.seed)
    out_dir = Path(resolve_under_home(req.output_dir)) / req.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    data = held_out_split(Path(resolve_under_home(req.cache_root)))
    train, test, text = data["train"], data["test"], data["text"]
    page_train: list[Path] = []
    page_eval: list[Path] = []
    if req.pages:
        root = Path(resolve_under_home(req.pages))
        page_train = sorted((root / "train").glob("*.safetensors"))
        page_eval = sorted((root / "eval").glob("*.safetensors"))
    seq_bands = cluster_token_bands(
        [item_tokens(it) for it in train + test + text + page_train + page_eval]
    )
    print(
        f"train {len(train)}  test {len(test)}  text {len(text)}  "
        f"pages {len(page_train)} / eval {len(page_eval)}  bands {seq_bands}  "
        f"points {', '.join(p.name for p in read_points)}",
        flush=True,
    )

    # --- models: VAE + frozen DiT (the adapter switched per point) ---
    vae, anima, network = load_models(req, device, seq_bands, adapter=req.adapter)
    page_cross = None
    if req.pages:
        arg_c, _ = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=req.page_caption,
            text_encoder_path=str(resolve_under_home(req.text_encoder)),
        )
        page_cross = arg_c["embed"][0].to(device, torch.bfloat16)
    rollout = X1Rollout(anima, vae, device, req.seed, network, page_cross)

    # --- head ---
    d_feat = anima.final_layer.hidden_size
    if req.init:
        src = Path(resolve_under_home(req.init))
        head = load_pixel_head(src, device, d_feat)
        assert not isinstance(head, PixelHeadX2), f"--init {src}: a ×1 head file"
        ls_init = head.meta.get("ls_init", "1.0")
    else:
        head = PixelHead(
            d_feat,
            LAT_PATCH_DIM,
            req.width,
            req.depth,
            ls_init=1.0,
            use_skip=False,
            up=req.up,
        ).to(device)
        ls_init = "1.0"
    head.train()
    n_params = sum(q.numel() for q in head.parameters())
    print(f"head {n_params / 1e6:.1f}M params from {req.init or 'fresh'}", flush=True)
    lp = Lpips(device, compile=req.compile)
    disc = None
    if cfg.pegan_ratio > 0:
        disc = PEDisc(device, lr=req.pegan_lr)  # the tower runs eager at any size
    trainer = X1Trainer(head, cfg, lpips=lp, disc=disc, compile=req.compile)

    # --- held-out: features once, on the CPU; built at the first read ---
    held_items = (
        [(ci, "test") for ci in test]
        + [(ci, "text") for ci in text]
        + [(page, "manga") for page in page_eval]
    )
    held = []  # (item, set, point, f, x_t patches, {target: decode}, original)
    sets: list[str] = []  # <point>.<set>

    def build_held() -> None:
        t = time.perf_counter()
        origs: dict[int, torch.Tensor] = {}
        for pt in read_points:  # the same noise per item at every point
            for j, (item, key) in enumerate(held_items):
                f, lat, tgts = rollout(
                    item, 10_000 + j, pt.sigma, pt.adapter, ("x0", "xhat0")
                )
                if j not in origs:  # one copy per item, shared by its points
                    hw = tuple(tgts["x0"].shape[-2:])
                    origs[j] = rollout.real(item, hw).to("cpu", torch.float16)
                held.append(
                    (
                        item,
                        key,
                        pt.name,
                        f.to("cpu", torch.bfloat16),
                        lat.cpu(),
                        {k: v.to("cpu", torch.float16) for k, v in tgts.items()},
                        origs[j],
                    )
                )
        sets.extend(dict.fromkeys(f"{h[2]}.{h[1]}" for h in held))
        print(
            f"held-out features {len(held)} in {time.perf_counter() - t:.0f}s",
            flush=True,
        )

    @torch.no_grad()
    def predict(f, lat) -> torch.Tensor:
        f, lat = f.to(device).float(), lat.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            pred = head(f, lat)
        return pred.float().clamp(-1, 1)

    def read(refs: bool = False) -> list[dict]:
        """Per held-out image and point: the head against VAE(x0), VAE(x̂0) and the
        original; ``refs`` adds the VAE's own columns against the original."""
        if not held:
            build_held()
        head.eval()
        rows = []
        for item, key, pt, f, lat, tgts, orig in held:
            pred, o = predict(f, lat), orig.to(device).float()
            tgt, hat = (tgts[k].to(device).float() for k in ("x0", "xhat0"))
            row = {
                "stem": item.stem,
                "artist": "pages"
                if isinstance(item, Path)
                else Path(item.npz_path).parent.name,
                "set": f"{pt}.{key}",
                "point": pt,
                "psnr_head": psnr(pred, tgt),
                "psnr_head_xhat0": psnr(pred, hat),
                "psnr_head_vs_orig": psnr(pred, o),
                "lpips_head_vs_orig": lp.image(pred, o),
            }
            if refs:
                row |= {
                    "psnr_vae_xhat0": psnr(hat, tgt),
                    "psnr_vae_vs_orig": psnr(tgt, o),
                    "lpips_vae_vs_orig": lp.image(tgt, o),
                    # the VAE path at this point: the decode of the DiT's own x̂0
                    "psnr_vaehat_vs_orig": psnr(hat, o),
                    "lpips_vaehat_vs_orig": lp.image(hat, o),
                }
            rows.append(row)
        head.train()
        return rows

    def by_set(rows: list[dict]) -> dict[str, float]:
        cols = [k for k in rows[0] if k.startswith(("psnr_", "lpips_"))]
        return {
            f"{st}_{col}": float(np.mean([r[col] for r in rows if r["set"] == st]))
            for st in sets
            for col in cols
        }

    def fmt(ev: dict[str, float]) -> str:
        return "  ".join(
            f"{st} {ev[f'{st}_psnr_head']:.2f} "
            f"(x̂0 {ev[f'{st}_psnr_head_xhat0']:.2f}, "
            f"orig {ev[f'{st}_psnr_head_vs_orig']:.2f} "
            f"lpips {ev[f'{st}_lpips_head_vs_orig']:.4f})"
            for st in sets
        )

    # --- train ---
    sink = open_sink(req, out_dir, "pixel_head_x1")
    order_rng = random.Random(req.seed + 1)
    page_rng = random.Random(req.seed + 2)  # own stream: colour order as without pages
    sig_rng = random.Random(req.seed + 4)
    order: list = []
    page_order: list = []
    n_page_rollouts = 0
    history: list[dict] = []
    if req.start_eval:
        ev = by_set(read())
        history.append({"rollout": 0, "step": 0, **ev})
        print(f"  eval @ 0 (as loaded): {fmt(ev)}", flush=True)
    ema: dict[str, float] = {}
    done = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with run_scope(sink, final_step=lambda: done):
        for r in range(cfg.rollouts):
            if page_train and page_rng.random() < req.page_share:
                if not page_order:
                    page_order = page_train[:]
                    page_rng.shuffle(page_order)
                item = page_order.pop()
                n_page_rollouts += 1
            else:
                if not order:
                    order = train[:]
                    order_rng.shuffle(order)
                item = order.pop()
            pt = train_points[r % len(train_points)]
            sg = pt.lo if pt.hi == pt.lo else sig_rng.uniform(pt.lo, pt.hi)
            f, lat, tgts = rollout(item, r, sg, pt.adapter, (pt.target,))
            trainer.push(f, lat, tgts[pt.target], pt.target == "real")
            for k, v in trainer.update().items():
                ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
            done = r + 1
            if done % 50 == 0:
                el = time.perf_counter() - t0
                meters = dict(trainer.gan_log)
                if trainer.lpips_ema is not None:
                    meters["lpips"] = trainer.lpips_ema
                print(
                    f"  rollout {done}/{cfg.rollouts}  loss {ema['loss']:.4f}  "
                    f"psnr~{10 * math.log10(4 / ema['mse']):.2f}  "
                    + "".join(f"{k} {v:.4g}  " for k, v in meters.items())
                    + (
                        f"gan steps {trainer.gan_steps}  "
                        f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB  "
                        if disc is not None
                        else ""
                    )
                    + f"{el / done:.2f}s/rollout  "
                    f"eta {(cfg.rollouts - done) * el / done / 60:.0f}m",
                    flush=True,
                )
                if sink is not None:
                    sink.log(
                        {f"loss/{k}": v for k, v in (ema | meters).items()},
                        global_step=done,
                        epoch=0,
                    )
            if done % req.eval_every == 0 or done == cfg.rollouts:
                ev = by_set(read())
                history.append({"rollout": done, "step": trainer.step, **ev})
                print(f"  eval @ {done}: {fmt(ev)}", flush=True)
    sec_per_rollout = (time.perf_counter() - t0) / max(1, cfg.rollouts)

    # --- the file, then rows and sheets ---
    stamp = {
        "ls_init": ls_init,
        "sigma": ",".join(
            f"{p.lo}" if p.lo == p.hi else f"{p.lo}-{p.hi}" for p in train_points
        ),
        "target": req.target,
        "points": req.points,
        "lambda_lpips": req.lambda_lpips,
        "stream": req.stream,
        "replay": req.replay,
        "adapter": req.adapter,
        "init": req.init,
        "rollouts": req.rollouts,
        "run": req.output_name,
    }
    if disc is not None:
        stamp |= {
            "pegan_ratio": cfg.pegan_ratio,
            "pegan_crop": cfg.pegan_crop,
            "pegan_lr": req.pegan_lr,
        }
    head_path = out_dir / "head.safetensors"
    save_x1_head(head, head_path, stamp)
    rows = read(refs=True)
    head.eval()
    counters: dict[tuple[str, str], int] = {}
    sheets = []
    for item, key, pt, f, lat, tgts, orig in held:
        n = counters.get((pt, key), 0)
        counters[(pt, key)] = n + 1
        if key == "test" and n >= req.n_sheets:
            continue
        tgt = tgts["x0"].to(device).float()
        cols = [
            ("original", orig),
            ("VAE decode", tgt),
            ("head", predict(f, lat)),
        ]
        name = f"{pt}_{key}_{n:02d}_{item.stem}.png"
        at = None if key == "test" else densest_window(tgt, 384)
        zoom_sheet(cols, out_dir / name, at=at)
        sheets.append(name)
    result = {
        "request": dataclasses.asdict(req),
        "train_points": [p._asdict() for p in train_points],
        "eval_points": [p._asdict() for p in read_points],
        "n_train": len(train),
        "n_page_train": len(page_train),
        "n_page_rollouts": n_page_rollouts,
        "opt_steps_total": trainer.step,
        "pegan_steps": trainer.gan_steps,
        "pegan_last": trainer.gan_log,
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
