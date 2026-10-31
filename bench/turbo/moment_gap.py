#!/usr/bin/env python3
"""How far apart are the dataset, CFG 1, CFG 4 and the turbo student in pooled moments?

Reads IDM's feature-moment loss (arXiv:2610.39633 Eq. 13) as a ruler, on the int8 base.
For the same N dataset items, same noise, same resolution:

    D     the cached dataset latent
    T1    teacher, CFG 1, dataset caption
    T4    teacher, CFG 4, dataset caption
    T1na  teacher, CFG 1, caption without its artist tag (proxy for the teacher's wider style)
    S     turbo student delta, 4 steps, CFG 1, dataset caption

Each set goes through one feature map: the teacher's last block, mean-pooled, on the
item's caption and a fixed noise per item, so a gap between two sets comes from the
images alone. Latents and decoded pixels are read the same way.

    W2² = ‖μ_a − μ_b‖² + ‖σ_a − σ_b‖², over pooled per-coordinate moments

``nw2`` divides it by D's summed per-coordinate variance. ``null_p95`` swaps a/b per item
at random, so a pair below it is not separable at this N. ``pull_cos`` asks whether
moving S toward D also moves it toward T4.

    make daemon-run ARGS="--stall-timeout 900 bench/turbo/moment_gap.py --n 128"
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from anime_tools.captions.position_clauses import (
    compose_caption,
    is_artist_tag,
    parse_caption,
)
from anime_tools.captions.variants import read_variants_sidecar, variants_sidecar_path
from PIL import Image

from anima_lora import GenerationRequest
from bench._anima import add_model_args
from bench._common import make_run_dir, write_result
from library.anima.uncond import default_uncond_path, load_uncond_crossattn
from library.env import resolve_under_home
from library.inference.models import load_dit_model, load_shared_models
from library.inference.sampling import get_timesteps_sigmas
from library.inference.text import prepare_text_inputs
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from library.models.qwen_vae import load_vae
from library.runtime.device import clean_memory_on_device

SETS = ("D", "T1", "T4", "T1na", "S")
PAIRS = (
    ("T1", "T4"),  # the CFG difference, teacher only
    ("D", "T1"),  # model against data, same guidance
    ("D", "T4"),  # what a data reference would see
    ("T1", "T1na"),  # the artist range, teacher only
    ("D", "T1na"),  # the data range against the wider teacher
    ("S", "T4"),  # the student's distance to its target: the yardstick
    ("S", "D"),
)
PIXEL_STATS = ("saturation", "luma_mean", "luma_std", "luma_p99_p1")
STUDENT_LORA = "models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors"


def pick_items(data_dir: Path, resized_dir: Path, n: int, seed: int) -> list[dict]:
    """A uniform draw of cached images, each with its v0 caption with and without artist."""
    pairs = discover_cached_pairs(str(data_dir))
    random.Random(seed).shuffle(pairs)
    items = []
    for pair in pairs:
        rel = Path(pair.te_path).parent.relative_to(data_dir)
        sidecar = variants_sidecar_path(resized_dir / rel / f"{pair.stem}.png")
        if not sidecar.is_file():
            continue
        caption = read_variants_sidecar(sidecar)[0][1]
        parsed = parse_caption(caption)
        kept = [t for t in parsed.flat_tags if not is_artist_tag(t)]
        items.append(
            {
                "stem": pair.stem,
                "group": str(rel),
                "npz": pair.npz_path,
                "te": pair.te_path,
                "caption": caption,
                "noartist": compose_caption(kept, parsed.clauses),
            }
        )
        if len(items) == n:
            break
    return items


def make_args(opts, device, lora: str | None) -> argparse.Namespace:
    req = GenerationRequest(
        dit=opts.dit,
        vae=opts.vae,
        text_encoder=opts.text_encoder,
        prompt="",
        save_path="output/tests/_moment_gap.png",
        infer_steps=opts.teacher_steps,
        guidance_scale=opts.teacher_cfg,
        image_size=(1024, 1024),
        seed=opts.seed,
    )
    args = req.to_args()
    args.device = device
    args.dtype = "bf16"
    args.attn_mode = getattr(args, "attn_mode", "flash") or "flash"
    args.compile = False
    args.compile_blocks = False
    args.compile_dynamic_seq = False
    args.int8 = True
    args.int8_file = opts.int8_file
    args.lora_weight = [lora] if lora else None
    args.lora_multiplier = 1.0
    return args


def velocity(model, x, sigma, c, **kw):
    pad = torch.zeros(1, 1, *x.shape[-2:], dtype=torch.bfloat16, device=x.device)
    t = torch.full((1,), float(sigma), dtype=torch.bfloat16, device=x.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.forward_mini_train_dit(
            x.to(torch.bfloat16).unsqueeze(2),
            t,
            c,
            padding_mask=pad,
            skip_pooled_text_proj=True,
            **kw,
        )


def rollout(model, eps, c, c_null, sigmas, cfg: float) -> torch.Tensor:
    x = eps
    for s_i, s_next in zip(sigmas[:-1], sigmas[1:]):
        v = velocity(model, x, s_i, c).squeeze(2).float()
        if cfg != 1.0:
            v_u = velocity(model, x, s_i, c_null).squeeze(2).float()
            v = v_u + cfg * (v - v_u)
        x = x - (s_i - s_next) * v
    return x


def feature(model, x0, eps_f, tau, c, block) -> np.ndarray:
    x_tau = (1.0 - tau) * x0 + tau * eps_f
    feats = velocity(
        model,
        x_tau,
        tau,
        c,
        return_block_features={block},
        return_features_early=True,
    )[block].float()
    return feats.mean(dim=tuple(range(1, feats.ndim - 1)))[0].cpu().numpy()


def pixel_stats(img: torch.Tensor) -> list[float]:
    """Stats of a decoded (3, H, W) image in [0, 1]."""
    mx, mn = img.amax(0), img.amin(0)
    sat = ((mx - mn) / mx.clamp_min(1e-3)).mean()
    luma = (img * torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0)
    q = torch.quantile(luma.flatten()[::16], torch.tensor([0.01, 0.99]))
    return [sat.item(), luma.mean().item(), luma.std().item(), (q[1] - q[0]).item()]


def w2(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mean = float(((a.mean(0) - b.mean(0)) ** 2).sum())
    std = float(((a.std(0) - b.std(0)) ** 2).sum())
    return mean, std


def gap(a, b, scale, rng, n_resample) -> dict:
    N = len(a)
    mean, std = w2(a, b)
    boot, null = [], []
    for _ in range(n_resample):
        idx = rng.integers(0, N, N)
        boot.append(sum(w2(a[idx], b[idx])) / scale)
        swap = rng.random(N) < 0.5
        aa = np.where(swap[:, None], b, a)
        bb = np.where(swap[:, None], a, b)
        null.append(sum(w2(aa, bb)) / scale)
    return {
        "nw2": (mean + std) / scale,
        "nw2_mean_part": mean / scale,
        "nw2_std_part": std / scale,
        "boot_p05": float(np.percentile(boot, 5)),
        "boot_p95": float(np.percentile(boot, 95)),
        "null_p95": float(np.percentile(null, 95)),
    }


def pull_cos(s, d, t4, rng, n_resample) -> dict:
    """cos(toward D, toward T4) from S, on the mean and on the std."""

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    def both(idx):
        ss, dd, tt = s[idx], d[idx], t4[idx]
        return (
            cos(dd.mean(0) - ss.mean(0), tt.mean(0) - ss.mean(0)),
            cos(dd.std(0) - ss.std(0), tt.std(0) - ss.std(0)),
        )

    N = len(s)
    point = both(np.arange(N))
    boot = np.array([both(rng.integers(0, N, N)) for _ in range(n_resample)])
    return {
        "mean": point[0],
        "mean_p05_p95": np.percentile(boot[:, 0], [5, 95]).tolist(),
        "std": point[1],
        "std_p05_p95": np.percentile(boot[:, 1], [5, 95]).tolist(),
    }


def contact_sheet(
    rows: list[list[torch.Tensor]], path: Path, height: int = 384
) -> None:
    tiles = []
    for row in rows:
        ims = []
        for img in row:
            arr = (img.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
            pil = Image.fromarray(arr)
            ims.append(pil.resize((pil.width * height // pil.height, height)))
        tiles.append(ims)
    width = max(sum(i.width for i in row) for row in tiles)
    sheet = Image.new("RGB", (width, height * len(tiles)))
    for y, row in enumerate(tiles):
        x = 0
        for im in row:
            sheet.paste(im, (x, y * height))
            x += im.width
    sheet.save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="post_image_dataset/lora")
    p.add_argument("--resized_dir", default="post_image_dataset/resized")
    p.add_argument("--student_lora", default=STUDENT_LORA)
    p.add_argument("--int8_file", default=None)
    p.add_argument("--teacher_steps", type=int, default=28)
    p.add_argument("--teacher_cfg", type=float, default=4.0)
    p.add_argument("--student_steps", type=int, default=4)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--taus", type=float, nargs="+", default=[0.8, 0.5])
    p.add_argument("--n_resample", type=int, default=500)
    p.add_argument("--label", default="moment-gap-int8")
    opts = p.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    run_dir = make_run_dir("turbo", label=opts.label)

    data_dir = Path(resolve_under_home(opts.data_dir)).resolve()
    resized_dir = Path(resolve_under_home(opts.resized_dir)).resolve()
    items = pick_items(data_dir, resized_dir, opts.n, opts.seed)
    N = len(items)
    n_stripped = sum(it["noartist"] != it["caption"] for it in items)
    print(f"[gap] {N} items, {n_stripped} with an artist tag removed", flush=True)

    teacher_sigmas = get_timesteps_sigmas(opts.teacher_steps, opts.flow_shift, "cpu")[1]
    teacher_sigmas = teacher_sigmas.tolist()
    student_sigmas = get_timesteps_sigmas(opts.student_steps, opts.flow_shift, "cpu")[1]
    student_sigmas = student_sigmas.tolist()
    c_null = load_uncond_crossattn(str(default_uncond_path()), device, dtype)
    if c_null.ndim == 2:
        c_null = c_null.unsqueeze(0)

    def conditioned(i: int):
        it = items[i]
        latent = load_cached_latents(it["npz"])[0].to(device).float().unsqueeze(0)
        c = load_cached_text_features(it["te"], variant=0)[0].to(device, dtype)
        gen = torch.Generator("cpu").manual_seed(opts.seed + i)
        eps = torch.randn(latent.shape, generator=gen).to(device)
        gen = torch.Generator("cpu").manual_seed(opts.seed + 100_000 + i)
        eps_f = torch.randn(latent.shape, generator=gen).to(device)
        return latent, c.unsqueeze(0), eps, eps_f

    x0s: dict[str, list[torch.Tensor]] = {k: [] for k in SETS}

    print("[gap] student rollouts", flush=True)
    model = load_dit_model(
        make_args(opts, device, opts.student_lora), device, torch.bfloat16
    )
    for i in range(N):
        _, c, eps, _ = conditioned(i)
        x0 = rollout(model, eps, c, c_null, student_sigmas, 1.0)
        x0s["S"].append(x0.cpu())
    del model
    clean_memory_on_device(device)

    print("[gap] teacher: encode captions without the artist", flush=True)
    args = make_args(opts, device, None)
    model = load_dit_model(args, device, torch.bfloat16)
    shared = load_shared_models(args)
    block = len(model.blocks) - 1
    te_cos, c_noartist = [], []
    for i, it in enumerate(items):
        full, stripped = prepare_text_inputs(
            args,
            device=device,
            anima=model,
            shared_models=shared,
            prompt=it["caption"],
            negative_prompt=it["noartist"],
            text_encoder_path=args.text_encoder,
        )
        c = conditioned(i)[1]
        te_cos.append(
            torch.nn.functional.cosine_similarity(
                full["embed"][0].to(device).float().flatten(),
                c.float().flatten(),
                dim=0,
            ).item()
        )
        c_noartist.append(stripped["embed"][0].to(device, dtype).reshape_as(c).cpu())
    print(f"[gap] fresh vs cached TE cos, min {min(te_cos):.5f}", flush=True)
    del shared
    clean_memory_on_device(device)

    print(f"[gap] teacher rollouts, tap block {block}", flush=True)
    feats = {tau: {k: [] for k in SETS} for tau in opts.taus}
    t0 = time.time()
    for i in range(N):
        latent, c, eps, eps_f = conditioned(i)
        x0s["D"].append(latent.cpu())
        for name, cond, cfg in (
            ("T1", c, 1.0),
            ("T4", c, opts.teacher_cfg),
            ("T1na", c_noartist[i].to(device), 1.0),
        ):
            x0 = rollout(model, eps, cond, c_null, teacher_sigmas, cfg)
            x0s[name].append(x0.cpu())
        for k in SETS:
            x0 = x0s[k][i].to(device)
            for tau in opts.taus:
                feats[tau][k].append(feature(model, x0, eps_f, tau, c, block))
        elapsed = time.time() - t0
        print(
            f"[gap] item {i + 1}/{N} {elapsed / (i + 1):.1f} s/item "
            f"ETA {elapsed / (i + 1) * (N - i - 1) / 60:.0f} min",
            flush=True,
        )
    feat_dim = int(next(iter(feats.values()))["D"][0].shape[0])
    del model
    clean_memory_on_device(device)

    print("[gap] decode", flush=True)
    vae = load_vae(args.vae, device=device, dtype=torch.bfloat16, eval=True)
    pixels = {k: [] for k in SETS}
    sheet = []
    for i in range(N):
        row = []
        for k in SETS:
            img = vae.decode_to_pixels(x0s[k][i].to(device, dtype).unsqueeze(2))
            img = ((img.squeeze(0).squeeze(1).float() + 1.0) / 2.0).clamp(0, 1).cpu()
            pixels[k].append(pixel_stats(img))
            row.append(img)
        if i < 6:
            sheet.append(row)
    contact_sheet(sheet, run_dir / "contact.png")
    del vae
    clean_memory_on_device(device)

    rng = np.random.default_rng(opts.seed)
    spaces = {f"tau{tau}": feats[tau] for tau in opts.taus}
    spaces["latent_channel_mean"] = {
        k: [x.float().mean(dim=(0, 2, 3)).numpy() for x in x0s[k]] for k in SETS
    }
    metrics: dict = {"N": N, "n_artist_stripped": n_stripped, "feat_dim": feat_dim}
    metrics["te_cos_min"] = min(te_cos)
    arrays = {}
    for space, sets in spaces.items():
        arr = {k: np.stack(v).astype(np.float64) for k, v in sets.items()}
        arrays.update({f"{space}_{k}": v for k, v in arr.items()})
        scale = float(arr["D"].var(0).sum())
        metrics[space] = {
            "scale_D_var": scale,
            "gaps": {
                f"{a}-{b}": gap(arr[a], arr[b], scale, rng, opts.n_resample)
                for a, b in PAIRS
            },
            "pull_cos_S_to_D_vs_T4": pull_cos(
                arr["S"], arr["D"], arr["T4"], rng, opts.n_resample
            ),
        }
    px = {k: np.array(v) for k, v in pixels.items()}
    metrics["pixels"] = {
        k: {
            stat: {
                "mean": float(px[k][:, j].mean()),
                "paired_vs_T4_mean": float((px[k][:, j] - px["T4"][:, j]).mean()),
                "paired_vs_T4_se": float(
                    (px[k][:, j] - px["T4"][:, j]).std() / np.sqrt(N)
                ),
            }
            for j, stat in enumerate(PIXEL_STATS)
        }
        for k in SETS
    }
    np.savez_compressed(run_dir / "features.npz", **arrays)
    write_result(
        run_dir,
        script=__file__,
        args=opts,
        metrics=metrics,
        artifacts=["contact.png", "features.npz"],
    )
    print(f"[gap] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
