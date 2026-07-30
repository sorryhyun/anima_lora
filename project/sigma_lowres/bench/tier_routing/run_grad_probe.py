#!/usr/bin/env python
"""tier_routing Phase 3a — the gradient-equivalence probe.

Validates the claim behind entropy-aware tier routing
(``_archive/proposals/traj_latent_stats.md`` §Phase 3a): that for high-redundancy
(flat, cel-style) images, training on a tier-down cache produces the same LoRA
gradient the native-tier cache does.

Per probe image, expected-gradient estimates at a fixed trained-LoRA operating
point, each accumulated over K draws on a shared stratified σ grid:

* ``native_a`` / ``native_b`` — native-tier cached latent, two disjoint noise
  seed sets. Their cosine is the **within-tier redraw floor** — the null. The
  verdict is never cos == 1.
* ``reenc`` — the native PNG re-encoded through the probe's own
  resize→VAE chain at the native bucket. Confound control: if
  cos(native, reenc) sits below the floor, the encode chain (not resolution)
  is contaminating the demoted arms and the probe is invalid.
* ``demote<edge>`` — the native PNG down-resized to the free-fit bucket of a
  lower tier and VAE-encoded. Per-image verdict:
  ``gap = floor − cos(native, demote)``.

Headline: Spearman(per-image static redundancy → gap). If redundancy predicts
gradient-content loss, the 3b routing threshold reads off that curve; if the
per-image spread is flat (the archived autoscale corpus-mean ≈ 0.75–0.78
regardless of content), the routing signal does not exist and the item dies
here.

Usage::

    uv run python project/sigma_lowres/bench/tier_routing/run_grad_probe.py \
        --adapter output/ckpt/anima_soup_sincos.safetensors --label phase3a
    uv run python project/sigma_lowres/bench/tier_routing/run_grad_probe.py \
        --adapter <ckpt> --smoke --label smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    ImageRecord,
    demoted_bucket,
    score_corpus,
    select_probe_set,
)
from library.io.cache import (  # noqa: E402
    load_cached_latents,
    load_cached_text_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
VAE = "models/vae/qwen_image_vae.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--adapter", required=True, help="trained plain-LoRA checkpoint")
    p.add_argument("--dit", default=DIT)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--num_images", type=int, default=40)
    p.add_argument("--draws", type=int, default=16, help="K draws per estimate")
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--demote_edges", default="896,768")
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--artists", default=None, help="csv restriction on the corpus")
    p.add_argument("--max_per_artist", type=int, default=None)
    p.add_argument("--score_limit", type=int, default=None)
    p.add_argument("--no_reenc_control", action="store_true")
    p.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="use gradient checkpointing INSTEAD of block compile (fallback)",
    )
    p.add_argument(
        "--activation_memory_budget",
        type=float,
        default=0.99,
        help="partitioner knapsack cap under compile (freefit knee; 1.0 = off)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.num_images = 4
        args.draws = 4
        args.demote_edges = "896"
        args.score_limit = args.score_limit or 120
    return args


def stratified_sigmas(k: int, shift: float) -> torch.Tensor:
    """σ grid at the K quantile midpoints of the training distribution.

    The trainer's ``timestep_sampling="shift"`` draws σ = sigmoid(randn), then
    σ ← σ·s/(1+(s−1)·σ); stratifying over quantile midpoints matches that
    density without the sampling variance.
    """
    u = (torch.arange(k, dtype=torch.float64) + 0.5) / k
    s0 = torch.sigmoid(torch.special.ndtri(u))
    sig = s0 * shift / (1.0 + (shift - 1.0) * s0)
    return sig.to(torch.float32)


def encode_probe_latents(
    records: list[ImageRecord],
    edges: list[int],
    vae_path: str,
    device: torch.device,
    reenc_control: bool,
    *,
    repromote: bool = False,
) -> dict[tuple[str, str], torch.Tensor]:
    """VAE-encode the non-cached arms; VAE is loaded, used, and freed before
    the DiT ever loads (lazy-loading invariant).

    ``repromote`` adds a ``repromote<e>`` arm per edge: the demoted pixels
    resized straight back up to the native bucket and encoded there — the same
    band destruction as ``demote<e>`` but on the NATIVE grid/graph (the
    operational data-branch intervention B of the interventional split; its
    pipeline cost is the demote arm's own down+up resize, with ``reenc`` as
    the same-grid encode-chain control)."""
    from library.datasets.image_utils import IMAGE_TRANSFORMS
    from library.models.qwen_vae import load_vae
    from library.preprocess.images import resize_to_bucket

    # vae_2d=True matches the caching default (2D-folded encode).
    vae = load_vae(
        vae_path, device=device, dtype=torch.bfloat16, eval=True, vae_2d=True
    )

    def encode(vae_, px_img) -> torch.Tensor:
        px = (
            IMAGE_TRANSFORMS(np.array(px_img))
            .unsqueeze(0)
            .to(device=device, dtype=vae_.dtype)
        )
        with torch.no_grad():
            return vae_.encode_pixels_to_latents(px)[0].float().cpu()

    out: dict[tuple[str, str], torch.Tensor] = {}
    for r in records:
        img = Image.open(r.png_path).convert("RGB")
        for e in edges:
            dem_img = resize_to_bucket(img, demoted_bucket(r, e))
            out[(r.stem, f"demote{e}")] = encode(vae, dem_img)
            if repromote:
                out[(r.stem, f"repromote{e}")] = encode(
                    vae, resize_to_bucket(dem_img, r.cached_px)
                )
        if reenc_control:
            out[(r.stem, "reenc")] = encode(vae, resize_to_bucket(img, r.cached_px))
    del vae
    torch.cuda.empty_cache()
    return out


def grad_estimate(
    bundle,
    latents: torch.Tensor,
    crossattn: torch.Tensor,
    sigmas: torch.Tensor,
    seeds: list[int],
) -> torch.Tensor:
    """Accumulated-gradient estimate over ``len(seeds)`` (σ, ε) draws.

    Returns the flattened LoRA gradient (float32, CPU) — the sum over draws,
    i.e. K·E[g]; every consumer is a cosine, so the scale is irrelevant.
    """
    device = bundle.device
    params = [
        p for _, p in sorted(bundle.network.named_parameters()) if p.requires_grad
    ]
    for p in params:
        p.grad = None
    lat = latents.unsqueeze(0).to(device)  # (1, 16, H, W) float32
    pad = torch.zeros(
        1, 1, lat.shape[-2], lat.shape[-1], dtype=torch.bfloat16, device=device
    )
    for sig, seed in zip(sigmas, seeds):
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, device=device, dtype=lat.dtype)
        sigma_b = sig.to(device).view(1)
        noisy = (1.0 - sigma_b) * lat + sigma_b * noise
        target = noise - lat
        noisy_5d = noisy.unsqueeze(2).to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = bundle.anima(noisy_5d, sigma_b, crossattn, padding_mask=pad)
        pred = pred.squeeze(2).float()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
    vec = torch.cat([p.grad.detach().float().flatten().cpu() for p in params])
    for p in params:
        p.grad = None
    return vec


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.dot(a.double(), b.double()) / (a.double().norm() * b.double().norm())
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def main() -> None:
    args = parse_args()
    edges = [int(e) for e in args.demote_edges.split(",") if e]
    reenc_control = not args.no_reenc_control
    device = torch.device("cuda")

    log.info("scoring corpus redundancy (per-image static column)…")
    artists = args.artists.split(",") if args.artists else None
    records = score_corpus(artists=artists, k=args.quant_k, limit=args.score_limit)
    probe = select_probe_set(
        records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
    )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records in the scored pool")
    log.info(
        f"probe set: {len(probe)} images, redundancy "
        f"{probe[0].redundancy:.3f}…{probe[-1].redundancy:.3f}, "
        f"{len({r.artist for r in probe})} artists"
    )

    log.info(f"encoding demoted arms ({edges}) + reenc={reenc_control}…")
    extra_latents = encode_probe_latents(probe, edges, args.vae, device, reenc_control)

    from library.runtime.harness import build_anima, compile_blocks_for_training

    args.gradient_checkpointing = args.grad_ckpt
    args.compile = False  # compile is wired below (needs dynamic-seq marks)
    bundle = build_anima(args, dit_path=args.dit, adapter=args.adapter, train_mode=True)

    if not args.grad_ckpt:
        # Block compile with dynamic-seq marks — the VRAM path (partitioner
        # recomputation instead of grad ckpt). Token budget derived from the
        # shapes the probe actually runs, native + all encoded arms.
        counts = {r.tokens for r in probe}
        counts.update(
            (t.shape[-2] // 2) * (t.shape[-1] // 2) for t in extra_latents.values()
        )
        compile_blocks_for_training(
            bundle.anima,
            bundle.network,
            backend="inductor",
            mode=None,
            n_token_families=len(counts),
            seq_range=(min(counts), max(counts)),
            dynamic_seq=True,
            activation_memory_budget=args.activation_memory_budget,
            partitioner_aggressive_recomputation=True,
            grad_ckpt=False,
        )

    sigmas = stratified_sigmas(args.draws, args.flow_shift)
    run_dir = make_run_dir(
        "tier_routing", args.label, root=Path(__file__).resolve().parent / "results"
    )
    rows_path = run_dir / "per_image.jsonl"
    rows: list[dict] = []
    t0 = time.time()

    for i, r in enumerate(probe):
        crossattn, _ = load_cached_text_features(r.te_path, variant=0)
        if crossattn is None:
            log.info(f"  [{i}] {r.artist}/{r.stem}: no crossattn_emb — skipped")
            continue
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        native = load_cached_latents(r.npz_path)[0]

        def seeds(arm_idx: int) -> list[int]:
            base = args.seed * 1_000_000 + i * 10_000 + arm_idx * 1_000
            return [base + d for d in range(args.draws)]

        g_a = grad_estimate(bundle, native, crossattn, sigmas, seeds(0))
        g_b = grad_estimate(bundle, native, crossattn, sigmas, seeds(1))
        floor = cosine(g_a, g_b)
        row = {
            "artist": r.artist,
            "stem": r.stem,
            "redundancy": round(r.redundancy, 4),
            "modal_share": round(r.modal_share, 4),
            "tokens_native": r.tokens,
            "cos_floor": round(floor, 5),
        }
        arm_idx = 2
        if reenc_control:
            re_lat = extra_latents[(r.stem, "reenc")]
            g_re = grad_estimate(bundle, re_lat, crossattn, sigmas, seeds(arm_idx))
            arm_idx += 1
            c = 0.5 * (cosine(g_a, g_re) + cosine(g_b, g_re))
            row["cos_reenc"] = round(c, 5)
            row["gap_reenc"] = round(floor - c, 5)
            # latent-space distance cache↔re-encode: isolates encode-chain
            # drift from gradient/draw noise (should be small and uniform)
            row["reenc_lat_rel_l2"] = round(
                float((re_lat - native).norm() / native.norm()), 5
            )
        for e in edges:
            lat = extra_latents[(r.stem, f"demote{e}")]
            g_d = grad_estimate(bundle, lat, crossattn, sigmas, seeds(arm_idx))
            arm_idx += 1
            c = 0.5 * (cosine(g_a, g_d) + cosine(g_b, g_d))
            row[f"tokens_{e}"] = (lat.shape[-2] // 2) * (lat.shape[-1] // 2)
            row[f"cos_{e}"] = round(c, 5)
            row[f"gap_{e}"] = round(floor - c, 5)
        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        log.info(
            f"  [{i + 1}/{len(probe)}] {r.artist}/{r.stem} red={r.redundancy:.3f} "
            + " ".join(
                f"{k}={v}" for k, v in row.items() if k.startswith(("cos_", "gap_"))
            )
        )

    if not rows:
        raise SystemExit("no per-image rows produced")

    red = np.array([r["redundancy"] for r in rows])
    headline: dict = {
        "n_images": len(rows),
        "draws": args.draws,
        "flow_shift": args.flow_shift,
        "adapter": args.adapter,
        "cos_floor_mean": round(float(np.mean([r["cos_floor"] for r in rows])), 5),
        "wall_time_s": round(time.time() - t0, 1),
    }
    if reenc_control:
        headline["gap_reenc_mean"] = round(
            float(np.mean([r["gap_reenc"] for r in rows])), 5
        )
    for e in edges:
        gaps = np.array([r[f"gap_{e}"] for r in rows])
        headline[f"cos_{e}_mean"] = round(
            float(np.mean([r[f"cos_{e}"] for r in rows])), 5
        )
        headline[f"gap_{e}_mean"] = round(float(gaps.mean()), 5)
        headline[f"gap_{e}_spread"] = round(float(gaps.max() - gaps.min()), 5)
        headline[f"spearman_redundancy_gap_{e}"] = round(spearman(red, gaps), 4)

    log.info(json.dumps(headline, indent=2))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=[rows_path],
        extra={"probe_set": [f"{r['artist']}/{r['stem']}" for r in rows]},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
