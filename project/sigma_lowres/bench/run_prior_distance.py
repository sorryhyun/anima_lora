#!/usr/bin/env python
"""sigma_lowres — forward-only prior/residual-distance probe (grid conditioning).

Discriminator (1) from the 1280→1024 read-through (account:
``record/hypothesis.md``; records: ``record/groundings.md`` G5/G6), generalized
to a σ-resolved curve. No adapter, no gradients: per image, grid and σ, the
K-draw **mean residual** ``r̄ = E_ε[v̂(z_σ) − (ε − x)]`` is the model's mean
prediction error on that grid — the exact ``r`` factor the LoRA gradient
``g = Jᵀr`` is built from. A perfect model has r̄ = 0 on every grid, so
cross-grid distances of r̄ measure *model-error differences only*, free of the
σ-independent content/resampling baseline that pollutes raw-prediction
comparisons (that baseline is still reported per pair as ``d_content``). At
σ = 1 the input is pure ε and r̄ = x − x̄_prior — the content-corrected form of
the v1 prior-drift probe (`results/20260726-2120/`, which compared x̄_prior
directly; numbers are close but not identical).

Per adjacent pair (hi → lo), ``r̄_hi`` is area-downsampled to the lo grid and
compared with relative L2 + cosine against ``r̄_lo``, with a split-half noise
floor (parity half-means; pair floor = ``0.5·√(f_hi↓² + f_lo²)`` since a
half-mean distance overestimates full-mean noise 2×). A same-grid reenc pair
(native cache vs re-encoded PNG at the native bucket) is the encode-chain
control.

Pre-registered σ-curve reads (the hybrid account, record/hypothesis.md):

- **S1-severity**: excess(σ) declines toward its σ=1 asymptote, and the
  1280→1024 curve stays elevated to HIGHER σ than 1024→896's — the forward
  analog of the gradient crossovers (σ\* ≈ 0.75–0.875 vs ≈ 0.5), with
  896→768 worst. If so, the route-dependent σ\* is prediction-side (S1).
- **Overlap**: the two routes' curves coincide → the crossover ordering is
  NOT carried by the residual; it would have to live in σ-dependence of J,
  breaking the "Floor is σ-independent" assumption — account rework.

Usage::

    make daemon-run ARGS="project/sigma_lowres/bench/run_prior_distance.py \
        --sigmas 0.125,0.375,0.625,0.875,1.0 --draws 8 --label rescurve"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    select_probe_set,
)
from project.sigma_lowres.bench.tier_routing.run_grad_probe import (  # noqa: E402
    DIT,
    VAE,
    encode_probe_latents,
)
from library.io.cache import (  # noqa: E402
    load_cached_latents,
    load_cached_text_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dit", default=DIT)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--num_images", type=int, default=16)
    p.add_argument(
        "--sigmas",
        default="1.0",
        help="csv σ list; at each σ the K-draw mean residual is compared "
        "across grids (σ=1.0 = the pure-noise prior case)",
    )
    p.add_argument("--draws", type=int, default=16, help="draws per grid per σ")
    p.add_argument("--tier", type=int, default=1280, help="native tier of the set")
    p.add_argument(
        "--edges",
        default="1024,896,768,512",
        help="demoted grids; pairs are consecutive over [native] + edges",
    )
    p.add_argument(
        "--data_root",
        default=str(Path(__file__).resolve().parent / "probe1280_cache"),
        help="dataset root holding lora/ + resized/ (default: the probe-local "
        "1280 cache from prep_1280_probe.py)",
    )
    p.add_argument("--max_per_artist", type=int, default=None)
    p.add_argument(
        "--uncond",
        action="store_true",
        help="condition every forward on the T5('') uncond crossattn (bundled "
        "sidecar asset) instead of each image's caption embedding. With full "
        "captions the model may never fall back to a grid-conditional content "
        "prior (composition pinned by text); this arm frees it to — the "
        "designated test of the 'small canvas → portrait prior' account of "
        "cross-route Δr̄ direction structure (cross-image direction "
        "consistency should appear at high σ if that account is real).",
    )
    p.add_argument(
        "--save_residuals",
        action="store_true",
        help="dump every per-image per-grid per-σ residual parity half-mean "
        "pair (m1, m2) as fp16 npz under <run_dir>/residuals/ — the raw "
        "objects behind d/floor/excess, so cross-route Δr̄ structure "
        "(pairwise cosines / stacked SVD; the 'is m route-uniform per-band "
        "or only in norm' read) is a CPU reanalysis instead of a rerun.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    return p.parse_args()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (0.5 * (a.norm() + b.norm())))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.flatten().double().dot(b.flatten().double()) / (a.norm() * b.norm()))


def down_to(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """(16, H, W) → area-downsample to (16, *hw)."""
    return F.interpolate(x.unsqueeze(0), size=hw, mode="area").squeeze(0)


def residual_halves(
    anima,
    device: torch.device,
    x_lat: torch.Tensor,  # (16, H, W) fp32 CPU — the grid's clean latent
    sigma: float,
    crossattn: torch.Tensor,
    draws: int,
    seed_base: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parity half-means (m1, m2) of the residual v̂(z_σ) − (ε − x) over
    ``draws`` noise draws on this grid. fp32, on device. At σ=1 the input is
    exactly ε (x drops out of z) and the residual is x − x̄_prior-shaped."""
    h, w = x_lat.shape[-2:]
    x = x_lat.unsqueeze(0).to(device)  # (1, 16, H, W) fp32
    pad = torch.zeros(1, 1, h, w, dtype=torch.bfloat16, device=device)
    sig = torch.full((1,), sigma, device=device)
    halves = [
        torch.zeros(16, h, w, device=device),
        torch.zeros(16, h, w, device=device),
    ]
    for d in range(draws):
        gen = torch.Generator(device=device).manual_seed(seed_base + d)
        eps = torch.randn(x.shape, generator=gen, device=device)
        z = (1.0 - sigma) * x + sigma * eps
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v = anima(
                z.unsqueeze(2).to(torch.bfloat16), sig, crossattn, padding_mask=pad
            )
        halves[d % 2] += (v.squeeze(2).float() - (eps - x))[0]
    n_half = [draws - draws // 2, draws // 2]
    return halves[0] / n_half[0], halves[1] / n_half[1]


def main() -> None:
    args = parse_args()
    start_heartbeat()
    edges = [int(e) for e in args.edges.split(",") if e]
    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    device = torch.device("cuda")

    records = score_corpus(k=args.quant_k, data_root=Path(args.data_root).resolve())
    probe = select_probe_set(
        records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
    )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records under {args.data_root}")
    log.info(
        f"probe set: {len(probe)} images ({len({r.artist for r in probe})} artists), "
        f"grids [{args.tier}]+reenc+{edges}, σ={sigmas}, {args.draws} draws each"
    )

    # VAE loads, encodes the demoted + reenc arms, and frees BEFORE the DiT
    # loads (lazy-loading invariant) — same chain as the gradient probes.
    extra_latents = encode_probe_latents(probe, edges, args.vae, device, True)

    from library.runtime.harness import build_anima

    args.compile = False
    args.gradient_checkpointing = False
    bundle = build_anima(args, dit_path=args.dit, adapter=None, train_mode=False)
    anima = bundle.anima

    uncond = None
    if args.uncond:
        from library.preprocess.uncond import ensure_uncond_crossattn

        uncond = ensure_uncond_crossattn(
            qwen3_path="",  # bundled sidecar asset — staging fallback unused
            dit_path=args.dit,
            device=device,
            dtype=torch.bfloat16,
        )

    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )
    rows_path = run_dir / "per_image.jsonl"
    rows: list[dict] = []
    native = str(args.tier)
    grid_names: list[str] = [native] + [str(e) for e in edges]
    pairs = list(zip(grid_names[:-1], grid_names[1:]))
    t0 = time.time()

    for i, r in enumerate(probe):
        if uncond is not None:
            crossattn = uncond
        else:
            crossattn, _ = load_cached_text_features(r.te_path, variant=0)
            if crossattn is None:
                log.info(f"  [{i}] {r.artist}/{r.stem}: no crossattn_emb — skipped")
                continue
            crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        lats: dict[str, torch.Tensor] = {native: load_cached_latents(r.npz_path)[0]}
        lats["reenc"] = extra_latents[(r.stem, "reenc")]
        for e in edges:
            lats[str(e)] = extra_latents[(r.stem, f"demote{e}")]

        halves: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for gi, g in enumerate([native, "reenc"] + [str(e) for e in edges]):
            for si, sig in enumerate(sigmas):
                halves[(g, si)] = residual_halves(
                    anima,
                    device,
                    lats[g],
                    sig,
                    crossattn,
                    args.draws,
                    args.seed * 1_000_000 + i * 10_000 + gi * 1_000 + si * 100,
                )
        if args.save_residuals:
            res_dir = run_dir / "residuals"
            res_dir.mkdir(exist_ok=True)
            for (g, si), (m1, m2) in halves.items():
                np.savez_compressed(
                    res_dir / f"{r.stem}__{g}__s{si}.npz",
                    m1=m1.half().cpu().numpy(),
                    m2=m2.half().cpu().numpy(),
                    sigma=np.float32(sigmas[si]),
                )

        row: dict = {
            "artist": r.artist,
            "stem": r.stem,
            "redundancy": round(r.redundancy, 4),
            "sigmas": sigmas,
            "tokens": {
                g: (v.shape[-2] // 2) * (v.shape[-1] // 2) for g, v in lats.items()
            },
            "res_norm": {
                g: [
                    round(
                        float((0.5 * (halves[(g, si)][0] + halves[(g, si)][1])).norm()),
                        3,
                    )
                    for si in range(len(sigmas))
                ]
                for g in grid_names
            },
        }
        # reenc control: same grid, cache latent vs re-encoded latent
        cmp_pairs = pairs + [(native, "reenc")]
        for hi, lo in cmp_pairs:
            x_lo = lats[lo].to(device)
            lo_hw = x_lo.shape[-2:]
            x_hi = lats[hi].to(device)
            x_hi = x_hi if x_hi.shape[-2:] == lo_hw else down_to(x_hi, lo_hw)
            row[f"d_content_{hi}_{lo}"] = round(rel_l2(x_hi, x_lo), 5)
            d_s, f_s, e_s, c_s = [], [], [], []
            for si in range(len(sigmas)):
                m1h, m2h = halves[(hi, si)]
                if m1h.shape[-2:] != lo_hw:
                    m1h, m2h = down_to(m1h, lo_hw), down_to(m2h, lo_hw)
                m1l, m2l = halves[(lo, si)]
                a, b = 0.5 * (m1h + m2h), 0.5 * (m1l + m2l)
                f_hi, f_lo = rel_l2(m1h, m2h), rel_l2(m1l, m2l)
                d = rel_l2(a, b)
                floor = 0.5 * (f_hi**2 + f_lo**2) ** 0.5
                d_s.append(round(d, 5))
                f_s.append(round(floor, 5))
                e_s.append(round(d - floor, 5))
                c_s.append(round(cos(a, b), 5))
            row[f"d_{hi}_{lo}"] = d_s
            row[f"floor_{hi}_{lo}"] = f_s
            row[f"excess_{hi}_{lo}"] = e_s
            row[f"cos_{hi}_{lo}"] = c_s
        del halves
        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        log.info(
            f"  [{i + 1}/{len(probe)}] {r.artist}/{r.stem} "
            + " ".join(
                f"x{hi}->{lo}={row[f'excess_{hi}_{lo}']}" for hi, lo in cmp_pairs
            )
        )

    if not rows:
        raise SystemExit("no rows produced")

    headline: dict = {
        "n_images": len(rows),
        "draws": args.draws,
        "sigmas": sigmas,
        "grids": grid_names,
        "dit": args.dit,
        "wall_time_s": round(time.time() - t0, 1),
    }
    for hi, lo in pairs + [(native, "reenc")]:
        headline[f"d_content_{hi}_{lo}"] = round(
            float(np.mean([r[f"d_content_{hi}_{lo}"] for r in rows])), 5
        )
        for k in ("d", "floor", "excess", "cos"):
            v = np.array([r[f"{k}_{hi}_{lo}"] for r in rows])  # (n, n_sigmas)
            headline[f"{k}_{hi}_{lo}"] = {
                "mean": [round(float(x), 5) for x in v.mean(axis=0)],
                "sem": [
                    round(float(x), 5)
                    for x in v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])
                ],
            }
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
