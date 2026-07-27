#!/usr/bin/env python
"""sigma_lowres — per-band VAE re-encode noise floor (the parameter-free δ).

The spectral null (SPD Prop 1) needs an error tolerance δ on per-band
recovered signal power. This script measures the one tolerance our
substitution pipeline fixes by construction: the per-frequency power of the
re-encode arm's latent error, D(f) = RAPSD(reenc(x) − cached(x)), on the
same probe set and with the same resize→encode chain as the gradient
probe's reenc control (``encode_probe_latents``). Band content below D(f)
is lost to the pipeline's own noise even with no resolution change, so
δ_reenc(e) := D(f_cut(e)) anchors the null with zero free parameters; the
implied boundary is t_ω(P(f_cut), δ_reenc) from Eq. spd.

Usage::

    uv run python project/sigma_lowres/bench/reenc_noise_floor.py --label reenc-floor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from project.sigma_lowres.bench.rapsd import rapsd  # noqa: E402
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    select_probe_set,
)
from project.sigma_lowres.bench.tier_routing.run_grad_probe import (  # noqa: E402
    encode_probe_latents,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NATIVE_EDGE = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    p.add_argument("--num_images", type=int, default=40)
    p.add_argument("--tier", type=int, default=NATIVE_EDGE)
    p.add_argument("--demote_edges", default="896,768,512")
    p.add_argument("--nbins", type=int, default=64)
    p.add_argument("--artists", default=None)
    p.add_argument("--max_per_artist", type=int, default=None)
    p.add_argument("--score_limit", type=int, default=None)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.num_images = 4
        args.score_limit = args.score_limit or 120
    return args


def t_omega(p_cut: float, delta: float) -> float:
    """SPD Prop 1 activation time at tolerance delta (paper Eq. spd)."""
    return 1.0 / (1.0 + float(np.sqrt(delta / (p_cut * (1.0 + p_cut - delta)))))


def main() -> None:
    args = parse_args()
    edges = [int(e) for e in args.demote_edges.split(",") if e]
    device = torch.device("cuda")

    artists = args.artists.split(",") if args.artists else None
    records = score_corpus(artists=artists, k=args.quant_k, limit=args.score_limit)
    probe = select_probe_set(
        records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
    )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records in the scored pool")
    log.info(
        f"probe set: {len(probe)} images, {len({r.artist for r in probe})} artists"
    )

    reenc = encode_probe_latents(probe, [], args.vae, device, reenc_control=True)

    p_profiles, d_profiles = [], []
    per_image: list[dict] = []
    centers = None
    for r in probe:
        d = np.load(r.npz_path)
        lat_key = next(k for k in d.files if k.startswith("latents_"))
        cached = d[lat_key].astype(np.float32)  # (16, H, W)
        fresh = reenc[(r.stem, "reenc")].numpy().astype(np.float32)
        if fresh.shape != cached.shape:
            log.warning(f"shape mismatch {r.stem}: {fresh.shape} vs {cached.shape}")
            continue
        centers, p_prof = rapsd(cached, args.nbins)
        _, d_prof = rapsd(fresh - cached, args.nbins)
        p_profiles.append(p_prof)
        d_profiles.append(d_prof)
        per_image.append(
            {
                "artist": r.artist,
                "stem": r.stem,
                "err_var": round(float((fresh - cached).var()), 6),
                "lat_var": round(float(cached.var()), 5),
            }
        )

    p_mean = np.nanmean(np.stack(p_profiles), axis=0)
    d_mean = np.nanmean(np.stack(d_profiles), axis=0)
    d_stack = np.stack(d_profiles)

    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )

    headline: dict = {
        "n_images": len(per_image),
        "nbins": args.nbins,
        "err_var_mean": round(float(np.mean([r["err_var"] for r in per_image])), 6),
        "lat_var_mean": round(float(np.mean([r["lat_var"] for r in per_image])), 5),
    }
    for e in edges:
        f_cut = 0.5 * e / args.tier
        p_cut = float(np.interp(f_cut, centers, p_mean))
        d_cut = float(np.interp(f_cut, centers, d_mean))
        d_imgs = np.array([np.interp(f_cut, centers, prof) for prof in d_stack])
        headline[f"route_{e}"] = {
            "f_cut": round(f_cut, 4),
            "P_cut": round(p_cut, 5),
            "delta_reenc": round(d_cut, 5),
            "delta_reenc_p10": round(float(np.percentile(d_imgs, 10)), 5),
            "delta_reenc_p90": round(float(np.percentile(d_imgs, 90)), 5),
            "t_star_reenc": round(t_omega(p_cut, d_cut), 4),
            "snr_cut_P_over_D": round(p_cut / d_cut, 3) if d_cut > 0 else None,
        }

    curves = {
        "freq": [round(float(f), 5) for f in centers],
        "P_mean": [round(float(v), 6) for v in p_mean],
        "D_mean": [round(float(v), 6) for v in d_mean],
    }
    (run_dir / "curves.json").write_text(json.dumps(curves, indent=1))
    rows_path = run_dir / "per_image.jsonl"
    with rows_path.open("w") as f:
        for row in per_image:
            f.write(json.dumps(row) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(centers, p_mean, color="purple", lw=2, label="clean latents P(f)")
    ax.loglog(centers, d_mean, color="crimson", lw=2, label="re-encode error D(f)")
    for e in edges:
        ax.axvline(0.5 * e / args.tier, ls="--", lw=1, label=f"{e} Nyquist")
    ax.set_xlabel("spatial frequency (cycles / latent px)")
    ax.set_ylabel("power")
    ax.set_title("Pipeline noise floor vs latent signal")
    ax.legend(fontsize=7)
    fig.tight_layout()
    plot_path = run_dir / "reenc_floor.png"
    fig.savefig(plot_path, dpi=140)

    log.info(json.dumps(headline, indent=2, default=str))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=[rows_path, run_dir / "curves.json", plot_path],
        extra={"probe_set": [f"{r.artist}/{r.stem}" for r in probe]},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
