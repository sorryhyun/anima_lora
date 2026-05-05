#!/usr/bin/env python
"""Per-step gap_LL plot with intra-prompt (seed) and inter-prompt bands separated.

Background
----------
``make dcw``'s gap_curves.png shows ``±1σ across (img×seed)`` as one
combined band. For DCW v4 calibration work it's useful to see seeds vs
prompts separately: σ_seed[i] is the within-(prompt,aspect) seed-noise
band — the part the head can't predict from prompt features alone —
while σ_prompt[i] is the between-prompt signal the head is supposed to
fit. The 2026-05-05 variance decomposition showed seed share ≈ 13.4%
of total target variance at production scale (vs ≈ 47% suggested by
the smaller bench-1543 pool); this plot makes the per-step shape of
that ratio visible.

Layout (matches gap_curves.png style — 1×3 row at 18×4.5):

1. Pooled gap_LL[i] mean across all rows, with σ_seed band (tight,
   dark) + σ_prompt band (wide, light).
2. Per-aspect gap_LL[i] mean curves (5 lines).
3. Variance decomposition fractions per step
   (σ_seed²/σ_total², σ_prompt²/σ_total²).

Usage
-----
::

    uv run python bench/dcw/plot_seed_band.py
    uv run python bench/dcw/plot_seed_band.py --runs_root output/dcw \\
        --pattern 'make-dcw-*'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_aspect_runs(runs_root: Path, pattern: str) -> list[dict]:
    """Load gaps_per_sample.npz + n_seeds from each matching run dir."""
    out: list[dict] = []
    for run_dir in sorted(runs_root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        npz_path = run_dir / "gaps_per_sample.npz"
        rj_path = run_dir / "result.json"
        if not (npz_path.exists() and rj_path.exists()):
            continue
        rj = json.loads(rj_path.read_text())
        a = rj.get("args", {})
        H, W = a.get("image_h"), a.get("image_w")
        n_seeds = int(a.get("n_seeds", 1))
        if n_seeds < 2:
            print(f"  skip {run_dir.name}: n_seeds={n_seeds} (need ≥2 for seed band)")
            continue
        z = np.load(npz_path, allow_pickle=True)
        if "gap_LL" not in z.files:
            continue
        gap_LL = np.asarray(z["gap_LL"], dtype=np.float64)  # (N, n_steps)
        stems = np.asarray(z["stems"]) if "stems" in z.files else None
        if stems is None:
            continue
        sigmas = np.asarray(z["sigmas"]) if "sigmas" in z.files else None
        N, n_steps = gap_LL.shape
        n_imgs = N // n_seeds
        # Reshape to (n_imgs, n_seeds, n_steps); rows are interleaved
        # (img0_seed0, img0_seed1, ..., img1_seed0, ...).
        if N != n_imgs * n_seeds:
            print(f"  skip {run_dir.name}: N={N} not divisible by n_seeds={n_seeds}")
            continue
        gap_grouped = gap_LL.reshape(n_imgs, n_seeds, n_steps)
        out.append(
            {
                "run": run_dir.name,
                "aspect": f"{W}x{H}" if (H and W) else run_dir.name,
                "n_imgs": n_imgs,
                "n_seeds": n_seeds,
                "gap_LL": gap_LL,  # (N, n_steps)
                "gap_grouped": gap_grouped,  # (n_imgs, n_seeds, n_steps)
                "stems": stems[:N],
                "sigmas": sigmas,
            }
        )
        print(
            f"  loaded {run_dir.name}: aspect={W}x{H} n_imgs={n_imgs} "
            f"n_seeds={n_seeds} n_steps={n_steps}"
        )
    return out


def variance_decomposition(
    gap_grouped_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-step σ_seed[i], σ_prompt[i], σ_total[i], gap_mean[i].

    Input: ``(n_groups, n_seeds, n_steps)`` where each group is one
    (prompt, aspect) pair pooled across all aspects (groups are unique
    by stem in this dataset since each stem appears at one aspect only).
    """
    n_groups, n_seeds, n_steps = gap_grouped_pool.shape
    # Per-step pooled total variance (over all rows = all groups × seeds)
    flat = gap_grouped_pool.reshape(-1, n_steps)
    sigma_total = flat.std(axis=0, ddof=0)  # (n_steps,)
    gap_mean = flat.mean(axis=0)
    # Per-step within-group variance averaged across groups
    # (= the "expected within-group variance" component of the total)
    sigma_seed_per_group = gap_grouped_pool.std(axis=1, ddof=0)  # (n_groups, n_steps)
    sigma_seed = np.sqrt((sigma_seed_per_group ** 2).mean(axis=0))  # (n_steps,)
    # Per-step between-group variance (variance of group means)
    group_means = gap_grouped_pool.mean(axis=1)  # (n_groups, n_steps)
    sigma_prompt = group_means.std(axis=0, ddof=0)  # (n_steps,)
    return sigma_seed, sigma_prompt, sigma_total, gap_mean


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--runs_root",
        type=Path,
        default=REPO_ROOT / "output" / "dcw",
        help="Directory holding make-dcw run dirs. Default = output/dcw/.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*-make-dcw-*",
        help="Glob pattern for run subdirs. Default matches make-dcw outputs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default = <runs_root>/seed_band_curves.png.",
    )
    p.add_argument(
        "--band",
        type=str,
        default="LL",
        choices=("LL", "LH", "HL", "HH"),
        help="Haar subband to plot (only LL is currently exported per-row "
        "by gaps_per_sample.npz).",
    )
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — run `uv add matplotlib` first")

    print(f"[1/3] loading runs from {args.runs_root}/{args.pattern} ...")
    runs = load_aspect_runs(args.runs_root, args.pattern)
    if not runs:
        sys.exit("no matching runs found")
    n_steps = runs[0]["gap_LL"].shape[1]
    print(f"  total: {len(runs)} aspects, n_steps={n_steps}")

    # Pool across all aspects for the population-level decomposition.
    # Each "group" is one (stem, aspect); since this dataset has each stem
    # at one aspect only, every (stem, aspect) is unique across the pool.
    print("[2/3] computing variance decomposition ...")
    pooled = np.concatenate([r["gap_grouped"] for r in runs], axis=0)
    print(f"  pooled groups: {pooled.shape[0]} × {pooled.shape[1]} seeds")
    sigma_seed, sigma_prompt, sigma_total, gap_mean = variance_decomposition(pooled)
    seed_share_int = float((sigma_seed ** 2).sum() / (sigma_total ** 2).sum())
    prompt_share_int = float((sigma_prompt ** 2).sum() / (sigma_total ** 2).sum())
    print(
        f"  integrated shares — seed={seed_share_int:.1%} "
        f"prompt={prompt_share_int:.1%}"
    )

    print(f"[3/3] plotting → {args.out or args.runs_root / 'seed_band_curves.png'}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True)
    xs = np.arange(n_steps)

    # Panel 1: pooled gap with σ_seed (tight) and σ_prompt (wide) bands.
    ax = axes[0]
    ax.fill_between(
        xs,
        gap_mean - sigma_prompt,
        gap_mean + sigma_prompt,
        color="#888888",
        alpha=0.18,
        label="±1σ_prompt (between-prompt)",
    )
    ax.fill_between(
        xs,
        gap_mean - sigma_seed,
        gap_mean + sigma_seed,
        color="#264653",
        alpha=0.45,
        label="±1σ_seed (intra-prompt)",
    )
    ax.plot(xs, gap_mean, color="#e76f51", lw=1.8, label="mean gap_LL[i]")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Pooled gap (band={args.band}) with seed vs prompt bands")
    ax.set_xlabel("step i")
    ax.set_ylabel("gap")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: per-aspect mean curves (no bands — comparison view).
    ax = axes[1]
    aspect_palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    for i, r in enumerate(runs):
        ax.plot(
            xs,
            r["gap_LL"].mean(axis=0),
            color=aspect_palette[i % len(aspect_palette)],
            lw=1.4,
            label=f"{r['aspect']} (n={r['n_imgs']}×{r['n_seeds']})",
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Per-aspect mean gap_LL[i]")
    ax.set_xlabel("step i")
    ax.set_ylabel("gap")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: per-step variance share (σ_seed² / σ_total², etc.).
    ax = axes[2]
    seed_frac = (sigma_seed ** 2) / np.maximum(sigma_total ** 2, 1e-12)
    prompt_frac = (sigma_prompt ** 2) / np.maximum(sigma_total ** 2, 1e-12)
    ax.fill_between(xs, 0, prompt_frac, color="#888888", alpha=0.5, label="σ²_prompt / σ²_total")
    ax.fill_between(
        xs,
        prompt_frac,
        prompt_frac + seed_frac,
        color="#264653",
        alpha=0.55,
        label="σ²_seed / σ²_total",
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Per-step variance share  "
        f"(integrated: seed={seed_share_int:.0%}, prompt={prompt_share_int:.0%})"
    )
    ax.set_xlabel("step i")
    ax.set_ylabel("share")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = args.out or args.runs_root / "seed_band_curves.png"
    fig.savefig(out_path, dpi=130)
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
