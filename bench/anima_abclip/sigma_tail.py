#!/usr/bin/env python3
"""Phase 2 read of ``--grad_sigma_log``: is the clipped tail a σ band?

Reads ``output/ckpt/<name>.grad_sigma.jsonl`` (one row per micro-batch under
``clip_per_micro``) and reports, per artist and pooled:

- per σ band: draws, median pre-clip norm, clip fraction under the run's own
  cap, and the band's share of the clipped tail against its share of draws;
- Spearman ρ of norm against σ;
- an offline replay of the median clip (k, window as the run) once with one
  window across all σ and once with a window per σ band, on the recorded
  pre-clip norms: how many units each clips, and how many of the global clip's
  hits the per-band one keeps;
- the image keys clipped most often.

Decision (docs/anima/findings/lora_gradient_noise_recipe.md): tail
concentrated in a band → per-timestep clip; tail spread over σ → training shift.

    .venv/bin/python bench/anima_abclip/sigma_tail.py sigmalog_cast_s42 sigmalog_ama_s42 ...
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import median

import torch

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home

EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]


def band(s: float) -> int:
    for i in range(len(EDGES) - 1):
        if s < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def label(i: int) -> str:
    return f"{EDGES[i]:.1f}-{min(EDGES[i + 1], 1.0):.1f}"


def load(name: str, skip: int) -> list[dict]:
    """Rows after the clip's warmup (``skip`` units at the fixed warmup cap)."""
    path = resolve_under_home("output/ckpt") / f"{name}.grad_sigma.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return [r for r in rows[skip:] if len(r["sigma"]) == 1]


def spearman(x: list[float], y: list[float]) -> float:
    rx = torch.tensor(x).argsort().argsort().float()
    ry = torch.tensor(y).argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def replay(rows: list[dict], k: float, window: int, per_band: bool) -> list[bool]:
    """Median clip on the recorded norms; units with fewer than 10 earlier norms
    in their window count as unclipped."""
    hist: dict[int, list[float]] = {}
    out = []
    for r in rows:
        key = band(r["sigma"][0]) if per_band else 0
        h = hist.setdefault(key, [])
        out.append(len(h) >= 10 and r["norm"] > k * median(h[-window:]))
        h.append(r["norm"])
    return out


def summarize(rows: list[dict], k: float, window: int) -> dict:
    n = len(rows)
    clipped = [r["cap"] is not None and r["norm"] > r["cap"] for r in rows]
    n_clip = sum(clipped)
    bands = []
    for i in range(len(EDGES) - 1):
        idx = [j for j, r in enumerate(rows) if band(r["sigma"][0]) == i]
        if not idx:
            continue
        c = sum(clipped[j] for j in idx)
        bands.append(
            {
                "band": label(i),
                "draws": len(idx),
                "draw_share": len(idx) / n,
                "median_norm": median(rows[j]["norm"] for j in idx),
                "clip_frac": c / len(idx),
                "tail_share": c / n_clip if n_clip else 0.0,
            }
        )
    g = replay(rows, k, window, per_band=False)
    b = replay(rows, k, window, per_band=True)
    keys = Counter(
        key for r, c in zip(rows, clipped) if c for key in r["keys"]
    ).most_common(8)
    return {
        "units": n,
        "clip_frac": n_clip / n,
        "spearman_norm_sigma": spearman(
            [r["sigma"][0] for r in rows], [r["norm"] for r in rows]
        ),
        "bands": bands,
        "replay_global_clips": sum(g),
        "replay_band_clips": sum(b),
        "replay_global_kept_by_band": sum(x and y for x, y in zip(g, b)),
        "top_clipped_keys": keys,
    }


def show(name: str, s: dict) -> None:
    print(
        f"\n== {name}: {s['units']} units, clipped {s['clip_frac']:.1%}, "
        f"Spearman(norm, σ) {s['spearman_norm_sigma']:+.2f}"
    )
    print(
        f"{'σ band':>9} {'draws':>6} {'share':>6} {'med‖g‖':>7} {'clip%':>6} {'tail%':>6}"
    )
    for b in s["bands"]:
        print(
            f"{b['band']:>9} {b['draws']:>6} {b['draw_share']:>6.1%} "
            f"{b['median_norm']:>7.2f} {b['clip_frac']:>6.1%} {b['tail_share']:>6.1%}"
        )
    print(
        f"replay k/window as run: global median clips {s['replay_global_clips']}, "
        f"per-band clips {s['replay_band_clips']}, "
        f"{s['replay_global_kept_by_band']} of the global hits also per-band"
    )
    print(
        "most clipped keys:", ", ".join(f"{k} ({c})" for k, c in s["top_clipped_keys"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="output_name of each logged run")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument(
        "--skip", type=int, default=10, help="warmup units clipped at max_grad_norm"
    )
    a = ap.parse_args()
    per = {name: load(name, a.skip) for name in a.names}
    results = {name: summarize(rows, a.k, a.window) for name, rows in per.items()}
    for name, s in results.items():
        show(name, s)
    if len(per) > 1:
        # Pooled bands only; the replay pools runs whose norm scales differ, so
        # it stays per run.
        pooled = summarize([r for rows in per.values() for r in rows], a.k, a.window)
        show("pooled (bands only; replay is per run)", pooled)
        results["pooled"] = pooled
    run_dir = make_run_dir("anima_abclip", label="sigma_tail")
    write_result(run_dir, script=__file__, args=a, metrics=results, label="sigma_tail")
    print(f"\n→ {run_dir}")


if __name__ == "__main__":
    main()
