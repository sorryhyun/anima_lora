#!/usr/bin/env python
"""[:K] supervision sweep — does the target window choice matter?

Background
----------
The v4 trainer's default target is the integrated *tail* residual
``sum_{t=K..N-1} (gap_LL[p, t] - μ_g_pop[t])`` with K=7 (the same K
that defines the warmup observation ``g_obs = v_rev_LL[:K]``).

Open question (2026-05-05): would supervising on the *head* (early
steps, ``gap_LL[:K]``) — or on individual steps — produce a more
learnable signal than the tail integral? The motivation:

- ``g_obs`` (the early-step observation feature) is dead at
  n=669 / 175-stem scale: capacity ratio 2.08:1 vs ``c_pool`` but no
  CV signal — c_proj bottleneck sweeps all degraded CV. So the v4
  AR-1 hypothesis ("early observation predicts late tail") isn't
  being borne out at current data scale.
- Single-seed labels are seed-noise dominated at every timestep
  (run 1543 — within-prompt seed-std ≈ between-prompt std on LL,
  with no clean window where the ratio drops). Tail integration may
  be averaging out *signal* faster than *noise*.

This bench falsifies or confirms whether moving the target off the
tail unlocks more learnable signal. We hold the FusionHead
architecture, K-fold CV, regularizers, and ``g_obs`` observation
window fixed — only the *target window* varies. Comparing
``r_alpha_mean`` (per-prompt) and ``nll_improvement`` across windows
tells us where on the timestep axis the prompt features actually
predict the gap.

Outputs
-------
``bench/dcw/results/<TS>-k-sup-sweep[-<label>]/``::

    summary.csv     # one row per window
    per_step.csv    # only when --per_step (one row per step t)
    result.json     # standard bench envelope

Usage
-----
::

    uv run python bench/dcw/k_supervision_sweep.py
    uv run python bench/dcw/k_supervision_sweep.py --per_step --label baseline
    uv run python bench/dcw/k_supervision_sweep.py \\
        --windows ":7" "7:" "0:7" "7:14" "14:21" "21:"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from scripts.dcw.fusion_data import (  # noqa: E402
    ASPECT_NAMES,
    N_ASPECTS,
    build_population_mu_g,
    load_bench_runs,
    load_text_features,
)
from scripts.dcw.train_fusion_head import pearson, train_one_fold  # noqa: E402


def parse_window(spec: str, n_steps: int) -> tuple[int, int]:
    """Parse 'a:b' / ':b' / 'a:' into (start, end). End is exclusive."""
    if ":" not in spec:
        raise ValueError(f"window must contain ':' (got {spec!r})")
    a, b = spec.split(":", 1)
    start = int(a) if a else 0
    end = int(b) if b else n_steps
    if start < 0:
        start = max(0, n_steps + start)
    if end < 0:
        end = max(0, n_steps + end)
    end = min(end, n_steps)
    if start >= end:
        raise ValueError(f"empty window {spec!r} for n_steps={n_steps}")
    return start, end


def integrated_target(rows, mu_g_pop, start: int, end: int) -> np.ndarray:
    return np.array(
        [float((r.gap_LL[start:end] - mu_g_pop[start:end]).sum()) for r in rows],
        dtype=np.float32,
    )


def per_step_target(rows, mu_g_pop, t: int) -> np.ndarray:
    return np.array(
        [float(r.gap_LL[t] - mu_g_pop[t]) for r in rows], dtype=np.float32
    )


def cv_run(
    targets: np.ndarray,
    features: dict,
    stems_arr: np.ndarray,
    aspect_arr: np.ndarray,
    fold_of_stem: dict,
    *,
    n_folds: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    c_proj_dim: int,
    lambda_sigma_aux: float,
    device: torch.device,
) -> dict:
    """One CV pass with the given target vector. Returns aggregate metrics."""
    sigma2_pop = float(targets.var())
    fold_assignments = np.array([fold_of_stem[s] for s in stems_arr])
    cv_alpha = np.full(len(targets), np.nan, dtype=np.float64)
    cv_log_sigma2 = np.full(len(targets), np.nan, dtype=np.float64)
    fold_scores: list[float] = []
    for f in range(n_folds):
        val_idx = np.where(fold_assignments == f)[0]
        train_idx = np.where(fold_assignments != f)[0]
        if len(val_idx) == 0 or len(train_idx) < 4:
            continue
        _, info = train_one_fold(
            train_idx,
            val_idx,
            features,
            targets,
            stems_arr,
            sigma2_pop=sigma2_pop,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            c_proj_dim=c_proj_dim,
            lambda_sigma_aux=lambda_sigma_aux,
            verbose=False,
        )
        cv_alpha[val_idx] = info["alpha_hat"]
        cv_log_sigma2[val_idx] = info["log_sigma2"]
        fold_scores.append(float(info["val_score"]))

    valid = ~np.isnan(cv_alpha)
    out = {
        "n_valid": int(valid.sum()),
        "sigma2_pop": sigma2_pop,
        "fold_scores_mean": float(np.mean(fold_scores)) if fold_scores else float("nan"),
        "r_alpha_mean": float("nan"),
        "r_alpha_seed": float("nan"),
        "r_sigma": float("nan"),
        "nll_head": float("nan"),
        "nll_baseline": float("nan"),
        "nll_improvement": float("nan"),
        "per_aspect_r": {},
    }
    if not valid.any():
        return out

    cv_a = cv_alpha[valid]
    cv_ls = cv_log_sigma2[valid]
    cv_t = targets[valid]
    stem_arr = stems_arr[valid]
    asp_arr = aspect_arr[valid]

    per_prompt_alpha, per_prompt_target = [], []
    per_prompt_sigma_hat, per_prompt_seed_std = [], []
    for s in np.unique(stem_arr):
        m = stem_arr == s
        per_prompt_alpha.append(cv_a[m].mean())
        per_prompt_target.append(cv_t[m].mean())
        if m.sum() >= 2:
            per_prompt_sigma_hat.append(np.exp(0.5 * cv_ls[m]).mean())
            per_prompt_seed_std.append(cv_t[m].std(ddof=1))

    out["r_alpha_mean"] = pearson(np.array(per_prompt_alpha), np.array(per_prompt_target))
    out["r_alpha_seed"] = pearson(cv_a, cv_t)
    if per_prompt_seed_std:
        out["r_sigma"] = pearson(
            np.array(per_prompt_sigma_hat), np.array(per_prompt_seed_std)
        )

    if sigma2_pop > 0.0:
        nll_baseline = float(
            ((cv_t - 0.0) ** 2 / (2 * sigma2_pop) + 0.5 * np.log(sigma2_pop)).mean()
        )
        sigma2_cv = np.exp(cv_ls)
        nll_head = float(
            ((cv_t - cv_a) ** 2 / (2 * sigma2_cv) + 0.5 * np.log(sigma2_cv)).mean()
        )
        out["nll_baseline"] = nll_baseline
        out["nll_head"] = nll_head
        out["nll_improvement"] = (nll_baseline - nll_head) / abs(nll_baseline)

    for a in range(N_ASPECTS):
        m = asp_arr == a
        if m.sum() < 4:
            continue
        out["per_aspect_r"][ASPECT_NAMES[a]] = {
            "n": int(m.sum()),
            "r_seed": pearson(cv_a[m], cv_t[m]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results_root",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "output" / "dcw",
            REPO_ROOT / "post_image_dataset" / "dcw",
            REPO_ROOT / "bench" / "dcw" / "results",
        ],
        help="Directories to scan for bench-script run dirs (same defaults "
        "as the production trainer). De-dups by run name.",
    )
    p.add_argument(
        "--dataset_dir", type=Path, default=REPO_ROOT / "post_image_dataset" / "lora"
    )
    p.add_argument("--text_variant", type=int, default=0)
    p.add_argument(
        "--k_warmup",
        type=int,
        default=7,
        help="Observation window for g_obs = v_rev_LL[:k_warmup]. Held fixed "
        "across the sweep so only the *target* window varies.",
    )
    p.add_argument(
        "--windows",
        type=str,
        nargs="+",
        default=[":7", "7:", "0:14", "14:", "0:7", "7:14", "14:21", "21:"],
        help="Target windows as 'start:end' (exclusive end). Default mixes "
        "head / tail / mid slices. ':7' is the user-proposed [:K] target; "
        "'7:' is the production trainer default.",
    )
    p.add_argument(
        "--per_step",
        action="store_true",
        help="Also run per-step targets (one CV per step t in 0..N-1). "
        "Expensive: n_folds × n_steps fold trainings.",
    )
    p.add_argument("--n_folds", type=int, default=8)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--c_proj_dim", type=int, default=0)
    p.add_argument("--lambda_sigma_aux", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", type=str, default=None)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("[1/5] loading bench runs ...")
    rows = load_bench_runs(args.results_root)
    if not rows:
        sys.exit("no bench rows found")
    n_steps = len(rows[0].gap_LL)
    print(
        f"  {len(rows)} rows over {len({r.run_id for r in rows})} runs, n_steps={n_steps}"
    )

    print("[2/5] loading text features ...")
    stems = sorted({r.stem for r in rows})
    feat = load_text_features(stems, args.dataset_dir, variant=args.text_variant)
    rows = [r for r in rows if r.stem in feat]
    n_unique = len({r.stem for r in rows})
    print(f"  {len(rows)} rows / {n_unique} unique stems after te-cache filter")
    if len(rows) < args.n_folds * 4:
        sys.exit(f"too few rows ({len(rows)}) for n_folds={args.n_folds}")

    print(f"[3/5] building feature matrices (k_obs={args.k_warmup}) ...")
    mu_g_pop = build_population_mu_g(rows, n_steps)
    c_pool_arr = np.stack([feat[r.stem]["c_pool"] for r in rows]).astype(np.float32)
    centroid = c_pool_arr.mean(axis=0)
    cap_len_arr = np.array(
        [feat[r.stem]["caption_length"] for r in rows], dtype=np.float32
    )
    tok_l2_arr = np.array(
        [feat[r.stem]["token_l2_std"] for r in rows], dtype=np.float32
    )
    cos_centroid = np.array(
        [
            np.dot(c, centroid) / (np.linalg.norm(c) * np.linalg.norm(centroid) + 1e-9)
            for c in c_pool_arr
        ],
        dtype=np.float32,
    )
    aux = np.stack([cap_len_arr, cos_centroid, tok_l2_arr], axis=-1)
    aux_n = (aux - aux.mean(axis=0)) / aux.std(axis=0).clip(min=1e-6)

    g_obs_arr = np.stack([r.v_rev_LL[: args.k_warmup] for r in rows]).astype(np.float32)
    g_obs_n = (g_obs_arr - g_obs_arr.mean(axis=0)) / g_obs_arr.std(axis=0).clip(min=1.0)

    aspect_arr = np.array([r.aspect_id for r in rows], dtype=np.int64)
    stems_arr = np.array([r.stem for r in rows])

    features = {
        "c_pool": c_pool_arr,
        "aspect": aspect_arr,
        "g_obs": g_obs_n,
        "aux": aux_n,
    }

    rng = np.random.default_rng(args.seed)
    unique_stems = sorted({r.stem for r in rows})
    rng.shuffle(unique_stems)
    fold_of_stem = {s: i % args.n_folds for i, s in enumerate(unique_stems)}

    label = f"k-sup-sweep-{args.label}" if args.label else "k-sup-sweep"
    run_dir = make_run_dir("dcw", label=label)
    print(f"[4/5] window sweep ({len(args.windows)} windows) -> {run_dir}")

    summary_path = run_dir / "summary.csv"
    summary_rows: list[dict] = []
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "window",
                "start",
                "end",
                "width",
                "n_valid",
                "sigma2_pop",
                "target_mean",
                "target_std",
                "r_alpha_mean",
                "r_alpha_seed",
                "r_sigma",
                "nll_head",
                "nll_baseline",
                "nll_improvement",
            ]
        )
        for spec in args.windows:
            try:
                start, end = parse_window(spec, n_steps)
            except ValueError as e:
                print(f"  skip {spec!r}: {e}")
                continue
            t = integrated_target(rows, mu_g_pop, start, end)
            print(
                f"  {spec!r} [{start}:{end}] W={end - start} "
                f"mean={t.mean():+.2f} std={t.std():.2f}"
            )
            m = cv_run(
                t,
                features,
                stems_arr,
                aspect_arr,
                fold_of_stem,
                n_folds=args.n_folds,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                c_proj_dim=args.c_proj_dim,
                lambda_sigma_aux=args.lambda_sigma_aux,
                device=device,
            )
            print(
                f"    r_α_mean={m['r_alpha_mean']:+.3f} "
                f"r_α_seed={m['r_alpha_seed']:+.3f} "
                f"r_σ={m['r_sigma']:+.3f} "
                f"ΔNLL={m['nll_improvement']:+.1%}"
            )
            w.writerow(
                [
                    spec,
                    start,
                    end,
                    end - start,
                    m["n_valid"],
                    f"{m['sigma2_pop']:.4f}",
                    f"{t.mean():.4f}",
                    f"{t.std():.4f}",
                    f"{m['r_alpha_mean']:.4f}",
                    f"{m['r_alpha_seed']:.4f}",
                    f"{m['r_sigma']:.4f}",
                    f"{m['nll_head']:.4f}",
                    f"{m['nll_baseline']:.4f}",
                    f"{m['nll_improvement']:.4f}",
                ]
            )
            summary_rows.append(
                {
                    "window": spec,
                    "start": start,
                    "end": end,
                    "width": end - start,
                    "target_mean": float(t.mean()),
                    "target_std": float(t.std()),
                    **m,
                }
            )
    artifacts = ["summary.csv"]

    if args.per_step:
        print(f"[4b/5] per-step sweep (n_steps={n_steps}) ...")
        ps_path = run_dir / "per_step.csv"
        per_step_rows: list[dict] = []
        with ps_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "t",
                    "n_valid",
                    "sigma2_pop",
                    "target_mean",
                    "target_std",
                    "r_alpha_mean",
                    "r_alpha_seed",
                    "r_sigma",
                    "nll_improvement",
                ]
            )
            for t_idx in range(n_steps):
                tv = per_step_target(rows, mu_g_pop, t_idx)
                m = cv_run(
                    tv,
                    features,
                    stems_arr,
                    aspect_arr,
                    fold_of_stem,
                    n_folds=args.n_folds,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    c_proj_dim=args.c_proj_dim,
                    lambda_sigma_aux=args.lambda_sigma_aux,
                    device=device,
                )
                print(
                    f"  t={t_idx:>2d} r_α_mean={m['r_alpha_mean']:+.3f} "
                    f"r_α_seed={m['r_alpha_seed']:+.3f} "
                    f"ΔNLL={m['nll_improvement']:+.1%}"
                )
                w.writerow(
                    [
                        t_idx,
                        m["n_valid"],
                        f"{m['sigma2_pop']:.4f}",
                        f"{tv.mean():.4f}",
                        f"{tv.std():.4f}",
                        f"{m['r_alpha_mean']:.4f}",
                        f"{m['r_alpha_seed']:.4f}",
                        f"{m['r_sigma']:.4f}",
                        f"{m['nll_improvement']:.4f}",
                    ]
                )
                per_step_rows.append(
                    {
                        "t": t_idx,
                        "target_mean": float(tv.mean()),
                        "target_std": float(tv.std()),
                        **m,
                    }
                )
        artifacts.append("per_step.csv")

    print("[5/5] writing result.json ...")
    metrics = {
        "n_rows": len(rows),
        "n_unique_stems": n_unique,
        "n_steps": n_steps,
        "k_warmup_observation": args.k_warmup,
        "windows": [
            {
                k: v
                for k, v in s.items()
                if k
                in (
                    "window",
                    "start",
                    "end",
                    "width",
                    "n_valid",
                    "sigma2_pop",
                    "target_mean",
                    "target_std",
                    "r_alpha_mean",
                    "r_alpha_seed",
                    "r_sigma",
                    "nll_improvement",
                    "per_aspect_r",
                )
            }
            for s in summary_rows
        ],
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=label,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )
    print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()
