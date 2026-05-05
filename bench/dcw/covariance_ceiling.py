"""DCW v5 — covariance-based predictability ceiling on the production 525-row pool.

Computes the Gaussian/linear R²_max = Σ_YZ Σ_ZZ⁻¹ Σ_ZY / Var(Y) for several
choices of feature set Z against the v5 supervision target Y. Splits each
ceiling into:

  - in-sample R² (population estimate; valid when N ≫ dim(Z))
  - 5-fold CV R² (honest out-of-sample, downward-biased relative to population)

Also reports the irreducible seed-noise floor: for each (stem, aspect) group
with ≥2 seeds, decomposes Var(Y) into between-group + within-group; the
within-group fraction caps every regressor whose features are (stem, aspect)-
constant (g_obs is per-row so it sees seed noise, but c_pool / aux / aspect
do not).

Usage::

    python bench/dcw/covariance_ceiling.py
    python bench/dcw/covariance_ceiling.py --supervision_window 7:28 --k_warmup 7

Reads from output/dcw/ (the same default as scripts/dcw/train_fusion_head.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from scripts.dcw.fusion_data import (  # noqa: E402
    ASPECT_NAMES,
    N_ASPECTS,
    load_bench_runs,
    load_text_features,
)
from safetensors import safe_open  # noqa: E402


def load_c_pool_with_method(
    stems: list[str], dataset_dir: Path, *, method: str, variant: int = 0
) -> dict[str, np.ndarray]:
    """Re-pool the cached cross-attention embeddings under the chosen method."""
    out: dict[str, np.ndarray] = {}
    for stem in stems:
        if stem in out:
            continue
        te_path = dataset_dir / f"{stem}_anima_te.safetensors"
        if not te_path.exists():
            continue
        with safe_open(str(te_path), framework="pt") as f:
            emb = f.get_tensor(f"crossattn_emb_v{variant}").float().numpy()  # (L, 1024)
            mask = f.get_tensor(f"attn_mask_v{variant}").bool().numpy()  # (L,)
        valid = emb[mask]  # (L_valid, 1024)
        if valid.shape[0] == 0:
            continue
        if method == "mean":
            v = valid.mean(axis=0)
        elif method == "max":
            v = valid.max(axis=0)
        elif method == "sum":
            v = valid.sum(axis=0)
        elif method == "mean_max":
            v = np.concatenate([valid.mean(axis=0), valid.max(axis=0)])
        elif method == "norm_weighted":
            n = np.linalg.norm(valid, axis=1, keepdims=True)
            v = (n * valid).sum(axis=0) / (n.sum() + 1e-9)
        else:
            raise ValueError(f"unknown c_pool_method {method!r}")
        out[stem] = v.astype(np.float32)
    return out


# ---------- ridge-aware linear R² ------------------------------------------------


def _ridge_r2(Z: np.ndarray, y: np.ndarray, ridge: float) -> float:
    """In-sample R² of the OLS/ridge fit y ~ [1, Z]. ridge=0 → plain OLS."""
    n = Z.shape[0]
    X = np.concatenate([np.ones((n, 1)), Z], axis=1)
    d = X.shape[1]
    A = X.T @ X
    if ridge > 0:
        A = A + ridge * np.eye(d)
        A[0, 0] -= ridge  # don't penalize intercept
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def _kfold_cv_r2(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None,
    n_folds: int,
    ridge: float,
    rng: np.random.Generator,
) -> float:
    """Out-of-sample R² via k-fold CV. If `groups` given, splits by unique group id
    (prompt-stratified) so the same prompt never crosses train/val."""
    n = len(y)
    if groups is None:
        idx = rng.permutation(n)
        folds = np.array_split(idx, n_folds)
    else:
        unique = np.array(sorted(set(groups.tolist())))
        rng.shuffle(unique)
        gfolds = np.array_split(unique, n_folds)
        folds = [np.where(np.isin(groups, gf))[0] for gf in gfolds]
    ss_res = 0.0
    ss_tot = 0.0
    y_mean_global = y.mean()
    for vi in folds:
        ti = np.setdiff1d(np.arange(n), vi)
        Z_t = Z[ti]
        y_t = y[ti]
        Z_v = Z[vi]
        y_v = y[vi]
        X_t = np.concatenate([np.ones((len(ti), 1)), Z_t], axis=1)
        X_v = np.concatenate([np.ones((len(vi), 1)), Z_v], axis=1)
        d = X_t.shape[1]
        A = X_t.T @ X_t
        if ridge > 0:
            A = A + ridge * np.eye(d)
            A[0, 0] -= ridge
        beta = np.linalg.solve(A, X_t.T @ y_t)
        yhat = X_v @ beta
        ss_res += float(((y_v - yhat) ** 2).sum())
        ss_tot += float(((y_v - y_mean_global) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def _pca(X: np.ndarray, k: int) -> np.ndarray:
    """Center + project to top-k principal components (no whitening)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T  # (n, k)


# ---------- main -----------------------------------------------------------------


def parse_window(spec: str, n_steps: int) -> tuple[int, int]:
    a, b = spec.split(":", 1)
    s = int(a) if a else 0
    e = int(b) if b else n_steps
    if s < 0:
        s = max(0, n_steps + s)
    if e < 0:
        e = max(0, n_steps + e)
    e = min(e, n_steps)
    if s >= e:
        raise SystemExit(f"empty window {spec!r} for n_steps={n_steps}")
    return s, e


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root",
        type=Path,
        nargs="+",
        default=[REPO_ROOT / "output" / "dcw"],
        help="Same default as the v5 trainer (output/dcw/).",
    )
    p.add_argument(
        "--dataset_dir", type=Path, default=REPO_ROOT / "post_image_dataset" / "lora"
    )
    p.add_argument("--text_variant", type=int, default=0)
    p.add_argument(
        "--supervision_window",
        type=str,
        default="0:4",
        help="Window over which Y = LSQ-optimal λ-scalar is computed. v5 default 0:4. "
        "Pass e.g. '7:28' to evaluate the report's full-tail integrated-residual framing.",
    )
    p.add_argument(
        "--k_prefix_list",
        type=int,
        nargs="+",
        default=[4, 7],
        help="Prefix lengths to evaluate g_obs[:k] against. v5 uses 4; report uses 7.",
    )
    p.add_argument(
        "--c_pool_pca_dim",
        type=int,
        default=32,
        help="Reduce 1024-D c_pool via PCA to this many components before regressing.",
    )
    p.add_argument(
        "--c_pool_norm",
        type=str,
        default="none",
        choices=("none", "l2", "standardize", "l2_then_standardize"),
        help="Preprocessing applied to c_pool BEFORE PCA. l2 = unit-norm per row "
        "(non-linear; removes magnitude differences across prompts). "
        "standardize = z-score per dim (linear; ceiling-invariant but improves "
        "conditioning). l2_then_standardize = both.",
    )
    p.add_argument(
        "--c_pool_method",
        type=str,
        default="mean",
        choices=("mean", "max", "sum", "mean_max", "norm_weighted"),
        help="Pooling over valid tokens. mean = current. max = per-dim max "
        "(skips attention-sink dilution). sum = unnormalized mean (length-coupled). "
        "mean_max = concat of mean+max (2048-d before PCA). norm_weighted = "
        "Σ ||t||·t / Σ ||t|| (emphasizes high-magnitude tokens).",
    )
    p.add_argument(
        "--seed_mean",
        action="store_true",
        help="Collapse Y and per-row features to per-(stem, aspect) seed-mean before "
        "computing R². Effective N drops to n_groups (175 here), but the within-group "
        "seed-noise floor is removed. Direct attack on the 21.5%% irreducible variance "
        "(see project_dcw_seed_variance_dominates).",
    )
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Ridge λ (small, just for numerical stability). 0 = plain OLS.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--label",
        type=str,
        default="covariance-ceiling",
        help="Label suffix for the bench/ run dir.",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- load pool ----------------------------------------------------------
    print("[1/4] loading bench runs ...")
    rows = load_bench_runs(args.results_root)
    if not rows:
        sys.exit("no rows found under " + ", ".join(str(r) for r in args.results_root))
    n_steps = len(rows[0].gap_LL)
    print(
        f"  {len(rows)} rows over {len({r.run_id for r in rows})} runs, "
        f"n_steps={n_steps}"
    )

    print("[2/4] loading text features ...")
    stems = sorted({r.stem for r in rows})
    feat = load_text_features(stems, args.dataset_dir, variant=args.text_variant)
    rows = [r for r in rows if r.stem in feat]
    print(f"  {len(feat)} unique stems, {len(rows)} rows after te-cache filter")
    n = len(rows)

    # ---- build Y ------------------------------------------------------------
    s_start, s_end = parse_window(args.supervision_window, n_steps)
    print(
        f"[3/4] building Y = LSQ-λ̂*_p over [{s_start}:{s_end}] "
        f"(s_i = 1 − σ_i, w_i = s_i, target_i = gap_LL_i) ..."
    )
    Y = np.empty(n, dtype=np.float64)
    for i, r in enumerate(rows):
        s = 1.0 - r.sigma_i[s_start:s_end]
        g = r.gap_LL[s_start:s_end]
        denom = float((s * s).sum())
        Y[i] = float((g * s).sum() / denom) if denom > 1e-9 else 0.0
    print(
        f"  Y stats: mean={Y.mean():+.2f} std={Y.std():.2f} "
        f"med={np.median(Y):+.2f} |Y|med={np.median(np.abs(Y)):.2f}"
    )

    # ---- build feature blocks ----------------------------------------------
    if args.c_pool_method == "mean":
        c_pool = np.stack([feat[r.stem]["c_pool"] for r in rows]).astype(np.float64)
    else:
        cp_alt = load_c_pool_with_method(
            stems, args.dataset_dir, method=args.c_pool_method, variant=args.text_variant
        )
        rows = [r for r in rows if r.stem in cp_alt]
        c_pool = np.stack([cp_alt[r.stem] for r in rows]).astype(np.float64)
        n = len(rows)
        print(f"  c_pool_method={args.c_pool_method} → c_pool shape {c_pool.shape}")
    cap_len = np.array(
        [feat[r.stem]["caption_length"] for r in rows], dtype=np.float64
    )
    tok_l2 = np.array(
        [feat[r.stem]["token_l2_std"] for r in rows], dtype=np.float64
    )
    centroid = c_pool.mean(axis=0)
    cos_centroid = np.array(
        [
            float(c @ centroid / (np.linalg.norm(c) * np.linalg.norm(centroid) + 1e-9))
            for c in c_pool
        ],
        dtype=np.float64,
    )
    aux = np.stack([cap_len, cos_centroid, tok_l2], axis=-1)

    aspect = np.array([r.aspect_id for r in rows], dtype=np.int64)
    aspect_oh = np.zeros((n, N_ASPECTS), dtype=np.float64)
    aspect_oh[np.arange(n), aspect] = 1.0
    aspect_oh = aspect_oh[:, 1:]  # drop one column to avoid collinearity with intercept

    g_obs_full = np.stack(
        [r.v_rev_LL[: max(args.k_prefix_list)] for r in rows]
    ).astype(np.float64)

    c_pool_proc = c_pool.copy()
    if args.c_pool_norm in ("l2", "l2_then_standardize"):
        norms = np.linalg.norm(c_pool_proc, axis=1, keepdims=True).clip(min=1e-9)
        c_pool_proc = c_pool_proc / norms
    if args.c_pool_norm in ("standardize", "l2_then_standardize"):
        mu = c_pool_proc.mean(axis=0, keepdims=True)
        sd = c_pool_proc.std(axis=0, keepdims=True).clip(min=1e-9)
        c_pool_proc = (c_pool_proc - mu) / sd
    print(f"  c_pool norm preprocessing: {args.c_pool_norm}")
    c_pool_pca = _pca(c_pool_proc, args.c_pool_pca_dim)

    # group id: same prompt at same aspect = one group (3 seeds inside)
    groups = np.array(
        [hash((r.stem, r.aspect_id)) for r in rows], dtype=np.int64
    )

    if args.seed_mean:
        # Collapse to per-group means: Y, g_obs (per-row), aspect/aux/c_pool
        # are already group-constant so just deduplicate them.
        unique_groups, inv = np.unique(groups, return_inverse=True)
        n_g = len(unique_groups)
        Y_g = np.zeros(n_g, dtype=Y.dtype)
        g_obs_g = np.zeros((n_g, g_obs_full.shape[1]), dtype=g_obs_full.dtype)
        aux_g = np.zeros((n_g, aux.shape[1]), dtype=aux.dtype)
        aspect_oh_g = np.zeros((n_g, aspect_oh.shape[1]), dtype=aspect_oh.dtype)
        c_pool_pca_g = np.zeros((n_g, c_pool_pca.shape[1]), dtype=c_pool_pca.dtype)
        aspect_g = np.zeros(n_g, dtype=aspect.dtype)
        for gi in range(n_g):
            mask = inv == gi
            Y_g[gi] = Y[mask].mean()
            g_obs_g[gi] = g_obs_full[mask].mean(axis=0)
            aux_g[gi] = aux[mask].mean(axis=0)
            aspect_oh_g[gi] = aspect_oh[mask][0]
            c_pool_pca_g[gi] = c_pool_pca[mask][0]
            aspect_g[gi] = aspect[mask][0]
        print(
            f"  --seed_mean: collapsed {n} rows → {n_g} groups, "
            f"Y std {Y.std():.2f} → {Y_g.std():.2f}"
        )
        Y = Y_g
        g_obs_full = g_obs_g
        aux = aux_g
        aspect_oh = aspect_oh_g
        c_pool_pca = c_pool_pca_g
        aspect = aspect_g
        groups = unique_groups
        n = n_g

    # ---- block table --------------------------------------------------------
    blocks: dict[str, np.ndarray] = {
        "aspect_oh (4)": aspect_oh,
        "aux (3)": aux,
        f"c_pool_pca{args.c_pool_pca_dim}": c_pool_pca,
    }
    for k in args.k_prefix_list:
        blocks[f"g_obs[:{k}]"] = g_obs_full[:, :k]
    # combinations
    for k in args.k_prefix_list:
        blocks[f"g_obs[:{k}] + aspect"] = np.concatenate(
            [g_obs_full[:, :k], aspect_oh], axis=1
        )
        blocks[f"g_obs[:{k}] + aux"] = np.concatenate(
            [g_obs_full[:, :k], aux], axis=1
        )
        blocks[f"g_obs[:{k}] + aux + c_pool_pca{args.c_pool_pca_dim}"] = np.concatenate(
            [g_obs_full[:, :k], aux, c_pool_pca], axis=1
        )
        blocks[
            f"g_obs[:{k}] + aux + c_pool_pca{args.c_pool_pca_dim} + aspect"
        ] = np.concatenate([g_obs_full[:, :k], aux, c_pool_pca, aspect_oh], axis=1)

    # ---- compute ceilings ---------------------------------------------------
    print(f"[4/4] computing R² ceilings (ridge={args.ridge}, k-fold={args.n_folds})")
    rows_out: list[dict] = []
    for name, Z in blocks.items():
        r2_in = _ridge_r2(Z, Y, ridge=args.ridge)
        r2_cv_random = _kfold_cv_r2(
            Z, Y, groups=None, n_folds=args.n_folds, ridge=args.ridge, rng=rng
        )
        r2_cv_grouped = _kfold_cv_r2(
            Z, Y, groups=groups, n_folds=args.n_folds, ridge=args.ridge, rng=rng
        )
        rows_out.append(
            {
                "block": name,
                "dim": int(Z.shape[1]),
                "r2_in_sample": float(r2_in),
                "r2_cv_random": float(r2_cv_random),
                "r2_cv_grouped": float(r2_cv_grouped),
                "r_cv_grouped": float(np.sign(r2_cv_grouped) * np.sqrt(max(r2_cv_grouped, 0.0))),
            }
        )

    # ---- seed-noise floor ---------------------------------------------------
    # Decompose Var(Y) = E[Var(Y|G)] + Var(E[Y|G]) where G = (stem, aspect).
    # within_frac = E[Var(Y|G)] / Var(Y) is the noise floor for any predictor
    # whose features are constant within G (aspect, c_pool, aux). g_obs is
    # per-row so it can in principle escape this floor.
    if args.seed_mean:
        within_frac = 0.0
        n_multi = n
        n_singleton = 0
    else:
        g_keys = np.array([(r.stem, r.aspect_id) for r in rows], dtype=object)
        g_str = np.array([f"{s}|{a}" for s, a in g_keys])
        within_ss = 0.0
        between_ss = 0.0
        y_mean_global = Y.mean()
        n_singleton = 0
        n_multi = 0
        for u in np.unique(g_str):
            mask = g_str == u
            yg = Y[mask]
            if len(yg) >= 2:
                n_multi += 1
            else:
                n_singleton += 1
            within_ss += float(((yg - yg.mean()) ** 2).sum())
            between_ss += float(len(yg) * (yg.mean() - y_mean_global) ** 2)
        total_ss = within_ss + between_ss
        within_frac = within_ss / max(total_ss, 1e-30)
    print(
        f"\n  groups: {n_multi} ≥2-seed, {n_singleton} singleton "
        f"({n_multi + n_singleton} total)"
    )
    print(
        f"  Var(Y) decomposition: within-group={within_frac:.3f} "
        f"between-group={1 - within_frac:.3f}"
    )
    print(
        f"  → group-constant feature ceiling: R²_max ≤ {1 - within_frac:.3f} "
        f"(r ≤ {np.sqrt(max(1 - within_frac, 0.0)):.3f})"
    )
    print(
        "  → per-row features (g_obs) can in principle exceed this; "
        "any block above this means features are picking up seed noise too."
    )

    # ---- print table --------------------------------------------------------
    print("\n" + "─" * 92)
    print(
        f"{'block':<55} {'dim':>4}  {'R²_in':>7}  {'R²_cv':>7}  {'R²_cv_g':>8}  {'r_cv_g':>7}"
    )
    print("─" * 92)
    for r in rows_out:
        print(
            f"{r['block']:<55} {r['dim']:>4}  "
            f"{r['r2_in_sample']:>+7.3f}  "
            f"{r['r2_cv_random']:>+7.3f}  "
            f"{r['r2_cv_grouped']:>+8.3f}  "
            f"{r['r_cv_grouped']:>+7.3f}"
        )
    print("─" * 92)
    print(
        "R²_in = in-sample (population est., valid when N ≫ dim(Z))\n"
        "R²_cv = 5-fold CV, random splits (honest out-of-sample)\n"
        "R²_cv_g = 5-fold CV, prompt-stratified (no stem leak across folds)\n"
        "r_cv_g = signed sqrt(R²_cv_g) for direct compare to fusion-head r_α."
    )

    # ---- per-aspect Y stats (sanity) ---------------------------------------
    print("\nper-aspect Y stats (sanity):")
    for a in range(N_ASPECTS):
        mask = aspect == a
        if mask.sum() == 0:
            continue
        ya = Y[mask]
        print(
            f"  {ASPECT_NAMES[a]:<10} n={mask.sum():>3}  "
            f"mean={ya.mean():+8.2f}  std={ya.std():>6.2f}  "
            f"|mean|/std={abs(ya.mean()) / max(ya.std(), 1e-9):.2f}"
        )

    # ---- write envelope -----------------------------------------------------
    run_dir = make_run_dir("dcw", label=args.label)
    metrics = {
        "n_rows": n,
        "n_unique_stems": len(stems),
        "n_groups_multi_seed": n_multi,
        "n_groups_singleton": n_singleton,
        "y": {
            "supervision_window": args.supervision_window,
            "mean": float(Y.mean()),
            "std": float(Y.std()),
            "median": float(np.median(Y)),
            "abs_median": float(np.median(np.abs(Y))),
        },
        "seed_noise_floor": {
            "within_group_var_fraction": float(within_frac),
            "max_r2_for_group_constant_features": float(1 - within_frac),
        },
        "ceilings": rows_out,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
    )
    print(f"\nwrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
