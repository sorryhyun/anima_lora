"""DCW v4 fusion head trainer (offline, prototype).

Pools every available `gaps_per_sample.npz` under bench/dcw/results/, joins
per-row text features from cached `{stem}_anima_te.safetensors`, and trains
the v4 fusion head with prompt-stratified K-fold CV.

The training data here is roughly 1/3 the size of the proposal's spec
(A3 = 200×3 seeds aspect-balanced); this script exists to prove the v4
hypothesis on existing bench data before committing to the A3 sample run.

Bucket conditioning was removed 2026-05-05 after the aggregate ablation
(memory `project_dcw_bucket_prior_cosmetic`) showed `aspect_emb` and
per-aspect μ_g residualization were both within-noise no-ops. The script
trains a single aggregate head; the artifact still writes per-aspect
tensors (broadcast from the single profile) for inference-side schema
compatibility.

See docs/proposal/dcw-learnable-calibrator-v4.md §A7 / §I2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from bench._common import make_run_dir, write_result  # noqa: E402
from networks.dcw import FusionHead  # noqa: E402
from scripts.dcw.fusion_data import (  # noqa: E402
    ASPECT_NAMES,
    N_ASPECTS,
    build_population_mu_g,
    load_bench_runs,
    load_text_features,
)


# Clamp range needs to span the realistic σ²_pop on Anima (~3e5 → log ≈ 12.7).
# A tighter [-10, 10] saturates at init and freezes the σ̂² head — discovered
# 2026-05-05 when the aux loss couldn't move log σ̂² off its initialised bias.
_LOG_SIGMA2_CLAMP = (-20.0, 20.0)


def gaussian_nll(alpha_hat, log_sigma2, r):
    log_sigma2 = log_sigma2.clamp(*_LOG_SIGMA2_CLAMP)
    sigma2 = torch.exp(log_sigma2)
    return ((r - alpha_hat) ** 2 / (2.0 * sigma2) + 0.5 * log_sigma2).mean()


def sigma_aux_loss(
    log_sigma2: torch.Tensor,
    targets: torch.Tensor,
    stem_groups: list[torch.Tensor],
    eps: float = 1e-3,
) -> torch.Tensor:
    """Per-prompt aggregate variance supervision.

    For each prompt with ≥2 seeds in the batch, compare:
      - prediction: mean of σ̂² across that prompt's rows
      - target:     unbiased seed variance of `target` across that prompt's rows
    in log space. Forces σ̂² to track *real* per-prompt seed variance — the
    per-row Gaussian NLL alone leaves σ̂² near-constant because single-seed
    labels can't disambiguate noise from true variance.
    """
    if not stem_groups:
        return log_sigma2.new_zeros(())
    sigma2 = torch.exp(log_sigma2.clamp(*_LOG_SIGMA2_CLAMP))
    pred_logs = []
    true_logs = []
    for idx in stem_groups:
        sigma2_p = sigma2[idx].mean()
        var_p = targets[idx].var(unbiased=True)
        pred_logs.append(torch.log(sigma2_p + eps))
        true_logs.append(torch.log(var_p + eps))
    pred = torch.stack(pred_logs)
    true = torch.stack(true_logs)
    return ((pred - true) ** 2).mean()


def _build_stem_groups(
    stems: np.ndarray, idx: np.ndarray, device: torch.device
) -> list[torch.Tensor]:
    """Group local indices (into the train/val tensor) by stem; keep ≥2 only."""
    sub = stems[idx]
    groups: list[torch.Tensor] = []
    for s in np.unique(sub):
        mask = np.where(sub == s)[0]
        if len(mask) >= 2:
            groups.append(torch.from_numpy(mask).long().to(device))
    return groups


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def train_one_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    features: dict,
    targets: np.ndarray,
    stems: np.ndarray,
    *,
    sigma2_pop: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    c_proj_dim: int,
    lambda_sigma_aux: float,
    verbose: bool = False,
) -> tuple[FusionHead, dict]:
    head = FusionHead(
        c_pool_dim=features["c_pool"].shape[1],
        n_aspects=N_ASPECTS,
        k=features["g_obs"].shape[1],
        aux_dim=features["aux"].shape[1],
        c_proj_dim=c_proj_dim,
        log_sigma2_init=float(np.log(max(sigma2_pop, 1e-6))),
    ).to(device)
    head.aspect_emb.weight.data.zero_()
    head.aspect_emb.weight.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    def to_dev(idx):
        return (
            torch.tensor(features["c_pool"][idx], device=device),
            torch.tensor(features["aspect"][idx], dtype=torch.long, device=device),
            torch.tensor(features["g_obs"][idx], device=device),
            torch.tensor(features["aux"][idx], device=device),
            torch.tensor(targets[idx], device=device),
        )

    c_t, a_t, g_t, x_t, r_t = to_dev(train_idx)
    c_v, a_v, g_v, x_v, r_v = to_dev(val_idx)
    stem_groups_train = _build_stem_groups(stems, train_idx, device)
    stem_groups_val = _build_stem_groups(stems, val_idx, device)

    best_val = float("inf")
    best_state = None
    patience, since_best = 50, 0
    last_train_aux = float("nan")
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        alpha_hat, log_sigma2 = head(c_t, a_t, g_t, x_t)
        nll = gaussian_nll(alpha_hat, log_sigma2, r_t)
        if lambda_sigma_aux > 0.0:
            aux = sigma_aux_loss(log_sigma2, r_t, stem_groups_train)
            loss = nll + lambda_sigma_aux * aux
            last_train_aux = float(aux.item())
        else:
            loss = nll
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            ah_v, ls_v = head(c_v, a_v, g_v, x_v)
            val_nll = gaussian_nll(ah_v, ls_v, r_v).item()
            val_aux = (
                float(sigma_aux_loss(ls_v, r_v, stem_groups_val).item())
                if stem_groups_val
                else float("nan")
            )
        # Early-stop on the same composite loss the trainer is minimising,
        # so the aux objective actually shapes the kept checkpoint.
        val_score = val_nll + (
            lambda_sigma_aux * val_aux if lambda_sigma_aux > 0.0 else 0.0
        )
        if val_score < best_val - 1e-4:
            best_val = val_score
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
        if since_best >= patience:
            break
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(
                f"  ep {ep:4d} train_nll={nll.item():.4f} "
                f"train_aux={last_train_aux:.4f} val_nll={val_nll:.4f} "
                f"val_aux={val_aux:.4f}"
            )

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        ah_v, ls_v = head(c_v, a_v, g_v, x_v)
    return head, {
        "val_score": best_val,
        "alpha_hat": ah_v.cpu().numpy(),
        "log_sigma2": ls_v.cpu().numpy(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root",
        type=Path,
        nargs="+",
        default=[REPO_ROOT / "output" / "dcw"],
        help="Directories holding bench-script run dirs. Default = output/dcw/ "
        "only (the clean `make dcw` baseline pool). Legacy roots "
        "(post_image_dataset/dcw, bench/dcw/results) are excluded by default "
        "because they contained heterogeneous configs (A2 calibration runs "
        "with dcw_sweep=True mixed with λ=0 baselines), which contaminated "
        "the 2026-05-05 seed-mean ablation. Pass them explicitly to opt back in.",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=REPO_ROOT / "output" / "dcw",
        help="Where to write the trained fusion_head.safetensors run dir. "
        "Default output/dcw/.",
    )
    p.add_argument(
        "--dataset_dir", type=Path, default=REPO_ROOT / "post_image_dataset" / "lora"
    )
    p.add_argument("--text_variant", type=int, default=0)
    p.add_argument("--k_warmup", type=int, default=7)
    p.add_argument(
        "--target_window",
        type=str,
        default=None,
        help="Target window as 'start:end' (exclusive end). Default = "
        "'k_warmup:n_steps' (full tail), matching the production inference "
        "contract. The 2026-05-05 k_supervision_sweep bench showed mid-tail "
        "windows (7:14, 21:) carry r_α≈0.52–0.55 vs full-tail's 0.47, so "
        "narrowing the supervision target is a cheap win — but the inference "
        "path must also be windowed to match (apply λ only over those steps).",
    )
    p.add_argument(
        "--seed_mean_targets",
        action="store_true",
        help="Replace each row's target with its per-prompt seed-mean. "
        "Direct attack on the seed-noise ceiling (project_dcw_seed_variance_dominates) "
        "for prompts with ≥2 seeds. Requires multi-seed pool (output/dcw/ "
        "make-dcw-* runs ship 3 seeds/prompt). Inputs (g_obs, c_pool, aux) "
        "stay per-row so the head still sees per-seed observations.",
    )
    p.add_argument("--n_folds", type=int, default=8)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", type=str, default="prototype")
    p.add_argument(
        "--c_proj_dim",
        type=int,
        default=0,
        help="Project c_pool down to this many dims before concat. 0 = "
        "identity (raw 1024-d). The 2026-05-05 sweep showed projection "
        "balances slot capacity but hurts CV r — kept as ablation knob.",
    )
    p.add_argument(
        "--lambda_sigma_aux",
        type=float,
        default=1.0,
        help="Weight on the per-prompt aggregate σ̂² supervision (log-MSE "
        "of mean σ̂²_p vs unbiased seed-var of target_p, only over stems "
        "with ≥2 seeds in batch). 0 disables. Default 1.0 because the loss "
        "is in log space and roughly scale-matched to NLL.",
    )
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("[1/6] loading bench runs ...")
    rows = load_bench_runs(args.results_root)
    if not rows:
        sys.exit("no bench rows found")
    n_steps = len(rows[0].gap_LL)
    print(
        f"  {len(rows)} rows over {len({r.run_id for r in rows})} runs, "
        f"n_steps={n_steps}"
    )
    counts = {a: sum(1 for r in rows if r.aspect_id == a) for a in range(N_ASPECTS)}
    print(
        "  per-aspect counts: "
        + ", ".join(f"{ASPECT_NAMES[a]}={c}" for a, c in counts.items())
    )

    print("[2/6] loading text features ...")
    stems = sorted({r.stem for r in rows})
    feat = load_text_features(stems, args.dataset_dir, variant=args.text_variant)
    rows = [r for r in rows if r.stem in feat]
    print(f"  {len(feat)} unique stems, {len(rows)} rows after te-cache filter")

    print("[3/6] computing population μ_g ...")
    mu_g_pop = build_population_mu_g(rows, n_steps)  # (n_steps,)
    mu_g = np.broadcast_to(mu_g_pop, (N_ASPECTS, n_steps)).copy()
    s_pop = np.zeros_like(mu_g)
    lam_scalar = np.zeros(N_ASPECTS, dtype=np.float32)
    print(
        f"  integrated μ_g = {mu_g_pop.sum():+.2f} "
        "(single profile, broadcast to all aspects in artifact)"
    )

    if args.target_window is not None:
        if ":" not in args.target_window:
            sys.exit(f"--target_window must contain ':' (got {args.target_window!r})")
        a, b = args.target_window.split(":", 1)
        t_start = int(a) if a else 0
        t_end = int(b) if b else n_steps
        if t_start < 0:
            t_start = max(0, n_steps + t_start)
        if t_end < 0:
            t_end = max(0, n_steps + t_end)
        t_end = min(t_end, n_steps)
        if t_start >= t_end:
            sys.exit(f"empty target_window {args.target_window!r} for n_steps={n_steps}")
    else:
        t_start, t_end = args.k_warmup, n_steps

    print(
        f"[4/6] building feature matrices (k_obs={args.k_warmup}, "
        f"target=[{t_start}:{t_end}]) ..."
    )
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
    aux_mean = aux.mean(axis=0)
    aux_std = aux.std(axis=0).clip(min=1e-6)
    aux_n = (aux - aux_mean) / aux_std

    aspect_arr = np.array([r.aspect_id for r in rows], dtype=np.int64)
    src_counts: dict[str, int] = {}
    for r in rows:
        src_counts[r.v_rev_source] = src_counts.get(r.v_rev_source, 0) + 1
    print("  v_rev sources: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))
    if "fallback" in src_counts:
        print(
            "  WARN: 'fallback' rows use raw gap_LL as g_obs — no v_fwd reference found. "
            "Add v_rev_LL via fresh bench run to fix."
        )
    g_obs_arr = np.stack([r.v_rev_LL[: args.k_warmup] for r in rows]).astype(np.float32)
    g_obs_mean = g_obs_arr.mean(axis=0)
    g_obs_std = g_obs_arr.std(axis=0).clip(min=1.0)
    g_obs_n = (g_obs_arr - g_obs_mean) / g_obs_std

    targets = np.array(
        [
            float((r.gap_LL[t_start:t_end] - mu_g_pop[t_start:t_end]).sum())
            for r in rows
        ],
        dtype=np.float32,
    )
    if args.seed_mean_targets:
        # Group by (stem, aspect_id) — same prompt at different aspects has
        # different gap shape (∂∫g/∂λ flips sign across aspects per the
        # bucket-cosmetic memo's table), so pooling those rows would erase
        # real per-aspect signal as if it were seed noise. Within one
        # (stem, aspect) group, each row is one seed of the same setup.
        groups_by_idx = np.array(
            [(r.stem, r.aspect_id) for r in rows], dtype=object
        )
        per_group_mean: dict[tuple, float] = {}
        per_group_n: dict[tuple, int] = {}
        for i, g in enumerate(groups_by_idx):
            t = tuple(g)
            per_group_n[t] = per_group_n.get(t, 0) + 1
        for t in per_group_n:
            mask = np.array([tuple(g) == t for g in groups_by_idx])
            per_group_mean[t] = float(targets[mask].mean())
        targets_seedmean = np.array(
            [per_group_mean[(r.stem, r.aspect_id)] for r in rows], dtype=np.float32
        )
        n_multi = sum(1 for n in per_group_n.values() if n >= 2)
        n_single = len(per_group_n) - n_multi
        std_before = float(targets.std())
        std_after = float(targets_seedmean.std())
        print(
            f"  --seed_mean_targets: collapsed labels to per-(stem,aspect) mean "
            f"({n_multi} groups w/ ≥2 seeds, {n_single} singleton groups, "
            f"{len(per_group_n)} total groups)"
        )
        print(
            f"  target std: per-row={std_before:.1f} → seed-mean={std_after:.1f} "
            f"(seed-noise share = 1 - (after/before)² = "
            f"{1 - (std_after / max(std_before, 1e-9)) ** 2:.1%})"
        )
        targets = targets_seedmean
    sigma2_pop = float(targets.var())
    print(
        f"  targets: mean={targets.mean():+.2f}, std={targets.std():.2f}, "
        f"sigma²_pop={sigma2_pop:.2f}"
    )

    features = {
        "c_pool": c_pool_arr,
        "aspect": aspect_arr,
        "g_obs": g_obs_n,
        "aux": aux_n,
    }

    print(f"[5/6] {args.n_folds}-fold prompt-stratified CV ...")
    rng = np.random.default_rng(args.seed)
    unique_stems = sorted({r.stem for r in rows})
    rng.shuffle(unique_stems)
    fold_of_stem = {s: i % args.n_folds for i, s in enumerate(unique_stems)}
    fold_assignments = np.array([fold_of_stem[r.stem] for r in rows])

    cv_alpha = np.full(len(rows), np.nan, dtype=np.float64)
    cv_log_sigma2 = np.full(len(rows), np.nan, dtype=np.float64)
    stems_arr = np.array([r.stem for r in rows])
    fold_scores = []
    for f in range(args.n_folds):
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
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            c_proj_dim=args.c_proj_dim,
            lambda_sigma_aux=args.lambda_sigma_aux,
            verbose=args.verbose,
        )
        cv_alpha[val_idx] = info["alpha_hat"]
        cv_log_sigma2[val_idx] = info["log_sigma2"]
        fold_scores.append(info["val_score"])
        print(
            f"  fold {f}: n_train={len(train_idx)} n_val={len(val_idx)} "
            f"val_score={info['val_score']:.4f}"
        )

    valid = ~np.isnan(cv_alpha)
    cv_a = cv_alpha[valid]
    cv_ls = cv_log_sigma2[valid]
    cv_t = targets[valid]
    cv_aspect = aspect_arr[valid]
    stem_arr = np.array([r.stem for r in rows])[valid]

    # Per-prompt seed-mean correlations
    per_prompt_alpha, per_prompt_target = [], []
    per_prompt_sigma_hat, per_prompt_seed_std = [], []
    for s in np.unique(stem_arr):
        m = stem_arr == s
        if m.sum() < 1:
            continue
        per_prompt_alpha.append(cv_a[m].mean())
        per_prompt_target.append(cv_t[m].mean())
        if m.sum() >= 2:
            per_prompt_sigma_hat.append(np.exp(0.5 * cv_ls[m]).mean())
            per_prompt_seed_std.append(cv_t[m].std(ddof=1))

    r_alpha_mean = pearson(np.array(per_prompt_alpha), np.array(per_prompt_target))
    r_alpha_seed = pearson(cv_a, cv_t)
    r_sigma = (
        pearson(np.array(per_prompt_sigma_hat), np.array(per_prompt_seed_std))
        if per_prompt_seed_std
        else float("nan")
    )

    # NLL improvement vs N(0, σ²_pop) baseline
    nll_baseline = float(
        ((cv_t - 0.0) ** 2 / (2 * sigma2_pop) + 0.5 * np.log(sigma2_pop)).mean()
    )
    sigma2_cv = np.exp(cv_ls)
    nll_head = float(
        ((cv_t - cv_a) ** 2 / (2 * sigma2_cv) + 0.5 * np.log(sigma2_cv)).mean()
    )
    nll_improve = (nll_baseline - nll_head) / abs(nll_baseline)

    # Per-aspect r
    per_aspect_metrics = {}
    for a in range(N_ASPECTS):
        m = cv_aspect == a
        if m.sum() < 4:
            continue
        per_aspect_metrics[ASPECT_NAMES[a]] = {
            "n": int(m.sum()),
            "r_alpha_seed": pearson(cv_a[m], cv_t[m]),
            "rmse": float(np.sqrt(((cv_a[m] - cv_t[m]) ** 2).mean())),
        }

    print("\n=== CV summary ===")
    print(f"r(α̂_p, mean_s r) [per-prompt] = {r_alpha_mean:+.3f}  (gate ≥ 0.6)")
    print(f"r(α̂_p,s, r_p,s) [seed-cond]   = {r_alpha_seed:+.3f}  (gate ≥ 0.7)")
    print(f"r(σ̂_p, std_s r)               = {r_sigma:+.3f}      (gate ≥ 0.4)")
    print(
        f"NLL head={nll_head:.3f} vs baseline={nll_baseline:.3f}  "
        f"({nll_improve:+.1%}, gate ≥ +15%)"
    )
    for k, v in per_aspect_metrics.items():
        print(f"  {k}: n={v['n']} r_seed={v['r_alpha_seed']:+.3f} rmse={v['rmse']:.1f}")

    print("[6/6] final fit on all data + dump artifact ...")
    full_idx = np.arange(len(rows))
    final_head, _ = train_one_fold(
        full_idx,
        full_idx,
        features,
        targets,
        stems_arr,
        sigma2_pop=sigma2_pop,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        c_proj_dim=args.c_proj_dim,
        lambda_sigma_aux=args.lambda_sigma_aux,
    )

    sigma2_prior_pop = float(np.exp(cv_ls).mean()) if cv_ls.size else float(sigma2_pop)
    sigma2_prior = np.full(N_ASPECTS, sigma2_prior_pop, dtype=np.float32)

    run_dir = make_run_dir(
        "dcw", label=f"v4-fusion-head-{args.label}", root=args.out_root
    )

    state = {f"head.{k}": v.cpu() for k, v in final_head.state_dict().items()}
    state["bucket_prior_mu_g"] = torch.tensor(mu_g, dtype=torch.float32)
    state["bucket_prior_S_pop"] = torch.tensor(s_pop, dtype=torch.float32)
    state["bucket_prior_lam_scalar"] = torch.tensor(lam_scalar, dtype=torch.float32)
    state["centroid_c_pool"] = torch.tensor(centroid, dtype=torch.float32)
    state["aux_mean"] = torch.tensor(aux_mean, dtype=torch.float32)
    state["aux_std"] = torch.tensor(aux_std, dtype=torch.float32)
    state["g_obs_mean"] = torch.tensor(g_obs_mean, dtype=torch.float32)
    state["g_obs_std"] = torch.tensor(g_obs_std, dtype=torch.float32)
    state["sigma2_prior"] = torch.tensor(sigma2_prior, dtype=torch.float32)
    meta = {
        "schema": "dcw_v4_fusion_head",
        "k_warmup": str(args.k_warmup),
        "target_start": str(t_start),
        "target_end": str(t_end),
        "n_aspects": str(N_ASPECTS),
        "aspect_names": ",".join(ASPECT_NAMES),
        "n_steps": str(n_steps),
        "text_variant": str(args.text_variant),
        "n_train_rows": str(len(rows)),
        "sigma2_pop": f"{sigma2_pop:.4f}",
        "c_proj_dim": str(args.c_proj_dim),
    }
    save_file(state, str(run_dir / "fusion_head.safetensors"), metadata=meta)

    cv_csv = run_dir / "cv_predictions.csv"
    with cv_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stem",
                "aspect",
                "seed_idx",
                "target_r",
                "alpha_hat",
                "sigma2_hat",
                "fold",
            ]
        )
        for i, r in enumerate(rows):
            if not np.isnan(cv_alpha[i]):
                w.writerow(
                    [
                        r.stem,
                        ASPECT_NAMES[r.aspect_id],
                        r.seed_idx,
                        f"{targets[i]:.4f}",
                        f"{cv_alpha[i]:.4f}",
                        f"{np.exp(cv_log_sigma2[i]):.4f}",
                        fold_of_stem[r.stem],
                    ]
                )

    metrics = {
        "n_rows": len(rows),
        "n_unique_stems": len(unique_stems),
        "per_aspect_counts": {ASPECT_NAMES[a]: counts[a] for a in range(N_ASPECTS)},
        "k_warmup": args.k_warmup,
        "target_start": t_start,
        "target_end": t_end,
        "n_folds": args.n_folds,
        "sigma2_pop": sigma2_pop,
        "sigma2_prior": {
            ASPECT_NAMES[a]: float(sigma2_prior[a]) for a in range(N_ASPECTS)
        },
        "lam_scalar": {ASPECT_NAMES[a]: float(lam_scalar[a]) for a in range(N_ASPECTS)},
        "cv_fold_val_scores": fold_scores,
        "r_alpha_mean_per_prompt": r_alpha_mean,
        "r_alpha_seed_conditional": r_alpha_seed,
        "r_sigma_per_prompt": r_sigma,
        "nll_head": nll_head,
        "nll_baseline": nll_baseline,
        "nll_improvement": nll_improve,
        "per_aspect_metrics": per_aspect_metrics,
        "gates": {
            "r_alpha_mean_pass": r_alpha_mean >= 0.6,
            "r_alpha_seed_pass": r_alpha_seed >= 0.7,
            "r_sigma_pass": (r_sigma >= 0.4) if not np.isnan(r_sigma) else None,
            "nll_improvement_pass": nll_improve >= 0.15,
        },
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
        artifacts=["fusion_head.safetensors", "cv_predictions.csv"],
    )
    print(f"\nartifacts written to {run_dir}")


if __name__ == "__main__":
    main()
