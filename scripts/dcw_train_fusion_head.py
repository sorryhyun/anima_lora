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
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from bench._common import make_run_dir, write_result  # noqa: E402
from networks.dcw import FusionHead  # noqa: E402

ASPECT_TABLE = {
    (832, 1248): 0,  # HD portrait — most common in cache
    (896, 1152): 1,  # 3:4 portrait
    (768, 1344): 2,  # tall portrait
    (1152, 896): 3,  # 3:4 landscape
    (1248, 832): 4,  # HD landscape
}
ASPECT_NAMES = ["832x1248", "896x1152", "768x1344", "1152x896", "1248x832"]
N_ASPECTS = 5


@dataclass
class Row:
    run_id: str
    aspect_id: int
    stem: str
    seed_idx: int
    gap_LL: np.ndarray  # (n_steps,) — used for target (residual on tail)
    v_rev_LL: np.ndarray  # (n_steps,) — used for input g_obs
    v_rev_source: str  # "native" | "synthetic" | "fallback"


def load_bench_runs(
    results_roots: Path | list[Path],
    *,
    require_cfg: float = 4.0,
    require_mod_w: float = 3.0,
    skip_with_lora: bool = True,
) -> list[Row]:
    if isinstance(results_roots, (str, Path)):
        results_roots = [Path(results_roots)]
    rows: list[Row] = []
    seen_run_names: set[str] = set()  # de-dup if same name appears in multiple roots
    candidate_dirs: list[Path] = []
    for root in results_roots:
        if not root.exists():
            continue
        candidate_dirs.extend(p for p in root.iterdir() if p.is_dir())
    for run_dir in sorted(candidate_dirs):
        if run_dir.name in seen_run_names:
            continue
        seen_run_names.add(run_dir.name)
        npz_path = run_dir / "gaps_per_sample.npz"
        rj_path = run_dir / "result.json"
        if not (npz_path.exists() and rj_path.exists()):
            continue
        rj = json.loads(rj_path.read_text())
        a = rj.get("args", {})
        H, W = a.get("image_h"), a.get("image_w")
        if (H, W) not in ASPECT_TABLE:
            print(f"skip {run_dir.name}: aspect {H}x{W} not in table")
            continue
        if a.get("guidance_scale") != require_cfg:
            print(
                f"skip {run_dir.name}: cfg={a.get('guidance_scale')} != {require_cfg}"
            )
            continue
        if a.get("mod_w") != require_mod_w:
            print(f"skip {run_dir.name}: mod_w={a.get('mod_w')} != {require_mod_w}")
            continue
        if skip_with_lora and a.get("lora_weight"):
            print(f"skip {run_dir.name}: has LoRA {a['lora_weight']}")
            continue
        n_seeds = int(a.get("n_seeds", 1))
        z = np.load(npz_path, allow_pickle=True)
        stems = z["stems"]
        gap_LL = z["gap_LL"]  # (N, n_steps)
        if "v_rev_LL" in z.files:
            v_rev_LL = z["v_rev_LL"]
            source = "native"
        else:
            v_fwd_pop = _load_v_fwd_pop_mean(run_dir, band="LL")
            if v_fwd_pop is not None:
                v_rev_LL = (
                    gap_LL + v_fwd_pop[None, :]
                )  # broadcast (n_steps,) → (N, n_steps)
                source = "synthetic"
            else:
                v_rev_LL = gap_LL
                source = "fallback"
        aspect_id = ASPECT_TABLE[(H, W)]
        for r in range(len(stems)):
            img_idx = r // n_seeds
            seed_idx = r % n_seeds
            rows.append(
                Row(
                    run_id=run_dir.name,
                    aspect_id=aspect_id,
                    stem=str(stems[r]),
                    seed_idx=int(
                        img_idx * 1000 + seed_idx
                    ),  # globally unique within run
                    gap_LL=np.asarray(gap_LL[r], dtype=np.float64),
                    v_rev_LL=np.asarray(v_rev_LL[r], dtype=np.float64),
                    v_rev_source=source,
                )
            )
    return rows


def _load_v_fwd_pop_mean(run_dir: Path, *, band: str = "LL") -> np.ndarray | None:
    """Read baseline_v_fwd_<band> column from per_step_bands.csv as a per-step mean."""
    csv_path = run_dir / "per_step_bands.csv"
    if not csv_path.exists():
        return None
    col = f"baseline_v_fwd_{band}"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            return np.array([float(r[col]) for r in reader], dtype=np.float64)
        except KeyError:
            return None


def load_text_features(
    stems: list[str], dataset_dir: Path, variant: int = 0
) -> dict[str, dict]:
    """Per-stem c_pool + caption_length + token_l2_std from te cache."""
    out: dict[str, dict] = {}
    for stem in stems:
        if stem in out:
            continue
        te_path = dataset_dir / f"{stem}_anima_te.safetensors"
        if not te_path.exists():
            print(f"warn: missing te cache for {stem}")
            continue
        with safe_open(str(te_path), framework="pt") as f:
            emb = f.get_tensor(f"crossattn_emb_v{variant}").float()  # (512, 1024)
            mask = f.get_tensor(f"attn_mask_v{variant}").bool()  # (512,)
        valid = emb[mask]  # (L, 1024)
        if valid.numel() == 0:
            continue
        c_pool = valid.mean(dim=0)  # (1024,)
        token_l2 = valid.norm(dim=-1)  # (L,)
        out[stem] = {
            "c_pool": c_pool.numpy().astype(np.float32),
            "caption_length": int(mask.sum().item()),
            "token_l2_std": float(token_l2.std().item()),
        }
    return out


def build_population_mu_g(rows: list[Row], n_steps: int) -> np.ndarray:
    """Single population-mean LL gap trajectory across all rows."""
    if not rows:
        return np.zeros(n_steps, dtype=np.float64)
    return np.stack([r.gap_LL for r in rows]).mean(axis=0)


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
        default=[
            REPO_ROOT / "output" / "dcw",
            REPO_ROOT / "post_image_dataset" / "dcw",
            REPO_ROOT / "bench" / "dcw" / "results",
        ],
        help="One or more directories holding bench-script run dirs. "
        "Default scans output/dcw/ (new `make dcw` output) first, then "
        "post_image_dataset/dcw/ (legacy `make dcw` output) and "
        "bench/dcw/results/ (legacy benches). De-dups by run name; "
        "first-found wins.",
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

    print(f"[4/6] building feature matrices (k={args.k_warmup}) ...")
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
            float((r.gap_LL[args.k_warmup :] - mu_g_pop[args.k_warmup :]).sum())
            for r in rows
        ],
        dtype=np.float32,
    )
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
