"""DCW v4 fusion head trainer (offline, prototype).

Pools every available `gaps_per_sample.npz` under bench/dcw/results/, joins
per-row text features from cached `{stem}_anima_te.safetensors`, and trains
the v4 fusion head with prompt-stratified K-fold CV.

The training data here is roughly 1/3 the size of the proposal's spec
(A3 = 200×3 seeds aspect-balanced); this script exists to prove the v4
hypothesis on existing bench data before committing to the A3 sample run.

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
    (1024, 1024): 0,  # square
    (832, 1248): 1,   # HD (portrait)
    (1248, 832): 2,   # inv-HD (landscape)
}
ASPECT_NAMES = ["1024x1024", "832x1248", "1248x832"]
N_ASPECTS = 3


@dataclass
class Row:
    run_id: str
    aspect_id: int
    stem: str
    seed_idx: int
    gap_LL: np.ndarray   # (n_steps,) — used for target (residual on tail)
    v_rev_LL: np.ndarray  # (n_steps,) — used for input g_obs
    v_rev_source: str     # "native" | "synthetic" | "fallback"


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
            print(f"skip {run_dir.name}: cfg={a.get('guidance_scale')} != {require_cfg}")
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
                v_rev_LL = gap_LL + v_fwd_pop[None, :]  # broadcast (n_steps,) → (N, n_steps)
                source = "synthetic"
            else:
                v_rev_LL = gap_LL
                source = "fallback"
        aspect_id = ASPECT_TABLE[(H, W)]
        for r in range(len(stems)):
            img_idx = r // n_seeds
            seed_idx = r % n_seeds
            rows.append(Row(
                run_id=run_dir.name,
                aspect_id=aspect_id,
                stem=str(stems[r]),
                seed_idx=int(img_idx * 1000 + seed_idx),  # globally unique within run
                gap_LL=np.asarray(gap_LL[r], dtype=np.float64),
                v_rev_LL=np.asarray(v_rev_LL[r], dtype=np.float64),
                v_rev_source=source,
            ))
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
            mask = f.get_tensor(f"attn_mask_v{variant}").bool()       # (512,)
        valid = emb[mask]                                  # (L, 1024)
        if valid.numel() == 0:
            continue
        c_pool = valid.mean(dim=0)                          # (1024,)
        token_l2 = valid.norm(dim=-1)                       # (L,)
        out[stem] = {
            "c_pool": c_pool.numpy().astype(np.float32),
            "caption_length": int(mask.sum().item()),
            "token_l2_std": float(token_l2.std().item()),
        }
    return out


def build_bucket_prior(rows: list[Row], n_steps: int) -> np.ndarray:
    """Per-aspect mean LL gap trajectory from baseline rows."""
    mu_g = np.zeros((N_ASPECTS, n_steps), dtype=np.float64)
    counts = np.zeros(N_ASPECTS, dtype=np.int64)
    for r in rows:
        mu_g[r.aspect_id] += r.gap_LL
        counts[r.aspect_id] += 1
    for a in range(N_ASPECTS):
        if counts[a] > 0:
            mu_g[a] /= counts[a]
        else:
            print(f"warn: no rows for aspect {ASPECT_NAMES[a]}")
    return mu_g


def load_a2_calibration(results_roots: Path | list[Path]) -> dict[str, dict]:
    """Read S_pop and λ_scalar from the three A2 runs' per_step_bands.csv.

    S_pop[i] := (g_λ[i] − g_base[i]) / λ.  λ_scalar by least-squares per-step
    zeroing: λ* = −Σ(μ_g · S_pop) / Σ(S_pop²). Searches all configured roots.
    """
    if isinstance(results_roots, (str, Path)):
        results_roots = [Path(results_roots)]
    a2_runs = {
        "1024x1024": "20260504-1648-v4-A2-1024x1024-pos01",
        "832x1248":  "20260504-1721-v4-A2-832x1248-pos01",
        "1248x832":  "20260504-1747-v4-A2-1248x832-pos01",
    }
    out: dict[str, dict] = {}
    for aspect_name, run_name in a2_runs.items():
        csv_path: Path | None = None
        for root in results_roots:
            cand = root / run_name / "per_step_bands.csv"
            if cand.exists():
                csv_path = cand
                break
        if csv_path is None:
            print(f"warn: missing A2 csv for {aspect_name}")
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rowdicts = list(reader)
        base_LL = np.array([float(r["baseline_gap_LL"]) for r in rowdicts])
        # find the λ=0.01 LL column
        lam_col = next(
            (k for k in rowdicts[0]
             if k.startswith("λ=0.01") and k.endswith("_gap_LL")),
            None,
        )
        if lam_col is None:
            print(f"warn: no λ=0.01 LL column in {run_name}")
            continue
        lam_LL = np.array([float(r[lam_col]) for r in rowdicts])
        s_pop = (lam_LL - base_LL) / 0.01
        lam_lsq = float(-(base_LL * s_pop).sum() / max(1e-12, (s_pop**2).sum()))
        out[aspect_name] = {
            "mu_g_baseline": base_LL.astype(np.float32),
            "S_pop": s_pop.astype(np.float32),
            "lam_scalar": lam_lsq,
        }
    return out


def gaussian_nll(alpha_hat, log_sigma2, r):
    log_sigma2 = log_sigma2.clamp(min=-10.0, max=10.0)
    sigma2 = torch.exp(log_sigma2)
    return ((r - alpha_hat) ** 2 / (2.0 * sigma2) + 0.5 * log_sigma2).mean()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def train_one_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    features: dict,
    targets: np.ndarray,
    *,
    sigma2_pop: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    verbose: bool = False,
    disable_aspect: bool = False,
) -> tuple[FusionHead, dict]:
    head = FusionHead(
        c_pool_dim=features["c_pool"].shape[1],
        k=features["g_obs"].shape[1],
        aux_dim=features["aux"].shape[1],
        log_sigma2_init=float(np.log(max(sigma2_pop, 1e-6))),
    ).to(device)
    if disable_aspect:
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

    best_val = float("inf")
    best_state = None
    patience, since_best = 50, 0
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        alpha_hat, log_sigma2 = head(c_t, a_t, g_t, x_t)
        loss = gaussian_nll(alpha_hat, log_sigma2, r_t)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            ah_v, ls_v = head(c_v, a_v, g_v, x_v)
            val_nll = gaussian_nll(ah_v, ls_v, r_v).item()
        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
        if since_best >= patience:
            break
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  ep {ep:4d} train_nll={loss.item():.4f} val_nll={val_nll:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        ah_v, ls_v = head(c_v, a_v, g_v, x_v)
    return head, {
        "val_nll": best_val,
        "alpha_hat": ah_v.cpu().numpy(),
        "log_sigma2": ls_v.cpu().numpy(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root", type=Path, nargs="+",
        default=[
            REPO_ROOT / "post_image_dataset" / "dcw",
            REPO_ROOT / "bench" / "dcw" / "results",
        ],
        help="One or more directories holding bench-script run dirs. "
        "Default scans both post_image_dataset/dcw/ (new `make dcw` output) "
        "and bench/dcw/results/ (legacy + A2 calibration). De-dups by run "
        "name; first-found wins.",
    )
    p.add_argument(
        "--out_root", type=Path, default=REPO_ROOT / "post_image_dataset" / "dcw",
        help="Where to write the trained fusion_head.safetensors run dir. "
        "Default post_image_dataset/dcw/.",
    )
    p.add_argument("--dataset_dir", type=Path,
                   default=REPO_ROOT / "post_image_dataset" / "lora")
    p.add_argument("--text_variant", type=int, default=0)
    p.add_argument("--k_warmup", type=int, default=7)
    p.add_argument("--n_folds", type=int, default=8)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", type=str, default="prototype")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--disable_aspect", action="store_true",
                   help="Ablate aspect_emb (force to zero) to test "
                   "whether the bucket-residualized target still has "
                   "aspect-conditional structure for the head to use.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("[1/6] loading bench runs ...")
    rows = load_bench_runs(args.results_root)
    if not rows:
        sys.exit("no bench rows found")
    n_steps = len(rows[0].gap_LL)
    print(f"  {len(rows)} rows over {len({r.run_id for r in rows})} runs, "
          f"n_steps={n_steps}")
    counts = {a: sum(1 for r in rows if r.aspect_id == a) for a in range(N_ASPECTS)}
    print("  per-aspect counts: " + ", ".join(
        f"{ASPECT_NAMES[a]}={c}" for a, c in counts.items()))

    print("[2/6] loading text features ...")
    stems = sorted({r.stem for r in rows})
    feat = load_text_features(stems, args.dataset_dir, variant=args.text_variant)
    rows = [r for r in rows if r.stem in feat]
    print(f"  {len(feat)} unique stems, {len(rows)} rows after te-cache filter")

    print("[3/6] computing bucket prior + A2 calibration ...")
    mu_g = build_bucket_prior(rows, n_steps)  # (N_ASPECTS, n_steps)
    a2 = load_a2_calibration(args.results_root)
    s_pop = np.zeros_like(mu_g)
    lam_scalar = np.zeros(N_ASPECTS, dtype=np.float32)
    for a, name in enumerate(ASPECT_NAMES):
        if name in a2:
            s_pop[a] = a2[name]["S_pop"]
            lam_scalar[a] = a2[name]["lam_scalar"]
    for a, name in enumerate(ASPECT_NAMES):
        print(f"  {name}: integrated μ_g = {mu_g[a].sum():+.2f}, "
              f"λ_scalar = {lam_scalar[a]:+.4f}")

    print(f"[4/6] building feature matrices (k={args.k_warmup}) ...")
    c_pool_arr = np.stack([feat[r.stem]["c_pool"] for r in rows]).astype(np.float32)
    centroid = c_pool_arr.mean(axis=0)
    cap_len_arr = np.array([feat[r.stem]["caption_length"] for r in rows], dtype=np.float32)
    tok_l2_arr = np.array([feat[r.stem]["token_l2_std"] for r in rows], dtype=np.float32)
    cos_centroid = np.array([
        np.dot(c, centroid) / (np.linalg.norm(c) * np.linalg.norm(centroid) + 1e-9)
        for c in c_pool_arr
    ], dtype=np.float32)
    aux = np.stack([cap_len_arr, cos_centroid, tok_l2_arr], axis=-1)
    aux_mean = aux.mean(axis=0); aux_std = aux.std(axis=0).clip(min=1e-6)
    aux_n = (aux - aux_mean) / aux_std

    aspect_arr = np.array([r.aspect_id for r in rows], dtype=np.int64)
    src_counts: dict[str, int] = {}
    for r in rows:
        src_counts[r.v_rev_source] = src_counts.get(r.v_rev_source, 0) + 1
    print("  v_rev sources: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))
    if "fallback" in src_counts:
        print("  WARN: 'fallback' rows use raw gap_LL as g_obs — no v_fwd reference found. "
              "Add v_rev_LL via fresh bench run to fix.")
    g_obs_arr = np.stack([r.v_rev_LL[: args.k_warmup] for r in rows]).astype(np.float32)
    g_obs_mean = g_obs_arr.mean(axis=0); g_obs_std = g_obs_arr.std(axis=0).clip(min=1.0)
    g_obs_n = (g_obs_arr - g_obs_mean) / g_obs_std

    targets = np.array([
        float((r.gap_LL[args.k_warmup:] - mu_g[r.aspect_id, args.k_warmup:]).sum())
        for r in rows
    ], dtype=np.float32)
    sigma2_pop = float(targets.var())
    print(f"  targets: mean={targets.mean():+.2f}, std={targets.std():.2f}, "
          f"sigma²_pop={sigma2_pop:.2f}")

    features = {
        "c_pool": c_pool_arr, "aspect": aspect_arr,
        "g_obs": g_obs_n, "aux": aux_n,
    }

    print(f"[5/6] {args.n_folds}-fold prompt-stratified CV ...")
    rng = np.random.default_rng(args.seed)
    unique_stems = sorted({r.stem for r in rows})
    rng.shuffle(unique_stems)
    fold_of_stem = {s: i % args.n_folds for i, s in enumerate(unique_stems)}
    fold_assignments = np.array([fold_of_stem[r.stem] for r in rows])

    cv_alpha = np.full(len(rows), np.nan, dtype=np.float64)
    cv_log_sigma2 = np.full(len(rows), np.nan, dtype=np.float64)
    fold_nlls = []
    for f in range(args.n_folds):
        val_idx = np.where(fold_assignments == f)[0]
        train_idx = np.where(fold_assignments != f)[0]
        if len(val_idx) == 0 or len(train_idx) < 4:
            continue
        _, info = train_one_fold(
            train_idx, val_idx, features, targets,
            sigma2_pop=sigma2_pop, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, device=device, verbose=args.verbose,
            disable_aspect=args.disable_aspect,
        )
        cv_alpha[val_idx] = info["alpha_hat"]
        cv_log_sigma2[val_idx] = info["log_sigma2"]
        fold_nlls.append(info["val_nll"])
        print(f"  fold {f}: n_train={len(train_idx)} n_val={len(val_idx)} "
              f"val_nll={info['val_nll']:.4f}")

    valid = ~np.isnan(cv_alpha)
    cv_a = cv_alpha[valid]; cv_ls = cv_log_sigma2[valid]
    cv_t = targets[valid]; cv_aspect = aspect_arr[valid]
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
    r_sigma = (pearson(np.array(per_prompt_sigma_hat), np.array(per_prompt_seed_std))
               if per_prompt_seed_std else float("nan"))

    # NLL improvement vs N(0, σ²_pop) baseline
    nll_baseline = float(((cv_t - 0.0) ** 2 / (2 * sigma2_pop)
                          + 0.5 * np.log(sigma2_pop)).mean())
    sigma2_cv = np.exp(cv_ls)
    nll_head = float(((cv_t - cv_a) ** 2 / (2 * sigma2_cv)
                      + 0.5 * np.log(sigma2_cv)).mean())
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
    print(f"NLL head={nll_head:.3f} vs baseline={nll_baseline:.3f}  "
          f"({nll_improve:+.1%}, gate ≥ +15%)")
    for k, v in per_aspect_metrics.items():
        print(f"  {k}: n={v['n']} r_seed={v['r_alpha_seed']:+.3f} rmse={v['rmse']:.1f}")

    print("[6/6] final fit on all data + dump artifact ...")
    full_idx = np.arange(len(rows))
    final_head, _ = train_one_fold(
        full_idx, full_idx, features, targets,
        sigma2_pop=sigma2_pop, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, device=device,
        disable_aspect=args.disable_aspect,
    )

    sigma2_prior = np.zeros(N_ASPECTS, dtype=np.float32)
    for a in range(N_ASPECTS):
        m = (cv_aspect == a) & (~np.isnan(cv_log_sigma2[valid]))
        if m.sum() > 0:
            sigma2_prior[a] = float(np.exp(cv_ls[m]).mean())
        else:
            sigma2_prior[a] = float(sigma2_pop)

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
    }
    save_file(state, str(run_dir / "fusion_head.safetensors"), metadata=meta)

    cv_csv = run_dir / "cv_predictions.csv"
    with cv_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "aspect", "seed_idx", "target_r", "alpha_hat",
                    "sigma2_hat", "fold"])
        for i, r in enumerate(rows):
            if not np.isnan(cv_alpha[i]):
                w.writerow([r.stem, ASPECT_NAMES[r.aspect_id], r.seed_idx,
                            f"{targets[i]:.4f}", f"{cv_alpha[i]:.4f}",
                            f"{np.exp(cv_log_sigma2[i]):.4f}", fold_of_stem[r.stem]])

    metrics = {
        "n_rows": len(rows),
        "n_unique_stems": len(unique_stems),
        "per_aspect_counts": {ASPECT_NAMES[a]: counts[a] for a in range(N_ASPECTS)},
        "k_warmup": args.k_warmup,
        "n_folds": args.n_folds,
        "sigma2_pop": sigma2_pop,
        "sigma2_prior": {ASPECT_NAMES[a]: float(sigma2_prior[a])
                         for a in range(N_ASPECTS)},
        "lam_scalar": {ASPECT_NAMES[a]: float(lam_scalar[a])
                       for a in range(N_ASPECTS)},
        "cv_fold_val_nlls": fold_nlls,
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
        run_dir, script=__file__, args=args, metrics=metrics,
        label=args.label, device=device,
        artifacts=["fusion_head.safetensors", "cv_predictions.csv"],
    )
    print(f"\nartifacts written to {run_dir}")


if __name__ == "__main__":
    main()
