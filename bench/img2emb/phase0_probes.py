#!/usr/bin/env python
"""Phase-0 probes over cached encoder features / crossattn_emb targets.

Runs three experiments against the artifacts produced by
``extract_features.py``:

1. **Pool linear probe.** Closed-form ridge regression from pooled encoder
   feature → target crossattn_emb, per slot. Tells us the best a
   single-vector summary of the image can do.
2. **Resampler probe.** 1-layer Perceiver-style cross-attention (512
   learnable queries) over patch tokens. Minimal trainable form of the
   proposed architecture; a negative result here is a real gating signal.
3. **Baselines.** (a) Predict dataset-mean target; (b) predict each image's
   variant-mean (trivial upper bound — already what we cached as the target,
   so this is really an on-sample sanity check that the eval loop is
   consistent).

Stratifies all metrics by slot position: prefix (0..7), mid (8..63),
tail (64..active_end). Writes per-probe JSONs and a consolidated
``summary.md`` that applies the decision rule from ``proposal.md``.

Usage:
    python bench/img2emb/phase0_probes.py
    python bench/img2emb/phase0_probes.py --encoders dinov3  --skip resampler
    python bench/img2emb/phase0_probes.py --resampler_steps 500  # fast smoke
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.io.cache import discover_cached_images  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache_dir",
        default=str(BENCH_DIR / "results" / "phase0"),
        help="Artifacts produced by extract_features.py",
    )
    p.add_argument(
        "--image_dir",
        default=None,
        help=(
            "Directory with per-image *_anima_te.safetensors (the targets). "
            "Defaults to the image_dir recorded in active_lengths.json."
        ),
    )
    p.add_argument(
        "--encoders",
        default="dinov3,siglip2",
        help="Comma-separated encoder names",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for on-demand target loading.",
    )
    p.add_argument(
        "--skip",
        default="",
        help="Comma-separated phases to skip: pool,resampler,baselines",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the resampler probe (pool probe stays on CPU).",
    )
    # Pool probe
    p.add_argument(
        "--pool_ridge",
        type=float,
        default=1e-2,
        help="Ridge regularization λ for closed-form regression.",
    )
    # Resampler probe
    p.add_argument("--resampler_dim", type=int, default=1024)
    p.add_argument("--resampler_heads", type=int, default=8)
    p.add_argument("--resampler_steps", type=int, default=2000)
    p.add_argument("--resampler_batch_size", type=int, default=16)
    p.add_argument("--resampler_lr", type=float, default=3e-4)
    p.add_argument("--resampler_seed", type=int, default=20260421)
    p.add_argument(
        "--cos_loss_weight",
        type=float,
        default=1.0,
        help="Weight on (1 - cos) relative to MSE in resampler training.",
    )
    p.add_argument(
        "--zero_pad_weight",
        type=float,
        default=0.01,
        help="Weight on padded-tail MSE (should stay near zero).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- helpers


def active_slice(active_lengths: list[int]) -> torch.Tensor:
    """(N, S) boolean mask — True where slot < that row's active length."""
    N = len(active_lengths)
    S = 512
    L = torch.tensor(active_lengths, dtype=torch.long)
    arange = torch.arange(S).unsqueeze(0).expand(N, -1)
    return arange < L.unsqueeze(1)  # (N, S)


def per_slot_metrics(
    pred: torch.Tensor,       # (N, S, D)
    target: torch.Tensor,     # (N, S, D)
    mask: torch.Tensor,       # (N, S) bool
) -> dict:
    """Per-slot R², cosine, MSE over the active rows of each slot.

    A slot is skipped in the aggregated stats if fewer than 2 rows are active.
    Stratified summaries for prefix / mid / tail / content-all are produced.
    """
    pred = pred.float()
    target = target.float()
    N, S, D = target.shape

    per_slot = []
    for s in range(S):
        m = mask[:, s]  # (N,)
        n_active = int(m.sum().item())
        if n_active < 2:
            per_slot.append({"slot": s, "n": n_active, "active": False})
            continue
        p = pred[m, s]      # (n, D)
        t = target[m, s]    # (n, D)
        # R²: 1 - sum((t - p)^2) / sum((t - t_mean)^2). Use mean-free total.
        t_mean = t.mean(dim=0, keepdim=True)
        ss_tot = float(((t - t_mean) ** 2).sum().item())
        ss_res = float(((t - p) ** 2).sum().item())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mse = float(((t - p) ** 2).mean().item())

        p_n = F.normalize(p, dim=-1, eps=1e-8)
        t_n = F.normalize(t, dim=-1, eps=1e-8)
        cos = (p_n * t_n).sum(dim=-1)  # (n,)
        per_slot.append(
            {
                "slot": s,
                "n": n_active,
                "active": True,
                "r2": r2,
                "mse": mse,
                "cos_mean": float(cos.mean().item()),
                "cos_median": float(cos.median().item()),
                "cos_p10": float(torch.quantile(cos, 0.10).item()),
            }
        )

    # Stratified summaries
    def _agg(slots, key):
        vals = [s[key] for s in slots if s["active"] and not math.isnan(s[key])]
        if not vals:
            return None
        return {
            "n_slots": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
        }

    strata = {
        "prefix_0_8": [s for s in per_slot if s["slot"] < 8],
        "mid_8_64": [s for s in per_slot if 8 <= s["slot"] < 64],
        "tail_64_256": [s for s in per_slot if 64 <= s["slot"] < 256],
        "content_all": [s for s in per_slot if 0 <= s["slot"] < 256],
    }
    stratified = {}
    for name, slots in strata.items():
        stratified[name] = {
            "r2": _agg(slots, "r2"),
            "cos_mean": _agg(slots, "cos_mean"),
            "cos_median": _agg(slots, "cos_median"),
            "cos_p10": _agg(slots, "cos_p10"),
            "mse": _agg(slots, "mse"),
        }

    return {"per_slot": per_slot, "stratified": stratified}


# --------------------------------------------------------------------------- cache loading


class _VariantMeanDataset(Dataset):
    """Load one TE file, zero-clamp the padded tail across every variant,
    return the variant-mean ``(S, D)`` slice. Parallelizable via DataLoader."""

    def __init__(self, te_paths: list[str], active_lengths: list[int]):
        self.te_paths = te_paths
        self.active_lengths = active_lengths

    def __len__(self) -> int:
        return len(self.te_paths)

    def __getitem__(self, idx: int):
        sd = load_file(self.te_paths[idx])
        variant_keys = sorted(k for k in sd.keys() if k.startswith("crossattn_emb_v"))
        if variant_keys:
            variants = [sd[k].float() for k in variant_keys]
        elif "crossattn_emb" in sd:
            variants = [sd["crossattn_emb"].float()]
        else:
            raise RuntimeError(f"No crossattn_emb in {self.te_paths[idx]}")
        stacked = torch.stack(variants, dim=0)  # (V, S, D)
        L = int(self.active_lengths[idx])
        if L < stacked.shape[1]:
            stacked[:, L:] = 0
        return idx, stacked.mean(dim=0), len(variants)


def _resolve_te_paths(image_dir: str, stems: list[str]) -> list[str]:
    """Align the TE file paths in ``image_dir`` to the cached ``stems`` order."""
    te_by_stem = {
        img.stem: img.te_path
        for img in discover_cached_images(image_dir)
        if img.te_path is not None
    }
    missing = [s for s in stems if s not in te_by_stem]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} stems listed in the cache have no TE file under "
            f"{image_dir} (first missing: {missing[0]}). Point --image_dir at "
            "the same dataset extract_features.py was run against."
        )
    return [te_by_stem[s] for s in stems]


def _load_targets_mean(
    te_paths: list[str],
    active_lengths: list[int],
    num_workers: int,
) -> tuple[torch.Tensor, int]:
    """Materialize the variant-mean targets ``(N, S, D)`` fp32 in RAM.

    Pre-allocated + filled by index via DataLoader workers — peak RAM is the
    final 2 GB tensor, not the old 8 GB per-variant stack.
    """
    ds = _VariantMeanDataset(te_paths, active_lengths)
    loader = DataLoader(
        ds,
        batch_size=16,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )
    N = len(ds)
    out: torch.Tensor | None = None
    V_seen: int | None = None
    for idx, mean_var, n_vs in tqdm(loader, desc="targets_mean"):
        batch_V = int(n_vs[0].item())
        if V_seen is None:
            V_seen = batch_V
            S, D = int(mean_var.shape[1]), int(mean_var.shape[2])
            out = torch.empty((N, S, D), dtype=torch.float32)
        elif batch_V != V_seen:
            raise RuntimeError(
                f"Variant count differs across images: {V_seen} vs {batch_V}"
            )
        out[idx] = mean_var.float()
    assert out is not None and V_seen is not None
    return out, V_seen


def load_cache(cache_dir: Path, image_dir: str, encoder: str, num_workers: int):
    """Return dict with pooled / tokens / targets_mean / active_lengths / split /
    te_paths.

    Targets are no longer cached in one big safetensors; variant-mean is
    materialized on demand from the per-image ``*_anima_te.safetensors`` in
    ``image_dir``, and ``te_paths`` is passed through for the resampler to
    stream per-sample variants during training.
    """
    stems = json.loads((cache_dir / "stems.json").read_text())
    split = json.loads((cache_dir / "split.json").read_text())
    act = json.loads((cache_dir / "active_lengths.json").read_text())

    te_paths = _resolve_te_paths(image_dir, stems)
    targets_mean, V = _load_targets_mean(te_paths, act["active_lengths"], num_workers)
    V = int(act.get("num_variants", V))

    pooled = load_file(str(cache_dir / "features" / f"{encoder}_pooled.safetensors"))[
        "pooled"
    ]
    tokens = load_file(str(cache_dir / "features" / f"{encoder}_tokens.safetensors"))[
        "tokens"
    ]
    return {
        "stems": stems,
        "split": split,
        "active_lengths": act["active_lengths"],
        "num_variants": V,
        "te_paths": te_paths,           # list[str] aligned to stems order
        "targets_mean": targets_mean,   # (N, S, D) fp32
        "pooled": pooled,               # (N, D_enc) fp32
        "tokens": tokens,               # (N, T, D_enc) bf16
    }


# --------------------------------------------------------------------------- pool probe


def run_pool_probe(cache: dict, ridge: float, out_path: Path, encoder: str):
    """Closed-form ridge regression from pooled → crossattn_emb, per slot.

    Solve per-slot in vectorized form:
        W = (XᵀX + λI)⁻¹ Xᵀ Y    with Y = targets flattened over (S*D)

    XᵀX is (D_enc, D_enc) — always tiny. We then multiply by the flattened
    target to get per-slot weights, without ever materializing a
    (D_enc, S*D) weight matrix in one shot — chunk over slot blocks.
    """
    logger.info(f"[pool/{encoder}] ridge={ridge}")
    train_idx = cache["split"]["train_idx"]
    eval_idx = cache["split"]["eval_idx"]
    X_train = cache["pooled"][train_idx].float()        # (N_tr, D)
    X_eval = cache["pooled"][eval_idx].float()          # (N_ev, D)
    # OLS with identical X rows collapses to predicting the mean target, so
    # the variant-averaged tensor is sufficient for this probe.
    Y_train = cache["targets_mean"][train_idx].float()  # (N_tr, S, D_y)
    Y_eval = cache["targets_mean"][eval_idx].float()    # (N_ev, S, D_y)
    act_eval = [cache["active_lengths"][i] for i in eval_idx]
    mask_eval = active_slice(act_eval)

    N_tr, D = X_train.shape
    S, D_y = Y_train.shape[1], Y_train.shape[2]

    XtX = X_train.T @ X_train + ridge * torch.eye(D)      # (D, D)
    XtX_inv = torch.linalg.inv(XtX)
    # A = (XᵀX + λI)⁻¹ Xᵀ  — shape (D, N_tr)
    A = XtX_inv @ X_train.T

    # Predict per-slot in chunks to bound memory at (D × chunk*D_y).
    chunk = 32  # slots
    pred_eval = torch.empty_like(Y_eval)
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        Y_chunk = Y_train[:, s0:s1, :].reshape(N_tr, -1)   # (N_tr, chunk*D_y)
        W_chunk = A @ Y_chunk                               # (D, chunk*D_y)
        pred_chunk = X_eval @ W_chunk                       # (N_ev, chunk*D_y)
        pred_eval[:, s0:s1, :] = pred_chunk.reshape(X_eval.shape[0], s1 - s0, D_y)

    metrics = per_slot_metrics(pred_eval, Y_eval, mask_eval)
    payload = {
        "encoder": encoder,
        "probe": "pool_linear",
        "ridge": ridge,
        "n_train": N_tr,
        "n_eval": X_eval.shape[0],
        "d_enc": D,
        "stratified": metrics["stratified"],
        # per-slot JSON is large — write under separate key to make it easy
        # to drop if we want to keep result files small.
        "per_slot": metrics["per_slot"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  → {out_path}")
    _log_strata(metrics["stratified"], f"pool/{encoder}")
    return payload


# --------------------------------------------------------------------------- resampler probe


class SingleLayerResampler(nn.Module):
    """512 learned queries attend to encoder patch tokens (one cross-attn + FFN)."""

    def __init__(self, d_enc: int, d_model: int = 1024, n_heads: int = 8, n_slots: int = 512, d_out: int = 1024):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.15)
        self.kv_proj = nn.Linear(d_enc, d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.q_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, bias=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.out = nn.Linear(d_model, d_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D_enc). Caller may pass bf16 from disk; cast to
        # the Linear layers' dtype (fp32) for numerically stable training on
        # small batches. bf16 end-to-end is fine at phase 1 scale; here the
        # cost is negligible.
        B = tokens.shape[0]
        tokens = tokens.to(dtype=self.kv_proj.weight.dtype)
        kv = self.kv_norm(self.kv_proj(tokens))
        q = self.q_norm(self.queries.expand(B, -1, -1))
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = q + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return self.out(x)


def _resampler_loss(
    pred: torch.Tensor,      # (B, S, D)
    target: torch.Tensor,    # (B, S, D)
    mask: torch.Tensor,      # (B, S) bool, True in active region
    cos_w: float,
    zero_w: float,
):
    pred_f = pred.float()
    target_f = target.float()
    active = mask.unsqueeze(-1).float()        # (B, S, 1)
    inactive = (1.0 - active)

    # MSE on active region (normalized by active element count)
    diff = (pred_f - target_f) ** 2
    denom_act = active.sum().clamp_min(1.0) * pred_f.shape[-1]
    mse_active = (diff * active).sum() / denom_act

    # Cosine loss on active slots
    if cos_w > 0:
        pred_n = F.normalize(pred_f, dim=-1, eps=1e-8)
        target_n = F.normalize(target_f, dim=-1, eps=1e-8)
        cos = (pred_n * target_n).sum(dim=-1)  # (B, S)
        cos_loss_num = ((1.0 - cos) * mask.float()).sum()
        cos_loss = cos_loss_num / mask.float().sum().clamp_min(1.0)
    else:
        cos_loss = torch.tensor(0.0, device=pred.device)

    # Zero-pad penalty
    if zero_w > 0:
        pad_sq = (pred_f ** 2) * inactive
        denom_inact = inactive.sum().clamp_min(1.0) * pred_f.shape[-1]
        pad_loss = pad_sq.sum() / denom_inact
    else:
        pad_loss = torch.tensor(0.0, device=pred.device)

    total = mse_active + cos_w * cos_loss + zero_w * pad_loss
    return total, {
        "mse_active": float(mse_active.detach().item()),
        "cos_loss": float(cos_loss.detach().item()),
        "pad_loss": float(pad_loss.detach().item()),
    }


class _ResamplerTrainDataset(Dataset):
    """Streams per-sample training targets on demand.

    Yields ``(full_idx, target[S, D] bf16, L)`` where ``full_idx`` is the index
    into the original (N-long) stems list and ``target`` is one uniformly-
    sampled variant (of V) with its padded tail zero-clamped. The caller
    looks up tokens via ``tokens_all[full_idx]`` in the main process.

    DataLoader re-seeds ``torch.initial_seed()`` per worker per epoch, so
    ``torch.randint`` below gives independent variant draws across workers.
    """

    def __init__(
        self,
        train_indices: list[int],
        te_paths: list[str],
        active_lengths: list[int],
        num_variants: int,
    ):
        self.train_indices = train_indices
        self.te_paths = te_paths
        self.active_lengths = active_lengths
        self.V = num_variants

    def __len__(self) -> int:
        return len(self.train_indices)

    def __getitem__(self, pos: int):
        full_idx = self.train_indices[pos]
        sd = load_file(self.te_paths[full_idx])
        if self.V > 1:
            v = int(torch.randint(0, self.V, (1,)).item())
            target = sd[f"crossattn_emb_v{v}"].float()
        else:
            target = sd["crossattn_emb"].float()
        L = int(self.active_lengths[full_idx])
        if L < target.shape[0]:
            target = target.clone()
            target[L:] = 0
        return full_idx, target.to(torch.bfloat16), L


def run_resampler_probe(
    cache: dict,
    args,
    out_path: Path,
    encoder: str,
    device: torch.device,
):
    logger.info(f"[resampler/{encoder}] steps={args.resampler_steps}")

    torch.manual_seed(args.resampler_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.resampler_seed)

    train_idx = cache["split"]["train_idx"]
    eval_idx = cache["split"]["eval_idx"]

    tokens_all = cache["tokens"]            # (N, T, D_enc) bf16 (stays on CPU)
    targets_mean = cache["targets_mean"]    # (N, S, D) fp32 — eval only
    V = int(cache["num_variants"])

    d_enc = tokens_all.shape[-1]
    d_y = targets_mean.shape[-1]
    S = targets_mean.shape[-2]

    model = SingleLayerResampler(
        d_enc=d_enc,
        d_model=args.resampler_dim,
        n_heads=args.resampler_heads,
        n_slots=S,
        d_out=d_y,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.resampler_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.resampler_steps, eta_min=args.resampler_lr * 0.05
    )

    train_ds = _ResamplerTrainDataset(
        train_indices=train_idx,
        te_paths=cache["te_paths"],
        active_lengths=cache["active_lengths"],
        num_variants=V,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.resampler_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    # Per-sample variant sampling via the Dataset: each image draws a random
    # variant in [0, V) on every __getitem__ call, so across epochs we see
    # ~V × len(train_idx) distinct (image, variant) pairs (the cos-0.99
    # spread across variants acts as implicit query augmentation).

    losses_log = []
    t0 = time.time()
    pbar = tqdm(total=args.resampler_steps, desc=f"train/{encoder}", leave=False)
    step = 0
    done = False
    while not done:
        for full_idx, tgt_b, L_b in loader:
            if step >= args.resampler_steps:
                done = True
                break
            tok_b = tokens_all[full_idx].to(device=device, dtype=torch.bfloat16)
            tgt_b = tgt_b.to(device=device, dtype=torch.float32)
            mask_b = (
                torch.arange(S).unsqueeze(0) < L_b.unsqueeze(1)
            ).to(device)

            pred = model(tok_b)
            loss, comps = _resampler_loss(
                pred, tgt_b, mask_b,
                cos_w=args.cos_loss_weight, zero_w=args.zero_pad_weight,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % 50 == 0 or step == args.resampler_steps - 1:
                losses_log.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().item()),
                        "mse": comps["mse_active"],
                        "cos": comps["cos_loss"],
                        "pad": comps["pad_loss"],
                        "lr": float(opt.param_groups[0]["lr"]),
                    }
                )
                pbar.set_postfix(
                    loss=f"{float(loss.detach().item()):.4f}",
                    mse=f"{comps['mse_active']:.4f}",
                    cos=f"{comps['cos_loss']:.4f}",
                )
            step += 1
            pbar.update(1)
    pbar.close()
    train_time = time.time() - t0

    # --- eval against the variant-mean target (stable, matches what the
    # model will serve downstream).
    model.eval()
    pred_eval_chunks = []
    eval_batch = 8
    with torch.no_grad():
        for i in range(0, len(eval_idx), eval_batch):
            ids = eval_idx[i : i + eval_batch]
            tok_b = tokens_all[ids].to(device=device, dtype=torch.bfloat16)
            pred_eval_chunks.append(model(tok_b).detach().float().cpu())
    pred_eval = torch.cat(pred_eval_chunks, dim=0)
    tgt_eval = targets_mean[eval_idx].float()
    mask_eval = active_slice([cache["active_lengths"][i] for i in eval_idx])

    metrics = per_slot_metrics(pred_eval, tgt_eval, mask_eval)
    payload = {
        "encoder": encoder,
        "probe": "cross_attn_resampler_1layer",
        "d_enc": d_enc,
        "d_model": args.resampler_dim,
        "n_heads": args.resampler_heads,
        "n_steps": args.resampler_steps,
        "lr": args.resampler_lr,
        "batch_size": args.resampler_batch_size,
        "num_variants": V,
        "variant_sampling": "per_sample_uniform",
        "eval_target": "variant_mean",
        "cos_loss_weight": args.cos_loss_weight,
        "zero_pad_weight": args.zero_pad_weight,
        "train_time_sec": train_time,
        "train_log": losses_log,
        "stratified": metrics["stratified"],
        "per_slot": metrics["per_slot"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  → {out_path}")
    _log_strata(metrics["stratified"], f"resampler/{encoder}")
    return payload


# --------------------------------------------------------------------------- baselines


def run_baselines(cache: dict, out_dir: Path):
    """Dataset-mean baseline + per-image-variant-mean sanity check.

    The variant-mean "baseline" isn't a true upper bound here because our
    target IS the variant mean — it just confirms that predicting the target
    returns metrics close to perfect (R²≈1, cos≈1). If those aren't ~1 there's
    a bug in ``per_slot_metrics`` or the mask.
    """
    train_idx = cache["split"]["train_idx"]
    eval_idx = cache["split"]["eval_idx"]
    Y_train = cache["targets_mean"][train_idx].float()
    Y_eval = cache["targets_mean"][eval_idx].float()
    mask_eval = active_slice([cache["active_lengths"][i] for i in eval_idx])

    # Dataset-mean
    mean_vec = Y_train.mean(dim=0, keepdim=True)  # (1, S, D)
    pred = mean_vec.expand_as(Y_eval).contiguous()
    m_mean = per_slot_metrics(pred, Y_eval, mask_eval)
    (out_dir / "baseline_mean.json").write_text(
        json.dumps(
            {
                "probe": "dataset_mean",
                "n_train": Y_train.shape[0],
                "n_eval": Y_eval.shape[0],
                "stratified": m_mean["stratified"],
                "per_slot": m_mean["per_slot"],
            },
            indent=2,
        )
    )
    _log_strata(m_mean["stratified"], "baseline/mean")

    # Self-prediction sanity check
    m_self = per_slot_metrics(Y_eval, Y_eval, mask_eval)
    (out_dir / "baseline_self.json").write_text(
        json.dumps({"probe": "self_identity", "stratified": m_self["stratified"]},
                   indent=2)
    )
    _log_strata(m_self["stratified"], "baseline/self")

    return m_mean, m_self


# --------------------------------------------------------------------------- logging + summary


def _log_strata(strat: dict, tag: str):
    for name in ("prefix_0_8", "mid_8_64", "tail_64_256", "content_all"):
        s = strat.get(name)
        if s is None:
            continue
        r2 = s.get("r2")
        cm = s.get("cos_median")
        cp = s.get("cos_p10")
        mse = s.get("mse")
        r2s = f"{r2['mean']:.3f}" if r2 else "-"
        cms = f"{cm['median']:.3f}" if cm else "-"
        cps = f"{cp['median']:.3f}" if cp else "-"
        mses = f"{mse['mean']:.4f}" if mse else "-"
        logger.info(
            f"  [{tag}/{name}] r2_mean={r2s} cos_med={cms} cos_p10_med={cps} mse_mean={mses}"
        )


def _row(strat: dict, key: str):
    """Flatten a stratified dict into one markdown row tuple."""
    def fmt(g, k, digits=3):
        if not g or g.get(k) is None:
            return "-"
        return f"{g[k]['mean']:.{digits}f}"
    return (
        fmt(strat.get("prefix_0_8"), "cos_median"),
        fmt(strat.get("mid_8_64"), "cos_median"),
        fmt(strat.get("tail_64_256"), "cos_median"),
        fmt(strat.get("prefix_0_8"), "r2"),
        fmt(strat.get("content_all"), "r2"),
        fmt(strat.get("content_all"), "mse", digits=4),
    )


def write_summary(out_dir: Path, results: dict):
    """Consolidate all probe results into a decision-rule-applied summary."""
    lines = []
    a = lines.append
    a("# Phase 0 — encoder fit bench")
    a("")

    # Table
    a("| encoder | probe | cos_med prefix | cos_med mid | cos_med tail | R² prefix | R² content | MSE content |")
    a("|---|---|---|---|---|---|---|---|")
    for row_key in sorted(results.keys()):
        payload = results[row_key]
        strat = payload["stratified"]
        row = _row(strat, row_key)
        a(f"| {payload.get('encoder', '-')} | {payload.get('probe', '-')} | "
          + " | ".join(row) + " |")
    a("")

    # Decision rule
    a("## Decision rule (from proposal.md)")
    a("")
    a("- Proceed to phase 1 if: cos_median on prefix_0_8 > 0.6 AND cos_median on content_all > 0.4.")
    a("- Swap encoder if all probes score cos_median < 0.3 everywhere.")
    a("- Middle band: run phase 1 anyway and re-measure — 1-layer linear probes underestimate what a 4-layer resampler can do.")
    a("")

    # Recommendation per encoder
    a("## Per-encoder verdict")
    a("")
    encoders = sorted(
        {
            p["encoder"]
            for p in results.values()
            if "encoder" in p and p["encoder"] not in ("-", None)
        }
    )
    for enc in encoders:
        a(f"### {enc}")
        by_probe = {p["probe"]: p for p in results.values() if p.get("encoder") == enc}
        best = None
        for probe_name in ("cross_attn_resampler_1layer", "pool_linear"):
            if probe_name in by_probe:
                best = by_probe[probe_name]
                break
        if best is None:
            a("- no probe ran")
            continue
        strat = best["stratified"]
        prefix_cos = strat.get("prefix_0_8", {}).get("cos_median", {})
        content_cos = strat.get("content_all", {}).get("cos_median", {})
        p_cos = prefix_cos.get("mean") if prefix_cos else None
        c_cos = content_cos.get("mean") if content_cos else None
        if p_cos is None or c_cos is None:
            a("- no content slots active in eval set; inconclusive")
            continue
        if p_cos > 0.6 and c_cos > 0.4:
            verdict = "**PROCEED** to phase 1"
        elif p_cos < 0.3 and c_cos < 0.3:
            verdict = "**SWAP** encoder"
        else:
            verdict = "**MARGINAL** — phase 1 full resampler before deciding"
        a(f"- best probe: `{best['probe']}`")
        a(f"- prefix_0_8 cos_med (mean over slots) = {p_cos:.3f}")
        a(f"- content_all cos_med (mean over slots) = {c_cos:.3f}")
        a(f"- verdict: {verdict}")
        a("")

    a("## Baselines")
    a("")
    for name in ("baseline_mean", "baseline_self"):
        p = results.get(name)
        if p is None:
            continue
        strat = p["stratified"]
        cos_med = strat.get("content_all", {}).get("cos_median", {}).get("mean")
        r2 = strat.get("content_all", {}).get("r2", {}).get("mean")
        cos_s = f"{cos_med:.3f}" if cos_med is not None else "-"
        r2_s = f"{r2:.3f}" if r2 is not None else "-"
        a(f"- **{name}**: content_all cos_med={cos_s}  r2={r2_s}")
    a("")

    (out_dir / "summary.md").write_text("\n".join(lines))
    logger.info(f"  → {out_dir / 'summary.md'}")


# --------------------------------------------------------------------------- main


def _resolve_image_dir(cache_dir: Path, override: str | None) -> str:
    """CLI override wins; otherwise pull from active_lengths.json; fall back
    to ``post_image_dataset`` as a last resort."""
    if override:
        return override
    act_path = cache_dir / "active_lengths.json"
    if act_path.exists():
        recorded = json.loads(act_path.read_text()).get("image_dir")
        if recorded:
            return recorded
    return "post_image_dataset"


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    image_dir = _resolve_image_dir(cache_dir, args.image_dir)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    device = torch.device(args.device)

    results: dict[str, dict] = {}

    for enc in encoders:
        logger.info(f"======== {enc} ========")
        try:
            cache = load_cache(cache_dir, image_dir, enc, args.num_workers)
        except FileNotFoundError as e:
            logger.warning(f"  missing cache for {enc}, skipping: {e}")
            continue

        if "pool" not in skip:
            out = cache_dir / f"{enc}_pool_probe.json"
            res = run_pool_probe(cache, args.pool_ridge, out, enc)
            results[f"{enc}/pool"] = res

        if "resampler" not in skip:
            out = cache_dir / f"{enc}_resampler_probe.json"
            res = run_resampler_probe(cache, args, out, enc, device)
            results[f"{enc}/resampler"] = res

        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "baselines" not in skip:
        logger.info("======== baselines ========")
        # Baselines only depend on the target, not the encoder. Pick any cache.
        any_enc = encoders[0] if encoders else None
        if any_enc is not None:
            try:
                cache = load_cache(cache_dir, image_dir, any_enc, args.num_workers)
                m_mean, m_self = run_baselines(cache, cache_dir)
                results["baseline_mean"] = {
                    "encoder": "-",
                    "probe": "dataset_mean",
                    "stratified": m_mean["stratified"],
                }
                results["baseline_self"] = {
                    "encoder": "-",
                    "probe": "self_identity",
                    "stratified": m_self["stratified"],
                }
                del cache
            except FileNotFoundError as e:
                logger.warning(f"baselines skipped: {e}")

    write_summary(cache_dir, results)
    logger.info("Done.")


if __name__ == "__main__":
    main()
