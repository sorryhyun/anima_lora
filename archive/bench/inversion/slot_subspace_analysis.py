#!/usr/bin/env python
"""Per-slot subspace analysis of cached T5 crossattn embeddings.

Goal: measure the structural properties of T5 caption embeddings that a
"natural-looking" inversion should match, so we can constrain future inversion
runs to (a) an image-appropriate content length and (b) a per-slot low-rank
subspace. Currently the invert script uses a flat ``--active_length`` default
of 128 and an unconstrained 1024-D vector per slot — this bench tells us how
wasteful that is.

The cached TE files (``post_image_dataset/*_anima_te.safetensors``, key
``crossattn_emb_v0``) have bit-zero padding (see ``library/anima/strategy.py``
line ~390: ``crossattn_emb[~t5_attn_mask] = 0``), so per-image content length
is recoverable directly from the tensor — no separate mask file needed.

Phases:
  A. Content-length distribution across the dataset.
  B. Per-slot SVD on content-only rows (drop images where slot is in padding).
     Records singular spectrum, effective rank at 80/90/95/99% variance,
     sample count per slot.
  C. Pooled SVD over all content rows (position-invariant subspace test).
     Measures variance-explained of slot-s rows under the pooled basis vs
     slot-s's own basis. If pooled ≈ per-slot, subspace is position-invariant.
  D. Summary: recommended ``--active_length`` (e.g. 95th percentile of caption
     length) and recommended low-rank K for the per-slot basis.

Run:
    python bench/inversion/slot_subspace_analysis.py
    python bench/inversion/slot_subspace_analysis.py --max_images 500 --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from library.io.cache import discover_cached_images  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image_dir", type=str, default="post_image_dataset")
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Subsample to first N TE caches (default: all)",
    )
    p.add_argument(
        "--zero_eps",
        type=float,
        default=1e-6,
        help="A row is padding if |row|_inf <= this. Cache stores bit-zero "
        "padding so default tolerates bf16/fp32 roundtrip.",
    )
    p.add_argument(
        "--min_samples_per_slot",
        type=int,
        default=16,
        help="Skip per-slot SVD if fewer images have content at this slot",
    )
    p.add_argument(
        "--pooled_q",
        type=int,
        default=64,
        help="Top-q right singular vectors for the pooled-content SVD",
    )
    p.add_argument(
        "--top_k_report",
        type=int,
        default=16,
        help="Number of leading singular values to record per slot",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(RESULTS_DIR / "slot_subspace"),
    )
    return p.parse_args()


def load_te_stack(image_dir, max_images, zero_eps, device):
    """Load all cached T5 crossattn embeddings into a single (N, S, D) fp32 tensor.

    Returns:
        emb: (N, S, D) float32 on ``device``
        seqlens: (N,) int64 — last-non-zero-row + 1 per image
        stems: list[str] — image stem per row (index-aligned)
    """
    images = discover_cached_images(image_dir)
    images = [img for img in images if img.te_path is not None]
    if not images:
        raise SystemExit(f"No TE caches found in {image_dir}")
    if max_images is not None:
        images = images[:max_images]

    logger.info(f"Loading {len(images)} TE caches from {image_dir}")

    # Peek one file to learn (S, D)
    sd0 = load_file(images[0].te_path)
    key = "crossattn_emb_v0" if "crossattn_emb_v0" in sd0 else "crossattn_emb"
    t0 = sd0[key]
    if t0.ndim != 2:
        raise RuntimeError(f"Unexpected shape {tuple(t0.shape)} at {images[0].te_path}")
    S, D = t0.shape
    logger.info(f"Embedding shape per image: S={S}, D={D}")

    emb = torch.empty((len(images), S, D), dtype=torch.float32, device=device)
    seqlens = torch.empty(len(images), dtype=torch.int64, device=device)
    stems = []

    for i, img in enumerate(images):
        sd = load_file(img.te_path)
        k = "crossattn_emb_v0" if "crossattn_emb_v0" in sd else "crossattn_emb"
        v = sd[k].to(torch.float32)
        if v.shape != (S, D):
            raise RuntimeError(
                f"Shape mismatch at {img.te_path}: {tuple(v.shape)} vs ({S},{D})"
            )
        emb[i] = v.to(device)
        # Content length = 1 + last row with any |x| > eps
        nonzero_rows = (v.abs().amax(dim=-1) > zero_eps)  # (S,) bool
        if nonzero_rows.any():
            seqlens[i] = int(nonzero_rows.nonzero(as_tuple=False)[-1].item()) + 1
        else:
            seqlens[i] = 0
        stems.append(img.stem)

        if (i + 1) % 200 == 0:
            logger.info(f"  loaded {i + 1}/{len(images)}")

    return emb, seqlens, stems


def phase_a_seqlen_dist(seqlens):
    """Content length distribution. Returns dict of summary stats + histogram."""
    sl = seqlens.cpu().numpy()
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    summary = {
        "n_images": int(sl.size),
        "min": int(sl.min()),
        "max": int(sl.max()),
        "mean": float(sl.mean()),
        "std": float(sl.std()),
        "percentiles": {str(p): float(np.percentile(sl, p)) for p in pcts},
    }
    # Histogram in 16-slot bins up to 512
    bin_edges = list(range(0, 513, 16))
    hist, _ = np.histogram(sl, bins=bin_edges)
    summary["histogram"] = {
        "bin_edges": bin_edges,
        "counts": hist.tolist(),
    }
    logger.info(
        f"  seqlen: mean={summary['mean']:.1f} "
        f"p50={summary['percentiles']['50']:.0f} "
        f"p95={summary['percentiles']['95']:.0f} "
        f"max={summary['max']}"
    )
    return summary


def _effective_rank(singular_values_sq, thresholds=(0.80, 0.90, 0.95, 0.99)):
    """Smallest k such that cum-variance-explained >= threshold. Returns dict."""
    total = singular_values_sq.sum()
    if total <= 0:
        return {f"k_{int(t * 100)}": 0 for t in thresholds}
    cum = np.cumsum(singular_values_sq) / total
    out = {}
    for t in thresholds:
        k = int(np.searchsorted(cum, t) + 1)
        out[f"k_{int(t * 100)}"] = k
    return out


def phase_b_per_slot_svd(emb, seqlens, args):
    """SVD on content-only rows of each slot.

    For slot s, includes image i iff seqlens[i] > s (i.e. slot s holds content).

    Returns dict keyed by slot index with: n_samples, sv_top_k, total_energy,
    effective_rank_{80,90,95,99}, mean_norm.
    """
    N, S, D = emb.shape
    out = {}

    # precompute per-slot mask via comparing seqlens to a (S,) range
    slot_idx = torch.arange(S, device=emb.device)  # (S,)
    # mask[i, s] = True iff seqlens[i] > s
    mask_all = seqlens.unsqueeze(1) > slot_idx.unsqueeze(0)  # (N, S)

    for s in range(S):
        mask = mask_all[:, s]  # (N,)
        n = int(mask.sum().item())
        if n < args.min_samples_per_slot:
            out[s] = {
                "n_samples": n,
                "skipped": True,
            }
            continue

        Xs = emb[mask, s, :]  # (n, D)
        mean_norm = float(Xs.norm(dim=-1).mean().item())

        # Non-centered SVD: we want total-energy subspace (direction-from-origin),
        # not variance-from-mean, because an inversion should match the raw
        # direction the model expects at that slot, not the residual around a mean.
        # For a sanity cross-check we also record the mean vector's norm above.
        try:
            Us, Ss_vals, _Vh = torch.linalg.svd(Xs, full_matrices=False)
        except RuntimeError as e:
            logger.warning(f"  SVD failed at slot {s}: {e}; skipping")
            out[s] = {"n_samples": n, "skipped": True, "error": str(e)}
            continue

        sv = Ss_vals.detach().float().cpu().numpy()
        sv_sq = sv * sv
        er = _effective_rank(sv_sq)
        out[s] = {
            "n_samples": n,
            "mean_row_norm": mean_norm,
            "total_energy": float(sv_sq.sum()),
            "sv_top_k": sv[: args.top_k_report].tolist(),
            **er,
        }

    # Log a summary of how effective-rank varies with slot index
    logged = [(s, out[s]["k_95"]) for s in range(S) if not out[s].get("skipped")]
    if logged:
        slots, k95 = zip(*logged)
        logger.info(
            f"  per-slot k@95%: active_slots={len(slots)}  "
            f"mean_k95={np.mean(k95):.1f}  p50={np.median(k95):.0f}  "
            f"p95={np.percentile(k95, 95):.0f}  max={max(k95)}"
        )
    return out


def phase_c_pooled_subspace(emb, seqlens, per_slot_results, args):
    """Pooled SVD over all content rows, then measure how well the pooled
    top-q basis explains each slot's variance vs that slot's own basis.

    Returns dict with pooled spectrum, and per-slot variance-explained under
    the pooled basis at q ∈ {8, 16, 32, 64}.
    """
    N, S, D = emb.shape
    slot_idx = torch.arange(S, device=emb.device)
    mask_all = seqlens.unsqueeze(1) > slot_idx.unsqueeze(0)  # (N, S)

    # Assemble (M, D) of all content rows across all images and slots
    content_rows = emb[mask_all]  # (M, D)
    M = content_rows.shape[0]
    logger.info(f"  pooled content rows: M={M}")

    # Top-q right singular vectors via randomized SVD
    q = min(args.pooled_q, D, max(1, M - 1))
    try:
        _U, Spool, Vpool = torch.svd_lowrank(content_rows, q=q, niter=3)
    except RuntimeError as e:
        logger.warning(f"  pooled SVD failed: {e}")
        return None

    Spool = Spool.detach().float().cpu().numpy()  # (q,)
    Vpool_cpu = Vpool.detach().float().cpu()  # (D, q) — right singular vectors

    total_pooled_energy = float(content_rows.float().pow(2).sum().item())

    pooled_cum = np.cumsum(Spool * Spool) / total_pooled_energy
    pooled_k95 = int(np.searchsorted(pooled_cum, 0.95) + 1) if pooled_cum[-1] >= 0.95 else q

    # Per-slot: variance explained under pooled basis at several Ks
    K_probes = [k for k in (8, 16, 32, 64) if k <= q]
    per_slot_pooled = {}
    for s in range(S):
        info = per_slot_results.get(s, {})
        if info.get("skipped"):
            continue
        mask = mask_all[:, s]
        Xs = emb[mask, s, :].float().cpu()  # (n, D)
        if Xs.shape[0] == 0:
            continue
        total_s = float(Xs.pow(2).sum().item())
        explained = {}
        # Project once onto full q, then reuse cumulative squared coefs for each K
        coefs = Xs @ Vpool_cpu  # (n, q)
        coef_energy = coefs.pow(2).sum(dim=0).numpy()  # (q,)
        cum = np.cumsum(coef_energy)
        for K in K_probes:
            explained[f"pooled_ve_K{K}"] = float(cum[K - 1] / total_s) if total_s > 0 else 0.0
        explained["pooled_ve_Kq"] = float(cum[-1] / total_s) if total_s > 0 else 0.0
        per_slot_pooled[s] = explained

    # Aggregate comparison: per-slot self-basis achieves k95_self at 95% VE (by construction).
    # Under pooled basis, how much VE do we get at the same K?
    summary_table = []
    for s, info in per_slot_results.items():
        if info.get("skipped"):
            continue
        k95_self = info["k_95"]
        if s not in per_slot_pooled:
            continue
        # Compute VE under pooled basis at K = k95_self (cap at q)
        mask = mask_all[:, s]
        Xs = emb[mask, s, :].float().cpu()
        total_s = float(Xs.pow(2).sum().item())
        if total_s <= 0:
            continue
        coefs = Xs @ Vpool_cpu
        K_eq = min(k95_self, q)
        ve_pooled_at_k95_self = (
            float(coefs.pow(2).sum(dim=0).numpy()[:K_eq].sum() / total_s)
            if K_eq > 0
            else 0.0
        )
        summary_table.append(
            {
                "slot": s,
                "k95_self": k95_self,
                "ve_pooled_at_same_K": ve_pooled_at_k95_self,
            }
        )

    if summary_table:
        ve_pooled_at_same_K = np.array([r["ve_pooled_at_same_K"] for r in summary_table])
        logger.info(
            f"  pooled-vs-self at K=k95_self: mean_ve={ve_pooled_at_same_K.mean():.3f} "
            f"(1.0 = position-invariant, <<1.0 = position-specific subspace)"
        )
        logger.info(
            f"  pooled k@95% over the whole pool: {pooled_k95} / q={q}"
        )

    return {
        "M": int(M),
        "q": int(q),
        "pooled_singular_values": Spool.tolist(),
        "pooled_total_energy": total_pooled_energy,
        "pooled_k95": pooled_k95,
        "per_slot_variance_explained_under_pooled_basis": per_slot_pooled,
        "pooled_vs_self_at_k95_self": summary_table,
    }


def phase_d_recommendations(phase_a, phase_b, phase_c):
    """Print actionable recommendations derived from A+B+C."""
    reco = {}
    p95 = int(round(phase_a["percentiles"]["95"]))
    p99 = int(round(phase_a["percentiles"]["99"]))
    reco["recommended_active_length_p95"] = p95
    reco["recommended_active_length_p99"] = p99

    active_slots = [s for s, v in phase_b.items() if not v.get("skipped")]
    k95_vals = [phase_b[s]["k_95"] for s in active_slots]
    k99_vals = [phase_b[s]["k_99"] for s in active_slots]
    if k95_vals:
        reco["per_slot_k95_median"] = float(np.median(k95_vals))
        reco["per_slot_k95_p95"] = float(np.percentile(k95_vals, 95))
        reco["per_slot_k99_median"] = float(np.median(k99_vals))
        reco["per_slot_k99_p95"] = float(np.percentile(k99_vals, 95))

    if phase_c is not None:
        reco["pooled_k95"] = phase_c["pooled_k95"]
        tbl = phase_c["pooled_vs_self_at_k95_self"]
        if tbl:
            reco["pooled_ve_at_self_k95_median"] = float(
                np.median([r["ve_pooled_at_same_K"] for r in tbl])
            )

    logger.info("")
    logger.info("=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    logger.info(f"  --active_length p95 = {p95}   (p99 = {p99})")
    if k95_vals:
        logger.info(
            f"  per-slot K @ 95% variance: median={reco['per_slot_k95_median']:.0f} "
            f"p95={reco['per_slot_k95_p95']:.0f}"
        )
        logger.info(
            f"  per-slot K @ 99% variance: median={reco['per_slot_k99_median']:.0f} "
            f"p95={reco['per_slot_k99_p95']:.0f}"
        )
    if phase_c is not None and "pooled_ve_at_self_k95_median" in reco:
        logger.info(
            f"  pooled basis at K=k95_self explains median "
            f"{reco['pooled_ve_at_self_k95_median']:.1%} of slot energy "
            f"(→ {'position-invariant' if reco['pooled_ve_at_self_k95_median'] > 0.9 else 'position-specific'} subspace)"
        )
    logger.info("=" * 60)
    return reco


def save_results(out_dir, phase_a, phase_b, phase_c, reco, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON
    payload = {
        "args": {
            "image_dir": args.image_dir,
            "max_images": args.max_images,
            "zero_eps": args.zero_eps,
            "min_samples_per_slot": args.min_samples_per_slot,
            "pooled_q": args.pooled_q,
            "top_k_report": args.top_k_report,
        },
        "phase_a_seqlen": phase_a,
        "phase_b_per_slot": phase_b,
        "phase_c_pooled": phase_c,
        "recommendations": reco,
    }
    json_path = out_dir / "slot_subspace_analysis.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote full JSON → {json_path}")

    # Compact CSV: one row per slot for easy plotting
    csv_path = out_dir / "per_slot.csv"
    with open(csv_path, "w") as f:
        f.write("slot,n_samples,mean_row_norm,total_energy,k80,k90,k95,k99,sv1,sv2,sv3,sv4\n")
        for s in sorted(phase_b.keys()):
            v = phase_b[s]
            if v.get("skipped"):
                f.write(f"{s},{v['n_samples']},,,,,,,,,,\n")
                continue
            svs = v["sv_top_k"] + [0.0] * 4
            f.write(
                f"{s},{v['n_samples']},{v['mean_row_norm']:.6f},"
                f"{v['total_energy']:.6f},"
                f"{v['k_80']},{v['k_90']},{v['k_95']},{v['k_99']},"
                f"{svs[0]:.6f},{svs[1]:.6f},{svs[2]:.6f},{svs[3]:.6f}\n"
            )
    logger.info(f"Wrote per-slot CSV → {csv_path}")


def main():
    args = parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    emb, seqlens, _stems = load_te_stack(
        args.image_dir, args.max_images, args.zero_eps, device
    )

    logger.info("")
    logger.info("Phase A — content-length distribution")
    phase_a = phase_a_seqlen_dist(seqlens)

    logger.info("")
    logger.info("Phase B — per-slot SVD on content-only rows")
    phase_b = phase_b_per_slot_svd(emb, seqlens, args)

    logger.info("")
    logger.info("Phase C — pooled subspace vs per-slot")
    phase_c = phase_c_pooled_subspace(emb, seqlens, phase_b, args)

    reco = phase_d_recommendations(phase_a, phase_b, phase_c)

    save_results(args.output_dir, phase_a, phase_b, phase_c, reco, args)
    logger.info("Done.")


if __name__ == "__main__":
    main()
