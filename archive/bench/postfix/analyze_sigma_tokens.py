#!/usr/bin/env python
"""Analyze σ-residual structure and T5-token-space interpretation of a
trained cond-timestep postfix checkpoint.

Two complementary analyses on the same checkpoint:

  Part A — σ-residual structure (weights-only, no TE load required)
    • ||sigma_residual(σ)|| vs σ — did zero-init sigma_mlp actually depart
      from cond-mode behavior?
    • Pairwise cosine(residual(σ_i), residual(σ_j)) — is σ change a gain
      knob, a rotation, or a direction flip between noise/detail regimes?
    • SVD on stacked residuals (num_σ, K·D) — how many independent σ-modes?
    • Per-token (K=32) residual norms — is σ-schedule carried by a few slots?
    • Per-channel residual variance — does the schedule live in a few axes
      of the T5-compatible space?
    • base vs residual magnitude ratio and slot-wise cosine alignment.

  Part B — T5-token NN probe ("can postfix be reproduced as text?")
    Build a T5-token lexicon from cached _anima_te.safetensors: each real
    position → (t5_token_id, post-adapter-vec). Mean-pool per token_id,
    keep tokens with >= --min_count occurrences.

    For residual-only (σ-sweep, per slot): top-k cosine-nearest T5 tokens.
    For full postfix(caption, σ) on selected sample captions: same.

    Consistent top-k across σ  → σ-residual just modulates gain along a
                                 fixed textual direction.
    Varying top-k across σ     → σ-residual carves between different textual
                                 directions per denoising step.

Usage:
    python bench/postfix/analyze_sigma_tokens.py \\
        --postfix_weight output/anima_postfix.safetensors \\
        --dataset_dir post_image_dataset \\
        --num_captions 128 \\
        --num_sigmas 33 \\
        --out_json bench/postfix/results/sigma_tokens_anima_postfix.json
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from library.anima import weights as anima_utils
from library.log import setup_logging
from networks import postfix_anima

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--postfix_weight", default="output/ckpt/anima_postfix.safetensors",
        help="cond-timestep postfix safetensors checkpoint",
    )
    p.add_argument(
        "--dataset_dir", default="post_image_dataset",
        help="Directory with <stem>_anima_te.safetensors files",
    )
    p.add_argument("--num_captions", type=int, default=128,
                   help="Cached TE files to use for base_postfix + lexicon")
    p.add_argument("--num_sigmas", type=int, default=33,
                   help="σ grid size for the sweep in [0, 1]")
    p.add_argument("--min_count", type=int, default=3,
                   help="Only keep T5 tokens that appear in >= this many positions")
    p.add_argument("--top_k", type=int, default=5,
                   help="Top-k nearest tokens per probe")
    p.add_argument("--probe_captions", type=int, default=3,
                   help="Sample captions for full-postfix NN probe (Part B.2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_postfix_network(weight_path, device):
    network, weights_sd = postfix_anima.create_network_from_weights(
        multiplier=1.0, file=weight_path, ae=None, text_encoders=None, unet=None,
    )
    if network.mode != "cond-timestep":
        raise ValueError(
            f"This bench expects mode='cond-timestep', got {network.mode!r}. "
            f"(For plain postfix/cond there's no σ-branch to analyze.)"
        )
    network.load_weights(weight_path)
    network.to(device).eval()
    for p in network.parameters():
        p.requires_grad_(False)
    return network


def find_cached_te(dataset_dir, n, seed):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*_anima_te.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *_anima_te.safetensors in {dataset_dir}")
    rng = np.random.default_rng(seed)
    if len(files) > n:
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx.tolist())]
    return files


def load_cached_te(path):
    """Return (crossattn_emb [512,D], t5_input_ids [512], attn_mask [512] bool, seqlen int)."""
    sd = load_file(path)
    emb = sd["crossattn_emb_v0"].float()            # [512, D]
    ids = sd["t5_input_ids_v0"].long()              # [512]
    mask = sd["attn_mask_v0"].bool()                # [512]
    seqlen = int(mask.sum().item())
    return emb, ids, mask, seqlen


def read_caption(te_path):
    stem = os.path.basename(te_path).replace("_anima_te.safetensors", "")
    txt = os.path.join(os.path.dirname(te_path), f"{stem}.txt")
    if os.path.exists(txt):
        with open(txt) as f:
            return f.read().strip()
    return None


# --------------------------------------------------------------------------- #
# Part A — σ-residual structure
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_residuals(network, sigmas, device):
    """Return (num_σ, K, D) residual tensor via sigma_mlp(sinusoidal(σ))."""
    s = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
    feats = network._sigma_features(s)              # [num_σ, sigma_feature_dim]
    feats = feats.to(next(network.sigma_mlp.parameters()).dtype)
    flat = network.sigma_mlp(feats)                 # [num_σ, K*D]
    return flat.view(len(sigmas), network.num_postfix_tokens, network.embed_dim).float()


@torch.no_grad()
def compute_base_postfixes(network, cached_files, device):
    """Return (M, K, D) caption-conditional base postfixes (no σ residual)."""
    K, D = network.num_postfix_tokens, network.embed_dim
    mlp_dtype = next(network.cond_mlp.parameters()).dtype
    bases = []
    for path in cached_files:
        emb, _ids, mask, seqlen = load_cached_te(path)
        if seqlen == 0:
            continue
        pooled = emb[:seqlen].mean(dim=0, keepdim=True).to(device=device, dtype=mlp_dtype)
        base = network.cond_mlp(pooled).view(1, K, D).float()
        bases.append(base.cpu())
    return torch.cat(bases, dim=0)  # [M, K, D]


def svd_effective_dof(residuals_flat, energy_threshold=0.9):
    """# singular values needed to capture `energy_threshold` fraction of variance."""
    _u, s, _v = np.linalg.svd(residuals_flat, full_matrices=False)
    energy = s ** 2
    cum = np.cumsum(energy) / energy.sum()
    dof = int(np.searchsorted(cum, energy_threshold) + 1)
    return s, dof


def pairwise_cosines(residuals):
    """residuals: (N, K, D) → (N, N) cosine matrix on flattened per-σ vectors."""
    flat = residuals.reshape(residuals.shape[0], -1)
    flat_n = F.normalize(flat, dim=-1)
    return (flat_n @ flat_n.T).numpy()


def per_slot_metrics(residuals, bases):
    """residuals: (N, K, D); bases: (M, K, D).
    Returns per-slot residual norm curves and mean residual/base alignment."""
    K, D = residuals.shape[1], residuals.shape[2]
    res_norm = residuals.norm(dim=-1).numpy()                   # [N, K]
    base_norm_mean = bases.norm(dim=-1).mean(dim=0).numpy()     # [K]

    # Alignment: cos(residual_slot(σ), base_slot(caption)) averaged over σ × caption
    # Same shape (K, D) on each side; flatten per slot.
    res_flat = F.normalize(residuals.reshape(-1, K, D), dim=-1)   # [N, K, D]
    base_flat = F.normalize(bases, dim=-1)                        # [M, K, D]
    # per-slot: (N·M, 1) inner products, then mean per slot
    align = torch.einsum("nkd,mkd->nmk", res_flat, base_flat)     # [N, M, K]
    align_mean = align.mean(dim=(0, 1)).numpy()                   # [K]
    return res_norm, base_norm_mean, align_mean


def slot_symmetry(tensor):
    """tensor: (..., K, D). Return max per-slot deviation from slot 0.
    0.0 → all K slots are literally equal (permutation symmetry unbroken)."""
    slot0 = tensor[..., 0:1, :]
    diff = (tensor - slot0).abs()
    return float(diff.max().item()), float(diff.mean().item())


# --------------------------------------------------------------------------- #
# Part B — T5 token lexicon + NN probe
# --------------------------------------------------------------------------- #

def build_lexicon(cached_files, min_count):
    """Walk cached TE files, collect (token_id → mean_vec, count)."""
    sums = {}   # token_id -> accumulator (D,)
    counts = {}
    D = None
    for path in cached_files:
        emb, ids, mask, seqlen = load_cached_te(path)
        if seqlen == 0:
            continue
        if D is None:
            D = emb.shape[-1]
        for pos in range(seqlen):
            tok = int(ids[pos].item())
            v = emb[pos]
            if tok in sums:
                sums[tok] += v
                counts[tok] += 1
            else:
                sums[tok] = v.clone()
                counts[tok] = 1

    filt_ids = sorted(t for t, c in counts.items() if c >= min_count)
    if not filt_ids:
        raise RuntimeError(
            f"No tokens with >= {min_count} occurrences. "
            f"Lower --min_count or raise --num_captions."
        )
    means = torch.stack([sums[t] / counts[t] for t in filt_ids], dim=0)  # [V, D]
    cnts = np.array([counts[t] for t in filt_ids], dtype=np.int64)
    return filt_ids, means, cnts


def top_k_nearest(query, lexicon_norm, k):
    """query: [D] or [B, D]. lexicon_norm: [V, D] unit-norm. Returns (top_k_idx, top_k_cos)."""
    q = F.normalize(query, dim=-1)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    sims = q @ lexicon_norm.T                         # [B, V]
    cos, idx = sims.topk(k=k, dim=-1)
    return idx.numpy(), cos.numpy()


def format_token_list(token_ids, cosines, tokenizer):
    items = []
    for tid, c in zip(token_ids, cosines):
        tok = tokenizer.convert_ids_to_tokens([int(tid)])[0]
        items.append(f"{tok!s}({c:+.3f})")
    return " ".join(items)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    logger.info(f"loading postfix from {args.postfix_weight}")
    network = load_postfix_network(args.postfix_weight, device)
    K, D = network.num_postfix_tokens, network.embed_dim
    logger.info(f"postfix: mode={network.mode} K={K} D={D} "
                f"sigma_feat={network.sigma_feature_dim} sigma_hidden={network.sigma_hidden_dim}")

    cached_files = find_cached_te(args.dataset_dir, args.num_captions, args.seed)
    logger.info(f"using {len(cached_files)} cached TE files")

    sigmas = np.linspace(0.0, 1.0, args.num_sigmas).tolist()

    # --------- Part A ---------
    logger.info("computing σ-residuals")
    residuals = compute_residuals(network, sigmas, device).cpu()         # [N, K, D]

    logger.info("computing caption-conditional base postfixes")
    bases = compute_base_postfixes(network, cached_files, device)        # [M, K, D]

    res_global_norm = residuals.reshape(residuals.shape[0], -1).norm(dim=-1).numpy()   # [N]
    base_global_norm = bases.reshape(bases.shape[0], -1).norm(dim=-1).numpy()          # [M]
    residual_over_base = res_global_norm / (base_global_norm.mean() + 1e-9)

    # Slot-symmetry diagnostic: zero-init final MLP + positional-symmetric splice
    # of K slots into zero-padding keeps permutation symmetry unbroken during training
    # (identical gradients → identical slots forever). If max-diff is 0, effective K=1.
    res_slot_max, res_slot_mean = slot_symmetry(residuals)
    base_slot_max, base_slot_mean = slot_symmetry(bases)
    slots_symmetric = (res_slot_max < 1e-4) and (base_slot_max < 1e-4)
    effective_K = 1 if slots_symmetric else K

    cos_mat = pairwise_cosines(residuals)                                 # [N, N]
    singular, dof90 = svd_effective_dof(residuals.reshape(residuals.shape[0], -1).numpy(), 0.9)
    res_norm_per_slot, base_norm_per_slot, align_per_slot = per_slot_metrics(residuals, bases)

    # Per-channel variance of residual across σ (averaged over K)
    per_channel_var = residuals.var(dim=0).mean(dim=0).numpy()           # [D]
    top_channel_idx = np.argsort(-per_channel_var)[:10]

    # --------- Part A report ---------
    print("\n" + "=" * 78)
    print(f"Part A — σ-residual structure  —  {os.path.basename(args.postfix_weight)}")
    print("=" * 78)
    print(f"  K={K}   D={D}   num_σ={args.num_sigmas}   base samples (M)={bases.shape[0]}")
    print("\n  Slot-symmetry check (are the K postfix tokens distinguishable?)")
    print(f"    residual  max inter-slot |diff|={res_slot_max:.2e}  mean={res_slot_mean:.2e}")
    print(f"    base      max inter-slot |diff|={base_slot_max:.2e}  mean={base_slot_mean:.2e}")
    if slots_symmetric:
        print(f"    → ALL K={K} SLOTS ARE IDENTICAL.  Effective K = 1.")
        print("      Cause: zero-init final MLP layer + positional-symmetric splice of K")
        print("      slots into the zero-padding region keeps permutation symmetry unbroken")
        print("      during training (each slot receives identical gradients → stays identical).")
        print("      Fix: random-init final layer OR add per-slot identity embedding:")
        print("        postfix[k] = mlp_out[k] + slot_embed[k]   # slot_embed: Param(K, D)")
    else:
        print("    → Slots are distinguishable (max |diff| > 0). Per-slot breakdown below.")
    print("\n  ||sigma_residual(σ)||   (global, flattened K·D)")
    print(f"    min = {res_global_norm.min():.3f}   mean = {res_global_norm.mean():.3f}   "
          f"max = {res_global_norm.max():.3f}")
    print(f"    at σ=0.0 : {res_global_norm[0]:.3f}")
    print(f"    at σ=0.5 : {res_global_norm[len(sigmas)//2]:.3f}")
    print(f"    at σ=1.0 : {res_global_norm[-1]:.3f}")
    print(f"  ||base_postfix||      mean over {bases.shape[0]} captions = {base_global_norm.mean():.3f}")
    print(f"  residual/base ratio   min={residual_over_base.min():.3f}  "
          f"mean={residual_over_base.mean():.3f}  max={residual_over_base.max():.3f}")
    if residual_over_base.mean() > 1.5:
        print("  → residual dominates base. Caption-conditional content is largely washed out;")
        print("    the postfix is mostly a σ-conditional (caption-independent) signal.")
    print("\n  pairwise cos(residual(σ_i), residual(σ_j))")
    print(f"    cos(σ=0.0, σ=1.0) = {cos_mat[0, -1]:+.3f}    "
          f"(+1 = same direction, -1 = flip, 0 = orthogonal)")
    print(f"    median off-diag   = {np.median(cos_mat[np.triu_indices_from(cos_mat, k=1)]):+.3f}")
    if cos_mat[0, -1] > 0.95:
        print("  → σ-residual is nearly direction-invariant — effectively a gain knob on one")
        print("    fixed direction, not a rotation between different textual regimes.")
    print("\n  SVD (flattened residuals):")
    print("    top-5 σ-modes  = " + " ".join(f"{s:.3f}" for s in singular[:5]))
    print(f"    effective DoF (90% energy) = {dof90}")
    print("\n  Top-10 channels by residual variance across σ:")
    print(f"    idx = {top_channel_idx.tolist()}")
    print(f"    var = {[f'{v:.2e}' for v in per_channel_var[top_channel_idx]]}")
    if not slots_symmetric:
        print("\n  Per-slot means (top-10 by residual norm):")
        print("    slot  ||residual|| ||base||   align(res,base)")
        order = np.argsort(-res_norm_per_slot.mean(axis=0))
        for slot in order[:10]:
            print(f"     {slot:3d}   {res_norm_per_slot[:, slot].mean():6.3f}      "
                  f"{base_norm_per_slot[slot]:6.3f}        {align_per_slot[slot]:+6.3f}")

    # --------- Part B ---------
    logger.info("building T5-token lexicon")
    tokenizer = anima_utils.load_t5_tokenizer()
    lex_ids, lex_vecs, lex_counts = build_lexicon(cached_files, args.min_count)
    lex_norm = F.normalize(lex_vecs, dim=-1)
    logger.info(f"lexicon: {len(lex_ids)} unique T5 tokens (min_count={args.min_count})")

    # Residual-only NN: for each σ, pool residual across K, rank nearest tokens.
    # Also per-slot at σ=0.5 (illustrative).
    residual_pooled = residuals.mean(dim=1)                              # [N, D]
    res_topk_idx, res_topk_cos = top_k_nearest(residual_pooled, lex_norm, args.top_k)  # [N, k]

    mid = len(sigmas) // 2
    per_slot_topk = []
    if not slots_symmetric:
        for k in range(K):
            idx, cos = top_k_nearest(residuals[mid, k], lex_norm, args.top_k)
            per_slot_topk.append((idx[0].tolist(), cos[0].tolist()))

    # Full postfix NN on sample captions
    rng = np.random.default_rng(args.seed)
    probe_idx = rng.choice(len(cached_files), size=min(args.probe_captions, len(cached_files)), replace=False)
    probe_files = [cached_files[i] for i in probe_idx]
    full_topk = {}
    for pf in probe_files:
        emb, _ids, _mask, seqlen = load_cached_te(pf)
        if seqlen == 0:
            continue
        mlp_dtype = next(network.cond_mlp.parameters()).dtype
        with torch.no_grad():
            pooled = emb[:seqlen].mean(dim=0, keepdim=True).to(device=device, dtype=mlp_dtype)
            base_one = network.cond_mlp(pooled).view(K, D).float().cpu()         # [K, D]
        # postfix(σ, caption) pooled across slots
        post = (residuals + base_one.unsqueeze(0))                           # [N, K, D]
        post_pooled = post.mean(dim=1)                                       # [N, D]
        idx, cos = top_k_nearest(post_pooled, lex_norm, args.top_k)
        cap = read_caption(pf)
        caption_preview = (cap[:60] + "…") if cap and len(cap) > 60 else (cap or "<no .txt>")
        full_topk[pf] = {
            "caption": caption_preview,
            "per_sigma": {f"{sigmas[i]:.3f}": {
                "ids": idx[i].tolist(),
                "tokens": tokenizer.convert_ids_to_tokens([int(x) for x in idx[i]]),
                "cos": cos[i].tolist(),
            } for i in range(len(sigmas))},
        }

    # --------- Part B report ---------
    print("\n" + "=" * 78)
    print("Part B — T5 token NN probe")
    print("=" * 78)
    print(f"  lexicon size: {len(lex_ids)}   (min_count={args.min_count}, "
          f"max count in corpus={int(lex_counts.max())})")

    print("\n  Residual-only NN (pooled over K slots)  — σ sweep")
    print(f"  σ      top-{args.top_k} nearest T5 tokens (cosine)")
    step = max(1, len(sigmas) // 11)
    for i in range(0, len(sigmas), step):
        toks = format_token_list(res_topk_idx[i], res_topk_cos[i], tokenizer)
        print(f"  {sigmas[i]:.3f}  {toks}")

    if not slots_symmetric:
        print(f"\n  Per-slot residual NN at σ={sigmas[mid]:.3f}  (anchors for each slot)")
        order = np.argsort(-res_norm_per_slot.mean(axis=0))
        show_slots = list(range(min(8, K))) + [int(x) for x in order[:8] if x >= 8][:4]
        for slot in show_slots:
            idx, cos = per_slot_topk[slot]
            toks = format_token_list(idx, cos, tokenizer)
            print(f"   slot {slot:3d}  ||res||={res_norm_per_slot[mid, slot]:5.3f}   {toks}")
    else:
        print("\n  (per-slot NN skipped — all K slots are identical.)")

    print(f"\n  Full postfix(caption, σ) NN  — {len(full_topk)} sample captions")
    for pf, entry in full_topk.items():
        print(f"\n   caption: {entry['caption']}")
        print(f"     σ      top-{args.top_k}")
        for i in range(0, len(sigmas), step):
            s_key = f"{sigmas[i]:.3f}"
            d = entry["per_sigma"][s_key]
            toks = " ".join(f"{t!s}({c:+.3f})" for t, c in zip(d["tokens"], d["cos"]))
            print(f"     {s_key}  {toks}")

    # --------- JSON out ---------
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "postfix_weight": args.postfix_weight,
            "K": K, "D": D,
            "sigmas": sigmas,
            "num_base_captions": int(bases.shape[0]),
            "num_lexicon_tokens": len(lex_ids),
            "part_a": {
                "slot_symmetry": {
                    "residual_max_interslot_diff": res_slot_max,
                    "residual_mean_interslot_diff": res_slot_mean,
                    "base_max_interslot_diff": base_slot_max,
                    "base_mean_interslot_diff": base_slot_mean,
                    "all_slots_identical": bool(slots_symmetric),
                    "effective_K": effective_K,
                },
                "residual_global_norm": res_global_norm.tolist(),
                "base_global_norm_mean": float(base_global_norm.mean()),
                "base_global_norm_std": float(base_global_norm.std()),
                "residual_over_base": residual_over_base.tolist(),
                "pairwise_cosine_endpoints": float(cos_mat[0, -1]),
                "pairwise_cosine_matrix": cos_mat.tolist(),
                "svd_top5": singular[:5].tolist(),
                "svd_dof_90pct": int(dof90),
                "per_slot_residual_norm_mean": res_norm_per_slot.mean(axis=0).tolist(),
                "per_slot_base_norm": base_norm_per_slot.tolist(),
                "per_slot_align_residual_base": align_per_slot.tolist(),
                "top_variance_channels": {
                    "idx": top_channel_idx.tolist(),
                    "var": per_channel_var[top_channel_idx].tolist(),
                },
            },
            "part_b": {
                "residual_pooled_topk": {
                    f"{sigmas[i]:.3f}": {
                        "ids": res_topk_idx[i].tolist(),
                        "tokens": tokenizer.convert_ids_to_tokens(
                            [int(x) for x in res_topk_idx[i]]),
                        "cos": res_topk_cos[i].tolist(),
                    } for i in range(len(sigmas))
                },
                "per_slot_topk_at_mid_sigma": {
                    f"{k}": {
                        "sigma": sigmas[mid],
                        "ids": per_slot_topk[k][0],
                        "tokens": tokenizer.convert_ids_to_tokens(
                            [int(x) for x in per_slot_topk[k][0]]),
                        "cos": per_slot_topk[k][1],
                    } for k in range(K)
                } if not slots_symmetric else "skipped (all slots identical)",
                "full_postfix_probe": full_topk,
            },
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
