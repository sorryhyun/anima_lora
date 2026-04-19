#!/usr/bin/env python
"""Analyze a cond-mode (no σ-branch) postfix checkpoint.

Companion to analyze_sigma_tokens.py — that one needs a σ residual; this one
runs on `mode=cond` checkpoints (e.g. output_temp/anima_postfix_exp.safetensors)
and answers the same underlying question: *what did the postfix learn?*

Diagnostics:
  1. Slot-symmetry check — are the K slots all identical? (Same zero-init
     failure mode as the cond-timestep analysis.)
  2. Caption-variance vs slot-variance — does cond_mlp actually vary with the
     caption, or does it collapse to a prompt-agnostic constant?
  3. SVD across captions — effective # of independent directions the postfix
     uses across the dataset. DoF=1 ⇒ one fixed direction, gain-modulated by
     caption. DoF=K·something ⇒ genuinely prompt-conditional.
  4. Per-slot norm + per-channel variance.
  5. T5-token NN probe — nearest tokens to pooled postfix, and per-caption NN.
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
        "--postfix_weight", default="output_temp/anima_postfix_exp.safetensors"
    )
    p.add_argument("--dataset_dir", default="post_image_dataset")
    p.add_argument("--num_captions", type=int, default=256)
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--probe_captions", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def load_network(weight_path, device):
    network, _ = postfix_anima.create_network_from_weights(
        multiplier=1.0,
        file=weight_path,
        ae=None,
        text_encoders=None,
        unet=None,
    )
    if network.mode not in ("cond",):
        raise ValueError(
            f"This bench expects mode='cond', got {network.mode!r}. "
            f"For cond-timestep use analyze_sigma_tokens.py; for plain postfix/prefix "
            f"the learned tensor is already static (no MLP)."
        )
    network.load_weights(weight_path)
    network.to(device).eval()
    for p in network.parameters():
        p.requires_grad_(False)
    return network


def find_cached(dataset_dir, n, seed):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*_anima_te.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *_anima_te.safetensors in {dataset_dir}")
    rng = np.random.default_rng(seed)
    if len(files) > n:
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx.tolist())]
    return files


def load_te(path):
    sd = load_file(path)
    emb = sd["crossattn_emb_v0"].float()
    ids = sd["t5_input_ids_v0"].long()
    mask = sd["attn_mask_v0"].bool()
    seqlen = int(mask.sum().item())
    return emb, ids, mask, seqlen


def read_caption(te_path):
    stem = os.path.basename(te_path).replace("_anima_te.safetensors", "")
    txt = os.path.join(os.path.dirname(te_path), f"{stem}.txt")
    if os.path.exists(txt):
        with open(txt) as f:
            return f.read().strip()
    return None


@torch.no_grad()
def compute_postfixes(network, cached_files, device):
    K, D = network.num_postfix_tokens, network.embed_dim
    mlp_dtype = next(network.cond_mlp.parameters()).dtype
    outs = []
    for path in cached_files:
        emb, _ids, mask, seqlen = load_te(path)
        if seqlen == 0:
            continue
        pooled = (
            emb[:seqlen].mean(dim=0, keepdim=True).to(device=device, dtype=mlp_dtype)
        )
        out = network.cond_mlp(pooled).view(1, K, D).float().cpu()
        outs.append(out)
    return torch.cat(outs, dim=0)  # [M, K, D]


def slot_symmetry(t):
    slot0 = t[..., 0:1, :]
    diff = (t - slot0).abs()
    return float(diff.max().item()), float(diff.mean().item())


def svd_dof(flat, energy=0.9):
    _u, s, _v = np.linalg.svd(flat, full_matrices=False)
    cum = np.cumsum(s**2) / (s**2).sum()
    return s, int(np.searchsorted(cum, energy) + 1)


def build_lexicon(cached_files, min_count):
    sums, counts, D = {}, {}, None
    for path in cached_files:
        emb, ids, mask, seqlen = load_te(path)
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
    filt = sorted(t for t, c in counts.items() if c >= min_count)
    if not filt:
        raise RuntimeError(f"no tokens with >= {min_count} occurrences")
    means = torch.stack([sums[t] / counts[t] for t in filt], dim=0)
    cnts = np.array([counts[t] for t in filt], dtype=np.int64)
    return filt, means, cnts


def top_k_nearest(q, lex_norm, k):
    qn = F.normalize(q, dim=-1)
    if qn.dim() == 1:
        qn = qn.unsqueeze(0)
    sims = qn @ lex_norm.T
    cos, idx = sims.topk(k=k, dim=-1)
    return idx.numpy(), cos.numpy()


def fmt_toks(ids, cos, tok):
    return " ".join(
        f"{tok.convert_ids_to_tokens([int(i)])[0]!s}({c:+.3f})"
        for i, c in zip(ids, cos)
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    logger.info(f"loading postfix from {args.postfix_weight}")
    network = load_network(args.postfix_weight, device)
    K, D = network.num_postfix_tokens, network.embed_dim
    logger.info(
        f"postfix: mode={network.mode} K={K} D={D} "
        f"cond_hidden={network.cond_hidden_dim}"
    )

    cached = find_cached(args.dataset_dir, args.num_captions, args.seed)
    logger.info(f"using {len(cached)} cached TE files")

    logger.info("computing caption-conditional postfixes")
    P = compute_postfixes(network, cached, device)  # [M, K, D]
    M = P.shape[0]

    # Slot symmetry
    slot_max, slot_mean = slot_symmetry(P)
    slots_identical = slot_max < 1e-4
    effective_K = 1 if slots_identical else K

    # Mean (prompt-agnostic component) vs caption-deviation
    P_mean = P.mean(dim=0, keepdim=True)  # [1, K, D]  — "bias" direction
    P_dev = P - P_mean  # [M, K, D]  — caption-specific part
    mean_norm = P_mean.view(-1).norm().item()
    per_caption_norm = P.reshape(M, -1).norm(dim=-1).numpy()
    per_caption_dev_norm = P_dev.reshape(M, -1).norm(dim=-1).numpy()
    dev_over_total = per_caption_dev_norm.mean() / (per_caption_norm.mean() + 1e-9)

    # Cosine spread across captions (flattened) — are all captions pointing the same way?
    flat = P.reshape(M, -1)
    flat_n = F.normalize(flat, dim=-1)
    cos_mat = (flat_n @ flat_n.T).numpy()
    off = cos_mat[np.triu_indices_from(cos_mat, k=1)]
    cos_min = float(off.min())
    cos_med = float(np.median(off))
    cos_mean = float(off.mean())

    # SVD: how many independent directions across captions?
    s_vals, dof90 = svd_dof(flat.numpy(), 0.9)

    # Per-slot norms (avg over captions)
    per_slot_norm = P.norm(dim=-1).mean(dim=0).numpy()  # [K]
    per_channel_var = P.var(dim=0).mean(dim=0).numpy()  # [D]
    top_ch = np.argsort(-per_channel_var)[:10]

    print("\n" + "=" * 78)
    print(f"cond-mode postfix analysis — {os.path.basename(args.postfix_weight)}")
    print("=" * 78)
    print(f"  K={K}  D={D}  M captions={M}  cond_hidden={network.cond_hidden_dim}")
    print(
        f"\n  Slot-symmetry  max inter-slot |diff| = {slot_max:.2e}   mean = {slot_mean:.2e}"
    )
    if slots_identical:
        print(f"    → ALL K={K} SLOTS ARE IDENTICAL across captions.  Effective K = 1.")
    else:
        print("    → slots are distinguishable (max |diff| > 0).")

    print("\n  Caption-variance probe")
    print(f"    mean postfix ||P_mean||                = {mean_norm:.3f}")
    print(
        f"    per-caption ||P||   mean = {per_caption_norm.mean():.3f}  "
        f"std = {per_caption_norm.std():.3f}"
    )
    print(
        f"    per-caption deviation from mean ||P-P_mean||  "
        f"mean = {per_caption_dev_norm.mean():.3f}"
    )
    print(f"    deviation / total ratio                = {dev_over_total:.3f}")
    if dev_over_total < 0.1:
        print(
            "    → caption deviation is <10% of total norm — postfix is "
            "largely prompt-agnostic (same direction regardless of caption)."
        )
    elif dev_over_total < 0.3:
        print(
            "    → caption deviation is small (<30%) — postfix has a dominant "
            "prompt-agnostic component with weak caption modulation."
        )
    else:
        print(
            "    → caption deviation is meaningful (>30%) — postfix actually "
            "reads the prompt."
        )
    print(f"\n  Pairwise cos(postfix_i, postfix_j)  over {M} captions")
    print(
        f"    min = {cos_min:+.3f}   median = {cos_med:+.3f}   mean = {cos_mean:+.3f}"
    )
    if cos_mean > 0.95:
        print(
            "    → every caption produces nearly the SAME postfix direction. "
            "Effective caption-DoF ≈ 1 (gain knob)."
        )

    print("\n  SVD across captions (flattened K·D):")
    print("    top-5 σ-values = " + " ".join(f"{s:.3f}" for s in s_vals[:5]))
    print(f"    effective DoF (90% energy) = {dof90}")
    print(f"\n  Per-slot ||postfix|| (K={K})")
    if slots_identical:
        print(f"    (all identical; showing slot 0)  = {per_slot_norm[0]:.3f}")
    else:
        order = np.argsort(-per_slot_norm)
        for slot in order[:10]:
            print(f"     slot {slot:3d}   ||P||_mean = {per_slot_norm[slot]:.3f}")
    print("\n  Top-10 channels by variance across captions:")
    print(f"    idx = {top_ch.tolist()}")
    print(f"    var = {[f'{v:.2e}' for v in per_channel_var[top_ch]]}")

    # ---------- T5 token NN ----------
    logger.info("building T5-token lexicon")
    tokenizer = anima_utils.load_t5_tokenizer()
    lex_ids, lex_vecs, lex_counts = build_lexicon(cached, args.min_count)
    lex_norm = F.normalize(lex_vecs, dim=-1)
    logger.info(f"lexicon: {len(lex_ids)} tokens")

    print("\n" + "=" * 78)
    print("T5-token NN probe")
    print("=" * 78)
    # mean postfix direction
    mean_pooled = P_mean.mean(dim=1).squeeze(0)  # [D]
    idx, cos = top_k_nearest(mean_pooled, lex_norm, args.top_k)
    print("\n  Mean postfix (pooled over slots, averaged over captions)")
    print(f"    top-{args.top_k}: {fmt_toks(idx[0], cos[0], tokenizer)}")
    if cos[0].max() < 0.2:
        print(
            f"    → low cosine to every T5 token (max={cos[0].max():.3f}) — "
            f"postfix lives off the T5 token manifold."
        )

    # per-slot NN on the mean postfix
    if not slots_identical:
        print("\n  Per-slot NN on mean postfix (slots sorted by ||·||)")
        order = np.argsort(-per_slot_norm)
        for slot in order[:8]:
            idx, cos = top_k_nearest(P_mean[0, slot], lex_norm, args.top_k)
            print(
                f"    slot {slot:3d}  ||={per_slot_norm[slot]:5.3f}   "
                f"{fmt_toks(idx[0], cos[0], tokenizer)}"
            )

    # probe captions: does NN change with caption?
    rng = np.random.default_rng(args.seed)
    probe_idx = rng.choice(
        len(cached), size=min(args.probe_captions, len(cached)), replace=False
    )
    print("\n  Per-caption NN — does postfix top-k vary with prompt?")
    for i in probe_idx:
        pf = cached[i]
        cap = read_caption(pf)
        preview = (cap[:60] + "…") if cap and len(cap) > 60 else (cap or "<no .txt>")
        # pool over slots for this caption
        pooled = P[i].mean(dim=0)  # [D]
        idx, cos = top_k_nearest(pooled, lex_norm, args.top_k)
        print(f"\n    caption: {preview}")
        print(f"      top-{args.top_k}: {fmt_toks(idx[0], cos[0], tokenizer)}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "postfix_weight": args.postfix_weight,
            "K": K,
            "D": D,
            "num_captions": M,
            "slot_symmetry": {
                "max_interslot_diff": slot_max,
                "mean_interslot_diff": slot_mean,
                "all_identical": bool(slots_identical),
                "effective_K": effective_K,
            },
            "caption_variance": {
                "mean_postfix_norm": mean_norm,
                "per_caption_norm_mean": float(per_caption_norm.mean()),
                "per_caption_norm_std": float(per_caption_norm.std()),
                "per_caption_deviation_mean": float(per_caption_dev_norm.mean()),
                "deviation_over_total": float(dev_over_total),
            },
            "pairwise_cosine": {"min": cos_min, "median": cos_med, "mean": cos_mean},
            "svd_top5": s_vals[:5].tolist(),
            "svd_dof_90pct": int(dof90),
            "per_slot_norm": per_slot_norm.tolist(),
            "top_variance_channels": {
                "idx": top_ch.tolist(),
                "var": per_channel_var[top_ch].tolist(),
            },
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
