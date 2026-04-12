#!/usr/bin/env python
"""Diagnose what T5 misses by comparing the cached caption embedding against
the inversion centroid for the same image, in raw + late-block functional space.

For one image:
  1. Load cached T5 caption embedding from <stem>_anima_te.safetensors (crossattn_emb_v0)
  2. Load inversion centroid + per-run from <stem>_inverted*.safetensors
  3. Forward all of them through DiT at requested blocks via probe_functional_space()
  4. Compute the "T5 gap": mean cosine(t5, inv_run*) at each block, plus how that
     compares to the natural inv-vs-inv pairwise variance — the bigger the gap,
     the more functional headroom an enhancement module has at that depth.
  5. Compute per-slot delta magnitudes ||inv_mean - t5||_2 to see which slots
     carry the largest discrepancy
  6. Save JSON + print human-readable summary

Usage:
    python bench/diagnose_t5_vs_inversion.py \\
        --image post_image_dataset/10811132.png \\
        --results_dir inversions_probe_test/results \\
        --te_dir post_image_dataset \\
        --dit models/diffusion_models/anima-preview3-base.safetensors \\
        --vae models/vae/qwen_image_vae.safetensors \\
        --probe_blocks 0,12,20,24,27
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
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import anima_utils, qwen_image_autoencoder_kl
from library.utils import setup_logging
from scripts.invert_embedding import probe_functional_space

setup_logging()
logger = logging.getLogger(__name__)

IMAGE_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument(
        "--results_dir",
        required=True,
        help="Directory holding <stem>_inverted*.safetensors",
    )
    p.add_argument(
        "--te_dir",
        required=True,
        help="Directory holding <stem>_anima_te.safetensors",
    )
    p.add_argument("--dit", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--probe_samples", type=int, default=4)
    p.add_argument(
        "--probe_blocks",
        type=str,
        default="0,12,20,24,27",
        help="Comma-separated DiT block indices to capture",
    )
    p.add_argument("--vae_chunk_size", type=int, default=64)
    p.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Output JSON path (default: <results_dir>/../logs/<stem>_t5_vs_inv.json)",
    )
    p.add_argument(
        "--top_k_slots",
        type=int,
        default=10,
        help="Report top-K slots by ||inv_mean - t5||_2",
    )
    return p.parse_args()


def load_t5_embedding(te_dir, stem):
    path = os.path.join(te_dir, f"{stem}_anima_te.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cached TE not found: {path}")
    sd = load_file(path)
    if "crossattn_emb_v0" in sd:
        emb = sd["crossattn_emb_v0"]
    elif "crossattn_emb" in sd:
        emb = sd["crossattn_emb"]
    else:
        raise KeyError(f"No crossattn_emb in {path}")
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)
    return emb.float()


def load_inversion_set(results_dir, stem):
    """Load (label, embed) for inv_run0..N + inv_mean (if it exists)."""
    labeled = []
    run_files = sorted(
        glob.glob(os.path.join(results_dir, f"{stem}_inverted_run*.safetensors"))
    )
    for rf in run_files:
        idx = os.path.basename(rf).split("_run")[-1].split(".")[0]
        sd = load_file(rf)
        emb = sd["crossattn_emb"]
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        labeled.append((f"inv_run{idx}", emb.float()))
    mean_path = os.path.join(results_dir, f"{stem}_inverted.safetensors")
    if os.path.exists(mean_path):
        sd = load_file(mean_path)
        emb = sd["crossattn_emb"]
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        labeled.append(("inv_mean", emb.float()))
    if not labeled:
        raise FileNotFoundError(
            f"No inversion artifacts for stem={stem} in {results_dir}"
        )
    return labeled


def encode_image(image_path, vae_path, device, chunk_size=64):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    h = (h // 32) * 32
    w = (w // 32) * 32
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    vae = qwen_image_autoencoder_kl.load_vae(
        vae_path, device="cpu", disable_mmap=True, spatial_chunk_size=chunk_size
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()
    tensor = IMAGE_TRANSFORMS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(tensor).to(torch.bfloat16)
    del vae
    torch.cuda.empty_cache()
    return latents


def per_slot_delta_analysis(t5_emb, inv_mean_emb, top_k=10):
    """||inv - t5||_2 per slot, plus magnitudes of t5 and inv for context."""
    t5 = t5_emb.squeeze(0).float().cpu()
    inv = inv_mean_emb.squeeze(0).float().cpu()
    delta = inv - t5
    delta_norm = delta.norm(dim=-1)
    t5_norm = t5.norm(dim=-1)
    inv_norm = inv.norm(dim=-1)

    rel = (delta_norm / t5_norm.clamp_min(1e-8)).numpy()
    delta_np = delta_norm.numpy()
    top_idx = np.argsort(-delta_np)[:top_k].tolist()

    return {
        "n_slots": int(t5.shape[0]),
        "delta_norm_mean": float(delta_np.mean()),
        "delta_norm_median": float(np.median(delta_np)),
        "delta_norm_max": float(delta_np.max()),
        "delta_to_t5_ratio_mean": float(rel.mean()),
        "delta_to_t5_ratio_median": float(np.median(rel)),
        "t5_norm_mean": float(t5_norm.mean().item()),
        "inv_norm_mean": float(inv_norm.mean().item()),
        "top_k_slots": [
            {
                "slot": int(i),
                "delta_norm": float(delta_np[i]),
                "delta_to_t5_ratio": float(rel[i]),
            }
            for i in top_idx
        ],
    }


def t5_gap_summary(probe_result):
    """Per block: mean cos of t5 vs inv_run* pairs, vs inv_run* pairwise mean.

    The gap (t5_vs_inv - inv_pairwise) tells you how far T5 sits from the
    cluster of inversions in functional space, normalized by how spread the
    inversion cluster itself is. More negative = bigger enhancement headroom.
    """
    labels = probe_result["labels"]
    if "t5" not in labels:
        return None

    out = {}
    for bi in probe_result["block_idxs"]:
        bi_str = str(bi)
        pw = probe_result["per_block"][bi_str]["pairwise_cos_flat"]
        t5_pairs = []
        inv_pairs = []
        for key, val in pw.items():
            a, b = key.split("__")
            if a == "t5" or b == "t5":
                other = b if a == "t5" else a
                if other != "inv_mean":
                    t5_pairs.append(val)
            else:
                if a != "inv_mean" and b != "inv_mean":
                    inv_pairs.append(val)
        out[bi_str] = {
            "block_idx": int(bi),
            "mean_cos_t5_vs_inv_runs": float(np.mean(t5_pairs)) if t5_pairs else None,
            "mean_cos_inv_runs_pairwise": float(np.mean(inv_pairs))
            if inv_pairs
            else None,
            "gap": (float(np.mean(t5_pairs)) - float(np.mean(inv_pairs)))
            if t5_pairs and inv_pairs
            else None,
        }
    return out


def raw_t5_inv_summary(probe_result):
    """Mean raw cosine of t5 vs inv_run* pairs, and t5 vs inv_mean."""
    labels = probe_result["labels"]
    if "t5" not in labels:
        return None
    pw = probe_result["raw_pairwise_cos_flat"]
    t5_runs = []
    t5_mean = None
    for key, val in pw.items():
        a, b = key.split("__")
        if a == "t5" or b == "t5":
            other = b if a == "t5" else a
            if other == "inv_mean":
                t5_mean = val
            else:
                t5_runs.append(val)
    return {
        "mean_raw_cos_t5_vs_inv_runs": float(np.mean(t5_runs)) if t5_runs else None,
        "raw_cos_t5_vs_inv_mean": t5_mean,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    logger.info(f"=== T5 vs inversion diagnostic for {stem} ===")

    inv_labeled = load_inversion_set(args.results_dir, stem)
    logger.info(
        f"Loaded {len(inv_labeled)} inversion embeddings: "
        f"{[lbl for lbl, _ in inv_labeled]}"
    )
    t5_emb = load_t5_embedding(args.te_dir, stem)
    logger.info(f"Loaded cached T5 embedding: shape={tuple(t5_emb.shape)}")

    inv_mean_emb = next((e for lbl, e in inv_labeled if lbl == "inv_mean"), None)
    if inv_mean_emb is not None and inv_mean_emb.shape != t5_emb.shape:
        logger.warning(
            f"shape mismatch: t5={tuple(t5_emb.shape)} "
            f"inv={tuple(inv_mean_emb.shape)}"
        )

    latents = encode_image(args.image, args.vae, device, chunk_size=args.vae_chunk_size)
    logger.info(f"Encoded latents: {tuple(latents.shape)}")

    logger.info("Loading DiT...")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)
    anima.split_attn = False
    anima.eval()
    anima.to(device)

    labeled = [("t5", t5_emb)] + inv_labeled

    class _A:
        pass

    block_idxs = [int(s) for s in args.probe_blocks.split(",") if s.strip()]
    probe = probe_functional_space(
        _A(),
        anima,
        latents,
        labeled,
        device,
        n_probes=args.probe_samples,
        block_idxs=block_idxs,
    )
    if probe is None:
        logger.error("Probe returned None")
        return 1

    delta_stats = None
    if inv_mean_emb is not None:
        delta_stats = per_slot_delta_analysis(
            t5_emb, inv_mean_emb, top_k=args.top_k_slots
        )

    gap = t5_gap_summary(probe)
    raw_summary = raw_t5_inv_summary(probe)

    payload = {
        "stem": stem,
        "image": args.image,
        "probe_blocks": block_idxs,
        "n_probes": args.probe_samples,
        "labels": probe["labels"],
        "raw_pairwise_cos_flat": probe["raw_pairwise_cos_flat"],
        "raw_summary": probe["raw_summary"],
        "raw_t5_vs_inv": raw_summary,
        "per_block": probe["per_block"],
        "t5_gap_per_block": gap,
        "delta_stats_t5_vs_inv_mean": delta_stats,
    }

    out_path = args.out_json
    if out_path is None:
        logs_dir = os.path.join(
            os.path.dirname(os.path.normpath(args.results_dir)), "logs"
        )
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"{stem}_t5_vs_inv.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {out_path}")

    print(f"\n=== T5 vs inversion diagnostic — {stem} ===")
    print(f"labels:    {probe['labels']}")
    print(f"n_probes:  {probe['n_probes']}")
    print()

    if raw_summary is not None:
        print("Raw embedding cosines:")
        if raw_summary["mean_raw_cos_t5_vs_inv_runs"] is not None:
            print(
                f"  mean cos(t5, inv_run*):   "
                f"{raw_summary['mean_raw_cos_t5_vs_inv_runs']:.4f}"
            )
        if raw_summary["raw_cos_t5_vs_inv_mean"] is not None:
            print(
                f"  cos(t5, inv_mean):        "
                f"{raw_summary['raw_cos_t5_vs_inv_mean']:.4f}"
            )
        print(
            f"  reference (mean over all): "
            f"{probe['raw_summary']['mean_raw_flat_cos']:.4f}"
        )
        print()

    if delta_stats is not None:
        print("Raw embedding delta (inv_mean - t5):")
        print(f"  n_slots:                  {delta_stats['n_slots']}")
        print(f"  ||delta||_2 mean:         {delta_stats['delta_norm_mean']:.4f}")
        print(f"  ||delta||_2 median:       {delta_stats['delta_norm_median']:.4f}")
        print(f"  ||delta||_2 max:          {delta_stats['delta_norm_max']:.4f}")
        print(
            f"  ||delta||/||t5|| mean:    "
            f"{delta_stats['delta_to_t5_ratio_mean']:.4f}"
        )
        print(
            f"  ||delta||/||t5|| median:  "
            f"{delta_stats['delta_to_t5_ratio_median']:.4f}"
        )
        print(f"  ||t5||_2 mean per slot:   {delta_stats['t5_norm_mean']:.4f}")
        print(f"  ||inv||_2 mean per slot:  {delta_stats['inv_norm_mean']:.4f}")
        print(f"  Top {len(delta_stats['top_k_slots'])} slots by ||delta||:")
        for s in delta_stats["top_k_slots"]:
            print(
                f"    slot {s['slot']:4d}  delta={s['delta_norm']:.3f}  "
                f"rel={s['delta_to_t5_ratio']:.3f}"
            )
        print()

    print("Functional probe per block (mean over ALL pairs incl. t5):")
    print("  block | flat cos | per-tok cos")
    print("  ------|----------|------------")
    for bi in probe["block_idxs"]:
        b = probe["per_block"][str(bi)]
        print(
            f"   {bi:3d}  |  {b['summary']['mean_functional_flat_cos']:.4f}  |  "
            f"{b['summary']['mean_functional_per_token_cos']:.4f}"
        )
    print()

    if gap is not None:
        print("T5 gap at each block (cosine in cross-attn output space):")
        print("  block | t5 ↔ inv_runs | inv_runs pairwise | gap (t5 - inv)")
        print("  ------|---------------|-------------------|---------------")
        for bi in probe["block_idxs"]:
            g = gap[str(bi)]
            t5v = g["mean_cos_t5_vs_inv_runs"]
            ivv = g["mean_cos_inv_runs_pairwise"]
            d = g["gap"]
            t5s = f"{t5v:.4f}" if t5v is not None else "  n/a "
            ivs = f"{ivv:.4f}" if ivv is not None else "  n/a "
            ds = f"{d:+.4f}" if d is not None else "  n/a "
            print(f"   {bi:3d}  |    {t5s}     |      {ivs}       |    {ds}")
        print()
        print(
            "Interpretation: more negative gap → T5 is FURTHER from inversions"
        )
        print(
            "than inversions are from each other → bigger 'enhancement headroom'"
        )
        print("at that depth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
