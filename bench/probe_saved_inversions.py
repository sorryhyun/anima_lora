#!/usr/bin/env python
"""Probe functional-space agreement between saved inversion embeddings.

Loads the per-run inversions produced by `--save_per_run` plus the aligned mean,
encodes a target image with the VAE, then calls
`scripts.invert_embedding.probe_functional_space` with a fixed (noise, sigma)
probe bank. Writes the probe block into an existing `<stem>_alignment.json`
(or prints to stdout if that file doesn't exist).

Usage:
    python bench/probe_saved_inversions.py \\
        --results_dir inversions_probe_test/results \\
        --logs_dir inversions_probe_test/logs \\
        --image post_image_dataset/10811132.png \\
        --dit models/diffusion_models/anima-preview3-base.safetensors \\
        --vae models/vae/qwen_image_vae.safetensors
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library.anima import weights as anima_utils
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.log import setup_logging
from scripts.invert_embedding import probe_functional_space

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--logs_dir", default=None)
    p.add_argument("--image", required=True, help="Target image used for inversion")
    p.add_argument("--dit", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--probe_samples", type=int, default=4)
    p.add_argument(
        "--probe_blocks",
        type=str,
        default="0",
        help="Comma-separated block indices, e.g. '0,4,8,12,16,20,24,27'",
    )
    p.add_argument("--vae_chunk_size", type=int, default=64)
    return p.parse_args()


def load_embeddings(results_dir, stem):
    """Find and load per-run and mean embeddings for a given stem."""
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
        labeled.append((f"run{idx}", emb.float()))
    mean_path = os.path.join(results_dir, f"{stem}_inverted.safetensors")
    if os.path.exists(mean_path):
        sd = load_file(mean_path)
        emb = sd["crossattn_emb"]
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        labeled.append(("mean", emb.float()))
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    labeled = load_embeddings(args.results_dir, stem)
    if not labeled:
        logger.error(f"No saved embeddings for stem '{stem}' in {args.results_dir}")
        return 1
    logger.info(f"Loaded {len(labeled)} embeddings: {[lbl for lbl, _ in labeled]}")

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

    class _A:
        pass

    probe_args = _A()
    block_idxs = [int(s) for s in args.probe_blocks.split(",") if s.strip()]
    probe = probe_functional_space(
        probe_args,
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

    logs_dir = args.logs_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.results_dir)), "logs"
    )
    diag_path = os.path.join(logs_dir, f"{stem}_alignment.json")
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            payload = json.load(f)
        payload["functional_probe"] = probe
        with open(diag_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Updated {diag_path}")
    else:
        print(json.dumps(probe, indent=2))

    print("\n=== Functional probe — depth sweep ===")
    print(f"labels:    {probe['labels']}")
    print(f"n_probes:  {probe['n_probes']}")
    print(f"raw flat cos (reference): {probe['raw_summary']['mean_raw_flat_cos']:.4f}")
    print("\nblock | flat cos | per-token cos")
    print("------|----------|-------------")
    for bi in probe["block_idxs"]:
        b = probe["per_block"][str(bi)]
        print(
            f"  {bi:3d} | {b['summary']['mean_functional_flat_cos']:.4f}   | "
            f"{b['summary']['mean_functional_per_token_cos']:.4f}"
        )
    print("\nPer-block per-pair flat cos:")
    print(
        "  pair             | " + " | ".join(f"b{bi:02d}" for bi in probe["block_idxs"])
    )
    pairs = list(
        probe["per_block"][str(probe["block_idxs"][0])]["pairwise_cos_flat"].keys()
    )
    for pair in pairs:
        row = [pair.ljust(16)]
        for bi in probe["block_idxs"]:
            row.append(
                f"{probe['per_block'][str(bi)]['pairwise_cos_flat'].get(pair, float('nan')):.3f}"
            )
        print(" | ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
