#!/usr/bin/env python
"""Inversion stability & multi-seed averaging benchmark.

Tests the hypothesis from inversion_idea.md: optimizing text embedding across
multiple z_T (noise seeds) and extracting the common component yields a more
robust "z_T-invariant semantic embedding."

Three experiments:
  E1: Optimization stability — run inversion N times with different RNG seeds
      on the same image, measure pairwise cosine similarity of optimized embeddings.
  E2: Generation consistency — take each inverted embedding + the mean embedding,
      generate images with M fresh seeds, measure LPIPS/cosine between pairs.
  E3: Mean vs individual — compare generation quality of mean embedding vs
      individual embeddings using cross-seed consistency as a proxy for robustness.

Run from anima_lora/:
    python bench/inversion_stability.py --image_dir post_image_dataset --num_inversions 5
    python bench/inversion_stability.py --image_dir post_image_dataset --num_inversions 5 --skip_generation
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import anima_models, anima_utils, inference_utils, qwen_image_autoencoder_kl
from library.device_utils import clean_memory_on_device

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
DIT_PATH = "models/diffusion_models/anima-preview3-base.safetensors"
VAE_PATH = "models/vae/qwen_image_vae.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Inversion core (adapted from scripts/invert_embedding.py)
# ---------------------------------------------------------------------------


def sample_sigmas(n, device, mode="uniform"):
    if mode == "sigmoid":
        return torch.sigmoid(torch.randn(n, device=device))
    return torch.rand(n, device=device)


def inversion_step(anima, latents, embed_bf16, sigmas, padding_mask):
    n_t = sigmas.shape[0]
    lat = latents.expand(n_t, -1, -1, -1)
    noise = torch.randn_like(lat)
    sv = sigmas.view(-1, 1, 1, 1)
    noisy = ((1.0 - sv) * lat + sv * noise).to(torch.bfloat16).unsqueeze(2)
    pred = anima(noisy, sigmas.to(torch.bfloat16), embed_bf16.expand(n_t, -1, -1),
                 padding_mask=padding_mask.expand(n_t, -1, -1, -1))
    target = noise - lat
    return F.mse_loss(pred.squeeze(2).float(), target.float())


def run_inversion(anima, latents, init_embed, device, *,
                  steps=100, lr=0.01, grad_accum=4, timesteps_per_step=1, seed=0):
    """Single inversion run. Returns (best_embed, best_loss, loss_curve)."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    embed = torch.nn.Parameter(init_embed.clone())
    opt = torch.optim.AdamW([embed], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    pm = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    best_loss = float("inf")
    best_embed = None
    losses = []

    pbar = tqdm(range(steps), desc=f"  seed={seed}", leave=False)
    for _step in pbar:
        opt.zero_grad()
        accum = 0.0
        for _ in range(grad_accum):
            sigmas = sample_sigmas(timesteps_per_step, device)
            loss = inversion_step(anima, latents, embed.to(torch.bfloat16), sigmas, pm)
            (loss / grad_accum).backward()
            accum += loss.item()
        torch.nn.utils.clip_grad_norm_([embed], max_norm=1.0)
        opt.step()
        sched.step()
        val = accum / grad_accum
        losses.append(val)
        if val < best_loss:
            best_loss = val
            best_embed = embed.detach().clone()
        pbar.set_postfix(loss=f"{val:.5f}", best=f"{best_loss:.5f}")

    return best_embed, best_loss, losses


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_from_embed(anima, vae, embed, h, w, device, *, seed=42, steps=50, flow_shift=5.0):
    """Generate a single image. Returns pixel tensor (C, H, W) in [0,1]."""
    embed_bf16 = embed.to(device=device, dtype=torch.bfloat16)
    if embed_bf16.ndim == 2:
        embed_bf16 = embed_bf16.unsqueeze(0)

    h_lat, w_lat = h // 8, w // 8
    pm = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        1, anima_models.Anima.LATENT_CHANNELS, 1, h_lat, w_lat,
        device=device, dtype=torch.bfloat16, generator=gen,
    )

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(steps, flow_shift, device)
    timesteps = (timesteps / 1000).to(device, dtype=torch.bfloat16)

    if hasattr(anima, "switch_block_swap_for_inference"):
        anima.switch_block_swap_for_inference()

    with torch.no_grad():
        for step_i, t in enumerate(timesteps):
            if hasattr(anima, "prepare_block_swap_before_forward"):
                anima.prepare_block_swap_before_forward()
            noise_pred = anima(latents, t.unsqueeze(0), embed_bf16, padding_mask=pm)
            latents = inference_utils.step(latents, noise_pred, sigmas, step_i).to(torch.bfloat16)

    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.squeeze(2))
    return ((pixels + 1.0) / 2.0).clamp(0, 1).squeeze(0).float().cpu()  # (C, H, W)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cosine_sim_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def cosine_sim_per_token(a: torch.Tensor, b: torch.Tensor) -> float:
    """Per-token cosine sim averaged across sequence (captures token-level structure)."""
    if a.ndim == 3:
        a, b = a.squeeze(0), b.squeeze(0)
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def pairwise_cosine_matrix(embeddings: list[torch.Tensor]) -> np.ndarray:
    """NxN pairwise cosine similarity matrix."""
    n = len(embeddings)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_sim_flat(embeddings[i], embeddings[j])
            mat[i, j] = mat[j, i] = s
    return mat


def pixel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a.float(), b.float()).item()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_one_image(image_dir: str):
    """Find first image with cached latents in directory."""
    import glob
    import os
    for png_path in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        stem = os.path.splitext(png_path)[0]
        npz_files = glob.glob(f"{stem}_*_anima.npz")
        if not npz_files:
            continue
        te_path = f"{stem}_anima_te.safetensors"
        return {
            "image_path": png_path,
            "npz_path": npz_files[0],
            "te_path": te_path if os.path.exists(te_path) else None,
            "stem": os.path.basename(stem),
        }
    return None


def load_cached_latents(npz_path, device):
    data = np.load(npz_path)
    latent_key = [k for k in data.keys() if k.startswith("latents_")][0]
    latents = torch.from_numpy(data[latent_key]).unsqueeze(0).to(device, dtype=torch.bfloat16)
    size_suffix = latent_key[len("latents_"):]
    size_key = f"original_size_{size_suffix}"
    if size_key in data:
        orig_w, orig_h = int(data[size_key][0]), int(data[size_key][1])
    else:
        orig_h, orig_w = latents.shape[-2] * 8, latents.shape[-1] * 8
    return latents, orig_h, orig_w


def load_cached_embedding(te_path, device):
    if te_path is None:
        return None
    sd = load_file(te_path)
    key = "crossattn_emb_v0" if "crossattn_emb_v0" in sd else "crossattn_emb"
    if key not in sd:
        return None
    return sd[key].unsqueeze(0).to(device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_e1(anima, latents, init_embed, device, args):
    """E1: Optimization stability — N inversions with different seeds."""
    print("\n" + "=" * 60)
    print("E1: Optimization stability across random seeds")
    print("=" * 60)

    embeddings = []
    losses = []
    for i in range(args.num_inversions):
        seed = args.base_seed + i * 1000
        print(f"  Inversion {i+1}/{args.num_inversions} (seed={seed})...")
        best_embed, best_loss, curve = run_inversion(
            anima, latents, init_embed, device,
            steps=args.steps, lr=args.lr, grad_accum=args.grad_accum,
            timesteps_per_step=args.timesteps_per_step, seed=seed,
        )
        embeddings.append(best_embed.cpu())
        losses.append(best_loss)
        print(f"    best_loss={best_loss:.6f}")

    # Pairwise cosine similarity
    cos_mat = pairwise_cosine_matrix(embeddings)
    upper_tri = cos_mat[np.triu_indices_from(cos_mat, k=1)]
    per_token_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            per_token_sims.append(cosine_sim_per_token(embeddings[i], embeddings[j]))

    # Aggregated embeddings
    stacked_embeds = torch.stack(embeddings)
    mean_embed = stacked_embeds.mean(dim=0)
    max_embed = stacked_embeds.max(dim=0).values

    # Distance from each individual to mean
    dists_to_mean = [cosine_sim_flat(e, mean_embed) for e in embeddings]

    print(f"\n  Pairwise cosine similarity (flattened):")
    print(f"    mean={np.mean(upper_tri):.4f}  std={np.std(upper_tri):.4f}  "
          f"min={np.min(upper_tri):.4f}  max={np.max(upper_tri):.4f}")
    print(f"  Per-token cosine similarity:")
    print(f"    mean={np.mean(per_token_sims):.4f}  std={np.std(per_token_sims):.4f}")
    print(f"  Cosine sim to mean embedding:")
    print(f"    mean={np.mean(dists_to_mean):.4f}  std={np.std(dists_to_mean):.4f}")
    print(f"  Loss values: {[f'{l:.6f}' for l in losses]}")

    # PCA of variation
    stacked = torch.stack(embeddings).view(len(embeddings), -1).float()
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    if len(embeddings) > 2:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        variance_explained = (S ** 2) / (S ** 2).sum()
        print(f"  PCA variance explained (top 5): {variance_explained[:5].tolist()}")
    else:
        variance_explained = None

    return {
        "embeddings": embeddings,
        "mean_embed": mean_embed,
        "max_embed": max_embed,
        "losses": losses,
        "pairwise_cos_sim": {
            "mean": float(np.mean(upper_tri)),
            "std": float(np.std(upper_tri)),
            "min": float(np.min(upper_tri)),
            "max": float(np.max(upper_tri)),
        },
        "per_token_cos_sim": {
            "mean": float(np.mean(per_token_sims)),
            "std": float(np.std(per_token_sims)),
        },
        "cos_sim_to_mean": {
            "mean": float(np.mean(dists_to_mean)),
            "std": float(np.std(dists_to_mean)),
        },
        "pca_variance_explained": variance_explained[:5].tolist() if variance_explained is not None else None,
    }


def run_e2(anima, vae, e1_results, h, w, device, args):
    """E2: Generation consistency — generate from each embedding + mean with M seeds."""
    print("\n" + "=" * 60)
    print("E2: Generation consistency across seeds")
    print("=" * 60)

    gen_seeds = [args.base_seed + i for i in range(args.num_gen_seeds)]
    embeddings_to_test = list(e1_results["embeddings"]) + [e1_results["mean_embed"], e1_results["max_embed"]]
    labels = [f"inv_{i}" for i in range(len(e1_results["embeddings"]))] + ["mean", "max"]

    all_images = {}  # label -> list of (C,H,W) tensors

    for label, embed in zip(labels, embeddings_to_test):
        print(f"  Generating for {label}...")
        images = []
        for seed in gen_seeds:
            img = generate_from_embed(
                anima, vae, embed, h, w, device,
                seed=seed, steps=args.gen_steps, flow_shift=args.flow_shift,
            )
            images.append(img)
        all_images[label] = images

    # Save sample images
    out_dir = RESULTS_DIR / "inversion_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image as PILImage
    for label, images in all_images.items():
        for i, img in enumerate(images):
            pixels = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            PILImage.fromarray(pixels).save(out_dir / f"{label}_seed{gen_seeds[i]}.png")

    # Measure within-embedding consistency (same embed, different seeds)
    consistency = {}
    for label, images in all_images.items():
        mse_vals = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                mse_vals.append(pixel_mse(images[i], images[j]))
        consistency[label] = {
            "pixel_mse_mean": float(np.mean(mse_vals)),
            "pixel_mse_std": float(np.std(mse_vals)),
        }
        print(f"  {label}: cross-seed pixel MSE = {np.mean(mse_vals):.6f} +/- {np.std(mse_vals):.6f}")

    # Measure cross-embedding similarity (different embeds, same seed)
    cross_embed = {}
    for si, seed in enumerate(gen_seeds):
        inv_images = [all_images[f"inv_{i}"][si] for i in range(len(e1_results["embeddings"]))]
        mean_image = all_images["mean"][si]
        # inv vs inv
        inv_mses = []
        for i in range(len(inv_images)):
            for j in range(i + 1, len(inv_images)):
                inv_mses.append(pixel_mse(inv_images[i], inv_images[j]))
        # inv vs mean
        mean_mses = [pixel_mse(img, mean_image) for img in inv_images]
        cross_embed[f"seed_{seed}"] = {
            "inv_vs_inv_mse": float(np.mean(inv_mses)) if inv_mses else 0.0,
            "inv_vs_mean_mse": float(np.mean(mean_mses)),
        }

    print(f"\n  Cross-embedding (same seed):")
    for k, v in cross_embed.items():
        print(f"    {k}: inv↔inv MSE={v['inv_vs_inv_mse']:.6f}  inv↔mean MSE={v['inv_vs_mean_mse']:.6f}")

    return {
        "within_embed_consistency": consistency,
        "cross_embed_similarity": cross_embed,
    }


def run_e3(e1_results, e2_results, args):
    """E3: Summary — are aggregated embeddings more robust?"""
    print("\n" + "=" * 60)
    print("E3: Aggregated embedding robustness summary")
    print("=" * 60)

    wc = e2_results["within_embed_consistency"]

    # Compare cross-seed consistency: aggregations vs individual embeddings
    inv_consistencies = []
    for i in range(len(e1_results["embeddings"])):
        label = f"inv_{i}"
        if label in wc:
            inv_consistencies.append(wc[label]["pixel_mse_mean"])

    avg_inv = float(np.mean(inv_consistencies)) if inv_consistencies else None
    mean_mse = wc.get("mean", {}).get("pixel_mse_mean", None)
    max_mse = wc.get("max", {}).get("pixel_mse_mean", None)

    print(f"  Avg individual cross-seed MSE: {avg_inv:.6f}" if avg_inv is not None else "  Avg individual: N/A")
    print(f"  Mean embed cross-seed MSE:     {mean_mse:.6f}" if mean_mse is not None else "  Mean embed: N/A")
    print(f"  Max  embed cross-seed MSE:     {max_mse:.6f}" if max_mse is not None else "  Max embed: N/A")

    results = {"avg_individual_mse": avg_inv}
    for name, val in [("mean", mean_mse), ("max", max_mse)]:
        if avg_inv is not None and val is not None:
            improvement = (avg_inv - val) / avg_inv * 100
            verdict = "MORE ROBUST" if val <= avg_inv else "LESS ROBUST"
            print(f"  {name:4s} improvement: {improvement:+.1f}% → {verdict}")
            results[f"{name}_embed_mse"] = val
            results[f"{name}_improvement_pct"] = improvement
            results[f"{name}_verdict"] = verdict

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Inversion stability & multi-seed averaging benchmark")
    p.add_argument("--image_dir", type=str, default="post_image_dataset", help="Dataset dir with cached latents/TE")
    p.add_argument("--image_stem", type=str, default=None, help="Specific image stem (default: first found)")
    p.add_argument("--dit", type=str, default=DIT_PATH)
    p.add_argument("--vae", type=str, default=VAE_PATH)
    p.add_argument("--attn_mode", type=str, default="torch")

    # Inversion params
    p.add_argument("--num_inversions", type=int, default=5, help="Number of independent inversions (E1)")
    p.add_argument("--steps", type=int, default=100, help="Optimization steps per inversion")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--timesteps_per_step", type=int, default=1)
    p.add_argument("--base_seed", type=int, default=42)

    # Generation params
    p.add_argument("--skip_generation", action="store_true", help="Skip E2/E3 (generation experiments)")
    p.add_argument("--num_gen_seeds", type=int, default=3, help="Seeds per embedding for generation (E2)")
    p.add_argument("--gen_steps", type=int, default=50, help="Denoising steps for generation")
    p.add_argument("--flow_shift", type=float, default=5.0)

    # Resume
    p.add_argument("--resume", action="store_true", help="Skip E1, load saved embeddings and run E2/E3 only")

    # VRAM
    p.add_argument("--blocks_to_swap", type=int, default=0)

    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Inversion Stability & Multi-Seed Averaging Benchmark")
    print("=" * 60)

    # ---- Find target image ----
    if args.image_stem:
        import glob, os
        stem_base = os.path.join(args.image_dir, args.image_stem)
        npz_files = glob.glob(f"{stem_base}_*_anima.npz")
        te_path = f"{stem_base}_anima_te.safetensors"
        img_info = {
            "stem": args.image_stem,
            "npz_path": npz_files[0],
            "te_path": te_path if os.path.exists(te_path) else None,
        }
    else:
        img_info = discover_one_image(args.image_dir)
    if img_info is None:
        print(f"ERROR: No images with cached latents found in {args.image_dir}")
        return 1

    print(f"Target image: {img_info['stem']}")

    # ---- Load latents & init embedding ----
    latents, orig_h, orig_w = load_cached_latents(img_info["npz_path"], DEVICE)
    print(f"Latent shape: {latents.shape}, image size: {orig_h}x{orig_w}")

    init_embed = load_cached_embedding(img_info["te_path"], DEVICE)
    if init_embed is None:
        print("No cached TE found, initializing from zeros")
        init_embed = torch.zeros(1, 512, 1024, dtype=torch.float32, device=DEVICE)
    else:
        print(f"Init embedding from cached TE: {img_info['te_path']}")

    # ---- E1: Optimization stability (or resume) ----
    embed_save_path = RESULTS_DIR / "inversion_stability_embeds.pt"

    if args.resume:
        print(f"\nResuming from {embed_save_path}")
        saved = torch.load(embed_save_path, map_location="cpu", weights_only=True)
        embeds = saved["embeddings"]
        stacked = torch.stack(embeds)
        e1 = {
            "embeddings": embeds,
            "mean_embed": stacked.mean(dim=0),
            "max_embed": stacked.max(dim=0).values,
            "losses": saved["losses"],
        }
        print(f"  Loaded {len(e1['embeddings'])} embeddings, losses: {e1['losses']}")
    else:
        # ---- Load DiT ----
        print("\nLoading DiT...")
        is_swapping = args.blocks_to_swap > 0
        anima = anima_utils.load_anima_model(
            device="cpu" if is_swapping else DEVICE,
            dit_path=args.dit,
            attn_mode=args.attn_mode,
            split_attn=True,
            loading_device="cpu" if is_swapping else DEVICE,
            dit_weight_dtype=torch.bfloat16,
        )
        anima.to(torch.bfloat16)
        anima.requires_grad_(False)
        anima.split_attn = False

        if is_swapping:
            anima.enable_block_swap(args.blocks_to_swap, DEVICE)
            anima.move_to_device_except_swap_blocks(DEVICE)
            anima.prepare_block_swap_before_forward()
        else:
            anima.to(DEVICE)
            # Enable gradient checkpointing for inversion (saves VRAM)
            anima.enable_gradient_checkpointing()
            for block in anima.blocks:
                block.train()

        e1 = run_e1(anima, latents, init_embed, DEVICE, args)

        # Save E1 embeddings for later analysis
        torch.save({
            "embeddings": e1["embeddings"],
            "mean_embed": e1["mean_embed"],
            "losses": e1["losses"],
            "stem": img_info["stem"],
        }, embed_save_path)

        if args.skip_generation:
            print("\nSkipping E2/E3 (--skip_generation)")
            results = {"e1": {k: v for k, v in e1.items() if k not in ("embeddings", "mean_embed")}}
            out_path = RESULTS_DIR / "inversion_stability.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved → {out_path}")
            return 0

        # Switch to inference mode (undo train mode from grad checkpointing)
        anima.eval()
        del anima
        clean_memory_on_device(DEVICE)

    # ---- Load DiT in inference mode for E2 ----
    print("\nLoading DiT for inference...")
    is_swapping = args.blocks_to_swap > 0
    anima = anima_utils.load_anima_model(
        device="cpu" if is_swapping else DEVICE,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device="cpu" if is_swapping else DEVICE,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)
    anima.eval()
    if is_swapping:
        anima.enable_block_swap(args.blocks_to_swap, DEVICE)
        anima.move_to_device_except_swap_blocks(DEVICE)
    else:
        anima.to(DEVICE)
        anima = torch.compile(anima)

    # ---- Load VAE for generation ----
    print("\nLoading VAE...")
    vae = qwen_image_autoencoder_kl.load_vae(VAE_PATH, device="cpu", disable_mmap=True, spatial_chunk_size=64)
    vae.to(DEVICE, dtype=torch.bfloat16)
    vae.eval()

    # ---- E2: Generation consistency ----
    e2 = run_e2(anima, vae, e1, orig_h, orig_w, DEVICE, args)

    # ---- E3: Mean embedding robustness ----
    e3 = run_e3(e1, e2, args)

    # ---- Save results ----
    results = {
        "image_stem": img_info["stem"],
        "config": {
            "num_inversions": args.num_inversions,
            "steps": args.steps,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "num_gen_seeds": args.num_gen_seeds,
            "gen_steps": args.gen_steps,
        },
        "e1": {k: v for k, v in e1.items() if k not in ("embeddings", "mean_embed", "max_embed")},
        "e2": e2,
        "e3": e3,
    }
    out_path = RESULTS_DIR / "inversion_stability.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved → {out_path}")
    print(f"Images saved → {RESULTS_DIR / 'inversion_stability/'}")
    print(f"Embeddings saved → {RESULTS_DIR / 'inversion_stability_embeds.pt'}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
