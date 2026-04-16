#!/usr/bin/env python
"""V3/V4 — Modulation sensitivity probe and injection point comparison.

V3: Perturb emb_B_T_D with projected quality direction, measure output sensitivity.
V4: Compare three injection points (before norm, after norm, into adaln_lora_B_T_3D).

Requires bench/quality_direction.pt and bench/test_embed.pt from V1/V2/V5.

Run from anima_lora/:
    python bench/v3v4_modulation_probe.py                  # latent-space analysis
    python bench/v3v4_modulation_probe.py --generate        # also generate images
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import anima_models, anima_utils
from library.inference import sampling as inference_utils
from library.device_utils import clean_memory_on_device

DIT_PATH = "models/diffusion_models/anima-preview3-base.safetensors"
VAE_PATH = "models/vae/qwen_image_vae.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
PROBE_TIMESTEPS = [0.1, 0.3, 0.5, 0.7, 0.9]  # representative timestep values


def build_random_projection(in_dim: int, out_dim: int, seed: int = 42) -> torch.Tensor:
    """Fixed random projection matrix (stand-in for untrained pooled_text_proj)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g) / (in_dim**0.5)
    return W


def single_forward(
    model: anima_models.Anima,
    latents: torch.Tensor,
    timestep: float,
    embed: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a single forward pass (no grad) and return noise prediction."""
    t = torch.tensor([timestep], device=latents.device, dtype=latents.dtype)
    with torch.no_grad():
        return model(latents, t, embed, padding_mask=padding_mask)


def probe_injection(
    model: anima_models.Anima,
    latents: torch.Tensor,
    embed: torch.Tensor,
    padding_mask: torch.Tensor,
    delta_2048: torch.Tensor,
    delta_6144: torch.Tensor,
    injection_point: str,
    alphas: list[float],
    timesteps: list[float],
) -> dict:
    """Measure output MSE at multiple alpha values and timesteps for one injection point.

    injection_point: "before_norm" | "after_norm" | "adaln_lora"
    Returns: {timestep: {alpha: mse, ...}, ...}
    """
    results = {}

    for ts in timesteps:
        t_tensor = torch.tensor([ts], device=DEVICE, dtype=torch.bfloat16)

        # Baseline (alpha=0)
        with torch.no_grad():
            baseline = model(latents, t_tensor, embed, padding_mask=padding_mask)

        mses = {}
        for alpha in alphas:
            if alpha == 0.0:
                mses[alpha] = 0.0
                continue

            delta = delta_2048 if injection_point != "adaln_lora" else delta_6144
            scaled = (alpha * delta).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

            # Register hook for injection
            if injection_point == "before_norm":
                # Hook on TimestepEmbedding (t_embedder[1]) output: modify emb before norm
                def hook_fn(module, input, output, _s=scaled):
                    emb, adaln_lora = output
                    return emb + _s.to(emb.dtype), adaln_lora

                handle = model.t_embedder[1].register_forward_hook(hook_fn)

            elif injection_point == "after_norm":
                # Hook on t_embedding_norm output: modify emb after norm
                def hook_fn(module, input, output, _s=scaled):
                    return output + _s.to(output.dtype)

                handle = model.t_embedding_norm.register_forward_hook(hook_fn)

            elif injection_point == "adaln_lora":
                # Hook on TimestepEmbedding output: modify adaln_lora_B_T_3D
                def hook_fn(module, input, output, _s=scaled):
                    emb, adaln_lora = output
                    return emb, adaln_lora + _s.to(adaln_lora.dtype)

                handle = model.t_embedder[1].register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    perturbed = model(
                        latents, t_tensor, embed, padding_mask=padding_mask
                    )
                mse = F.mse_loss(perturbed.float(), baseline.float()).item()
                mses[alpha] = mse
            finally:
                handle.remove()

        results[ts] = mses

    return results


def run_full_generation(
    model: anima_models.Anima,
    embed: torch.Tensor,
    neg_embed: torch.Tensor,
    delta_2048: torch.Tensor,
    alpha: float,
    injection_point: str,
    seed: int = 42,
    steps: int = 20,
    guidance_scale: float = 4.0,
    height: int = 1024,
    width: int = 1024,
) -> torch.Tensor:
    """Run full denoising loop with optional modulation perturbation. Returns latent."""
    seed_g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, 16, 1, height // 8, width // 8)
    latents = torch.randn(shape, generator=seed_g, dtype=torch.bfloat16).to(DEVICE)
    padding_mask = torch.zeros(
        1, 1, height // 8, width // 8, device=DEVICE, dtype=torch.bfloat16
    )

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(steps, 1.0, DEVICE)
    timesteps = (timesteps / 1000).to(DEVICE, dtype=torch.bfloat16)

    # Install persistent hook if alpha > 0
    handle = None
    if alpha > 0.0:
        scaled = (alpha * delta_2048).unsqueeze(0).unsqueeze(0)
        if injection_point == "after_norm":

            def hook_fn(module, input, output, _s=scaled):
                return output + _s.to(output.dtype)

            handle = model.t_embedding_norm.register_forward_hook(hook_fn)
        elif injection_point == "before_norm":

            def hook_fn(module, input, output, _s=scaled):
                emb, adaln = output
                return emb + _s.to(emb.dtype), adaln

            handle = model.t_embedder[1].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for i, t in enumerate(
                tqdm(timesteps, desc=f"α={alpha:.1f} ({injection_point})", leave=False)
            ):
                t_expand = t.unsqueeze(0)
                noise_pred = model(latents, t_expand, embed, padding_mask=padding_mask)
                uncond_pred = model(
                    latents, t_expand, neg_embed, padding_mask=padding_mask
                )
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
                latents = inference_utils.step(latents, noise_pred, sigmas, i).to(
                    latents.dtype
                )
    finally:
        if handle is not None:
            handle.remove()

    return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate", action="store_true", help="Also generate full images (slow)"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Inference steps for image generation"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load artifacts from V1/V2/V5
    qdir_path = BENCH_DIR / "quality_direction.pt"
    temb_path = BENCH_DIR / "test_embed.pt"
    if not qdir_path.exists() or not temb_path.exists():
        print("ERROR: Run v1v2v5_embedding_analysis.py first to generate artifacts.")
        return 1

    quality_dir = torch.load(qdir_path, weights_only=True)  # (1024,)
    test_data = torch.load(temb_path, weights_only=True)
    test_embed = test_data["test_embed"]  # (1, 512, 1024)
    neg_embed = test_data["neg_embed"]  # (1, 512, 1024)

    print("=" * 60)
    print("Modulation Guidance — Sensitivity Probe (V3/V4)")
    print("=" * 60)

    # Build random projections
    proj_2048 = build_random_projection(1024, 2048)
    proj_6144 = build_random_projection(1024, 6144, seed=43)
    delta_2048 = (
        (quality_dir.float() @ proj_2048).to(torch.bfloat16).to(DEVICE)
    )  # (2048,)
    delta_6144 = (
        (quality_dir.float() @ proj_6144).to(torch.bfloat16).to(DEVICE)
    )  # (6144,)

    # Normalize deltas so alpha is interpretable
    delta_2048 = delta_2048 / delta_2048.norm()
    delta_6144 = delta_6144 / delta_6144.norm()

    # Load DiT
    print("\nLoading DiT model...")
    model = anima_utils.load_anima_model(
        device=DEVICE,
        dit_path=DIT_PATH,
        attn_mode="torch",
        split_attn=False,
        loading_device=DEVICE,
        dit_weight_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False)

    # Prepare fixed inputs
    latents = torch.randn(
        1, 16, 1, 128 // 8, 128 // 8, device=DEVICE, dtype=torch.bfloat16
    )
    torch.manual_seed(42)  # for reproducibility
    padding_mask = torch.zeros(
        1, 1, 128 // 8, 128 // 8, device=DEVICE, dtype=torch.bfloat16
    )
    embed = test_embed.to(DEVICE, dtype=torch.bfloat16)

    # ==============================================================
    # V3: Modulation sensitivity (after_norm injection)
    # ==============================================================
    print("\n" + "=" * 60)
    print("V3: Modulation sensitivity probe")
    print("=" * 60)

    v3_results = probe_injection(
        model,
        latents,
        embed,
        padding_mask,
        delta_2048,
        delta_6144,
        "after_norm",
        ALPHAS,
        PROBE_TIMESTEPS,
    )

    # Check: does MSE increase with alpha?
    avg_sensitivity = {}
    for ts, mses in v3_results.items():
        print(f"\n  Timestep {ts}:")
        for alpha, mse in sorted(mses.items()):
            print(f"    α={alpha:5.2f}  MSE={mse:.6e}")
        # Sensitivity = MSE at alpha=4.0 (or max alpha tested)
        avg_sensitivity[ts] = mses[ALPHAS[-1]]

    overall_sensitivity = float(np.mean(list(avg_sensitivity.values())))
    v3_pass = overall_sensitivity > 1e-8  # perturbations cause measurable change

    print(f"\n  Overall sensitivity (MSE @ α={ALPHAS[-1]}): {overall_sensitivity:.6e}")
    print(
        f"  VERDICT: {'PASS' if v3_pass else 'FAIL'} — output {'is' if v3_pass else 'is NOT'} sensitive to emb perturbations"
    )

    # Check smoothness: MSE should increase monotonically with alpha
    monotonic_count = 0
    total_ts = 0
    for ts, mses in v3_results.items():
        vals = [mses[a] for a in sorted(mses.keys())]
        is_mono = all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))
        monotonic_count += int(is_mono)
        total_ts += 1
    smoothness = monotonic_count / total_ts
    print(f"  Smoothness (fraction monotonic): {smoothness:.2f}")

    # ==============================================================
    # V4: Injection point comparison
    # ==============================================================
    print("\n" + "=" * 60)
    print("V4: Injection point comparison")
    print("=" * 60)

    injection_points = ["before_norm", "after_norm", "adaln_lora"]
    v4_results = {}

    for ip in injection_points:
        print(f"\n  Probing: {ip}")
        r = probe_injection(
            model,
            latents,
            embed,
            padding_mask,
            delta_2048,
            delta_6144,
            ip,
            ALPHAS,
            [0.5],  # single representative timestep
        )
        v4_results[ip] = r

        mses = r[0.5]
        for alpha, mse in sorted(mses.items()):
            print(f"    α={alpha:5.2f}  MSE={mse:.6e}")

    # Compare sensitivity at a mid-range alpha
    ref_alpha = 2.0
    print(f"\n  Sensitivity comparison (MSE @ α={ref_alpha}, t=0.5):")
    best_ip = None
    best_sens = 0
    for ip in injection_points:
        mse = v4_results[ip][0.5][ref_alpha]
        print(f"    {ip:15s}: {mse:.6e}")
        if mse > best_sens:
            best_sens = mse
            best_ip = ip

    # Check stability: does output collapse at high alpha?
    print("\n  Stability check (MSE growth rate α=4→8):")
    stable_ips = []
    for ip in injection_points:
        m4 = v4_results[ip][0.5][4.0]
        m8 = v4_results[ip][0.5][8.0]
        growth = m8 / (m4 + 1e-12)
        is_stable = growth < 10  # less than 10x growth = graceful
        stable_ips.append(is_stable)
        print(
            f"    {ip:15s}: growth={growth:.2f}x {'(stable)' if is_stable else '(collapse)'}"
        )

    v4_best = best_ip
    print(f"\n  Best injection point: {v4_best} (highest sensitivity)")

    # ==============================================================
    # Optional: full image generation
    # ==============================================================
    if args.generate:
        print("\n" + "=" * 60)
        print("Generating images for visual comparison")
        print("=" * 60)

        neg = neg_embed.to(DEVICE, dtype=torch.bfloat16)
        gen_embed = test_embed.to(DEVICE, dtype=torch.bfloat16)
        gen_alphas = [0.0, 1.0, 4.0]

        from library import qwen_image_autoencoder_kl
        from library.inference.output import decode_latent
        from PIL import Image

        all_latents = {}
        for alpha in gen_alphas:
            lat = run_full_generation(
                model,
                gen_embed,
                neg,
                delta_2048,
                alpha=alpha,
                injection_point="after_norm",
                seed=42,
                steps=args.steps,
            )
            all_latents[alpha] = lat.cpu()

        # Free DiT, load VAE
        del model
        gc.collect()
        clean_memory_on_device(DEVICE)

        print("Loading VAE...")
        vae = qwen_image_autoencoder_kl.load_vae(
            VAE_PATH, spatial_chunk_size=64, disable_cache=True
        )
        vae.to(DEVICE, dtype=torch.bfloat16).eval()

        img_dir = RESULTS_DIR / "v3_images"
        img_dir.mkdir(exist_ok=True)
        for alpha, lat in all_latents.items():
            pixels = decode_latent(vae, lat, DEVICE)
            img = Image.fromarray(
                ((pixels.permute(1, 2, 0).float().numpy() + 1) / 2 * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            img.save(img_dir / f"alpha_{alpha:.1f}.png")
            print(f"  Saved: {img_dir}/alpha_{alpha:.1f}.png")

        del vae
        gc.collect()
        clean_memory_on_device(DEVICE)

    # ==============================================================
    # Save results
    # ==============================================================
    # Convert results to JSON-serializable format
    def jsonify(d):
        if isinstance(d, dict):
            return {str(k): jsonify(v) for k, v in d.items()}
        if isinstance(d, (np.floating, float)):
            return float(d)
        return d

    results = {
        "v3": {
            "injection_point": "after_norm",
            "alphas": ALPHAS,
            "timesteps": PROBE_TIMESTEPS,
            "mse_by_timestep": jsonify(v3_results),
            "overall_sensitivity": overall_sensitivity,
            "smoothness": smoothness,
            "pass": v3_pass,
        },
        "v4": {
            "alpha": ref_alpha,
            "timestep": 0.5,
            "sensitivity_by_injection": {
                ip: v4_results[ip][0.5][ref_alpha] for ip in injection_points
            },
            "best_injection_point": v4_best,
            "stability": {ip: bool(s) for ip, s in zip(injection_points, stable_ips)},
        },
    }

    out_path = RESULTS_DIR / "v3v4_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY:  V3={'PASS' if v3_pass else 'FAIL'}  V4: best={v4_best}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
