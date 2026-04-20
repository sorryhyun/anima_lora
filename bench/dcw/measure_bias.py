#!/usr/bin/env python
"""Direct SNR-t bias measurement for Anima (flow-matching DiT).

Reproduces Fig. 1c of Yu et al. *Elucidating the SNR-t Bias of DPMs*
(CVPR 2026, arXiv:2604.16044) on Anima and uses the *measured gap* as the
tuning objective for the DCW post-step correction. This is the most
principled way to pick DCW's scaler/schedule: DCW exists to close this gap,
so we measure the gap directly and pick the knob that closes it the most.

Consumes cached samples in ``post_image_dataset/`` — both latents
(``*_anima.npz``) and post-LLMAdapter text embeds (``*_anima_te.safetensors``)
are already precomputed there, so no VAE or text-encoder loading is needed.

What it measures
----------------
For each timestep ``i`` in the inference schedule:

    v_fwd(i) = || v_θ(x_t_fwd, t_i) ||           # forward-noised from cached x_0
    v_rev(i) = || v_θ(x_hat_i, t_i) ||           # reverse-sampled from noise
    gap(i)   = v_rev(i) − v_fwd(i)

Paper's Key Finding 2: for all i, gap(i) > 0 (reverse samples have lower SNR
than forward samples at the same timestep, so the network — which is
SNR-monotone in its prediction norm per Key Finding 1 — outputs a larger
velocity norm). A correctly-tuned DCW should *reduce* the integrated gap.
Over-tuned DCW flips the sign.

Modes
-----
- **Diagnostic (default)**: baseline only. Answers "does the bias exist
  on Anima at our scale?" — a precondition before investing in DCW.
- **Sweep** (``--dcw_sweep``): also runs reverse trajectories with DCW
  correction at a grid of ``(scaler, schedule)`` pairs. Picks the winner
  by integrated |gap(i)| over steps.

Schedule forms (pixel-mode DCW):
    const             scaler(i) = λ
    sigma_i           scaler(i) = λ · σ_i            (paper Eq. 20)
    one_minus_sigma   scaler(i) = λ · (1 − σ_i)      (inverse)

Outputs (bench/dcw/results/<timestamp>/)
----------------------------------------
    config.json            CLI args
    summary.json           integrated gap + |gap| per DCW config, ranked
    per_step.csv           wide: step, v_fwd, v_rev, gap per config
    gap_curves.png         matplotlib Fig 1c reproduction + gap overlay

Usage
-----
    # Diagnostic (~3 min on 4090 for 4 cached samples × 2 seeds × 20 steps)
    uv run python bench/dcw/measure_bias.py \
        --dit models/diffusion_models/anima-preview3-base.safetensors

    # Full sweep — pixel-mode.
    #
    # Anima-specific note (2026-04-20 baseline, shift=1.0, 28 steps):
    # the measured gap is predominantly NEGATIVE (reverse velocity < forward) —
    # opposite sign from Yu et al. Fig 1c. Paper-form positive λ would widen
    # |gap|; closing it requires NEGATIVE λ. The grid below sweeps both sides
    # (positive values serve as a direction-sanity-check — they should make
    # |gap| worse on Anima). `one_minus_sigma` concentrates correction at low
    # σ, where |gap| is largest.
    uv run python bench/dcw/measure_bias.py \
        --dit models/... \
        --dcw_sweep \
        --dcw_scalers -0.3 -0.1 -0.03 -0.01 0.0 0.01 0.1 \
        --dcw_schedules one_minus_sigma const
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.anima import weights as anima_utils
from library.inference import sampling as inference_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dcw-bench")

# post_image_dataset/<stem>_<HHHH>x<WWWW>_anima.npz  (latent is C×H×W, already H/8×W/8)
_LATENT_RE = re.compile(r"^(?P<stem>.+)_(?P<h>\d+)x(?P<w>\d+)_anima\.npz$")


# ------------------------------------------------------------------
# DCW correction (inlined; see plan.md for production version)
# ------------------------------------------------------------------


def dcw_scaler(lam: float, sigma_i: float, schedule: str) -> float:
    if schedule == "const":
        return lam
    if schedule == "sigma_i":
        return lam * sigma_i
    if schedule == "one_minus_sigma":
        return lam * (1.0 - sigma_i)
    raise ValueError(f"unknown schedule: {schedule}")


def apply_dcw_pixel(prev: torch.Tensor, x0_pred: torch.Tensor, s: float) -> torch.Tensor:
    """Eq. 17 (pixel-space): prev += s · (prev − x0_pred)."""
    if s == 0.0:
        return prev
    return prev + s * (prev - x0_pred)


# ------------------------------------------------------------------
# Cache loading: one (latent, crossattn_emb) pair per sample
# ------------------------------------------------------------------


def pick_cached_samples(dataset_dir: Path, n: int, text_variant: int) -> list[tuple[str, Path, Path]]:
    """Return list of (stem, latent_npz_path, text_safetensors_path)."""
    out = []
    for npz_path in sorted(dataset_dir.glob("*_anima.npz")):
        m = _LATENT_RE.match(npz_path.name)
        if not m:
            continue
        stem = m.group("stem")
        te_path = dataset_dir / f"{stem}_anima_te.safetensors"
        if not te_path.exists():
            continue
        out.append((stem, npz_path, te_path))
        if len(out) >= n:
            break
    return out


def load_cached(
    npz_path: Path, te_path: Path, text_variant: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_0 as (1,16,1,H,W) bfloat16, embed as (1,512,1024) bfloat16)."""
    with np.load(npz_path) as z:
        latent_keys = [k for k in z.keys() if k.startswith("latents_")]
        if not latent_keys:
            raise RuntimeError(f"no latents_* key in {npz_path}")
        lat = torch.from_numpy(z[latent_keys[0]])  # (16, H, W) float32
    x_0 = lat.unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)  # (1, 16, 1, H, W)

    sd = load_file(str(te_path))
    key = f"crossattn_emb_v{text_variant}"
    if key not in sd:
        raise KeyError(f"{key} not in {te_path}; available: {[k for k in sd if k.startswith('crossattn_emb_')]}")
    embed = sd[key].to(device, dtype=torch.bfloat16).unsqueeze(0)  # (1, 512, 1024)
    return x_0, embed


# ------------------------------------------------------------------
# Core measurement: one reverse trajectory with paired forward-noise evals
# ------------------------------------------------------------------


@torch.no_grad()
def run_trajectory(
    anima,
    x_0: torch.Tensor,              # (1, 16, 1, H, W) bf16
    embed: torch.Tensor,            # (1, 512, D) bf16
    sigmas: torch.Tensor,           # (num_steps+1,) float32
    *,
    noise_seed: int,
    dcw_lam: float,
    dcw_schedule: str,
    device: torch.device,
) -> dict:
    """Run reverse sampling + paired forward-noise measurement at each step.

    Forward-noise RNG is re-seeded identically across DCW configs → the
    forward branch is bit-identical across configs → any change in the gap
    is attributable to DCW changing the reverse trajectory alone.
    """
    h_lat, w_lat = x_0.shape[-2], x_0.shape[-1]
    num_steps = len(sigmas) - 1

    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    g_init = torch.Generator(device="cpu").manual_seed(noise_seed)
    g_fwd = torch.Generator(device="cpu").manual_seed(noise_seed + 10_000)

    eps_init = torch.randn(x_0.shape, generator=g_init).to(device, torch.bfloat16)
    x_hat = eps_init.clone()  # reverse trajectory starts at σ=1

    v_fwd_norms = np.zeros(num_steps, dtype=np.float64)
    v_rev_norms = np.zeros(num_steps, dtype=np.float64)

    for i in range(num_steps):
        sigma_i = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        t_i = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)

        # Forward branch
        eps_fwd = torch.randn(x_0.shape, generator=g_fwd).to(device, torch.bfloat16)
        x_t_fwd = (1.0 - sigma_i) * x_0 + sigma_i * eps_fwd
        v_fwd = anima(x_t_fwd, t_i, embed, padding_mask=padding_mask)
        v_fwd_norms[i] = v_fwd.float().flatten().norm().item()

        # Reverse branch: evaluate velocity on current x_hat
        v_rev = anima(x_hat, t_i, embed, padding_mask=padding_mask)
        v_rev_norms[i] = v_rev.float().flatten().norm().item()

        # Euler step + optional DCW correction
        prev = x_hat.float() + (sigma_next - sigma_i) * v_rev.float()
        if dcw_lam != 0.0 and sigma_next > 0.0:
            x0_pred = x_hat.float() - sigma_i * v_rev.float()
            s = dcw_scaler(dcw_lam, sigma_i, dcw_schedule)
            prev = apply_dcw_pixel(prev, x0_pred, s)
        x_hat = prev.to(torch.bfloat16)

    return dict(v_fwd=v_fwd_norms, v_rev=v_rev_norms, gap=v_rev_norms - v_fwd_norms)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit", type=str, required=True, help="DiT .safetensors path")
    p.add_argument("--dataset_dir", type=str, default="post_image_dataset",
                   help="Directory with cached *_anima.npz + *_anima_te.safetensors pairs")
    p.add_argument("--text_variant", type=int, default=0,
                   help="Which cached caption variant to use (crossattn_emb_v<N>); 0 = canonical")
    p.add_argument("--attn_mode", type=str, default="flash",
                   help="torch | sdpa | xformers | sage | flash ")
    p.add_argument("--n_images", type=int, default=6, help="Number of cached samples to use")
    p.add_argument("--n_seeds", type=int, default=3, help="Seeds per sample")
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=1.0, help="Sigma shift (matches inference.py default)")
    p.add_argument("--seed_base", type=int, default=1234)
    # DCW sweep
    p.add_argument("--dcw_sweep", action="store_true", help="Also run DCW-corrected trajectories")
    # Default grid sweeps both signs: Anima's baseline gap is negative, so we
    # expect negative λ to close it and positive λ to widen it. Including a
    # couple of positive values acts as a direction-sanity check that the
    # correction and the measurement agree on sign.
    p.add_argument("--dcw_scalers", type=float, nargs="+",
                   default=[-0.3, -0.1, -0.03, -0.01, 0.0, 0.01, 0.1],
                   help="λ values to sweep (negative expected on Anima; see docstring)")
    p.add_argument("--dcw_schedules", type=str, nargs="+", default=["one_minus_sigma", "const"],
                   choices=["const", "sigma_i", "one_minus_sigma"], help="Schedule forms to sweep")
    default_out = Path(__file__).resolve().parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    p.add_argument("--output", type=str, default=str(default_out), help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log.info(f"output → {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # -------- DiT only (latents + crossattn_emb are cached) --------
    log.info("loading DiT…")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.eval().requires_grad_(False)

    # -------- Pick data --------
    samples = pick_cached_samples(Path(args.dataset_dir), args.n_images, args.text_variant)
    if not samples:
        raise SystemExit(
            f"no cached samples found under {args.dataset_dir}. "
            "Expected *_anima.npz + *_anima_te.safetensors pairs (from make preprocess)."
        )
    log.info(f"using {len(samples)} cached samples (variant v{args.text_variant})")

    # -------- Schedule (same math as inference.py) --------
    _, sigmas_t = inference_utils.get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigmas = sigmas_t.cpu()
    num_steps = args.infer_steps
    log.info(f"infer_steps={num_steps}, flow_shift={args.flow_shift}, σ₀={float(sigmas[0]):.3f}, σₙ={float(sigmas[-1]):.3f}")

    # -------- DCW configs --------
    configs: list[tuple[str, float, str]] = [("baseline", 0.0, "const")]
    if args.dcw_sweep:
        for sched in args.dcw_schedules:
            for lam in args.dcw_scalers:
                if lam == 0.0:
                    continue  # already covered by baseline
                configs.append((f"λ={lam}_{sched}", lam, sched))
    log.info(f"{len(configs)} configs × {len(samples)} samples × {args.n_seeds} seeds "
             f"= {len(configs) * len(samples) * args.n_seeds} trajectories")

    # -------- Preload cached data onto device --------
    log.info("loading cached latents + text embeds…")
    encoded = []
    for stem, npz_path, te_path in samples:
        x_0, embed = load_cached(npz_path, te_path, args.text_variant, device)
        encoded.append((stem, x_0, embed))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------- Sweep trajectories --------
    accum = {
        name: {"v_fwd": np.zeros(num_steps), "v_rev": np.zeros(num_steps),
               "gap": np.zeros(num_steps), "n": 0}
        for name, _, _ in configs
    }

    pbar = tqdm(total=len(configs) * len(samples) * args.n_seeds, desc="trajectories")
    t0 = time.time()
    for name, lam, sched in configs:
        for img_idx, (stem, x_0, embed) in enumerate(encoded):
            for seed_idx in range(args.n_seeds):
                seed = args.seed_base + 1000 * img_idx + seed_idx
                res = run_trajectory(
                    anima, x_0, embed, sigmas,
                    noise_seed=seed, dcw_lam=lam, dcw_schedule=sched, device=device,
                )
                for k in ("v_fwd", "v_rev", "gap"):
                    accum[name][k] += res[k]
                accum[name]["n"] += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{name} stem={stem} seed={seed}")
    pbar.close()
    log.info(f"done in {time.time() - t0:.0f}s")

    for name in accum:
        n = accum[name]["n"]
        for k in ("v_fwd", "v_rev", "gap"):
            accum[name][k] = accum[name][k] / n

    # -------- Summary --------
    ranked = sorted(
        [(name, float(np.abs(accum[name]["gap"]).sum()), float(accum[name]["gap"].sum()))
         for name in accum],
        key=lambda t: t[1],
    )
    summary = {
        "infer_steps": num_steps,
        "n_samples": len(samples),
        "n_seeds": args.n_seeds,
        "text_variant": args.text_variant,
        "configs_ranked_by_integrated_abs_gap": [
            {"config": name, "integrated_abs_gap": a, "integrated_signed_gap": s}
            for name, a, s in ranked
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # -------- CSV --------
    csv_path = out_dir / "per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["step", "sigma_i"]
        for name in accum:
            headers += [f"{name}_v_fwd", f"{name}_v_rev", f"{name}_gap"]
        w.writerow(headers)
        for i in range(num_steps):
            row = [i, float(sigmas[i])]
            for name in accum:
                row += [accum[name]["v_fwd"][i], accum[name]["v_rev"][i], accum[name]["gap"][i]]
            w.writerow(row)
    log.info(f"CSV → {csv_path}")

    # -------- Plot --------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

        base = accum["baseline"]
        axes[0].plot(range(num_steps), base["v_fwd"], label="forward ||v(x_t, t)||", color="#2a9d8f")
        axes[0].plot(range(num_steps), base["v_rev"], label="reverse ||v(x̂_t, t)||", color="#e76f51")
        axes[0].set_title("Baseline: forward vs reverse velocity norm (Fig 1c reproduction)")
        axes[0].set_xlabel("step i")
        axes[0].set_ylabel("||v||₂")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for name in accum:
            axes[1].plot(range(num_steps), accum[name]["gap"], label=name, alpha=0.85)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_title("gap(i) = ||v_rev|| − ||v_fwd||  (closer to 0 = better)")
        axes[1].set_xlabel("step i")
        axes[1].set_ylabel("gap")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        png_path = out_dir / "gap_curves.png"
        fig.savefig(png_path, dpi=130)
        log.info(f"plot → {png_path}")
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")

    print("\n=== SNR-t bias measurement ===")
    print(f"baseline integrated signed gap: {accum['baseline']['gap'].sum():+.3f}  "
          f"(>0 means reverse > forward, as the paper predicts)")
    if args.dcw_sweep:
        print("\nconfigs ranked by integrated |gap|  (smaller = closer alignment):")
        for rank, (name, a, s) in enumerate(ranked, 1):
            print(f"  {rank:>2}. {name:<30s}  |gap|={a:7.3f}  signed={s:+7.3f}")


if __name__ == "__main__":
    main()
