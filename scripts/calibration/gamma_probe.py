#!/usr/bin/env python3
"""CNS Phase-0 γ-matrix probe (read-only) — arXiv 2605.30332, Algorithm 2.

Measures the *completion matrix* γ(f, t) for Anima: how resolved each radial
frequency band is at each sampling step. CNS (Colored Noise Sampling) recolors
the SDE-injected noise by sqrt(1−γ) so the fixed stochastic-energy budget lands
in the bands the network has NOT yet built. This probe checks the staircase is
sharp on Anima. See _archive/bench/cns/plan.md.

What it does (no engine edits):
  1. Drive deterministic ODE (euler) generations through the real pipeline via
     GenerationRequest (sampler defaults to "euler").
  2. Wrap `library.inference.sampling.step` (the Euler ODE step, called as
     `inference_utils.step` in generation.py) to record, per step, the clean
     prediction x0_pred = latents − σ_i · v.  Zero changes to generation.py.
  3. γ(f,t) = 1 − |X_pred(f,t) − X₀(f)|² / |X₀(f)|², radially binned over the
     latent (H, W) FFT grid, averaged over channels and trajectories.
  4. Report staircase sharpness + emit beta_preview = the exact per-step colored
     scale CNS would apply (Eq. 11).

Run from repo root (anima_lora/):
    python scripts/calibration/gamma_probe.py --steps 28 --cfg 1.0
    python scripts/calibration/gamma_probe.py --cfg 4.0 --size 1248 832
    # probe the adapter (does the LoRA flatten the bias?):
    python scripts/calibration/gamma_probe.py --extra --lora_weight output/ckpt/<x>.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
)
from library.inference import sampling as inference_utils

_ckpts = default_checkpoints()
DIT, VAE, TEXT_ENCODER = _ckpts.dit, _ckpts.vae, _ckpts.text_encoder

# Varied content so γ reflects the model, not one prompt's spectrum.
DEFAULT_PROMPTS = [
    "1girl, detailed face, intricate kimono, cherry blossoms, masterpiece",
    "a sprawling cyberpunk city at night, neon signs, rain, wide shot",
    "a calm watercolor landscape, rolling hills, soft gradients, minimal detail",
    "a close-up portrait of an old fisherman, weathered skin, sharp texture",
]


# --------------------------------------------------------------------------- #
# Per-step capture via a wrapped Euler step.
# --------------------------------------------------------------------------- #
class _StepCapture:
    """Monkeypatch `inference_utils.step` to record x0_pred per ODE step.

    generation.py calls `inference_utils.step(latents, noise_pred, sigmas, i)`
    only on the euler path (er_sde is None). We compute the clean prediction
    x0_pred = latents − sigmas[i]·noise_pred, store it on CPU fp16, then delegate
    to the real step so the trajectory is unchanged.
    """

    def __init__(self) -> None:
        self._orig = inference_utils.step
        self.x0_preds: list[np.ndarray] = []
        self.sigmas: np.ndarray | None = None

    def __enter__(self) -> "_StepCapture":
        def _wrapped(latents, noise_pred, sigmas, step_i):
            x0 = latents.float() - float(sigmas[step_i]) * noise_pred.float()
            # (1, C, 1, H, W) -> (C, H, W)
            self.x0_preds.append(x0.squeeze(0).squeeze(1).half().cpu().numpy())
            if self.sigmas is None:
                self.sigmas = sigmas.float().cpu().numpy()
            return self._orig(latents, noise_pred, sigmas, step_i)

        inference_utils.step = _wrapped
        return self

    def __exit__(self, *exc) -> None:
        inference_utils.step = self._orig


# --------------------------------------------------------------------------- #
# γ computation.
# --------------------------------------------------------------------------- #
def _radial_bins(h: int, w: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Radial-frequency bin index per FFT cell + bin centers in [0, 1]."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy**2 + fx**2)
    r = r / r.max()  # normalize so Nyquist-ish corner -> ~1
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return idx, centers


def _gamma_for_trajectory(
    x0_preds: list[np.ndarray], x0_final: np.ndarray, bin_idx: np.ndarray, n_bins: int
) -> np.ndarray:
    """γ[T, F] for one trajectory. x0_* are (C, H, W) arrays."""
    Xf = np.fft.fft2(x0_final.astype(np.float32), axes=(-2, -1))
    pf = np.abs(Xf) ** 2  # (C, H, W)
    eps = 1e-8 * pf.mean(axis=(-2, -1), keepdims=True) + 1e-12
    T = len(x0_preds)
    out = np.zeros((T, n_bins), dtype=np.float64)
    flat_idx = bin_idx.reshape(-1)
    counts = np.bincount(flat_idx, minlength=n_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    for t, xp in enumerate(x0_preds):
        Xp = np.fft.fft2(xp.astype(np.float32), axes=(-2, -1))
        g = 1.0 - (np.abs(Xp - Xf) ** 2) / (pf + eps)  # (C, H, W)
        g = np.clip(g.mean(axis=0), 0.0, 1.0)  # avg channels -> (H, W)
        sums = np.bincount(flat_idx, weights=g.reshape(-1), minlength=n_bins)
        out[t] = sums / counts
    return out


def _sigma_at_gamma_half(gamma_col: np.ndarray, sigma_mid: np.ndarray) -> float:
    """σ where this freq band crosses γ=0.5 (linear interp). sigma_mid decreasing."""
    above = gamma_col >= 0.5
    if not above.any():
        return 0.0  # never resolves -> resolves at σ→0 end
    if above.all():
        return float(sigma_mid[0])
    i = int(np.argmax(above))  # first step at/above 0.5
    if i == 0:
        return float(sigma_mid[0])
    g0, g1 = gamma_col[i - 1], gamma_col[i]
    s0, s1 = sigma_mid[i - 1], sigma_mid[i]
    frac = (0.5 - g0) / (g1 - g0 + 1e-12)
    return float(s0 + frac * (s1 - s0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=28, help="ODE sampling steps.")
    p.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="Guidance scale (1.0 = clean single-forward ODE).",
    )
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="One prompt per line; overrides defaults.",
    )
    p.add_argument("--n_bins", type=int, default=32, help="Radial frequency bins (F).")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("output/calibration/cns_gamma_probe"),
        help="Directory for the probe's gamma.npz + heatmaps + metrics.",
    )
    p.add_argument("--dit", type=str, default=DIT)
    p.add_argument("--vae", type=str, default=VAE)
    p.add_argument("--text_encoder", type=str, default=TEXT_ENCODER)
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Verbatim extra inference flags (e.g. --lora_weight <path>). Must be LAST.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = (
        [
            ln.strip()
            for ln in Path(args.prompts_file).read_text().splitlines()
            if ln.strip()
        ]
        if args.prompts_file
        else DEFAULT_PROMPTS
    )
    H, W = args.size
    h_lat, w_lat = H // 8, W // 8
    bin_idx, centers = _radial_bins(h_lat, w_lat, args.n_bins)

    per_traj: list[np.ndarray] = []
    sigmas_ref: np.ndarray | None = None
    n_ok = 0
    for prompt in prompts:
        for seed in args.seeds:
            req = GenerationRequest(
                dit=args.dit,
                vae=args.vae,
                text_encoder=args.text_encoder,
                prompt=prompt,
                image_size=(H, W),
                infer_steps=args.steps,
                guidance_scale=args.cfg,
                flow_shift=args.flow_shift,
                sampler="euler",
                seed=seed,
                output_type="latent",
                extra_argv=tuple(args.extra),
            )
            gen_args = req.to_args()
            gen_args.device = device
            settings = get_generation_settings(gen_args)
            with _StepCapture() as cap, torch.no_grad():
                latent = generate(gen_args, settings)
            if len(cap.x0_preds) != args.steps:
                print(
                    f"  ! captured {len(cap.x0_preds)} steps (expected {args.steps}); skipping"
                )
                continue
            x0_final = latent.float().squeeze(0).squeeze(1).cpu().numpy()  # (C, H, W)
            g = _gamma_for_trajectory(cap.x0_preds, x0_final, bin_idx, args.n_bins)
            per_traj.append(g)
            sigmas_ref = cap.sigmas if sigmas_ref is None else sigmas_ref
            n_ok += 1
            print(f"  ok: seed={seed} '{prompt[:40]}...'  ({n_ok} trajectories)")

    if not per_traj:
        raise SystemExit("No trajectories captured — check model paths / flags.")

    gamma = np.mean(np.stack(per_traj, 0), 0)  # [T, F]
    sigma_mid = sigmas_ref[:-1]  # σ fed to the model at each step (decreasing)

    # Eq. 11: colored scale CNS would apply, RMS-normalized per step across freq.
    one_minus = np.clip(1.0 - gamma, 0.0, 1.0)
    rms = np.sqrt(np.maximum(one_minus.mean(axis=1, keepdims=True), 1e-12))
    beta_preview = np.sqrt(one_minus) / rms  # [T, F]

    # Linear training target γ_target = 1 − σ (paper App. C.1). Deviation = bias.
    gamma_linear = np.clip(1.0 - sigma_mid[:, None], 0.0, 1.0) * np.ones_like(gamma)
    linear_mae = float(np.abs(gamma - gamma_linear).mean())

    # Staircase sharpness: σ at γ=0.5 per freq band; low-freq should cross at
    # HIGH σ (early), high-freq at LOW σ (late). Spread > 0 ⇒ staircase present.
    s50 = np.array(
        [_sigma_at_gamma_half(gamma[:, f], sigma_mid) for f in range(args.n_bins)]
    )
    lo = int(args.n_bins * 0.15)
    hi = int(args.n_bins * 0.85)
    s50_low = float(np.nanmean(s50[: lo + 1]))
    s50_high = float(np.nanmean(s50[hi:]))
    s50_spread = s50_low - s50_high
    aggregate_s50 = _sigma_at_gamma_half(gamma.mean(axis=1), sigma_mid)

    run_dir = args.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "gamma.npz",
        gamma=gamma,
        beta_preview=beta_preview,
        sigmas=sigmas_ref,
        timesteps=sigma_mid,
        radial_centers=centers,
        gamma_linear_target=gamma_linear,
    )

    artifacts = ["gamma.npz"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        ext = [centers[0], centers[-1], sigma_mid[-1], sigma_mid[0]]
        im0 = ax[0].imshow(
            gamma,
            aspect="auto",
            origin="lower",
            extent=ext,
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax[0].set(
            title="γ(f, t)  completion", xlabel="radial freq f", ylabel="σ (→0 = done)"
        )
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(
            beta_preview, aspect="auto", origin="lower", extent=ext, cmap="magma"
        )
        ax[1].set(
            title="β preview = sqrt(1−γ), RMS-norm (CNS scale)",
            xlabel="radial freq f",
            ylabel="σ",
        )
        fig.colorbar(im1, ax=ax[1])
        ax[2].plot(centers, s50, ".-")
        ax[2].set(
            title=f"σ@γ=0.5 vs freq  (spread={s50_spread:+.3f})",
            xlabel="radial freq f",
            ylabel="σ at γ=0.5",
        )
        ax[2].invert_yaxis()
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "gamma_matrix.png", dpi=110)
        artifacts.append("gamma_matrix.png")
    except Exception as e:  # plotting is best-effort
        print(f"  (plot skipped: {e})")

    metrics = {
        "n_trajectories": n_ok,
        "steps": args.steps,
        "cfg": args.cfg,
        "size_hw": [H, W],
        "n_bins": args.n_bins,
        "staircase_s50_spread": s50_spread,  # >0 ⇒ low-freq resolves earlier (CNS premise)
        "s50_lowfreq": s50_low,
        "s50_highfreq": s50_high,
        "aggregate_s50": aggregate_s50,  # cross-check vs project_sigma_signal (~0.45)
        "linear_target_mae": linear_mae,  # bigger ⇒ stronger spectral bias to exploit
    }
    metrics["device"] = str(device)
    metrics["artifacts"] = artifacts
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n=== CNS γ-probe ===")
    print(f"  trajectories       : {n_ok}")
    print(
        f"  staircase σ50 spread: {s50_spread:+.3f}  (low {s50_low:.3f} → high {s50_high:.3f})"
    )
    print(
        f"  aggregate σ50       : {aggregate_s50:.3f}   (cf. project_sigma_signal ≈ 0.45)"
    )
    print(f"  linear-target MAE   : {linear_mae:.3f}   (↑ = more bias to exploit)")
    print(f"  → {run_dir}")
    verdict = (
        "GO premise"
        if (s50_spread > 0.08 and linear_mae > 0.05)
        else "WEAK — staircase shallow"
    )
    print(f"  Phase-0 read       : {verdict}")


if __name__ == "__main__":
    main()
