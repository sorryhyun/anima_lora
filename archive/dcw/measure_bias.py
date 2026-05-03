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

Outputs (bench/dcw/results/<YYYYMMDD-HHMM>[-<label>]/)
------------------------------------------------------
    result.json            standard envelope (args, git, env, metrics, artifacts)
    per_step.csv           wide: step, v_fwd, v_rev, gap per config
    gap_curves.png         matplotlib Fig 1c reproduction + gap overlay

Usage
-----
    # Diagnostic (~3 min on 4090 for 4 cached samples × 2 seeds × 20 steps)
    uv run python bench/dcw/measure_bias.py \
        --dit models/diffusion_models/anima-preview3-base.safetensors

    # Full sweep — pixel-mode. Default --dcw_scalers / --dcw_schedules
    # already encode the negative-λ + one_minus_sigma grid that Anima's
    # 2026-04-20 baseline points to (positive values stay in as a
    # direction-sanity check; they should make |gap| worse).
    uv run python bench/dcw/measure_bias.py \
        --dit models/diffusion_models/anima-preview3-base.safetensors \
        --dcw_sweep

    # Same sweep + decode the first 4 sweep-winner latents per config to PNG
    # (loads the VAE; ~+30 s).
    uv run python bench/dcw/measure_bias.py \
        --dit models/diffusion_models/anima-preview3-base.safetensors \
        --dcw_sweep --save_images 4
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result
from library.anima import weights as anima_utils
from library.inference import sampling as inference_utils
from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma
from library.inference.models import _is_hydra_moe

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


def apply_dcw_pixel(
    prev: torch.Tensor, x0_pred: torch.Tensor, s: float
) -> torch.Tensor:
    """Eq. 17 (pixel-space): prev += s · (prev − x0_pred)."""
    if s == 0.0:
        return prev
    return prev + s * (prev - x0_pred)


# ------------------------------------------------------------------
# Cache loading: one (latent, crossattn_emb) pair per sample
# ------------------------------------------------------------------


def pick_cached_samples(
    dataset_dir: Path, n: int, text_variant: int
) -> list[tuple[str, Path, Path]]:
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
    x_0 = (
        lat.unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)
    )  # (1, 16, 1, H, W)

    sd = load_file(str(te_path))
    key = f"crossattn_emb_v{text_variant}"
    if key not in sd:
        raise KeyError(
            f"{key} not in {te_path}; available: {[k for k in sd if k.startswith('crossattn_emb_')]}"
        )
    embed = sd[key].to(device, dtype=torch.bfloat16).unsqueeze(0)  # (1, 512, 1024)
    return x_0, embed


# ------------------------------------------------------------------
# Core measurement: one reverse trajectory with paired forward-noise evals
# ------------------------------------------------------------------


@torch.no_grad()
def _padding_mask(x_0: torch.Tensor, device: torch.device) -> torch.Tensor:
    h_lat, w_lat = x_0.shape[-2], x_0.shape[-1]
    return torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)


@torch.no_grad()
def measure_forward_norms(
    anima,
    x_0: torch.Tensor,  # (1, 16, 1, H, W) bf16
    embed: torch.Tensor,  # (1, 512, D) bf16
    sigmas: torch.Tensor,  # (num_steps+1,) float32
    *,
    noise_seed: int,
    device: torch.device,
) -> np.ndarray:
    """Forward branch only: ||v_θ((1−σ)x_0 + σε, t)|| at every step.

    Bit-identical across DCW configs given the same (x_0, noise_seed), so we
    compute it once per (image, seed) and cache it for every config below.
    """
    num_steps = len(sigmas) - 1
    padding_mask = _padding_mask(x_0, device)
    g_fwd = torch.Generator(device="cpu").manual_seed(noise_seed + 10_000)
    v_fwd_norms = np.zeros(num_steps, dtype=np.float64)
    for i in range(num_steps):
        sigma_i = float(sigmas[i])
        t_i = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)
        eps_fwd = torch.randn(x_0.shape, generator=g_fwd).to(device, torch.bfloat16)
        x_t_fwd = (1.0 - sigma_i) * x_0 + sigma_i * eps_fwd
        set_hydra_sigma(anima, t_i)
        v_fwd = anima(x_t_fwd, t_i, embed, padding_mask=padding_mask)
        v_fwd_norms[i] = v_fwd.float().flatten().norm().item()
    return v_fwd_norms


@torch.no_grad()
def run_reverse(
    anima,
    x_0: torch.Tensor,  # (1, 16, 1, H, W) bf16, used only for shape
    embed: torch.Tensor,  # (1, 512, D) bf16
    sigmas: torch.Tensor,  # (num_steps+1,) float32
    *,
    noise_seed: int,
    dcw_lam: float,
    dcw_schedule: str,
    device: torch.device,
) -> dict:
    """Reverse trajectory from σ=1 noise, with optional DCW correction.

    Returns ``v_rev`` (norm per step) and the final ``x_hat`` (cpu, bf16).
    """
    num_steps = len(sigmas) - 1
    padding_mask = _padding_mask(x_0, device)
    g_init = torch.Generator(device="cpu").manual_seed(noise_seed)
    eps_init = torch.randn(x_0.shape, generator=g_init).to(device, torch.bfloat16)
    x_hat = eps_init  # reverse trajectory starts at σ≈1

    v_rev_norms = np.zeros(num_steps, dtype=np.float64)
    for i in range(num_steps):
        sigma_i = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        t_i = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)

        set_hydra_sigma(anima, t_i)
        v_rev = anima(x_hat, t_i, embed, padding_mask=padding_mask)
        v_rev_norms[i] = v_rev.float().flatten().norm().item()

        prev = x_hat.float() + (sigma_next - sigma_i) * v_rev.float()
        if dcw_lam != 0.0 and sigma_next > 0.0:
            x0_pred = x_hat.float() - sigma_i * v_rev.float()
            s = dcw_scaler(dcw_lam, sigma_i, dcw_schedule)
            prev = apply_dcw_pixel(prev, x0_pred, s)
        x_hat = prev.to(torch.bfloat16)

    return dict(v_rev=v_rev_norms, x_hat=x_hat.detach().to("cpu"))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dit", type=str, required=True, help="DiT .safetensors path")
    p.add_argument(
        "--lora_weight",
        type=str,
        nargs="+",
        default=None,
        help="Optional LoRA / HydraLoRA adapter(s) to stack on the base DiT. "
        "Auto-detects HydraLoRA moe (lora_ups.* keys) and attaches router-live "
        "via dynamic forward hooks; plain LoRA goes through the same dynamic "
        "path (math-equivalent to static merge for this measurement).",
    )
    p.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="+",
        default=[1.0],
        help="Multiplier per --lora_weight entry (broadcast if a single value).",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="post_image_dataset/lora",
        help="Directory with cached *_anima.npz + *_anima_te.safetensors pairs "
        "(make preprocess writes here via subset cache_dir).",
    )
    p.add_argument(
        "--text_variant",
        type=int,
        default=0,
        help="Which cached caption variant to use (crossattn_emb_v<N>); 0 = canonical",
    )
    p.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="torch | sdpa | xformers | sage | flash ",
    )
    p.add_argument(
        "--n_images", type=int, default=4, help="Number of cached samples to use"
    )
    p.add_argument("--n_seeds", type=int, default=2, help="Seeds per sample")
    p.add_argument("--infer_steps", type=int, default=24)
    p.add_argument(
        "--flow_shift",
        type=float,
        default=1.0,
        help="Sigma shift (matches inference.py default)",
    )
    p.add_argument("--seed_base", type=int, default=1234)
    # DCW sweep
    p.add_argument(
        "--dcw_sweep", action="store_true", help="Also run DCW-corrected trajectories"
    )
    # Default grid sweeps both signs: Anima's baseline gap is negative, so we
    # expect negative λ to close it and positive λ to widen it. Including a
    # couple of positive values acts as a direction-sanity check that the
    # correction and the measurement agree on sign.
    p.add_argument(
        "--dcw_scalers",
        type=float,
        nargs="+",
        default=[-0.01, 0.01],
        help="λ values to sweep (negative expected on Anima; see docstring)",
    )
    p.add_argument(
        "--dcw_schedules",
        type=str,
        nargs="+",
        default=["one_minus_sigma", "const"],
        choices=["const", "sigma_i", "one_minus_sigma"],
        help="Schedule forms to sweep",
    )
    # Image decoding (optional eyeball check on sweep winners)
    p.add_argument(
        "--save_images",
        type=int,
        default=0,
        help="If >0, decode the final reverse-trajectory latent for the first "
        "N samples (first seed only) per config and save as PNG. Loads the VAE "
        "lazily; off by default to keep the bench DiT-only.",
    )
    p.add_argument(
        "--vae",
        type=str,
        default="models/vae/qwen_image_vae.safetensors",
        help="VAE path (only loaded when --save_images > 0).",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label appended to the run dir (bench/dcw/results/<ts>-<label>/).",
    )
    p.add_argument(
        "--report_optimal_lambda",
        action="store_true",
        help="Compute and print the (1−σ)-weighted closed-form λ* for each "
        "schedule in the sweep, plus per-step response slopes s_i = ∂gap/∂λ. "
        "Requires --dcw_sweep with ≥2 nonzero λ values per schedule.",
    )
    return p.parse_args()


def compute_optimal_lambda(
    accum: dict,
    configs: list[tuple[str, float, str]],
    sigmas: torch.Tensor,
    num_steps: int,
) -> dict:
    """Closed-form (1−σ)-weighted least-squares λ* per schedule.

    For each schedule s_kind, gathers anchor configs with that schedule
    (plus baseline at λ=0, which is schedule-agnostic), fits per-step slope
    s_i = ∂gap/∂λ via least-squares over the anchors, then applies

        λ* = − Σ w_i · g_i · s_i  /  Σ w_i · s_i²        (i ≥ N/2)

    with w_i = (1 − σ_i) and g_i = baseline gap. See plan.md §3.

    Returns a dict keyed by schedule name, each value:
      {anchors: [(λ, name)], slopes: np.ndarray (num_steps,),
       lambda_star: float, late_idx: (start, end)}
    """
    sigmas_np = sigmas.numpy()
    g_baseline = accum["baseline"]["gap"]
    late_start = num_steps // 2
    w = 1.0 - sigmas_np[:num_steps]

    by_schedule: dict[str, list[tuple[float, str]]] = {}
    for name, lam, sched in configs:
        if name == "baseline":
            continue
        by_schedule.setdefault(sched, []).append((lam, name))

    out: dict[str, dict] = {}
    for sched, anchors in by_schedule.items():
        # Include baseline (λ=0) in the fit — it's schedule-agnostic.
        pts = [(0.0, "baseline")] + sorted(anchors, key=lambda t: t[0])
        if len(pts) < 2:
            continue
        lam_arr = np.array([p[0] for p in pts], dtype=np.float64)
        gap_mat = np.stack([accum[p[1]]["gap"] for p in pts], axis=0)  # (A, num_steps)
        # Per-step least-squares slope: gap_i(λ) ≈ a + s_i · λ.
        lam_centered = lam_arr - lam_arr.mean()
        denom = float((lam_centered ** 2).sum())
        if denom == 0.0:
            continue
        slopes = (lam_centered[:, None] * (gap_mat - gap_mat.mean(axis=0))).sum(axis=0) / denom

        # λ* over the late half, weighted by (1−σ_i).
        w_late = w[late_start:]
        g_late = g_baseline[late_start:]
        s_late = slopes[late_start:]
        num = -float((w_late * g_late * s_late).sum())
        den = float((w_late * s_late ** 2).sum())
        lambda_star = num / den if den != 0.0 else float("nan")

        out[sched] = {
            "anchors": pts,
            "slopes": slopes,
            "lambda_star": lambda_star,
            "late_idx": (late_start, num_steps),
        }
    return out


def main() -> None:
    args = parse_args()
    out_dir = make_run_dir("dcw", label=args.label)
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
    # load_anima_model only moves checkpoint-resident submodules; the
    # non-persistent mod-guidance buffers (`_mod_guidance_delta` etc.) are
    # created on CPU during __init__ and need an explicit move. Production
    # inference does this in library/inference/models.py:108. Then
    # reset_mod_guidance() zeros them so the per-block addition is identity.
    anima.to(device, dtype=dtype)
    anima.reset_mod_guidance()
    anima.eval().requires_grad_(False)

    # -------- Optional LoRA / HydraLoRA stacking --------
    # Mirrors library/inference/models.py:152-198 (router-live attach via
    # dynamic forward hooks). Plain LoRA goes through the same path; with
    # multiplier=1.0 it is mathematically equivalent to a static merge for
    # the measurement here. Keeps the bench DiT-only loader untouched.
    if args.lora_weight:
        from networks import lora_anima

        hydra_flags = [_is_hydra_moe(p) for p in args.lora_weight]
        if any(hydra_flags) and not all(hydra_flags):
            raise SystemExit(
                "Mixing HydraLoRA moe files with regular LoRA files in --lora_weight "
                "is not supported (matches the inference-time restriction)."
            )
        any_hydra = any(hydra_flags)
        kind = "router-live HydraLoRA" if any_hydra else "LoRA"
        log.info(f"attaching {len(args.lora_weight)} adapter(s) as {kind} hooks…")

        mults = args.lora_multiplier
        if len(mults) == 1:
            mults = mults * len(args.lora_weight)
        if len(mults) != len(args.lora_weight):
            raise SystemExit(
                f"--lora_multiplier has {len(mults)} entries but --lora_weight has "
                f"{len(args.lora_weight)}. Pass one multiplier per weight, or one shared."
            )

        for path, mult in zip(args.lora_weight, mults):
            lora_sd = load_file(path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
            network, weights_sd = lora_anima.create_network_from_weights(
                multiplier=mult,
                file=None,
                ae=None,
                text_encoders=[],
                unet=anima,
                weights_sd=lora_sd,
                for_inference=True,
            )
            network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights_sd, strict=False)
            if info.unexpected_keys:
                log.warning(
                    f"{path}: unexpected keys (first 5): {info.unexpected_keys[:5]}"
                )
            if info.missing_keys:
                log.warning(
                    f"{path}: missing keys (first 5): {info.missing_keys[:5]}"
                )
            network.to(device, dtype=dtype)
            network.eval().requires_grad_(False)
            if any_hydra:
                hydra_networks = list(getattr(anima, "_hydra_networks", []))
                hydra_networks.append(network)
                anima._hydra_networks = hydra_networks
                anima._hydra_network = network
            log.info(
                f"  attached {path} (mult={mult}, modules={len(network.unet_loras)})"
            )

    # -------- Pick data --------
    samples = pick_cached_samples(
        Path(args.dataset_dir), args.n_images, args.text_variant
    )
    if not samples:
        raise SystemExit(
            f"no cached samples found under {args.dataset_dir}. "
            "Expected *_anima.npz + *_anima_te.safetensors pairs (from make preprocess)."
        )
    log.info(f"using {len(samples)} cached samples (variant v{args.text_variant})")

    # -------- Schedule (same math as inference.py) --------
    _, sigmas_t = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas_t.cpu()
    num_steps = args.infer_steps
    log.info(
        f"infer_steps={num_steps}, flow_shift={args.flow_shift}, σ₀={float(sigmas[0]):.3f}, σₙ={float(sigmas[-1]):.3f}"
    )

    # -------- DCW configs --------
    configs: list[tuple[str, float, str]] = [("baseline", 0.0, "const")]
    if args.dcw_sweep:
        for sched in args.dcw_schedules:
            for lam in args.dcw_scalers:
                if lam == 0.0:
                    continue  # already covered by baseline
                configs.append((f"λ={lam}_{sched}", lam, sched))
    n_fwd = len(samples) * args.n_seeds
    n_rev = len(configs) * n_fwd
    log.info(
        f"{len(configs)} configs × {len(samples)} samples × {args.n_seeds} seeds: "
        f"{n_fwd} forward-branch + {n_rev} reverse-branch trajectories "
        f"({n_fwd + n_rev} total; was {2 * n_rev} before fwd caching)"
    )

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
        name: {
            "v_fwd": np.zeros(num_steps),
            "v_rev": np.zeros(num_steps),
            "gap": np.zeros(num_steps),
            "n": 0,
        }
        for name, _, _ in configs
    }

    # (config_name, stem) → final x_hat (cpu, bf16) for the FIRST seed of the
    # first --save_images samples. Populated only when --save_images > 0.
    saved_latents: dict[tuple[str, str], torch.Tensor] = {}

    t0 = time.time()

    # -------- Phase 1: forward-branch norms (once per (img, seed)) --------
    fwd_cache: dict[tuple[int, int], np.ndarray] = {}
    pbar_fwd = tqdm(total=n_fwd, desc="fwd  ")
    for img_idx, (stem, x_0, embed) in enumerate(encoded):
        for seed_idx in range(args.n_seeds):
            seed = args.seed_base + 1000 * img_idx + seed_idx
            fwd_cache[(img_idx, seed_idx)] = measure_forward_norms(
                anima, x_0, embed, sigmas, noise_seed=seed, device=device
            )
            pbar_fwd.update(1)
            pbar_fwd.set_postfix_str(f"stem={stem} seed={seed}")
    pbar_fwd.close()

    # -------- Phase 2: reverse trajectories (per config) --------
    pbar_rev = tqdm(total=n_rev, desc="rev  ")
    for name, lam, sched in configs:
        for img_idx, (stem, x_0, embed) in enumerate(encoded):
            for seed_idx in range(args.n_seeds):
                seed = args.seed_base + 1000 * img_idx + seed_idx
                res = run_reverse(
                    anima,
                    x_0,
                    embed,
                    sigmas,
                    noise_seed=seed,
                    dcw_lam=lam,
                    dcw_schedule=sched,
                    device=device,
                )
                v_fwd = fwd_cache[(img_idx, seed_idx)]
                accum[name]["v_fwd"] += v_fwd
                accum[name]["v_rev"] += res["v_rev"]
                accum[name]["gap"] += res["v_rev"] - v_fwd
                accum[name]["n"] += 1
                if (
                    args.save_images > 0
                    and img_idx < args.save_images
                    and seed_idx == 0
                ):
                    saved_latents[(name, stem)] = res["x_hat"]
                pbar_rev.update(1)
                pbar_rev.set_postfix_str(f"{name} stem={stem} seed={seed}")
    pbar_rev.close()
    clear_hydra_sigma(anima)
    log.info(f"done in {time.time() - t0:.0f}s")

    for name in accum:
        n = accum[name]["n"]
        for k in ("v_fwd", "v_rev", "gap"):
            accum[name][k] = accum[name][k] / n

    # -------- Summary --------
    ranked = sorted(
        [
            (
                name,
                float(np.abs(accum[name]["gap"]).sum()),
                float(accum[name]["gap"].sum()),
            )
            for name in accum
        ],
        key=lambda t: t[1],
    )
    metrics = {
        "infer_steps": num_steps,
        "n_samples": len(samples),
        "n_seeds": args.n_seeds,
        "text_variant": args.text_variant,
        "configs_ranked_by_integrated_abs_gap": [
            {"config": name, "integrated_abs_gap": a, "integrated_signed_gap": s}
            for name, a, s in ranked
        ],
    }

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
                row += [
                    accum[name]["v_fwd"][i],
                    accum[name]["v_rev"][i],
                    accum[name]["gap"][i],
                ]
            w.writerow(row)
    log.info(f"CSV → {csv_path}")

    # -------- Plot --------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

        base = accum["baseline"]
        axes[0].plot(
            range(num_steps),
            base["v_fwd"],
            label="forward ||v(x_t, t)||",
            color="#2a9d8f",
        )
        axes[0].plot(
            range(num_steps),
            base["v_rev"],
            label="reverse ||v(x̂_t, t)||",
            color="#e76f51",
        )
        axes[0].set_title(
            "Baseline: forward vs reverse velocity norm (Fig 1c reproduction)"
        )
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
        plot_written = True
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        plot_written = False

    # -------- Optional image decode (sweep-winner eyeball) --------
    image_artifacts: list[str] = []
    if saved_latents:
        log.info(
            f"decoding {len(saved_latents)} latents (first {args.save_images} sample(s) × "
            f"{len(configs)} config(s), seed_idx=0)…"
        )
        # Free the DiT-side activations before bringing in the VAE.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from library.models import qwen_vae
        from library.inference import output as out_io
        from PIL import Image

        vae = qwen_vae.load_vae(args.vae, device="cpu", disable_mmap=True)
        vae.to(torch.bfloat16)
        vae.eval()

        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for (name, stem), latent in saved_latents.items():
            pixels = out_io.decode_latent(vae, latent, device)  # (C,H,W) f32 in [-1,1]
            x = torch.clamp(pixels, -1.0, 1.0)
            x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
            x = x.transpose(1, 2, 0)
            safe_name = name.replace("/", "_").replace(" ", "")
            png = img_dir / f"{stem}__{safe_name}.png"
            Image.fromarray(x).save(png)
            image_artifacts.append(f"images/{png.name}")

        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(f"images → {img_dir} ({len(image_artifacts)} files)")

    artifacts = (
        ["per_step.csv"]
        + (["gap_curves.png"] if plot_written else [])
        + image_artifacts
    )
    result_path = write_result(
        out_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )
    log.info(f"result → {result_path}")

    print("\n=== SNR-t bias measurement ===")
    print(
        f"baseline integrated signed gap: {accum['baseline']['gap'].sum():+.3f}  "
        f"(>0 means reverse > forward, as the paper predicts)"
    )
    if args.dcw_sweep:
        print("\nconfigs ranked by integrated |gap|  (smaller = closer alignment):")
        for rank, (name, a, s) in enumerate(ranked, 1):
            print(f"  {rank:>2}. {name:<30s}  |gap|={a:7.3f}  signed={s:+7.3f}")

    if args.report_optimal_lambda:
        if not args.dcw_sweep:
            print(
                "\n--report_optimal_lambda needs --dcw_sweep — skipping calibrator."
            )
        else:
            calib = compute_optimal_lambda(accum, configs, sigmas, num_steps)
            if not calib:
                print(
                    "\n--report_optimal_lambda: no schedule had ≥1 nonzero anchor; "
                    "add nonzero --dcw_scalers values."
                )
            else:
                print("\n=== closed-form λ* (late half, (1−σ)-weighted) ===")
                for sched, info in calib.items():
                    anchors_str = ", ".join(f"{lam:+g}" for lam, _ in info["anchors"])
                    a, b = info["late_idx"]
                    print(
                        f"  schedule={sched:<18s}  anchors=[{anchors_str}]  "
                        f"λ*={info['lambda_star']:+.4f}  (over steps {a}..{b - 1})"
                    )
                metrics["optimal_lambda"] = {
                    sched: {
                        "lambda_star": info["lambda_star"],
                        "anchors": [lam for lam, _ in info["anchors"]],
                        "slopes_per_step": info["slopes"].tolist(),
                        "late_step_range": list(info["late_idx"]),
                    }
                    for sched, info in calib.items()
                }
                # Re-emit result.json with the calibrator block included.
                write_result(
                    out_dir,
                    script=__file__,
                    args=args,
                    label=args.label,
                    metrics=metrics,
                    artifacts=artifacts,
                    device=device,
                )


if __name__ == "__main__":
    main()
