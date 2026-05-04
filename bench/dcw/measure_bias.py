#!/usr/bin/env python
"""Direct SNR-t bias measurement for Anima (flow-matching DiT).

Reproduces Fig. 1c of Yu et al. *Elucidating the SNR-t Bias of Diffusion
Probabilistic Models* (arXiv:2604.16044) on Anima and dumps the
per-(sample, step) baseline LL/LH/HL/HH gap arrays consumed by the
dcw-learnable-calibrator analysis phase (transfer hypothesis, PCA,
S_pop sensitivity profile, overshoot guard).

Reads cached samples from ``post_image_dataset/lora`` (latents +
post-LLMAdapter text embeds), so no VAE / T5 loading is needed.

Measurement
-----------
For each timestep ``i`` in the inference schedule:

    v_fwd(i) = || v_θ((1 − σ_i)·x_0 + σ_i·ε, σ_i) ||
    v_rev(i) = || v_θ(x̂_i, σ_i) ||
    gap(i)   = v_rev(i) − v_fwd(i)

Per-Haar-subband variants come from a single-level orthonormal 2D Haar
DWT on the velocity tensor's (H, W) plane. v2's controller acts on the
LL band only; LH/HL/HH are recorded so the LL-only assumption can be
verified (paper §5.3, dcw-learnable-calibrator-v2 §"What this is not").

Modes
-----
- **Diagnostic (default)**: baseline only. Pair with
  ``--dump_per_sample_gaps`` to emit ``gaps_per_sample.npz`` for the
  transfer-hypothesis / PCA / S_pop analysis scripts.
- **--dcw_sweep**: also runs reverse trajectories with LL-only DCW
  correction (one_minus_sigma schedule) at a grid of λ values. Used by
  A4 to estimate per-step λ-sensitivity ``S_pop(σ_i)``.

Outputs (bench/dcw/results/<YYYYMMDD-HHMM>[-<label>]/)
------------------------------------------------------
    result.json            standard envelope (args, git, env, metrics, artifacts)
    per_step.csv           wide: step, σ_i, v_fwd / v_rev / gap per config
    per_step_bands.csv     same as per_step.csv but split by Haar subband
    gap_curves.png         (1×3) Fig 1c reproduction, gap overlay across
                           configs, baseline gap broken out by subband
    gaps_per_sample.npz    optional, --dump_per_sample_gaps; per-(traj, step)
                           baseline LL/LH/HL/HH gap arrays

Usage
-----
    # A1: production-env baseline (CFG=4, 28 steps, mod-on by default)
    uv run python bench/dcw/measure_bias.py \\
        --dit models/diffusion_models/anima-preview3-base.safetensors \\
        --infer_steps 28 --n_images 48 --n_seeds 2 \\
        --guidance_scale 4.0 \\
        --dump_per_sample_gaps --label v2-prod-env

    # A4: λ-sweep for sensitivity profile S_pop(σ_i)
    uv run python bench/dcw/measure_bias.py \\
        --dit ... --infer_steps 28 --guidance_scale 4.0 \\
        --dcw_sweep --dcw_scalers 0 -0.015 -0.020 -0.025 \\
        --label v2-S_pop

Caveats
-------
- CFG: ``--guidance_scale`` defaults to 4.0 (production env). Setting
  ``--guidance_scale > 1`` live-encodes the unconditional embed via the
  same transient text-encoder block that mod-guidance uses (default
  ``--negative_prompt ""`` mirrors ``inference.py``) and runs the
  cond+uncond pair as a single **batched** DiT forward per step,
  combining as ``v_uncond + s · (v_cond - v_uncond)``. Adds ~30-50% wall
  time vs CFG=1 (the batched path; two-separate-forwards would be ~2×).
  Cached ``_anima_te.safetensors`` sidecars are still cond-only — uncond
  is encoded once at startup and reused across every prompt.

Speed notes
-----------
The hot loop fuses three speedups vs the v1 implementation:

1. **Batched CFG.** Cond and uncond run as a single forward at batch=2·B
   (see ``_cfg_velocity``).
2. **Batched λ sweep.** When ``--dcw_sweep`` is set, all configured λ
   trajectories share the same step's DiT forward at batch=N_λ (or
   2·N_λ under CFG > 1). See ``run_reverse_batched``.
3. **GPU-resident norm accumulation.** Per-step ``‖v‖`` and Haar-band
   norms accumulate on-device; one ``.cpu()`` sync at trajectory end
   instead of 5 syncs per step.

Combined, the prod-env A1 / A4 runs land in ~30-45 min on a 5060 Ti at
1024² (down from several hours at v1 cadence).
- Mod guidance: ON by default with the production-baseline
  ``output/ckpt/pooled_text_proj-0429.safetensors`` checkpoint
  (delta = proj(pos) − proj(neg), schedule applied on blocks
  [mod_start_layer, mod_end_layer)). Setup loads T5 transiently to encode
  the pos/neg prompts, then frees it before the bench loop. Pass
  ``--pooled_text_proj ''`` (empty) for the base-DiT calibration target.
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

BANDS = ("LL", "LH", "HL", "HH")


# ------------------------------------------------------------------
# Single-level 2D orthonormal Haar DWT on the latent (H, W) plane.
# Σ_b ||v_b||² = ||v||² (Parseval); iDWT(DWT(x)) == x to float roundoff.
# ------------------------------------------------------------------


@torch.no_grad()
def haar_dwt_2d(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = v[..., 0::2, 0::2]
    b = v[..., 0::2, 1::2]
    c = v[..., 1::2, 0::2]
    d = v[..., 1::2, 1::2]
    s = 0.5
    LL = (a + b + c + d) * s
    LH = (a + b - c - d) * s
    HL = (a - b + c - d) * s
    HH = (a - b - c + d) * s
    return LL, LH, HL, HH


@torch.no_grad()
def haar_idwt_2d(
    LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor
) -> torch.Tensor:
    s = 0.5
    a = (LL + LH + HL + HH) * s
    b = (LL + LH - HL - HH) * s
    c = (LL - LH + HL - HH) * s
    d = (LL - LH - HL + HH) * s
    out = torch.empty(
        *LL.shape[:-2],
        LL.shape[-2] * 2,
        LL.shape[-1] * 2,
        dtype=LL.dtype,
        device=LL.device,
    )
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


@torch.no_grad()
def haar_band_norms_batched(v: torch.Tensor) -> torch.Tensor:
    """Per-batch Haar-subband L2 norms, on-device.

    v: (B, ...) — DiT velocity. Returns (B, 4) float32 with columns
    ordered as ``BANDS`` = (LL, LH, HL, HH). Stays on GPU so the caller
    can stack ``n_steps`` rows into a single accumulator and sync once
    at trajectory end (avoids 4 + 1 ``.item()`` syncs per step).
    """
    LL, LH, HL, HH = haar_dwt_2d(v.float())
    return torch.stack(
        [
            LL.flatten(start_dim=1).norm(dim=1),
            LH.flatten(start_dim=1).norm(dim=1),
            HL.flatten(start_dim=1).norm(dim=1),
            HH.flatten(start_dim=1).norm(dim=1),
        ],
        dim=1,
    )


def apply_dcw_LL_only_batched(
    prev: torch.Tensor, x0_pred: torch.Tensor, scalars: torch.Tensor
) -> torch.Tensor:
    """LL-only pixel-mode DCW with per-row scalar.

    prev / x0_pred: (B, C, T, H, W) float. scalars: (B,) float — already
    multiplied by the schedule (e.g. ``λ · (1 − σ_i)``). Rows with
    scalar=0 produce a zero correction and are bit-identical to the
    unbatched ``λ=0`` early-out (no early-out kept here so the call is
    graph-stable for ``torch.compile``).
    """
    diff = prev - x0_pred
    LL, LH, HL, HH = haar_dwt_2d(diff)
    z = torch.zeros_like(LL)
    masked = haar_idwt_2d(LL, z, z, z)
    sc = scalars.view(-1, *([1] * (prev.dim() - 1)))
    return prev + sc * masked


# ------------------------------------------------------------------
# Cache loading: one (latent, crossattn_emb) pair per sample.
# ------------------------------------------------------------------


def pick_cached_samples(
    dataset_dir: Path,
    n: int,
    image_h: int | None = None,
    image_w: int | None = None,
) -> list[tuple[str, Path, Path]]:
    """Return list of (stem, latent_npz_path, text_safetensors_path).

    When ``image_h`` and ``image_w`` are both set, restricts to samples whose
    cache filename encodes exactly that resolution (filename format:
    ``<stem>_<H>x<W>_anima.npz``). Required for ``--compile`` to converge to
    a single graph, and for direct cross-run comparability of v_fwd / v_rev
    norms (different bucket resolutions → different patch counts → different
    norms — see CLAUDE.md "Constant-token bucketing").
    """
    out = []
    for npz_path in sorted(dataset_dir.glob("*_anima.npz")):
        m = _LATENT_RE.match(npz_path.name)
        if not m:
            continue
        if image_h is not None and int(m.group("h")) != image_h:
            continue
        if image_w is not None and int(m.group("w")) != image_w:
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
    """Return (x_0 as (1,16,1,H,W) bf16, embed as (1,512,1024) bf16)."""
    with np.load(npz_path) as z:
        latent_keys = [k for k in z.keys() if k.startswith("latents_")]
        if not latent_keys:
            raise RuntimeError(f"no latents_* key in {npz_path}")
        lat = torch.from_numpy(z[latent_keys[0]])
    x_0 = lat.unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)

    sd = load_file(str(te_path))
    key = f"crossattn_emb_v{text_variant}"
    if key not in sd:
        raise KeyError(
            f"{key} not in {te_path}; available: "
            f"{[k for k in sd if k.startswith('crossattn_emb_')]}"
        )
    embed = sd[key].to(device, dtype=torch.bfloat16).unsqueeze(0)
    return x_0, embed


# ------------------------------------------------------------------
# Trajectory measurement.
# ------------------------------------------------------------------


def _padding_mask(x_0: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.zeros(
        1, 1, x_0.shape[-2], x_0.shape[-1], dtype=torch.bfloat16, device=device
    )


@torch.no_grad()
def encode_uncond_embed(
    anima,
    text_encoder,
    negative_prompt: str,
    device: torch.device,
) -> torch.Tensor:
    """Encode the unconditional crossattn embed for CFG.

    Mirrors ``library/inference/text.py:80-94`` — tokenize → encode →
    ``anima._preprocess_text_embeds(...)`` → zero-pad to 512. Returns
    a single (1, 512, 1024) bf16 tensor on ``device``.

    Strategies are assumed primed by the caller (the existing
    transient text-encoder block in ``main`` does this).
    """
    from library.anima import text_strategies

    tok = text_strategies.TokenizeStrategy.get_strategy()
    enc = text_strategies.TextEncodingStrategy.get_strategy()
    tokens = tok.tokenize(negative_prompt)
    embed = enc.encode_tokens(tok, [text_encoder], tokens)
    crossattn, _ = anima._preprocess_text_embeds(
        source_hidden_states=embed[0].to(anima.device),
        target_input_ids=embed[2].to(anima.device),
        target_attention_mask=embed[3].to(anima.device),
        source_attention_mask=embed[1].to(anima.device),
    )
    crossattn[~embed[3].bool()] = 0
    if crossattn.shape[1] < 512:
        crossattn = torch.nn.functional.pad(
            crossattn, (0, 0, 0, 512 - crossattn.shape[1])
        )
    return crossattn.to(device, dtype=torch.bfloat16)


@torch.no_grad()
def _cfg_velocity(
    anima,
    x: torch.Tensor,
    t: torch.Tensor,
    embed: torch.Tensor,
    pad: torch.Tensor,
    *,
    embed_uncond: torch.Tensor | None,
    cfg_scale: float,
) -> torch.Tensor:
    """One DiT forward (CFG=1) or **batched** uncond+cond forward (CFG > 1).

    Combination matches ``library/inference/generation.py``:
        v = v_uncond + s · (v_cond − v_uncond).

    Under CFG > 1 the [uncond, cond] pair is concatenated along the
    batch axis and run as a single forward at batch = 2·B; this halves
    the per-step kernel-launch + attention setup overhead vs two
    separate calls and is the dominant speedup for prod-env (CFG=4)
    bench runs. ``embed_uncond`` is broadcast to match the cond batch
    when needed.
    """
    if cfg_scale == 1.0 or embed_uncond is None:
        return anima(x, t, embed, padding_mask=pad)
    B = x.shape[0]
    embed_u = (
        embed_uncond.expand(B, -1, -1).contiguous()
        if embed_uncond.shape[0] != B
        else embed_uncond
    )
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([t, t], dim=0)
    e2 = torch.cat([embed_u, embed], dim=0)
    p2 = torch.cat([pad, pad], dim=0)
    v = anima(x2, t2, e2, padding_mask=p2)
    v_uncond = v[:B]
    v_cond = v[B:]
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def measure_forward_norms(
    anima,
    x_0: torch.Tensor,
    embed: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    noise_seed: int,
    device: torch.device,
    embed_uncond: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Forward branch only: ‖v_θ((1−σ)x_0 + σε, σ)‖ at every step.

    Bit-identical across DCW configs given the same (x_0, noise_seed),
    so cache once per (image, seed) and reuse across the λ sweep.

    Under CFG > 1 each step runs the cond+uncond pair as a single
    batched forward (see ``_cfg_velocity``); norms / bands are taken on
    the combined velocity. Per-step ``‖v‖`` and band norms accumulate
    on-device and sync once at trajectory end.
    """
    n_steps = len(sigmas) - 1
    pad = _padding_mask(x_0, device)
    g = torch.Generator(device="cpu").manual_seed(noise_seed + 10_000)

    norms_gpu = torch.zeros(n_steps, dtype=torch.float32, device=device)
    bands_gpu = torch.zeros(n_steps, 4, dtype=torch.float32, device=device)
    for i in range(n_steps):
        sigma_i = float(sigmas[i])
        t_i = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)
        eps = torch.randn(x_0.shape, generator=g).to(device, torch.bfloat16)
        x_t = (1.0 - sigma_i) * x_0 + sigma_i * eps
        set_hydra_sigma(anima, t_i)
        v = _cfg_velocity(
            anima,
            x_t,
            t_i,
            embed,
            pad,
            embed_uncond=embed_uncond,
            cfg_scale=cfg_scale,
        )
        norms_gpu[i] = v.float().flatten().norm()
        bands_gpu[i] = haar_band_norms_batched(v)[0]

    norms = norms_gpu.cpu().numpy().astype(np.float64)
    bands_arr = bands_gpu.cpu().numpy().astype(np.float64)
    bands = {b: bands_arr[:, j] for j, b in enumerate(BANDS)}
    return norms, bands


@torch.no_grad()
def run_reverse_batched(
    anima,
    x_0: torch.Tensor,
    embed: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    noise_seed: int,
    dcw_lams: list[float],
    device: torch.device,
    embed_uncond: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Run ``len(dcw_lams)`` reverse trajectories in parallel along batch.

    All trajectories share (x_0, embed, schedule, initial noise); only
    the DCW correction λ differs per row, so they diverge after step 0.
    Each step does **one** DiT forward at batch = ``len(dcw_lams)`` (or
    2× under CFG > 1, see ``_cfg_velocity``), replacing what was
    previously ``len(dcw_lams) × n_steps`` separate forwards. For the
    common A4 sweep (4 λ values, CFG=4) this is ~3-4× faster on the
    reverse branch — the dominant cost when ``--dcw_sweep`` is set.

    DCW correction (when ``λ != 0``): LL-only with
    ``scalar_i = λ · (1 − σ_i)``, applied to ``(prev − x0_pred)``
    independently per row.

    Returns one (norms, bands) tuple per ``dcw_lams`` entry, in input
    order. ``norms`` shape: (n_steps,). ``bands[b]`` shape: (n_steps,).
    """
    n_lams = len(dcw_lams)
    n_steps = len(sigmas) - 1
    pad_one = _padding_mask(x_0, device)
    pad = pad_one.expand(n_lams, -1, -1, -1).contiguous()
    embed_b = embed.expand(n_lams, -1, -1).contiguous()

    g = torch.Generator(device="cpu").manual_seed(noise_seed)
    x_hat0 = torch.randn(x_0.shape, generator=g).to(device, torch.bfloat16)
    x_hat = x_hat0.expand(n_lams, -1, -1, -1, -1).contiguous()

    lams_t = torch.tensor(dcw_lams, dtype=torch.float32, device=device)

    norms_gpu = torch.zeros(n_steps, n_lams, dtype=torch.float32, device=device)
    bands_gpu = torch.zeros(n_steps, n_lams, 4, dtype=torch.float32, device=device)

    for i in range(n_steps):
        sigma_i = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        # σ is shared across rows; pass shape (1,) to the Hydra router
        # (state is a scalar) and shape (n_lams,) to the model forward
        # to match the batch.
        t_one = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)
        t_b = torch.full((n_lams,), sigma_i, device=device, dtype=torch.bfloat16)

        set_hydra_sigma(anima, t_one)
        v = _cfg_velocity(
            anima,
            x_hat,
            t_b,
            embed_b,
            pad,
            embed_uncond=embed_uncond,
            cfg_scale=cfg_scale,
        )
        norms_gpu[i] = v.float().flatten(start_dim=1).norm(dim=1)
        bands_gpu[i] = haar_band_norms_batched(v)

        v_f = v.float()
        prev = x_hat.float() + (sigma_next - sigma_i) * v_f
        if sigma_next > 0.0:
            x0_pred = x_hat.float() - sigma_i * v_f
            scalars = lams_t * (1.0 - sigma_i)
            prev = apply_dcw_LL_only_batched(prev, x0_pred, scalars)
        x_hat = prev.to(torch.bfloat16)

    norms_np = norms_gpu.cpu().numpy().astype(np.float64)
    bands_np = bands_gpu.cpu().numpy().astype(np.float64)

    out: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for j in range(n_lams):
        bands_dict = {b: bands_np[:, j, k] for k, b in enumerate(BANDS)}
        out.append((norms_np[:, j], bands_dict))
    return out


# ------------------------------------------------------------------
# Adapter attach (LoRA / HydraLoRA). Mirrors library/inference/models.py.
# ------------------------------------------------------------------


def attach_loras(
    anima,
    paths: list[str],
    mults: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    from networks import lora_anima

    hydra_flags = [_is_hydra_moe(p) for p in paths]
    if any(hydra_flags) and not all(hydra_flags):
        raise SystemExit(
            "Mixing HydraLoRA moe files with regular LoRA files in --lora_weight "
            "is not supported (matches inference-time restriction)."
        )
    any_hydra = any(hydra_flags)
    log.info(
        f"attaching {len(paths)} adapter(s) as "
        f"{'router-live HydraLoRA' if any_hydra else 'LoRA'} hooks…"
    )

    if len(mults) == 1:
        mults = mults * len(paths)
    if len(mults) != len(paths):
        raise SystemExit(
            f"--lora_multiplier has {len(mults)} entries but --lora_weight has "
            f"{len(paths)}. Pass one multiplier per weight, or one shared."
        )

    for path, mult in zip(paths, mults):
        sd = load_file(path)
        sd = {k: v for k, v in sd.items() if k.startswith("lora_unet_")}
        network, weights_sd = lora_anima.create_network_from_weights(
            multiplier=mult,
            file=None,
            ae=None,
            text_encoders=[],
            unet=anima,
            weights_sd=sd,
            for_inference=True,
        )
        network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
        info = network.load_state_dict(weights_sd, strict=False)
        if info.unexpected_keys:
            log.warning(f"{path}: unexpected (first 5): {info.unexpected_keys[:5]}")
        if info.missing_keys:
            log.warning(f"{path}: missing (first 5): {info.missing_keys[:5]}")
        network.to(device, dtype=dtype)
        network.eval().requires_grad_(False)
        if any_hydra:
            hydra_networks = list(getattr(anima, "_hydra_networks", []))
            hydra_networks.append(network)
            anima._hydra_networks = hydra_networks
            anima._hydra_network = network
        log.info(f"  attached {path} (mult={mult}, modules={len(network.unet_loras)})")


# ------------------------------------------------------------------
# CLI.
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-preview3-base.safetensors",
        help="DiT .safetensors path (default: anima-preview3-base).",
    )
    p.add_argument(
        "--lora_weight",
        type=str,
        nargs="+",
        default=None,
        help="Optional LoRA / HydraLoRA adapter(s) to stack on the base DiT. "
        "Auto-detects HydraLoRA moe (lora_ups.* keys) and attaches router-live "
        "via dynamic forward hooks; plain LoRA goes through the same dynamic path "
        "(math-equivalent to static merge for this measurement).",
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
        help="Directory with cached *_anima.npz + *_anima_te.safetensors pairs.",
    )
    p.add_argument(
        "--text_variant",
        type=int,
        default=0,
        help="Cached caption variant (crossattn_emb_v<N>); 0 = canonical.",
    )
    p.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="torch | sdpa | xformers | sage | flash",
    )
    p.add_argument("--n_images", type=int, default=16, help="Cached samples to use")
    p.add_argument("--n_seeds", type=int, default=2, help="Seeds per sample")
    p.add_argument(
        "--image_h",
        type=int,
        default=None,
        help="Restrict to cached samples with this image-space height (the "
        "<H> in <stem>_<H>x<W>_anima.npz). Required (with --image_w) for "
        "--compile to converge to a single graph and for direct cross-run "
        "comparability of velocity norms.",
    )
    p.add_argument(
        "--image_w",
        type=int,
        default=None,
        help="Restrict to cached samples with this image-space width.",
    )
    p.add_argument(
        "--infer_steps",
        type=int,
        default=28,
        help="Inference schedule length (v2 prod env = 28).",
    )
    p.add_argument(
        "--flow_shift", type=float, default=1.0, help="σ shift (matches inference.py)."
    )
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the DiT before the bench loop. Each unique latent "
        "(H, W) pays a one-time warm-up; steady-state is much faster. Best "
        "amortized when n_images is moderate (every sample's ~150+ forwards "
        "run at the same shape, and dynamo auto-flips to dynamic shapes after "
        "the second distinct (H, W)).",
    )
    p.add_argument(
        "--dcw_sweep",
        action="store_true",
        help="Also run LL-only DCW-corrected trajectories at --dcw_scalers "
        "(one_minus_sigma schedule). Used by v2 §A4 to estimate S_pop(σ_i).",
    )
    p.add_argument(
        "--dcw_scalers",
        type=float,
        nargs="+",
        default=[-0.010, 0, 0.010],
        help="λ values to sweep when --dcw_sweep is set (negative on Anima; "
        "v2 §A4 uses {0, -0.015, -0.020, -0.025}).",
    )
    p.add_argument(
        "--dump_per_sample_gaps",
        action="store_true",
        help="Dump per-(traj, step) baseline LL/LH/HL/HH gap arrays "
        "(shape (n_images*n_seeds, n_steps)) to gaps_per_sample.npz. "
        "Consumed by the dcw-learnable-calibrator analysis scripts "
        "(transfer hypothesis, PCA, S_pop).",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale. 1.0 = single conditional "
        "forward (matches v1 calibration). >1 live-encodes the "
        "unconditional embed at startup and runs an extra DiT forward "
        "per step, combining as v_uncond + s · (v_cond − v_uncond) "
        "(matches inference.py). v2 §A1 production env = 4.0.",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Unconditional prompt for CFG > 1 (default '' matches "
        "inference.py default).",
    )

    # Modulation guidance (off by default — base-DiT calibration target).
    # When --pooled_text_proj is set, mirrors inference.py's mod-guidance
    # pipeline so v2 §A1 can run a production-mod-on cross-check.
    g_mod = p.add_argument_group("modulation guidance (optional)")
    g_mod.add_argument(
        "--pooled_text_proj",
        type=str,
        default="output/ckpt/pooled_text_proj-0429.safetensors",
        help="Path to trained pooled_text_proj weights (.safetensors). "
        "Default enables modulation guidance with the production-baseline "
        "0429 checkpoint and the pos/neg prompts below. Pass an empty "
        "string (--pooled_text_proj '') to disable for the base-DiT "
        "calibration measurement.",
    )
    g_mod.add_argument(
        "--text_encoder",
        type=str,
        default="models/text_encoders/qwen_3_06b_base.safetensors",
        help="Qwen3 text encoder path; only loaded when mod guidance is on, "
        "freed after the steering delta is computed.",
    )
    g_mod.add_argument("--mod_w", type=float, default=3.0)
    g_mod.add_argument(
        "--mod_pos_prompt", type=str, default="absurdres, masterpiece, score_9"
    )
    g_mod.add_argument(
        "--mod_neg_prompt",
        type=str,
        default="worst quality, low quality, score_1, score_2, score_3",
    )
    g_mod.add_argument("--mod_start_layer", type=int, default=8)
    g_mod.add_argument("--mod_end_layer", type=int, default=27)
    g_mod.add_argument("--mod_taper", type=int, default=0)
    g_mod.add_argument("--mod_taper_scale", type=float, default=0.25)
    g_mod.add_argument("--mod_final_w", type=float, default=0.0)

    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label appended to the run dir (bench/dcw/results/<ts>-<label>/).",
    )
    return p.parse_args()


# ------------------------------------------------------------------
# Output helpers.
# ------------------------------------------------------------------


def write_per_step_csv(
    out_dir: Path, accum: dict, sigmas: torch.Tensor, n_steps: int
) -> Path:
    csv_path = out_dir / "per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["step", "sigma_i"]
        for name in accum:
            headers += [
                f"{name}_v_fwd",
                f"{name}_v_rev",
                f"{name}_gap",
                f"{name}_gap_std",
            ]
        w.writerow(headers)
        for i in range(n_steps):
            row: list = [i, float(sigmas[i])]
            for name in accum:
                row += [
                    accum[name]["v_fwd"][i],
                    accum[name]["v_rev"][i],
                    accum[name]["gap"][i],
                    accum[name]["gap_std"][i],
                ]
            w.writerow(row)
    return csv_path


def write_per_band_csv(
    out_dir: Path, accum: dict, sigmas: torch.Tensor, n_steps: int
) -> Path:
    csv_path = out_dir / "per_step_bands.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["step", "sigma_i"]
        for name in accum:
            for b in BANDS:
                headers += [
                    f"{name}_v_fwd_{b}",
                    f"{name}_v_rev_{b}",
                    f"{name}_gap_{b}",
                ]
        w.writerow(headers)
        for i in range(n_steps):
            row: list = [i, float(sigmas[i])]
            for name in accum:
                for b in BANDS:
                    row += [
                        accum[name]["v_fwd_bands"][b][i],
                        accum[name]["v_rev_bands"][b][i],
                        accum[name]["gap_bands"][b][i],
                    ]
            w.writerow(row)
    return csv_path


def make_plot(out_dir: Path, accum: dict, n_steps: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True)
    base = accum["baseline"]
    xs = range(n_steps)

    axes[0].plot(xs, base["v_fwd"], label="forward ‖v(x_t, t)‖", color="#2a9d8f")
    axes[0].plot(xs, base["v_rev"], label="reverse ‖v(x̂_t, t)‖", color="#e76f51")
    axes[0].set_title("Baseline forward vs reverse velocity (Fig 1c)")
    axes[0].set_xlabel("step i")
    axes[0].set_ylabel("‖v‖₂")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name in accum:
        axes[1].plot(xs, accum[name]["gap"], label=name, alpha=0.85)
    axes[1].fill_between(
        xs,
        base["gap"] - base["gap_std"],
        base["gap"] + base["gap_std"],
        color="#888888",
        alpha=0.20,
        label="baseline ±1σ across (img×seed)",
    )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("gap(i) = ‖v_rev‖ − ‖v_fwd‖  (closer to 0 = better)")
    axes[1].set_xlabel("step i")
    axes[1].set_ylabel("gap")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    band_colors = {"LL": "#264653", "LH": "#2a9d8f", "HL": "#e9c46a", "HH": "#e76f51"}
    for b in BANDS:
        axes[2].plot(xs, base["gap_bands"][b], label=b, color=band_colors[b], alpha=0.9)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("Baseline gap by Haar subband")
    axes[2].set_xlabel("step i")
    axes[2].set_ylabel("gap (band)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "gap_curves.png"
    fig.savefig(png_path, dpi=130)
    log.info(f"plot → {png_path}")
    return True


def print_summary(accum: dict, ranked: list, dcw_sweep: bool) -> None:
    base = accum["baseline"]
    print("\n=== SNR-t bias measurement ===")
    print(
        f"baseline integrated signed gap: {base['gap'].sum():+.3f}  "
        f"(Anima flow-matching: gap is typically negative — forward > reverse — "
        f"opposite of the DDPM noise-pred sign in the paper)"
    )
    print(
        f"baseline gap std across {base['n']} (img×seed) trajectories: "
        f"mean σ_step = {base['gap_std'].mean():.3f}, "
        f"max σ_step = {base['gap_std'].max():.3f}"
    )

    print("\nbaseline integrated signed gap by Haar subband:")
    print(f"  {'band':<4s}  {'signed':>9s}  {'|gap|':>8s}")
    for b in BANDS:
        g = float(base["gap_bands"][b].sum())
        a = float(np.abs(base["gap_bands"][b]).sum())
        print(f"  {b:<4s}  {g:+9.3f}  {a:8.3f}")

    if dcw_sweep:
        print("\nconfigs ranked by integrated |gap| (smaller = closer alignment):")
        for rank, (name, a, s) in enumerate(ranked, 1):
            print(f"  {rank:>2}. {name:<24s}  |gap|={a:7.3f}  signed={s:+7.3f}")


# ------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if (args.image_h is None) != (args.image_w is None):
        raise SystemExit(
            "--image_h and --image_w must be set together (or both omitted)."
        )
    out_dir = make_run_dir("dcw", label=args.label)
    log.info(f"output → {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    log.info("loading DiT…")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    # Mod-guidance: pooled_text_proj params are meta tensors when not in the
    # base checkpoint, so load weights BEFORE .to() (matches
    # library/inference/models.py:95-99). Bench-default is off → buffers stay
    # at init zeros after reset, giving an identity per-block addition.
    if args.pooled_text_proj:
        anima_utils.load_pooled_text_proj(anima, args.pooled_text_proj, "cpu")
    anima.to(device, dtype=dtype)
    anima.eval().requires_grad_(False)

    # Transient text-encoder block. Triggered when either mod-guidance is on
    # (encodes pos/neg pooled deltas) or CFG > 1 (encodes uncond crossattn).
    # Loads the Qwen3 text encoder once, runs both encodes, frees it.
    embed_uncond: torch.Tensor | None = None
    needs_text_encoder = bool(args.pooled_text_proj) or args.guidance_scale != 1.0
    if needs_text_encoder:
        from library.anima import strategy as strategy_anima, text_strategies
        from library.inference.models import load_text_encoder

        # Mirror inference.py:909-918 — mod_guidance.tokenize_strategy.tokenize()
        # reads the module-level singletons, so they have to be primed before
        # setup_mod_guidance encodes the pos/neg prompts.
        text_strategies.TokenizeStrategy.set_strategy(
            strategy_anima.AnimaTokenizeStrategy(
                qwen3_path=args.text_encoder,
                t5_tokenizer_path=None,
                qwen3_max_length=512,
                t5_max_length=512,
            )
        )
        text_strategies.TextEncodingStrategy.set_strategy(
            strategy_anima.AnimaTextEncodingStrategy()
        )

        text_encoder = load_text_encoder(args, dtype=torch.bfloat16, device=device)
        text_encoder.eval()

        if args.guidance_scale != 1.0:
            log.info(
                f"CFG={args.guidance_scale}; encoding uncond "
                f"(negative_prompt='{args.negative_prompt}')"
            )
            embed_uncond = encode_uncond_embed(
                anima, text_encoder, args.negative_prompt, device
            )

        if args.pooled_text_proj:
            from library.inference.mod_guidance import setup_mod_guidance

            setup_mod_guidance(
                args,
                anima,
                device,
                shared_models={"text_encoder": text_encoder},
            )
        else:
            anima.reset_mod_guidance()

        # Free the text encoder — neither CFG nor mod-guidance needs it during
        # the bench loop (uncond is one frozen tensor, mod delta is baked).
        del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        anima.reset_mod_guidance()

    if args.lora_weight:
        attach_loras(anima, args.lora_weight, list(args.lora_multiplier), device, dtype)

    # Compile last: after LoRA + mod-guidance attach so the wrapped graph sees
    # them. set_hydra_sigma already routes through ``_orig_mod`` so writes to
    # router state work on the OptimizedModule wrapper.
    if args.compile:
        log.info("torch.compile(DiT)…")
        anima = torch.compile(anima)

    samples = pick_cached_samples(
        Path(args.dataset_dir),
        args.n_images,
        image_h=args.image_h,
        image_w=args.image_w,
    )
    if not samples:
        shape_msg = (
            f" matching {args.image_h}x{args.image_w}"
            if args.image_h is not None
            else ""
        )
        raise SystemExit(
            f"no cached samples{shape_msg} under {args.dataset_dir}. "
            "Expected *_anima.npz + *_anima_te.safetensors pairs (make preprocess)."
        )
    shape_info = (
        f" @ {args.image_h}x{args.image_w}"
        if args.image_h is not None
        else " (mixed shapes)"
    )
    log.info(
        f"using {len(samples)} cached samples (variant v{args.text_variant}){shape_info}"
    )

    _, sigmas_t = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas_t.cpu()
    n_steps = args.infer_steps
    log.info(
        f"infer_steps={n_steps}, flow_shift={args.flow_shift}, "
        f"σ₀={float(sigmas[0]):.3f}, σₙ={float(sigmas[-1]):.3f}"
    )

    # DCW configs: baseline + optional λ sweep (LL-only, one_minus_sigma).
    configs: list[tuple[str, float]] = [("baseline", 0.0)]
    if args.dcw_sweep:
        for lam in args.dcw_scalers:
            if lam == 0.0:
                continue
            configs.append((f"λ={lam}_LL_oneminussigma", lam))
    n_fwd = len(samples) * args.n_seeds
    n_rev = len(configs) * n_fwd
    log.info(
        f"{len(configs)} config(s) × {len(samples)} samples × {args.n_seeds} seeds: "
        f"{n_fwd} fwd calls + {n_fwd} batched-rev calls "
        f"(each rev advances {len(configs)} λ trajectories at batch={len(configs)}"
        f"{' × 2 under CFG' if args.guidance_scale > 1.0 else ''})"
    )

    # Preload cached data onto device.
    log.info("loading cached latents + text embeds…")
    encoded = []
    for stem, npz, te in samples:
        x_0, embed = load_cached(npz, te, args.text_variant, device)
        encoded.append((stem, x_0, embed))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accum: dict = {
        name: dict(
            v_fwd=np.zeros(n_steps),
            v_rev=np.zeros(n_steps),
            gap=np.zeros(n_steps),
            gap_sq=np.zeros(n_steps),
            v_fwd_bands={b: np.zeros(n_steps) for b in BANDS},
            v_rev_bands={b: np.zeros(n_steps) for b in BANDS},
            gap_bands={b: np.zeros(n_steps) for b in BANDS},
            n=0,
        )
        for name, _ in configs
    }

    per_sample_bands: dict[str, np.ndarray] | None = None
    per_sample_stems: list[str] | None = None
    if args.dump_per_sample_gaps:
        n_traj = len(encoded) * args.n_seeds
        per_sample_bands = {b: np.zeros((n_traj, n_steps)) for b in BANDS}
        per_sample_stems = [""] * n_traj

    t0 = time.time()

    # Phase 1: forward-branch norms (cached per (img, seed) across configs).
    fwd_cache: dict[tuple[int, int], tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    pbar = tqdm(total=n_fwd, desc="fwd")
    for img_idx, (stem, x_0, embed) in enumerate(encoded):
        for seed_idx in range(args.n_seeds):
            seed = args.seed_base + 1000 * img_idx + seed_idx
            fwd_cache[(img_idx, seed_idx)] = measure_forward_norms(
                anima,
                x_0,
                embed,
                sigmas,
                noise_seed=seed,
                device=device,
                embed_uncond=embed_uncond,
                cfg_scale=args.guidance_scale,
            )
            pbar.update(1)
            pbar.set_postfix_str(f"{stem} seed={seed}")
    pbar.close()

    # Phase 2: all-λ reverse trajectories batched per (img, seed).
    config_lams = [lam for _, lam in configs]
    pbar = tqdm(total=n_fwd, desc=f"rev (×{len(configs)} λ batched)")
    for img_idx, (stem, x_0, embed) in enumerate(encoded):
        for seed_idx in range(args.n_seeds):
            seed = args.seed_base + 1000 * img_idx + seed_idx
            rev_results = run_reverse_batched(
                anima,
                x_0,
                embed,
                sigmas,
                noise_seed=seed,
                dcw_lams=config_lams,
                device=device,
                embed_uncond=embed_uncond,
                cfg_scale=args.guidance_scale,
            )
            v_fwd, fwd_bands = fwd_cache[(img_idx, seed_idx)]
            for j, (name, _lam) in enumerate(configs):
                rev_norms, rev_bands = rev_results[j]
                gap = rev_norms - v_fwd
                accum[name]["v_fwd"] += v_fwd
                accum[name]["v_rev"] += rev_norms
                accum[name]["gap"] += gap
                accum[name]["gap_sq"] += gap**2
                for b in BANDS:
                    gap_b = rev_bands[b] - fwd_bands[b]
                    accum[name]["v_fwd_bands"][b] += fwd_bands[b]
                    accum[name]["v_rev_bands"][b] += rev_bands[b]
                    accum[name]["gap_bands"][b] += gap_b
                    if name == "baseline" and per_sample_bands is not None:
                        row = img_idx * args.n_seeds + seed_idx
                        per_sample_bands[b][row] = gap_b
                        per_sample_stems[row] = stem
                accum[name]["n"] += 1
            pbar.update(1)
            pbar.set_postfix_str(f"{stem} seed={seed}")
    pbar.close()
    clear_hydra_sigma(anima)
    log.info(f"done in {time.time() - t0:.0f}s")

    # Reduce: mean over (img × seed); std on the gap from running Σ gap².
    for name in accum:
        n = accum[name]["n"]
        mean_g = accum[name]["gap"] / n
        mean_g_sq = accum[name]["gap_sq"] / n
        accum[name]["gap_std"] = np.sqrt(np.maximum(mean_g_sq - mean_g**2, 0.0))
        for k in ("v_fwd", "v_rev", "gap"):
            accum[name][k] = accum[name][k] / n
        for k in ("v_fwd_bands", "v_rev_bands", "gap_bands"):
            for b in BANDS:
                accum[name][k][b] = accum[name][k][b] / n

    # Metrics envelope.
    ranked = sorted(
        (
            (
                name,
                float(np.abs(accum[name]["gap"]).sum()),
                float(accum[name]["gap"].sum()),
            )
            for name in accum
        ),
        key=lambda t: t[1],
    )
    metrics = {
        "infer_steps": n_steps,
        "n_samples": len(samples),
        "n_seeds": args.n_seeds,
        "text_variant": args.text_variant,
        "configs_ranked_by_integrated_abs_gap": [
            {"config": name, "integrated_abs_gap": a, "integrated_signed_gap": s}
            for name, a, s in ranked
        ],
        "per_band_integrated_signed_gap": {
            name: {b: float(accum[name]["gap_bands"][b].sum()) for b in BANDS}
            for name in accum
        },
    }

    # Write artifacts.
    csv_path = write_per_step_csv(out_dir, accum, sigmas, n_steps)
    log.info(f"CSV → {csv_path}")
    band_csv_path = write_per_band_csv(out_dir, accum, sigmas, n_steps)
    log.info(f"per-band CSV → {band_csv_path}")

    per_sample_path: Path | None = None
    if per_sample_bands is not None:
        per_sample_path = out_dir / "gaps_per_sample.npz"
        np.savez(
            per_sample_path,
            sigmas=sigmas.numpy()[:n_steps],
            stems=np.array(per_sample_stems, dtype=object),
            **{f"gap_{b}": per_sample_bands[b] for b in BANDS},
        )
        log.info(f"per-sample gaps → {per_sample_path}")

    plot_written = make_plot(out_dir, accum, n_steps)

    artifacts = (
        ["per_step.csv", "per_step_bands.csv"]
        + (["gap_curves.png"] if plot_written else [])
        + (["gaps_per_sample.npz"] if per_sample_path is not None else [])
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

    print_summary(accum, ranked, args.dcw_sweep)


if __name__ == "__main__":
    main()
