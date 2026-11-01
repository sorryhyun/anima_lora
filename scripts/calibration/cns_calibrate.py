#!/usr/bin/env python3
"""CNS Phase-1 completion-matrix calibration — arXiv 2605.30332, Algorithm 2.

Builds the *deployable* γ(f, t) artifact that `--cns` consumes at inference.
Where `gamma_probe.py` is the read-only Phase-0 staircase check (one config),
this drives it across the **deploy config** — cfg=4.0 at the top-N most-frequent
aspect buckets — and bundles the per-aspect γ matrices into a single
`networks/calibration/cns_gamma.npz`.

γ is measured from the *deterministic euler ODE* trajectory at the deploy cfg
(Alg. 2): per step, x0_pred = latents − σ·v, then
γ(f,t) = 1 − |X_pred − X₀|² / |X₀|², radially binned. The recoloring it feeds
then applies on the **er_sde** path (CNS only acts on injected noise). Aspects
come from `DCW_ASPECT_BUCKETS` (the dataset's measured top-5 (H, W) by
frequency); `--n_aspects 3` takes the three most common.

Prompts are **real captions** sampled from the preprocessed training subset
(post_image_dataset/lora stems → image_dataset/<artist>/<stem>.txt), picking the
richest (most tags) across distinct artists so γ reflects the model on its own
data distribution. `--prompts_file` overrides; DEFAULT_PROMPTS is the fallback.

Load-once: the DiT is loaded (and, with `--compile`, compiled) a single time;
all text is precomputed up front and the encoder freed before the trajectory
loop (the TE→free→DiT memory invariant), so a 16GB box never holds TE+DiT at
once. The three default aspects are all token-count 4200, so `--compile`'s
`compile_blocks` builds ONE native-flatten graph reused across every aspect.

Run from repo root (anima_lora/):
    python scripts/calibration/cns_calibrate.py --cfg 4.0 --n_aspects 3   # compiled (default)
    python scripts/calibration/cns_calibrate.py --cfg 4.0 --no-compile    # eager
    # probe the adapter instead of base (γ is LoRA-transparent per Phase 0):
    python scripts/calibration/cns_calibrate.py --cfg 4.0 --extra --lora_weight output/ckpt/<x>.safetensors

Output:
    networks/calibration/cns_gamma.npz   (the shipped artifact `--cns auto` loads)
    (a per-aspect σ50 staircase summary prints to stdout — no report file)
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


import numpy as np
import torch
from scripts.calibration.gamma_probe import (
    DEFAULT_PROMPTS,
    _gamma_for_trajectory,
    _radial_bins,
    _sigma_at_gamma_half,
    _StepCapture,
)

from anima_lora import GenerationRequest, default_checkpoints, get_generation_settings
from library.datasets.buckets import DCW_ASPECT_BUCKETS
from library.env import resolve_under_home
from library.inference.generation import generate
from library.inference.models import load_dit_model, load_text_encoder
from library.inference.text import prepare_text_inputs
from library.runtime.device import clean_memory_on_device

_ckpts = default_checkpoints()
DIT, VAE, TEXT_ENCODER = _ckpts.dit, _ckpts.vae, _ckpts.text_encoder
DEFAULT_OUT = "networks/calibration/cns_gamma.npz"

# Real captions come from the actually-preprocessed training subset: cached TE
# stems under post_image_dataset/lora/<artist>/<stem>_anima_te.safetensors map
# 1:1 to the caption master at image_dataset/<artist>/<stem>.txt.
LORA_CACHE_DIR = "post_image_dataset/lora"
CAPTION_DIR = "image_dataset"
_TE_SUFFIX = "_anima_te.safetensors"


def _sample_real_prompts(n: int) -> list[str]:
    """Pick the ``n`` richest real captions (most tags) across distinct artists.

    Long, tag-dense captions excite the cross-attn (and thus the spectral
    staircase) the hardest. Deterministic (longest-first, one per artist for
    spectral diversity); falls back to DEFAULT_PROMPTS if the dataset isn't
    present (e.g. a fresh checkout).
    """
    cache_root = resolve_under_home(LORA_CACHE_DIR)
    cap_root = resolve_under_home(CAPTION_DIR)
    if not cache_root.exists() or not cap_root.exists():
        print(
            f"  ! dataset not found ({cache_root} / {cap_root}); using DEFAULT_PROMPTS"
        )
        return list(DEFAULT_PROMPTS)

    cands: list[tuple[int, str, str]] = []  # (n_tags, artist, caption)
    for te in cache_root.rglob(f"*{_TE_SUFFIX}"):
        artist = te.parent.name
        stem = te.name[: -len(_TE_SUFFIX)]
        txt = cap_root / artist / f"{stem}.txt"
        if not txt.exists():
            continue
        caption = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if caption:
            cands.append((caption.count(",") + 1, artist, caption))

    if not cands:
        print("  ! no captions resolved from cache stems; using DEFAULT_PROMPTS")
        return list(DEFAULT_PROMPTS)

    cands.sort(key=lambda c: c[0], reverse=True)
    picked: list[str] = []
    seen_artists: set[str] = set()
    for n_tags, artist, caption in cands:  # distinct artists first
        if artist not in seen_artists:
            picked.append(caption)
            seen_artists.add(artist)
            print(f"  prompt[{len(picked)}]: {artist}/{n_tags}tags — {caption[:60]}...")
        if len(picked) == n:
            break
    for n_tags, artist, caption in cands:  # top up if < n distinct artists
        if len(picked) == n:
            break
        if caption not in picked:
            picked.append(caption)
    return picked[:n]


def _make_args(args, prompt, hw, seed, extra, device):
    """Build a per-call inference namespace (euler ODE, latent out) at (H,W)=hw."""
    req = GenerationRequest(
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        prompt=prompt,
        image_size=(hw[0], hw[1]),
        infer_steps=args.steps,
        guidance_scale=args.cfg,
        flow_shift=args.flow_shift,
        sampler="euler",
        seed=seed,
        output_type="latent",
        extra_argv=tuple(extra),
    )
    ns = req.to_args()
    ns.device = device
    return ns


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", type=float, default=4.0, help="Deploy guidance scale.")
    p.add_argument("--n_aspects", type=int, default=3, help="Top-N DCW_ASPECT_BUCKETS.")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n_bins", type=int, default=32, help="Radial frequency bins (F).")
    p.add_argument(
        "--n_prompts",
        type=int,
        default=4,
        help="How many real captions to sample (distinct artists, most tags).",
    )
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="One prompt per line; overrides real-caption sampling.",
    )
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Calibration npz path.")
    p.add_argument(
        "--average_aspects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ship one aspect-averaged γ (shape (1,T,F)) — cross-aspect variation "
        "is cosmetic (β MAD ~0.01), so a single γ serves any resolution. "
        "--no-average-aspects keeps the full per-aspect (A,T,F) table instead.",
    )
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the DiT once and compile_blocks it (1 graph for the default "
        "3×4200-token aspects). --no-compile runs eager. Compile-after-apply is "
        "respected (load_dit_model compiles post adapter-attach).",
    )
    p.add_argument("--dit", type=str, default=DIT)
    p.add_argument("--vae", type=str, default=VAE)
    p.add_argument("--text_encoder", type=str, default=TEXT_ENCODER)
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Verbatim extra inference flags (e.g. --lora_weight <path>). Last.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text().splitlines()
            if ln.strip()
        ]
    else:
        print(f"sampling {args.n_prompts} real captions from {LORA_CACHE_DIR} ...")
        prompts = _sample_real_prompts(args.n_prompts)
    aspects = list(DCW_ASPECT_BUCKETS[: args.n_aspects])  # (H, W) tuples

    # Compile rides the inference flags via extra_argv; compile_blocks keys the
    # graph on token count, so all same-token-count aspects share one graph.
    extra = list(args.extra)
    if args.compile and not any(f in extra for f in ("--compile_blocks", "--compile")):
        extra = extra + ["--compile_blocks"]

    # ---- load the DiT ONCE (and compile), reused across every trajectory -----
    base_args = _make_args(args, prompts[0], aspects[0], args.seeds[0], extra, device)
    settings = get_generation_settings(base_args)
    print(
        f"loading DiT (compile={args.compile}) — one graph for {len(aspects)} aspects ..."
    )
    anima = load_dit_model(base_args, device, torch.bfloat16)
    shared = {"model": anima}

    # ---- precompute text per unique prompt, then free the encoder ------------
    print("encoding text for all prompts (then freeing the encoder) ...")
    text_encoder = load_text_encoder(base_args, dtype=torch.bfloat16, device=device)
    txt_shared = {"text_encoder": text_encoder, "conds_cache": {}}
    text_by_prompt: dict[str, dict] = {}
    for prompt in prompts:
        pa = _make_args(args, prompt, aspects[0], args.seeds[0], extra, device)
        ctx, ctx_null = prepare_text_inputs(pa, device, anima, txt_shared)
        text_by_prompt[prompt] = {"context": ctx, "context_null": ctx_null}
    del text_encoder, txt_shared
    gc.collect()
    clean_memory_on_device(device)

    # ---- per-aspect γ, reusing the compiled model + cached text --------------
    gammas: list[np.ndarray] = []
    sigmas_ref: np.ndarray | None = None
    centers_ref: np.ndarray | None = None
    s50_spreads: list[float] = []
    agg_s50s: list[float] = []
    for hw in aspects:
        print(f"\n[aspect {hw[0]}x{hw[1]}]  cfg={args.cfg}")
        bin_idx, centers = _radial_bins(hw[0] // 8, hw[1] // 8, args.n_bins)
        per_traj: list[np.ndarray] = []
        for prompt in prompts:
            for seed in args.seeds:
                ca = _make_args(args, prompt, hw, seed, extra, device)
                with _StepCapture() as cap, torch.no_grad():
                    latent = generate(
                        ca,
                        settings,
                        shared_models=shared,
                        precomputed_text_data=text_by_prompt[prompt],
                    )
                if len(cap.x0_preds) != args.steps:
                    print(
                        f"    ! {hw[0]}x{hw[1]} seed={seed}: {len(cap.x0_preds)} steps; skip"
                    )
                    continue
                x0_final = latent.float().squeeze(0).squeeze(1).cpu().numpy()  # (C,H,W)
                per_traj.append(
                    _gamma_for_trajectory(cap.x0_preds, x0_final, bin_idx, args.n_bins)
                )
                sigmas_ref = cap.sigmas if sigmas_ref is None else sigmas_ref
                print(
                    f"    ok: {hw[0]}x{hw[1]} seed={seed} '{prompt[:32]}...' ({len(per_traj)} traj)"
                )

        if not per_traj:
            raise SystemExit(
                f"No trajectories captured at {hw[0]}x{hw[1]} — check paths."
            )
        g = np.mean(np.stack(per_traj, 0), 0)  # (T, F)
        gammas.append(g)
        centers_ref = centers if centers_ref is None else centers_ref
        sig_mid = sigmas_ref[:-1]
        s50 = np.array(
            [_sigma_at_gamma_half(g[:, f], sig_mid) for f in range(args.n_bins)]
        )
        lo, hi = int(args.n_bins * 0.15), int(args.n_bins * 0.85)
        spread = float(np.nanmean(s50[: lo + 1]) - np.nanmean(s50[hi:]))
        agg = _sigma_at_gamma_half(g.mean(axis=1), sig_mid)
        s50_spreads.append(spread)
        agg_s50s.append(agg)
        print(f"  → staircase σ50 spread {spread:+.3f}  aggregate σ50 {agg:.3f}")

    gamma = np.stack(gammas, 0)  # (A, T, F)
    aspects_arr = np.array(aspects, dtype=np.int32)  # (A, 2) (H, W)

    # Ship the calibration artifact `--cns auto` loads. Default: aspect-averaged
    # single-γ (the recolorer's nearest-aspect select degrades to index 0).
    if args.average_aspects:
        ship_gamma = gamma.mean(axis=0, keepdims=True).astype(np.float32)  # (1,T,F)
        ship_aspects = np.array([[0, 0]], dtype=np.int32)  # aspect-agnostic sentinel
    else:
        ship_gamma, ship_aspects = gamma.astype(np.float32), aspects_arr
    out_path = resolve_under_home(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        gamma=ship_gamma,
        aspects=ship_aspects,
        sigmas=sigmas_ref,
        radial_centers=centers_ref,
        cfg=np.float32(args.cfg),
        flow_shift=np.float32(args.flow_shift),
        steps=np.int32(args.steps),
        source_aspects=aspects_arr,
        averaged=np.bool_(args.average_aspects),
    )
    print(
        f"\nwrote calibration → {out_path}  (gamma {ship_gamma.shape}, "
        f"averaged={args.average_aspects})"
    )

    print("\n=== CNS calibration ===")
    for hw, sp, ag in zip(aspects, s50_spreads, agg_s50s):
        print(f"  {hw[0]}x{hw[1]}: σ50 spread {sp:+.3f}  aggregate {ag:.3f}")
    print(f"  artifact → {out_path}")
    print(
        "  next: A/B `--sampler er_sde --cns auto` vs er_sde white, CMMD + read grids."
    )


if __name__ == "__main__":
    main()
