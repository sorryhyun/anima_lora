#!/usr/bin/env python
"""Does the shipped LoRA channel-scaling calibration fit the few-step TURBO student?

Context
-------
`channel_scaling_alpha > 0` makes the LoRA family absorb a SmoothQuant per-channel
input rebalance into every `lora_down`, using the vendored
`networks/calibration/channel_stats.safetensors` — per-channel mean|x| of the
DiT's Linear inputs, collected by `analyze_lora_input_channels.py` on the BASE
DiT at *analytically* noised latents `(1-σ)·clean + σ·noise` over a 5-point σ
grid {0.1..0.9}.

The turbo (DP-DMD) student never sees that distribution. It runs a 4-step Euler
rollout *from pure noise on its own trajectory* (its step-k input is the student's
own previous-step output, off the clean-data manifold), at only the 4 student
σ-grid points. The stored A/B result `project_channel_scaling_hurts_turbo`
attributes the α=0.5 harm to "few-step calib mismatch" (among other things) — i.e.
the shipped base-DiT calibration may simply be the *wrong* per-channel scale for
the regime the student actually optimizes in. If that's true, lowering α to 0.01
doesn't find a sweet spot, it just walks back toward the α=0 we already ship.

This probe tests that mismatch DIRECTLY, with no training, by mirroring
`cond_stream_profile.py`'s transfer-efficiency methodology — but on the turbo
student's real rollout activations instead of the EasyControl cond stream.

What it does
------------
1. Loads the base DiT, attaches a trained turbo checkpoint (a normal LoRA;
   `--no_lora` profiles the base DiT over the same rollout as a control), and
   hooks every Linear input.
2. For each sample, rolls the student's `student_steps`-point Euler grid from
   pure noise (cached caption as conditioning) and accumulates per-channel
   mean|x| — the exact tensor channel scaling would rebalance, in the regime it
   would be applied.
3. Per Linear, compares the turbo-rollout profile against the shipped
   (base-DiT) calibration:
     - cosine(turbo, shipped)        — do the per-channel shapes even agree?
     - dom_raw  = max/median mean|x|  — turbo skew (if <~3, scaling is a near-noop
                                        on turbo regardless → keep α=0).
     - dom_self(α) = dominance after a TURBO-derived scale (ideal floor at α).
     - dom_xfer(α) = dominance after the SHIPPED base-derived scale (what
                     turning channel scaling back on actually achieves).
     - xfer_eff(α) = (dom_raw - dom_xfer)/(dom_raw - dom_self): fraction of the
                     achievable flattening the shipped scale captures on turbo.
                     ~1 = transfers; ≤0 = shipped scale doesn't help (or hurts).
4. Sweeps α ∈ {0.01, 0.1, 0.25, 0.5} so you can read off, on the *turbo*
   trajectory, exactly how much each α moves dominance (α=0.01 is expected to be
   a near-noop: 80× → ~75×) and whether the shipped scale diverges from a
   bespoke one. Optionally dumps a turbo-specific calibration.

Decision rule (printed at the end), for whether an α A/B is even worth GPU:
    dom_raw low                       → turbo isn't channel-skewed; keep α=0.
    xfer_eff high + cosine high        → shipped calib fits turbo; an α sweep in
                                         {0.1, 0.25} is the meaningful test (skip
                                         0.01 — it's ≈α=0).
    xfer_eff low / cosine low          → shipped calib is MISCALIBRATED for the
                                         few-step student; don't sweep the shipped
                                         scale — either retrain a turbo calib (the
                                         dumped file) or keep α=0. This corroborates
                                         the "few-step calib mismatch" diagnosis.

Usage
-----
    python bench/channel_stats/turbo_rollout_profile.py --per_artist \
        --lora_weight output/ckpt/anima_turbo_N2_noch_500.safetensors \
        --dump_turbo_stats bench/channel_stats/results/turbo_channel_stats.safetensors \
        --out_json bench/channel_stats/results/turbo_rollout_profile.json
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file

from bench._anima import DEFAULT_DIT
from library.anima import weights as anima_utils
from library.inference.sampling import get_timesteps_sigmas
from library.log import setup_logging
from networks import lora_anima

# Reuse the base collector's dataset/hook/dump helpers verbatim — same stems,
# same key/fused-mirror convention, so the turbo dump is drop-in swappable with
# the shipped main-stream file.
from bench.channel_stats.analyze_lora_input_channels import (
    classify_module,
    dump_channel_stats_safetensors,
    find_sample_stems,
    install_channel_hooks,
    load_cached_te,
    load_latent_npz,
)

setup_logging()
logger = logging.getLogger(__name__)

_SHIPPED_STATS = "networks/calibration/channel_stats.safetensors"
_DEFAULT_LORA = "output/ckpt/anima_turbo_N2_noch_500.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default=DEFAULT_DIT)
    p.add_argument(
        "--lora_weight",
        default=_DEFAULT_LORA,
        help="Trained turbo checkpoint (a normal LoRA). Use the α=0 'noch' "
        "student so the profiled trajectory is the production regime an α>0 run "
        "would rebalance. Pass '' / --no_lora to profile the base DiT over the "
        "same rollout (control: isolates the σ-grid+trajectory effect from "
        "LoRA weight drift).",
    )
    p.add_argument(
        "--no_lora",
        action="store_true",
        help="Skip the adapter — profile the base DiT over the few-step rollout.",
    )
    p.add_argument("--dataset_dir", default="post_image_dataset/lora")
    p.add_argument("--num_samples", type=int, default=48)
    p.add_argument("--per_artist", action="store_true")
    p.add_argument("--per_artist_n", type=int, default=1)
    p.add_argument(
        "--student_steps",
        type=int,
        default=4,
        help="N — student Euler steps (turbo.toml student_steps; default 4).",
    )
    p.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="σ-schedule shift for the student Euler grid (turbo.toml flow_shift).",
    )
    p.add_argument(
        "--alphas",
        default="0.01,0.1,0.25,0.5",
        help="channel_scaling_alpha values to evaluate transfer at.",
    )
    p.add_argument(
        "--headline_alpha",
        type=float,
        default=0.5,
        help="α used for the per-group table (default 0.5, the known-bad anchor).",
    )
    p.add_argument("--shipped_stats", default=_SHIPPED_STATS)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dump_turbo_stats",
        default=None,
        help="Write the turbo-rollout per-channel mean|x| to a safetensors "
        "(same keys as the shipped file) — a bespoke turbo calibration.",
    )
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def _scale_from_mean_abs(mean_abs: np.ndarray, alpha: float) -> np.ndarray:
    """Replicate factory._load_channel_scales: clamp_min(1e-6).pow(alpha), then
    divide by its (clamped) mean → a pure per-channel rebalance."""
    s = np.clip(mean_abs.astype(np.float64), 1e-6, None) ** alpha
    return s / max(float(s.mean()), 1e-12)


def _dominance(v: np.ndarray) -> float:
    med = float(np.median(v))
    return float(v.max() / med) if med > 0 else float("inf")


def attach_turbo_lora(anima, lora_weight, device):
    """Attach a saved turbo checkpoint as a plain LoRA (turbo output is a normal
    LoRA — single head when per_step_expert=false, which is the default)."""
    from safetensors import safe_open

    with safe_open(lora_weight, framework="pt") as f:
        lora_metadata = dict(f.metadata() or {})
    lora_sd = load_file(lora_weight)
    lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
    network, weights_sd = lora_anima.create_network_from_weights(
        multiplier=1.0,
        file=None,
        ae=None,
        text_encoders=[],
        unet=anima,
        weights_sd=lora_sd,
        metadata=lora_metadata,
        for_inference=False,
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    if info.missing_keys:
        logger.warning(
            f"missing keys: {len(info.missing_keys)} (first: {info.missing_keys[:3]})"
        )
    if info.unexpected_keys:
        logger.warning(
            f"unexpected keys: {len(info.unexpected_keys)} "
            f"(first: {info.unexpected_keys[:3]})"
        )
    network.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return network


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    use_lora = not args.no_lora and bool(args.lora_weight)

    stems = find_sample_stems(
        args.dataset_dir,
        args.num_samples,
        args.seed,
        per_artist=args.per_artist,
        per_artist_n=args.per_artist_n,
    )
    logger.info(f"selected {len(stems)} samples")

    # Student Euler grid (σ: 1 → 0, student_steps+1 points). Step i integrates
    # from sigmas[i] to sigmas[i+1]; identical construction to the turbo loop.
    student_sigmas = get_timesteps_sigmas(args.student_steps, args.flow_shift, "cpu")[
        1
    ].tolist()
    logger.info(
        f"student σ grid ({args.student_steps} steps): "
        f"{['%.3f' % s for s in student_sigmas]}"
    )

    logger.info(f"loading DiT from {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.eval().requires_grad_(False)
    anima.to(device)

    if use_lora:
        logger.info(f"attaching turbo LoRA: {args.lora_weight}")
        attach_turbo_lora(anima, args.lora_weight, device)
    else:
        logger.info("no LoRA — profiling base DiT over the few-step rollout (control)")

    stats, _handles = install_channel_hooks(anima)
    if not stats:
        raise RuntimeError("no Linear modules found to hook")

    logger.info("rolling student trajectories to collect channel stats")
    n_forward = 0
    with torch.no_grad():
        for stem, npz_path, te_path in stems:
            lat = load_latent_npz(npz_path).to(device)  # [C, H, W]
            C, H, W = lat.shape
            emb = load_cached_te(te_path).to(device, dtype=torch.bfloat16)  # [1,L,D]
            pad = torch.zeros(1, 1, H, W, dtype=torch.bfloat16, device=device)

            # Pure-noise start at the cached latent's bucket shape, then roll the
            # student's own Euler trajectory (NOT analytically noised real data).
            x = torch.randn(1, C, H, W, device=device, dtype=torch.float32)
            for i in range(args.student_steps):
                s_i = student_sigmas[i]
                s_next = student_sigmas[i + 1]
                t = torch.tensor([s_i], device=device, dtype=torch.bfloat16)
                x_in = x.to(torch.bfloat16).unsqueeze(2)  # [1,C,1,H,W]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    v = anima(x_in, t, emb, padding_mask=pad).squeeze(2)
                x = x - (s_i - s_next) * v.float()
                n_forward += 1
            logger.info(f"  rolled {stem} ({args.student_steps} steps)")

    logger.info(f"collected stats from {n_forward} forward passes")

    # ---- compare against the shipped (base-DiT) calibration ----
    shipped = load_file(args.shipped_stats)
    logger.info(f"loaded shipped stats: {len(shipped)} keys from {args.shipped_stats}")

    per_module = {}
    groups = defaultdict(list)
    for path, st in stats.items():
        if st.get("count", 0) == 0:
            continue
        turbo_mean = (st["sum_abs"] / st["count"]).numpy()
        lora_key = "lora_unet_" + path.replace(".", "_")
        if lora_key not in shipped:
            continue
        main_mean = shipped[lora_key].float().numpy()
        if main_mean.shape != turbo_mean.shape:
            logger.warning(
                f"{path}: shape mismatch turbo{turbo_mean.shape} vs "
                f"shipped{main_mean.shape}"
            )
            continue

        dom_raw = _dominance(turbo_mean)
        cos = float(
            np.dot(turbo_mean, main_mean)
            / (np.linalg.norm(turbo_mean) * np.linalg.norm(main_mean) + 1e-12)
        )
        rec = {
            "name": lora_key,
            "module_path": path,
            "group": classify_module(path),
            "in_dim": int(turbo_mean.shape[0]),
            "cosine_turbo_shipped": cos,
            "dom_raw": dom_raw,
        }
        for a in alphas:
            s_turbo = _scale_from_mean_abs(turbo_mean, a)
            s_main = _scale_from_mean_abs(main_mean, a)
            dom_self = _dominance(turbo_mean / s_turbo)
            dom_xfer = _dominance(turbo_mean / s_main)
            denom = dom_raw - dom_self
            rec[f"dom_self@{a}"] = dom_self
            rec[f"dom_xfer@{a}"] = dom_xfer
            rec[f"xfer_eff@{a}"] = (
                (dom_raw - dom_xfer) / denom if denom > 1e-9 else float("nan")
            )
        per_module[lora_key] = rec
        groups[rec["group"]].append(rec)

    def _med(rows, k):
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(vals)) if vals else float("nan")

    all_rows = list(per_module.values())
    tag = os.path.basename(args.lora_weight) if use_lora else "<base DiT, no LoRA>"
    ha = args.headline_alpha

    print("\n" + "=" * 94)
    print("Turbo few-step rollout channel profile vs shipped base-DiT calibration")
    print(
        f"  adapter={tag}  samples={len(stems)}  student_steps={args.student_steps}  "
        f"flow_shift={args.flow_shift}"
    )
    print(
        f"  dom_raw: turbo skew (max/median mean|x|).  xfer_eff@{ha}: 1.0 = shipped "
        f"scale flattens turbo"
    )
    print(
        "           as well as a bespoke calib; ≤0 = shipped scale doesn't help "
        "(few-step mismatch)."
    )
    print("=" * 94)
    print(
        f"  {'group':<20s} {'n':>3s} {'cosine':>7s} {'dom_raw':>8s} "
        f"{'dom_self':>8s} {'dom_xfer':>8s} {'xfer_eff':>8s}   (@α={ha})"
    )
    for g in sorted(groups):
        rows = groups[g]
        print(
            f"  {g:<20s} {len(rows):>3d} {_med(rows, 'cosine_turbo_shipped'):>7.3f} "
            f"{_med(rows, 'dom_raw'):>8.2f} {_med(rows, f'dom_self@{ha}'):>8.2f} "
            f"{_med(rows, f'dom_xfer@{ha}'):>8.2f} {_med(rows, f'xfer_eff@{ha}'):>8.2f}"
        )
    o_dom = _med(all_rows, "dom_raw")
    o_cos = _med(all_rows, "cosine_turbo_shipped")
    o_eff = _med(all_rows, f"xfer_eff@{ha}")
    print("-" * 94)
    print(
        f"  {'OVERALL':<20s} {len(all_rows):>3d} {o_cos:>7.3f} {o_dom:>8.2f} "
        f"{_med(all_rows, f'dom_self@{ha}'):>8.2f} "
        f"{_med(all_rows, f'dom_xfer@{ha}'):>8.2f} {o_eff:>8.2f}"
    )

    # ---- α-sweep leverage table: what each α ACTUALLY does on the turbo traj ----
    print("\nα-sweep (median over modules) — how much each α moves turbo dominance:")
    print(
        f"  {'alpha':>6s} {'dom_self':>9s} {'dom_xfer':>9s} {'xfer_eff':>9s}   "
        f"(dom_raw={o_dom:.2f}; α=0.01 ≈ no-op confirms 'weaker≈α=0')"
    )
    for a in alphas:
        print(
            f"  {a:>6.2f} {_med(all_rows, f'dom_self@{a}'):>9.2f} "
            f"{_med(all_rows, f'dom_xfer@{a}'):>9.2f} "
            f"{_med(all_rows, f'xfer_eff@{a}'):>9.2f}"
        )

    # ---- verdict ----
    print("\nVERDICT:")
    if o_dom < 3.0:
        verdict = (
            f"turbo rollout is NOT channel-skewed (overall dom_raw={o_dom:.2f} < 3) "
            "— channel scaling is a near-noop on the few-step student regardless of "
            "α. Keep α=0; an α sweep can't help."
        )
    elif o_eff >= 0.8 and o_cos >= 0.9:
        verdict = (
            f"shipped base-DiT calib TRANSFERS to turbo (xfer_eff@{ha}={o_eff:.2f}, "
            f"cosine={o_cos:.2f}) — calib mismatch is NOT the cause of the α=0.5 "
            "harm. A training A/B at α∈{0.1,0.25} is the meaningful next test "
            "(skip 0.01 — it's ≈α=0; see the sweep table)."
        )
    else:
        verdict = (
            f"shipped calib is MISCALIBRATED for the few-step student "
            f"(xfer_eff@{ha}={o_eff:.2f}, cosine={o_cos:.2f}) — this corroborates "
            "the 'few-step calib mismatch' diagnosis. Don't sweep the shipped scale; "
            "either retrain a turbo-specific calib (the dumped file) and A/B that, "
            "or keep α=0."
        )
    print(f"  {verdict}")

    if args.dump_turbo_stats:
        dump_channel_stats_safetensors(stats, args.dump_turbo_stats)
        print(f"\n  turbo-specific calibration written to {args.dump_turbo_stats}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "dit": args.dit,
            "lora_weight": args.lora_weight if use_lora else None,
            "num_samples": len(stems),
            "student_steps": args.student_steps,
            "flow_shift": args.flow_shift,
            "student_sigmas": student_sigmas,
            "alphas": alphas,
            "headline_alpha": ha,
            "sample_stems": [s[0] for s in stems],
            "shipped_stats": args.shipped_stats,
            "overall": {
                "dom_raw": o_dom,
                "cosine_turbo_shipped": o_cos,
                f"xfer_eff@{ha}": o_eff,
                "alpha_sweep": {
                    str(a): {
                        "dom_self": _med(all_rows, f"dom_self@{a}"),
                        "dom_xfer": _med(all_rows, f"dom_xfer@{a}"),
                        "xfer_eff": _med(all_rows, f"xfer_eff@{a}"),
                    }
                    for a in alphas
                },
            },
            "groups": {
                g: {
                    "n": len(rs),
                    "cosine": _med(rs, "cosine_turbo_shipped"),
                    "dom_raw": _med(rs, "dom_raw"),
                    f"dom_self@{ha}": _med(rs, f"dom_self@{ha}"),
                    f"dom_xfer@{ha}": _med(rs, f"dom_xfer@{ha}"),
                    f"xfer_eff@{ha}": _med(rs, f"xfer_eff@{ha}"),
                }
                for g, rs in groups.items()
            },
            "per_module": per_module,
            "verdict": verdict,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    main()
