#!/usr/bin/env python
"""EasyControl's cond-stream LoRA input profile against the main-stream channel scaling.

`channel_scaling_alpha > 0` folds a SmoothQuant-style per-channel rebalance into every
`lora_down`, using `networks/calibration/channel_stats.safetensors`: per-channel
mean|x| of the DiT Linear inputs on the main (target) stream, from
`analyze_lora_input_channels.py`. EasyControl's cond LoRA (`_LoRAProj`) targets the same
Linears on the cond stream and reads its own `cond_channel_stats.safetensors`
(`networks/methods/easycontrol.py`); this script measures that profile and regenerates
the file.

A fresh (untrained) EasyControlNetwork runs in eval with `set_cond(clean_latent)`, and
a pre-hook on every `_LoRAProj` input accumulates per-channel mean|x|. The cond stream
is σ-invariant (cond_temb = t_embedder(zeros)) and ΔW = 0 does not change the inputs,
so one forward per sample measures the base-DiT-on-cond regime the shipped calibration
uses.

Per cond Linear, against the shipped main-stream stats:
  - cosine(cond_profile, main_profile)
  - dom_raw: max/median of cond mean|x|; below ~3 scaling is a near no-op on it
  - dom_self / dom_xfer: dominance after a cond-derived / the shipped main-derived scale
  - xfer_eff = (dom_raw - dom_xfer) / (dom_raw - dom_self): the share of the achievable
    flattening the shipped scale captures (~1.0 = as good as a bespoke calibration,
    ≤0 = no help or worse)

    python scripts/calibration/cond_stream_profile.py --per_artist \
        --dump_cond_stats networks/calibration/cond_channel_stats.safetensors

The comparison and verdict print to stdout; ``--dump_cond_stats`` is the only file
written (same keys and fused-qkv mirroring as the shipped file).
"""

import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file


from anima_lora import default_checkpoints
from library.io.cache import (
    load_cached_crossattn_emb,
    load_cached_latents,
)
from library.log import setup_logging
from library.runtime.harness import build_anima
from networks.methods import easycontrol

# The main-stream collector's stem selection and key / fused-qkv convention, so the
# cond dump is a drop-in for the shipped file.
from scripts.calibration.analyze_lora_input_channels import (
    classify_module,
    dump_channel_stats_safetensors,
    find_sample_stems,
)

DEFAULT_DIT = default_checkpoints().dit

setup_logging()
logger = logging.getLogger(__name__)

_SHIPPED_STATS = "networks/calibration/channel_stats.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default=DEFAULT_DIT)
    p.add_argument(
        "--dataset_dir",
        default="post_image_dataset/lora",
        help="Cached <stem>_*_anima.npz + <stem>_anima_te.safetensors. The clean "
        "latent is fed as BOTH target and cond (ref==target). The cond-stream "
        "channel profile is a domain statistic (σ-insensitive — see "
        "project_channel_scaling_sigma_insensitive), so the clean self-cond is a "
        "faithful proxy for any natural-image cond (incl. the sanitize _tags twin).",
    )
    p.add_argument(
        "--cond_dataset_dir",
        default=None,
        help="Optional: draw cond latents from a parallel cache (e.g. the "
        "sanitize cond dir) instead of ref==target. Paired by index order.",
    )
    p.add_argument("--num_samples", type=int, default=48)
    p.add_argument("--per_artist", action="store_true")
    p.add_argument("--per_artist_n", type=int, default=1)
    p.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Target noising σ. The cond stream is σ-invariant; this only "
        "sets the (discarded) target activations, so one σ is enough.",
    )
    p.add_argument("--network_dim", type=int, default=32)
    p.add_argument("--network_alpha", type=float, default=32.0)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="channel_scaling_alpha to evaluate the transfer at (config default 0.5).",
    )
    p.add_argument("--shipped_stats", default=_SHIPPED_STATS)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dump_cond_stats",
        default=None,
        help="Write the cond-stream per-channel mean|x| to a safetensors "
        "(same keys as the shipped file) — the bespoke cond calibration.",
    )
    return p.parse_args()


# cond _LoRAProj instance -> the dotted DiT module path whose input it shares,
# so the dump keys (lora_unet_blocks_N_...) line up with the shipped file.
def _cond_proj_paths(net):
    """Yield (loraproj_module, dotted_module_path) for every cond _LoRAProj."""
    for idx in range(net.num_blocks):
        yield net.cond_lora_qkv[idx], f"blocks.{idx}.self_attn.qkv_proj"
        yield net.cond_lora_o[idx], f"blocks.{idx}.self_attn.output_proj"
        if net.apply_ffn_lora:
            yield net.cond_lora_ffn1[idx], f"blocks.{idx}.mlp.layer1"
            yield net.cond_lora_ffn2[idx], f"blocks.{idx}.mlp.layer2"


def install_cond_proj_hooks(net):
    """forward_pre_hook on each cond _LoRAProj input — same accumulation shape as
    the base collector's stats dict (keyed by dotted module path)."""
    stats = {}
    handles = []

    def make_hook(stat_ref):
        def hook(_mod, inputs):
            if not inputs:
                return
            x = inputs[0]
            with torch.no_grad():
                xf = x.detach().to(torch.float32).abs().reshape(-1, x.shape[-1])
                stat_ref["sum_abs"] += xf.sum(dim=0).double().cpu()
                stat_ref["count"] += xf.shape[0]

        return hook

    for module, path in _cond_proj_paths(net):
        in_dim = module.in_dim
        stats[path] = {
            "sum_abs": torch.zeros(in_dim, dtype=torch.float64),
            "count": 0,
            "in_dim": in_dim,
            "module_path": path,
        }
        handles.append(module.register_forward_pre_hook(make_hook(stats[path])))
    logger.info(f"installed forward_pre_hooks on {len(stats)} cond _LoRAProj inputs")
    return stats, handles


def _scale_from_mean_abs(mean_abs: np.ndarray, alpha: float) -> np.ndarray:
    """Replicate factory._load_channel_scales exactly: clamp_min(1e-6).pow(alpha),
    then divide by its (clamped) mean → a pure per-channel rebalance."""
    s = np.clip(mean_abs.astype(np.float64), 1e-6, None) ** alpha
    return s / max(float(s.mean()), 1e-12)


def _dominance(v: np.ndarray) -> float:
    med = float(np.median(v))
    return float(v.max() / med) if med > 0 else float("inf")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    stems = find_sample_stems(
        args.dataset_dir,
        args.num_samples,
        args.seed,
        per_artist=args.per_artist,
        per_artist_n=args.per_artist_n,
    )
    cond_stems = None
    if args.cond_dataset_dir:
        cond_stems = find_sample_stems(
            args.cond_dataset_dir,
            args.num_samples,
            args.seed,
            per_artist=args.per_artist,
            per_artist_n=args.per_artist_n,
        )
        n = min(len(stems), len(cond_stems))
        stems, cond_stems = stems[:n], cond_stems[:n]
    logger.info(
        f"{len(stems)} samples; cond={'paired dir' if cond_stems else 'ref==target'}"
    )

    logger.info(f"loading base DiT from {args.dit}")
    args.device = str(device)
    args.dtype = "bf16"
    anima = build_anima(args, dit_path=args.dit, adapter=None, train_mode=False).anima

    net = easycontrol.create_network(
        1.0,
        args.network_dim,
        args.network_alpha,
        None,
        [],
        anima,
        apply_ffn_lora=1,
        cond_res_scale=1.0,
    )
    net.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    net.to(device).eval().requires_grad_(False)

    stats, _handles = install_cond_proj_hooks(net)

    logger.info("running cond-active forward passes")
    sv = torch.tensor(args.sigma, device=device).view(1, 1, 1, 1)
    with torch.no_grad():
        for i, (stem, npz_path, te_path) in enumerate(stems):
            lat = (  # [1,C,H,W]
                load_cached_latents(npz_path)[0].to(device).unsqueeze(0).float()
            )
            emb = (
                load_cached_crossattn_emb(te_path)
                .unsqueeze(0)
                .to(device, dtype=torch.bfloat16)
            )
            H, W = lat.shape[-2], lat.shape[-1]
            pad = torch.zeros(1, 1, H, W, dtype=torch.bfloat16, device=device)

            if cond_stems is not None:
                cond_lat = (
                    load_cached_latents(cond_stems[i][1])[0]
                    .to(device)
                    .unsqueeze(0)
                    .float()
                )
            else:
                cond_lat = lat
            net.set_cond(cond_lat.to(torch.bfloat16))  # clean cond at t=0

            noisy = ((1.0 - sv) * lat + sv * torch.randn_like(lat)).to(torch.bfloat16)
            t = torch.tensor([args.sigma], device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _ = anima(noisy.unsqueeze(2), t, emb, padding_mask=pad)
            logger.info(f"  done {stem} ({i + 1}/{len(stems)})")

    # Compare against the shipped main-stream calibration.
    shipped = load_file(args.shipped_stats)
    logger.info(
        f"loaded shipped main-stream stats: {len(shipped)} keys from {args.shipped_stats}"
    )

    per_module = {}
    groups = defaultdict(list)
    for path, st in stats.items():
        if st["count"] == 0:
            logger.warning(f"{path}: never invoked — skipping")
            continue
        cond_mean = (st["sum_abs"] / st["count"]).numpy()
        lora_key = "lora_unet_" + path.replace(".", "_")
        if lora_key not in shipped:
            logger.warning(f"{path}: no shipped key {lora_key} — skipping comparison")
            continue
        main_mean = shipped[lora_key].float().numpy()
        if main_mean.shape != cond_mean.shape:
            logger.warning(
                f"{path}: shape mismatch cond{cond_mean.shape} vs main{main_mean.shape}"
            )
            continue

        s_cond = _scale_from_mean_abs(cond_mean, args.alpha)
        s_main = _scale_from_mean_abs(main_mean, args.alpha)
        dom_raw = _dominance(cond_mean)
        dom_self = _dominance(cond_mean / s_cond)
        dom_xfer = _dominance(cond_mean / s_main)
        denom = dom_raw - dom_self
        xfer_eff = (dom_raw - dom_xfer) / denom if denom > 1e-9 else float("nan")
        cos = float(
            np.dot(cond_mean, main_mean)
            / (np.linalg.norm(cond_mean) * np.linalg.norm(main_mean) + 1e-12)
        )
        rec = {
            "name": lora_key,
            "module_path": path,
            "group": classify_module(path),
            "in_dim": int(cond_mean.shape[0]),
            "cosine_cond_main": cos,
            "dom_raw": dom_raw,
            "dom_self": dom_self,
            "dom_xfer": dom_xfer,
            "xfer_efficiency": xfer_eff,
        }
        per_module[lora_key] = rec
        groups[rec["group"]].append(rec)

    def _med(rows, k):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.median(vals)) if vals else float("nan")

    print("\n" + "=" * 90)
    print("EasyControl cond-stream channel profile vs shipped main-stream calibration")
    print(
        f"  samples={len(stems)}  alpha={args.alpha}  cond={'paired' if cond_stems else 'ref==target'}"
    )
    print(
        "  dom_raw: cond skew (max/median mean|x|).  xfer_eff: 1.0=shipped scale flattens"
    )
    print(
        "           cond as well as a bespoke calib; <=0 = shipped scale doesn't help cond."
    )
    print("=" * 90)
    print(
        f"  {'group':<20s} {'n':>3s} {'cosine':>7s} {'dom_raw':>8s} {'dom_self':>8s} {'dom_xfer':>8s} {'xfer_eff':>8s}"
    )
    for g in sorted(groups):
        rows = groups[g]
        print(
            f"  {g:<20s} {len(rows):>3d} {_med(rows, 'cosine_cond_main'):>7.3f} "
            f"{_med(rows, 'dom_raw'):>8.2f} {_med(rows, 'dom_self'):>8.2f} "
            f"{_med(rows, 'dom_xfer'):>8.2f} {_med(rows, 'xfer_efficiency'):>8.2f}"
        )
    all_rows = list(per_module.values())
    o_dom = _med(all_rows, "dom_raw")
    o_cos = _med(all_rows, "cosine_cond_main")
    o_eff = _med(all_rows, "xfer_efficiency")
    print("-" * 90)
    print(
        f"  {'OVERALL':<20s} {len(all_rows):>3d} {o_cos:>7.3f} {o_dom:>8.2f} "
        f"{'':>8s} {_med(all_rows, 'dom_xfer'):>8.2f} {o_eff:>8.2f}"
    )

    print("\nVERDICT:")
    if o_dom < 3.0:
        verdict = (
            "cond stream is NOT channel-skewed (overall dom_raw < 3) — channel "
            "scaling is a near-noop on the cond stream. Don't bother wiring it."
        )
    elif o_eff >= 0.8 and o_cos >= 0.9:
        verdict = (
            "shipped main-stream calibration TRANSFERS (high xfer_eff + cosine) — "
            "wire channel scaling into _LoRAProj using the vendored channel_stats."
        )
    elif o_eff >= 0.8:
        verdict = (
            "shipped scale flattens cond well (high xfer_eff) despite modest cosine — "
            "reusing the vendored file is fine; the dumped cond stats are the safer pick."
        )
    else:
        verdict = (
            "cond profile DIVERGES (low xfer_eff) — reusing the shipped file would "
            "mis-scale the cond stream. Use the dumped cond_channel_stats instead."
        )
    print(f"  {verdict}")

    if args.dump_cond_stats:
        dump_channel_stats_safetensors(stats, args.dump_cond_stats)
        print(f"\n  cond-specific calibration written to {args.dump_cond_stats}")


if __name__ == "__main__":
    main()
