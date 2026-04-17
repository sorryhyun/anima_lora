#!/usr/bin/env python
"""Channel-dominance analysis for LoRA-adapted Linear inputs in the Anima DiT.

Motivation
----------
GraLoRA (arXiv:2505.20355) argues that LoRA degrades at rank >=64 because a few
"outlier" input channels dominate the gradient of the down-projection, and the
low-rank shared B matrix propagates that distortion across every rank. Their
evidence is LLaMA3-8B layer-1 down-proj: channel ~198 has mean|x| ~100 while
typical channels sit at ~1-5 (20-100x dominance). Their fix — partitioning
weights into k*k adapter sub-blocks — only helps if the same pathology exists
in *your* model.

What this script checks
-----------------------
For each Linear layer that an already-trained LoRA adapter wraps in the DiT,
collect per-input-channel statistics (|x| averaged over tokens) across a small
batch of real dataset samples at several flow-matching timesteps, then report:

  * mean_abs[c] = average |x| over all tokens we saw
  * channel_dominance = max(mean_abs) / median(mean_abs) — the single number
    from the GraLoRA paper. >=10 means severe skew; <3 probably means the
    motivation for GraLoRA doesn't transfer to your model.
  * top-K offending channels and their indices

Stats are grouped by module role (self_attn.q_proj, cross_attn.kv_proj, ...)
so you can see where the worst channel concentration lives — if it's in the
cross-attention inputs (T5 features), that's different from the self-attention
inputs (image patch tokens), and the remediation differs.

Usage
-----
    # With a trained adapter (channel stats from LoRA-wrapped linears' inputs):
    python bench/analyze_lora_input_channels.py \\
        --lora_weight output/anima-tlora-0415-12.safetensors \\
        --out_json bench/results/channel_dominance_0415-12.json

    # Base model only (no LoRA) — use this to separate base-model pathology
    # from LoRA-induced activation drift:
    python bench/analyze_lora_input_channels.py \\
        --out_json bench/results/channel_dominance_base.json

Separating sinks from DC bias
-----------------------------
For each module we also report peak_to_mean = max|x| / mean|x| on the most
dominant channel. Interpretation:
  * ~1-3x  : uniform high values across tokens (DC bias / global offset)
  * ~10-50x: bimodal — a minority of tokens carry the outlier magnitude
  * >=50x  : extreme concentration, consistent with attention-sink /
             register-token behavior (a few tokens act as scratchpads)
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library.anima import weights as anima_utils
from library.log import setup_logging
from networks import lora_anima

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lora_weight",
        default=None,
        help="Optional LoRA adapter to attach before collecting stats. "
        "Omit to analyze the base DiT only.",
    )
    p.add_argument(
        "--dit", default="models/diffusion_models/anima-preview3-base.safetensors"
    )
    p.add_argument(
        "--dataset_dir",
        default="post_image_dataset",
        help="Dir with cached <stem>_*_anima.npz latents and <stem>_anima_te.safetensors",
    )
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument(
        "--sigmas",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated flow-matching sigmas to forward at",
    )
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_k_channels", type=int, default=8)
    p.add_argument("--out_json", default=None)
    p.add_argument(
        "--dump_channel_stats",
        default=None,
        help="Write full per-input-channel mean|x| vectors to a safetensors file "
        "(keyed by LoRA module name, e.g. `lora_unet_blocks_0_self_attn_qkv_proj`) "
        "consumable by `per_channel_scaling=true` / `channel_stats_path=...` in training configs.",
    )
    p.add_argument(
        "--print_top_n_modules",
        type=int,
        default=20,
        help="How many worst-dominance modules to print per group",
    )
    return p.parse_args()


def dump_channel_stats_safetensors(stats, out_path, prefix="lora_unet_"):
    """Save full per-input-channel mean_abs vectors keyed by LoRA module name.

    Each entry in `stats` is keyed by a dot-separated module path (e.g.
    `blocks.0.self_attn.qkv_proj`); we prepend `prefix` and replace dots with
    underscores to match the `lora_name` format used by `networks/lora_anima.py`.
    """
    tensors = {}
    for module_path, stat in stats.items():
        if stat.get("count", 0) == 0:
            continue
        mean_abs = (stat["sum_abs"] / stat["count"]).float().contiguous()
        lora_name = prefix + module_path.replace(".", "_")
        tensors[lora_name] = mean_abs
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, out_path)
    logger.info(
        f"wrote per-channel mean|x| stats for {len(tensors)} modules to {out_path}"
    )


def find_sample_stems(dataset_dir, n, seed):
    te_files = sorted(glob.glob(os.path.join(dataset_dir, "*_anima_te.safetensors")))
    if not te_files:
        raise FileNotFoundError(f"no *_anima_te.safetensors found in {dataset_dir}")

    stems = []
    for te_path in te_files:
        base = os.path.basename(te_path).replace("_anima_te.safetensors", "")
        npz_candidates = glob.glob(os.path.join(dataset_dir, f"{base}_*_anima.npz"))
        if not npz_candidates:
            continue
        stems.append((base, npz_candidates[0], te_path))

    if not stems:
        raise FileNotFoundError(f"no paired latent/TE samples in {dataset_dir}")

    rng = np.random.default_rng(seed)
    if len(stems) > n:
        idx = rng.choice(len(stems), size=n, replace=False)
        stems = [stems[i] for i in sorted(idx.tolist())]
    return stems


def load_latent_npz(npz_path):
    z = np.load(npz_path)
    lat_key = next(
        (k for k in z.files if k.startswith("latents_") and "flip" not in k), None
    )
    if lat_key is None:
        raise KeyError(f"no latents_* key in {npz_path}")
    lat = torch.from_numpy(z[lat_key]).float()  # [C, H, W]
    return lat


def load_cached_te(te_path):
    sd = load_file(te_path)
    key = "crossattn_emb_v0" if "crossattn_emb_v0" in sd else "crossattn_emb"
    if key not in sd:
        raise KeyError(f"no crossattn_emb* key in {te_path}: {list(sd.keys())[:5]}")
    emb = sd[key].float()  # [L, D]
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)  # [1, L, D]
    return emb


def install_channel_hooks(model: torch.nn.Module):
    """Register a forward_pre_hook on every nn.Linear in the model.

    Pre-hooks fire inside nn.Module.__call__ *before* self.forward runs, so they
    capture the Linear's input x whether or not LoRA has monkey-patched forward.

    Returns (stats, handles) — stats maps a cleaned module path to a dict with
    cumulative per-input-channel statistics.
    """
    stats = {}
    handles = []

    def make_hook(stat_ref):
        def hook(_mod, inputs):
            if not inputs:
                return
            x = inputs[0]
            with torch.no_grad():
                xf = x.detach().to(torch.float32).abs()
                xf_flat = xf.reshape(-1, xf.shape[-1])  # [N, C]
                stat_ref["sum_abs"] += xf_flat.sum(dim=0).double().cpu()
                cur_max = xf_flat.max(dim=0).values.double().cpu()
                stat_ref["max_abs"] = torch.maximum(stat_ref["max_abs"], cur_max)
                stat_ref["count"] += xf_flat.shape[0]

        return hook

    for module_path, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        clean_path = module_path.replace("_orig_mod.", "")
        if clean_path in stats:
            continue
        in_dim = module.in_features
        stats[clean_path] = {
            "sum_abs": torch.zeros(in_dim, dtype=torch.float64),
            "max_abs": torch.zeros(in_dim, dtype=torch.float64),
            "count": 0,
            "in_dim": in_dim,
            "module_path": clean_path,
        }
        handles.append(module.register_forward_pre_hook(make_hook(stats[clean_path])))

    logger.info(f"installed forward_pre_hooks on {len(stats)} Linear modules")
    return stats, handles


def classify_module(module_path: str) -> str:
    """Map a dotted module path to a coarse role used for grouping.

    Works for both LoRA key names (`lora_unet_blocks_0_self_attn_qkv_proj`)
    and native module paths (`blocks.0.self_attn.qkv_proj`) by normalizing
    underscores to dots first.
    """
    p = module_path.lower().replace("_", ".")
    if "llm.adapter" in p:
        return "llm_adapter"
    if "self.attn.qkv" in p:
        return "self_attn.qkv_in"
    if "self.attn.output.proj" in p or "self.attn.out.proj" in p:
        return "self_attn.out_in"
    if "cross.attn.q.proj" in p:
        return "cross_attn.q_in"
    if "cross.attn.kv.proj" in p:
        return "cross_attn.kv_in"
    if "cross.attn.output.proj" in p or "cross.attn.out.proj" in p:
        return "cross_attn.out_in"
    if (
        "mlp.layer1" in p
        or "mlp.fc1" in p
        or "mlp.gate.proj" in p
        or "mlp.up.proj" in p
    ):
        return "mlp.layer1_in"
    if "mlp.layer2" in p or "mlp.fc2" in p or "mlp.down.proj" in p:
        return "mlp.layer2_in"
    return "other"


def summarize_module(stat, top_k):
    mean_abs = (stat["sum_abs"] / max(stat["count"], 1)).numpy()
    max_abs = stat["max_abs"].numpy()
    med = float(np.median(mean_abs))
    mx = float(mean_abs.max())
    mx_idx = int(mean_abs.argmax())
    dominance = mx / med if med > 0 else float("inf")
    # peak_to_mean on the most-dominant channel: low (~1-3) means DC bias,
    # high (>50) means a few tokens carry the magnitude (attention sink / register).
    peak_to_mean = (
        float(max_abs[mx_idx] / mean_abs[mx_idx])
        if mean_abs[mx_idx] > 0
        else float("inf")
    )
    top_idx = np.argsort(-mean_abs)[:top_k]
    return {
        "in_dim": stat["in_dim"],
        "mean_abs_median": med,
        "mean_abs_max": mx,
        "mean_abs_max_channel": mx_idx,
        "dominance_ratio": dominance,
        "peak_to_mean_on_top_channel": peak_to_mean,
        "max_abs_on_top_channel": float(max_abs[mx_idx]),
        "top_channels": [
            {
                "channel": int(c),
                "mean_abs": float(mean_abs[c]),
                "max_abs": float(max_abs[c]),
                "peak_to_mean": float(max_abs[c] / mean_abs[c])
                if mean_abs[c] > 0
                else float("inf"),
            }
            for c in top_idx.tolist()
        ],
    }


def print_group_summary(group_name, modules, n_worst):
    rows = sorted(modules, key=lambda m: -m["dominance_ratio"])
    dominances = np.array([m["dominance_ratio"] for m in rows])
    p2m = np.array([m["peak_to_mean_on_top_channel"] for m in rows])
    print(f"\n  [{group_name}]  n={len(rows)}")
    print(
        f"    dominance ratio (mean|x|.max / median):  "
        f"mean={dominances.mean():.2f}  median={np.median(dominances):.2f}  "
        f"p95={np.percentile(dominances, 95):.2f}  max={dominances.max():.2f}"
    )
    print(
        f"    peak_to_mean on top channel              :  "
        f"mean={p2m.mean():.1f}  median={np.median(p2m):.1f}  max={p2m.max():.1f}"
    )
    print(f"    worst {min(n_worst, len(rows))} modules:")
    for m in rows[:n_worst]:
        print(
            f"      {m['name']:<64s}  ratio={m['dominance_ratio']:6.2f}  "
            f"mean|x|={m['mean_abs_max']:8.4f}  max|x|={m['max_abs_on_top_channel']:9.3f}  "
            f"p2m={m['peak_to_mean_on_top_channel']:6.1f}  ch={m['mean_abs_max_channel']}"
        )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    stems = find_sample_stems(args.dataset_dir, args.num_samples, args.seed)
    logger.info(f"selected {len(stems)} samples: {[s[0] for s in stems]}")

    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    logger.info(f"sigmas: {sigmas}")

    logger.info(f"loading DiT from {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.eval().requires_grad_(False)

    if args.lora_weight:
        logger.info(f"loading LoRA adapter from {args.lora_weight}")
        lora_sd = load_file(args.lora_weight)
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
        network, weights_sd = lora_anima.create_network_from_weights(
            multiplier=1.0,
            file=None,
            ae=None,
            text_encoders=[],
            unet=anima,
            weights_sd=lora_sd,
            for_inference=False,  # keep OrthoLoRAModule for T-LoRA
        )
        network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
        info = network.load_state_dict(weights_sd, strict=False)
        if info.missing_keys:
            logger.warning(
                f"missing keys: {len(info.missing_keys)} (first: {info.missing_keys[:3]})"
            )
        if info.unexpected_keys:
            logger.warning(
                f"unexpected keys: {len(info.unexpected_keys)} (first: {info.unexpected_keys[:3]})"
            )
        network.to(device, dtype=torch.bfloat16)
        network.eval()
        for p in network.parameters():
            p.requires_grad_(False)
    else:
        logger.info("no LoRA adapter — analyzing base DiT activations")

    stats, _handles = install_channel_hooks(anima)
    if not stats:
        raise RuntimeError("no Linear modules found to hook")

    logger.info("running forward passes to collect channel stats")
    n_forward = 0
    with torch.no_grad():
        for stem, npz_path, te_path in stems:
            lat = load_latent_npz(npz_path).to(device)  # [C, H, W]
            lat_4d = lat.unsqueeze(0).float()  # [1, C, H, W]
            emb = load_cached_te(te_path).to(device, dtype=torch.bfloat16)  # [1, L, D]
            h_lat, w_lat = lat_4d.shape[-2], lat_4d.shape[-1]
            padding_mask = torch.zeros(
                1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device
            )

            for sigma in sigmas:
                noise = torch.randn_like(lat_4d)
                sv = torch.tensor(sigma, device=device).view(1, 1, 1, 1)
                noisy = ((1.0 - sv) * lat_4d + sv * noise).to(torch.bfloat16)
                noisy_5d = noisy.unsqueeze(2)  # [1, C, 1, H, W]
                t = torch.tensor([sigma], device=device, dtype=torch.bfloat16)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _ = anima(noisy_5d, t, emb, padding_mask=padding_mask)

                n_forward += 1
            logger.info(f"  done sample {stem}  ({len(sigmas)} timesteps)")

    logger.info(f"collected stats from {n_forward} forward passes")

    per_module = {}
    groups = defaultdict(list)
    for key, stat in stats.items():
        summary = summarize_module(stat, args.top_k_channels)
        summary["name"] = key
        summary["module_path"] = stat["module_path"]
        summary["group"] = classify_module(stat["module_path"])
        per_module[key] = summary
        groups[summary["group"]].append(summary)

    all_dominances = np.array([m["dominance_ratio"] for m in per_module.values()])

    tag = (
        os.path.basename(args.lora_weight)
        if args.lora_weight
        else "<base DiT, no LoRA>"
    )
    print("\n" + "=" * 78)
    print(f"Channel-dominance analysis for {tag}")
    print(
        f"  samples: {len(stems)}   sigmas: {sigmas}   "
        f"forward passes: {n_forward}   hooked modules: {len(per_module)}"
    )
    print("  GraLoRA reference: severe ~20-100x, moderate 5-20x, negligible <3x")
    print("=" * 78)
    print("\nOverall dominance ratio distribution (mean|x|.max / median):")
    print(
        f"  mean={all_dominances.mean():.2f}   median={np.median(all_dominances):.2f}   "
        f"p90={np.percentile(all_dominances, 90):.2f}   "
        f"p99={np.percentile(all_dominances, 99):.2f}   max={all_dominances.max():.2f}"
    )

    for group_name in sorted(groups.keys()):
        print_group_summary(group_name, groups[group_name], args.print_top_n_modules)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "lora_weight": args.lora_weight,
            "dit": args.dit,
            "num_samples": len(stems),
            "sample_stems": [s[0] for s in stems],
            "sigmas": sigmas,
            "n_forward_passes": n_forward,
            "overall": {
                "mean_dominance": float(all_dominances.mean()),
                "median_dominance": float(np.median(all_dominances)),
                "p90_dominance": float(np.percentile(all_dominances, 90)),
                "p99_dominance": float(np.percentile(all_dominances, 99)),
                "max_dominance": float(all_dominances.max()),
            },
            "groups": {
                g: {
                    "n": len(ms),
                    "mean_dominance": float(
                        np.mean([m["dominance_ratio"] for m in ms])
                    ),
                    "median_dominance": float(
                        np.median([m["dominance_ratio"] for m in ms])
                    ),
                    "max_dominance": float(max(m["dominance_ratio"] for m in ms)),
                }
                for g, ms in groups.items()
            },
            "per_module": per_module,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out_json}")

    if args.dump_channel_stats:
        dump_channel_stats_safetensors(stats, args.dump_channel_stats)


if __name__ == "__main__":
    main()
