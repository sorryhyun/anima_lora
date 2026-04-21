#!/usr/bin/env python
"""HydraLoRA router diagnostic: σ-correlation + expert balance / collapse.

Two independent diagnostics over the same captured gates:

1. σ-correlation (pre-analysis gate for proposal.md Track B / Phase B0).
   Does the router already route differently per σ bucket? Proposal Option 2
   (σ-conditional router) is only justified if the current router is not
   already implicitly σ-aware.

   Decision tree (from proposal §Track B pre-analysis):
     * Case A (median max-pairwise JS < 0.05)  → implement Option 2
     * Case B (0.05 ≤ median JS ≤ 0.5)         → skip Option 2; prioritize Track A
     * Case C (median JS > 0.5)                → consider Option 3 (full 2D grid)

2. Expert balance / collapse.
   Are all experts actually being used, or has the router collapsed onto a
   subset? Reports per-module normalized entropy of the marginal gate, dead-
   expert count, dominant-top1 fraction, and per-sample routing sharpness.
   Collapse here is measured against the *marginal* expert usage aggregated
   over all (sample, σ) forwards — a module with norm_entropy ≈ 1 uses all
   experts roughly equally; ≈ 0 means one expert has absorbed all mass.

Usage
-----
    python bench/hydralora/analyze_router_sigma_correlation.py \\
        --lora_weight output/anima-hydralora-XXXX.safetensors \\
        --dataset_dir post_image_dataset \\
        --num_samples 32 \\
        --out_json bench/hydralora/results/sigma_correlation_XXXX.json

Notes
-----
- Hooks the existing HydraLoRAModule instances after .apply_to(); no code
  changes in networks/ required.
- Runs forward passes in eval mode with gradients disabled; the gate is
  captured from _compute_gate's return value regardless of self.training.
- σ is sampled uniformly over `--sigmas` (comma list) — caller should match
  the checkpoint's training σ distribution if it matters. Default covers the
  logit-normal bulk.
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
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from library.anima import weights as anima_utils
from library.log import setup_logging
from networks import lora_anima
from networks.lora_modules import HydraLoRAModule

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lora_weight", required=True, help="Trained HydraLoRA safetensors checkpoint"
    )
    p.add_argument(
        "--dit", default="models/diffusion_models/anima-preview3-base.safetensors"
    )
    p.add_argument(
        "--dataset_dir",
        default="post_image_dataset",
        help="Dir with cached <stem>_*_anima.npz latents and <stem>_anima_te.safetensors",
    )
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument(
        "--sigmas",
        default="0.05,0.15,0.3,0.45,0.6,0.75,0.9",
        help="Comma-separated flow-matching sigmas to forward at. Denser is better "
             "for the equal-frequency bucketing to be meaningful.",
    )
    p.add_argument("--num_buckets", type=int, default=3)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default=None)
    p.add_argument(
        "--print_top_n",
        type=int,
        default=10,
        help="Per group, show this many most-σ-correlated modules",
    )
    p.add_argument(
        "--collapse_entropy_threshold",
        type=float,
        default=0.5,
        help="Module is flagged 'collapsed' when marginal-gate normalized entropy < this",
    )
    p.add_argument(
        "--balanced_entropy_threshold",
        type=float,
        default=0.8,
        help="Module is flagged 'balanced' when marginal-gate normalized entropy > this",
    )
    p.add_argument(
        "--dominant_top1_threshold",
        type=float,
        default=0.5,
        help="A forward is 'top-1 dominant' when its max gate weight exceeds this",
    )
    return p.parse_args()


# --------- I/O helpers (copied from bench/analyze_lora_input_channels.py) --------


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
    return torch.from_numpy(z[lat_key]).float()


def load_cached_te(te_path):
    sd = load_file(te_path)
    key = "crossattn_emb_v0" if "crossattn_emb_v0" in sd else "crossattn_emb"
    if key not in sd:
        raise KeyError(f"no crossattn_emb* key in {te_path}: {list(sd.keys())[:5]}")
    emb = sd[key].float()
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)
    return emb


# --------- gate capture ---------


def install_gate_hooks(network):
    """Wrap each HydraLoRAModule._compute_gate to cache (sigma, gate) at each call.

    Returns (captures, restore). `captures` is a dict keyed by module.lora_name:
      {"gate_chunks": [gate tensor per forward], "sigma_chunks": [sigma per forward]}
    `restore()` unwraps in-place.
    """
    captures = {}
    originals = []

    # current σ gets written here from the outer forward loop; each hook reads it.
    shared = {"sigma": None}

    hydra_modules = []
    for module in network.modules():
        if isinstance(module, HydraLoRAModule):
            hydra_modules.append(module)

    if not hydra_modules:
        raise RuntimeError(
            "no HydraLoRAModule instances found — checkpoint may not be HydraLoRA"
        )

    logger.info(f"hooking {len(hydra_modules)} HydraLoRAModule instances")

    def make_wrapper(module, orig):
        name = module.lora_name
        captures[name] = {"gates": [], "sigmas": [], "num_experts": module.num_experts}

        def wrapped(x_lora):
            gate = orig(x_lora)  # (B, num_experts)
            sigma = shared["sigma"]
            if sigma is not None:
                captures[name]["gates"].append(
                    gate.detach().to(torch.float32).cpu()
                )
                # broadcast σ to batch size
                B = gate.shape[0]
                s = torch.full((B,), float(sigma), dtype=torch.float32)
                captures[name]["sigmas"].append(s)
            return gate

        return wrapped

    for module in hydra_modules:
        orig = module._compute_gate
        originals.append((module, orig))
        module._compute_gate = make_wrapper(module, orig)

    def restore():
        for module, orig in originals:
            module._compute_gate = orig

    return captures, shared, restore


# --------- JS divergence ---------


def js_divergence(p, q, eps=1e-12):
    """Jensen–Shannon divergence (base-2 log, range [0, 1])."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log2(p) - np.log2(m)))
    kl_qm = np.sum(q * (np.log2(q) - np.log2(m)))
    return 0.5 * (kl_pm + kl_qm)


def expert_balance_metrics(gates: np.ndarray, dominant_top1_threshold: float):
    """Per-module collapse/balance diagnostics from a (N, E) gate matrix.

    Assumes each row already sums to ~1 (softmax output). We renormalize
    defensively for entropy computations so odd shapes can't NaN the result.
    """
    N, E = gates.shape
    eps = 1e-12

    # marginal expert usage — average gate weight per expert across all forwards
    mean_gate = gates.mean(axis=0).astype(np.float64)  # (E,)

    # normalized entropy of the marginal. 1 = uniform usage, 0 = one expert.
    p = mean_gate / (mean_gate.sum() + eps)
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log2(p))
    max_H = np.log2(E) if E > 1 else 1.0
    norm_entropy = float(H / max_H)

    # coefficient of variation of marginal (alt-view of balance; 0 = uniform)
    cv = float(mean_gate.std(ddof=0) / (mean_gate.mean() + eps))

    # dead experts: those whose marginal sits below half-uniform
    dead_threshold = 0.5 / E
    dead_mask = mean_gate < dead_threshold
    dead_count = int(dead_mask.sum())

    # per-forward routing sharpness — mean of normalized entropy over rows.
    # Low value + low norm_entropy = full collapse; low + high = decisive-but-balanced.
    row_sums = gates.sum(axis=1, keepdims=True)
    psg = gates / (row_sums + eps)
    psg = np.clip(psg, eps, 1.0)
    per_row_H = -np.sum(psg * np.log2(psg), axis=1)
    mean_per_sample_norm_entropy = float(per_row_H.mean() / max_H)

    # fraction of forwards where a single expert takes >threshold of the mass
    top1 = gates.max(axis=1)
    dominant_frac = float((top1 > dominant_top1_threshold).mean())

    return {
        "mean_gate": mean_gate.tolist(),
        "normalized_entropy": norm_entropy,
        "gate_cv": cv,
        "dead_experts": dead_count,
        "mean_per_sample_norm_entropy": mean_per_sample_norm_entropy,
        "dominant_top1_fraction": dominant_frac,
    }


def equal_frequency_buckets(sigmas, num_buckets):
    """Return bucket index for each sigma by quantile (equal-frequency bins)."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    quantiles = np.quantile(sigmas, np.linspace(0, 1, num_buckets + 1))
    # np.digitize returns 1..K for K bin edges interior; we want 0..K-1.
    idx = np.digitize(sigmas, quantiles[1:-1], right=False)
    return idx, quantiles


def classify_module(lora_name: str) -> str:
    """Coarse grouping for reporting — mirrors bench/analyze_lora_input_channels.py."""
    p = lora_name.lower().replace("_", ".")
    if "self.attn.qkv" in p:
        return "self_attn.qkv"
    if "self.attn.output.proj" in p or "self.attn.out.proj" in p:
        return "self_attn.out"
    if "cross.attn.q.proj" in p:
        return "cross_attn.q"
    if "cross.attn.kv.proj" in p:
        return "cross_attn.kv"
    if "cross.attn.output.proj" in p or "cross.attn.out.proj" in p:
        return "cross_attn.out"
    if "mlp.layer1" in p or "mlp.fc1" in p or "mlp.gate.proj" in p or "mlp.up.proj" in p:
        return "mlp.layer1"
    if "mlp.layer2" in p or "mlp.fc2" in p or "mlp.down.proj" in p:
        return "mlp.layer2"
    return "other"


def block_depth(lora_name: str) -> int:
    """Extract DiT block index from lora_unet_blocks_<N>_... names; -1 if not a block."""
    parts = lora_name.split("_")
    try:
        i = parts.index("blocks")
        return int(parts[i + 1])
    except (ValueError, IndexError):
        return -1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    stems = find_sample_stems(args.dataset_dir, args.num_samples, args.seed)
    logger.info(f"selected {len(stems)} samples")

    sigma_list = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    logger.info(f"sigmas: {sigma_list}")

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

    logger.info(f"loading HydraLoRA adapter from {args.lora_weight}")
    lora_sd = load_file(args.lora_weight)
    lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
    network, weights_sd = lora_anima.create_network_from_weights(
        multiplier=1.0,
        file=None,
        ae=None,
        text_encoders=[],
        unet=anima,
        weights_sd=lora_sd,
        for_inference=False,
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    if info.missing_keys:
        logger.warning(f"missing keys: {len(info.missing_keys)}")
    if info.unexpected_keys:
        logger.warning(f"unexpected keys: {len(info.unexpected_keys)}")
    network.to(device, dtype=torch.bfloat16)
    network.eval()
    for p in network.parameters():
        p.requires_grad_(False)

    captures, shared, restore = install_gate_hooks(network)

    logger.info("running forward passes to collect gates")
    n_forward = 0
    with torch.no_grad():
        for stem, npz_path, te_path in stems:
            lat = load_latent_npz(npz_path).to(device)
            lat_4d = lat.unsqueeze(0).float()
            emb = load_cached_te(te_path).to(device, dtype=torch.bfloat16)
            h_lat, w_lat = lat_4d.shape[-2], lat_4d.shape[-1]
            padding_mask = torch.zeros(
                1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device
            )

            for sigma in sigma_list:
                noise = torch.randn_like(lat_4d)
                sv = torch.tensor(sigma, device=device).view(1, 1, 1, 1)
                noisy = ((1.0 - sv) * lat_4d + sv * noise).to(torch.bfloat16)
                noisy_5d = noisy.unsqueeze(2)
                t = torch.tensor([sigma], device=device, dtype=torch.bfloat16)

                shared["sigma"] = sigma
                # Propagate σ to every HydraLoRA module with σ-routing enabled.
                # Without this the router would receive zero-padding for the
                # σ-feature slice and JS would be trivially 0 regardless of
                # how well-trained the σ weights are. batch dim=1 here.
                sigma_tensor = torch.tensor([sigma], device=device, dtype=torch.float32)
                if hasattr(network, "set_sigma"):
                    network.set_sigma(sigma_tensor)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _ = anima(noisy_5d, t, emb, padding_mask=padding_mask)
                n_forward += 1
            logger.info(f"  done {stem}  ({len(sigma_list)} sigmas)")
    shared["sigma"] = None
    if hasattr(network, "clear_sigma"):
        network.clear_sigma()
    restore()

    logger.info(f"captured gates from {n_forward} forward passes")

    # --------- per-module JS analysis ---------
    per_module = {}
    for name, cap in captures.items():
        if not cap["gates"]:
            continue
        gates = torch.cat(cap["gates"], dim=0).numpy()  # (N, E)
        sigmas = torch.cat(cap["sigmas"], dim=0).numpy()  # (N,)

        bucket_idx, edges = equal_frequency_buckets(sigmas, args.num_buckets)
        per_bucket_mean = []
        bucket_counts = []
        for b in range(args.num_buckets):
            mask = bucket_idx == b
            bucket_counts.append(int(mask.sum()))
            if mask.sum() == 0:
                per_bucket_mean.append(
                    np.full(cap["num_experts"], 1.0 / cap["num_experts"])
                )
            else:
                per_bucket_mean.append(gates[mask].mean(axis=0))
        per_bucket_mean = np.stack(per_bucket_mean, axis=0)  # (K, E)

        # pairwise JS across buckets
        pairwise = []
        for i in range(args.num_buckets):
            for j in range(i + 1, args.num_buckets):
                pairwise.append(
                    js_divergence(per_bucket_mean[i], per_bucket_mean[j])
                )
        max_pairwise = float(max(pairwise)) if pairwise else 0.0

        balance = expert_balance_metrics(gates, args.dominant_top1_threshold)

        per_module[name] = {
            "name": name,
            "group": classify_module(name),
            "block": block_depth(name),
            "num_experts": cap["num_experts"],
            "num_samples": int(len(sigmas)),
            "bucket_edges": edges.tolist(),
            "bucket_counts": bucket_counts,
            "per_bucket_mean_gate": per_bucket_mean.tolist(),
            "pairwise_js": [float(x) for x in pairwise],
            "max_pairwise_js": max_pairwise,
            **balance,
        }

    all_js = np.array([m["max_pairwise_js"] for m in per_module.values()])

    # --------- report ---------
    print("\n" + "=" * 78)
    print(f"HydraLoRA router σ-correlation  —  {os.path.basename(args.lora_weight)}")
    print(
        f"  modules: {len(per_module)}   samples: {len(stems)}   sigmas: {len(sigma_list)}  "
        f"  forward passes: {n_forward}"
    )
    print("=" * 78)

    if len(all_js) == 0:
        print("No HydraLoRA modules captured gates — nothing to analyze.")
        return

    print("\nMax-pairwise JS divergence across σ buckets, per module:")
    print(
        f"  mean={all_js.mean():.4f}   median={np.median(all_js):.4f}   "
        f"p90={np.percentile(all_js, 90):.4f}   max={all_js.max():.4f}"
    )

    median = float(np.median(all_js))
    print("\nDecision tree (proposal Track B / Phase B0):")
    if median < 0.05:
        case = "A"
        msg = "implement Option 2 (σ-conditional router); explicit σ is free gain"
    elif median > 0.5:
        case = "C"
        msg = "consider Option 3 (full 2D grid); router is already strongly σ-specialized"
    else:
        case = "B"
        msg = "skip Option 2; router is already implicitly σ-aware — prioritize Track A"
    print(f"  → Case {case}  (median={median:.4f}): {msg}")

    # per-group summary
    groups = defaultdict(list)
    for m in per_module.values():
        groups[m["group"]].append(m)

    print("\nPer-group summary (max-pairwise JS):")
    for g in sorted(groups.keys()):
        rows = sorted(groups[g], key=lambda r: -r["max_pairwise_js"])
        js_arr = np.array([r["max_pairwise_js"] for r in rows])
        print(
            f"  [{g:<16s}] n={len(rows):3d}  "
            f"mean={js_arr.mean():.4f}  median={np.median(js_arr):.4f}  "
            f"max={js_arr.max():.4f}"
        )
        for r in rows[: args.print_top_n]:
            blk = f"blk{r['block']:02d}" if r["block"] >= 0 else "----"
            print(f"      {blk}  JS={r['max_pairwise_js']:.4f}  {r['name']}")

    # per-block-depth aggregation (Open Question #2 in proposal)
    by_block = defaultdict(list)
    for m in per_module.values():
        if m["block"] >= 0:
            by_block[m["block"]].append(m["max_pairwise_js"])
    if by_block:
        print("\nPer-block-depth median JS (Open Q #2: is σ-correlation depth-dependent?):")
        for blk in sorted(by_block.keys()):
            vals = np.array(by_block[blk])
            print(
                f"  block {blk:02d}  n={len(vals):3d}  median={np.median(vals):.4f}  "
                f"max={vals.max():.4f}"
            )

    # ==================== expert balance / collapse ====================
    norm_H = np.array([m["normalized_entropy"] for m in per_module.values()])
    dead_per_module = np.array([m["dead_experts"] for m in per_module.values()])
    dom_frac = np.array([m["dominant_top1_fraction"] for m in per_module.values()])
    row_H = np.array([m["mean_per_sample_norm_entropy"] for m in per_module.values()])
    total_experts = sum(m["num_experts"] for m in per_module.values())

    collapsed_modules = [
        m for m in per_module.values()
        if m["normalized_entropy"] < args.collapse_entropy_threshold
    ]
    balanced_modules = [
        m for m in per_module.values()
        if m["normalized_entropy"] > args.balanced_entropy_threshold
    ]

    print("\n" + "=" * 78)
    print("Expert balance / collapse")
    print("=" * 78)
    print(
        f"  norm_entropy (marginal):  mean={norm_H.mean():.4f}  "
        f"median={np.median(norm_H):.4f}  p10={np.percentile(norm_H, 10):.4f}  "
        f"min={norm_H.min():.4f}"
    )
    print(
        f"  dead experts / module:    mean={dead_per_module.mean():.2f}  "
        f"median={int(np.median(dead_per_module))}  max={int(dead_per_module.max())}  "
        f"(total dead across all modules: {int(dead_per_module.sum())} / {total_experts})"
    )
    print(
        f"  top1 dominant fraction:   mean={dom_frac.mean():.4f}  "
        f"median={np.median(dom_frac):.4f}  (threshold={args.dominant_top1_threshold})"
    )
    print(
        f"  per-sample sharpness:     mean_row_norm_entropy={row_H.mean():.4f}  "
        f"(1=uniform, 0=one-hot)"
    )
    print(
        f"  collapsed modules (norm_H < {args.collapse_entropy_threshold}): "
        f"{len(collapsed_modules)} / {len(per_module)}"
    )
    print(
        f"  balanced modules  (norm_H > {args.balanced_entropy_threshold}): "
        f"{len(balanced_modules)} / {len(per_module)}"
    )

    # Coarse verdict on training health
    median_H = float(np.median(norm_H))
    if median_H < args.collapse_entropy_threshold:
        verdict = "COLLAPSED — majority of modules routing to a single expert"
    elif median_H < args.balanced_entropy_threshold:
        verdict = "PARTIAL — router using experts unevenly; specialization or slow warm-up"
    else:
        verdict = "HEALTHY — experts utilized broadly across modules"
    print(f"  → verdict: {verdict}")

    # per-group balance summary
    print("\nPer-group expert balance (marginal norm_entropy):")
    for g in sorted(groups.keys()):
        rows = sorted(groups[g], key=lambda r: r["normalized_entropy"])
        H_arr = np.array([r["normalized_entropy"] for r in rows])
        dead_arr = np.array([r["dead_experts"] for r in rows])
        print(
            f"  [{g:<16s}] n={len(rows):3d}  "
            f"mean_H={H_arr.mean():.4f}  median_H={np.median(H_arr):.4f}  "
            f"min_H={H_arr.min():.4f}  dead/mod={dead_arr.mean():.2f}"
        )
        # show the most collapsed modules in this group
        for r in rows[: args.print_top_n]:
            blk = f"blk{r['block']:02d}" if r["block"] >= 0 else "----"
            print(
                f"      {blk}  H={r['normalized_entropy']:.4f}  "
                f"dead={r['dead_experts']}/{r['num_experts']}  "
                f"dom={r['dominant_top1_fraction']:.2f}  {r['name']}"
            )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(
                {
                    "lora_weight": args.lora_weight,
                    "sigmas": sigma_list,
                    "num_buckets": args.num_buckets,
                    "num_samples": len(stems),
                    "n_forward": n_forward,
                    "overall": {
                        "mean_max_js": float(all_js.mean()),
                        "median_max_js": float(np.median(all_js)),
                        "p90_max_js": float(np.percentile(all_js, 90)),
                        "max_max_js": float(all_js.max()),
                        "case": case,
                    },
                    "expert_balance": {
                        "mean_norm_entropy": float(norm_H.mean()),
                        "median_norm_entropy": float(np.median(norm_H)),
                        "p10_norm_entropy": float(np.percentile(norm_H, 10)),
                        "min_norm_entropy": float(norm_H.min()),
                        "mean_dead_per_module": float(dead_per_module.mean()),
                        "total_dead_experts": int(dead_per_module.sum()),
                        "total_experts": int(total_experts),
                        "mean_dominant_top1_fraction": float(dom_frac.mean()),
                        "mean_per_sample_norm_entropy": float(row_H.mean()),
                        "num_collapsed_modules": len(collapsed_modules),
                        "num_balanced_modules": len(balanced_modules),
                        "collapse_entropy_threshold": args.collapse_entropy_threshold,
                        "balanced_entropy_threshold": args.balanced_entropy_threshold,
                        "dominant_top1_threshold": args.dominant_top1_threshold,
                        "verdict": verdict,
                    },
                    "per_module": per_module,
                },
                f,
                indent=2,
            )
        logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
