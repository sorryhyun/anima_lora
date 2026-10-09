#!/usr/bin/env python3
"""Merge several LoRA adapters into a SINGLE LoRA adapter (LoRA ⊕ LoRA → LoRA).

Unlike ``merge_to_dit.py`` (which bakes one adapter into a full ~GB DiT), this
combines N adapters into one small ``.safetensors`` usable via the normal
``--lora_weight`` inference path. The merge is **exact** — no SVD/truncation:

    ΔW = Σ_i w_i · (α_i/r_i) · up_i @ down_i

is reproduced by row/col concatenation of the rank-r factors:

    down_merged = vstack(down_i)                 # (Σr_i, in)
    up_merged   = hstack( (α_i/r_i)·w_i · up_i )  # (out, Σr_i)
    alpha_merged = Σr_i   →  effective scale 1, so multiplier=1.0 reproduces Σ w_i ΔW_i

The merged rank grows to Σr_i (e.g. 3×32 = 96) — fine for inference, which runs
full rank. Per-LoRA strength via --weights. Modules present in only some inputs
contribute their own block (union of stems). Heterogeneous α/r across inputs is
handled because each input's scale is folded into its own up factor.

--normalize (default 'global') rescales the merged delta back to single-LoRA
magnitude: a raw sum of N near-orthogonal deltas grows energy ~√N (quadrature),
overdriving the DiT into high-freq noise. 'global' applies one scalar so the
global RMS ‖ΔW‖_F matches the average single-LoRA norm (≈1/√N when orthogonal,
→1/N when similar); 'off' reproduces the raw exact-concat sum.

--adaln (default 'average') merges the adaln modulation modules
(``adaln_modulation_*_2``) as a weighted average instead: their input is a
σ-only vector, so each input's adaln delta is a global per-channel
shift/scale/gate bias on every image, not an orthogonally addressed term.
Averaged modules are left out of the --normalize energy budget. 'sum' treats
them like every other module.

Usage:
    uv run python scripts/toolkits/merge_loras.py \
        output/ckpt/anima_artist1.safetensors \
        output/ckpt/anima_artist2.safetensors \
        output/ckpt/anima_artist3.safetensors \
        --out output/ckpt/anima_artist123_merged.safetensors
    # weighted:  --weights 1.0,0.7,0.7
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library.anima import merge_analysis
from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_lora(path: str):
    """Return (downs, ups, alphas) dicts keyed by module stem; tensors fp32."""
    downs, ups, alphas = {}, {}, {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.endswith(".lora_down.weight"):
                downs[k[: -len(".lora_down.weight")]] = f.get_tensor(k).float()
            elif k.endswith(".lora_up.weight"):
                ups[k[: -len(".lora_up.weight")]] = f.get_tensor(k).float()
            elif k.endswith(".alpha"):
                alphas[k[: -len(".alpha")]] = float(f.get_tensor(k).item())
    return downs, ups, alphas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("loras", nargs="+", help="≥2 LoRA safetensors to merge")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to <first-stem>_merged<N>.safetensors next "
        "to the first input.",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="comma-separated per-LoRA strengths (default all 1.0)",
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="Dry-run: report weight-space interference between the inputs "
        "(pairwise constructive/destructive cosine + worst layers) and exit "
        "WITHOUT writing a merged file.",
    )
    ap.add_argument(
        "--normalize",
        choices=["off", "global", "per_module"],
        default="global",
        help="Rescale the merged delta to single-LoRA magnitude. Summing N "
        "near-orthogonal deltas grows energy ~√N (quadrature), so a raw sum "
        "overdrives the DiT (high-freq noise). 'global' (default) applies one "
        "scalar so the global RMS ‖ΔW‖_F matches the avg single-LoRA norm "
        "(≈1/√N when orthogonal, →1/N when similar); 'per_module' matches each "
        "module exactly but reweights layers; 'off' = raw exact concat sum.",
    )
    ap.add_argument(
        "--adaln",
        choices=["average", "sum"],
        default="average",
        help="How the adaln modulation modules merge: 'average' (default) takes "
        "the weighted mean of the inputs' adaln deltas and skips --normalize; "
        "'sum' concatenates and normalizes them like every other module.",
    )
    args = ap.parse_args()

    if len(args.loras) < 2:
        ap.error("need at least 2 LoRAs to merge")
    if args.out is None:
        first = Path(args.loras[0])
        args.out = first.parent / f"{first.stem}_merged{len(args.loras)}.safetensors"
    weights = (
        [float(x) for x in args.weights.split(",")]
        if args.weights
        else [1.0] * len(args.loras)
    )
    if len(weights) != len(args.loras):
        ap.error(f"--weights has {len(weights)} entries, need {len(args.loras)}")
    out_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    loaded = [load_lora(p) for p in args.loras]
    names = [Path(p).stem for p in args.loras]

    # Weight-space interference: how much the inputs reinforce vs cancel once
    # summed. A one-line summary always; the full report (and an early exit) on
    # --analyze. Cheap — same Gram-trace identity as fro2, no out×in ΔW.
    report = merge_analysis.analyze(loaded, names, weights)
    if args.analyze:
        print(merge_analysis.format_report(report))
        # Machine-readable trailer so the GUI can render a pass/warning banner
        # without re-parsing the human report. Ignored on the CLI.
        print("ANALYZE_RESULT " + json.dumps(report.marker_payload()))
        return 0
    logger.info(report.summary_line())

    logger.info("merging %d LoRAs: %s  weights=%s", len(names), names, weights)

    def fro2(up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        """‖up@down‖_F² via Gram-trace identity (no out×in materialization)."""
        return ((up.T @ up) * (down @ down.T)).sum()

    # union of module stems across all inputs
    stems = sorted({s for downs, _, _ in loaded for s in downs})

    # pass 1: build concatenated factors (scale + weight folded into up) and
    # measure merged vs single-LoRA energy per module.
    built: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    rank_hist: dict[int, int] = {}
    sumsq_actual = 0.0  # Σ_m ‖merged ΔW_m‖²  (at chosen weights)
    sumsq_target = 0.0  # Σ_m (avg single-LoRA ‖ΔW_m‖)²
    averaged: set[str] = set()
    for stem in stems:
        avg = args.adaln == "average" and "adaln_modulation" in stem
        w_sum = sum(w for (downs, _, _), w in zip(loaded, weights) if stem in downs)
        downs_cat, ups_cat, indiv = [], [], []
        for (downs, ups, alphas), w in zip(loaded, weights):
            if stem not in downs:
                continue
            r = downs[stem].shape[0]
            s = alphas.get(stem, float(r)) / r  # α/r
            if avg:
                w = w / w_sum
            downs_cat.append(downs[stem])
            ups_cat.append(ups[stem] * (s * w))  # (α/r)·w folded into up
            indiv.append(s * fro2(ups[stem], downs[stem]).sqrt().item())  # weight-1 ref
        down = torch.cat(downs_cat, dim=0)  # (Σr, in)
        up = torch.cat(ups_cat, dim=1)  # (out, Σr)
        actual = fro2(up, down).item()
        target = sum(indiv) / len(indiv)  # avg single-LoRA norm at this module
        if avg:
            averaged.add(stem)
        else:
            sumsq_actual += actual
            sumsq_target += target * target
        built[stem] = (down, up, target)
        rank = down.shape[0]
        rank_hist[rank] = rank_hist.get(rank, 0) + 1

    # pass 2: resolve normalization scale, then write.
    g_scale = (sumsq_target**0.5) / (sumsq_actual**0.5 + 1e-12)
    if args.normalize == "global":
        logger.info(
            "normalize=global: scale=%.4f (1/√N=%.4f for N=%d)",
            g_scale,
            1.0 / (len(args.loras) ** 0.5),
            len(args.loras),
        )
    merged: dict[str, torch.Tensor] = {}
    if averaged:
        logger.info("adaln=average: %d modules averaged, not normalized", len(averaged))
    for stem, (down, up, target) in built.items():
        if stem in averaged:
            pass
        elif args.normalize == "global":
            up = up * g_scale
        elif args.normalize == "per_module":
            actual = fro2(up, down).sqrt().item()
            up = up * (target / (actual + 1e-12))
        rank = down.shape[0]
        merged[f"{stem}.lora_down.weight"] = down.to(out_dtype)
        merged[f"{stem}.lora_up.weight"] = up.to(out_dtype)
        # alpha = rank → effective (α/r) = 1; multiplier=1.0 applies the merged ΔW
        merged[f"{stem}.alpha"] = torch.tensor(float(rank), dtype=out_dtype)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "merged_from": ",".join(names),
        "merge_weights": ",".join(str(w) for w in weights),
        "merge_kind": "concat_exact",
        "normalize": args.normalize,
        "normalize_global_scale": f"{g_scale:.6f}",
        "adaln_merge": args.adaln,
    }
    save_file(merged, str(args.out), metadata=metadata)
    logger.info(
        "wrote %s  (%d modules, rank histogram %s)",
        args.out,
        len(stems),
        rank_hist,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
