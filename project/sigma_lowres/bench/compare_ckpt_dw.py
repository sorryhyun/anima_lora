"""Paired ΔW cosine comparison between LoRA checkpoints (sigma_lowres Phase 1b).

For every checkpoint pair, computes the cosine between the full adapter
displacements ΔW = (alpha/r)·up@down, accumulated module-by-module in RANK
space — ⟨ΔWa, ΔWb⟩ = s_a·s_b·Σ((UaᵀUb) ⊙ (Da·Dbᵀ)) — so the (out×in) product
is never materialized (the full-ΔW version eats all RAM). Emits the global
cosine plus a per-block depth profile (the demote signature lives in late
blocks — record/report.md §"Phase 1b in-vivo weight-space A/B").

Meaningful on PAIRED runs only (--seed N --paired_step_rng): unpaired arms
sit at the ~0.09–0.10 init/noise-lottery floor regardless of intervention.
Compare ΔW, never raw checkpoints — frozen Ortho/SVD bases mask the lottery
in whole-checkpoint cosines.

Usage::

    python compare_ckpt_dw.py output/ckpt/anima_lora_tenth4s_{base,sigma,yarnsig}.safetensors
    python compare_ckpt_dw.py <ckpts...> --json results/dw.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
from safetensors import safe_open

_BLOCK_RE = re.compile(r"_blocks_(\d+)_")


def load_modules(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """{module: (up, down, scale)} — float64 CPU, scale = alpha/rank."""
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), "pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    mods: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for k, v in tensors.items():
        if not k.endswith(".lora_up.weight"):
            continue
        m = k[: -len(".lora_up.weight")]
        down = tensors.get(f"{m}.lora_down.weight")
        if down is None:
            continue
        up = v.double().flatten(1)  # conv 1x1 tolerated; all Linear here
        down = down.double().flatten(1)
        rank = up.shape[1]
        alpha = tensors.get(f"{m}.alpha")
        scale = (float(alpha) / rank) if alpha is not None else 1.0
        mods[m] = (up, down, scale)
    return mods


def block_of(module: str) -> str:
    m = _BLOCK_RE.search(module)
    return m.group(1) if m else "other"


def pair_stats(a, b) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Per-block (⟨ΔWa,ΔWb⟩, ‖ΔWa‖², ‖ΔWb‖²), keys = block ids + 'other'."""
    ip: dict[str, float] = defaultdict(float)
    na: dict[str, float] = defaultdict(float)
    nb: dict[str, float] = defaultdict(float)
    common = sorted(set(a) & set(b))
    missing = (set(a) | set(b)) - set(common)
    if missing:
        print(
            f"  WARN: {len(missing)} modules not in both checkpoints", file=sys.stderr
        )
    for m in common:
        ua, da, sa = a[m]
        ub, db, sb = b[m]
        blk = block_of(m)
        ip[blk] += sa * sb * float(((ua.T @ ub) * (da @ db.T)).sum())
        na[blk] += sa * sa * float(((ua.T @ ua) * (da @ da.T)).sum())
        nb[blk] += sb * sb * float(((ub.T @ ub) * (db @ db.T)).sum())
    return ip, na, nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write results here")
    args = ap.parse_args()

    arms = {}
    for p in args.checkpoints:
        name = p.stem.split("tenth4s_")[-1] if "tenth4s_" in p.stem else p.stem
        arms[name] = load_modules(p)
        print(f"loaded {name}: {len(arms[name])} modules")

    out = {
        "checkpoints": {n: str(p) for n, p in zip(arms, args.checkpoints)},
        "pairs": {},
    }
    for x, y in combinations(arms, 2):
        ip, na, nb = pair_stats(arms[x], arms[y])
        g_ip, g_na, g_nb = sum(ip.values()), sum(na.values()), sum(nb.values())
        cos_g = g_ip / (g_na**0.5 * g_nb**0.5)
        per_block = {
            blk: ip[blk] / (na[blk] ** 0.5 * nb[blk] ** 0.5)
            for blk in sorted(
                ip, key=lambda s: (s == "other", int(s) if s.isdigit() else 0)
            )
        }
        out["pairs"][f"{x}~{y}"] = {
            "cos_global": round(cos_g, 4),
            "cos_per_block": {k: round(v, 4) for k, v in per_block.items()},
        }
        print(f"\n{x} ↔ {y}: global cos = {cos_g:.3f}")
        blocks = [k for k in per_block if k != "other"]
        prof = "  ".join(f"{k}:{per_block[k]:.2f}" for k in blocks)
        print(f"  per-block: {prof}")
        if "other" in per_block:
            print(f"  other: {per_block['other']:.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
