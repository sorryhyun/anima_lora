#!/usr/bin/env python3
"""How much of a LoRA's ΔW do two runs share?

ΔW = (alpha/r)·up·down per module. Prints |ΔW| (Frobenius, over all modules) of
each checkpoint and the cosine between every pair, computed with the Gram trick
(no full ΔW is formed), so it runs on the CPU in seconds. With ``--groups`` the
first pair is also split by module kind and by block depth.

For two runs of one objective from one init, the cosine reads as the share of
ΔW energy the runs have in common; the rest is run-specific (the batch-1 walk,
``scripts/distill_turbo/plan.md``).

    .venv/bin/python bench/turbo/dw_overlap.py a.safetensors b.safetensors [...] [--groups]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from safetensors import safe_open

from library.env import resolve_under_home


def load(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    f = safe_open(str(path), "pt")
    keys = set(f.keys())
    mods = {}
    for k in keys:
        if not k.endswith(".lora_down.weight"):
            continue
        m = k[: -len(".lora_down.weight")]
        down = f.get_tensor(k).double().flatten(1)
        up = f.get_tensor(m + ".lora_up.weight").double().flatten(1)
        r = down.shape[0]
        alpha = f.get_tensor(m + ".alpha").item() if m + ".alpha" in keys else r
        mods[m] = (up * (alpha / r), down)
    return mods


def inner(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]) -> float:
    return torch.trace((a[0].T @ b[0]) @ (b[1] @ a[1].T)).item()


def total(x: dict, y: dict) -> float:
    return sum(inner(x[m], y[m]) for m in x.keys() & y.keys())


def kind(m: str) -> str:
    for k in ("adaln", "self_attn", "cross_attn"):
        if k in m:
            return k
    return "mlp"


def depth(m: str) -> str:
    b = int(re.search(r"blocks_(\d+)_", m).group(1)) // 7 * 7
    return f"blocks {b:02d}-{b + 6:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--groups", action="store_true", help="split the first pair by module kind and depth")
    args = ap.parse_args()

    paths = [resolve_under_home(p) for p in args.ckpts]
    names = [p.stem for p in paths]
    nets = [load(p) for p in paths]
    norms = [total(n, n) ** 0.5 for n in nets]
    for name, n in zip(names, norms):
        print(f"{n:8.3f}  {name}")
    print("cos")
    for i, x in enumerate(nets):
        row = [1.0 if i == j else total(x, y) / (norms[i] * norms[j]) for j, y in enumerate(nets)]
        print(" ".join(f"{c:+.3f}" for c in row), names[i])

    if args.groups and len(nets) >= 2:
        x, y = nets[0], nets[1]
        for keyf in (kind, depth):
            acc: dict[str, list[float]] = {}
            for m in x.keys() & y.keys():
                s = acc.setdefault(keyf(m), [0.0, 0.0, 0.0])
                s[0] += inner(x[m], x[m])
                s[1] += inner(y[m], y[m])
                s[2] += inner(x[m], y[m])
            tot = sum(v[0] for v in acc.values())
            for g, (xx, yy, xy) in sorted(acc.items()):
                print(f"  {g:14s} energy {xx / tot:5.1%}  cos {xy / (xx * yy) ** 0.5:+.3f}")


if __name__ == "__main__":
    main()
