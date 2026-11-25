#!/usr/bin/env python
"""Row walk — how far a G1 run's rows moved, and how much of it is one direction (CPU).

Reads a run's ``log.jsonl`` and ``rows_s<steps>.pt`` under ``output/cjk_rl/g1/<run>/``
and prints, in row norms:

- the length of a walk of N steps of ``eta`` whose successive directions are tied only
  by the momentum (cosine ``beta^k`` at lag k), beside each row's drift at every save;
- per row the segments between saves, their lengths and the cosines between them;
- the cosine between two rows' displacements, and the norm of the rows' mean
  displacement beside a row's own;
- per row and pass the mean gradient's norm and the training-side ``exact`` count.

    .venv/bin/python project/cjk_rl/probes/row_walk.py --run g1long_1125_1335
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
G1_OUT = REPO / "output" / "cjk_rl" / "g1"


def walk(n: int, eta: float, beta: float) -> float:
    """``eta · sqrt(Σ_{i,j<n} beta^|i−j|)``."""
    s = n * (1 + beta) / (1 - beta) - 2 * beta * (1 - beta**n) / (1 - beta) ** 2
    return eta * math.sqrt(s)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    d = G1_OUT / a.run
    log = [
        json.loads(x) for x in (d / "log.jsonl").read_text("utf-8").splitlines() if x
    ]
    saves = {
        int(f.stem.split("_s")[1]): torch.load(
            f, map_location="cpu", weights_only=False
        )
        for f in sorted(d.glob("rows_s*.pt"))
    }
    ns = sorted(saves)
    last = saves[ns[-1]]
    cfg, init, sc = last["cfg"], last["init"].float(), last["row_scale"]
    eta, beta = cfg["eta"], cfg["beta"]
    per_pass = len(cfg["train_scenes"]) * 4 // cfg["accum"]
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))  # noqa: E731

    print(f"{a.run}: eta {eta}, beta {beta}, saves at {ns} steps")
    print("a momentum walk on noise:", {n: round(walk(n, eta, beta), 3) for n in ns})
    by = defaultdict(list)
    for r in log:
        by[r["key"]].append(r)
    seg, tot = {}, {}
    print("row | drift at each save | segment lengths | cosines of segment pairs")
    for k, i in last["rows"].items():
        at = [init[i]] + [saves[n]["raw"][i].float() for n in ns]
        seg[k] = [(y - x) / sc for x, y in zip(at, at[1:])]
        tot[k] = (at[-1] - at[0]) / sc
        pairs = itertools.combinations(range(len(seg[k])), 2)
        print(
            k,
            " ".join(f"{float((x - at[0]).norm()) / sc:.3f}" for x in at[1:]),
            "|",
            " ".join(f"{float(s.norm()):.3f}" for s in seg[k]),
            "|",
            " ".join(
                f"{p + 1}{q + 1} {cos(seg[k][p], seg[k][q]):+.3f}" for p, q in pairs
            ),
        )
    cc = [cos(tot[x], tot[y]) for x, y in itertools.combinations(tot, 2)]
    mean = torch.stack(list(tot.values())).mean(0)
    print(
        f"two rows' displacements to the last save: cosine mean {sum(cc) / len(cc):.3f} "
        f"({min(cc):+.3f} … {max(cc):+.3f}); the mean displacement {float(mean.norm()):.3f}, "
        f"a row's {sum(float(v.norm()) for v in tot.values()) / len(tot):.3f}"
    )
    print("row | mean gradient norm by pass | exact of the step's renders by pass")
    for k, rs in by.items():
        ps = [rs[p : p + per_pass] for p in range(0, len(rs), per_pass)]
        print(
            k,
            " ".join(f"{sum(x['gbar'] for x in p) / len(p):.3f}" for p in ps),
            "|",
            " ".join(f"{sum(x['exact'] for x in p) / len(p):4.1f}" for p in ps),
        )


if __name__ == "__main__":
    main()
