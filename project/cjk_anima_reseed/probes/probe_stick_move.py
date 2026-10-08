#!/usr/bin/env python
"""probe_stick_move — the trained moves in stick / ball terms (2026-10-08)

Per family, for f0 − start, pres − f0 and pres − start (effective units,
start = `seed_fixed_1005_stick080`): the share of the move's energy in its
shared vector (Δstick: every row by the same vector, the ball translated),
|Δstick| / |stick|, cos(Δstick, stick), the angle the stick turns, its
length, and the per-row residual's rms against the ball's. Then the
families' Δstick against each other and pres's against f0's. CPU, seconds;
reads the rows only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reseed import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import probe_jl as J  # noqa: E402
from probe_geom import _families  # noqa: E402


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    rows = {
        "start": J._offsets(J.STICK080),
        "f0": J._offsets(J.F0),
        "pres": J._offsets(J.PRES),
    }
    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    fam = {f: [ids[k] for k in ks] for f, ks in _families(ids).items()}
    dstick = {}
    for f in ("kana", "kanji"):
        ks = [e for e in fam[f] if np.linalg.norm(rows["start"][e]) > 0]
        R = {n: np.stack([r[e] for e in ks]) for n, r in rows.items()}
        for a, b in (("f0", "start"), ("pres", "f0"), ("pres", "start")):
            X = R[a] - R[b]
            v = X.mean(0)
            dstick[f, a, b] = v
            s0, s1 = R[b].mean(0), R[a].mean(0)
            res = X - v
            ball = R[b] - s0
            ang = np.degrees(np.arccos(np.clip(_cos(s1, s0), -1, 1)))
            print(
                f"{f:5s} {a} − {b:5s} rows {len(X):4d}  "
                f"shared {len(X) * (v**2).sum() / (X**2).sum():.2f}  "
                f"|Δstick|/|stick| {np.linalg.norm(v) / np.linalg.norm(s0):.3f}  "
                f"cos(Δstick, stick) {_cos(v, s0):+.3f}  turn {ang:.2f}°  "
                f"|stick| {np.linalg.norm(s0):.1f}→{np.linalg.norm(s1):.1f}  "
                f"resid/ball {np.sqrt((res**2).sum(1).mean() / (ball**2).sum(1).mean()):.3f}"
            )
    for a, b in (("f0", "start"), ("pres", "f0")):
        print(
            f"cos(Δstick {a} − {b}: kana, kanji) "
            f"{_cos(dstick['kana', a, b], dstick['kanji', a, b]):+.3f}"
        )
    for f in ("kana", "kanji"):
        print(
            f"{f} cos(Δstick pres − f0, Δstick f0 − start) "
            f"{_cos(dstick[f, 'pres', 'f0'], dstick[f, 'f0', 'start']):+.3f}"
        )


if __name__ == "__main__":
    main()
