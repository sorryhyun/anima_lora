"""Tallies per group (all, per bin, the unseen strings) and the paired
tests of one arm against another."""

from __future__ import annotations

from eval.ruler import BINS
from eval.ruler.score import BOOL, REAL


def sign_p(g: int, lo: int) -> float:
    from math import comb

    n = g + lo
    if not n:
        return 1.0
    return float(
        f"{min(1.0, 2 * sum(comb(n, k) for k in range(min(g, lo) + 1)) / 2**n):.2g}"
    )


def groups(recs_a: dict) -> dict:
    out = {"all": list(recs_a)}
    for b in BINS:
        out[b] = [i for i, r in recs_a.items() if r["bin"] == b]
    out["unseen"] = [i for i, r in recs_a.items() if r["unseen"]]
    out["unseen_short"] = [i for i in out["unseen"] if recs_a[i]["bin"] == "short"]
    return out


def tally(recs: dict) -> dict:
    out = {}
    for a, ra in recs.items():
        out[a] = {}
        for g, ks in groups(ra).items():
            c = {"n": len(ks)} | {k: sum(ra[i][k] for i in ks) for k in BOOL}
            for k in REAL:
                xs = [ra[i][k] for i in ks if ra[i][k] is not None]
                c[k] = round(sum(xs) / len(xs), 4) if xs else None
            out[a][g] = c
    return out


def paired(ra: dict, rb: dict) -> dict:
    """Per group: McNemar on the booleans (gained / lost / p), the sign test
    and mean difference on the reals (a − b; ``cer`` lower is better)."""
    out = {}
    for g, ks in groups(ra).items():
        c = {"n": len(ks)}
        for k in BOOL:
            gn = sum(ra[i][k] and not rb[i][k] for i in ks)
            ls = sum(rb[i][k] and not ra[i][k] for i in ks)
            c[k] = [gn, ls, sign_p(gn, ls)]
        for k in REAL:
            dif = [
                ra[i][k] - rb[i][k]
                for i in ks
                if ra[i][k] is not None and rb[i][k] is not None
            ]
            up, dn = sum(x > 1e-9 for x in dif), sum(x < -1e-9 for x in dif)
            c[k] = [round(sum(dif) / max(1, len(dif)), 4), up, dn, sign_p(up, dn)]
        out[g] = c
    return out
