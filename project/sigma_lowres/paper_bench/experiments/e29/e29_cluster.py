"""E29 — native-block test (roadmap §1(c)).

Contiguity-constrained clustering (frozen instrument, see README) on the
committed across-σ pair-cos tables. Gates run first on synthetic +
committed-public tables; the unread e26 verdict tables are opened only
after every gate passes. Output: e29_read.json.
"""

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E26_JSON = HERE.parent / "e26" / "e26_grid_across_sigma.json"
E28_JSON = HERE.parent / "e28" / "e28_read768.json"

BINS = ["s0.3", "s0.4333", "s0.5667", "s0.7", "s0.8333"]
N = len(BINS)
TAU = 0.30  # frozen


def to_mat(pairs):
    m = {}
    for p in pairs:
        a, b = p["pair"].split("~")
        m[(a, b)] = p["cos"]
        m[(b, a)] = p["cos"]
    for i, j in itertools.combinations(range(N), 2):
        assert (BINS[i], BINS[j]) in m, f"missing pair {BINS[i]}~{BINS[j]}"
    return m


def contiguous_partitions(n, k):
    for cuts in itertools.combinations(range(1, n), k - 1):
        segs, prev = [], 0
        for c in list(cuts) + [n]:
            segs.append(tuple(range(prev, c)))
            prev = c
        yield tuple(segs)


def cost(m, segs):
    d, n = 0.0, 0
    for seg in segs:
        for i, j in itertools.combinations(seg, 2):
            d += 1.0 - m[(BINS[i], BINS[j])]
            n += 1
    return d / n if n else None


def classify(m):
    """Frozen selection rule: sequential elbow with tau on pooled-mean cost."""
    best = {}
    for k in (1, 2, 3):
        # itertools.combinations yields cut sets in lexicographic order;
        # min() keeps the first of tied costs => frozen tie-break.
        best[k] = min(
            ((cost(m, s), s) for s in contiguous_partitions(N, k)),
            key=lambda t: t[0],
        )
    k_star = 1
    if best[1][0] - best[2][0] >= TAU:
        k_star = 2
        if best[2][0] - best[3][0] >= TAU:
            k_star = 3
    segs = best[k_star][1]
    within = [
        m[(BINS[i], BINS[j])]
        for seg in segs
        for i, j in itertools.combinations(seg, 2)
    ]
    cross = [
        m[(BINS[i], BINS[j])]
        for sa, sb in itertools.combinations(segs, 2)
        for i in sa
        for j in sb
    ]
    margin = (min(within) - max(cross)) if within and cross else None
    return {
        "k_star": k_star,
        "partition": [[BINS[i] for i in seg] for seg in segs],
        "cost": {str(k): round(best[k][0], 6) for k in (1, 2, 3)},
        "gap12": round(best[1][0] - best[2][0], 6),
        "gap23": round(best[2][0] - best[3][0], 6),
        "separation_margin": round(margin, 6) if margin is not None else None,
    }


def abs_variant(m):
    return {k: abs(v) for k, v in m.items()}


# ---------------------------------------------------------------- gates

def synth_mat(fn):
    return {
        (BINS[i], BINS[j]): fn(i, j)
        for i in range(N)
        for j in range(N)
        if i != j
    }


def perturb_stable(base_fn, expect_k, expect_partition=None, eps=0.05):
    """Exhaustive deterministic ±eps on the 10 pairs; classification must hold."""
    pair_idx = list(itertools.combinations(range(N), 2))
    for signs in itertools.product((-1.0, 1.0), repeat=len(pair_idx)):
        delta = {p: s * eps for p, s in zip(pair_idx, signs)}
        m = synth_mat(
            lambda i, j: base_fn(i, j) + delta[(min(i, j), max(i, j))]
        )
        r = classify(m)
        if r["k_star"] != expect_k:
            return False, r
        if expect_partition is not None and r["partition"] != expect_partition:
            return False, r
    return True, None


def run_gates(e28):
    gates = {}

    # gate 1 — planted two-block, all 4 boundary positions, ±0.05 exhaustive
    ok_all = True
    for b in range(1, N):
        fn = lambda i, j, b=b: 0.6 if (i < b) == (j < b) else -0.3
        expect = [[BINS[i] for i in range(b)], [BINS[i] for i in range(b, N)]]
        ok, bad = perturb_stable(fn, 2, expect)
        ok_all &= ok
        if not ok:
            gates.setdefault("gate1_failures", []).append(
                {"boundary": b, "example": bad}
            )
    gates["gate1_planted_blocks"] = "PASS" if ok_all else "FAIL"

    # gate 2 — smooth exponential decay, k*=1, ±0.05 exhaustive
    ok_all = True
    for rho in (0.5, 0.7, 0.85, 0.95):
        ok, bad = perturb_stable(lambda i, j, r=rho: r ** abs(i - j), 1)
        ok_all &= ok
        if not ok:
            gates.setdefault("gate2_failures", []).append(
                {"rho_adj": rho, "example": bad}
            )
    gates["gate2_smooth_control"] = "PASS" if ok_all else "FAIL"

    # gate 3 — committed positive controls: frozen-frame R̂ and B̂ -> k=2, 3|2
    expect = [BINS[:3], BINS[3:]]
    r_frozen_R = classify(to_mat(e28["c_rows"]["R_across_sigma_frozen"]))
    r_frozen_B = classify(to_mat(e28["frozen_across_sigma_B"]["pairs"]))
    ok = all(
        r["k_star"] == 2 and r["partition"] == expect
        for r in (r_frozen_R, r_frozen_B)
    )
    gates["gate3_frozen_frame_positive"] = "PASS" if ok else "FAIL"
    gates["gate3_rows"] = {"frozen_R": r_frozen_R, "frozen_B": r_frozen_B}

    # gate 4 — committed smooth anchor: twin B̂ -> k=1
    r_twin_B = classify(to_mat(e28["twin_across_sigma_B"]["pairs"]))
    gates["gate4_twin_B_smooth"] = "PASS" if r_twin_B["k_star"] == 1 else "FAIL"
    gates["gate4_row"] = r_twin_B

    return gates, r_twin_B


def gate5_provenance(e26, e28):
    a = {p["pair"]: p["cos"] for p in e28["twin_across_sigma_B"]["pairs"]}
    b = {
        p["pair"]: p["cos"]
        for p in e26["adapters"]["sincos_twin"]["across_sigma_B"]["pairs"]
    }
    if set(a) != set(b):
        return "FAIL", None
    dev = max(abs(a[k] - b[k]) for k in a)
    return ("PASS" if dev <= 1e-3 else "FAIL"), dev


def main():
    e28 = json.loads(E28_JSON.read_text())

    gates, _ = run_gates(e28)

    # gate 5 needs the e26 file, but touches ONLY the twin-B table
    # (public); nothing else is read until all gates pass.
    e26 = json.loads(E26_JSON.read_text())
    g5, dev = gate5_provenance(e26, e28)
    gates["gate5_provenance_tie"] = g5
    gates["gate5_max_dev"] = dev

    gate_keys = [
        "gate1_planted_blocks",
        "gate2_smooth_control",
        "gate3_frozen_frame_positive",
        "gate4_twin_B_smooth",
        "gate5_provenance_tie",
    ]
    all_pass = all(gates[k] == "PASS" for k in gate_keys)
    out = {"generated_by": "e29_cluster.py", "tau": TAU, "gates": gates}

    if not all_pass:
        out["verdict"] = "GATES-FAILED — no verdict table read"
        (HERE / "e29_read.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        sys.exit(1)

    # ------------------------------------------------ verdict + rows
    def read_table(adapter, leg):
        m = to_mat(e26["adapters"][adapter][f"across_sigma_{leg}"]["pairs"])
        return {
            "signed": classify(m),
            "abs_variant": classify(abs_variant(m)),
            "pairs": {
                f"{BINS[i]}~{BINS[j]}": round(m[(BINS[i], BINS[j])], 5)
                for i, j in itertools.combinations(range(N), 2)
            },
        }

    primary = read_table("sincos_twin", "R")
    k = primary["signed"]["k_star"]
    verdict = "NATIVE-SMOOTH" if k == 1 else "NATIVE-BLOCKS"

    replication = {a: read_table(a, "R") for a in ("flat", "dirty")}
    rep_consistent = {
        a: replication[a]["signed"]["k_star"] == k for a in replication
    }

    descriptive = {
        f"{a}_{leg}": read_table(a, leg)
        for a in ("flat", "dirty", "sincos_twin")
        for leg in ("B", "C")
    }

    out.update(
        {
            "verdict_29_1": verdict,
            "primary_twin_R": primary,
            "replication_29_2": replication,
            "replication_consistent_with_29_1": rep_consistent,
            "descriptive_B_C": descriptive,
            "note": (
                "twin_B is a calibration anchor (README) — descriptive only, "
                "no verdict weight"
            ),
        }
    )
    (HERE / "e29_read.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("verdict_29_1",)}, indent=1))
    print("primary:", json.dumps(primary["signed"], indent=1))
    print("replication k*:", {a: replication[a]["signed"]["k_star"] for a in replication},
          "consistent:", rep_consistent)


if __name__ == "__main__":
    main()
