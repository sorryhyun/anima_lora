"""E7 cell readout: per-cell means + factorial contrasts from the two probe runs.

Bin-mean first read of the pre-registered E7 quantities
(required_experiments.md): per-cell means over the in-window sigma bins,
membership contrast (S1 vs S2), probe-style main effect, and the paired
adapter x probe-style interaction on the shared cross-cluster stems.
Source of the numbers quoted in the paper's Appendix E7 section.

Usage:
    python project/sigma_lowres/paper_bench/e7_cells.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "runs"
# Artifact names: "flat" = the paper's high-redundancy cluster, "dirty" = low-redundancy.
RUNS = {"flat": ROOT / "20260729-1349", "dirty": ROOT / "20260729-1702"}
INWIN = slice(
    0, 4
)  # in-window sigma bins (0.5625..0.9375); index 4 = sigma=1.0 endpoint


def load(run):
    return [json.loads(line) for line in (run / "per_image.jsonl").open()]


def cellmean(rows, key, sl=INWIN):
    vals = np.array([np.nanmean(np.array(r[key], dtype=float)[sl]) for r in rows])
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), len(vals)


data = {a: load(p) for a, p in RUNS.items()}

print("=== per-cell means, in-window sigma bins (0.5625-0.9375), mean +/- sem (n) ===")
for adapter, rows in data.items():
    cells = sorted({r["cell"] for r in rows})
    print(f"\n-- adapter: {adapter} --")
    hdr = f"{'cell':5s} {'style':6s} {'n':>3s}"
    for key in (
        "cos_floor",
        "debiased_gap_896",
        "debiased_gap_768",
        "debiased_gap_reenc",
    ):
        hdr += f"  {key:>20s}"
    print(hdr)
    for c in cells:
        sub = [r for r in rows if r["cell"] == c]
        styles = {r["style"] for r in sub}
        line = f"{c:5s} {'/'.join(sorted(styles)):6s} {len(sub):3d}"
        for key in (
            "cos_floor",
            "debiased_gap_896",
            "debiased_gap_768",
            "debiased_gap_reenc",
        ):
            m, s, n = cellmean(sub, key)
            line += f"  {m:+.4f} +/- {s:.4f}  "
        print(line)

print("\n=== membership effect: S1 - S2 (style-matched, within adapter), per route ===")
for adapter, rows in data.items():
    s1 = {r["stem"]: r for r in rows if r["cell"] == "S1"}
    s2 = {r["stem"]: r for r in rows if r["cell"] == "S2"}
    for key in ("debiased_gap_896", "debiased_gap_768", "cos_floor"):
        a = np.array(
            [np.nanmean(np.array(r[key], dtype=float)[INWIN]) for r in s1.values()]
        )
        b = np.array(
            [np.nanmean(np.array(r[key], dtype=float)[INWIN]) for r in s2.values()]
        )
        d = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"  {adapter:5s} {key:20s} S1-S2 = {d:+.4f} +/- {se:.4f}")

print(
    "\n=== probe-style main effect & interaction (paired on shared cross-cluster stems) ==="
)
# For each adapter run: own-style probe = S3s (never-seen artists, same style),
# cross-style probe = S3x. Shared stems: flat-run S3x stems should equal dirty-run S3s stems etc.
for key in ("debiased_gap_896", "debiased_gap_768"):
    stats = {}
    for adapter, rows in data.items():
        own = np.array(
            [
                np.nanmean(np.array(r[key], dtype=float)[INWIN])
                for r in rows
                if r["cell"] == "S3s"
            ]
        )
        cross = np.array(
            [
                np.nanmean(np.array(r[key], dtype=float)[INWIN])
                for r in rows
                if r["cell"] == "S3x"
            ]
        )
        stats[adapter] = (own, cross)
    # probe-style main effect: dirty-style probes vs flat-style probes, averaged over adapters
    dirty_probes = np.concatenate([stats["dirty"][0], stats["flat"][1]])
    flat_probes = np.concatenate([stats["flat"][0], stats["dirty"][1]])
    me = dirty_probes.mean() - flat_probes.mean()
    me_se = np.sqrt(
        dirty_probes.var(ddof=1) / len(dirty_probes)
        + flat_probes.var(ddof=1) / len(flat_probes)
    )
    # interaction: (cross - own)_flat_adapter - (cross - own)_dirty_adapter... define as
    # own-style advantage difference: [own - cross]_dirty - [own - cross]_flat is symmetric variant.
    ia = (stats["dirty"][0].mean() - stats["dirty"][1].mean()) - (
        stats["flat"][1].mean() - stats["flat"][0].mean()
    )
    print(
        f"  {key:20s} style main effect (dirty-flat probes) = {me:+.4f} +/- {me_se:.4f}"
    )

    # paired interaction on shared stems
    for a, b, tag in (
        (("dirty", "S3s"), ("flat", "S3x"), "dirty-style stems"),
        (("flat", "S3s"), ("dirty", "S3x"), "flat-style stems"),
    ):
        ra = {
            r["stem"]: np.nanmean(np.array(r[key], dtype=float)[INWIN])
            for r in data[a[0]]
            if r["cell"] == a[1]
        }
        rb = {
            r["stem"]: np.nanmean(np.array(r[key], dtype=float)[INWIN])
            for r in data[b[0]]
            if r["cell"] == b[1]
        }
        shared = sorted(set(ra) & set(rb))
        if shared:
            d = np.array(
                [ra[s] - rb[s] for s in shared]
            )  # own-adapter minus cross-adapter, same stem
            print(
                f"    {key} paired own-vs-cross adapter on {tag} (n={len(shared)}): "
                f"{d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.4f}"
            )
        else:
            print(f"    {key} {tag}: NO shared stems between {a} and {b}")

print("\n=== VERDICT: adapter x probe-style interaction, RAW gaps, paired per stem ===")
for key in ("gap_896", "gap_768"):
    effs = {}
    for tag, (own, cross) in {
        "dirty-style": (("dirty", "S3s"), ("flat", "S3x")),
        "flat-style": (("flat", "S3s"), ("dirty", "S3x")),
    }.items():
        ra = {
            r["stem"]: np.nanmean(np.array(r[key], dtype=float)[INWIN])
            for r in data[own[0]]
            if r["cell"] == own[1]
        }
        rb = {
            r["stem"]: np.nanmean(np.array(r[key], dtype=float)[INWIN])
            for r in data[cross[0]]
            if r["cell"] == cross[1]
        }
        shared = sorted(set(ra) & set(rb))
        # adapter effect per stem group = value under dirty adapter minus under flat adapter
        if tag == "dirty-style":
            d = np.array([ra[s] - rb[s] for s in shared])
        else:
            d = np.array([rb[s] - ra[s] for s in shared])
        effs[tag] = d
        print(
            f"  {key} adapter effect on {tag:11s} stems (n={len(d)}): "
            f"{d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.4f}"
        )
    ia = effs["dirty-style"].mean() - effs["flat-style"].mean()
    se = np.sqrt(
        effs["dirty-style"].var(ddof=1) / len(effs["dirty-style"])
        + effs["flat-style"].var(ddof=1) / len(effs["flat-style"])
    )
    print(f"  {key} INTERACTION = {ia:+.4f} +/- {se:.4f}")

print("\n=== endpoint (sigma=1.0) debiased gaps, sanity (should be ~0 everywhere) ===")
for adapter, rows in data.items():
    for key in ("debiased_gap_896", "debiased_gap_768"):
        v = np.array([r[key][4] for r in rows])
        print(
            f"  {adapter:5s} {key:20s} {v.mean():+.4f} +/- {v.std(ddof=1) / np.sqrt(len(v)):.4f}"
        )
