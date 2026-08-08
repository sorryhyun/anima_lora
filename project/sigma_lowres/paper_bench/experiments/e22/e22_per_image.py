#!/usr/bin/env python
"""sigma_lowres E22.2 — per-image ledger digest + figures (CPU).

Pre-registered in README.md (committed before the 22.1 instrument existed).
Reads a ``run_sigma_probe.py --per_image_ledger`` run's
``per_image_ledger.jsonl`` and produces the 22.3 verdict quantities:

 * per-σ distributions of gated ρ_i (global — the verdict quantity; per
   band — secondary; per cell — exploratory, no verdict weight)
 * reliability accounting (rel ≥ 0.5 gate counts; the ≥ 8-images floor)
 * stratum overlays (near-native vs heavy realized resize factor) —
   OBSERVATIONAL only (strata differ in content by construction)
 * pooled same-run cross-check against ``<run>/ledger.json`` when present
   (``vector_ledger.py --data_ref reenc`` output) — internal consistency,
   NOT comparable to E14/E21 (stratified corpus, selection differs)

Verdict wording (standing caveat): "per-image at this operating point,
probe corpus" — never "per-user-sample at training time".

Outputs (this dir): ``e22_per_image.json`` (the record), ``e22_rho_i.png``.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e22/e22_per_image.py \
        --run project/sigma_lowres/paper_bench/runs/<run>
    ... --figs_only   # redraw from the committed digest (22.3 re-reads)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]

VERDICT_SIGMA = (0.4333, 0.5667, 0.7)
ROUTE = "768"
REL_GATE = 0.5
FLOOR_N = 8  # reliability floor: fewer gated images at a σ ⇒ that σ is void
POOLED_RHO_REF = -0.91  # E14/E19 pooled deep constant (reference line only)
BANDS = ("adaln", "cross_attn", "self_attn", "mlp")


def gated(row: dict) -> bool:
    rb, rc = (
        row.get("rel_cos_B", row.get("relB")),
        row.get("rel_cos_C", row.get("relC")),
    )
    return (
        rb is not None
        and rc is not None
        and not (math.isnan(rb) or math.isnan(rc))
        and rb >= REL_GATE
        and rc >= REL_GATE
    )


def sigma_class(vals: list[float]) -> str:
    """The 22.3 per-σ reading on gated global ρ_i."""
    if len(vals) < FLOOR_N:
        return "VOID"
    med = float(np.median(vals))
    frac_deep = float(np.mean([v <= -0.7 for v in vals]))
    if med <= -0.7 and frac_deep >= 0.75:
        return "HOLDS"
    if med >= -0.5 or frac_deep < 0.5:
        return "POOLED_ONLY"
    return "MIXED"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run", default=None, help="22.1 run dir (holds the jsonl)")
    ap.add_argument("--figs_only", action="store_true")
    ap.add_argument("--out", default=str(HERE / "e22_per_image.json"))
    args = ap.parse_args()
    out_json = Path(args.out)

    if args.figs_only:
        digest = json.loads(out_json.read_text())
        draw(digest)
        report(digest)
        return
    if not args.run:
        raise SystemExit("--run is required unless --figs_only")

    run = Path(args.run).resolve()
    rows = [
        json.loads(ln)
        for ln in (run / "per_image_ledger.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    centers = rows[0]["sigma_centers"]
    bins = [centers.index(s) for s in VERDICT_SIGMA if s in centers]
    assert len(bins) == len(VERDICT_SIGMA), f"run grid {centers} != {VERDICT_SIGMA}"

    digest: dict = {
        "meta": {
            "run": str(run),
            "route": ROUTE,
            "rel_gate": REL_GATE,
            "floor_n": FLOOR_N,
            "verdict_sigma": list(VERDICT_SIGMA),
            "n_images": len(rows),
            "note": (
                "per-image at this operating point (the E14 adapter), probe "
                "corpus; stratified selection makes pooled values "
                "non-comparable to E14/E21. Resize-factor reads are "
                "OBSERVATIONAL (strata differ in content by construction)."
            ),
        },
        "per_sigma": [],
        "bands": {},
        "cells": [],
        "strata": [],
        "verdict": {},
    }

    # ---------- global ρ_i: the verdict quantity ----------
    classes = []
    for s, bi in zip(VERDICT_SIGMA, bins):
        entries = []
        for r in rows:
            g = r["ledger"][ROUTE][bi]["global"]
            entries.append(
                {
                    "artist": r["artist"],
                    "stem": r["stem"],
                    "stratum": r.get("stratum"),
                    "resize_factor": r.get("resize_factor"),
                    "gated": gated(g),
                    **{
                        k: g[k]
                        for k in (
                            "rho",
                            "sig_rho",
                            "S",
                            "F",
                            "I",
                            "rel_cos_B",
                            "rel_cos_C",
                            "amp_ratio",
                            "h_B_plus_C",
                            "h_B",
                            "h_C",
                            "G",
                        )
                    },
                }
            )
        vals = [e["rho"] for e in entries if e["gated"] and not math.isnan(e["rho"])]
        cls = sigma_class(vals)
        classes.append(cls)
        digest["per_sigma"].append(
            {
                "sigma": s,
                "n_gated": len(vals),
                "n_images": len(entries),
                "median_rho_i": round(float(np.median(vals)), 4) if vals else None,
                "q25": round(float(np.percentile(vals, 25)), 4) if vals else None,
                "q75": round(float(np.percentile(vals, 75)), 4) if vals else None,
                "min": round(min(vals), 4) if vals else None,
                "max": round(max(vals), 4) if vals else None,
                "frac_le_m07": round(float(np.mean([v <= -0.7 for v in vals])), 4)
                if vals
                else None,
                "frac_le_m05": round(float(np.mean([v <= -0.5 for v in vals])), 4)
                if vals
                else None,
                "median_sig_rho_gated": round(
                    float(np.median([e["sig_rho"] for e in entries if e["gated"]])),
                    4,
                )
                if vals
                else None,
                "class": cls,
                "images": entries,
            }
        )

    # ---------- run verdict (22.3 table) ----------
    live = [c for c in classes if c != "VOID"]
    if not live:
        run_verdict = "INSTRUMENT-LIMITED"
    elif any(c == "VOID" for c in classes):
        # a void σ can't confirm "every verdict σ" — degrade to MIXED unless
        # the surviving σ all say POOLED_ONLY (one pre-registered stroke
        # still kills per-sample correction on the σ it measured)
        run_verdict = (
            "POOLED-ONLY" if all(c == "POOLED_ONLY" for c in live) else "MIXED"
        )
    elif all(c == "HOLDS" for c in classes):
        run_verdict = "PER-SAMPLE HOLDS"
    elif all(c == "POOLED_ONLY" for c in classes):
        run_verdict = "POOLED-ONLY"
    else:
        run_verdict = "MIXED"
    digest["verdict"] = {"per_sigma": classes, "run": run_verdict}

    # ---------- bands (secondary) ----------
    for band in BANDS:
        digest["bands"][band] = []
        for s, bi in zip(VERDICT_SIGMA, bins):
            vals = []
            n_gated = 0
            for r in rows:
                b = r["ledger"][ROUTE][bi]["bands"][band]
                if gated(b) and not math.isnan(b["rho"]):
                    n_gated += 1
                    vals.append(b["rho"])
            digest["bands"][band].append(
                {
                    "sigma": s,
                    "n_gated": n_gated,
                    "median_rho_i": round(float(np.median(vals)), 4) if vals else None,
                    "q25": round(float(np.percentile(vals, 25)), 4) if vals else None,
                    "q75": round(float(np.percentile(vals, 75)), 4) if vals else None,
                }
            )

    # ---------- cells (exploratory only, no verdict weight) ----------
    for s, bi in zip(VERDICT_SIGMA, bins):
        per_image_medians = []
        gate_counts = []
        for r in rows:
            cells = r["ledger"][ROUTE][bi]["cells"]
            vals = [
                c["rho"]
                for c in cells.values()
                if gated(c) and not math.isnan(c["rho"])
            ]
            gate_counts.append(len(vals))
            if vals:
                per_image_medians.append(float(np.median(vals)))
        digest["cells"].append(
            {
                "sigma": s,
                "n_cells_per_image": len(rows[0]["ledger"][ROUTE][bi]["cells"]),
                "gated_cells_per_image": [int(v) for v in gate_counts],
                "median_of_per_image_cell_medians": round(
                    float(np.median(per_image_medians)), 4
                )
                if per_image_medians
                else None,
            }
        )

    # ---------- stratum overlays (OBSERVATIONAL) ----------
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None
    for s, bi in zip(VERDICT_SIGMA, bins):
        entry = {"sigma": s, "quantities": {}}
        for qty, getter in (
            ("rho_i", lambda g: g["rho"]),
            ("b_amp", lambda g: math.sqrt(2.0 * g["S"]) if g["S"] > 0 else None),
            ("h_B_plus_C", lambda g: g["h_B_plus_C"]),
        ):
            groups = {"near_native": [], "heavy": []}
            for r in rows:
                g = r["ledger"][ROUTE][bi]["global"]
                if not gated(g):
                    continue
                v = getter(g)
                if v is not None and not math.isnan(v):
                    groups[r["stratum"]].append(v)
            nn, hv = groups["near_native"], groups["heavy"]
            q = {
                "near_native_median": round(float(np.median(nn)), 4) if nn else None,
                "heavy_median": round(float(np.median(hv)), 4) if hv else None,
                "n": [len(nn), len(hv)],
            }
            if mannwhitneyu is not None and nn and hv:
                q["mannwhitney_p"] = round(
                    float(mannwhitneyu(nn, hv, alternative="two-sided").pvalue), 4
                )
            entry["quantities"][qty] = q
        digest["strata"].append(entry)

    # ---------- pooled same-run cross-check ----------
    pooled_path = run / "ledger.json"
    if pooled_path.exists():
        pooled = json.loads(pooled_path.read_text())
        digest["pooled_crosscheck"] = {
            "source": "vector_ledger.py (same run, reenc ref)",
            "note": "internal consistency only — stratified corpus, NOT "
            "comparable to E14/E21 pooled values",
            "rho": [
                pooled["bc_ledger"][ROUTE][centers.index(s)]["rho"]
                for s in VERDICT_SIGMA
            ],
        }

    out_json.write_text(json.dumps(digest, indent=1))
    print(f"digest -> {out_json}")
    report(digest)
    draw(digest)


def report(digest: dict) -> None:
    print("\n== E22.3 verdict (gated images, global rho_i) ==")
    for row in digest["per_sigma"]:
        print(
            f"sigma={row['sigma']:<7} gated {row['n_gated']}/{row['n_images']}  "
            f"median={row['median_rho_i']}  IQR=[{row['q25']},{row['q75']}]  "
            f"frac<=-0.7: {row['frac_le_m07']}  -> {row['class']}"
        )
    print("run verdict:", digest["verdict"]["run"])
    if "pooled_crosscheck" in digest:
        print("pooled same-run rho:", digest["pooled_crosscheck"]["rho"])
    print("\n== bands (secondary, gated) ==")
    for band, rows_ in digest["bands"].items():
        cells = "  ".join(
            f"{r['sigma']}: {r['median_rho_i']} (n={r['n_gated']})" for r in rows_
        )
        print(f"  {band:<11} {cells}")
    print("\n== strata (OBSERVATIONAL) ==")
    for e in digest["strata"]:
        for qty, q in e["quantities"].items():
            p = f" p={q.get('mannwhitney_p')}" if "mannwhitney_p" in q else ""
            print(
                f"  sigma={e['sigma']:<7} {qty:<11} near={q['near_native_median']} "
                f"heavy={q['heavy_median']} n={q['n']}{p}"
            )


def draw(digest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = [r["sigma"] for r in digest["per_sigma"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # panel 1: gated rho_i distribution per sigma, colored by stratum
    ax = axes[0]
    data = []
    for r in digest["per_sigma"]:
        data.append(
            [e["rho"] for e in r["images"] if e["gated"] and not math.isnan(e["rho"])]
        )
    vp = ax.violinplot(data, showmedians=True, widths=0.7)
    for body in vp["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_alpha(0.6)
    vp["cmedians"].set_color("0.1")
    rng = np.random.default_rng(0)
    for xi, r in enumerate(digest["per_sigma"]):
        for e in r["images"]:
            if not e["gated"] or math.isnan(e["rho"]):
                continue
            col = "#1f77b4" if e["stratum"] == "near_native" else "#d62728"
            ax.plot(
                xi + 1 + rng.uniform(-0.08, 0.08),
                e["rho"],
                "o",
                color=col,
                ms=4,
                alpha=0.85,
            )
    if "pooled_crosscheck" in digest:
        ax.plot(
            range(1, len(sigmas) + 1),
            digest["pooled_crosscheck"]["rho"],
            "D",
            color="0.2",
            ms=6,
            label="pooled (same run)",
        )
    ax.axhline(POOLED_RHO_REF, color="0.4", ls="--", lw=1, label="E14 pooled -0.91")
    ax.axhline(-0.7, color="0.7", ls=":", lw=1, label="HOLDS floor -0.7")
    ax.axhline(-0.5, color="0.85", ls=":", lw=1, label="POOLED-ONLY line -0.5")
    ax.set_xticks(range(1, len(sigmas) + 1))
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.set_xlabel("sigma bin")
    ax.set_ylabel("rho_i (gated images)")
    ax.set_title(
        "global per-image rho_i, route 1024-768\n"
        "blue = near-native, red = heavy resize stratum"
    )
    ax.legend(fontsize=7, loc="upper right")

    # panel 2: band medians
    ax = axes[1]
    for band, rows_ in digest["bands"].items():
        med = [r["median_rho_i"] for r in rows_]
        ax.plot(sigmas, med, "o-", label=band)
    glob = [r["median_rho_i"] for r in digest["per_sigma"]]
    ax.plot(sigmas, glob, "k^--", label="global")
    ax.axhline(-0.7, color="0.7", ls=":", lw=1)
    ax.set_xlabel("sigma bin")
    ax.set_ylabel("median gated rho_i")
    ax.set_title("per-band medians (secondary)")
    ax.legend(fontsize=8)

    # panel 3: rho_i vs realized resize factor (OBSERVATIONAL)
    ax = axes[2]
    for r in digest["per_sigma"]:
        xs, ys, cols = [], [], []
        for e in r["images"]:
            if not e["gated"] or math.isnan(e["rho"]):
                continue
            xs.append(e["resize_factor"])
            ys.append(e["rho"])
            cols.append("#1f77b4" if e["stratum"] == "near_native" else "#d62728")
        ax.scatter(
            xs,
            ys,
            c=cols,
            s=22,
            alpha=0.5 + 0.5 * (r["sigma"] == 0.7),
            label=f"sigma={r['sigma']}" if r["sigma"] == 0.7 else None,
        )
    ax.set_xscale("log")
    ax.axhline(-0.7, color="0.7", ls=":", lw=1)
    ax.set_xlabel("realized resize factor (source area / resized area, log)")
    ax.set_ylabel("rho_i (gated)")
    ax.set_title(
        "OBSERVATIONAL: rho_i vs resize factor\n(content-confounded — no causal read)"
    )

    fig.suptitle(
        "E22 per-image g-ledger — per-image at this operating point, probe corpus "
        f"(n={digest['meta']['n_images']}, rel gate {REL_GATE}, floor {FLOOR_N})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "e22_rho_i.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
