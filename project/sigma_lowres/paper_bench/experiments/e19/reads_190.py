#!/usr/bin/env python
"""E19.0 — re-reads of the committed E14 ledgers (zero GPU).

Three deliverables from ``runs/20260801-2304-e14-ledger-probematched``:

1. ρ(σ) × route table {896, 768, 512} with the completeness read
   (cancellation fraction 1 − h(B+C)/(h(B)+h(C))), debiased amplitude
   ratio |B⊥|/|C⊥| = √(S/F), and the ratio=1 crossing per route.
   Pre-registered (proposal §19.0): the scale-covariance account predicts
   completeness ordered by the safety map — 512 least complete. The
   refutation branch — 512 keeping ρ ≈ −0.9 with only the amplitude
   ratio broken — keeps only the "unsafe = mismatched magnitudes" version.
2. Shared-arm validity: ``I_sameset_biascheck`` vs cross-set I per
   (route, bin) — closes the "is the headline an artifact?" question
   from committed numbers alone.
3. The frozen target curve for 19.1: per (route, bin) ρ + amplitude
   ratio with reliability gates, emitted to ``frozen_target_190.json``.
   19.1's closure prediction is scored against THIS table.

Reliability gate (inherited from E14): any cell with rel_cos_B or
rel_cos_C < 0.5 is marked non-reading and excluded from summaries.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e19/reads_190.py
"""

from __future__ import annotations

import json
from pathlib import Path

PAPER_BENCH = Path(__file__).resolve().parents[2]
E14_RUN = PAPER_BENCH / "runs" / "20260801-2304-e14-ledger-probematched"
OUT = Path(__file__).resolve().parent / "frozen_target_190.json"

REL_GATE = 0.5
ROUTES = ["896", "768", "512"]
# E9's window localizations: 768 reduction-failure window center at the
# crossing σ ≈ 0.688; 896 crossing ≈ 0.5. Window bins tabulated for §3.
WINDOWS = {"896": (0.30, 0.70), "768": (0.43, 0.83)}


def amp_ratio(row: dict) -> float:
    """Debiased |B⊥|/|C⊥| = √(S/F); nan when debiasing pushed S or F ≤ 0."""
    s, f = row["S"], row["F"]
    if s <= 0 or f <= 0:
        return float("nan")
    return (s / f) ** 0.5


def cancel_frac(row: dict) -> float:
    """Cancellation completeness 1 − h(B+C)/(h(B)+h(C)) ∈ (−∞, 1]."""
    add = row["h_B"] + row["h_C"]
    if add <= 0:
        return float("nan")
    return 1.0 - row["h_B_plus_C"] / add


def reliable(row: dict) -> bool:
    return row["rel_cos_B"] >= REL_GATE and row["rel_cos_C"] >= REL_GATE


def crossings(sigmas: list[float], ratios: list[float]) -> list[float]:
    """σ of every ratio=1 crossing (linear interpolation, nan-skipping)."""
    pts = [(s, r) for s, r in zip(sigmas, ratios) if r == r]
    out = []
    for (s0, r0), (s1, r1) in zip(pts[:-1], pts[1:]):
        if (r0 - 1.0) * (r1 - 1.0) < 0:
            out.append(round(s0 + (1.0 - r0) / (r1 - r0) * (s1 - s0), 4))
    return out


def route_table(led: dict, route: str, sigmas: list[float]) -> list[dict]:
    rows = []
    for s, r in zip(sigmas, led["bc_ledger"][route]):
        rows.append(
            {
                "sigma": s,
                "rho": r["rho"],
                "amp_ratio_BoverC": round(amp_ratio(r), 4),
                "cancel_frac": round(cancel_frac(r), 4),
                "h_B": r["h_B"],
                "h_C": r["h_C"],
                "h_B_plus_C": r["h_B_plus_C"],
                "I": r["I"],
                "I_sameset": r["I_sameset_biascheck"],
                "rel_cos_B": r["rel_cos_B"],
                "rel_cos_C": r["rel_cos_C"],
                "reads": reliable(r),
            }
        )
    return rows


def main() -> None:
    sigmas = None
    ledgers = {}
    for name, fn in (("reenc", "ledger.json"), ("native", "ledger_native.json")):
        led = json.loads((E14_RUN / fn).read_text())
        ledgers[name] = led
        sigmas = led["sigma_centers"]

    result: dict = {"run": str(E14_RUN), "sigma_centers": sigmas, "rel_gate": REL_GATE}

    for ref, led in ledgers.items():
        block: dict = {}
        for route in ROUTES:
            tab = route_table(led, route, sigmas)
            ok = [r for r in tab if r["reads"]]
            block[route] = {
                "table": tab,
                "crossings_sigma": crossings(
                    [r["sigma"] for r in tab],
                    [
                        r["amp_ratio_BoverC"] if r["reads"] else float("nan")
                        for r in tab
                    ],
                ),
                "mean_cancel_frac_reliable": round(
                    sum(r["cancel_frac"] for r in ok) / len(ok), 4
                )
                if ok
                else None,
                "mean_abs_rho_reliable": round(
                    sum(abs(r["rho"]) for r in ok) / len(ok), 4
                )
                if ok
                else None,
                "n_reliable": len(ok),
            }
        result[f"ref_{ref}"] = block

    # ---- printed reads ----------------------------------------------------
    for ref in ("reenc", "native"):
        print(f"\n================ data_ref = {ref} ================")
        for route in ROUTES:
            b = result[f"ref_{ref}"][route]
            print(
                f"\n-- route {route}  (crossings σ = {b['crossings_sigma']}, "
                f"mean cancel = {b['mean_cancel_frac_reliable']}, "
                f"mean |ρ| = {b['mean_abs_rho_reliable']}, "
                f"reliable {b['n_reliable']}/{len(b['table'])}) --"
            )
            print(
                "σ        ρ       |B|/|C|  cancel   h(B)     h(C)     h(B+C)  "
                "I        I_same   gate"
            )
            for r in b["table"]:
                w = WINDOWS.get(route)
                mark = "w" if w and w[0] <= r["sigma"] <= w[1] else " "
                print(
                    f"{r['sigma']:<7} {r['rho']:+.3f}  {r['amp_ratio_BoverC']:>7.3f}"
                    f"  {r['cancel_frac']:+.3f}  {r['h_B']:.5f}  {r['h_C']:.5f}"
                    f"  {r['h_B_plus_C']:.5f} {r['I']:+.4f}  {r['I_sameset']:+.4f}"
                    f"  {'ok' if r['reads'] else 'X '}{mark}"
                )

    # ---- pre-registered scores -------------------------------------------
    print("\n================ pre-registered reads ================")
    for ref in ("reenc", "native"):
        cf = {
            rt: result[f"ref_{ref}"][rt]["mean_cancel_frac_reliable"] for rt in ROUTES
        }
        rho = {rt: result[f"ref_{ref}"][rt]["mean_abs_rho_reliable"] for rt in ROUTES}
        ordered = cf["896"] >= cf["768"] > cf["512"]
        result[f"ref_{ref}"]["completeness_ordering_896_768_512"] = cf
        result[f"ref_{ref}"]["completeness_ordered_as_predicted"] = bool(ordered)
        print(
            f"[{ref}] completeness 896/768/512 = {cf['896']}/{cf['768']}/{cf['512']}"
            f" → 512-least-complete {'CONFIRMED' if ordered else 'NOT confirmed'};"
            f" mean|ρ| = {rho['896']}/{rho['768']}/{rho['512']}"
        )

    # shared-arm validity: same-set minus cross-set I, worst and typical
    for ref in ("reenc", "native"):
        deltas = []
        for route in ROUTES:
            for r in result[f"ref_{ref}"][route]["table"]:
                if r["reads"] and r["I"] == r["I"]:
                    deltas.append(abs(r["I_sameset"] - r["I"]))
        result[f"ref_{ref}"]["I_bias_absdelta_max"] = round(max(deltas), 5)
        result[f"ref_{ref}"]["I_bias_absdelta_mean"] = round(
            sum(deltas) / len(deltas), 5
        )
        print(
            f"[{ref}] |I_same − I_cross|: mean = "
            f"{result[f'ref_{ref}']['I_bias_absdelta_mean']}, max = "
            f"{result[f'ref_{ref}']['I_bias_absdelta_max']}"
        )

    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nfrozen target → {OUT}")


if __name__ == "__main__":
    main()
