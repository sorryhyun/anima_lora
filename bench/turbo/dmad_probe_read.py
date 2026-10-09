#!/usr/bin/env python3
"""Read a DMAD Phase −1 probe run (docs/proposal/turbo_dmad.md § Phase −1).

Input is the ``dmad_probe.jsonl`` a ``--dmad_probe`` turbo run writes into its log
dir (one row per step, ``scripts/distill_turbo/dmad_probe.py``). CPU-only.

Pre-registered read (fixed before the first run):

* Read window = the second half of the rows (by step); the first half is head-T
  burn-in.
* UNCONVERGED — read-window disc accuracy < 0.75: head T does not separate
  teacher from student yet, so its gradient says nothing about Prop. 1.
* KILL — paired ``cos − cos_null`` within 3 SEM of 0: indistinguishable from the
  permutation null.
* PASS — mean ``cos`` ≥ 0.5 × mean ``ceil_cos`` (two independent DM draws) AND
  ``agree − agree_null`` > 3 SEM.
* WEAK — anything between: aligned above the null but well short of how two DM
  draws agree with each other.

    python bench/turbo/dmad_probe_read.py output/logs/turbo/<run>/dmad_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

N_TAU_BINS = 8
ACC_FLOOR = 0.75
CEIL_FRACTION = 0.5
SEM_K = 3.0


def _mean_sem(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def _col(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if key in r]


def read(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["step"])
    window = rows[len(rows) // 2 :]

    agg = {
        k: _mean_sem(_col(window, k))
        for k in (
            "cos",
            "cos_null",
            "agree",
            "agree_null",
            "ceil_cos",
            "ceil_agree",
            "cos_dm2",
            "acc",
            "margin",
            "bce",
        )
    }
    d_cos = _mean_sem([r["cos"] - r["cos_null"] for r in window])
    d_agree = _mean_sem([r["agree"] - r["agree_null"] for r in window])

    acc = agg["acc"][0]
    ceil = agg["ceil_cos"][0]
    if not acc >= ACC_FLOOR:
        verdict = "UNCONVERGED"
    elif abs(d_cos[0]) <= SEM_K * d_cos[1]:
        verdict = "KILL"
    elif (
        not math.isnan(ceil)
        and agg["cos"][0] >= CEIL_FRACTION * ceil
        and d_agree[0] > SEM_K * d_agree[1]
    ):
        verdict = "PASS"
    else:
        verdict = "WEAK"

    tau_bins = []
    for b in range(N_TAU_BINS):
        lo, hi = b / N_TAU_BINS, (b + 1) / N_TAU_BINS
        sel = [
            r
            for r in window
            if lo <= r["tau_dm"] < hi or (b == N_TAU_BINS - 1 and r["tau_dm"] == 1.0)
        ]
        tau_bins.append(
            {
                "tau": [lo, hi],
                "n": len(sel),
                "cos": _mean_sem(_col(sel, "cos")),
                "cos_null": _mean_sem(_col(sel, "cos_null")),
                "agree": _mean_sem(_col(sel, "agree")),
                "ceil_cos": _mean_sem(_col(sel, "ceil_cos")),
            }
        )
    by_grad_step = {}
    for g in sorted({r["grad_step"] for r in window}):
        sel = [r for r in window if r["grad_step"] == g]
        by_grad_step[str(g)] = {
            "n": len(sel),
            "cos": _mean_sem(_col(sel, "cos")),
            "ceil_cos": _mean_sem(_col(sel, "ceil_cos")),
            "acc": _mean_sem(_col(sel, "acc")),
        }
    # Head-T trajectory in deciles over the whole run (burn-in visible).
    deciles = []
    n = len(rows)
    for i in range(10):
        sel = rows[i * n // 10 : (i + 1) * n // 10]
        if sel:
            deciles.append(
                {
                    "steps": [sel[0]["step"], sel[-1]["step"]],
                    "acc": _mean_sem(_col(sel, "acc"))[0],
                    "margin": _mean_sem(_col(sel, "margin"))[0],
                    "cos": _mean_sem(_col(sel, "cos"))[0],
                }
            )
    return {
        "verdict": verdict,
        "n_rows": n,
        "n_window": len(window),
        "window_steps": [window[0]["step"], window[-1]["step"]] if window else None,
        "aggregate": agg,
        "d_cos": d_cos,
        "d_agree": d_agree,
        "tau_bins": tau_bins,
        "by_grad_step": by_grad_step,
        "deciles": deciles,
    }


def _fmt(ms: tuple[float, float]) -> str:
    return f"{ms[0]:+.4f} ± {ms[1]:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--label", default="dmad_probe")
    ap.add_argument("--no_write", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line]
    res = read(rows)
    a = res["aggregate"]
    print(
        f"rows {res['n_rows']}, read window {res['window_steps']} (n={res['n_window']})"
    )
    print(
        f"disc acc {_fmt(a['acc'])}  margin {_fmt(a['margin'])}  bce {_fmt(a['bce'])}"
    )
    print(f"cos(g_T, DM)      {_fmt(a['cos'])}   null {_fmt(a['cos_null'])}")
    print(f"  paired Δ        {_fmt(res['d_cos'])}")
    print(f"agree-energy      {_fmt(a['agree'])}   null {_fmt(a['agree_null'])}")
    print(f"  paired Δ        {_fmt(res['d_agree'])}")
    print(
        f"ceiling cos(DM, DM') {_fmt(a['ceil_cos'])}   cos(g_T, DM') {_fmt(a['cos_dm2'])}"
    )
    print("τ-binned cos (n | cos | null | ceiling):")
    for b in res["tau_bins"]:
        print(
            f"  τ∈[{b['tau'][0]:.3f},{b['tau'][1]:.3f})  {b['n']:4d} | "
            f"{_fmt(b['cos'])} | {_fmt(b['cos_null'])} | {_fmt(b['ceil_cos'])}"
        )
    print("by DMD grad step (n | cos | ceiling | acc):")
    for g, v in res["by_grad_step"].items():
        print(
            f"  g={g}  {v['n']:4d} | {_fmt(v['cos'])} | {_fmt(v['ceil_cos'])} | {_fmt(v['acc'])}"
        )
    print("deciles (steps | acc | margin | cos):")
    for d in res["deciles"]:
        print(f"  {d['steps']}  {d['acc']:.3f} | {d['margin']:+.3f} | {d['cos']:+.4f}")
    print(f"VERDICT: {res['verdict']}")

    if not args.no_write:
        run_dir = make_run_dir("turbo", args.label)
        write_result(
            run_dir,
            script=__file__,
            args=args,
            metrics=res,
            label=args.label,
            extra={"source": str(args.jsonl)},
        )
        print(f"result: {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
