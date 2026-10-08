#!/usr/bin/env python3
"""Phase 0 of the memorization low-σ reweight proposal — retrain ± reweight.

Orchestrates ``_archive/proposals/memorization_lowsigma_reweight.md`` Phase 0: retrain
the banked known-overfit recipe (``bench_sincos_half_plain`` — sincos, 167-member
half shard, 4 epochs, dim16/α90, AUC 0.82) with the online Δ-gap tracker ON, then
judge it with the FROZEN existing instruments — nothing new is built to score
this (the proposal is explicitly not an eval proposal):

  G1  ``loss_gap.py`` member-vs-holdout AUC: banked 0.82 → gate **≤ 0.65**.
  SANITY  the online per-item Δ-EMA ranking (``<ckpt>_memgap.json``) must
      correlate (Spearman, member rows) with the offline probe's per-item
      delta — they estimate the same statistic; disagreement means the online
      estimator is broken and G1/G2 must not be read.
  CHURN  kill criterion 3: flagged-set Jaccard churn between snapshots
      > ~50% ⇒ estimator too noisy at trainable batch sizes.
  G2/G3  generation-side probe + no-harm render grid are printed as follow-up
      commands (heavier; run manually once G1+SANITY look sane).

Arms:
  * ``--arm reweight``  (default) — Arm A, the intervention.
  * ``--arm measure``   — tracker only, loss untouched. AUC should match the
    banked 0.82 (the measurement itself must be inert); validates the
    measure-only report that ships regardless.

Usage (GPU must be free — one ~670-step training run + one probe pass)::

    uv run python bench/memorization/reweight_bench.py --arm reweight
    uv run python bench/memorization/reweight_bench.py --arm measure
    # re-score an existing checkpoint without retraining:
    uv run python bench/memorization/reweight_bench.py --skip_train \
        --adapter output/ckpt/bench_sincos_half_rw_reweight.safetensors

Envelope lands in ``bench/memorization/results/<ts>-reweight_<arm>[_<label>]/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

RESULTS = REPO_ROOT / "bench" / "memorization" / "results"
BANKED_RESULT = RESULTS / "20260704-1839-sincos_half_plain" / "result.json"
BANKED_AUC_FALLBACK = 0.822

# The banked overfit recipe — mirrors output/ckpt/bench_sincos_half_plain
# .snapshot.toml. Everything else (down_init=weight_svd, REPA, caption_dropout
# 0.1) comes from configs/methods/lora.toml — but that file's comment-toggle
# defaults MUTATE over time (the 2026-07-06 state drifted to lr 2e-5 + the
# active T-LoRA mask, which killed the overfit and invalidated the first
# Phase-0 run), so every knob the banked snapshot disagrees with today's TOML
# on is pinned here explicitly. sample_ratio is pinned on CLI because it is not
# in tracked config (docs/experimental/soup.md).
RECIPE = [
    "--method",
    "lora",
    "--preset",
    "default",
    "--path_pattern",
    "sincos/*",
    "--sample_ratio",
    "0.5",
    "--max_train_epochs",
    "4",
    "--network_dim",
    "16",
    "--network_alpha",
    "90",
    "--learning_rate",
    "5e-5",
    "--network_args",
    "use_timestep_mask=false",
]


def _run(cmd: list[str], dry: bool) -> None:
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return
    r = subprocess.run([str(c) for c in cmd], cwd=REPO_ROOT)
    if r.returncode:
        sys.exit(r.returncode)


def _newest_result_dir(label: str) -> Path:
    cands = sorted(RESULTS.glob(f"*-{label}"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no results dir matching *-{label} under {RESULTS}")
    return cands[-1]


def _banked_auc(path: Path) -> tuple[float, str]:
    try:
        d = json.loads(path.read_text())
        return float(d["metrics"]["auc_member_vs_same"]), str(path)
    except (OSError, KeyError, ValueError):
        return BANKED_AUC_FALLBACK, "fallback (banked result.json unreadable)"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ via Pearson on ranks (average-rank ties; no scipy dep)."""

    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty(len(x))
        sx = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0
            i = j + 1
        return ranks

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _sanity_correlation(memgap: dict, per_item_csv: Path) -> dict:
    """Join online Δ-EMA with the offline probe's member deltas on stem."""
    online = {
        it["stem"]: it["ema_corrected"]
        for it in memgap.get("items", [])
        if it.get("count", 0) > 0
    }
    offline = {}
    with per_item_csv.open() as f:
        for row in csv.DictReader(f):
            if row["group"] == "member":
                offline[row["stem"]] = float(row["delta"])
    stems = sorted(set(online) & set(offline))
    out = {"n_joined": len(stems), "n_online": len(online), "n_offline": len(offline)}
    if len(stems) >= 10:
        a = np.array([online[s] for s in stems])
        b = np.array([offline[s] for s in stems])
        out["spearman"] = round(_spearman(a, b), 4)
    else:
        out["spearman"] = None
    return out


def _flag_churn(memgap: dict) -> float | None:
    """Mean Jaccard-distance between consecutive flagged-set snapshots."""
    hist = [set(h["stems"]) for h in memgap.get("flag_history", [])]
    hist = [h for h in hist if h]
    if len(hist) < 2:
        return None
    ds = []
    for a, b in zip(hist, hist[1:]):
        union = a | b
        ds.append(1.0 - len(a & b) / len(union))
    return round(float(np.mean(ds)), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arm", choices=["reweight", "measure"], default="reweight")
    ap.add_argument("--label", default=None, help="suffix for names/dirs.")
    ap.add_argument(
        "--skip_train",
        action="store_true",
        help="skip training; score --adapter as-is.",
    )
    ap.add_argument(
        "--adapter",
        default=None,
        help="checkpoint to score (default: the one this run trains).",
    )
    ap.add_argument(
        "--baseline_result",
        default=str(BANKED_RESULT),
        help="banked OFF-run loss_gap result.json (G1 reference).",
    )
    ap.add_argument("--g1_auc_gate", type=float, default=0.65)
    ap.add_argument("--sanity_rho_gate", type=float, default=0.2)
    ap.add_argument("--churn_gate", type=float, default=0.5)
    # tracker knobs forwarded to train.py (defaults = the flag defaults there,
    # except measure_every: bs=1 × 668 steps needs every-step coverage).
    ap.add_argument("--mem_measure_every", type=int, default=1)
    ap.add_argument("--mem_z_threshold", type=float, default=1.5)
    ap.add_argument("--mem_ema_beta", type=float, default=0.3)
    ap.add_argument("--mem_weight_temp", type=float, default=0.5)
    ap.add_argument("--mem_weight_floor", type=float, default=0.0)
    ap.add_argument("--mem_warmup_updates", type=int, default=100)
    ap.add_argument(
        "--mem_extra_sigmas",
        type=float,
        nargs="*",
        default=None,
        help="enable multi-draw measurement at this σ grid (e.g. 0.5 0.7) — "
        "the Phase-0.5 estimator fix; ~3x per-item counts, ~4x lower variance.",
    )
    ap.add_argument("--mem_extra_noise", type=int, default=2)
    ap.add_argument(
        "--train_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra argv appended to the train.py invocation.",
    )
    ap.add_argument("--dry_run", action="store_true", help="print commands only.")
    args = ap.parse_args()

    suffix = f"{args.arm}" + (f"_{args.label}" if args.label else "")
    name = f"bench_sincos_half_rw_{suffix}"
    ckpt = (
        Path(args.adapter)
        if args.adapter
        else REPO_ROOT / "output" / "ckpt" / f"{name}.safetensors"
    )
    memgap_path = ckpt.parent / f"{ckpt.stem}_memgap.json"

    # ── 1. train (the banked recipe + tracker) ────────────────────────────
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            REPO_ROOT / "train.py",
            *RECIPE,
            "--output_name",
            name,
            "--mem_reweight_mode",
            args.arm,
            "--mem_measure_every",
            args.mem_measure_every,
            "--mem_z_threshold",
            args.mem_z_threshold,
            "--mem_ema_beta",
            args.mem_ema_beta,
            "--mem_weight_temp",
            args.mem_weight_temp,
            "--mem_weight_floor",
            args.mem_weight_floor,
            "--mem_warmup_updates",
            args.mem_warmup_updates,
            *(
                ["--mem_extra_sigmas", *args.mem_extra_sigmas]
                + ["--mem_extra_noise", args.mem_extra_noise]
                if args.mem_extra_sigmas
                else []
            ),
            *args.train_args,
        ]
        _run(train_cmd, args.dry_run)

    # ── 2. offline probe (frozen instrument) ──────────────────────────────
    gap_label = f"rw_{suffix}"
    _run(
        [
            sys.executable,
            REPO_ROOT / "bench" / "memorization" / "loss_gap.py",
            "--adapter",
            ckpt,
            "--label",
            gap_label,
        ],
        args.dry_run,
    )
    if args.dry_run:
        print("[dry_run] stopping before analysis")
        return

    gap_dir = _newest_result_dir(gap_label)
    gap = json.loads((gap_dir / "result.json").read_text())["metrics"]
    auc = float(gap["auc_member_vs_same"])
    baseline_auc, baseline_src = _banked_auc(Path(args.baseline_result))

    # ── 3. sanity: online estimator vs offline probe ──────────────────────
    if not memgap_path.exists():
        raise SystemExit(
            f"{memgap_path} missing — train.py did not dump tracker state "
            "(was --mem_reweight_mode actually set?)"
        )
    memgap = json.loads(memgap_path.read_text())
    sanity = _sanity_correlation(memgap, gap_dir / "per_item.csv")
    churn = _flag_churn(memgap)

    # ── 4. gates ───────────────────────────────────────────────────────────
    rho = sanity["spearman"]
    sanity_ok = rho is not None and rho >= args.sanity_rho_gate
    churn_ok = churn is None or churn <= args.churn_gate
    if args.arm == "measure":
        # measurement must be inert: AUC should stay at the banked overfit.
        g1 = None
        g1_txt = (
            f"n/a (measure arm) — AUC {auc:.3f} vs banked {baseline_auc:.3f}; "
            "a large drop here would mean the measurement itself perturbs "
            "training (it must not)."
        )
    else:
        g1 = auc <= args.g1_auc_gate
        g1_txt = (
            f"{'PASS' if g1 else 'FAIL'} — AUC {auc:.3f} "
            f"(banked OFF {baseline_auc:.3f}, gate ≤ {args.g1_auc_gate})"
        )
        if not sanity_ok:
            g1_txt += "  [UNRELIABLE: sanity gate failed — fix estimator first]"

    metrics = {
        "arm": args.arm,
        "adapter": str(ckpt),
        "auc_member_vs_same": auc,
        "auc_baseline_banked": baseline_auc,
        "baseline_source": baseline_src,
        "per_sigma": gap.get("per_sigma"),
        "perm_p_member_vs_same": gap.get("perm_p_member_vs_same"),
        "g1_pass": g1,
        "sanity_spearman": rho,
        "sanity_n_joined": sanity["n_joined"],
        "sanity_pass": sanity_ok,
        "flag_churn": churn,
        "churn_pass": churn_ok,
        "n_flagged_final": len(memgap.get("flagged_stems", [])),
        "tracker_updates": memgap.get("updates"),
        "loss_gap_run": str(gap_dir.relative_to(REPO_ROOT)),
    }

    run_dir = make_run_dir("memorization", label=f"reweight_{suffix}")
    write_result(
        run_dir, script=__file__, args=args, metrics=metrics, label=f"reweight_{suffix}"
    )

    M = [
        "# Phase 0 — memorization low-σ reweight (proposal gate run)\n",
        f"- arm: **{args.arm}** · adapter `{ckpt.name}` · "
        f"tracker updates {memgap.get('updates')} · "
        f"flagged {len(memgap.get('flagged_stems', []))} items",
        f"- offline probe run: `{gap_dir.name}`\n",
        "## Gates\n",
        "| gate | verdict |",
        "|---|---|",
        f"| G1 (AUC ≤ {args.g1_auc_gate}) | {g1_txt} |",
        f"| SANITY (online vs offline, Spearman ≥ {args.sanity_rho_gate}) | "
        f"{'PASS' if sanity_ok else 'FAIL'} — ρ={rho} on {sanity['n_joined']} members |",
        f"| CHURN (≤ {args.churn_gate}) | {'PASS' if churn_ok else 'FAIL'} — {churn} |",
        "\n## Follow-ups (heavier instruments — run manually)\n",
        "G2 (generation-side xerox check):\n",
        f"    uv run python bench/memorization/probe.py --adapter {ckpt}",
        "\nG3 (no-harm eyeball — flagged items' artist must keep fine detail; "
        "compare against the banked OFF adapter at matched seeds):\n",
        f"    uv run python bench/memorization/eyeball.py --adapter {ckpt}",
        "\nFlagged stems (the 'which images is this LoRA xeroxing' report):\n",
        "    " + ", ".join(memgap.get("flagged_stems", []) or ["(none)"]),
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(M), encoding="utf-8")
    print(f"\nwrote {run_dir}/result.json + summary.md")
    for line in M[4:12]:
        print(line)


if __name__ == "__main__":
    main()
