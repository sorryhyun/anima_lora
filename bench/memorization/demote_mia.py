#!/usr/bin/env python3
"""Does σ-demoted training regularize memorization? Paired MIA bench.

Combo (`--sigma_lowres` with both spans cleared, the unscheduled E16 stacked
router: 1024→768 on σ∈(0.65,0.95), 1024→896 on the rest of σ>0.5) trains every
high-σ step on a lower-res sibling latent. High σ is where composition binds to
the conditioning, so this bench measures whether demotion lowers member-vs-holdout
membership signal.

DESIGN. N arms per seed, exactly paired (same seed → identical data order +
FM noise; the sample_ratio half is keyed to validation_seed, so all arms
train on the identical member half and the other half is a clean same-artist
holdout, the `bench_sincos_half_*` operating point):

  * native  — the calibrated plain-LoRA overfit recipe (dim 16 / α 90, 4 ep,
    sample_ratio 0.5, T-LoRA mask off).
  * combo   — identical + `--sigma_lowres`, spans cleared (stacked router).
  * half896 — identical + `--sigma_lowres`, span cleared, route2 cleared:
    the single half-line route 1024→896 on σ>0.5 only.

Each checkpoint then goes through ``loss_gap.py`` (the calibrated loss-MIA
probe) unchanged; this driver aggregates the paired comparison.

THE CONFOUND. ``loss_gap.py`` scores re-noised *native-resolution* latents at every σ. The combo arm trained σ>0.5 at demoted
resolution, so a lower member-vs-holdout AUC at σ>0.5 is ambiguous: less
member signal in the weights, or member signal stored at the demoted grid
where a native-res probe can't see it. Only σ ≤ threshold (both arms trained
native there) is an apples-to-apples read — so the HEADLINE metric is the
clean-σ composite AUC (per-item mean of delta_s over σ≤0.5, member vs
same-artist holdout), and the σ>0.5 rows are reported but labeled confounded.
Earlier runs put the member signal at σ≤0.7, mostly inside the clean window.

Usage::

    make daemon-run ARGS="bench/memorization/demote_mia.py --seeds 1001 1002"

    # probe-only rerun against existing checkpoints:
    .venv/bin/python bench/memorization/demote_mia.py --seeds 1001 1002 --skip_train

Cost: 2 arms x len(seeds) short trainings (~7 min each at this recipe; combo
slightly less) + one loss_gap probe per checkpoint. Existing checkpoints are
reused unless --force_retrain.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


from bench._common import make_run_dir, write_result
from bench.memorization.loss_gap import _auc, _perm_p

REPO_ROOT = Path(__file__).resolve().parents[2]

# arm -> extra train.py argv. base.toml carries the stacked-router combolate
# values (routes + thresholds); the arms only clear pieces of that surface.
ARM_EXTRA = {
    "native": [],
    # both spans cleared -> unscheduled stacked router (combo, E16)
    "combo": ["--sigma_lowres", "--sigma_lowres_span", "", "--sigma_lowres_span2", ""],
    # route2 cleared -> single certified half-line 1024:896 on sigma>0.5
    "half896": [
        "--sigma_lowres",
        "--sigma_lowres_span",
        "",
        "--sigma_lowres_route2",
        "",
    ],
}
RESULTS_DIR = REPO_ROOT / "bench" / "memorization" / "results"
CKPT_DIR = REPO_ROOT / "output" / "ckpt"
DEFAULT_SIGMAS = [0.3, 0.5, 0.6, 0.7, 0.9]  # 0.6 reads the 896-only route


def ckpt_name(arm: str, seed: int, epochs: int) -> str:
    # Epochs in the name so operating-point escalations (--epochs 8/12) never
    # collide with — or silently reuse — another point's checkpoints. NB the
    # first run (2026-08-18, 4 ep) predates this and is named without `_e4`.
    return f"bench_demomia_{arm}_e{epochs}_s{seed}"


def resolve_ckpt(arm: str, seed: int, epochs: int) -> Path:
    """Checkpoint path, falling back to the legacy no-`_e4` name for the
    2026-08-18 first run so its arms are reused, not retrained."""
    ckpt = CKPT_DIR / f"{ckpt_name(arm, seed, epochs)}.safetensors"
    if not ckpt.exists() and epochs == 4:
        legacy = CKPT_DIR / f"bench_demomia_{arm}_s{seed}.safetensors"
        if legacy.exists():
            return legacy
    return ckpt


def train_argv(arm: str, seed: int, a) -> list[str]:
    """The calibrated sincos-half plain-LoRA overfit recipe (report.md), one
    knob varied. Everything memorization-relevant is pinned on the CLI — the
    July repro trap: sample_ratio=0.5 lives nowhere in tracked config."""
    argv = [
        sys.executable,
        "train.py",
        "--method",
        a.method,
        "--preset",
        a.preset,
        "--network_dim",
        str(a.network_dim),
        "--network_alpha",
        str(a.network_alpha),
        "--max_train_epochs",
        str(a.epochs),
        "--sample_ratio",
        str(a.sample_ratio),
        "--path_pattern",
        a.path_pattern,
        "--caption_dropout_rate",
        str(a.caption_dropout_rate),
        "--seed",
        str(seed),
        "--network_args",
        "use_timestep_mask=false",
        "--output_name",
        ckpt_name(arm, seed, a.epochs),
    ]
    argv += ARM_EXTRA[arm]
    return argv


def run(argv: list[str], what: str) -> None:
    print(f"\n=== {what}\n    {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {proc.returncode})")


def probe_run_dir(label: str, legacy_label: str | None = None) -> Path | None:
    """Newest loss_gap run dir for `label`, trying `legacy_label` (the
    pre-`_e4` naming of the 2026-08-18 first run) second. None if neither."""
    for lb in (label, legacy_label):
        if lb:
            hits = sorted(RESULTS_DIR.glob(f"*-{lb}"))
            if hits:
                return hits[-1]
    return None


def composite_auc(
    per_item: Path, sigmas: list[float], seed: int
) -> dict[str, dict[str, float]]:
    """Clean-σ / demoted-σ composite member-vs-holdout AUCs from a loss_gap
    per_item.csv (per-item mean of delta_s over each σ subset)."""
    rows = list(csv.DictReader(per_item.open()))
    out: dict[str, dict[str, float]] = {}
    for name, subset in (
        ("clean", [s for s in sigmas if s <= 0.5]),
        ("demoted", [s for s in sigmas if s > 0.5]),
    ):
        if not subset:
            continue

        def comp(group: str) -> np.ndarray:
            return np.array(
                [
                    np.mean([float(r[f"delta_s{s:g}"]) for s in subset])
                    for r in rows
                    if r["group"] == group
                ],
                dtype=np.float64,
            )

        mem, same = comp("member"), comp("same_artist")
        out[name] = {
            "sigmas": subset,
            "auc": round(_auc(mem, same), 4),
            "member_gap": round(float(mem.mean() - same.mean()), 5),
            "perm_p": round(_perm_p(mem, same, np.random.default_rng(seed)), 5),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["native", "combo"],
        choices=sorted(ARM_EXTRA),
        help="arms to run; 'native' is the pairing baseline and is required.",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002])
    ap.add_argument("--network_dim", type=int, default=16)
    ap.add_argument("--network_alpha", type=float, default=90.0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--sample_ratio", type=float, default=0.5)
    ap.add_argument("--path_pattern", default="sincos/*")
    ap.add_argument("--caption_dropout_rate", type=float, default=0.1)
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument("--num_members", type=int, default=48)
    ap.add_argument("--num_holdout", type=int, default=48)
    ap.add_argument("--n_probes", type=int, default=4)
    ap.add_argument(
        "--skip_train",
        action="store_true",
        help="probe-only: every arm checkpoint must already exist.",
    )
    ap.add_argument(
        "--force_retrain",
        action="store_true",
        help="retrain arms even when their checkpoint exists.",
    )
    ap.add_argument(
        "--force_probe",
        action="store_true",
        help="re-run loss_gap even when a probe run dir for the arm exists. "
        "Probe noise is stem-seeded (deterministic), so reuse is exact — "
        "force only after a loss_gap.py or cache change.",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="run-dir label (default: demote_mia_e<epochs>). NB `make "
        "daemon-run` plucks --label out of ARGS for itself — rely on the "
        "default when submitting through the daemon.",
    )
    args = ap.parse_args()
    if args.label is None:
        args.label = f"demote_mia_e{args.epochs}"
    if "native" not in args.arms:
        raise SystemExit("--arms must include 'native' (the pairing baseline).")
    arms = ["native"] + [a for a in args.arms if a != "native"]

    clean_sigmas = [s for s in args.sigmas if s <= 0.5]
    if not clean_sigmas:
        raise SystemExit(
            "--sigmas has no σ ≤ 0.5 — the clean (native-trained-in-both-arms) "
            "window is the headline; include at least one such σ."
        )

    # ── train the arms (sequential — one GPU) ─────────────────────────────
    for seed in args.seeds:
        for arm in arms:
            ckpt = resolve_ckpt(arm, seed, args.epochs)
            if ckpt.exists() and not args.force_retrain:
                print(f"reusing existing {ckpt.name}", flush=True)
                continue
            if args.skip_train:
                raise RuntimeError(f"--skip_train but {ckpt} does not exist")
            run(train_argv(arm, seed, args), f"train {arm} seed {seed}")
            ckpt = CKPT_DIR / f"{ckpt_name(arm, seed, args.epochs)}.safetensors"
            if not ckpt.exists():
                raise RuntimeError(f"training finished but {ckpt} was not written")

    # ── probe each checkpoint with the unmodified loss_gap ────────────────
    results: dict[str, dict] = {}  # f"{arm}_s{seed}" -> record
    for seed in args.seeds:
        for arm in arms:
            key = f"{arm}_s{seed}"
            label = f"demomia_e{args.epochs}_{key}"
            legacy_label = f"demomia_{key}" if args.epochs == 4 else None
            rd = probe_run_dir(label, legacy_label)
            if rd is not None and not args.force_probe:
                print(f"reusing probe run {rd.name} for {key}", flush=True)
            else:
                argv = [
                    sys.executable,
                    "bench/memorization/loss_gap.py",
                    "--adapter",
                    str(resolve_ckpt(arm, seed, args.epochs)),
                    "--method",
                    args.method,
                    "--preset",
                    args.preset,
                    "--num_members",
                    str(args.num_members),
                    "--num_holdout",
                    str(args.num_holdout),
                    "--n_probes",
                    str(args.n_probes),
                    "--sigmas",
                    *[f"{s:g}" for s in args.sigmas],
                    "--label",
                    label,
                ]
                run(argv, f"loss_gap {key}")
                rd = probe_run_dir(label)
                if rd is None:
                    raise RuntimeError(f"loss_gap ran but no run dir *-{label}")
            metrics = json.loads((rd / "result.json").read_text())["metrics"]
            results[key] = {
                "run_dir": str(rd.relative_to(REPO_ROOT)),
                "auc_all": metrics.get("auc_member_vs_same"),
                "perm_p_all": metrics.get("perm_p_member_vs_same"),
                "member_gap_all": metrics.get("member_gap"),
                "per_sigma": metrics.get("per_sigma", {}),
                **{
                    f"{k}_{f}": v[f]
                    for k, v in composite_auc(
                        rd / "per_item.csv", args.sigmas, seed
                    ).items()
                    for f in ("auc", "member_gap", "perm_p")
                },
            }

    # ── the paired comparisons (each demote arm vs native) ────────────────
    demote_arms = [a for a in arms if a != "native"]
    deltas_by_arm = {
        arm: [
            results[f"{arm}_s{s}"]["clean_auc"] - results[f"native_s{s}"]["clean_auc"]
            for s in args.seeds
        ]
        for arm in demote_arms
    }
    verdicts: dict[str, str] = {}
    for arm, deltas in deltas_by_arm.items():
        mean_delta = float(np.mean(deltas))
        consistent = all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
        if mean_delta <= -0.07 and consistent:
            v = (
                f"REGULARIZING signal — {arm} lowers the clean-σ "
                f"member-vs-holdout AUC by {mean_delta:+.3f} on average, same "
                f"sign in every paired seed. Confirm generation-side with "
                f"probe.py before claiming it."
            )
        elif mean_delta >= 0.07 and consistent:
            v = (
                f"{arm} WORSENS weight-side memorization on the clean window "
                f"({mean_delta:+.3f} mean paired ΔAUC, consistent across seeds)."
            )
        elif consistent or abs(mean_delta) < 0.07:
            v = (
                f"NEUTRAL — mean paired clean-σ ΔAUC {mean_delta:+.3f} is "
                f"inside the ±0.05 probe noise band; {arm} is not a "
                f"memorization lever at this operating point."
            )
        else:
            v = (
                f"MIXED (seed-dependent): paired clean-σ ΔAUC "
                f"{[round(d, 3) for d in deltas]} — add seeds before reading "
                f"a direction."
            )
        verdicts[arm] = v
        print(f"\n[{arm}] {v}", flush=True)

    # Interpolation readout: per seed, where a partial arm's clean-σ AUC lands
    # on the native(t=0) → combo(t=1) axis. t≈0.5 = demote coverage is a dial.
    interp: dict[str, list[float]] = {}
    if "combo" in deltas_by_arm:
        for arm in demote_arms:
            if arm == "combo":
                continue
            interp[arm] = [
                round(deltas_by_arm[arm][i] / deltas_by_arm["combo"][i], 3)
                if abs(deltas_by_arm["combo"][i]) > 1e-9
                else float("nan")
                for i in range(len(args.seeds))
            ]
            print(
                f"[{arm}] interpolation t (0=native, 1=combo), per seed: {interp[arm]}",
                flush=True,
            )

    # ── artifacts ─────────────────────────────────────────────────────────
    run_dir = make_run_dir("memorization", label=args.label)
    M = [f"# σ-demoted ({'/'.join(demote_arms)}) vs native — paired weight-side MIA\n"]
    M.append(
        f"- recipe: `--method {args.method} --preset {args.preset}` · "
        f"dim {args.network_dim} / α {args.network_alpha:g} · "
        f"{args.epochs} ep · sample_ratio {args.sample_ratio} · "
        f"`{args.path_pattern}` · T-LoRA mask off · seeds {args.seeds}"
    )
    M.append(
        f"- probe: loss_gap {args.num_members}/{args.num_holdout}, "
        f"σ {args.sigmas}, K={args.n_probes}. July anchors: plain 0.82, "
        f"tenth 0.54."
    )
    M.append("\n## Verdict\n")
    for arm in demote_arms:
        M.append(f"- **{arm}**: {verdicts[arm]}")
    for arm, ts in interp.items():
        M.append(
            f"- **{arm}** interpolation t on the native(0)→combo(1) clean-σ "
            f"AUC axis, per seed: {ts}"
        )
    M.append(
        "\nHeadline = **clean-σ composite** (σ≤0.5 — native-trained in every "
        "arm). σ>0.5 rows are resolution-confounded for the demote arms "
        "(their high-σ steps trained at the demoted grid; the probe scores "
        "native): a drop there is NOT interpretable as less memorization.\n"
    )
    M.append(
        "| arm | seed | AUC clean (σ≤0.5) | p | gap | AUC demoted (σ>0.5)† | AUC all-σ |"
    )
    M.append("|---|---|---|---|---|---|---|")
    for seed in args.seeds:
        for arm in arms:
            r = results[f"{arm}_s{seed}"]
            M.append(
                f"| {arm} | {seed} | **{r['clean_auc']:.3f}** "
                f"| {r['clean_perm_p']:.4f} | {r['clean_member_gap']:+.4f} "
                f"| {r.get('demoted_auc', float('nan')):.3f} "
                f"| {r['auc_all']:.3f} |"
            )
    M.append("\n† confounded column — see above.\n")
    for arm in demote_arms:
        M.append(
            f"Paired clean-σ ΔAUC ({arm} − native) per seed: "
            f"{[round(d, 3) for d in deltas_by_arm[arm]]} · "
            f"mean {float(np.mean(deltas_by_arm[arm])):+.3f}\n"
        )
    M.append("## Per-σ AUC (member vs same-artist holdout)\n")
    M.append(
        "| σ | " + " | ".join(f"{a}_s{s}" for s in args.seeds for a in arms) + " |"
    )
    M.append("|---|" + "---|" * (len(arms) * len(args.seeds)))
    for s in args.sigmas:
        cells = [
            f"{results[f'{arm}_s{seed}']['per_sigma'].get(f'{s:g}', {}).get('auc', float('nan')):.3f}"
            for seed in args.seeds
            for arm in arms
        ]
        mark = "" if s <= 0.5 else " †"
        M.append(f"| {s:g}{mark} | " + " | ".join(cells) + " |")
    M.append("\nPer-arm probe artifacts:\n")
    for key, r in results.items():
        M.append(f"- {key}: `{r['run_dir']}`")
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "verdicts": verdicts,
            "mean_paired_delta_auc_clean": {
                arm: round(float(np.mean(d)), 4) for arm, d in deltas_by_arm.items()
            },
            "paired_delta_auc_clean": {
                arm: [round(x, 4) for x in d] for arm, d in deltas_by_arm.items()
            },
            "interp_t_vs_combo": interp,
            "clean_sigmas": clean_sigmas,
            "arms": results,
        },
        label=args.label,
        artifacts=["summary.md"],
    )
    print(f"\n  -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
