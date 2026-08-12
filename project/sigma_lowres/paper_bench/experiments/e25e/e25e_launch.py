"""E25e grid launcher — submits the frozen 19-run training grid.

Registered design (README, frozen 2026-08-12): E16.1 protocol — 2
corpora x 3 seeds (1001-1003) x 480 steps bs 1, --deterministic
--paired_step_rng, stock lora recipe. 3 arms (native / combo /
rescond_c = combo + --sigma_lowres_res_cond
--sigma_lowres_res_cond_centered) + one determinism-control duplicate
(combo, hews, s1001). Corpus path_patterns from the frozen E4 manifest
dir; combo argv matches E25b Stage 2 verbatim.

Submission order fail-fasts the one never-run flag combination first
(rescond_c: the centered variant has trained zero steps anywhere).

Usage: python e25e_launch.py [--dry-run] [--skip N]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
MANIFEST_DIR = REPO / "project/sigma_lowres/paper_bench/runs/20260729-1537-e4-manifest"
sys.path.insert(0, str(REPO))

from anima_daemon.client import DaemonClient  # noqa: E402

COMBO = [
    "--sigma_lowres",
    "--sigma_lowres_route2",
    "1024:768",
    "--sigma_lowres_threshold2",
    "0.65",
    "--sigma_lowres_threshold2_max",
    "0.95",
]
RES_COND_C = ["--sigma_lowres_res_cond", "--sigma_lowres_res_cond_centered"]

ARMS = {
    "native": [],
    "combo": COMBO,
    "rescond_c": COMBO + RES_COND_C,
}

# (short name, pattern file, epochs) — 60x8 / 15x32 = 480 steps each
CORPORA = [
    ("hews", "path_pattern_hews_train.txt", 8),
    ("channel", "path_pattern_channel_caststation_train.txt", 32),
]
SEEDS = [1001, 1002, 1003]

FAIL_FAST = "e25e_hews_rescond_c_s1001"


def cell_extra(
    pattern: str, epochs: int, seed: int, arm_argv: list[str], out: str
) -> list[str]:
    return [
        "--seed",
        str(seed),
        "--deterministic",
        "--paired_step_rng",
        "--path_pattern",
        pattern,
        "--max_train_epochs",
        str(epochs),
        "--save_every_n_epochs",
        str(epochs),
        "--checkpointing_epochs",
        str(epochs),
        *arm_argv,
        "--output_name",
        out,
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip",
        type=int,
        default=0,
        help="skip the first N cells (resume a partial submission)",
    )
    args = ap.parse_args()

    patterns = {
        short: (MANIFEST_DIR / fname).read_text(encoding="utf-8").strip()
        for short, fname, _ in CORPORA
    }

    cells: list[tuple[str, list[str]]] = []
    # fail-fast cell first: the never-run centered flag combination
    cells.append(
        (FAIL_FAST, cell_extra(patterns["hews"], 8, 1001, ARMS["rescond_c"], FAIL_FAST))
    )
    for short, _fname, epochs in CORPORA:
        for seed in SEEDS:
            for arm, arm_argv in ARMS.items():
                out = f"e25e_{short}_{arm}_s{seed}"
                if out == FAIL_FAST:
                    continue  # already queued as the fail-fast cell
                cells.append(
                    (out, cell_extra(patterns[short], epochs, seed, arm_argv, out))
                )
    # determinism control: exact duplicate argv, distinct output name
    cells.append(
        (
            "e25e_hews_combo_s1001_ctrl2",
            cell_extra(
                patterns["hews"], 8, 1001, ARMS["combo"], "e25e_hews_combo_s1001_ctrl2"
            ),
        )
    )
    assert len(cells) == 19, len(cells)

    if args.dry_run:
        for name, extra in cells:
            print(name, " ".join(a if len(a) < 70 else a[:40] + "..." for a in extra))
        return

    client = DaemonClient()
    out_path = HERE / "e25e_jobs.json"
    ids = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name, extra in cells[args.skip :]:
        rec = client.submit(method="lora", preset="default", extra=extra)
        ids[name] = rec.get("job_id") or rec["id"]
        print(f"{ids[name]}  {name}")
    out_path.write_text(json.dumps(ids, indent=1), encoding="utf-8")
    print(f"\n{len(ids)} jobs queued; ids -> e25e_jobs.json")


if __name__ == "__main__":
    main()
