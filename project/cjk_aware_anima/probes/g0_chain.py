#!/usr/bin/env python3
"""plan_zh3 G0/G1 generation chain as ONE daemon job: for each arm, TE
re-cache through its pack (skipped for the seed-control arm) -> C9-recipe
LoRA train -> 8-row x 3-seed grid; then compose + push the double-blind
pair sets so the next session only grades.

Arms run in order; a failing arm is logged and skipped so the rest still
land. Every stage is a direct subprocess (never a nested daemon job).

    make daemon-run ARGS="--label g0-chain --stall-timeout 0 \
        project/cjk_aware_anima/probes/g0_chain.py --queue"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
PY = sys.executable
R2 = PROJ / "run_unmask_r2.py"
BP = HERE / "blind_pairs.py"
RECORDS = "post_image_dataset/cjk_unmask/ocr_records_sincos_ppocr.jsonl"
MIRROR = "post_image_dataset/cjk_unmask/mirror_sincos_ppocr"

# arm id -> (method, ext_prefix, te_out, skip_cache)
ARMS = {
    "C9s2": (
        "cjk_unmask_c9_seed2",
        "output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256",
        "post_image_dataset/cjk_unmask/te/sincos_jakozh1sym_r256",
        True,
    ),
    "HOT": (
        "cjk_unmask_hot",
        "output/ckpt/cjk_vocab_pack_hot_r256",
        "post_image_dataset/cjk_unmask/te/sincos_hot_r256",
        False,
    ),
    "COLLIDE": (
        "cjk_unmask_collide",
        "output/ckpt/cjk_vocab_pack_collide_r256",
        "post_image_dataset/cjk_unmask/te/sincos_collide_r256",
        False,
    ),
    "ROTATE": (
        "cjk_unmask_rotate",
        "output/ckpt/cjk_vocab_pack_rotate_r256",
        "post_image_dataset/cjk_unmask/te/sincos_rotate_r256",
        False,
    ),
    "COLLAPSE": (
        "cjk_unmask_collapse",
        "output/ckpt/cjk_vocab_pack_collapse_r256",
        "post_image_dataset/cjk_unmask/te/sincos_collapse_r256",
        False,
    ),
    "ISO1": (
        "cjk_unmask_iso1",
        "output/ckpt/cjk_vocab_pack_iso1_r256",
        "post_image_dataset/cjk_unmask/te/sincos_iso1_r256",
        False,
    ),
    "COLD": (
        "cjk_unmask_cold",
        "output/ckpt/cjk_vocab_pack_cold_r256",
        "post_image_dataset/cjk_unmask/te/sincos_cold_r256",
        False,
    ),
    "INIT": (
        "cjk_unmask_init",
        "bench/cjk_adapter/assets/ext_embed",
        "post_image_dataset/cjk_unmask/te/sincos_init_v2",
        False,
    ),
    # DiT line D1 gate: C9's trained rows + a seed-0 isotropic mirror block
    # for quoted spans (make_random_pack.py --mode iso-partition).
    "C9ISOQ": (
        "cjk_unmask_c9_isoq",
        "output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256_isoq",
        "post_image_dataset/cjk_unmask/te/sincos_jakozh1sym_r256_isoq",
        False,
    ),
}

# blind sets to compose once the grids exist: set name -> arms
SETS = [
    ("s02_C9_vs_C9s2", ["C9", "C9s2"]),  # grader noise floor: same recipe, two seeds
    ("s03_C9_vs_R", ["C9", "R"]),
    ("s04_P_vs_R", ["P", "R"]),
    ("s05_C9_vs_HOT", ["C9", "HOT"]),
    ("s06_C9_vs_COLLIDE", ["C9", "COLLIDE"]),
    ("s07_C9_vs_ROTATE", ["C9", "ROTATE"]),
    ("s08_C9_vs_INIT", ["C9", "INIT"]),
]


def run(stage: str, argv: list[str]) -> None:
    print(f"\n=== [{stage}] {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arms", nargs="*", default=list(ARMS), help="subset / order of arms"
    )
    ap.add_argument("--skip_sets", action="store_true")
    ap.add_argument("--no_push", action="store_true")
    opts = ap.parse_args()

    done, failed = [], []
    for arm in opts.arms:
        method, ext, te, skip_cache = ARMS[arm]
        argv = [
            PY,
            str(R2),
            "--ext_prefix",
            ext,
            "--method",
            method,
            "--te_out",
            te,
            "--mirror",
            MIRROR,
            "--arm",
            f"arm{arm}",
            "--ocr_format",
            "tags",
            "--records",
            RECORDS,
        ]
        if skip_cache:
            argv.append("--skip_cache")
        try:
            run(f"arm {arm}", argv)
            done.append(arm)
        except Exception:
            traceback.print_exc()
            failed.append(arm)
            print(f"!!! arm {arm} failed — continuing", flush=True)

    print(f"\n=== arms done: {done}  failed: {failed}", flush=True)
    if opts.skip_sets:
        return
    grids = {a for a in ARMS} | {"C9", "P", "R"}
    have = {
        a
        for a in grids
        if (REPO / "output/tests/cjk_unmask_eval2" / f"arm{a}_s42").exists()
    }
    for name, arms in SETS:
        if not all(a in have for a in arms):
            print(
                f"--- set {name}: skipped (missing {[a for a in arms if a not in have]})",
                flush=True,
            )
            continue
        argv = [PY, str(BP), "make", "--set", name, "--arms", *arms, "--overwrite"]
        if not opts.no_push:
            argv.append("--push")
        try:
            run(f"blind set {name}", argv)
        except Exception:
            traceback.print_exc()
            print(
                f"!!! set {name} failed — next session: rerun `blind_pairs.py make --set {name} --arms {' '.join(arms)} --overwrite --push`",
                flush=True,
            )
    print("\n=== chain done", flush=True)


if __name__ == "__main__":
    main()
