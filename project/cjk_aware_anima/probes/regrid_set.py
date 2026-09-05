#!/usr/bin/env python3
"""Re-render already-trained unmask arms at fresh seeds and (re)build a blind
set from them — for pairing two arms the grader has already seen at the
default seeds (42/7/1234), where reusing the grids would break the blind.

    .venv/bin/python project/cjk_aware_anima/probes/regrid_set.py \
        --set s09_HOT_vs_COLLIDE --arms HOT COLLIDE --seeds 1 2 3 --push

Arms are looked up in ``g0_chain.ARMS`` (method name = LoRA file stem);
``C9`` / ``P`` / ``R`` map to their trained method names below. One daemon
command job: ``make daemon-run ARGS="<this file> ... --queue"``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
PY = sys.executable

sys.path.insert(0, str(HERE))
from g0_chain import ARMS, MIRROR, RECORDS  # noqa: E402

METHOD = {a: v[0] for a, v in ARMS.items()} | {
    "C9": "cjk_unmask_c9",
    "P": "cjk_unmask_presence",
    "R": "cjk_unmask_random",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--eval_dir", default="output/tests/cjk_unmask_eval2")
    ap.add_argument(
        "--prompts", default=str(PROJ / "assets" / "unmask_eval_prompts.txt")
    )
    ap.add_argument(
        "--train_arms",
        nargs="*",
        default=[],
        help="arms to cache+train first (g0_chain.ARMS entries); the rest render only",
    )
    ap.add_argument("--skip_render", action="store_true")
    ap.add_argument("--push", action="store_true")
    o = ap.parse_args()
    for arm in o.arms:
        method = METHOD[arm]
        if (
            arm not in o.train_arms
            and not (REPO / f"output/ckpt/{method}.safetensors").exists()
        ):
            sys.exit(f"missing LoRA for arm {arm}: {method}")
    for arm in o.train_arms:
        method, ext, te, _ = ARMS[arm]
        cmd = [
            PY,
            str(PROJ / "run_unmask_r2.py"),
            "--ext_prefix",
            ext,
            "--method",
            method,
            "--te_out",
            te,
            "--mirror",
            MIRROR,
            "--records",
            RECORDS,
            "--ocr_format",
            "tags",
            "--arm",
            f"arm{arm}",
            "--seeds",
            *map(str, o.seeds),
            "--eval_dir",
            o.eval_dir,
            "--prompts",
            o.prompts,
        ]
        print("\n===", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    n_rows = len(
        [
            ln
            for ln in Path(o.prompts).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    )
    if not o.skip_render:
        for arm in o.arms:
            if arm in o.train_arms:
                continue
            if all(
                len(list((REPO / o.eval_dir / f"arm{arm}_s{s}").glob("*.png")))
                == n_rows
                for s in o.seeds
            ):
                print(
                    f"=== arm{arm} already rendered at seeds {o.seeds} in {o.eval_dir}, skip",
                    flush=True,
                )
                continue
            cmd = [
                PY,
                str(PROJ / "run_unmask_r2.py"),
                "--method",
                METHOD[arm],
                "--arm",
                f"arm{arm}",
                "--skip_cache",
                "--skip_train",
                "--seeds",
                *map(str, o.seeds),
                "--eval_dir",
                o.eval_dir,
                "--prompts",
                o.prompts,
            ]
            print("\n===", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=REPO, check=True)
    cmd = [
        PY,
        str(HERE / "blind_pairs.py"),
        "make",
        "--set",
        o.set,
        "--arms",
        *o.arms,
        "--seeds",
        *map(str, o.seeds),
        "--eval_dir",
        o.eval_dir,
        "--prompts",
        o.prompts,
        "--overwrite",
    ] + (["--push"] if o.push else [])
    print("\n===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
