#!/usr/bin/env python3
"""Re-score every reader row of the sincos gate on the current labels (one daemon job).

    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima_dit/ocr/reeval_sfx_all.py"

Rows = the ``reports/ocr_eval_sfx_<name>.md`` set that still has a runnable
reader: stock manga-ocr / VL-1.6 / the Hub SFX reader, and the fine-tunes under
``output/ocr/<run>/best``. The PP-era ``record_hybrid*`` rows are gone (their
boxes no longer match the AnimeText labels); ``record`` = the pipeline's own
AnimeText read, scored on CPU by ``eval_sfx.py --reader record``.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "project/cjk_aware_anima_dit/ocr/eval_sfx.py"
ROWS = [
    ("manga_ocr_stock", ["--reader", "manga_ocr"]),
    ("mocr_lr2e-5", ["--reader", "manga_ocr", "--ckpt", "output/ocr/mocr_lr2e-5/best"]),
    ("mocr_lr5e-5", ["--reader", "manga_ocr", "--ckpt", "output/ocr/mocr_lr5e-5/best"]),
    ("vl16_stock", ["--reader", "vl16"]),
    ("vl16_lr1e-4", ["--reader", "vl16", "--ckpt", "output/ocr/vl16_lr1e-4/best"]),
    (
        "vl16_tower_lr1e-5",
        ["--reader", "vl16", "--ckpt", "output/ocr/vl16_tower_lr1e-5/best"],
    ),
    (
        "vl16_tower_col100",
        ["--reader", "vl16", "--ckpt", "output/ocr/vl16_tower_col100/best"],
    ),
    ("sfx_pkg", ["--reader", "sfx"]),
]

if __name__ == "__main__":
    only = set(sys.argv[1:])
    failed = []
    for name, args in ROWS:
        if only and name not in only:
            continue
        print(f"\n===== {name} =====", flush=True)
        rc = subprocess.call(
            [sys.executable, str(EVAL), *args, "--name", name], cwd=REPO
        )
        if rc:
            failed.append(name)
    print("\nfailed:", failed or "none", flush=True)
    sys.exit(1 if failed else 0)
