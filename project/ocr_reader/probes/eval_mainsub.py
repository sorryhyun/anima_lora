#!/usr/bin/env python3
"""A reader on the main / sub val set (``crop_dataset.load_mainsub("val")``, the rows a
``finetune_vl16_lora.py --mainsub`` run scores at ``--val_limit``), each row under its
prompt (``OCR:`` main, ``OCR2:`` sub) or all under one (``--prompt``): exact / sim per
``metric``, for a reader trained before main / sub (``sfxpc_*``) beside a main / sub run.

    make daemon-run ARGS="--queue project/ocr_reader/probes/eval_mainsub.py \\
        --ckpt output/ocr/sfxpc_1121_0117/best --name sfxpc_1121_0117 --prompt OCR:"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import crop_dataset as cd  # noqa: E402
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--val_limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prompt", help="read every row under this prompt (default: its own)"
    )
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    df = cd.load_mainsub("val", limit=a.val_limit, seed=a.seed)
    d = m109.derived_root()
    rd = ev.Vl16Reader(a.ckpt, "cuda")
    preds = [""] * len(df)
    for task, g in df.groupby("task"):
        ev.TASK = a.prompt or task  # Vl16Reader.read's prompt
        crops = [cv2.imread(str(d / p)) for p in g.path]
        for i, p in zip(g.index, rd.read(crops, list(g.orient), a.bs)):
            preds[i] = p
    scored = ev.score(df, preds)
    m = {}
    for k, g in scored.groupby("metric"):
        m[k] = dict(n=len(g), exact=float(g.exact.mean()), sim=float(g.sim.mean()))
    out = m109.REPO / "output/ocr/eval" / f"mainsub_{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ckpt=a.ckpt, prompt=a.prompt, metrics=m), indent=1))
    scored.to_json(
        out.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False
    )
    for k, v in m.items():
        print(
            f"{k:10s} n {v['n']:4d} exact {100 * v['exact']:.1f} % sim {v['sim']:.3f}"
        )
    print(f"→ {out}")


if __name__ == "__main__":
    main()
