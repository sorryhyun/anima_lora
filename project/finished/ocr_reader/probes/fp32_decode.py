#!/usr/bin/env python3
"""Does fp32 compute change v4's reads? The upper bound of an fp32 stream.

Reads K3 (KO / ZH, 96 rows) and a COO val subset with the merged v4 reader in
bf16, then the same weights cast to fp32 (every activation fp32), and counts
the reads that change and the exact-match delta per set.

    make daemon-run ARGS="--stall-timeout 600 project/finished/ocr_reader/probes/fp32_decode.py"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import crop_dataset as cd  # noqa: E402
import eval_k3 as k3  # noqa: E402
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/ocr/vl16_p2b_kozh_ep2/ep2")
    ap.add_argument("--val_per_kind", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()

    k3_df = k3.load_labels()
    sets = {"k3": (k3_df, k3.load_crops(k3_df), k3.score)}
    va = cd.load_split("val", limit=a.val_per_kind, seed=0)
    va_crops = [cv2.imread(str(m109.derived_root() / p)) for p in va.path]
    sets["coo_val"] = (va, va_crops, ev.score)

    reader = ev.Vl16Reader(str(m109.REPO / a.ckpt), "cuda")
    out = {}
    for dtype in ("bf16", "fp32"):
        if dtype == "fp32":
            reader.model = reader.model.float()
        for name, (df, crops, score) in sets.items():
            t0 = time.time()
            preds = reader.read(crops, list(df.orient), a.bs)
            out[(dtype, name)] = (score(df, preds), time.time() - t0)

    res = {}
    for name in sets:
        b, tb = out[("bf16", name)]
        f, tf = out[("fp32", name)]
        diff = b.pred != f.pred
        grp = "lang" if "lang" in b.columns else "kind"
        per = {
            str(k): dict(
                n=int(len(g)),
                changed=int(diff[g.index].sum()),
                exact_bf16=int(b.exact[g.index].sum()),
                exact_fp32=int(f.exact[g.index].sum()),
                fp32_wins=int((f.exact & ~b.exact)[g.index].sum()),
                bf16_wins=int((b.exact & ~f.exact)[g.index].sum()),
            )
            for k, g in b.groupby(grp)
        }
        res[name] = dict(per=per, wall_bf16=round(tb, 1), wall_fp32=round(tf, 1))
        print(f"== {name}  wall bf16 {tb:.0f}s fp32 {tf:.0f}s")
        for k, v in per.items():
            print(f"   {k}: {v}")
        for i in diff[diff].index[:8]:
            print(f"   bf16 {b.pred[i]!r}  fp32 {f.pred[i]!r}  label {b.text[i]!r}")
    dst = m109.REPO / "output/ocr/eval/fp32_decode_v4.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
