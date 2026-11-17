#!/usr/bin/env python3
"""Apply the nested-box decision to swaprelax_v5 (input: ``nested_inner_read.py``'s parquet).

A row whose box holds smaller AnimeText boxes is decided by how much of its label the
inner boxes' v6 reads cover (character multiset, whitespace-blind):

* coverage ≥ ``--thresh`` (0.85): the inner box holds the whole text (the outer box was
  loose) → the row takes the inner box (the union when the detector split the block),
  label unchanged;
* below: the inner box holds part of it (one line of a block, or one of several blocks the
  outer box merged) → the row is dropped.

Coverage is taken against the manifest's text (review edits included) where the row is in
``manifest_pseudo_swaprelax_v5``, else against ``labels.jsonl``. Writes
``swap_relax_v5/nested_fix.json`` (``build.py`` applies it on a rebuild), then patches what
it would have produced: ``labels.jsonl`` (box, crop_box, status), ``crops/{id}.png`` (12 %),
``manifest_pseudo_swaprelax_v5`` and ``manifest_sfxp_kozh`` (crop re-cut at the 4 px
margin, w / h / orient; dropped rows removed) and the review page's v6 flags (the inner
read). Each file is copied to ``*.pre_nested.*`` once before its first change.

    .venv/bin/python project/finished/ocr_reader/probes/apply_nested_fix.py \\
        --inner output/ocr/eval/swaprelax_nested_inner.parquet [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402
from build_sfx_prefix import margin_box  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
SR = Path("/media/sorryhyun/new/dataset/hito_ocr/swap_relax_v5")
FLAGS = REPO / "output/ocr/eval/ko_v6_reads_flags.parquet"


def norm(t: str) -> str:
    return re.sub(r"\s+", "", t.replace("...", "…"))


def coverage(text: str, reads: list[str]) -> float:
    L, R = Counter(norm(text)), Counter(norm("".join(reads)))
    n = sum(L.values())
    return sum(min(v, R[c]) for c, v in L.items()) / n if n else 0.0


def backup(path: Path) -> None:
    bak = path.with_name(path.stem + ".pre_nested" + path.suffix)
    if not bak.exists():
        shutil.copy2(path, bak)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--inner", type=Path, required=True)
    ap.add_argument("--thresh", type=float, default=0.85)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    d = m109.derived_root()
    pm_path = d / "manifest_pseudo_swaprelax_v5.parquet"
    kz_path = d / "manifest_sfxp_kozh.parquet"
    pm, kz = pd.read_parquet(pm_path), pd.read_parquet(kz_path)
    ptext = dict(zip(pm.id, pm.text))
    labels = [json.loads(x) for x in open(SR / "labels.jsonl", encoding="utf-8")]
    by_id = {r["id"]: r for r in labels}

    inner = pd.read_parquet(a.inner)
    fix, reads_of, tally = {}, {}, Counter()
    for lid, g in inner.groupby("id"):
        r = by_id[lid]
        if r["status"] in ("drop", "excluded"):
            continue
        mid = f"{r['stem']}_{r['line']}"
        text = ptext.get(mid, r["label"])
        cov = coverage(text, list(g.read))
        if cov >= a.thresh:
            bs = list(g.inner)
            box = [
                int(min(b[0] for b in bs)),
                int(min(b[1] for b in bs)),
                int(max(b[2] for b in bs)),
                int(max(b[3] for b in bs)),
            ]
            fix[lid] = {"box": box, "outer": r["box"], "cov": round(cov, 3)}
            g = g.assign(y=[b[1] for b in g.inner], x=[b[0] for b in g.inner])
            g = g.sort_values(["y", "x"])
            reads_of[mid] = ("\n".join(g.read), float(g.conf.min()))
            tally["box"] += 1
        else:
            fix[lid] = {
                "drop": f"nested: inner AnimeText boxes cover {cov:.0%} of the label",
                "outer": r["box"],
                "cov": round(cov, 3),
            }
            tally["drop"] += 1
    in_pm = Counter(
        k
        for lid, f in fix.items()
        if f"{by_id[lid]['stem']}_{by_id[lid]['line']}" in ptext
        for k in ("box" if "box" in f else "drop",)
    )
    print(f"[nested] all rows {dict(tally)}; in the manifest {dict(in_pm)}", flush=True)
    if a.dry_run:
        return

    (SR / "nested_fix.json").write_text(
        json.dumps(fix, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    # labels.jsonl + crops/{id}.png, as build.py would write them
    backup(SR / "labels.jsonl")
    img, cur, size = None, None, {}
    for r in sorted((by_id[i] for i in fix), key=lambda r: r["page"]):
        f = fix[r["id"]]
        if "drop" in f:
            r.update(status="drop", label="", drop_reason=f["drop"])
            continue
        if cur != r["page"]:
            img, cur = cv2.imread(str(REPO / r["page"])), r["page"]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = f["box"]
        g = round(0.12 * max(x1 - x0, y1 - y0))
        cb = [max(x0 - g, 0), max(y0 - g, 0), min(x1 + g, W), min(y1 + g, H)]
        cv2.imwrite(str(SR / r["crop"]), img[cb[1] : cb[3], cb[0] : cb[2]])
        r.update(box=f["box"], crop_box=cb)
        size[f"{r['stem']}_{r['line']}"] = (cb[2] - cb[0], cb[3] - cb[1], r, img)
    with open(SR / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in labels:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    drop_mid = {
        f"{by_id[i]['stem']}_{by_id[i]['line']}" for i, f in fix.items() if "drop" in f
    }
    # manifest_pseudo_swaprelax_v5: the 12 % crop's size and orient
    backup(pm_path)
    pm = pm[~pm.id.isin(drop_mid)].reset_index(drop=True)
    orient = {}
    for i, mid in enumerate(pm.id):
        if mid in size:
            w, h = size[mid][:2]
            o = "vertical" if h > w else "horizontal"
            pm.at[i, "w"], pm.at[i, "h"], pm.at[i, "orient"] = w, h, o
            pm.at[i, "poly"] = json.dumps([0.0, 0.0, w, 0.0, w, h, 0.0, h])
            orient[mid] = o
    pm.to_parquet(pm_path, index=False)
    # manifest_sfxp_kozh: the 4 px crop, re-cut from the page
    backup(kz_path)
    sr = kz.source == "swaprelax_v5"
    kz = kz[~(sr & kz.id.isin(drop_mid))].reset_index(drop=True)
    for i in kz.index[(kz.source == "swaprelax_v5") & kz.id.isin(orient.keys())]:
        mid = kz.at[i, "id"]
        _, _, r, page_img = size[mid]
        H, W = page_img.shape[:2]
        c = margin_box(*r["box"], W, H)
        cv2.imwrite(str(d / kz.at[i, "path"]), page_img[c[1] : c[3], c[0] : c[2]])
        kz.at[i, "w"], kz.at[i, "h"] = c[2] - c[0], c[3] - c[1]
        kz.at[i, "orient"] = orient[mid]
    kz.to_parquet(kz_path, index=False)
    # the review page's v6 read: the inner read replaces the outer crop's
    if FLAGS.exists():
        backup(FLAGS)
        fl = pd.read_parquet(FLAGS)
        srf = fl.source == "swaprelax_v5"
        fl = fl[~(srf & fl.id.isin(drop_mid))].reset_index(drop=True)
        for i in fl.index[(fl.source == "swaprelax_v5") & fl.id.isin(reads_of.keys())]:
            rd, cf = reads_of[fl.at[i, "id"]]
            fl.at[i, "read"], fl.at[i, "conf"], fl.at[i, "edge"] = rd, cf, None
        fl.to_parquet(FLAGS, index=False)
    print(
        f"[nested] pseudo {len(pm)} rows, sfxp_kozh {len(kz)} rows "
        f"({int((kz.source == 'swaprelax_v5').sum())} swaprelax)",
        flush=True,
    )


if __name__ == "__main__":
    main()
