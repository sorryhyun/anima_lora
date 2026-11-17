#!/usr/bin/env python3
"""swaprelax_v5 rows whose group was never swapped: the staged target keeps the Japanese
lettering there (the target is the JA page with the clean KO groups pasted in), while the
row carries the KO edition's label and box. Found by pixels — inside the row's box the
target equals the JA source page (share of pixels changed by > 40 below ``--chg``, 0.05;
1,1xx rows sit at ≥ 0.05 and every one checked is Korean).

Each such row becomes a Japanese row: the JA page's AnimeText box (``det.json`` ``src``)
that overlaps the KO box most, cut from the JA source page, labelled by v6's read.

    make daemon-run ARGS="--stall-timeout 600 \\
        project/finished/ocr_reader/probes/unswapped_ja.py --read --out <ja.parquet>"
    .venv/bin/python project/finished/ocr_reader/probes/unswapped_ja.py --apply --out <ja.parquet>

``--apply`` writes ``swap_relax_v5/unswapped_ja.json`` (``build.py`` applies it on a
rebuild) and patches ``labels.jsonl`` (page → the JA source, box, crop_box, label, lang),
``crops/{id}.png``, ``manifest_pseudo_swaprelax_v5``, ``manifest_sfxp_kozh`` and the review
page's v6 flags. A row with no JA box over it is dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402
from apply_nested_fix import FLAGS, REPO, SR, backup  # noqa: E402
from build_sfx_prefix import margin_box  # noqa: E402

DET = REPO / "post_image_dataset/i2i/swap_text_relax/det.json"


def src_of(page: str) -> str:
    return page.replace(".tgt.", ".src.")


def find_rows(chg_max: float) -> list[dict]:
    pm = set(
        pd.read_parquet(m109.derived_root() / "manifest_pseudo_swaprelax_v5.parquet").id
    )
    labels = [json.loads(x) for x in open(SR / "labels.jsonl", encoding="utf-8")]
    rows = sorted(
        (
            r
            for r in labels
            if f"{r['stem']}_{r['line']}" in pm and r.get("lang") != "ja"
        ),
        key=lambda r: r["page"],
    )
    det = json.loads(DET.read_text())
    out, cur, t, s = [], None, None, None
    for r in rows:
        if r["page"] != cur:
            cur = r["page"]
            t = cv2.imread(str(REPO / cur))
            s = cv2.imread(str(REPO / src_of(cur)))
        if t is None or s is None:
            continue
        x0, y0, x1, y1 = r["box"]
        d = np.abs(t[y0:y1, x0:x1].astype(np.int16) - s[y0:y1, x0:x1]).mean(axis=2)
        chg = float((d > 40).mean())
        if chg >= chg_max:
            continue
        work, page, lang = r["stem"].split("_")
        srcb = det.get(f"{lang}/{work}/{page}", {}).get("src") or []

        def inter(b):
            w = min(b[2], x1) - max(b[0], x0)
            h = min(b[3], y1) - max(b[1], y0)
            return max(w, 0) * max(h, 0)

        best = max(srcb, key=inter, default=None)
        ov = inter(best) / ((best[2] - best[0]) * (best[3] - best[1])) if best else 0.0
        out.append(
            dict(
                id=r["id"],
                mid=f"{r['stem']}_{r['line']}",
                page=src_of(r["page"]),
                ko_box=r["box"],
                ko_label=r["label"],
                chg=chg,
                ja_box=[int(v) for v in best] if best else None,
                ov=ov,
            )
        )
    return out


def read(a) -> None:
    rows = find_rows(a.chg)
    crops, keep = [], []
    img, cur = None, None
    for r in rows:
        if r["ja_box"] is None or r["ov"] < a.min_ov:
            continue
        if cur != r["page"]:
            img, cur = cv2.imread(str(REPO / r["page"])), r["page"]
        H, W = img.shape[:2]
        c = margin_box(*r["ja_box"], W, H)
        crops.append(img[c[1] : c[3], c[0] : c[2]])
        keep.append(r)
    print(
        f"[ja] {len(rows)} unswapped rows, {len(keep)} with a JA box over them",
        flush=True,
    )
    from anime_tools.ocr.reader import OcrReader

    reader = OcrReader.load(device=a.device, batch_size=32)
    res = reader.read_scored(crops)
    for r, x in zip(keep, res):
        r["read"], r["conf"] = (x[0], float(x[1])) if x else ("", float("nan"))
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[ja] → {a.out}", flush=True)


def apply(a) -> None:
    d = m109.derived_root()
    df = pd.read_parquet(a.out)
    # one row per JA box (two KO rows can sit over the same JA block: the larger overlap
    # keeps it), and only reads v6 is sure of
    key = df.ja_box.map(lambda b: tuple(b) if b is not None else None)
    first = (
        df.assign(k=key).sort_values("ov", ascending=False).drop_duplicates("k").index
    )
    ok = (
        df.read.notna()
        & (df.read.fillna("").str.strip() != "")
        & (df.conf >= a.min_conf)
        & df.index.isin(first)
    )
    fix = {
        r.id: (
            {
                "page": r.page,
                "box": list(map(int, r.ja_box)),
                "label": r.read,
                "lang": "ja",
            }
            if good
            else {
                "drop": "unswapped group: the target keeps the JA lettering; no JA box "
                "over it, a duplicate JA box, or a v6 read below --min_conf"
            }
        )
        for r, good in zip(df.itertuples(), ok)
    }
    (SR / "unswapped_ja.json").write_text(
        json.dumps(fix, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    labels = [json.loads(x) for x in open(SR / "labels.jsonl", encoding="utf-8")]
    backup(SR / "labels.jsonl")
    new, drop, img, cur = {}, set(), None, None
    for r in sorted(
        (r for r in labels if r["id"] in fix),
        key=lambda r: fix[r["id"]].get("page", ""),
    ):
        f, mid = fix[r["id"]], f"{r['stem']}_{r['line']}"
        if "drop" in f:
            r.update(status="drop", label="", drop_reason=f["drop"])
            drop.add(mid)
            continue
        if cur != f["page"]:
            img, cur = cv2.imread(str(REPO / f["page"])), f["page"]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = f["box"]
        g = round(0.12 * max(x1 - x0, y1 - y0))
        cb = [max(x0 - g, 0), max(y0 - g, 0), min(x1 + g, W), min(y1 + g, H)]
        cv2.imwrite(str(SR / r["crop"]), img[cb[1] : cb[3], cb[0] : cb[2]])
        r.update(page=f["page"], box=f["box"], crop_box=cb, label=f["label"], lang="ja")
        c = margin_box(x0, y0, x1, y1, W, H)
        new[mid] = dict(
            cb=cb, c=c, img=img[c[1] : c[3], c[0] : c[2]].copy(), text=f["label"]
        )
    with open(SR / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in labels:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    pm_path, kz_path = (
        d / "manifest_pseudo_swaprelax_v5.parquet",
        d / "manifest_sfxp_kozh.parquet",
    )
    backup(pm_path)
    pm = pd.read_parquet(pm_path)
    pm = pm[~pm.id.isin(drop)].reset_index(drop=True)
    orient = {}
    for i in pm.index[pm.id.isin(new.keys())]:
        n = new[pm.at[i, "id"]]
        w, h = n["cb"][2] - n["cb"][0], n["cb"][3] - n["cb"][1]
        o = "vertical" if h > w else "horizontal"
        pm.loc[i, ["w", "h", "orient", "text"]] = [w, h, o, n["text"]]
        pm.at[i, "poly"] = json.dumps([0.0, 0.0, w, 0.0, w, h, 0.0, h])
        orient[pm.at[i, "id"]] = o
    pm.to_parquet(pm_path, index=False)
    backup(kz_path)
    kz = pd.read_parquet(kz_path)
    sr = kz.source == "swaprelax_v5"
    kz = kz[~(sr & kz.id.isin(drop))].reset_index(drop=True)
    for i in kz.index[(kz.source == "swaprelax_v5") & kz.id.isin(new.keys())]:
        mid = kz.at[i, "id"]
        n = new[mid]
        cv2.imwrite(str(d / kz.at[i, "path"]), n["img"])
        kz.loc[i, ["w", "h", "orient", "text"]] = [
            n["c"][2] - n["c"][0],
            n["c"][3] - n["c"][1],
            orient[mid],
            n["text"],
        ]
    kz.to_parquet(kz_path, index=False)
    if FLAGS.exists():
        backup(FLAGS)
        fl = pd.read_parquet(FLAGS)
        fl = fl[~((fl.source == "swaprelax_v5") & fl.id.isin(drop))].reset_index(
            drop=True
        )
        conf = dict(zip(df.mid, df.conf))
        for i in fl.index[(fl.source == "swaprelax_v5") & fl.id.isin(new.keys())]:
            mid = fl.at[i, "id"]
            fl.at[i, "read"], fl.at[i, "conf"], fl.at[i, "edge"] = (
                new[mid]["text"],
                conf[mid],
                None,
            )
        fl.to_parquet(FLAGS, index=False)
    print(
        f"[ja] relabelled {len(new)}, dropped {len(drop)}; pseudo {len(pm)}, kozh {len(kz)}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--chg", type=float, default=0.05)
    ap.add_argument(
        "--min_ov",
        type=float,
        default=0.3,
        help="share of the JA box inside the KO box",
    )
    ap.add_argument("--min_conf", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.read:
        read(a)
    if a.apply:
        apply(a)


if __name__ == "__main__":
    main()
