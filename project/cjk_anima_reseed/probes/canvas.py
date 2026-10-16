#!/usr/bin/env python
"""canvas — what a hiragana row's training occurrences sit on (CPU; proposal_template.md
L3: a template is what stays constant across a glyph's occurrences other than the
glyph).

``meta`` (every data dir, from the records; retrain_kana's images are gone): per
hiragana glyph occurrence its canvas (``scene`` = lettered into a scene's bubble,
``grid`` = a grid cell, ``flat`` = one glyph on a flat page), the upper edge of its
σ band and its glyph px; the glyph-weighted shares by canvas × upper edge, the
share at σ ≥ 0.8 on a flat / grid canvas, and per row the distinct scenes over
its scene occurrences.

``pixels`` (data dirs whose images exist): per glyph occurrence, from the item's image: the crop is its box (a grid cell's
own box; a window's one box for every glyph in it) padded by a quarter of its
size each side. ``bg`` is the crop's median colour; ``ink`` the mean of the 10 %
of crop pixels farthest from ``bg``. ``plain`` = bg luminance ≥ 0.85, bg
saturation ≤ 0.12, ink luminance ≤ 0.35, ink saturation ≤ 0.25: dark neutral type
on a pale neutral ground, retrain_kana's habit. Per row: the plain share of its
occurrences and the entropy (bits) of its occurrences over a 4 × 3 × 3 grid of
(bg lum, bg sat, ink lum).

    .venv/bin/python project/cjk_anima_reseed/probes/canvas.py --sample 3000
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, OUT, SCALE_OUT  # noqa: E402

DATA = {
    "retrain_kana": SCALE_OUT / "retrain_kana" / "data",
    "hira_cold_hp": OUT / "hira_cold_hp" / "data",
    "hira_cold_mix": OUT / "hira_cold_mix" / "data",
}
HIRA = set(
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをんゔ"
)


def _lum_sat(rgb):
    import numpy as np

    rgb = np.asarray(rgb, np.float32) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return lum, sat


def occurrences(rec: dict) -> list:
    """``[(glyphs, box)]``: a grid cell's glyph with its own box, else the
    item's glyphs with its one box."""
    units = [u.replace(" ", "") for u in rec["units"]]
    boxes = rec.get("boxes") or ([rec["box"]] if rec.get("box") else [])
    if len(boxes) == len(units) and len(boxes) > 1:
        return list(zip(units, boxes))
    if not boxes:
        return []
    return [("".join(units), boxes[0])]


def measure(job):
    import numpy as np
    from PIL import Image

    rec = job
    try:
        im = np.asarray(Image.open(rec["file"]).convert("RGB"))
    except Exception:
        return []
    H, W, _ = im.shape
    out = []
    for glyphs, (x0, y0, x1, y1) in occurrences(rec):
        px, py = max(2, (x1 - x0) // 4), max(2, (y1 - y0) // 4)
        c = im[max(0, y0 - py) : min(H, y1 + py), max(0, x0 - px) : min(W, x1 + px)]
        flat = c.reshape(-1, 3).astype(np.float32)
        if len(flat) < 16:
            continue
        bg = np.median(flat, 0)
        d = np.linalg.norm(flat - bg, axis=1)
        ink = flat[d >= np.quantile(d, 0.9)].mean(0)
        bl, bs = _lum_sat(bg)
        il, is_ = _lum_sat(ink)
        out.append(
            {
                "glyphs": glyphs,
                "tier": rec.get("tier") or rec.get("kind"),
                "bl": float(bl),
                "bs": float(bs),
                "il": float(il),
                "is": float(is_),
            }
        )
    return out


def plain(o) -> bool:
    return o["bl"] >= 0.85 and o["bs"] <= 0.12 and o["il"] <= 0.35 and o["is"] <= 0.25


def cell(o) -> tuple:
    return (
        min(3, int(o["bl"] * 4)),
        min(2, int(o["bs"] * 3)),
        min(2, int(o["il"] * 3)),
    )


def entropy(cs: Counter) -> float:
    n = sum(cs.values())
    return -sum(v / n * math.log2(v / n) for v in cs.values()) if n else 0.0


def meta(name: str) -> dict:
    recs = [
        json.loads(ln)
        for ln in (DATA[name] / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    share: Counter = Counter()
    px_hi: list = []
    scenes = defaultdict(list)
    for r in recs:
        canvas = "scene" if r["layout"] == "scene" else r["layout"]
        hi = float(r["band"][1])
        hb = (
            "≤0.6"
            if hi <= 0.6
            else ("0.7" if hi <= 0.75 else ("0.8" if hi <= 0.85 else "≥0.9"))
        )
        for g in "".join(u.replace(" ", "") for u in r["units"]):
            if g not in HIRA:
                continue
            share[canvas, hb] += 1
            if canvas != "scene" and hi >= 0.8:
                px_hi.append(float(r["px"]))
            if canvas == "scene":
                scenes[g].append(r.get("scene"))
    n = sum(share.values())
    reuse = sorted(len(set(v)) / len(v) for v in scenes.values())
    return {
        "items": len(recs),
        "glyph_occ": n,
        "share": {f"{c} {h}": v / n for (c, h), v in sorted(share.items())},
        "plain_hi_share": len(px_hi) / n,
        "plain_hi_px_median": sorted(px_hi)[len(px_hi) // 2] if px_hi else 0.0,
        "scene_distinct_per_occ_median": reuse[len(reuse) // 2] if reuse else 0.0,
    }


def read(name: str, sample: int, workers: int) -> dict:
    recs = [
        json.loads(ln)
        for ln in (DATA[name] / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    rng = random.Random(0)
    if sample and len(recs) > sample:
        recs = rng.sample(recs, sample)
    with mp.Pool(workers) as p:
        occ = [o for os_ in p.imap_unordered(measure, recs, chunksize=32) for o in os_]
    per_row = defaultdict(list)
    for o in occ:
        for g in o["glyphs"]:
            if g in HIRA:
                per_row[g].append(o)
    rows = {
        g: {
            "n": len(os_),
            "plain": sum(map(plain, os_)) / len(os_),
            "H": entropy(Counter(map(cell, os_))),
        }
        for g, os_ in per_row.items()
    }
    by_tier = defaultdict(list)
    for o in occ:
        by_tier[o["tier"]].append(o)
    occ_h = [o for o in occ if set(o["glyphs"]) & HIRA]
    w = [sum(g in HIRA for g in o["glyphs"]) for o in occ_h]  # glyph-weighted
    return {
        "items": len(recs),
        "plain_glyph_share": sum(wi * plain(o) for wi, o in zip(w, occ_h)) / sum(w),
        "H_all": entropy(
            Counter(cell(o) for wi, o in zip(w, occ_h) for _ in range(wi))
        ),
        "row_plain_median": sorted(r["plain"] for r in rows.values())[len(rows) // 2],
        "row_H_median": sorted(r["H"] for r in rows.values())[len(rows) // 2],
        "by_tier": {
            t: {"n": len(v), "plain": sum(map(plain, v)) / len(v)}
            for t, v in sorted(by_tier.items())
        },
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sample", type=int, default=3000, help="items a data dir (0 = all)"
    )
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    res = {"meta": {n: meta(n) for n in DATA}}
    for n, m in res["meta"].items():
        print(
            f"{n}: {m['glyph_occ']} hiragana glyph occurrences in {m['items']} items; "
            f"flat / grid at σ ≥ 0.8 {m['plain_hi_share']:.3f} (px median "
            f"{m['plain_hi_px_median']:.0f}); distinct scenes / scene occurrence "
            f"(row median) {m['scene_distinct_per_occ_median']:.3f}",
            flush=True,
        )
        print("    " + ", ".join(f"{k} {v:.3f}" for k, v in m["share"].items()))
    have = [n for n in DATA if (DATA[n] / "img").is_dir()]
    res["pixels"] = {n: read(n, a.sample, a.workers) for n in have}
    for n, r in res["pixels"].items():
        print(
            f"{n}: {r['items']} items, plain {r['plain_glyph_share']:.3f} of hiragana "
            f"glyph occurrences, H {r['H_all']:.2f} bits; per row median plain "
            f"{r['row_plain_median']:.3f}, H {r['row_H_median']:.2f}",
            flush=True,
        )
        for t, v in r["by_tier"].items():
            print(f"    {t:<20} n {v['n']:>6}  plain {v['plain']:.3f}")
    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_anima_reseed", label="canvas", root=HOME / "results")
    write_result(
        run_dir,
        script="project/cjk_anima_reseed/probes/canvas.py",
        args=vars(a),
        metrics=res,
    )
    print(f"canvas → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
