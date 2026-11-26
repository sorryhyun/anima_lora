#!/usr/bin/env python
"""Orient read — which way a run's string pages are lettered (forward only, pages on disk).

Reads the read pages of an ``s1.py`` run (``output/cjk_rl/g1/<run>/records.json`` and the
files it names; classes ``val`` and ``seen``) again with the detector and the tuned
reader, picks a page's box as training does (``gd.pick``) and keeps its width and height.
A picked box is ``tall`` when its height is over 1.15 times its width, ``wide`` the other
way round, else ``square``.

Per arm and string length (2, 3 – 4, 5 – 8, 9 – 12 glyphs): pages, tall / wide / square /
no box, the picked reads that hold a line break, and the share of the string's positions
the read hits (``s1.score``) on the tall and on the wide pages.

Result → ``results/<ts>-orient_read-<run>/``.

    make daemon-run ARGS="--queue --stall-timeout 600 project/cjk_rl/probes/orient_read.py --run <run>"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import turbo_read as T

import gd  # noqa: E402  (src/, on the path through turbo_read)
import s1  # noqa: E402

LINE = Path(__file__).resolve().parents[1]
EDGE = 1.15


def bin_of(n: int) -> str:
    return "2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9-12"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="an s1 run under output/cjk_rl/g1/")
    ap.add_argument("--conf_min", type=float, default=0.6)
    a = ap.parse_args()
    import numpy as np
    from PIL import Image

    recs = [
        r
        for r in json.loads(
            (T.REPO / "output" / "cjk_rl" / "g1" / a.run / "records.json").read_text(
                "utf-8"
            )
        )
        if r["cls"] in ("val", "seen")
    ]
    rdr = gd.Reader("cuda")
    rows = []
    for i in range(0, len(recs), 16):
        chunk = recs[i : i + 16]
        bgrs = [
            np.ascontiguousarray(
                np.asarray(Image.open(r["file"]).convert("RGB"))[:, :, ::-1]
            )
            for r in chunk
        ]
        for r, reads in zip(chunk, rdr.rd.read_images(bgrs)):
            case, box = gd.pick(r["key"], reads, a.conf_min)
            au = s1.units(r["key"])
            row = {
                "arm": r["arm"],
                "cls": r["cls"],
                "key": r["key"],
                "scene": r["scene"],
                "seed": r["seed"],
                "bin": bin_of(len(au)),
                "case": case,
                "way": "none",
                "asked": len(au),
                "hit": 0,
            }
            if box is not None:
                x0, y0, x1, y1 = box["box"]
                w, h = x1 - x0, y1 - y0
                got = box["sfx"] or ""
                row |= {
                    "w": w,
                    "h": h,
                    "way": "tall"
                    if h > EDGE * w
                    else "wide"
                    if w > EDGE * h
                    else "square",
                    "read": got,
                    "breaks": got.count("\n"),
                    "hit": sum(s1.score(au, got)[0]),
                }
            rows.append(row)
    out = {}
    print(
        "| arm | glyphs | pages | tall | wide | square | no box | reads with a break | positions hit, tall | positions hit, wide |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        for b in ("2", "3-4", "5-8", "9-12", "all"):
            xs = [r for r in rows if r["arm"] == arm and b in ("all", r["bin"])]
            if not xs:
                continue

            def share(way: str) -> float | None:
                ys = [r for r in xs if r["way"] == way]
                n = sum(r["asked"] for r in ys)
                return round(sum(r["hit"] for r in ys) / n, 3) if n else None

            cell = {
                w: sum(r["way"] == w for r in xs)
                for w in ("tall", "wide", "square", "none")
            }
            cell |= {
                "pages": len(xs),
                "breaks": sum(bool(r.get("breaks")) for r in xs),
                "hit_tall": share("tall"),
                "hit_wide": share("wide"),
            }
            out[f"{arm}/{b}"] = cell
            print(
                f"| {arm} | {b} | {cell['pages']} | {cell['tall']} | {cell['wide']} | "
                f"{cell['square']} | {cell['none']} | {cell['breaks']} | "
                f"{cell['hit_tall']} | {cell['hit_wide']} |",
                flush=True,
            )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_rl", label=f"orient_read-{a.run}", root=LINE / "results"
    )
    (run_dir / "pages.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), "utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/orient_read.py",
        args=vars(a) | {"edge": EDGE},
        label=a.run,
        metrics={"by_arm_and_length": out},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
