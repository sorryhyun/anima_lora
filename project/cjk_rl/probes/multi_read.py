#!/usr/bin/env python
"""Multi read — pages that ask for several strings in one clause, counted per string (CPU).

Reads the ``records.json`` of a ``row_line.py`` result (records of class ``multi``: the
key is the strings joined by ``", "``, with the page's box reads). Per page each string
takes a box of its own, the assignment with the highest summed sequence score
(``word_read.align`` hits over the longer of the string and the box). Counted:

- a page: every string read exactly by some box (``all``), the same with as many boxes as
  strings (``clean``), a box that holds two of the strings (``joined``), a string read
  exactly by two boxes or more (``twice``), no box (``none``), boxes a page;
- a string, by its place in the clause and by its length: read exactly by a box (``own``),
  and the share of its positions on the alignment with its assigned box.

``--alone`` names a ``row_line.py`` result that read strings one a caption (class ``word``
or ``en``), and ``--alone_arm`` its arm (the multi result's own arm when absent): for the
strings asked here, on the scenes and seeds read here, their ``own`` and positions hit
there are printed beside. Both counts are given with the
strings as written and with a small kana folded onto its full-size glyph. ``--sheets``
writes a labelled sheet a caption (its pages, each with its boxes' reads).

    .venv/bin/python project/cjk_rl/probes/multi_read.py project/cjk_rl/results/<ts>-row_line-<stem> \\
        --alone project/cjk_rl/results/<ts>-row_line-<stem> --sheets
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import word_read as W

SEP = '", "'  # g1.SEP


def score(t: str, b: str) -> float:
    return sum(W.align(t, b)) / max(len(t), len(b))


def assign(ts: list, bs: list) -> list:
    """Per string the index of its box (``None``: no box left)."""
    slots = list(range(len(bs))) + [None] * len(ts)
    best, pick = -1.0, [None] * len(ts)
    for p in set(permutations(slots, len(ts))):
        s = sum(score(t, bs[j]) for t, j in zip(ts, p) if j is not None)
        if s > best:
            best, pick = s, list(p)
    return pick


def page(ts: list, boxes: list, folded: bool) -> dict:
    f = W.fold if folded else (lambda s: s)
    ts, bs = [f(t) for t in ts], [f(b) for b in boxes]
    at = assign(ts, bs)
    own = [int(t in bs) for t in ts]
    return {
        "own": own,
        "hit": [
            W.align(t, bs[j]) if j is not None else [False] * len(t)
            for t, j in zip(ts, at)
        ],
        "all": int(all(own)),
        "clean": int(all(own) and len(bs) == len(ts)),
        "joined": int(any(sum(t in b for t in ts) >= 2 for b in bs)),
        "twice": int(any(bs.count(t) >= 2 for t in ts)),
        "none": int(not bs),
        "boxes": len(bs),
    }


def share(h: int, n: int) -> str:
    return f"{h} / {n} ({h / n:.2f})" if n else "–"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path, help="a row_line result directory")
    ap.add_argument(
        "--alone", type=Path, help="a row_line result with the strings alone"
    )
    ap.add_argument("--alone_arm", help="the arm of --alone's records to take")
    ap.add_argument("--sheets", action="store_true")
    a = ap.parse_args()
    recs = [
        r
        for r in json.loads((a.result / "records.json").read_text("utf-8"))
        if r["cls"] == "multi"
    ]
    arms = list(dict.fromkeys(r["arm"] for r in recs))
    alone = []
    if a.alone:
        asked = {t for r in recs for t in r["key"].split(SEP)}
        pages = {(r["scene"], r["seed"]) for r in recs}
        alone = [
            r
            for r in json.loads((a.alone / "records.json").read_text("utf-8"))
            if r["cls"] in ("word", "en")
            and r["key"] in asked
            and (r["scene"], r["seed"]) in pages
        ]
    out = {}
    for reader, field in (("tuned", "boxes"), ("hayai", "hy_boxes")):
        for folded in (False, True):
            tag = f"{reader}, {'folded' if folded else 'as written'}"
            print(f"\n## {tag}")
            print(
                "| arm | strings | pages | all | clean | joined | twice | no box | boxes a page |"
            )
            print("|---|---|---|---|---|---|---|---|---|")
            by = defaultdict(lambda: [0, 0, 0, 0])  # own, strings, hits, positions
            for arm in arms:
                for n in sorted({len(r["key"].split(SEP)) for r in recs}):
                    xs = [
                        r
                        for r in recs
                        if r["arm"] == arm and len(r["key"].split(SEP)) == n
                    ]
                    ps = [page(r["key"].split(SEP), r[field], folded) for r in xs]
                    cell = {
                        k: sum(p[k] for p in ps)
                        for k in ("all", "clean", "joined", "twice", "none")
                    } | {
                        "pages": len(xs),
                        "boxes": round(sum(p["boxes"] for p in ps) / len(ps), 2),
                    }
                    out[f"{tag}/{arm}/{n}"] = cell
                    print(
                        f"| {arm} | {n} | {cell['pages']} | {cell['all']} | {cell['clean']} | "
                        f"{cell['joined']} | {cell['twice']} | {cell['none']} | {cell['boxes']} |"
                    )
                    for r, p in zip(xs, ps):
                        for i, t in enumerate(r["key"].split(SEP)):
                            for k in (
                                (arm, n, "place", i + 1),
                                (arm, n, "len", len(t)),
                            ):
                                c = by[k]
                                c[0] += p["own"][i]
                                c[1] += 1
                                c[2] += sum(p["hit"][i])
                                c[3] += len(t)
                for r in alone:
                    if r["arm"] != (a.alone_arm or arm):
                        continue
                    p = W.page(r["key"], r[field], folded)
                    k = W.fold(r["key"]) if folded else r["key"]
                    bs = [W.fold(b) if folded else b for b in r[field]]
                    for key in ((arm, 1, "place", 1), (arm, 1, "len", len(k))):
                        c = by[key]
                        c[0] += int(k in bs)
                        c[1] += 1
                        c[2] += sum(p["hit"])
                        c[3] += len(k)
            print(
                "\n| arm | strings a caption | by | value | own | positions hit |\n|---|---|---|---|---|---|"
            )
            for (arm, n, kind, v), c in sorted(by.items(), key=lambda kv: kv[0]):
                if n == 1 and not any(
                    k[1] > 1 and k[2:] == (kind, v) for k in by
                ):  # a length the captions of several strings do not hold
                    continue
                out[f"{tag}/{arm}/{n}/{kind}/{v}"] = {
                    "own": c[0],
                    "strings": c[1],
                    "pos_hit": round(c[2] / c[3], 3),
                }
                print(
                    f"| {arm} | {n} | {kind} | {v} | {share(c[0], c[1])} | {c[2] / c[3]:.2f} |"
                )
    (a.result / "multi_read.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), "utf-8"
    )
    print(f"\n→ {a.result / 'multi_read.json'}", file=sys.stderr)

    if a.sheets:
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[2] / "cjk_anima_reseed")
        )
        from reseed import bootstrap

        bootstrap()
        from common.readers import contact_sheet
        from PIL import Image

        for arm in arms:
            d = a.result / f"sheets_{arm}"
            d.mkdir(exist_ok=True)
            for key in dict.fromkeys(r["key"] for r in recs):
                xs = [r for r in recs if r["arm"] == arm and r["key"] == key]
                rows = [
                    (
                        Image.open(r["file"]).convert("RGB"),
                        [
                            f"sc{r['scene']:02d} s{r['seed']}",
                            " / ".join(r["boxes"])[:26],
                        ],
                    )
                    for r in sorted(xs, key=lambda r: (r["scene"], r["seed"]))
                ]
                contact_sheet(
                    rows, d / f"{key.replace(SEP, '+')}.png", thumb=384, cols=4
                )
            print(f"→ {d}", file=sys.stderr)


if __name__ == "__main__":
    main()
