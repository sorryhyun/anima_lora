#!/usr/bin/env python
"""Word read — a read's word pages counted per position (CPU).

Reads the ``records.json`` of a ``row_line.py`` result (records of class ``word``: the
string, the page's box reads). Per page the box closest to the string is taken
(``reward.seq_score``, as ``gd.pick`` orders boxes; the confidence floor is not in the
records) and aligned to the string by a longest common subsequence; a position is a hit
when its glyph is on the alignment.

Per arm and string length: pages, ``solo`` under the tuned reader and hayai, the share of
positions hit, the glyphs the box holds over the glyphs asked, pages whose box is the
first glyph alone, pages with no box read. Then the hit share at each position, and per
glyph (a row) its hits over the positions it was asked at. With ``--fold`` a small kana
and its full-size glyph are one glyph in the alignment (``g1.SMALL``), printed beside the
strict count.

    .venv/bin/python project/cjk_rl/probes/word_read.py project/cjk_rl/results/<ts>-row_line-<stem>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]
SMALL = dict(
    zip(
        "ぁぃぅぇぉっゃゅょァィゥェォッャュョヵヶ",
        "あいうえおつやゆよアイウエオツヤユヨカケ",
    )
)


def align(t: str, c: str) -> list:
    """A bool per position of ``t``: on a longest common subsequence with ``c``."""
    n, m = len(t), len(c)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = (
                L[i + 1][j + 1] + 1 if t[i] == c[j] else max(L[i + 1][j], L[i][j + 1])
            )
    hit, i, j = [False] * n, 0, 0
    while i < n and j < m:
        if t[i] == c[j]:
            hit[i] = True
            i, j = i + 1, j + 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return hit


def fold(s: str) -> str:
    return "".join(SMALL.get(ch, ch) for ch in s)


def page(key: str, boxes: list, folded: bool) -> dict:
    f = fold if folded else (lambda s: s)
    k, bs = f(key), [f(b) for b in boxes]
    if not bs:
        return {"hit": [False] * len(k), "drawn": 0, "first": 0, "none": 1, "solo": 0}
    best = max(bs, key=lambda b: sum(align(k, b)) / max(len(k), len(b)))
    return {
        "hit": align(k, best),
        "drawn": len(best),
        "first": int(best == k[0]),
        "none": 0,
        "solo": int(k in bs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path, help="a row_line result directory")
    ap.add_argument("--bins", default="2,3,4,5-8", help="string lengths a column")
    a = ap.parse_args()
    recs = [
        r
        for r in json.loads((a.result / "records.json").read_text("utf-8"))
        if r["cls"] == "word"
    ]
    arms = list(dict.fromkeys(r["arm"] for r in recs))
    bins = [tuple(int(x) for x in b.split("-")) for b in a.bins.split(",")]
    bins = [(b[0], b[-1]) for b in bins]
    out = {}
    for folded in (False, True):
        tag = "folded" if folded else "strict"
        print(f"\n## {tag}")
        print(
            "| arm | length | pages | solo | hayai solo | positions hit | drawn / asked | first glyph alone | no box read |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        by_pos = defaultdict(lambda: [0, 0])
        by_row = defaultdict(lambda: [0, 0])
        for arm in arms:
            for lo, hi in [*bins, (0, 99)]:
                xs = [r for r in recs if r["arm"] == arm and lo <= len(r["key"]) <= hi]
                if not xs:
                    continue
                ps = [page(r["key"], r["boxes"], folded) for r in xs]
                hits = sum(sum(p["hit"]) for p in ps)
                asked = sum(len(r["key"]) for r in xs)
                name = (
                    "all"
                    if (lo, hi) == (0, 99)
                    else (f"{lo}" if lo == hi else f"{lo}-{hi}")
                )
                cell = {
                    "pages": len(xs),
                    "solo": sum(p["solo"] for p in ps)
                    if folded
                    else sum(r["solo"] for r in xs),
                    "hy_solo": sum(r.get("hy_solo", 0) for r in xs),
                    "pos_hit": round(hits / asked, 3),
                    "drawn_over_asked": round(sum(p["drawn"] for p in ps) / asked, 2),
                    "first_alone": sum(p["first"] for p in ps),
                    "none": sum(p["none"] for p in ps),
                }
                out[f"{tag}/{arm}/{name}"] = cell
                print(
                    f"| {arm} | {name} | {cell['pages']} | {cell['solo']} | "
                    f"{'–' if folded else cell['hy_solo']} | {cell['pos_hit']} | "
                    f"{cell['drawn_over_asked']} | {cell['first_alone']} | {cell['none']} |"
                )
                if (lo, hi) == (0, 99):
                    for r, p in zip(xs, ps):
                        k = fold(r["key"]) if folded else r["key"]
                        for i, h in enumerate(p["hit"]):
                            by_pos[(arm, min(i, 7))][0] += h
                            by_pos[(arm, min(i, 7))][1] += 1
                            by_row[(arm, k[i])][0] += h
                            by_row[(arm, k[i])][1] += 1
        print("\nhit share by position (1 … 8+):")
        for arm in arms:
            row = [by_pos[(arm, i)] for i in range(8) if by_pos[(arm, i)][1]]
            print(f"  {arm}: " + " ".join(f"{h / n:.2f}" for h, n in row))
        print("\nrows by their hit share (asked at 8 positions or more), share: rows")
        for arm in arms:
            sh = {g: h / n for (x, g), (h, n) in by_row.items() if x == arm and n >= 8}
            edges = [0, 0.25, 0.5, 0.75, 1.0001]
            cnt = [
                sum(lo <= v < hi for v in sh.values())
                for lo, hi in zip(edges, edges[1:])
            ]
            print(
                f"  {arm}: under 0.25 {cnt[0]}, 0.25 – 0.5 {cnt[1]}, 0.5 – 0.75 {cnt[2]}, "
                f"0.75 or over {cnt[3]}; of {len(sh)} rows, mean {sum(sh.values()) / max(len(sh), 1):.2f}"
            )
            out[f"{tag}/{arm}/rows"] = {g: round(v, 3) for g, v in sorted(sh.items())}
    (a.result / "word_read.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), "utf-8"
    )
    print(f"\n→ {a.result / 'word_read.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
