"""What a row budget buys in covered lines: single kanji vs multi-glyph pieces.

A line is *covered* when every Qwen piece of it has a warm row. Reads a phrase
TSV (``dialogue_2_10.tsv``), classes every piece with a pack row (single kana /
single kanji / multi-glyph JA piece / other) and prints line coverage for
kanji-only, multi-only and the joint frequency ranking on top of the step-1
inventory (single kana + punctuation + ``kanji:200``). CPU only, ≈ 1 min.

    .venv/bin/python project/cjk_renderable_anima/src/probe/piece_coverage.py \
        --phrase_file <manga109s>/derived/dialogue_2_10.tsv [--out ranked.tsv]

reports/piece_coverage_2026_09_21.md is the read.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402,F401  (sys.path bootstrap)
from data.inventory import phrase_file_lines, qwen_pieces  # noqa: E402

JA = re.compile(r"^[぀-ヿ一-鿿々ー]+$")
KANJI = re.compile(r"^[一-鿿々]$")
PUNCT = set("、 。 ・ ー ～ 〜 ！ ？ 「 」 ！！ ・・・ ・・・・".split())
SENT_PIECES = 4  # a covered line of this many pieces counts as a sentence


def piece_class(p: str, row) -> str:
    if row is None:
        return "norow"
    if not JA.match(p):
        return "other"
    if len(p) == 1:
        return "kanji1" if KANJI.match(p) else "kana1"
    return "multi"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase_file", required=True)
    ap.add_argument("--kanji_today", type=int, default=200)
    ap.add_argument("--out", help="ranked cold pieces → TSV (rank, piece, count, class, glyphs)")
    a = ap.parse_args()

    tok, q = qwen_pieces()
    lines = [t for t, _b, _n in phrase_file_lines(Path(a.phrase_file), 1, 99, norm=True)]
    dec: dict = {}
    tl = []
    for ids in tok(lines, add_special_tokens=False)["input_ids"]:
        ps = []
        for i in ids:
            if i not in dec:
                p = tok.decode([i])
                dec[i] = (p, piece_class(p, q.get(i)))
            ps.append(dec[i])
        tl.append(ps)

    cnt: Counter = Counter(pc for ps in tl for pc in ps)
    mass: Counter = Counter()
    for (_, c), n in cnt.items():
        mass[c] += n
    tot = sum(mass.values())
    print(f"lines {len(tl)}; piece-token mass " + ", ".join(f"{k} {v / tot:.3f}" for k, v in mass.most_common()))

    multi = [p for (p, c), _ in cnt.most_common() if c == "multi"]
    kanji = [p for (p, c), _ in cnt.most_common() if c == "kanji1"]
    today = {p for (p, c) in cnt if c == "kana1"} | PUNCT | set(kanji[: a.kanji_today])
    print(f"distinct: multi {len(multi)}, single kanji {len(kanji)}; today ≈ {len(today)} rows")

    def cov(warm: set) -> tuple[int, int]:
        full = [ps for ps in tl if all(p in warm for p, _ in ps)]
        return len(full), sum(len(ps) >= SENT_PIECES for ps in full)

    def row(name: str, warm: set) -> None:
        f, s = cov(warm)
        print(f"  {name:34s} rows {len(warm):5d}  lines {f:6d} ({f / len(tl):.3f})  sentences {s:6d}")

    row("today", today)
    for n in (600, len(kanji)):
        row(f"kanji only: kanji:{n}", today | set(kanji[:n]))
    for n in (100, 200, 500, 1000):
        row(f"multi only: +{n}", today | set(multi[:n]))
    cold = [(p, n, c) for (p, c), n in cnt.most_common() if c in ("multi", "kanji1") and p not in today]
    for n in (200, 400, 600, 800, 1000, 1500, 2000, 3000):
        add = cold[:n]
        nk = sum(c == "kanji1" for _, _, c in add)
        gl = Counter(min(len(p), 5) for p, _, c in add if c == "multi")
        row(
            f"joint +{n} (kanji {nk}, multi {n - nk}; 2/3/4/5+ glyphs {gl[2]}/{gl[3]}/{gl[4]}/{gl[5]})",
            today | {p for p, _, _ in add},
        )
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("# rank\tpiece\tcount\tclass\tglyphs\n")
            for r, (p, n, c) in enumerate(cold, 1):
                fh.write(f"{r}\t{p}\t{n}\t{c}\t{len(p)}\n")


if __name__ == "__main__":
    main()
