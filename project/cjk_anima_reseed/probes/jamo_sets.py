#!/usr/bin/env python
"""jamo_sets — the phase-1 syllable sets (``proposal_jamo.md`` § 4) and the
font check (§ 3 prerequisite 4).

**Order.** Every KS X 1001 syllable, picked greedily: first the most cells
(``jamo.cells``: 57 C + 21 V + 28 F + 6 layouts = 112) not yet covered, then
the most cells covered once so far, then the most frequent. The trained sets
are prefixes of the one order (``J64`` ⊂ ``J96`` ⊂ ``J128``), so a bigger arm
adds rows and keeps the smaller one's.

**Frequency.** No KO frequency list here; the proxy is Qwen3's byte-BPE
merge order — a syllable that is one Qwen token ranks by its id (이 0th,
다 1st, 는 3rd, 가 8th), the 8 660 byte-split syllables after every one of
them. Two cells exist in one KS X 1001 syllable only (ㅉ before a compound
vowel; the final ㄿ, 읊), so no prefix covers every cell twice.

**H.** The 32 most frequent syllables outside ``J128`` whose four cells
``J64`` covers: no arm trains them (the zero-shot read, R2).

**Fonts.** For every face a ``lang`` run draws (``pools.lang_fonts``:
``find_fonts()`` + ``kozh/``): Hangul coverage of KS X 1001 and of all
11 172, and the set glyphs it maps to an empty outline.

    .venv/bin/python project/cjk_anima_reseed/probes/jamo_sets.py

CPU, seconds. Writes ``assets/jamo_sets.json``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from reseed import bootstrap  # noqa: E402

bootstrap()

OUT = HOME / "assets" / "jamo_sets.json"
SIZES = (64, 96, 128)
N_HELD = 32


def qwen_rank() -> dict:
    """syllable → its Qwen id if it is one token, else past every such id."""
    from transformers import AutoTokenizer

    from library.env import resolve_under_home
    from reseed.jamo import ALL

    tok = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    big = len(tok)
    out = {}
    for s in ALL:
        ids = tok.encode(s, add_special_tokens=False)
        out[s] = ids[0] if len(ids) == 1 else big + ord(s)
    return out


def order(pool, rank: dict, n: int) -> list:
    from reseed.jamo import cells

    seen: Counter = Counter()
    left = set(pool)
    out = []
    while len(out) < n:

        def key(s):
            c = cells(s)
            return (
                -sum(seen[x] == 0 for x in c),
                -sum(seen[x] == 1 for x in c),
                rank[s],
            )

        s = min(left, key=key)
        out.append(s)
        left.discard(s)
        seen.update(cells(s))
    return out


def held(pool, rank: dict, trained: list, cover: list) -> list:
    from reseed.jamo import cells

    have = set().union(*map(cells, cover))
    out = [s for s in sorted(pool, key=rank.get) if s not in set(trained)]
    return [s for s in out if cells(s) <= have][:N_HELD]


def fonts_check(glyphs: str) -> list:
    from common.render.flat import FONT_DIR, find_fonts, font_covers
    from reseed.jamo import ALL, KSX1001
    from reseed.pools import KOZH_FONTS, empty_glyphs

    kozh = sorted(str(p) for p in (FONT_DIR / KOZH_FONTS).glob("*.[ot]tf"))
    out = []
    for f in find_fonts() + kozh:
        ks = sum(font_covers(f, c) for c in KSX1001)
        if not ks:
            continue
        out.append(
            {
                "face": Path(f).name,
                "ksx1001": ks,
                "all": sum(font_covers(f, c) for c in ALL),
                "sets_missing": "".join(c for c in glyphs if not font_covers(f, c)),
                "sets_empty": empty_glyphs(f, glyphs),
            }
        )
    return out


def main() -> None:
    from reseed.jamo import KSX1001, cells, layout

    rank = qwen_rank()
    top = order(KSX1001, rank, max(SIZES))
    sets = {f"J{n}": "".join(top[:n]) for n in SIZES}
    h = "".join(held(KSX1001, rank, top, top[:64]))
    assert len(h) == N_HELD, len(h)
    report = {}
    for name, s in {**sets, "H": h}.items():
        cov = Counter(x for c in s for x in cells(c))
        report[name] = {
            "n": len(s),
            "cells": len(cov),
            "cells_twice": sum(v >= 2 for v in cov.values()),
            "one_token": sum(rank[c] < 10**6 for c in s),
            "layouts": dict(sorted(Counter(map(layout, s)).items())),
        }
    fonts = fonts_check("".join(dict.fromkeys(sets[f"J{max(SIZES)}"] + h)))
    for name, r in report.items():
        print(f"{name}: {r}")
        print(f"  {sets.get(name, h)}")
    print("fonts (Hangul faces): face, KS X 1001 / 2350, all / 11172, missing / empty")
    for f in fonts:
        print(
            f"  {f['face']:36s} {f['ksx1001']:5d} {f['all']:6d}  "
            f"{f['sets_missing'] or '-'} / {f['sets_empty'] or '-'}"
        )
    OUT.write_text(
        json.dumps(
            {**sets, "H": h, "report": report, "fonts": fonts},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
