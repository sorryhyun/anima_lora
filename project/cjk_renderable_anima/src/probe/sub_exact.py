#!/usr/bin/env python
"""sub_exact — sub-exact rulers for multi-glyph eval groups, off ``eval_reads.json``.

Exact match is a **floor-saturated** ruler for `short` / `phrase` / `line` /
`corpus` / `word`: every sentence arm to date reads 0/32 there, so it cannot
order two arms (2026-09-18). These rulers read the OCR strings the eval stage
already wrote, so they are free — no GPU, no re-render.

    glyph recall  |{c ∈ set(ref) : c ∈ read}| / |set(ref)|, per item
    perm control  the same recall of *other items' refs* in the same group
                  against this item's read, averaged — the chance level for a
                  read of this length and glyph distribution
    lift          recall − perm control; the number to compare arms on

The control shares the read, so read length and the arm's general JA-glyph
habits are held; what is left is whether the read carries *this* item's glyphs.
A positive lift on `_held` groups is content the table never saw as a string.

    python src/probe/sub_exact.py <arm dir> [<arm dir> …] [--groups a,b] [--boot 5000]
    python src/probe/sub_exact.py <arm dir>/native … --groups en,swap   # sentence natives

Two arm dirs print a bootstrap CI on the difference of pooled lifts.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

GROUPS = ("short", "short_held", "phrase", "phrase_held", "line", "combo", "corpus", "word", "word_held", "gword", "gword_held")


def _read(rec) -> str:
    """The longest sfx read over the record's boxes (whole-image + per-box)."""
    return max((r.get("sfx") or "" for r in rec["reads"]), key=len, default="")


def lifts(path: Path, groups) -> dict:
    """group → (n, recall, control, [per-item lift]). ``path`` is an arm dir
    (``eval_reads.json``, groups = eval groups) or its ``native/`` dir
    (``native_reads.json``, groups = clauses — sentence natives, S2a)."""
    f = path / "eval_reads.json"
    if not f.exists():
        f = path / "native_reads.json"
    recs = json.loads(f.read_text())
    out = {}
    for g in groups:
        rows = [
            r
            for r in recs
            if r.get("group", r.get("clause")) == g and r.get("cond") == "trained"
        ]
        if not rows:
            continue
        refs = [r["text"] for r in rows]
        reads = [_read(r) for r in rows]

        def rec(ref, read):
            s = set(ref)
            return sum(1 for c in s if c in read) / len(s)

        per = []
        for i, (ref, read) in enumerate(zip(refs, reads)):
            ctrl = [rec(refs[j], read) for j in range(len(refs)) if j != i]
            per.append(rec(ref, read) - (st.mean(ctrl) if ctrl else 0.0))
        real = st.mean(rec(a, b) for a, b in zip(refs, reads))
        out[g] = (len(rows), real, real - st.mean(per), per)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    groups = [g for g in a.groups.split(",") if g]
    pooled = {}
    for d in a.dirs:
        t = lifts(d, groups)
        print(f"\n== {d.parent.name + '/' if d.name == 'native' else ''}{d.name} ==")
        print(f"{'group':>12} | {'n':>3} | recall | control |   lift")
        for g, (n, real, ctrl, per) in t.items():
            print(f"{g:>12} | {n:3d} | {real:6.3f} |  {ctrl:6.3f} | {st.mean(per):+6.3f}")
        pooled[d.name] = [x for _, _, _, per in t.values() for x in per]
        print(f"{'pooled':>12} | {len(pooled[d.name]):3d} | {'':6} |  {'':6} | {st.mean(pooled[d.name]):+6.3f}")
    if len(a.dirs) == 2:
        rng = random.Random(1)
        x, y = (pooled[d.name] for d in a.dirs)
        diffs = sorted(
            st.mean([rng.choice(y) for _ in y]) - st.mean([rng.choice(x) for _ in x])
            for _ in range(a.boot)
        )
        lo, hi = diffs[a.boot // 40], diffs[-a.boot // 40]
        print(
            f"\npooled lift {a.dirs[1].name} − {a.dirs[0].name} = "
            f"{st.mean(y) - st.mean(x):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
            f"P(>0) = {sum(d > 0 for d in diffs) / len(diffs):.3f}"
        )


if __name__ == "__main__":
    main()
