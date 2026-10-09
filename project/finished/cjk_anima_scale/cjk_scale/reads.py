"""Per-render scoring the experiments read arms with — Stage B's scoring
(``experiments/stage_b``, 2026-09-26), shared by every experiment since.

    hits     a ``native_reads.json`` → per-render booleans, keyed for pairing
    tally    per-key + words / singles totals (printed)
    paired   McNemar of two arms on their shared renders

The paired unit is the render: ``(text, clause, pi, seed)``, so two arms read
on the same prompts × seeds pair one-to-one.
"""

from __future__ import annotations

import json
import re
from math import comb
from pathlib import Path

METRICS = ("official", "contained", "le1", "le2", "repeat", "kana", "dup", "le1c")


def hits(path: Path, chars, clauses: str) -> dict:
    """``{(text, clause, pi, seed): {official, contained, le1, le2, repeat,
    dup, le1c}}`` per render — the paired unit. ``dup``: a word's read holds a
    doubled glyph (ひまわりり; no target repeats a glyph); ``le1c``: ≤ 1 edit
    once doubled glyphs are collapsed, i.e. what in-word doubling costs."""
    from common.text import lev, norm

    cl = set(clauses.split(","))
    out = {}
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] not in chars or m["clause"] not in cl:
            continue
        t = norm(m["text"])
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        best = min((lev(r, t) for r in reads if r), default=len(t))
        best_c = min(
            (lev(re.sub(r"(.)\1+", r"\1", r), t) for r in reads if r), default=len(t)
        )
        out[(m["text"], m["clause"], m["pi"], m["seed"])] = {
            "official": bool(m.get("hit_sfx")) and bool(m.get("hit_vl")),
            "contained": any(t in r for r in reads),
            "le1": len(t) > 2 and best <= 1,
            "le2": len(t) > 2 and best <= 2,
            "repeat": len(t) == 1 and any(r.count(t) >= 2 for r in reads),
            "kana": any(re.search(r"[ぁ-ヿ]", r) for r in reads),
            "dup": len(t) > 1 and any(re.search(r"(.)\1", r) for r in reads),
            "le1c": len(t) > 2 and best_c <= 1,
        }
    return out


def tally(h: dict) -> dict:
    per: dict = {}
    for (text, clause, _pi, _s), v in h.items():
        c = per.setdefault(f"{text}|{clause}", dict.fromkeys(("n", *METRICS), 0))
        c["n"] += 1
        for k in METRICS:
            c[k] += v[k]
    words = {k: v for k, v in per.items() if len(k.split("|")[0].replace(" ", "")) > 1}
    singles = {k: v for k, v in per.items() if k not in words}
    tot = {
        name: {k: sum(v[k] for v in grp.values()) for k in ("n", *METRICS)}
        for name, grp in (("words", words), ("singles", singles))
    }
    for k, c in sorted(per.items()):
        print(
            f"  {k:<16} off {c['official']:>2}  cont {c['contained']:>2}  "
            f"≤1 {c['le1']:>2}  ≤2 {c['le2']:>2}  rep {c['repeat']:>2}  "
            f"dup {c['dup']:>2}  ≤1c {c['le1c']:>2} / {c['n']}",
            flush=True,
        )
    for name, c in tot.items():
        print(
            f"  {name.upper():<16} off {c['official']:>3}  cont {c['contained']:>3}  "
            f"≤1 {c['le1']:>3}  ≤2 {c['le2']:>3}  rep {c['repeat']:>3}  "
            f"kana {c['kana']:>3}  dup {c['dup']:>3}  ≤1c {c['le1c']:>3} / {c['n']}",
            flush=True,
        )
    return {"per_key": per, "total": tot}


def paired(a: dict, b: dict) -> dict:
    """McNemar (exact two-sided binomial) of ``a`` vs ``b`` on their shared
    renders: ``{metric: [gained, lost, p]}``."""
    keys = sorted(set(a) & set(b))
    out = {}
    for k in METRICS:
        g = sum(a[x][k] and not b[x][k] for x in keys)
        lo = sum(b[x][k] and not a[x][k] for x in keys)
        n = g + lo
        p = (
            min(1.0, 2 * sum(comb(n, i) for i in range(min(g, lo) + 1)) / 2**n)
            if n
            else 1.0
        )
        out[k] = [g, lo, float(f"{p:.2g}")]
    return {"n": len(keys), **out}
